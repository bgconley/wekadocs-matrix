"""
GraphService implements projection-only Cypher queries with cursor-aware paging.

It is the canonical place for:
  * describe_nodes
  * expand_neighbors
  * get_paths_between
  * child/parent/entity lookups

All queries adhere to the guardrails in docs/cdx-outputs/retrieval_fix.json:
  - Never select raw property maps (no implicit projections)
  - Optional snippets are truncated server-side
  - Cursor + page_size based pagination
  - Session-scoped delta deduplication
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neo4j import Driver

from src.services.context_budget_manager import BudgetExceeded, ContextBudgetManager
from src.services.delta_cache import SessionDeltaCache
from src.shared.observability import get_logger
from src.shared.observability.metrics import projection_violations_total

logger = get_logger(__name__)


class ProjectionViolation(RuntimeError):
    """Raised when a caller attempts to access non-whitelisted fields."""


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("utf-8")).decode("utf-8")


def _decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    try:
        return int(base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8"))
    except Exception:
        return 0


NODE_FIELD_PROJECTIONS = {
    "id": "n.id AS id",
    "label": "labels(n)[0] AS label",
    "title": "coalesce(n.title, n.name, n.heading, '') AS title",
    "level": "n.level AS level",
    "tokens": "n.tokens AS tokens",
    "doc_tag": "n.doc_tag AS doc_tag",
    "anchor": "n.anchor AS anchor",
}


# Use canonical relationship allow-list from schema to stay aligned with ingestion
from src.neo.schema import RELATIONSHIP_TYPES as _CANONICAL_REL_TYPES  # noqa: E402

ALLOWED_REL_TYPES = set(_CANONICAL_REL_TYPES)


@dataclass
class GraphResult:
    payload: Dict[str, Any]
    partial: bool = False
    limit_reason: str = "none"
    tokens_estimate: int = 0
    bytes_estimate: int = 0
    duplicates_suppressed: int = 0


class GraphService:
    def __init__(
        self,
        driver: Driver,
        *,
        default_page_size: int = 25,
        snippet_chars: int = 200,
        delta_cache: Optional[SessionDeltaCache] = None,
    ) -> None:
        self.driver = driver
        self.default_page_size = default_page_size
        self.snippet_chars = snippet_chars
        self.delta_cache = delta_cache or SessionDeltaCache()

    # ------------------------------------------------------------------ helpers
    def _ensure_fields(self, fields: Sequence[str]) -> None:
        unknown = set(fields) - set(NODE_FIELD_PROJECTIONS.keys())
        if unknown:
            for field in unknown:
                projection_violations_total.labels(field=field).inc()
            raise ProjectionViolation(
                f"Disallowed projection fields: {sorted(unknown)}"
            )

    def _project_clause(self, alias: str, fields: Sequence[str]) -> str:
        clauses = []
        for field in fields:
            projection = NODE_FIELD_PROJECTIONS.get(field)
            if not projection:
                projection_violations_total.labels(field=field).inc()
                raise ProjectionViolation(f"Field '{field}' is not whitelisted")
            clauses.append(projection.replace("n.", f"{alias}."))
        return ", ".join(clauses)

    def _with_budget(
        self,
        payload: Dict[str, Any],
        phase: str,
        budget: Optional[ContextBudgetManager],
    ) -> Tuple[int, int]:
        body = json.dumps(payload)
        bytes_estimate = len(body.encode("utf-8"))
        tokens_estimate = (
            budget.estimate_tokens(body) if budget else max(1, len(body) // 4)
        )
        if budget:
            try:
                budget.consume(tokens_estimate, bytes_estimate, phase)
            except BudgetExceeded:
                raise
        return tokens_estimate, bytes_estimate

    # ---------------------------------------------------------------- describe
    def describe_nodes(
        self,
        node_ids: Sequence[str],
        fields: Optional[Sequence[str]] = None,
        *,
        budget: Optional[ContextBudgetManager] = None,
        phase: str = "neighbors",
    ) -> GraphResult:
        if not node_ids:
            return GraphResult(
                payload={"results": [], "cursor": None, "next_cursor": None}
            )

        fields = fields or ("id", "label", "title", "tokens", "doc_tag")
        self._ensure_fields(fields)

        projection = self._project_clause("n", fields)
        query = f"""
        MATCH (n) WHERE (n:Chunk OR n:Document OR n:Entity) AND n.id IN $node_ids
        RETURN {projection}
        """
        logger.debug("describe_nodes query=%s", query)
        with self.driver.session() as session:
            rows = session.run(query, node_ids=list(node_ids))
            results = [dict(row) for row in rows]

        payload = {
            "results": results,
            "cursor": None,
            "next_cursor": None,
        }
        partial = False
        limit_reason = "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(payload, phase, budget)
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)

        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

    # ------------------------------------------------------------- expand neighbors
    def expand_neighbors(
        self,
        *,
        node_ids: Sequence[str],
        rel_types: Optional[Sequence[str]] = None,
        direction: str = "both",
        max_hops: int = 1,
        include_snippet: bool = False,
        page_size: Optional[int] = None,
        cursor: Optional[str] = None,
        session_id: Optional[str] = None,
        budget: Optional[ContextBudgetManager] = None,
    ) -> GraphResult:
        if not node_ids:
            return GraphResult(
                payload={
                    "nodes": [],
                    "edges": [],
                    "cursor": cursor,
                    "next_cursor": None,
                }
            )

        rel_types = rel_types or list(ALLOWED_REL_TYPES)
        invalid = set(rel_types) - ALLOWED_REL_TYPES
        if invalid:
            raise ValueError(
                f"Invalid relationship type(s): {sorted(invalid)}. "
                f"Allowed: {sorted(ALLOWED_REL_TYPES)}"
            )

        hop_limit = min(max_hops, 3)
        limit = min(page_size or self.default_page_size, 100)
        skip = _decode_cursor(cursor)
        rel_pattern = "|".join(rel_types)
        if direction == "out":
            pattern = f"-[r:{rel_pattern}*1..{hop_limit}]->"
        elif direction == "in":
            pattern = f"<-[r:{rel_pattern}*1..{hop_limit}]-"
        else:
            pattern = f"-[r:{rel_pattern}*1..{hop_limit}]-"

        snippet_clause = "CASE WHEN $include_snippet THEN substring(target.text, 0, $snippet_chars) ELSE NULL END AS raw_snippet"

        query = f"""
        UNWIND $node_ids AS start_id
        MATCH (start {{id: start_id}})
        MATCH path=(start){pattern}(target)
        WITH target, path
        ORDER BY length(path) ASC
        WITH target, collect(path) AS paths
        WITH target,
             paths[0] AS sample_path,
             length(paths[0]) AS dist
        ORDER BY dist ASC, target.id ASC
        SKIP $skip
        LIMIT $limit_plus_one
        RETURN
            target.id AS id,
            CASE
                WHEN 'Chunk' IN labels(target) THEN 'Chunk'
                WHEN 'Document' IN labels(target) THEN 'Document'
                WHEN 'Procedure' IN labels(target) THEN 'Procedure'
                WHEN 'Step' IN labels(target) THEN 'Step'
                WHEN 'Command' IN labels(target) THEN 'Command'
                WHEN 'Configuration' IN labels(target) THEN 'Configuration'
                WHEN 'Error' IN labels(target) THEN 'Error'
                WHEN 'Concept' IN labels(target) THEN 'Concept'
                WHEN 'Example' IN labels(target) THEN 'Example'
                WHEN 'Parameter' IN labels(target) THEN 'Parameter'
                WHEN 'Component' IN labels(target) THEN 'Component'
                ELSE head(labels(target))
            END AS label,
            coalesce(target.title, target.name, target.heading, '') AS title,
            target.level AS level,
            target.tokens AS tokens,
            target.doc_tag AS doc_tag,
            target.anchor AS anchor,
            {snippet_clause},
            [rel IN relationships(sample_path) | {{
                src: startNode(rel).id,
                dst: endNode(rel).id,
                type: type(rel)
            }}] AS sample_edges
        """

        params = {
            "node_ids": list(node_ids),
            "skip": skip,
            "limit_plus_one": limit + 1,
            "include_snippet": bool(include_snippet),
            "snippet_chars": self.snippet_chars,
        }

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, str]] = []
        with self.driver.session() as session:
            rows = list(session.run(query, **params))

        more = len(rows) > limit
        rows = rows[:limit]
        for record in rows:
            snippet_value = record.get("raw_snippet")
            node_entry = {
                "id": record.get("id"),
                "label": record.get("label"),
                "title": record.get("title"),
                "level": record.get("level"),
                "tokens": record.get("tokens"),
                "doc_tag": record.get("doc_tag"),
                "anchor": record.get("anchor"),
            }
            if include_snippet and snippet_value:
                node_entry["snippet"] = snippet_value[: self.snippet_chars]
            nodes.append(node_entry)
            edges.extend(record.get("sample_edges") or [])

        duplicates = 0
        if session_id:
            nodes, node_dupes = self.delta_cache.filter_nodes(session_id, nodes)
            edges, edge_dupes = self.delta_cache.filter_edges(session_id, edges)
            duplicates = node_dupes + edge_dupes

        next_cursor = _encode_cursor(skip + limit) if more else None
        payload = {
            "nodes": nodes,
            "edges": edges,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "dedupe_applied": bool(session_id),
        }

        partial = bool(next_cursor)
        limit_reason = "page_size" if next_cursor else "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(
                payload, "neighbors", budget
            )
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)

        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
            duplicates_suppressed=duplicates,
        )

    # -------------------------------------------------------------- paths between
    def get_paths_between(
        self,
        *,
        a_ids: Sequence[str],
        b_ids: Sequence[str],
        rel_types: Optional[Sequence[str]] = None,
        max_hops: int = 3,
        max_paths: int = 10,
        cursor: Optional[str] = None,
        budget: Optional[ContextBudgetManager] = None,
    ) -> GraphResult:
        if not a_ids or not b_ids:
            return GraphResult(
                payload={"paths": [], "cursor": cursor, "next_cursor": None}
            )

        rel_types = rel_types or list(ALLOWED_REL_TYPES)
        invalid = set(rel_types) - ALLOWED_REL_TYPES
        if invalid:
            raise ValueError(
                f"Invalid relationship type(s): {sorted(invalid)}. "
                f"Allowed: {sorted(ALLOWED_REL_TYPES)}"
            )

        hop_limit = min(max_hops, 5)
        limit = min(max_paths, 50)
        skip = _decode_cursor(cursor)
        rel_pattern = "|".join(rel_types)

        query = f"""
        UNWIND $a_ids AS a_id
        UNWIND $b_ids AS b_id
        MATCH (a {{id: a_id}})
        MATCH (b {{id: b_id}})
        MATCH path=shortestPath((a)-[:{rel_pattern}*..{hop_limit}]-(b))
        WITH path
        WHERE path IS NOT NULL
        WITH path
        ORDER BY length(path) ASC
        SKIP $skip
        LIMIT $limit_plus_one
        RETURN [n IN nodes(path) | n.id] AS node_ids,
               [r IN relationships(path) | type(r)] AS rel_types,
               length(path) AS length
        """
        params = {
            "a_ids": list(a_ids),
            "b_ids": list(b_ids),
            "skip": skip,
            "limit_plus_one": limit + 1,
        }
        with self.driver.session() as session:
            rows = list(session.run(query, **params))

        more = len(rows) > limit
        rows = rows[:limit]
        paths = [
            {
                "nodes": row["node_ids"],
                "types": row["rel_types"],
                "length": row["length"],
            }
            for row in rows
        ]

        next_cursor = _encode_cursor(skip + limit) if more else None
        payload = {
            "paths": paths,
            "cursor": cursor,
            "next_cursor": next_cursor,
        }
        partial = bool(next_cursor)
        limit_reason = "page_size" if next_cursor else "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(
                payload, "neighbors", budget
            )
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)

        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

    # ----------------------------------------------------------- hierarchy helpers
    def list_children(
        self,
        parent_id: str,
        *,
        page_size: Optional[int] = None,
        cursor: Optional[str] = None,
        budget: Optional[ContextBudgetManager] = None,
    ) -> GraphResult:
        if not parent_id:
            return GraphResult(
                payload={"children": [], "cursor": cursor, "next_cursor": None}
            )

        limit = min(page_size or self.default_page_size, 100)
        skip = _decode_cursor(cursor)

        query = """
        MATCH (parent {id: $parent_id})
        OPTIONAL MATCH (parent)-[:PARENT_OF|HAS_CHUNK|CONTAINS_STEP]->(child)
        WITH child
        WHERE child IS NOT NULL
        ORDER BY child.order ASC, child.title ASC
        SKIP $skip
        LIMIT $limit_plus_one
        RETURN child.id AS id,
               labels(child)[0] AS label,
               coalesce(child.title, child.name, child.heading, '') AS title,
               child.level AS level
        """
        params = {
            "parent_id": parent_id,
            "skip": skip,
            "limit_plus_one": limit + 1,
        }
        with self.driver.session() as session:
            rows = list(session.run(query, **params))

        more = len(rows) > limit
        rows = rows[:limit]
        children = [dict(row) for row in rows]

        next_cursor = _encode_cursor(skip + limit) if more else None
        payload = {"children": children, "cursor": cursor, "next_cursor": next_cursor}
        partial = bool(next_cursor)
        limit_reason = "page_size" if next_cursor else "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(
                payload, "neighbors", budget
            )
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)

        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

    def list_parents(
        self,
        section_ids: Sequence[str],
        *,
        budget: Optional[ContextBudgetManager] = None,
    ) -> GraphResult:
        if not section_ids:
            return GraphResult(payload={"results": []})

        query = """
        UNWIND $section_ids AS sid
        MATCH (child {id: sid})
        OPTIONAL MATCH (parent)-[:PARENT_OF|HAS_CHUNK|CONTAINS_STEP]->(child)
        WITH child, parent
        WHERE parent IS NOT NULL
        RETURN child.id AS section_id,
               parent.id AS parent_id,
               coalesce(parent.title, parent.name, parent.heading, '') AS parent_title
        """
        with self.driver.session() as session:
            rows = session.run(query, section_ids=list(section_ids))
            results = [dict(row) for row in rows]

        payload = {"results": results}
        partial = False
        limit_reason = "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(
                payload, "neighbors", budget
            )
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)
        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

    # ------------------------------------------------------------- entity helpers
    def get_entities_for_sections(
        self,
        section_ids: Sequence[str],
        *,
        labels: Optional[Sequence[str]] = None,
        max_per_section: int = 20,
        budget: Optional[ContextBudgetManager] = None,
    ) -> GraphResult:
        if not section_ids:
            return GraphResult(payload={"results": []})

        where_clause = "WHERE labels(entity)[0] IN $labels" if labels else ""
        query = f"""
        MATCH (section {{id: $section_id}})-[:MENTIONS]->(entity)
        {where_clause}
        RETURN {{
            id: entity.id,
            label: labels(entity)[0],
            name: coalesce(entity.name, entity.title, entity.heading, '')
        }} AS entity
        LIMIT $max_per
        """
        aggregated: Dict[str, List[Dict[str, Any]]] = {}
        with self.driver.session() as session:
            for section_id in section_ids:
                rows = session.run(
                    query,
                    section_id=section_id,
                    labels=list(labels) if labels else None,
                    max_per=max_per_section,
                )
                aggregated[section_id] = [row["entity"] for row in rows]

        results = [
            {"section_id": sid, "entities": ents} for sid, ents in aggregated.items()
        ]
        payload = {"results": results}
        partial = False
        limit_reason = "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(
                payload, "neighbors", budget
            )
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)
        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

    def get_sections_for_entities(
        self,
        entity_ids: Sequence[str],
        *,
        max_per: int = 20,
        budget: Optional[ContextBudgetManager] = None,
    ) -> GraphResult:
        if not entity_ids:
            return GraphResult(payload={"results": []})

        query = """
        MATCH (section)-[:MENTIONS]->(entity)
        WHERE entity.id IN $entity_ids
        WITH entity, section
        ORDER BY section.level ASC, section.title ASC
        RETURN entity.id AS entity_id,
               collect({section_id: section.id, title: coalesce(section.title, section.name, section.heading, '')})[0..$max_per] AS sections
        """
        with self.driver.session() as session:
            rows = session.run(
                query,
                entity_ids=list(entity_ids),
                max_per=max_per,
            )
            results = [dict(row) for row in rows]

        payload = {"results": results}
        partial = False
        limit_reason = "none"
        try:
            tokens_estimate, bytes_estimate = self._with_budget(
                payload, "neighbors", budget
            )
        except BudgetExceeded as exc:
            partial = True
            limit_reason = exc.limit_reason
            tokens_estimate = exc.usage.get("tokens", 0)
            bytes_estimate = exc.usage.get("bytes", 0)
        return GraphResult(
            payload=payload,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

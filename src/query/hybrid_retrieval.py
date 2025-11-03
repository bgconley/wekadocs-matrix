"""
Phase 7E-2: Hybrid Retrieval Implementation
Combines vector search (Qdrant) with BM25/keyword search (Neo4j full-text)
Implements RRF fusion, weighted fusion, bounded adjacency expansion, and context budget enforcement

Phase 7E-4: Enhanced with comprehensive metrics collection and SLO monitoring

Reference: Phase 7E Canonical Spec L1421-1444, L3781-3788
"""

import json
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import Driver
from qdrant_client import QdrantClient

# Phase 7E-4: Monitoring imports
from src.monitoring.metrics import get_metrics_aggregator
from src.providers.tokenizer_service import TokenizerService
from src.shared.config import get_config
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    retrieval_expansion_chunks_added,
    retrieval_expansion_rate_current,
    retrieval_expansion_total,
)

logger = get_logger(__name__)


try:
    CITATIONUNIT_BOOST = float(os.getenv("BM25_CITATIONUNIT_BOOST", "1.25"))
except ValueError:
    CITATIONUNIT_BOOST = 1.25


class FusionMethod(str, Enum):
    """Available fusion methods for hybrid retrieval."""

    RRF = "rrf"  # Reciprocal Rank Fusion (default, robust)
    WEIGHTED = "weighted"  # Weighted linear combination (requires tuning)


class ExpandWhen(str, Enum):
    """Controls when adjacency expansion is triggered."""

    AUTO = "auto"  # Long query OR scores close (spec-compliant default)
    QUERY_LENGTH_ONLY = "query_length_only"  # Only expand for long queries
    NEVER = "never"  # Disable expansion
    ALWAYS = "always"  # Always expand (for debugging)


@dataclass
class ChunkResult:
    """A chunk retrieval result with all scoring metadata."""

    chunk_id: str  # Phase 7E: canonical 'id' field
    document_id: str
    parent_section_id: str
    order: int
    level: int
    heading: str
    text: str
    token_count: int

    # Metadata
    is_combined: bool = False
    is_split: bool = False
    original_section_ids: List[str] = None
    boundaries_json: str = "{}"
    doc_tag: Optional[str] = None  # document scoping tag (e.g., REGPACK-01)
    document_total_tokens: int = 0
    source_path: Optional[str] = None
    is_microdoc: bool = False

    # Expansion tracking
    is_expanded: bool = False  # Was this chunk added via expansion?
    expansion_source: Optional[str] = None  # Which chunk triggered expansion

    # Scoring metadata
    fusion_method: Optional[str] = None  # Method used for fusion
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None  # BM25/keyword score
    vector_rank: Optional[int] = None
    vector_score: Optional[float] = None  # Vector similarity score
    fused_score: Optional[float] = None  # Final fused score

    # Retrieval context metadata
    embedding_version: Optional[str] = None
    tenant: Optional[str] = None
    is_microdoc_extra: bool = False

    # Citation labels (order, title, level) derived from CitationUnits
    citation_labels: List[Tuple[int, str, int]] = field(default_factory=list)

    def __post_init__(self):
        """Ensure required fields are populated."""
        if self.original_section_ids is None:
            self.original_section_ids = []


class BM25Retriever:
    """
    BM25/keyword retriever using Neo4j full-text search.
    Neo4j's full-text search uses Lucene under the hood, providing BM25 scoring.
    """

    def __init__(self, neo4j_driver: Driver, index_name: Optional[str] = None):
        self.driver = neo4j_driver
        self.index_name = index_name or os.getenv(
            "BM25_FT_INDEX_NAME", "chunk_text_index_v3"
        )
        self._ensure_fulltext_index()

    def _list_indexes(self, session) -> List[Dict[str, Any]]:
        """
        Return normalized index metadata across Neo4j 4.x and 5.x.
        """
        try:
            query = """
            SHOW INDEXES
            YIELD name, type, entityType, labelsOrTypes, properties, state, options
            RETURN name, type, entityType, labelsOrTypes, properties, state, options
            """
            return session.run(query).data()
        except Exception:
            pass

        rows = session.run("CALL db.index.fulltext.list()").data()
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "name": row.get("name"),
                    "type": "FULLTEXT",
                    "entityType": "NODE",
                    "labelsOrTypes": row.get("labels"),
                    "properties": row.get("properties"),
                    "state": row.get("state", "ONLINE"),
                    "options": row.get("options"),
                }
            )
        return normalized

    def _ensure_fulltext_index(self):
        """
        Ensure the full-text index targets both Chunk and CitationUnit nodes with text and heading fields.
        If an index with the same name exists but uses a different definition, drop and recreate it.
        """
        desired_labels = {"Chunk", "CitationUnit"}
        desired_props = {"text", "heading"}

        with self.driver.session() as session:
            defn = None
            rows = self._list_indexes(session)
            for row in rows:
                if row.get("name") == self.index_name:
                    defn = {
                        "type": (row.get("type") or "").upper(),
                        "labels": set(row.get("labelsOrTypes") or []),
                        "properties": set(row.get("properties") or []),
                        "state": row.get("state"),
                        "raw": row,
                    }
                    break

            need_create = False
            if defn is None:
                need_create = True
            else:
                if (
                    defn["type"] != "FULLTEXT"
                    or defn["labels"] != desired_labels
                    or not desired_props.issubset(defn["properties"])
                ):
                    logger.warning(
                        "Dropping mismatched full-text index",
                        extra={"name": self.index_name, "current": defn.get("raw")},
                    )
                    dropped = False
                    try:
                        session.run(f"DROP INDEX {self.index_name} IF EXISTS")
                        dropped = True
                    except Exception:
                        pass

                    if not dropped:
                        session.run(
                            "CALL db.index.fulltext.drop($name)", name=self.index_name
                        )
                    need_create = True

            if need_create:
                logger.info(
                    "Creating full-text index",
                    extra={
                        "name": self.index_name,
                        "labels": list(desired_labels),
                        "props": list(desired_props),
                    },
                )
                created = False
                try:
                    session.run(
                        f"CREATE FULLTEXT INDEX {self.index_name} "
                        "FOR (n:Chunk|CitationUnit) ON EACH [n.text, n.heading]"
                    )
                    created = True
                except Exception:
                    pass

                if not created:
                    session.run(
                        "CALL db.index.fulltext.createNodeIndex($name, $labels, $props)",
                        name=self.index_name,
                        labels=["Chunk", "CitationUnit"],
                        props=["text", "heading"],
                    )

            # Wait for the index to come online (best effort)
            try:
                for _ in range(60):
                    row = session.run(
                        """
                        SHOW INDEXES YIELD name, state
                        WHERE name = $name
                        RETURN state
                        """,
                        name=self.index_name,
                    ).single()
                    if row and row["state"] == "ONLINE":
                        break
                    time.sleep(0.25)
            except Exception:
                # Neo4j 4.x doesn't support SHOW; nothing further required.
                pass

            logger.info("Full-text index ensured", extra={"name": self.index_name})

    def search(
        self, query: str, top_k: int = 20, filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkResult]:
        """
        Perform citation-aware BM25 search using Neo4j full-text search.
        """
        start_time = time.time()

        # Build WHERE clause for filters (apply to resolved chunk)
        where_clauses = []
        params: Dict[str, Any] = {
            "query": query,
            "limit": max(top_k, 1),
            "index_name": self.index_name,
        }

        if filters:
            for key, value in filters.items():
                param_name = f"filter_{key}"
                where_clauses.append(f"chunk.{key} = ${param_name}")
                params[param_name] = value

        where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        search_query = f"""
CALL db.index.fulltext.queryNodes($index_name, $query)
YIELD node, score
OPTIONAL MATCH (node)-[:IN_CHUNK]->(parent:Chunk)
WITH node, parent, score
OPTIONAL MATCH (fallback:Chunk {{id: node.parent_chunk_id}})
WITH
  CASE
    WHEN node:Chunk THEN node
    WHEN parent IS NOT NULL THEN parent
    WHEN fallback IS NOT NULL THEN fallback
    ELSE NULL
  END AS chunk,
  CASE WHEN node:CitationUnit THEN node ELSE NULL END AS citation,
  score
WHERE chunk IS NOT NULL{where_clause}
RETURN
  chunk.id AS chunk_id,
  chunk.document_id AS document_id,
  chunk.parent_section_id AS parent_section_id,
  chunk.order AS `order`,
  chunk.level AS level,
  chunk.heading AS chunk_heading,
  chunk.text AS chunk_text,
  chunk.token_count AS token_count,
  chunk.is_combined AS is_combined,
  chunk.is_split AS is_split,
  chunk.original_section_ids AS original_section_ids,
  chunk.boundaries_json AS boundaries_json,
  chunk.doc_tag AS doc_tag,
  chunk.document_total_tokens AS document_total_tokens,
  chunk.is_microdoc AS is_microdoc,
  chunk.source_path AS source_path,
  chunk.embedding_version AS embedding_version,
  chunk.tenant AS tenant,
  score AS score,
  (citation IS NOT NULL) AS is_citation,
  citation.order AS citation_order,
  citation.heading AS citation_heading
ORDER BY score DESC
LIMIT $limit
"""

        aggregates: Dict[str, Dict[str, Any]] = {}

        try:
            with self.driver.session() as session:
                records = session.run(search_query, params)
                for record in records:
                    chunk_id = record["chunk_id"]
                    entry = aggregates.get(chunk_id)
                    if not entry:
                        entry = {
                            "chunk_id": chunk_id,
                            "document_id": record["document_id"],
                            "parent_section_id": record["parent_section_id"],
                            "order": int(record["order"]),
                            "level": int(record["level"]),
                            "heading": record["chunk_heading"] or "",
                            "text": record["chunk_text"] or "",
                            "token_count": int(record["token_count"] or 0),
                            "is_combined": bool(record["is_combined"]),
                            "is_split": bool(record["is_split"]),
                            "original_section_ids": record["original_section_ids"]
                            or [],
                            "boundaries_json": record["boundaries_json"] or "{}",
                            "doc_tag": record.get("doc_tag"),
                            "document_total_tokens": int(
                                record.get("document_total_tokens") or 0
                            ),
                            "is_microdoc": bool(record.get("is_microdoc")),
                            "source_path": record.get("source_path"),
                            "embedding_version": record.get("embedding_version"),
                            "tenant": record.get("tenant"),
                            "best_chunk_score": 0.0,
                            "best_cu_score": 0.0,
                            "citations": [],
                        }
                        aggregates[chunk_id] = entry

                    score = float(record.get("score") or 0.0)
                    is_citation = bool(record.get("is_citation"))
                    citation_heading = record.get("citation_heading")
                    citation_order = record.get("citation_order")

                    if is_citation:
                        entry["best_cu_score"] = max(entry["best_cu_score"], score)
                        if citation_heading:
                            order_value = (
                                int(citation_order)
                                if citation_order is not None
                                else entry["order"]
                            )
                            entry["citations"].append((order_value, citation_heading))
                    else:
                        entry["best_chunk_score"] = max(
                            entry["best_chunk_score"], score
                        )

                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    "BM25 search completed",
                    extra={
                        "query_preview": query[:50],
                        "unique_chunks": len(aggregates),
                        "elapsed_ms": f"{elapsed_ms:.2f}",
                    },
                )
        except Exception as exc:
            logger.error("BM25 search failed", extra={"error": str(exc)})
            raise

        results: List[ChunkResult] = []
        for entry in aggregates.values():
            best_chunk_score = entry["best_chunk_score"]
            best_cu_score = entry["best_cu_score"]
            final_score = (
                best_cu_score * CITATIONUNIT_BOOST
                if best_cu_score > 0.0
                else best_chunk_score
            )

            raw_labels = entry["citations"]
            deduped: List[Tuple[int, str, int]] = []
            seen: Set[Tuple[int, str]] = set()
            for order_val, title in raw_labels:
                normalized_order = int(
                    order_val if order_val is not None else entry["order"]
                )
                normalized_title = title or entry["heading"] or "Section"
                key = (normalized_order, normalized_title)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append((normalized_order, normalized_title, entry["level"]))

            deduped.sort(key=lambda item: (item[0], item[1]))

            results.append(
                ChunkResult(
                    chunk_id=entry["chunk_id"],
                    document_id=entry["document_id"],
                    parent_section_id=entry["parent_section_id"],
                    order=entry["order"],
                    level=entry["level"],
                    heading=entry["heading"],
                    text=entry["text"],
                    token_count=entry["token_count"],
                    is_combined=entry["is_combined"],
                    is_split=entry["is_split"],
                    original_section_ids=list(entry["original_section_ids"] or []),
                    boundaries_json=entry["boundaries_json"],
                    doc_tag=entry.get("doc_tag"),
                    document_total_tokens=entry.get("document_total_tokens", 0),
                    source_path=entry.get("source_path"),
                    is_microdoc=entry.get("is_microdoc", False),
                    embedding_version=entry.get("embedding_version"),
                    tenant=entry.get("tenant"),
                    bm25_score=final_score,
                    citation_labels=deduped,
                )
            )

        results.sort(
            key=lambda chunk: (
                len(chunk.citation_labels or []),
                chunk.bm25_score or 0.0,
            ),
            reverse=True,
        )
        for idx, chunk in enumerate(results, start=1):
            chunk.bm25_rank = idx

        return results[:top_k]


class VectorRetriever:
    """Vector retriever using Qdrant."""

    def __init__(
        self, qdrant_client: QdrantClient, embedder, collection_name: str = "chunks"
    ):
        self.client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name

    def search(
        self, query: str, top_k: int = 50, filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkResult]:
        """
        Perform vector search on chunks using Qdrant.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filters (e.g., document_id)

        Returns:
            List of ChunkResult objects with vector scores
        """
        start_time = time.time()

        # Embed the query
        query_vector = self.embedder.embed_query(query)

        # Build Qdrant filter
        qdrant_filter = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Search Qdrant
        results = []
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            for rank, hit in enumerate(search_results, start=1):
                payload = hit.payload
                results.append(
                    ChunkResult(
                        chunk_id=payload.get("id", hit.id),
                        document_id=payload.get("document_id", ""),
                        parent_section_id=payload.get("parent_section_id", ""),
                        order=payload.get("order", 0),
                        level=payload.get("level", 3),
                        heading=payload.get("heading", ""),
                        text=payload.get("text", ""),
                        token_count=payload.get("token_count", 0),
                        vector_rank=rank,
                        vector_score=hit.score,
                        is_combined=payload.get("is_combined", False),
                        is_split=payload.get("is_split", False),
                        original_section_ids=payload.get("original_section_ids", []),
                        boundaries_json=payload.get("boundaries_json", "{}"),
                        doc_tag=payload.get("doc_tag"),
                        document_total_tokens=payload.get("document_total_tokens", 0),
                        source_path=payload.get("source_path"),
                        is_microdoc=payload.get("is_microdoc", False),
                        embedding_version=payload.get("embedding_version"),
                        tenant=payload.get("tenant"),
                        citation_labels=[],
                    )
                )

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Vector search completed: query='{query[:50]}...', "
                f"results={len(results)}, time={elapsed_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

        return results


class HybridRetriever:
    """
    Main hybrid retrieval engine implementing Phase 7E-2 requirements.
    Combines vector and BM25 search with fusion and expansion.
    """

    def __init__(
        self,
        neo4j_driver: Driver,
        qdrant_client: QdrantClient,
        embedder,
        tokenizer: Optional[TokenizerService] = None,
    ):
        """
        Initialize hybrid retriever with all components.

        Args:
            neo4j_driver: Neo4j driver for BM25 and graph operations
            qdrant_client: Qdrant client for vector search
            embedder: Embedding provider for query vectorization
            tokenizer: Tokenizer service for token counting (optional)
        """
        self.neo4j_driver = neo4j_driver
        self.bm25_retriever = BM25Retriever(neo4j_driver)
        self.vector_retriever = VectorRetriever(qdrant_client, embedder)
        self.tokenizer = tokenizer or TokenizerService()

        # Load configuration
        config = get_config()
        search_config = config.search

        # Fusion configuration
        self.fusion_method = FusionMethod(
            getattr(search_config.hybrid, "method", "rrf")
        )
        self.rrf_k = getattr(search_config.hybrid, "rrf_k", 60)
        self.fusion_alpha = getattr(search_config.hybrid, "fusion_alpha", 0.6)

        # Expansion configuration
        expansion_config = getattr(search_config.hybrid, "expansion", {})
        if hasattr(expansion_config, "enabled"):
            self.expansion_enabled = expansion_config.enabled
            self.expansion_max_neighbors = getattr(expansion_config, "max_neighbors", 1)
            self.expansion_query_min_tokens = getattr(
                expansion_config, "query_min_tokens", 12
            )
            self.expansion_score_delta_max = getattr(
                expansion_config, "score_delta_max", 0.02
            )
        else:
            # Default expansion settings if not configured
            self.expansion_enabled = True
            self.expansion_max_neighbors = 1
            self.expansion_query_min_tokens = 12
            self.expansion_score_delta_max = 0.02

        # Micro-doc stitching configuration
        self.micro_max_neighbors = int(os.getenv("MICRODOC_MAX_NEIGHBORS", "2"))
        self.microdoc_enabled = self.micro_max_neighbors > 0
        self.micro_min_tokens = int(os.getenv("MICRODOC_MIN_TOKENS", "600"))
        self.micro_doc_max = int(os.getenv("MICRODOC_DOC_MAX", "2000"))
        self.micro_dir_depth = int(os.getenv("MICRODOC_DIR_DEPTH", "2"))
        self.micro_knn_limit = int(os.getenv("MICRODOC_KNN_LIMIT", "5"))
        self.micro_sim_threshold = float(os.getenv("MICRODOC_SIM_THRESHOLD", "0.76"))
        self.micro_per_doc_budget = int(
            os.getenv("MICRODOC_PER_DOC_TOKEN_BUDGET", "300")
        )
        self.micro_total_budget = int(os.getenv("MICRODOC_MAX_STITCH_TOKENS", "1200"))
        if self.micro_total_budget < self.micro_per_doc_budget:
            self.micro_total_budget = self.micro_per_doc_budget

        # Context budget
        self.context_max_tokens = getattr(
            search_config.response, "answer_context_max_tokens", 4500
        )

        logger.info(
            f"HybridRetriever initialized: fusion={self.fusion_method.value}, "
            f"rrf_k={self.rrf_k}, alpha={self.fusion_alpha}, "
            f"expansion={'enabled' if self.expansion_enabled else 'disabled'}, "
            f"context_budget={self.context_max_tokens}, "
            f"microdoc_neighbors={self.micro_max_neighbors}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        expand: bool = True,
        expand_when: str = "auto",
    ) -> Tuple[List[ChunkResult], Dict[str, Any]]:
        """
        Perform hybrid retrieval with fusion and optional expansion.

        Args:
            query: Search query text
            top_k: Number of final results to return
            filters: Optional filters for search
            expand: Whether to perform adjacency expansion

        Returns:
            Tuple of (results, metrics) where metrics contains timing and diagnostic info
        """
        start_time = time.time()
        metrics = {}

        # Step 1: Parallel BM25 and vector search
        # Retrieve more candidates for fusion (3x top_k)
        candidate_k = min(top_k * 3, 100)

        # BM25 search
        bm25_start = time.time()
        bm25_results = self.bm25_retriever.search(query, candidate_k, filters)
        metrics["bm25_time_ms"] = (time.time() - bm25_start) * 1000
        metrics["bm25_count"] = len(bm25_results)

        # Vector search
        vec_start = time.time()
        vec_results = self.vector_retriever.search(query, candidate_k, filters)
        metrics["vec_time_ms"] = (time.time() - vec_start) * 1000
        metrics["vec_count"] = len(vec_results)

        # Step 2: Fuse rankings
        fusion_start = time.time()
        if self.fusion_method == FusionMethod.RRF:
            fused_results = self._rrf_fusion(bm25_results, vec_results)
        else:
            fused_results = self._weighted_fusion(bm25_results, vec_results)
        metrics["fusion_time_ms"] = (time.time() - fusion_start) * 1000
        metrics["fusion_method"] = self.fusion_method.value

        # Step 3: Take top_k from fusion as seeds
        fused_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
        seeds = fused_results[:top_k]
        metrics["seed_count"] = len(seeds)

        # Hydrate vector-only winners with citation labels
        self._hydrate_missing_citations(seeds)

        # Prefer chunks that carry richer citation labels
        seeds.sort(
            key=lambda chunk: (
                len(chunk.citation_labels or []),
                chunk.fused_score or 0.0,
            ),
            reverse=True,
        )

        dt = filters.get("doc_tag") if filters else None
        if dt:
            before = len(seeds)
            seeds = [c for c in seeds if getattr(c, "doc_tag", None) == dt]
            metrics["seed_count"] = len(seeds)
            logger.info(
                "Filtered seeds by doc_tag=%s: kept %d/%d", dt, len(seeds), before
            )

        # Step 4: Optional dominance gating before expansion (stabilize doc continuity)
        # Gate seeds to a primary document only if dominance is clear
        def _gate_to_primary_document(
            cs: List[ChunkResult], sample_size: int = 8
        ) -> List[ChunkResult]:
            if not cs:
                return cs
            from collections import Counter

            sample_ids = [
                getattr(c, "document_id", None)
                for c in cs[:sample_size]
                if getattr(c, "document_id", None)
            ]
            if not sample_ids:
                return cs
            doc_id, count = Counter(sample_ids).most_common(1)[0]
            top3 = sample_ids[:3]
            dominance = (
                (top3.count(doc_id) >= 2)
                or (count >= 3)
                or (count / max(1, len(sample_ids)) >= 0.5)
            )
            if dominance:
                gated = [c for c in cs if getattr(c, "document_id", None) == doc_id]
                logger.info(
                    "Gating seeds to primary document %s: kept %d/%d",
                    (doc_id or "")[:8],
                    len(gated),
                    len(cs),
                )
                return gated
            return cs

        seeds = _gate_to_primary_document(seeds)

        microdoc_extras: List[ChunkResult] = []
        microdoc_tokens = 0
        if self.microdoc_enabled:
            microdoc_extras, microdoc_tokens = self._expand_microdoc_results(
                query, fused_results, seeds, filters or {}
            )
            metrics["microdoc_extras"] = len(microdoc_extras)
            metrics["microdoc_tokens"] = microdoc_tokens
        else:
            metrics["microdoc_extras"] = 0
            metrics["microdoc_tokens"] = 0

        # Step 5: Gating decision for expansion
        when = ExpandWhen(expand_when)
        triggered, reason, score_delta, query_tokens = self._should_expand(
            query, seeds, when
        )
        metrics["expansion_triggered"] = bool(expand and triggered)
        metrics["expansion_reason"] = reason
        metrics["scores_close_delta"] = score_delta
        metrics["query_tokens"] = query_tokens

        # Step 6: Optional bounded adjacency expansion
        # Note: Expansion ADDS neighbors without re-limiting to top_k
        all_results = list(seeds)
        if expand and triggered and self.expansion_enabled:
            expansion_start = time.time()
            expanded_results = self._bounded_expansion(query, seeds)

            # Enforce doc_tag and same-document continuity for neighbors
            dt = filters.get("doc_tag") if filters else None
            if dt:
                expanded_results = [
                    n for n in expanded_results if getattr(n, "doc_tag", None) == dt
                ]
            seed_docs = {c.chunk_id: c.document_id for c in seeds}
            expanded_results = [
                n
                for n in expanded_results
                if seed_docs.get(getattr(n, "expansion_source", None)) == n.document_id
            ]

            all_results.extend(expanded_results)
            metrics["expansion_time_ms"] = (time.time() - expansion_start) * 1000
            metrics["expansion_count"] = len(expanded_results)
            metrics["expanded_source_count"] = min(
                len(seeds), getattr(self, "max_sources_to_expand", 5)
            )
        else:
            metrics["expansion_count"] = 0
            metrics["expanded_source_count"] = 0

        # Include micro-doc extras prior to dedup
        if microdoc_extras:
            all_results.extend(microdoc_extras)

        # Step 6: Dedup and final sort (no top_k cap - budget enforces limit)
        all_results = self._dedup_results(all_results)
        self._hydrate_missing_citations(all_results)
        all_results.sort(
            key=lambda chunk: (
                len(chunk.citation_labels or []),
                chunk.fused_score or 0.0,
            ),
            reverse=True,
        )
        final_results = all_results

        # Gentle document continuity boost before final assembly
        final_results = self._apply_doc_continuity_boost(final_results, alpha=0.12)

        # Step 6: Context assembly with budget enforcement
        context_start = time.time()
        final_results = self._enforce_context_budget(final_results)
        metrics["context_assembly_ms"] = (time.time() - context_start) * 1000
        metrics["final_count"] = len(final_results)
        metrics["total_tokens"] = sum(r.token_count for r in final_results)

        metrics["total_time_ms"] = (time.time() - start_time) * 1000

        # Phase 7E-4: Record metrics to aggregator if enabled
        config = get_config()
        if config.monitoring.metrics_aggregation_enabled:
            aggregator = get_metrics_aggregator()
            aggregator.record_retrieval(
                latency_ms=metrics["total_time_ms"],
                chunks_returned=len(final_results),
                expanded=metrics.get("expansion_triggered", False),
                fusion_method=self.fusion_method.value,
            )

            # Emit Prometheus metrics for expansion
            if metrics.get("expansion_triggered", False):
                retrieval_expansion_total.labels(
                    expansion_reason=metrics.get("expansion_reason", "unknown")
                ).inc()
                retrieval_expansion_chunks_added.observe(
                    metrics.get("expansion_count", 0)
                )

            # Update current expansion rate gauge (rolling window)
            window_metrics = aggregator.get_window_metrics(window_seconds=300)
            if "retrieval" in window_metrics:
                expansion_rate = window_metrics["retrieval"].expansion_rate
                retrieval_expansion_rate_current.set(expansion_rate)
                metrics["expansion_rate_5min"] = expansion_rate

            # Phase 7E-4: Check SLOs if monitoring enabled
            if config.monitoring.slo_monitoring_enabled:
                # Get recent p95 from aggregator
                if "retrieval" in window_metrics:
                    p95_latency = window_metrics["retrieval"].p95_latency
                    expansion_rate = window_metrics["retrieval"].expansion_rate

                    # Check retrieval p95 SLO (target: ≤500ms)
                    if p95_latency > config.monitoring.retrieval_p95_target_ms:
                        logger.warning(
                            "Retrieval p95 SLO violation",
                            p95_ms=p95_latency,
                            target_ms=config.monitoring.retrieval_p95_target_ms,
                            query_preview=query[:50],
                        )

                    # Check expansion rate SLO (target: 10-40%)
                    if expansion_rate < config.monitoring.expansion_rate_min:
                        logger.warning(
                            "Expansion rate below minimum",
                            expansion_rate=expansion_rate,
                            min_threshold=config.monitoring.expansion_rate_min,
                        )
                    elif expansion_rate > config.monitoring.expansion_rate_max:
                        logger.warning(
                            "Expansion rate above maximum",
                            expansion_rate=expansion_rate,
                            max_threshold=config.monitoring.expansion_rate_max,
                        )

        logger.info(
            f"Hybrid retrieval complete: query='{query[:50]}...', "
            f"results={len(final_results)}, tokens={metrics['total_tokens']}, "
            f"microdocs={metrics['microdoc_extras']}, "
            f"time={metrics['total_time_ms']:.2f}ms"
        )

        return final_results, metrics

    def _rrf_fusion(
        self, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Reciprocal Rank Fusion (RRF) - robust fusion without parameter tuning.

        RRF formula: score = Σ(1 / (k + rank_i))
        where k is a constant (default 60) and rank_i is the rank in result list i

        Reference: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
        """
        # Build rank dictionaries
        bm25_ranks: Dict[str, int] = {}
        for i, r in enumerate(bm25_results):
            rank = i + 1
            bm25_ranks[r.chunk_id] = rank
            if r.bm25_rank is None:
                r.bm25_rank = rank

        vec_ranks: Dict[str, int] = {}
        for i, r in enumerate(vec_results):
            rank = i + 1
            vec_ranks[r.chunk_id] = rank
            if r.vector_rank is None:
                r.vector_rank = rank

        # Build combined result map
        all_chunks = {}
        for r in bm25_results:
            all_chunks[r.chunk_id] = r
        for r in vec_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r

        # Calculate RRF scores
        for chunk_id, chunk in all_chunks.items():
            bm25_rank = bm25_ranks.get(chunk_id, float("inf"))
            vec_rank = vec_ranks.get(chunk_id, float("inf"))

            # RRF formula
            rrf_score = 0.0
            if bm25_rank != float("inf"):
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)
            if vec_rank != float("inf"):
                rrf_score += 1.0 / (self.rrf_k + vec_rank)

            chunk.fused_score = rrf_score
            chunk.fusion_method = "rrf"

            # Preserve original scores for diagnostics
            if chunk_id in bm25_ranks:
                chunk.bm25_score = chunk.bm25_score or 0
            if chunk_id in vec_ranks:
                chunk.vector_score = chunk.vector_score or 0

        return list(all_chunks.values())

    def _weighted_fusion(
        self, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Weighted linear combination fusion.

        score = α * vector_score + (1-α) * bm25_score
        where α is the vector weight (default 0.6)

        Note: Requires score normalization since BM25 and vector scores have different ranges.
        """

        def normalize_scores(results: List[ChunkResult], score_attr: str):
            scores = [
                getattr(r, score_attr, 0) for r in results if getattr(r, score_attr)
            ]
            if not scores:
                return
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return

            for r in results:
                score = getattr(r, score_attr, 0)
                if score:
                    normalized = (score - min_score) / (max_score - min_score)
                    setattr(r, f"norm_{score_attr}", normalized)

        normalize_scores(bm25_results, "bm25_score")
        normalize_scores(vec_results, "vector_score")

        all_chunks: Dict[str, ChunkResult] = {}
        for r in bm25_results:
            all_chunks[r.chunk_id] = r
            r.norm_bm25_score = getattr(r, "norm_bm25_score", 0)
        for r in vec_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r
            else:
                all_chunks[r.chunk_id].vector_score = r.vector_score
                all_chunks[r.chunk_id].norm_vector_score = getattr(
                    r, "norm_vector_score", 0
                )

        for chunk in all_chunks.values():
            norm_vec = getattr(chunk, "norm_vector_score", 0)
            norm_bm25 = getattr(chunk, "norm_bm25_score", 0)

            chunk.fused_score = (
                self.fusion_alpha * norm_vec + (1 - self.fusion_alpha) * norm_bm25
            )
            chunk.fusion_method = "weighted"

        return list(all_chunks.values())

    def _apply_doc_continuity_boost(
        self, chunks: List[ChunkResult], alpha: float = 0.12
    ) -> List[ChunkResult]:
        """Slightly favor documents that dominate the seed set."""
        if not chunks:
            return chunks

        from collections import Counter

        sample = chunks[:10]
        counts = Counter(getattr(c, "document_id", None) for c in sample)
        boosted: List[ChunkResult] = []
        for c in chunks:
            base = float(c.fused_score or c.vector_score or c.bm25_score or 0.0)
            doc_count = counts.get(getattr(c, "document_id", None), 0)
            c.fused_score = base * (1.0 + alpha * doc_count)
            boosted.append(c)
        boosted.sort(key=lambda x: x.fused_score or 0.0, reverse=True)
        return boosted

    def _hydrate_missing_citations(self, chunks: List[ChunkResult]) -> None:
        """
        Ensure chunks surfaced by vectors also emit citation labels by fetching their CitationUnit headings.
        """
        query_ids = list({chunk.chunk_id for chunk in chunks})

        lookup: Dict[str, List[Tuple[int, str, int]]] = {}
        if query_ids:
            query = """
            UNWIND $ids AS cid
            MATCH (c:Chunk {id: cid})
            OPTIONAL MATCH (u:CitationUnit)-[:IN_CHUNK]->(c)
            WITH c, u ORDER BY u.order ASC
            RETURN c.id AS chunk_id, collect([u.order, u.heading, u.level]) AS labels
            """

            with self.neo4j_driver.session() as session:
                rows = session.run(query, ids=query_ids).data()

            lookup = {
                row["chunk_id"]: [
                    (int(order or 0), heading, int(level or 0))
                    for order, heading, level in (row.get("labels") or [])
                    if heading
                ]
                for row in rows
            }

        pending_sections: List[str] = []
        chunk_by_id: Dict[str, ChunkResult] = {c.chunk_id: c for c in chunks}

        for chunk in chunks:
            labels: List[Tuple[int, str, int]] = []
            if lookup:
                labels.extend(lookup.get(chunk.chunk_id, []) or [])

            existing = getattr(chunk, "citation_labels", None) or []
            if existing:
                for item in existing:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        order_val = int(item[0] or 0)
                        title = (item[1] or "").strip()
                        level_val = (
                            int(item[2]) if len(item) > 2 and item[2] is not None else 0
                        )
                        if title:
                            labels.append((order_val, title, level_val))

            try:
                boundaries = json.loads(chunk.boundaries_json or "{}")
                items = (
                    boundaries
                    if isinstance(boundaries, list)
                    else boundaries.get("sections", [])
                )
                parsed: List[Tuple[int, str, int]] = []
                for section in items or []:
                    heading_val = (
                        section.get("heading") or section.get("title") or ""
                    ).strip()
                    if not heading_val:
                        continue
                    order_val = section.get("order") or 0
                    try:
                        order_int = int(order_val)
                    except (TypeError, ValueError):
                        order_int = 0
                    level_val = int(section.get("level", 0))
                    parsed.append((order_int, heading_val, level_val))
                if parsed:
                    labels.extend(parsed)
            except Exception:
                pass

            if not labels and chunk.original_section_ids:
                pending_sections.append(chunk.chunk_id)
                continue

            if labels:
                self._assign_normalized_citations(chunk, labels)

        if pending_sections:
            query = """
            UNWIND $ids AS cid
            MATCH (c:Chunk {id: cid})
            UNWIND coalesce(c.original_section_ids, []) AS sid
            MATCH (s:Section {id: sid})
            RETURN c.id AS chunk_id, collect([s.order, s.heading, s.level]) AS labels
            """
            with self.neo4j_driver.session() as session:
                rows = session.run(query, ids=pending_sections).data()

            fallback_lookup = {
                row["chunk_id"]: [
                    (int(order or 0), heading, int(level or 0))
                    for order, heading, level in (row.get("labels") or [])
                    if heading
                ]
                for row in rows
            }

            for chunk_id, labels in fallback_lookup.items():
                chunk = chunk_by_id.get(chunk_id)
                if chunk and not getattr(chunk, "citation_labels", None):
                    if labels:
                        self._assign_normalized_citations(chunk, labels)

    def _assign_normalized_citations(
        self, chunk: ChunkResult, labels: List[Tuple[int, str, int]]
    ) -> None:
        """Normalize citation labels while preserving document order and hierarchy."""
        entries: List[Tuple[int, str, int]] = []
        seen: Set[Tuple[int, str]] = set()

        for order_val, title, level in labels:
            clean_title = (title or "").strip()
            if not clean_title:
                continue
            order_int = int(order_val or 0)
            level_int = int(level or 0)
            key = (order_int, clean_title.lower())
            if key in seen:
                continue
            seen.add(key)
            entries.append((order_int, clean_title, level_int))

        if not entries:
            return

        entries.sort(key=lambda x: (x[0], x[2], x[1].lower()))

        deep_levels = [level for order, _, level in entries if order > 0 and level >= 3]
        if deep_levels:
            min_level = min(deep_levels)
            entries = [
                (order, title, level)
                for order, title, level in entries
                if order == 0 or level >= min_level
            ]

        heading_lower = (chunk.heading or "").strip().lower()
        final_labels: List[Tuple[int, str, int]] = []
        for order_int, title, level_int in entries:
            if (
                heading_lower
                and order_int == 0
                and title.lower() == heading_lower
                and len(entries) > 1
            ):
                continue
            final_labels.append((order_int, title, level_int))

        if not final_labels and entries:
            final_labels = [entries[0]]

        chunk.citation_labels = final_labels

    def _should_expand(
        self, query: str, seeds: List[ChunkResult], when: ExpandWhen
    ) -> Tuple[bool, str, float, int]:
        """
        Determine if expansion should be triggered based on gating conditions.

        Returns: (triggered, reason, score_delta, query_tokens)
        """
        query_tokens = self.tokenizer.count_tokens(query)
        scores = [r.fused_score or 0.0 for r in seeds]

        # Calculate score delta between top results
        if len(scores) >= 2:
            score_delta = abs(scores[0] - scores[1])
        else:
            score_delta = 1.0  # Large delta if only one result

        reason = "none"
        triggered = False

        if when == ExpandWhen.NEVER:
            reason = "disabled"
        elif when == ExpandWhen.ALWAYS:
            triggered, reason = True, "forced"
        elif when == ExpandWhen.QUERY_LENGTH_ONLY:
            if query_tokens >= self.expansion_query_min_tokens:
                triggered, reason = True, "query_long"
        else:  # AUTO (spec-compliant default)
            if query_tokens >= self.expansion_query_min_tokens:
                triggered, reason = True, "query_long"
            elif len(scores) >= 2 and score_delta <= self.expansion_score_delta_max:
                triggered, reason = True, "scores_close"

        return triggered, reason, score_delta, query_tokens

    def _result_id(self, r: ChunkResult) -> Tuple:
        """Get unique identifier for a chunk result."""
        return ("chunk_id", r.chunk_id)

    def _dedup_results(self, results: List[ChunkResult]) -> List[ChunkResult]:
        """Remove duplicate chunks by identity."""
        seen = set()
        deduped = []
        for r in results:
            rid = self._result_id(r)
            if rid not in seen:
                seen.add(rid)
                deduped.append(r)
        return deduped

    def _bounded_expansion(
        self, query: str, seeds: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Bounded adjacency expansion via NEXT_CHUNK relationships.

        Adds ±1 neighbors from the first N seeds (default N=5).
        Neighbors never outrank their source (score = source_score × 0.5).

        Reference: Phase 7E Canonical Spec L1425-1434
        """
        if not seeds:
            return []

        # Limit to first 5 seeds for expansion (bounded)
        max_sources = getattr(self, "max_sources_to_expand", 5)
        eligible = seeds[:max_sources]

        # Track existing chunks to avoid duplicates
        seen_ids = {self._result_id(r) for r in seeds}
        expanded = []

        chunk_ids_to_expand = [r.chunk_id for r in eligible]

        expansion_query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {id: chunk_id})

        // Find previous chunk
        OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)

        // Find next chunk
        OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)

        WITH chunk_id,
             collect(DISTINCT prev) AS prev_chunks,
             collect(DISTINCT next) AS next_chunks

        UNWIND (prev_chunks + next_chunks) AS neighbor
        WITH chunk_id, neighbor
        WHERE neighbor IS NOT NULL

        RETURN DISTINCT
            neighbor.id AS chunk_id,
            neighbor.document_id AS document_id,
            neighbor.parent_section_id AS parent_section_id,
            neighbor.order AS `order`,
            neighbor.level AS level,
            neighbor.heading AS heading,
            neighbor.text AS text,
            neighbor.token_count AS token_count,
            neighbor.is_combined AS is_combined,
            neighbor.is_split AS is_split,
            neighbor.original_section_ids AS original_section_ids,
            neighbor.boundaries_json AS boundaries_json,
            neighbor.doc_tag AS doc_tag,
            neighbor.document_total_tokens AS document_total_tokens,
            neighbor.is_microdoc AS is_microdoc,
            neighbor.source_path AS source_path,
            neighbor.embedding_version AS embedding_version,
            neighbor.tenant AS tenant,
            chunk_id AS source_chunk
        """

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(expansion_query, chunk_ids=chunk_ids_to_expand)

                for record in result:
                    neighbor_id = record["chunk_id"]
                    rid = ("chunk_id", neighbor_id)

                    if rid in seen_ids:
                        continue  # Skip duplicates

                    seen_ids.add(rid)

                    # Find source chunk to get its score
                    source_chunk_id = record["source_chunk"]
                    source_score = 0.0
                    for seed in seeds:
                        if seed.chunk_id == source_chunk_id:
                            source_score = seed.fused_score or 0.0
                            break

                    # Neighbor score = source_score × 0.5 (never outrank source)
                    neighbor_score = source_score * 0.5

                    expanded.append(
                        ChunkResult(
                            chunk_id=neighbor_id,
                            document_id=record["document_id"],
                            parent_section_id=record["parent_section_id"],
                            order=record["order"],
                            level=record["level"],
                            heading=record["heading"] or "",
                            text=record["text"],
                            token_count=record["token_count"],
                            is_combined=record["is_combined"],
                            is_split=record["is_split"],
                            original_section_ids=record["original_section_ids"] or [],
                            boundaries_json=record["boundaries_json"] or "{}",
                            doc_tag=record.get("doc_tag"),
                            document_total_tokens=record.get(
                                "document_total_tokens", 0
                            ),
                            source_path=record.get("source_path"),
                            is_microdoc=record.get("is_microdoc", False),
                            embedding_version=record.get("embedding_version"),
                            tenant=record.get("tenant"),
                            is_expanded=True,
                            expansion_source=source_chunk_id,
                            fused_score=neighbor_score,
                            citation_labels=[],
                        )
                    )

            # Calculate query tokens for logging
            query_token_count = self.tokenizer.count_tokens(query)
            logger.info(
                f"Adjacency expansion: query_tokens={query_token_count}, "
                f"expanded={len(expanded)} chunks"
            )

        except Exception as e:
            logger.error(f"Adjacency expansion failed: {e}")
            # Don't fail the whole search if expansion fails

        return expanded

    def _expand_microdoc_results(
        self,
        query: str,
        fused_results: List[ChunkResult],
        seeds: List[ChunkResult],
        filters: Dict[str, Any],
    ) -> Tuple[List[ChunkResult], int]:
        """Stitch additional micro-doc chunks when base results are inherently small."""
        if not self.microdoc_enabled or not seeds or self.micro_max_neighbors <= 0:
            return [], 0

        extras: List[ChunkResult] = []
        stitched_tokens = 0
        used_docs = {r.document_id for r in seeds if r.document_id}

        fused_pool = [
            r for r in fused_results if r.document_id and r.document_id not in used_docs
        ]
        fused_pool.sort(key=lambda x: x.fused_score or 0.0, reverse=True)

        for base in seeds:
            if not self._is_microdoc_candidate(base):
                continue

            remaining = self.micro_max_neighbors
            cohort: List[ChunkResult] = []

            cohort.extend(
                self._microdoc_from_fused(base, fused_pool, used_docs, remaining)
            )
            remaining = self.micro_max_neighbors - len(cohort)

            if remaining > 0:
                cohort.extend(self._microdoc_from_directory(base, used_docs, remaining))
                remaining = self.micro_max_neighbors - len(cohort)

            if remaining > 0:
                cohort.extend(
                    self._microdoc_from_knn(base, used_docs, remaining, filters)
                )

            for candidate in cohort:
                if candidate.document_id in used_docs:
                    continue

                truncated_text, truncated_tokens = self._truncate_text(
                    candidate.text, self.micro_per_doc_budget
                )
                if truncated_tokens == 0:
                    continue
                if stitched_tokens + truncated_tokens > self.micro_total_budget:
                    logger.info(
                        "Microdoc stitching budget exhausted",
                        extra={"tokens": stitched_tokens},
                    )
                    return extras, stitched_tokens

                patched = replace(
                    candidate,
                    text=truncated_text,
                    token_count=truncated_tokens,
                    parent_section_id=candidate.chunk_id,
                    is_microdoc_extra=True,
                    expansion_source=base.chunk_id,
                )
                extras.append(patched)
                used_docs.add(candidate.document_id)
                stitched_tokens += truncated_tokens

        if extras:
            logger.info(
                "Microdoc stitching added extras",
                extra={
                    "base_count": len(seeds),
                    "extras": len(extras),
                    "tokens": stitched_tokens,
                },
            )
        return extras, stitched_tokens

    def _is_microdoc_candidate(self, chunk: ChunkResult) -> bool:
        return (
            chunk.token_count < self.micro_min_tokens
            and (chunk.document_total_tokens or 0) <= self.micro_doc_max
        )

    def _is_microdoc_source(self, chunk: ChunkResult) -> bool:
        return (chunk.document_total_tokens or 0) <= self.micro_doc_max

    def _microdoc_from_fused(
        self,
        base: ChunkResult,
        fused_pool: List[ChunkResult],
        used_docs: Set[str],
        limit: int,
    ) -> List[ChunkResult]:
        if limit <= 0:
            return []
        extras: List[ChunkResult] = []
        for candidate in fused_pool:
            if candidate.document_id in used_docs:
                continue
            if candidate.document_id == base.document_id:
                continue
            if not self._is_microdoc_source(candidate):
                continue
            extras.append(replace(candidate))
            if len(extras) >= limit:
                break
        return extras

    def _microdoc_from_directory(
        self, base: ChunkResult, used_docs: Set[str], limit: int
    ) -> List[ChunkResult]:
        if limit <= 0:
            return []
        prefix = self._path_prefix(base.source_path)
        if not prefix:
            return []

        query = """
        MATCH (c:Chunk)
        WHERE c.document_id <> $document_id
          AND c.document_total_tokens <= $doc_max
          AND c.source_path STARTS WITH $prefix
        RETURN c
        ORDER BY c.document_total_tokens ASC, c.token_count ASC
        LIMIT $limit
        """

        extras: List[ChunkResult] = []
        try:
            with self.neo4j_driver.session() as session:
                records = session.run(
                    query,
                    document_id=base.document_id,
                    doc_max=self.micro_doc_max,
                    prefix=prefix,
                    limit=self.micro_knn_limit,
                )
                for record in records:
                    node = record.get("c")
                    if not node:
                        continue
                    candidate = self._chunk_from_props(dict(node))
                    if candidate.document_id in used_docs:
                        continue
                    extras.append(candidate)
                    if len(extras) >= limit:
                        break
        except Exception as exc:
            logger.debug(
                "Microdoc directory lookup failed",
                extra={"error": str(exc), "doc": base.document_id},
            )
        return extras

    def _microdoc_from_knn(
        self,
        base: ChunkResult,
        used_docs: Set[str],
        limit: int,
        filters: Dict[str, Any],
    ) -> List[ChunkResult]:
        if limit <= 0:
            return []
        text = (base.text or "").strip()
        if not text:
            return []
        try:
            vectors = self.vector_retriever.embedder.embed_documents([text])
            if not vectors:
                return []
            query_vector = vectors[0]
        except Exception as exc:
            logger.debug(
                "Microdoc embedding failed",
                extra={"error": str(exc), "doc": base.document_id},
            )
            return []

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError:
            return []

        must_conditions = []
        must_not_conditions = [
            FieldCondition(key="document_id", match=MatchValue(value=base.document_id))
        ]
        doc_tag = filters.get("doc_tag")
        if doc_tag:
            must_conditions.append(
                FieldCondition(key="doc_tag", match=MatchValue(value=doc_tag))
            )

        qdrant_filter = None
        if must_conditions or must_not_conditions:
            qdrant_filter = Filter(
                must=must_conditions or None, must_not=must_not_conditions or None
            )

        extras: List[ChunkResult] = []
        try:
            hits = self.vector_retriever.client.search(
                collection_name=self.vector_retriever.collection_name,
                query_vector=query_vector,
                limit=self.micro_knn_limit,
                with_payload=True,
                score_threshold=self.micro_sim_threshold,
                query_filter=qdrant_filter,
            )
        except Exception as exc:
            logger.debug(
                "Microdoc kNN search failed",
                extra={"error": str(exc), "doc": base.document_id},
            )
            return []

        for hit in hits:
            payload = hit.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id or doc_id in used_docs or doc_id == base.document_id:
                continue
            doc_total = payload.get("document_total_tokens", 0)
            if doc_total and doc_total > self.micro_doc_max:
                continue
            candidate = ChunkResult(
                chunk_id=payload.get("id", hit.id),
                document_id=doc_id,
                parent_section_id=payload.get("parent_section_id", ""),
                order=payload.get("order", 0),
                level=payload.get("level", 3),
                heading=payload.get("heading", ""),
                text=payload.get("text", ""),
                token_count=payload.get("token_count", 0),
                is_combined=payload.get("is_combined", False),
                is_split=payload.get("is_split", False),
                original_section_ids=payload.get("original_section_ids", []),
                boundaries_json=payload.get("boundaries_json", "{}"),
                doc_tag=payload.get("doc_tag"),
                document_total_tokens=doc_total,
                source_path=payload.get("source_path"),
                is_microdoc=payload.get("is_microdoc", False),
                embedding_version=payload.get("embedding_version"),
                tenant=payload.get("tenant"),
                fused_score=hit.score,
                citation_labels=[],
            )
            extras.append(candidate)
            if len(extras) >= limit:
                break
        return extras

    def _truncate_text(self, text: str, token_budget: int) -> Tuple[str, int]:
        if token_budget <= 0 or not text:
            return "", 0
        total = self.tokenizer.count_tokens(text)
        if total <= token_budget:
            return text, total
        tokens = self.tokenizer.backend.encode(text)
        truncated_tokens = tokens[:token_budget]
        truncated_text = self.tokenizer.backend.decode(truncated_tokens)
        return truncated_text, len(truncated_tokens)

    def _chunk_from_props(self, props: Dict[str, Any]) -> ChunkResult:
        boundaries = props.get("boundaries_json", "{}")
        if isinstance(boundaries, dict):
            boundaries_json = json.dumps(boundaries, separators=(",", ":"))
        else:
            boundaries_json = boundaries or "{}"
        return ChunkResult(
            chunk_id=props.get("id"),
            document_id=props.get("document_id", ""),
            parent_section_id=props.get("parent_section_id", ""),
            order=int(props.get("order", 0)),
            level=int(props.get("level", 3)),
            heading=props.get("heading", ""),
            text=props.get("text", ""),
            token_count=int(props.get("token_count", 0)),
            is_combined=bool(props.get("is_combined", False)),
            is_split=bool(props.get("is_split", False)),
            original_section_ids=props.get("original_section_ids", []),
            boundaries_json=boundaries_json,
            doc_tag=props.get("doc_tag"),
            document_total_tokens=int(props.get("document_total_tokens", 0)),
            source_path=props.get("source_path"),
            is_microdoc=bool(props.get("is_microdoc", False)),
            embedding_version=props.get("embedding_version"),
            tenant=props.get("tenant"),
            citation_labels=[],
        )

    def _path_prefix(self, source_path: Optional[str]) -> Optional[str]:
        if not source_path:
            return None
        path = Path(source_path)
        parts = [p for p in path.parts if p not in ("", os.sep)]
        if not parts:
            normalized = source_path.replace("\\", "/")
            parts = [p for p in normalized.split("/") if p]
        if not parts:
            return None
        depth = min(self.micro_dir_depth, len(parts))
        return "/".join(parts[:depth])

    def _enforce_context_budget(self, results: List[ChunkResult]) -> List[ChunkResult]:
        """
        Enforce context budget by limiting total tokens.

        Groups by parent_section_id, sorts by order, and trims from tail
        when exceeding budget.

        Reference: Phase 7E Canonical Spec - Context Budget: Max 4,500 tokens
        """
        if not results:
            return results

        # Group by parent_section_id for proper ordering
        grouped = {}
        for r in results:
            parent = r.parent_section_id or r.chunk_id
            if parent not in grouped:
                grouped[parent] = []
            grouped[parent].append(r)

        # Sort within each group by order
        for chunks in grouped.values():
            chunks.sort(key=lambda x: x.order)

        # Assemble final list, respecting token budget
        final_results = []
        total_tokens = 0

        # Process groups by best score first
        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: max(c.fused_score or 0 for c in x[1]),
            reverse=True,
        )

        for parent_id, chunks in sorted_groups:
            for chunk in chunks:
                if total_tokens + chunk.token_count <= self.context_max_tokens:
                    final_results.append(chunk)
                    total_tokens += chunk.token_count
                else:
                    # Budget exceeded
                    logger.info(
                        f"Context budget enforced: {len(final_results)} chunks, "
                        f"{total_tokens} tokens (max: {self.context_max_tokens})"
                    )
                    break

            if total_tokens >= self.context_max_tokens:
                break

        # Sort final results by parent_section_id and order for coherent context
        final_results.sort(key=lambda x: (x.parent_section_id or "", x.order))

        return final_results

    def assemble_context(self, chunks: List[ChunkResult]) -> str:
        """
        Assemble chunks into coherent context string with headings preserved.

        Args:
            chunks: List of chunks to assemble

        Returns:
            Stitched context string
        """
        if not chunks:
            return ""

        # Group by parent and maintain order
        context_parts = []
        current_parent = None

        for chunk in chunks:
            # Add parent section heading if switching contexts
            if chunk.parent_section_id != current_parent:
                if chunk.heading:
                    context_parts.append(f"\n## {chunk.heading}\n")
                current_parent = chunk.parent_section_id

            # Add chunk text
            context_parts.append(chunk.text)

            # Add expansion indicator if applicable
            if chunk.is_expanded:
                context_parts.append(f" [expanded from: {chunk.expansion_source}]")

        return "\n".join(context_parts)

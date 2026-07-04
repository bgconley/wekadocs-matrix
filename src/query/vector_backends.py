# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade re-export), query_service.py, tests
# =============================================================================
"""
Vector backend retriever implementations: BM25, multi-vector Qdrant, and
VectorRetriever wrapper.

Extracted from hybrid_retrieval.py to isolate retrieval backend
implementations from orchestration logic.
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector as QdrantSparseVector
from qdrant_client.models import (
    FieldCondition,
)
from qdrant_client.models import Filter as QdrantFilter
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    HasIdCondition,
    MatchAny,
    MatchValue,
    Prefetch,
)

from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.factory import ProviderFactory
from src.providers.settings import EmbeddingSettings
from src.query.retrieval_types import ChunkResult, _deduplicate_entity_metadata
from src.shared.config import get_embedding_plan, get_embedding_settings, get_settings
from src.shared.observability import get_logger

logger = get_logger(__name__)

try:
    CITATIONUNIT_BOOST = float(os.getenv("BM25_CITATIONUNIT_BOOST", "1.25"))
except ValueError:
    CITATIONUNIT_BOOST = 1.25


class BM25Retriever:
    """
    BM25/keyword retriever using Neo4j full-text search.
    Neo4j's full-text search uses Lucene under the hood, providing BM25 scoring.
    """

    def __init__(
        self,
        neo4j_driver: Driver,
        index_name: Optional[str] = None,
        *,
        timeout_seconds: float = 2.0,
        allow_index_migration: bool = False,
    ):
        self.driver = neo4j_driver
        env_index_name = os.getenv("BM25_FT_INDEX_NAME")
        if env_index_name:
            self.index_name = env_index_name
        elif index_name:
            self.index_name = index_name
        else:
            self.index_name = "chunk_text_index_v3"
        self.timeout_seconds = timeout_seconds
        self.allow_index_migration = allow_index_migration
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
        Ensure the full-text index targets Chunk and CitationUnit nodes.

        Uses text and heading fields. If an index with the same name exists
        but uses a different definition, drop and recreate it.
        """
        desired_labels = {"Chunk", "CitationUnit"}
        desired_props = {"text", "heading"}

        with self.driver.session() as session:
            defn = None
            compatible_name: Optional[str] = None
            rows = self._list_indexes(session)
            for row in rows:
                labels = set(row.get("labelsOrTypes") or [])
                props = set(row.get("properties") or [])
                idx_type = (row.get("type") or "").upper()
                name = row.get("name")
                if labels == desired_labels and desired_props.issubset(props):
                    compatible_name = compatible_name or name
                if name == self.index_name:
                    defn = {
                        "type": idx_type,
                        "labels": labels,
                        "properties": props,
                        "state": row.get("state"),
                        "raw": row,
                    }

            # Reuse any compatible index, even if the name differs
            if defn is None and compatible_name:
                logger.info(
                    "Reusing existing full-text index for BM25",
                    extra={
                        "requested_name": self.index_name,
                        "existing_name": compatible_name,
                    },
                )
                self.index_name = compatible_name
                return

            need_create = False
            if defn is None:
                need_create = True
            else:
                mismatch = (
                    defn["type"] != "FULLTEXT"
                    or defn["labels"] != desired_labels
                    or not desired_props.issubset(defn["properties"])
                )
                if mismatch and not self.allow_index_migration:
                    logger.error(
                        "Full-text index mismatch detected",
                        extra={
                            "name": self.index_name,
                            "current": defn.get("raw"),
                            "expected_labels": list(desired_labels),
                            "expected_props": list(desired_props),
                        },
                    )
                    raise RuntimeError(
                        f"Full-text index {self.index_name} mismatched; "
                        "set HYBRID_ALLOW_INDEX_MIGRATION=true to recreate"
                    )
                if mismatch and self.allow_index_migration:
                    logger.warning(
                        "Dropping mismatched full-text index",
                        extra={"name": self.index_name, "current": defn.get("raw")},
                    )
                    session.run(f"DROP INDEX {self.index_name} IF EXISTS")
                    need_create = True
                if not mismatch:
                    return

            if need_create:
                logger.info(
                    "Creating full-text index",
                    extra={
                        "name": self.index_name,
                        "labels": list(desired_labels),
                        "props": list(desired_props),
                    },
                )
                session.run(
                    f"CREATE FULLTEXT INDEX {self.index_name} "
                    "FOR (n:Chunk|CitationUnit) ON EACH [n.text, n.heading]"
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
        params: Dict[str, Any] = {
            "query": query,
            "limit": max(top_k, 1),
            "index_name": self.index_name,
        }
        where_clauses: List[str] = []
        if filters:
            for key, value in filters.items():
                param_name = f"filter_{key}"
                if isinstance(value, list):
                    where_clauses.append(f"chunk.{key} IN ${param_name}")
                else:
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
  chunk.doc_is_microdoc AS doc_is_microdoc,
  chunk.is_microdoc_stub AS is_microdoc_stub,
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
                records = session.run(
                    search_query, params, timeout=self.timeout_seconds
                )
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
                            "doc_is_microdoc": bool(record.get("doc_is_microdoc")),
                            "is_microdoc_stub": bool(record.get("is_microdoc_stub")),
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
                    doc_is_microdoc=entry.get("doc_is_microdoc", False),
                    is_microdoc_stub=entry.get("is_microdoc_stub", False),
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


class QdrantMultiVectorRetriever:
    """Multi-field Qdrant retriever with weighted fusion across named vectors."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder,
        collection_name: str = "chunks_multi",
        field_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
        payload_keys: Optional[List[str]] = None,
        embedding_settings: Optional[EmbeddingSettings] = None,
        embedding_plan: Optional[Any] = None,
        *,
        use_query_api: bool = False,
        query_api_weighted_fusion: bool = False,
        multi_vector_fusion_method: str = "weighted",  # "weighted" or "rrf"
        query_api_dense_limit: int = 200,
        query_api_sparse_limit: int = 200,
        query_api_candidate_limit: int = 200,
        primary_vector_name: str = "content",
        schema_supports_sparse: bool = False,
        schema_supports_colbert: bool = False,
        schema_supports_doc_title_sparse: bool = False,
        schema_supports_title_sparse: bool = False,  # NEW: Lexical heading matching
        schema_supports_entity_sparse: bool = False,  # NEW: Lexical entity matching
        rrf_debug_logging: bool = False,  # NEW: Per-field RRF contribution logging
        rrf_field_weights: Optional[
            Dict[str, float]
        ] = None,  # NEW: Per-field RRF weights
    ):
        settings = get_settings()
        env = (
            getattr(settings, "env", None) or os.getenv("ENV", "development")
        ).lower()
        strict_env = env not in ("development", "dev", "test")
        self.client = qdrant_client
        self.embedder = embedder
        self.embedding_plan = embedding_plan
        self.sparse_embedder = embedder
        self.colbert_embedder = embedder
        if embedding_plan:
            if (
                embedding_plan.sparse
                and embedding_plan.sparse.profile_name
                != embedding_plan.dense.profile_name
            ):
                self.sparse_embedder = (
                    ProviderFactory.create_embedding_provider_for_role(
                        embedding_plan.sparse
                    )
                )
            if embedding_plan.colbert:
                if (
                    embedding_plan.sparse
                    and embedding_plan.colbert.profile_name
                    == embedding_plan.sparse.profile_name
                ):
                    self.colbert_embedder = self.sparse_embedder
                elif (
                    embedding_plan.colbert.profile_name
                    != embedding_plan.dense.profile_name
                ):
                    self.colbert_embedder = (
                        ProviderFactory.create_embedding_provider_for_role(
                            embedding_plan.colbert
                        )
                    )
        self.collection = collection_name
        self.rrf_k = rrf_k
        self.embedding_settings = embedding_settings
        self.embedding_version = (
            embedding_settings.version if embedding_settings else None
        )
        self.use_query_api = use_query_api
        self.query_api_weighted_fusion = query_api_weighted_fusion
        self.multi_vector_fusion_method = multi_vector_fusion_method
        self.query_api_dense_limit = query_api_dense_limit
        self.query_api_sparse_limit = query_api_sparse_limit
        self.query_api_candidate_limit = query_api_candidate_limit
        self.primary_vector_name = primary_vector_name or "content"
        # Prefer explicit flags; otherwise infer from embedding capabilities
        caps = getattr(embedding_settings, "capabilities", None)
        if self.embedding_plan:
            caps_sparse = bool(
                self.embedding_plan.sparse
                and self.embedding_plan.sparse.profile.capabilities.supports_sparse
            )
            caps_colbert = bool(
                self.embedding_plan.colbert
                and self.embedding_plan.colbert.profile.capabilities.supports_colbert
            )
        else:
            caps_sparse = bool(getattr(caps, "supports_sparse", False))
            caps_colbert = bool(getattr(caps, "supports_colbert", False))
        self.schema_supports_sparse = bool(schema_supports_sparse and caps_sparse)
        self.schema_supports_colbert = bool(schema_supports_colbert and caps_colbert)
        # doc_title-sparse: only enable if general sparse support is also enabled
        self.schema_supports_doc_title_sparse = (
            schema_supports_doc_title_sparse and self.schema_supports_sparse
        )
        # NEW: title-sparse and entity-sparse support flags
        self.schema_supports_title_sparse = (
            schema_supports_title_sparse and self.schema_supports_sparse
        )
        self.schema_supports_entity_sparse = (
            schema_supports_entity_sparse and self.schema_supports_sparse
        )
        # NEW: RRF debug logging flag
        self.rrf_debug_logging = rrf_debug_logging
        # NEW: RRF per-field weights (default 1.0 for all fields)
        self.rrf_field_weights = rrf_field_weights or {}
        if self.schema_supports_colbert and not self.use_query_api:
            message = (
                "Configuration error: ColBERT enabled but Query API is disabled. "
                "Enable search.vector.qdrant.use_query_api or disable enable_colbert."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Auto-enabling Query API in non-strict env.", message)
            self.use_query_api = True
        # Capability guardrails
        if self.schema_supports_sparse and not (
            hasattr(self.sparse_embedder, "embed_sparse")
            or hasattr(self.sparse_embedder, "embed_query_all")
        ):
            message = (
                "Configuration error: Sparse vectors enabled but embedder "
                "does not support embed_sparse."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Disabling sparse support in non-strict env.", message)
            self.schema_supports_sparse = False
            self.schema_supports_doc_title_sparse = False
        if self.schema_supports_colbert and not hasattr(
            self.colbert_embedder, "embed_query_all"
        ):
            message = (
                "Configuration error: ColBERT enabled but embedder "
                "does not support embed_query_all/multivector."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Disabling ColBERT in non-strict env.", message)
            self.schema_supports_colbert = False
        self.last_stats: Dict[str, Any] = {}
        self.field_weights = {
            name: float(weight)
            for name, weight in (field_weights or {"content": 1.0}).items()
            if float(weight) > 0
        }
        if not self.field_weights:
            self.field_weights = {"content": 1.0}
        self.payload_keys = payload_keys or [
            "id",
            "node_id",
            "document_id",
            "doc_id",
            "parent_section_id",
            "order",
            "level",
            "heading",
            "text",
            "token_count",
            "doc_tag",
            "document_total_tokens",
            "source_path",
            "is_microdoc",
            "embedding_version",
            "embedding_provider",
            "embedding_dimensions",
            "tenant",
            "lang",
            "version",
            "boundaries_json",
            "original_section_ids",
            "text_hash",
            "shingle_hash",
            # GLiNER entity metadata for Phase 4 entity-aware retrieval
            "entity_metadata",
        ]
        self.supports_sparse = (
            hasattr(self.sparse_embedder, "embed_sparse")
            and self.schema_supports_sparse
        )
        self.supports_colbert = (
            hasattr(self.colbert_embedder, "embed_colbert")
            and self.schema_supports_colbert
        )
        self.sparse_query_name = "text-sparse"
        self.sparse_field_name = None
        for field_name in self.field_weights.keys():
            lname = field_name.lower()
            if lname in {"lexical", "sparse"} or field_name == self.sparse_query_name:
                self.sparse_field_name = field_name
                break
        if self.sparse_field_name and not self.supports_sparse:
            self.field_weights[self.sparse_field_name] = 0.0
        self.dense_vector_names = [
            name for name in self.field_weights.keys() if name != self.sparse_field_name
        ] or ["content"]

    def get_queried_vector_fields(self) -> List[str]:
        """Return list of vector fields actually queried based on schema_supports_* flags.

        This provides accurate metrics about which vectors participate in multi-vector
        fusion, rather than the legacy vector_field_weights config which is only used
        for weighted (non-RRF) fusion.

        Returns:
            List of vector field names that are included in Prefetch queries.
        """
        fields = []
        # Dense vectors (always included if in field_weights)
        for name in self.dense_vector_names:
            if name in self.field_weights:
                fields.append(name)
        # Sparse vectors based on schema support flags
        if self.schema_supports_sparse:
            fields.append("text-sparse")
        if self.schema_supports_doc_title_sparse:
            fields.append("doc_title-sparse")
        if self.schema_supports_title_sparse:
            fields.append("title-sparse")
        if self.schema_supports_entity_sparse:
            fields.append("entity-sparse")
        return fields

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        ef: Optional[int] = 256,
        lexical_query: Optional[str] = None,
    ) -> List[ChunkResult]:
        self.last_stats = {"path": "legacy"}
        if self.use_query_api and self._query_api_supported():
            try:
                bundle = self._build_query_bundle(query, lexical_query=lexical_query)
                if self.query_api_weighted_fusion:
                    return self._search_via_query_api_weighted(bundle, top_k, filters)
                return self._search_via_query_api(bundle, top_k, filters)
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "Query API search failed; falling back to legacy search",
                    error=str(exc),
                )
                self.last_stats = {
                    "path": "legacy",
                    "fallback_reason": str(exc),
                }
        return self._search_legacy(
            query, top_k, filters, ef, lexical_query=lexical_query
        )

    def _search_legacy(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        ef: Optional[int],
        lexical_query: Optional[str] = None,
    ) -> List[ChunkResult]:
        start_time = time.time()
        query_vectors = self._build_query_vectors(query, lexical_query=lexical_query)
        qdrant_filter = self._build_filter(filters)

        rankings: Dict[str, List[Tuple[str, float]]] = {}
        payload_by_id: Dict[str, Dict[str, Any]] = {}
        vec_score_by_id: Dict[Tuple[str, str], float] = {}

        for vector_name, vector_kind, vector in query_vectors:
            weight = self.field_weights.get(vector_name, 0.0)
            if weight <= 0:
                continue
            if vector_kind == "sparse":
                hits = self._search_sparse(vector, top_k, qdrant_filter)
            else:
                hits = self._search_single(
                    vector_name, vector, top_k, qdrant_filter, ef
                )
            hits_sorted = sorted(hits, key=lambda h: h.score or 0.0, reverse=True)
            rankings[vector_name] = []
            for hit in hits_sorted:
                pid = str(hit.id)
                score = float(hit.score or 0.0)
                rankings[vector_name].append((pid, score))
                vec_score_by_id[(pid, vector_name)] = score
                if pid not in payload_by_id:
                    payload = dict(hit.payload or {})
                    if self.payload_keys:
                        payload = {k: payload.get(k) for k in self.payload_keys}
                    payload_by_id[pid] = payload

        fused = self._fuse_rankings(rankings)
        results: List[ChunkResult] = []
        for pid, fused_score in sorted(
            fused.items(), key=lambda kv: kv[1], reverse=True
        ):
            payload = payload_by_id.get(pid, {})
            chunk = self._chunk_from_payload(
                pid,
                payload,
                fused_score=float(fused_score),
                vector_score=float(fused_score),
                title_vec_score=vec_score_by_id.get((pid, "title"), 0.0),
                doc_title_vec_score=vec_score_by_id.get((pid, "doc_title"), 0.0),
                doc_title_sparse_score=vec_score_by_id.get(
                    (pid, "doc_title-sparse"), 0.0
                ),
                title_sparse_score=vec_score_by_id.get((pid, "title-sparse"), 0.0),
                entity_vec_score=vec_score_by_id.get((pid, "entity-sparse"), 0.0),
                lexical_vec_score=(
                    vec_score_by_id.get((pid, self.sparse_field_name), 0.0)
                    if self.sparse_field_name
                    else None
                ),
            )
            chunk.fusion_method = "weighted"
            chunk.vector_score_kind = "weighted_fusion"
            results.append(chunk)

        elapsed_ms = (time.time() - start_time) * 1000
        self.last_stats = {
            "path": "legacy",
            "duration_ms": elapsed_ms,
            "results": len(results),
        }
        logger.info(
            "Multi-vector search completed (legacy path)",
            fields=list(self.field_weights.keys()),
            results=len(results),
            time_ms=f"{elapsed_ms:.2f}",
        )
        return results

    def search_named_vector(
        self,
        vector_name: str,
        vector: Sequence[float],
        limit: int,
        query_filter: Optional[QdrantFilter] = None,
        score_threshold: Optional[float] = None,
    ):
        try:
            result = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                using=vector_name,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=score_threshold,
            )
            return result.points
        except Exception as exc:
            logger.debug(
                "Named vector search failed", field=vector_name, error=str(exc)
            )
            return []

    def _build_query_vectors(
        self,
        query: str,
        lexical_query: Optional[str] = None,
    ) -> List[Tuple[str, str, Sequence[float]]]:
        base_vector = self.embedder.embed_query(query)
        sparse_q = lexical_query or query
        vectors: List[Tuple[str, str, Sequence[float]]] = []
        for vector_name in self.field_weights.keys():
            if vector_name == self.sparse_field_name:
                sparse_vector = self._build_sparse_query(sparse_q)
                if sparse_vector:
                    vectors.append((vector_name, "sparse", sparse_vector))
                continue
            vectors.append((vector_name, "dense", base_vector))
        return vectors

    def _build_sparse_query(self, query: str) -> Optional[Dict[str, List[float]]]:
        if not self.supports_sparse or not hasattr(
            self.sparse_embedder, "embed_sparse"
        ):
            return None
        try:
            sparse_vectors = self.sparse_embedder.embed_sparse([query])
            if sparse_vectors:
                sparse_vector = sparse_vectors[0]
                indices = (
                    sparse_vector.get("indices")
                    if isinstance(sparse_vector, dict)
                    else None
                )
                values = (
                    sparse_vector.get("values")
                    if isinstance(sparse_vector, dict)
                    else None
                )
                if indices and values:
                    return sparse_vector
        except Exception as exc:
            logger.warning(
                "Sparse query embedding failed; skipping lexical search leg",
                error=str(exc),
            )
        return None

    def _build_filter(
        self, filters: Optional[Dict[str, Any]]
    ) -> Optional[QdrantFilter]:
        must: List[FieldCondition] = []
        filters = dict(filters or {})
        filters.pop("embedding_version", None)
        if self.embedding_version:
            must.append(
                FieldCondition(
                    key="embedding_version",
                    match=MatchValue(value=self.embedding_version),
                )
            )
        for key, value in filters.items() if filters else []:
            if isinstance(value, list):
                if not value:
                    continue
                try:
                    must.append(FieldCondition(key=key, match=MatchAny(any=value)))
                except Exception:
                    # Fallback: include as OR of separate matches
                    for v in value:
                        must.append(FieldCondition(key=key, match=MatchValue(value=v)))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return QdrantFilter(must=must) if must else None

    def _search_single(
        self,
        vector_name: str,
        vector: Sequence[float],
        top_k: int,
        query_filter: Optional[QdrantFilter],
        ef: Optional[int],
    ):
        search_params = None
        try:
            from qdrant_client.http.models import SearchParams

            search_params = SearchParams(hnsw_ef=ef) if ef else None
        except Exception:
            search_params = None

        result = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=vector_name,
            limit=top_k,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
            with_vectors=False,
        )
        return result.points

    def _search_sparse(
        self,
        sparse_vector: Dict[str, List[float]],
        top_k: int,
        query_filter: Optional[QdrantFilter],
    ):
        if not sparse_vector:
            return []
        indices = (
            sparse_vector.get("indices") if isinstance(sparse_vector, dict) else None
        )
        values = (
            sparse_vector.get("values") if isinstance(sparse_vector, dict) else None
        )
        if not indices or not values:
            return []
        try:
            sparse_query = QdrantSparseVector(indices=indices, values=values)
        except Exception as exc:
            logger.debug(
                "Failed to construct SparseVector; skipping sparse search",
                error=str(exc),
            )
            return []
        result = self.client.query_points(
            collection_name=self.collection,
            query=sparse_query,
            using=self.sparse_query_name,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        return result.points

    def _chunk_from_payload(
        self,
        pid: str,
        payload: Dict[str, Any],
        *,
        fused_score: Optional[float] = None,
        vector_score: Optional[float] = None,
        title_vec_score: Optional[float] = None,
        doc_title_vec_score: Optional[float] = None,
        doc_title_sparse_score: Optional[float] = None,
        title_sparse_score: Optional[float] = None,
        entity_vec_score: Optional[float] = None,
        lexical_vec_score: Optional[float] = None,
    ) -> ChunkResult:
        chunk_id = payload.get("id") or payload.get("node_id") or pid
        doc_id = payload.get("document_id") or payload.get("doc_id") or ""
        original_ids = payload.get("original_section_ids") or []
        return ChunkResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            parent_section_id=payload.get("parent_section_id", ""),
            order=payload.get("order", 0),
            level=payload.get("level", 3),
            heading=payload.get("heading", ""),
            text=payload.get("text", ""),
            token_count=payload.get("token_count", 0),
            is_combined=payload.get("is_combined", False),
            is_split=payload.get("is_split", False),
            original_section_ids=list(original_ids),
            boundaries_json=payload.get("boundaries_json", "{}"),
            doc_tag=payload.get("doc_tag"),
            snapshot_scope=payload.get("snapshot_scope"),
            document_total_tokens=payload.get("document_total_tokens", 0),
            source_path=payload.get("source_path"),
            is_microdoc=payload.get("is_microdoc", False),
            doc_is_microdoc=payload.get("doc_is_microdoc", False),
            is_microdoc_stub=payload.get("is_microdoc_stub", False),
            embedding_version=payload.get("embedding_version"),
            tenant=payload.get("tenant"),
            fused_score=fused_score,
            vector_score=vector_score,
            title_vec_score=title_vec_score,
            doc_title_vec_score=doc_title_vec_score,
            doc_title_sparse_score=doc_title_sparse_score,
            title_sparse_score=title_sparse_score,
            entity_vec_score=entity_vec_score,
            lexical_vec_score=lexical_vec_score,
            citation_labels=payload.get("citation_labels") or [],
            # Phase 4: Entity metadata for GLiNER-aware retrieval boosting
            # Deduplicate to clean up repeated entity extractions from same chunk
            entity_metadata=_deduplicate_entity_metadata(
                payload.get("entity_metadata")
            ),
            # Phase 5: Structural metadata for query-type adaptive boosting
            structural_metadata={
                "has_code": payload.get("has_code", False),
                "has_table": payload.get("has_table", False),
                "parent_path_depth": payload.get("parent_path_depth", 0),
                "block_type": payload.get("block_type", "paragraph"),
                "code_ratio": payload.get("code_ratio", 0.0),
            },
        )

    def _build_query_bundle(
        self,
        query: str,
        lexical_query: Optional[str] = None,
    ) -> QueryEmbeddingBundle:
        """Build embedding bundle for query.

        Args:
            query: Reformulated query for dense/ColBERT embeddings.
            lexical_query: Original keywords for sparse embeddings. When None,
                ``query`` is used for all embeddings (backward compat).
        """
        from src.providers.embeddings.embedding_service import (
            _to_multivector,
            _to_sparse_embedding,
        )

        sparse_query = lexical_query or query
        dense = self.embedder.embed_query(query)
        sparse = None
        multivector = None

        # When sparse and colbert are the SAME provider (e.g., BGE-M3),
        # use embed_query_all for efficiency (one call, all heads).
        # When they're DIFFERENT providers (e.g., SPLADEv3 + ColBERTv2),
        # call each independently with the appropriate query form.
        same_provider = self.sparse_embedder is self.colbert_embedder

        if same_provider and hasattr(self.colbert_embedder, "embed_query_all"):
            # Same provider produces both — use reformulated query
            # (can't split sparse/colbert when they share a provider)
            bundle = self.colbert_embedder.embed_query_all(query)
            if self.schema_supports_sparse:
                sparse = bundle.sparse
            if self.schema_supports_colbert:
                multivector = bundle.multivector
        else:
            # Separate providers — sparse gets lexical keywords,
            # ColBERT gets reformulated query (semantic token matching)
            if self.schema_supports_sparse and hasattr(
                self.sparse_embedder, "embed_sparse"
            ):
                sparse_list = self.sparse_embedder.embed_sparse([sparse_query])
                if sparse_list:
                    sparse = _to_sparse_embedding(sparse_list[0])

            if self.schema_supports_colbert and hasattr(
                self.colbert_embedder, "embed_colbert"
            ):
                colbert_list = self.colbert_embedder.embed_colbert([query])
                if colbert_list:
                    multivector = _to_multivector(colbert_list[0])

        return QueryEmbeddingBundle(
            dense=list(dense),
            sparse=sparse,
            multivector=multivector,
        )

    def _search_via_query_api(
        self,
        bundle: QueryEmbeddingBundle,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        start_time = time.time()
        qdrant_filter = self._build_filter(filters)
        prefetch_entries = self._build_prefetch_entries(bundle, qdrant_filter, top_k)
        prefetch_arg = None
        if prefetch_entries:
            fusion_mode = getattr(Fusion, "DBSF", Fusion.RRF)
            prefetch_arg = Prefetch(
                prefetch=prefetch_entries,
                query=FusionQuery(fusion=fusion_mode),
                limit=max(top_k, self.query_api_candidate_limit),
                filter=qdrant_filter,
            )
        query_payload, using_name = self._build_query_api_query(bundle)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_payload,
            using=using_name,
            prefetch=prefetch_arg,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )
        points = getattr(response, "points", response)
        results: List[ChunkResult] = []
        for idx, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            if self.payload_keys:
                payload = {k: payload.get(k) for k in self.payload_keys}
            chunk = self._chunk_from_payload(
                str(point.id),
                payload,
                fused_score=float(point.score or 0.0),
                vector_score=float(point.score or 0.0),
            )
            chunk.vector_rank = idx
            chunk.vector_score_kind = "similarity"
            chunk.fusion_method = "weighted"
            results.append(chunk)

        elapsed_ms = (time.time() - start_time) * 1000
        self.last_stats = {
            "path": "query_api",
            "duration_ms": elapsed_ms,
            "prefetch_count": len(prefetch_entries),
            "results": len(results),
            "colbert_used": using_name == "late-interaction",
            "sparse_prefetch": any(
                getattr(entry, "using", "") == self.sparse_query_name
                for entry in (prefetch_entries or [])
            ),
        }
        logger.info(
            "Multi-vector search completed via Query API",
            results=len(results),
            time_ms=f"{elapsed_ms:.2f}",
        )
        return results

    def _build_id_filter(
        self, base_filter: Optional[QdrantFilter], candidate_ids: List[str]
    ) -> QdrantFilter:
        """Augment an existing filter with an ID constraint."""
        id_condition = HasIdCondition(has_id=list(candidate_ids))
        if base_filter is None:
            return QdrantFilter(must=[id_condition])
        must = list(getattr(base_filter, "must", []) or [])
        must.append(id_condition)
        return QdrantFilter(
            must=must,
            must_not=getattr(base_filter, "must_not", None),
            should=getattr(base_filter, "should", None),
        )

    def _search_via_query_api_weighted(
        self,
        bundle: QueryEmbeddingBundle,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        """
        Two-stage Query API search with Python-side weighted fusion (Strategy 2).
        Stage 1: Query API + prefetch for candidate recall.
        Stage 2: Per-field scoring on candidates, fused via _fuse_rankings.
        """
        start_time = time.time()
        qdrant_filter = self._build_filter(filters)
        prefetch_entries = self._build_prefetch_entries(bundle, qdrant_filter, top_k)
        prefetch_arg = None
        if prefetch_entries:
            fusion_mode = getattr(Fusion, "DBSF", Fusion.RRF)
            prefetch_arg = Prefetch(
                prefetch=prefetch_entries,
                query=FusionQuery(fusion=fusion_mode),
                limit=max(top_k, self.query_api_candidate_limit),
                filter=qdrant_filter,
            )

        query_payload, using_name = self._build_query_api_query(bundle)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_payload,
            using=using_name,
            prefetch=prefetch_arg,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
            limit=max(top_k, self.query_api_candidate_limit),
        )
        points = getattr(response, "points", response) or []
        if not points:
            return []

        candidate_ids = [str(p.id) for p in points]
        # Limit how many candidates we rescore per-field to bound latency
        scoring_cap = max(top_k * 3, self.query_api_candidate_limit)
        scoring_ids = candidate_ids[: min(len(candidate_ids), scoring_cap)]
        rankings: Dict[str, List[Tuple[str, float]]] = {}
        vec_score_by_id: Dict[Tuple[str, str], float] = {}

        # Dense fields
        for field_name in self.dense_vector_names:
            field_hits = self._search_named_vector_candidates(
                bundle.dense, field_name, scoring_ids, qdrant_filter
            )
            rankings[field_name] = [
                (str(hit.id), float(hit.score or 0.0)) for hit in field_hits
            ]
            # Track individual vector scores for debugging and analysis
            for hit in field_hits:
                vec_score_by_id[(str(hit.id), field_name)] = float(hit.score or 0.0)

        # Sparse field (text-sparse)
        if (
            self.supports_sparse
            and bundle.sparse
            and bundle.sparse.indices
            and bundle.sparse.values
        ):
            sparse_hits = self._search_sparse_candidates(
                bundle.sparse.indices,
                bundle.sparse.values,
                scoring_ids,
                qdrant_filter,
            )
            rankings[self.sparse_query_name] = [
                (str(hit.id), float(hit.score or 0.0)) for hit in sparse_hits
            ]
            for hit in sparse_hits:
                vec_score_by_id[(str(hit.id), self.sparse_query_name)] = float(
                    hit.score or 0.0
                )

        # doc_title-sparse prefetch is handled in _build_prefetch_entries
        # but we need to score it separately if enabled
        if (
            self.schema_supports_doc_title_sparse
            and bundle.sparse
            and bundle.sparse.indices
            and bundle.sparse.values
        ):
            try:
                doc_title_sparse_hits = self._search_sparse_candidates(
                    bundle.sparse.indices,
                    bundle.sparse.values,
                    scoring_ids,
                    qdrant_filter,
                    sparse_vector_name="doc_title-sparse",
                )
                rankings["doc_title-sparse"] = [
                    (str(hit.id), float(hit.score or 0.0))
                    for hit in doc_title_sparse_hits
                ]
                for hit in doc_title_sparse_hits:
                    vec_score_by_id[(str(hit.id), "doc_title-sparse")] = float(
                        hit.score or 0.0
                    )
            except Exception as e:
                # doc_title-sparse may not exist in older collections
                # Log at debug level to aid troubleshooting without noise
                logger.debug(
                    "sparse_scoring_skipped",
                    sparse_field="doc_title-sparse",
                    reason="field_unavailable_or_error",
                    error_type=type(e).__name__,
                    error_msg=str(e)[:100],
                )

        # NEW: title-sparse scoring - lexical heading matching
        if (
            self.schema_supports_title_sparse
            and bundle.sparse
            and bundle.sparse.indices
            and bundle.sparse.values
        ):
            try:
                title_sparse_hits = self._search_sparse_candidates(
                    bundle.sparse.indices,
                    bundle.sparse.values,
                    scoring_ids,
                    qdrant_filter,
                    sparse_vector_name="title-sparse",
                )
                rankings["title-sparse"] = [
                    (str(hit.id), float(hit.score or 0.0)) for hit in title_sparse_hits
                ]
                for hit in title_sparse_hits:
                    vec_score_by_id[(str(hit.id), "title-sparse")] = float(
                        hit.score or 0.0
                    )
            except Exception as e:
                # title-sparse may not exist in older collections
                # Log at debug level to aid troubleshooting without noise
                logger.debug(
                    "sparse_scoring_skipped",
                    sparse_field="title-sparse",
                    reason="field_unavailable_or_error",
                    error_type=type(e).__name__,
                    error_msg=str(e)[:100],
                )

        # NEW: entity-sparse scoring - lexical entity name matching
        if (
            self.schema_supports_entity_sparse
            and bundle.sparse
            and bundle.sparse.indices
            and bundle.sparse.values
        ):
            try:
                entity_sparse_hits = self._search_sparse_candidates(
                    bundle.sparse.indices,
                    bundle.sparse.values,
                    scoring_ids,
                    qdrant_filter,
                    sparse_vector_name="entity-sparse",
                )
                rankings["entity-sparse"] = [
                    (str(hit.id), float(hit.score or 0.0)) for hit in entity_sparse_hits
                ]
                for hit in entity_sparse_hits:
                    vec_score_by_id[(str(hit.id), "entity-sparse")] = float(
                        hit.score or 0.0
                    )
            except Exception as e:
                # entity-sparse may not exist in older collections
                # Log at debug level to aid troubleshooting without noise
                logger.debug(
                    "sparse_scoring_skipped",
                    sparse_field="entity-sparse",
                    reason="field_unavailable_or_error",
                    error_type=type(e).__name__,
                    error_msg=str(e)[:100],
                )

        sparse_scored = len(rankings.get(self.sparse_query_name, []))

        # Choose fusion method based on config
        if self.multi_vector_fusion_method == "rrf":
            fused_scores = self._fuse_rankings_rrf(rankings, k=self.rrf_k)
            fusion_method_used = "rrf"
            # NEW: Log per-field contributions for debugging
            if self.rrf_debug_logging:
                self._log_rrf_field_contributions(
                    rankings,
                    fused_scores,
                    top_k=min(top_k, 10),  # Limit log size to top 10
                    k=self.rrf_k,
                )
        else:
            fused_scores = self._fuse_rankings(rankings)
            fusion_method_used = "weighted"

        sparse_ids = {
            pid
            for pid, _ in rankings.get(self.sparse_query_name, [])  # type: ignore[arg-type]
        }

        results: List[ChunkResult] = []
        for idx, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            if self.payload_keys:
                payload = {k: payload.get(k) for k in self.payload_keys}
            pid = str(point.id)
            fused_score = float(fused_scores.get(pid, point.score or 0.0))
            chunk = self._chunk_from_payload(
                pid,
                payload,
                fused_score=fused_score,
                vector_score=fused_score,
                title_vec_score=vec_score_by_id.get((pid, "title"), 0.0),
                doc_title_vec_score=vec_score_by_id.get((pid, "doc_title"), 0.0),
                doc_title_sparse_score=vec_score_by_id.get(
                    (pid, "doc_title-sparse"), 0.0
                ),
                title_sparse_score=vec_score_by_id.get((pid, "title-sparse"), 0.0),
                entity_vec_score=vec_score_by_id.get((pid, "entity-sparse"), 0.0),
                lexical_vec_score=(
                    vec_score_by_id.get((pid, self.sparse_query_name), 0.0)
                    if self.sparse_query_name
                    else None
                ),
            )
            chunk.vector_rank = idx
            chunk.vector_score_kind = f"{fusion_method_used}_fusion"
            chunk.fusion_method = fusion_method_used
            # Populate RRF field contributions when debug logging is enabled
            if self.rrf_debug_logging and fusion_method_used == "rrf":
                chunk.rrf_field_contributions = (
                    self._compute_rrf_contributions_for_chunk(
                        pid, rankings, k=self.rrf_k
                    )
                )
            results.append(chunk)

        results.sort(key=lambda x: x.fused_score or 0.0, reverse=True)
        results = results[:top_k]

        # Compute sparse coverage among top-k results (Phase B.4)
        sparse_in_topk = 0
        if sparse_ids and results:
            for r in results:
                if str(r.chunk_id) in sparse_ids:
                    sparse_in_topk += 1

        elapsed_ms = (time.time() - start_time) * 1000
        self.last_stats = {
            "path": "query_api_weighted",
            "duration_ms": elapsed_ms,
            "prefetch_count": len(prefetch_entries),
            "results": len(results),
            "candidates": len(points),
            "scored_candidates": len(scoring_ids),
            "sparse_scored": sparse_scored,
            "sparse_scored_ratio": (
                (sparse_scored / len(scoring_ids)) if scoring_ids else 0.0
            ),
            "sparse_in_topk": sparse_in_topk,
            "sparse_topk_ratio": (sparse_in_topk / len(results)) if results else 0.0,
            "sparse_prefetch": any(
                getattr(entry, "using", "") == self.sparse_query_name
                for entry in (prefetch_entries or [])
            ),
        }
        logger.info(
            "Multi-vector search completed via Query API (weighted)",
            extra={
                "results": len(results),
                "time_ms": f"{elapsed_ms:.2f}",
                "candidates": len(points),
                "scored_candidates": len(scoring_ids),
            },
        )
        return results

    def _build_prefetch_entries(
        self,
        bundle: QueryEmbeddingBundle,
        qdrant_filter: Optional[QdrantFilter],
        top_k: int,
    ) -> List[Prefetch]:
        entries: List[Prefetch] = []
        candidate_limit = max(top_k, self.query_api_candidate_limit)
        dense_limit = min(candidate_limit, self.query_api_dense_limit)
        dense_vector = list(bundle.dense)
        for field_name in self.dense_vector_names:
            entries.append(
                Prefetch(
                    query=dense_vector,
                    using=field_name,
                    limit=dense_limit,
                    filter=qdrant_filter,
                )
            )
        if (
            self.schema_supports_sparse
            and bundle.sparse
            and bundle.sparse.indices
            and bundle.sparse.values
        ):
            sparse_query = QdrantSparseVector(
                indices=list(bundle.sparse.indices), values=list(bundle.sparse.values)
            )
            # text-sparse: BM25 matching against chunk content
            entries.append(
                Prefetch(
                    query=sparse_query,
                    using=self.sparse_query_name,
                    limit=min(candidate_limit, self.query_api_sparse_limit),
                    filter=qdrant_filter,
                )
            )
            # doc_title-sparse: BM25 matching against document titles
            # Uses same sparse embedding but searches title text index
            if self.schema_supports_doc_title_sparse:
                entries.append(
                    Prefetch(
                        query=sparse_query,
                        using="doc_title-sparse",
                        limit=100,  # Raised from 50: eliminates truncation bias at k=30
                        filter=qdrant_filter,
                    )
                )
            # NEW: title-sparse - lexical matching for section headings
            if self.schema_supports_title_sparse:
                entries.append(
                    Prefetch(
                        query=sparse_query,
                        using="title-sparse",
                        limit=100,  # Raised from 50: eliminates truncation bias at k=30
                        filter=qdrant_filter,
                    )
                )
            # NEW: entity-sparse - lexical matching for entity names
            if self.schema_supports_entity_sparse:
                entries.append(
                    Prefetch(
                        query=sparse_query,
                        using="entity-sparse",
                        limit=100,  # Raised from 50: eliminates truncation bias at k=30
                        filter=qdrant_filter,
                    )
                )
        return entries

    def _search_named_vector_candidates(
        self,
        vector: Sequence[float],
        vector_name: str,
        candidate_ids: List[str],
        base_filter: Optional[QdrantFilter],
    ):
        if not candidate_ids:
            return []
        try:
            id_filter = self._build_id_filter(base_filter, candidate_ids)
            result = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                using=vector_name,
                limit=len(candidate_ids),
                query_filter=id_filter,
                with_payload=False,
                with_vectors=False,
            )
            return getattr(result, "points", result) or []
        except Exception as exc:
            logger.debug(
                "Named vector candidate scoring failed",
                extra={"field": vector_name, "error": str(exc)},
            )
            return []

    def _search_sparse_candidates(
        self,
        indices: Sequence[int],
        values: Sequence[float],
        candidate_ids: List[str],
        base_filter: Optional[QdrantFilter],
        sparse_vector_name: Optional[str] = None,
    ):
        if not candidate_ids or not indices or not values:
            return []
        try:
            sparse_query = QdrantSparseVector(
                indices=list(indices), values=list(values)
            )
            id_filter = self._build_id_filter(base_filter, candidate_ids)
            # Use provided name or default to text-sparse
            using = sparse_vector_name or self.sparse_query_name
            result = self.client.query_points(
                collection_name=self.collection,
                query=sparse_query,
                using=using,
                limit=len(candidate_ids),
                query_filter=id_filter,
                with_payload=False,
                with_vectors=False,
            )
            return getattr(result, "points", result) or []
        except Exception as exc:
            logger.debug("Sparse candidate scoring failed", extra={"error": str(exc)})
            return []

    def _build_query_api_query(
        self, bundle: QueryEmbeddingBundle
    ) -> Tuple[Sequence[Sequence[float]] | Sequence[float], str]:
        return list(bundle.dense), self.primary_vector_name

    def _query_api_supported(self) -> bool:
        return hasattr(self.client, "query_points")

    def _fuse_rankings(
        self, rankings: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, float]:
        fused: Dict[str, float] = defaultdict(float)
        max_by_field: Dict[str, float] = {}
        for vector_name, items in rankings.items():
            if not items:
                continue
            scores = [score for _, score in items if score is not None]
            if not scores:
                continue
            max_by_field[vector_name] = max(scores) or 0.0

        for vector_name, items in rankings.items():
            weight = self.field_weights.get(vector_name, 1.0)
            if weight <= 0:
                continue
            max_score = max_by_field.get(vector_name) or 0.0
            for pid, raw_score in items:
                if raw_score is None:
                    continue
                normalized = (raw_score / max_score) if max_score > 0 else 0.0
                fused[pid] += weight * normalized
        return fused

    def _fuse_rankings_rrf(
        self, rankings: Dict[str, List[Tuple[str, float]]], k: int = 60
    ) -> Dict[str, float]:
        """
        Weighted Reciprocal Rank Fusion across multiple vector fields.

        RRF score = Σ weight_i * 1/(k + rank_i) for each field where the document appears.
        Uses rank position only, not score magnitude - robust to scale differences.
        Field weights allow boosting specific signals (e.g., title-sparse for heading matches).

        Args:
            rankings: Dict mapping field name to list of (doc_id, score) tuples
            k: RRF constant (default 60) - higher k reduces impact of top ranks

        Returns:
            Dict mapping doc_id to fused RRF score
        """
        fused: Dict[str, float] = defaultdict(float)

        for field_name, items in rankings.items():
            if not items:
                continue
            # Get field weight (default 1.0 if not specified)
            weight = self.rrf_field_weights.get(field_name, 1.0)
            # Sort by score descending to establish ranks
            sorted_items = sorted(items, key=lambda x: x[1] or 0.0, reverse=True)
            for rank, (doc_id, _score) in enumerate(sorted_items, start=1):
                fused[doc_id] += weight * 1.0 / (k + rank)

        return fused

    def _log_rrf_field_contributions(
        self,
        rankings: Dict[str, List[Tuple[str, float]]],
        fused_scores: Dict[str, float],
        top_k: int = 10,
        k: int = 60,
    ) -> None:
        """Log per-field RRF contributions for debugging.

        Shows how each field contributed to the final fused score for top results.
        Helps diagnose which fields are providing useful signal vs noise.

        Weighted RRF formula: score(d) = Σ weight_i * 1/(k + rank_i(d)) for each field i

        Args:
            rankings: Dict mapping field name to list of (doc_id, score) tuples
            fused_scores: Dict mapping doc_id to fused RRF score
            top_k: Number of top results to log details for
            k: RRF constant (default 60)

        Log output example:
            {
                "chunk_id": "section_12345...",
                "fused_score": 0.05892,
                "fields": {
                    "content": {"rank": 2, "contribution": 0.01613},
                    "title": {"rank": 5, "contribution": 0.01538},
                    "text-sparse": {"rank": 1, "contribution": 0.01639},
                    "title-sparse": {"rank": 3, "contribution": 0.01587}
                }
            }
        """
        if not fused_scores:
            return

        # Sort by fused score to get top results
        sorted_ids = sorted(
            fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
        )[:top_k]

        # Build per-field rank lookup
        field_ranks: Dict[str, Dict[str, int]] = {}
        for field_name, items in rankings.items():
            if not items:
                continue
            sorted_items = sorted(items, key=lambda x: x[1] or 0.0, reverse=True)
            field_ranks[field_name] = {
                doc_id: rank for rank, (doc_id, _) in enumerate(sorted_items, start=1)
            }

        # Build detailed breakdown for each top result
        rrf_details = []
        for doc_id in sorted_ids:
            contributions = {}
            total_contribution = 0.0

            for field_name in rankings.keys():
                rank = field_ranks.get(field_name, {}).get(doc_id)
                if rank:
                    weight = self.rrf_field_weights.get(field_name, 1.0)
                    contribution = weight * 1.0 / (k + rank)
                    total_contribution += contribution
                    contributions[field_name] = {
                        "rank": rank,
                        "weight": weight,
                        "contribution": round(contribution, 5),
                    }

            rrf_details.append(
                {
                    "chunk_id": doc_id[:20] + "..." if len(doc_id) > 20 else doc_id,
                    "fused_score": round(fused_scores[doc_id], 5),
                    "computed_sum": round(total_contribution, 5),
                    "fields": contributions,
                }
            )

        logger.info(
            "rrf_field_contributions",
            extra={
                "event": "rrf_fusion_debug",
                "top_k_shown": len(rrf_details),
                "rrf_k": k,
                "fields_used": list(rankings.keys()),
                "field_counts": {k: len(v) for k, v in rankings.items()},
                "details": rrf_details,
            },
        )

    def _compute_rrf_contributions_for_chunk(
        self,
        chunk_id: str,
        rankings: Dict[str, List[Tuple[str, float]]],
        k: int = 60,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-field RRF contributions for a single chunk.

        Used to populate ChunkResult.rrf_field_contributions when rrf_debug_logging=true.

        Args:
            chunk_id: The chunk ID to compute contributions for
            rankings: Dict mapping field name to list of (doc_id, score) tuples
            k: RRF constant (default 60)

        Returns:
            Dict mapping field_name to {"rank": int, "weight": float, "contribution": float}
        """
        contributions: Dict[str, Dict[str, float]] = {}

        for field_name, items in rankings.items():
            if not items:
                continue
            # Sort by score descending to establish ranks
            sorted_items = sorted(items, key=lambda x: x[1] or 0.0, reverse=True)
            # Find rank of this chunk in this field
            for rank, (doc_id, _score) in enumerate(sorted_items, start=1):
                if doc_id == chunk_id:
                    weight = self.rrf_field_weights.get(field_name, 1.0)
                    contribution = weight * 1.0 / (k + rank)
                    contributions[field_name] = {
                        "rank": rank,
                        "weight": weight,
                        "contribution": round(contribution, 6),
                    }
                    break

        return contributions


class VectorRetriever(QdrantMultiVectorRetriever):
    """
    Backward-compatible wrapper for legacy VectorRetriever tests.
    Uses the new multi-vector retriever under the hood.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder,
        collection_name: str = "chunks_multi",
        similarity: str = "cosine",  # Kept for signature compatibility
        embedding_settings: Optional[EmbeddingSettings] = None,
    ):
        settings = embedding_settings or get_embedding_settings()
        plan = get_embedding_plan()
        super().__init__(
            qdrant_client,
            embedder,
            collection_name=collection_name,
            field_weights={"content": 1.0},
            embedding_settings=settings,
            embedding_plan=plan,
        )
        self.collection_name = collection_name

"""
Phase 7E-2: Hybrid Retrieval Implementation
Combines vector search (Qdrant) with BM25/keyword search (Neo4j full-text)
Implements RRF fusion, weighted fusion, bounded adjacency expansion, and context budget enforcement

Phase 7E-4: Enhanced with comprehensive metrics collection and SLO monitoring

Reference: Phase 7E Canonical Spec L1421-1444, L3781-3788
"""

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    NamedSparseVector,
)
from qdrant_client.http.models import SparseVector as QdrantSparseVector
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter as QdrantFilter
from qdrant_client.models import Fusion, FusionQuery, MatchAny, MatchValue, Prefetch

# Phase 7E-4: Monitoring imports
from src.monitoring.metrics import get_metrics_aggregator
from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.factory import ProviderFactory
from src.providers.rerank.base import RerankProvider
from src.providers.settings import EmbeddingSettings
from src.providers.tokenizer_service import TokenizerService
from src.shared.config import (
    get_config,
    get_embedding_settings,
    get_expected_namespace_suffix,
    get_settings,
)
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    retrieval_expansion_chunks_added,
    retrieval_expansion_rate_current,
    retrieval_expansion_total,
)
from src.shared.qdrant_schema import validate_qdrant_schema
from src.shared.schema import ensure_schema_version

logger = get_logger(__name__)
HYBRID_INIT_LOGGED = False


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
    doc_tag: Optional[str] = None  # document scoping tag (per-document)
    snapshot_scope: Optional[str] = None  # snapshot-level scope tag
    document_total_tokens: int = 0
    source_path: Optional[str] = None
    is_microdoc: bool = False
    doc_is_microdoc: bool = False
    is_microdoc_stub: bool = False

    # Expansion tracking
    is_expanded: bool = False  # Was this chunk added via expansion?
    expansion_source: Optional[str] = None  # Which chunk triggered expansion

    # Scoring metadata
    fusion_method: Optional[str] = None  # Method used for fusion
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None  # BM25/keyword score
    vector_rank: Optional[int] = None
    vector_score: Optional[float] = None  # Vector similarity score
    vector_score_kind: Optional[str] = None  # Similarity metric (cosine, dot, etc.)
    title_vec_score: Optional[float] = None
    entity_vec_score: Optional[float] = None
    lexical_vec_score: Optional[float] = None
    fused_score: Optional[float] = None  # Final fused score
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    rerank_original_rank: Optional[int] = None
    reranker: Optional[str] = None
    inherited_score: Optional[float] = None  # propagated semantic score from seed

    # Retrieval context metadata
    embedding_version: Optional[str] = None
    tenant: Optional[str] = None
    is_microdoc_extra: bool = False

    # Citation labels (order, title, level) derived from CitationUnits
    citation_labels: List[Tuple[int, str, int]] = field(default_factory=list)
    # Graph enrichment (Phase 2.3 legacy parity)
    graph_distance: int = 0
    graph_score: float = 0.0
    graph_path: Optional[List[str]] = None
    connection_count: int = 0
    mention_count: int = 0

    def __post_init__(self):
        """Ensure required fields are populated."""
        if self.original_section_ids is None:
            self.original_section_ids = []


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
        Ensure the full-text index targets both Chunk and CitationUnit nodes with text and heading fields.
        If an index with the same name exists but uses a different definition, drop and recreate it.
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
        *,
        use_query_api: bool = False,
        query_api_dense_limit: int = 200,
        query_api_sparse_limit: int = 200,
        query_api_candidate_limit: int = 200,
        primary_vector_name: str = "content",
        schema_supports_sparse: bool = False,
        schema_supports_colbert: bool = False,
    ):
        settings = get_settings()
        env = (
            getattr(settings, "env", None) or os.getenv("ENV", "development")
        ).lower()
        strict_env = env not in ("development", "dev", "test")
        self.client = qdrant_client
        self.embedder = embedder
        self.collection = collection_name
        self.rrf_k = rrf_k
        self.embedding_settings = embedding_settings
        self.embedding_version = (
            embedding_settings.version if embedding_settings else None
        )
        self.use_query_api = use_query_api
        self.query_api_dense_limit = query_api_dense_limit
        self.query_api_sparse_limit = query_api_sparse_limit
        self.query_api_candidate_limit = query_api_candidate_limit
        self.primary_vector_name = primary_vector_name or "content"
        # Prefer explicit flags; otherwise infer from embedding capabilities
        caps = getattr(embedding_settings, "capabilities", None)
        self.schema_supports_sparse = schema_supports_sparse or bool(
            getattr(caps, "supports_sparse", False)
        )
        self.schema_supports_colbert = schema_supports_colbert or bool(
            getattr(caps, "supports_colbert", False)
        )
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
        if self.schema_supports_sparse and not hasattr(self.embedder, "embed_sparse"):
            message = (
                "Configuration error: Sparse vectors enabled but embedder "
                "does not support embed_sparse."
            )
            if strict_env:
                raise ValueError(message)
            logger.warning("%s Disabling sparse support in non-strict env.", message)
            self.schema_supports_sparse = False
        if self.schema_supports_colbert and not hasattr(
            self.embedder, "embed_query_all"
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
        ]
        self.supports_sparse = (
            hasattr(self.embedder, "embed_sparse") and self.schema_supports_sparse
        )
        self.supports_colbert = (
            hasattr(self.embedder, "embed_colbert") and self.schema_supports_colbert
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

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        ef: Optional[int] = 256,
    ) -> List[ChunkResult]:
        self.last_stats = {"path": "legacy"}
        if self.use_query_api and self._query_api_supported():
            try:
                bundle = self._build_query_bundle(query)
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
        return self._search_legacy(query, top_k, filters, ef)

    def _search_legacy(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        ef: Optional[int],
    ) -> List[ChunkResult]:
        start_time = time.time()
        query_vectors = self._build_query_vectors(query)
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
                entity_vec_score=vec_score_by_id.get((pid, "entity"), 0.0),
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
            query_vector = {"name": vector_name, "vector": list(vector)}
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            logger.debug(
                "Named vector search failed", field=vector_name, error=str(exc)
            )
            return []

    def _build_query_vectors(
        self, query: str
    ) -> List[Tuple[str, str, Sequence[float]]]:
        base_vector = self.embedder.embed_query(query)
        vectors: List[Tuple[str, str, Sequence[float]]] = []
        for vector_name in self.field_weights.keys():
            if vector_name == self.sparse_field_name:
                sparse_vector = self._build_sparse_query(query)
                if sparse_vector:
                    vectors.append((vector_name, "sparse", sparse_vector))
                continue
            vectors.append((vector_name, "dense", base_vector))
        return vectors

    def _build_sparse_query(self, query: str) -> Optional[Dict[str, List[float]]]:
        if not self.supports_sparse or not hasattr(self.embedder, "embed_sparse"):
            return None
        try:
            sparse_vectors = self.embedder.embed_sparse([query])
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

        query_vector = {"name": vector_name, "vector": list(vector)}

        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
            with_vectors=False,
        )

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
            named_sparse = NamedSparseVector(
                name=self.sparse_query_name,
                vector=QdrantSparseVector(indices=indices, values=values),
            )
        except Exception as exc:
            logger.debug(
                "Failed to construct NamedSparseVector; skipping sparse search",
                error=str(exc),
            )
            return []
        return self.client.search(
            collection_name=self.collection,
            query_vector=named_sparse,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

    def _chunk_from_payload(
        self,
        pid: str,
        payload: Dict[str, Any],
        *,
        fused_score: Optional[float] = None,
        vector_score: Optional[float] = None,
        title_vec_score: Optional[float] = None,
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
            entity_vec_score=entity_vec_score,
            lexical_vec_score=lexical_vec_score,
            citation_labels=payload.get("citation_labels") or [],
        )

    def _build_query_bundle(self, query: str) -> QueryEmbeddingBundle:
        if hasattr(self.embedder, "embed_query_all"):
            bundle = self.embedder.embed_query_all(query)
        else:
            dense = self.embedder.embed_query(query)
            return QueryEmbeddingBundle(dense=list(dense))
        return QueryEmbeddingBundle(
            dense=list(bundle.dense),
            sparse=bundle.sparse,
            multivector=bundle.multivector,
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
            entries.append(
                Prefetch(
                    query=sparse_query,
                    using=self.sparse_query_name,
                    limit=min(candidate_limit, self.query_api_sparse_limit),
                    filter=qdrant_filter,
                )
            )
        return entries

    def _build_query_api_query(
        self, bundle: QueryEmbeddingBundle
    ) -> Tuple[Sequence[Sequence[float]] | Sequence[float], str]:
        if (
            self.schema_supports_colbert
            and bundle.multivector
            and bundle.multivector.vectors
        ):
            return [list(vec) for vec in bundle.multivector.vectors], "late-interaction"
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
        super().__init__(
            qdrant_client,
            embedder,
            collection_name=collection_name,
            field_weights={"content": 1.0},
            embedding_settings=settings,
        )
        self.collection_name = collection_name


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
        embedding_settings: Optional[EmbeddingSettings] = None,
    ):
        """
        Initialize hybrid retriever with all components.

        Args:
            neo4j_driver: Neo4j driver for BM25 and graph operations
            qdrant_client: Qdrant client for vector search
            embedder: Embedding provider for query vectorization
            tokenizer: Tokenizer service for token counting (optional)
        """
        config = get_config()
        settings = get_settings()
        self.embedding_settings = embedding_settings or get_embedding_settings()

        self.expected_schema_version = getattr(config.graph_schema, "version", None)
        if self.expected_schema_version:
            ensure_schema_version(neo4j_driver, self.expected_schema_version)

        if (
            self.embedding_settings
            and hasattr(embedder, "dims")
            and embedder.dims != self.embedding_settings.dims
        ):
            raise ValueError(
                f"HybridRetriever embedder dims ({getattr(embedder, 'dims', 'unknown')}) "
                f"do not match embedding profile dims ({self.embedding_settings.dims})."
            )

        hybrid_config = getattr(config.search, "hybrid", None)
        qdrant_vector_cfg = getattr(config.search.vector, "qdrant", None)
        bm25_config = getattr(config.search, "bm25", None)
        self.hybrid_mode = getattr(hybrid_config, "mode", "legacy")
        # Timeouts and index migration toggle (set early for downstream use)
        timeout_ms = getattr(hybrid_config, "bm25_timeout_ms", None)
        self.bm25_timeout_seconds = (
            float(timeout_ms) / 1000.0 if timeout_ms is not None else 2.0
        )
        expansion_timeout_ms = getattr(hybrid_config, "expansion_timeout_ms", None)
        self.expansion_timeout_seconds = (
            float(expansion_timeout_ms) / 1000.0
            if expansion_timeout_ms is not None
            else 2.0
        )
        self.allow_index_migration = (
            os.getenv("HYBRID_ALLOW_INDEX_MIGRATION", "false").lower() == "true"
        )

        self.neo4j_driver = neo4j_driver
        bm25_index_name = getattr(bm25_config, "index_name", None)
        bm25_enabled = getattr(bm25_config, "enabled", False)
        self.bm25_retriever = None
        if self.hybrid_mode != "bge_reranker" and bm25_enabled:
            self.bm25_retriever = BM25Retriever(
                neo4j_driver,
                index_name=bm25_index_name,
                timeout_seconds=self.bm25_timeout_seconds,
                allow_index_migration=self.allow_index_migration,
            )
        search_config = config.search
        qdrant_collection = getattr(
            getattr(search_config.vector, "qdrant", None), "collection_name", None
        )
        namespace_mode = getattr(settings, "embedding_namespace_mode", "none")
        namespace_suffix = get_expected_namespace_suffix(
            self.embedding_settings, namespace_mode
        )
        bm25_namespaced = False
        if self.bm25_retriever is not None:
            bm25_namespaced = bool(
                namespace_suffix
                and self.bm25_retriever.index_name
                and str(self.bm25_retriever.index_name).endswith(namespace_suffix)
            )
        qdrant_namespaced = bool(
            namespace_suffix
            and qdrant_collection
            and str(qdrant_collection).endswith(namespace_suffix)
        )
        strict_env = getattr(settings, "env", "development").lower() not in (
            "development",
            "dev",
            "test",
        )
        if (
            self.bm25_retriever is not None
            and qdrant_namespaced
            and not bm25_namespaced
        ):
            message = (
                "BM25 index appears global while Qdrant collection is namespaced. "
                "Set search.bm25.index_name to the namespaced value or allow override."
            )
            override = os.getenv("ALLOW_NAMESPACE_MISMATCH", "false").lower() == "true"
            if strict_env and not override:
                raise RuntimeError(
                    f"{message} bm25_index={self.bm25_retriever.index_name} "
                    f"qdrant_collection={qdrant_collection}"
                )
            logger.warning(
                message,
                extra={
                    "bm25_index_name": self.bm25_retriever.index_name,
                    "qdrant_collection_name": qdrant_collection,
                    "namespace_mode": namespace_mode,
                    "override": override,
                },
            )
        self.namespace_mode = namespace_mode
        self.qdrant_collection_name = qdrant_collection
        global HYBRID_INIT_LOGGED
        if not HYBRID_INIT_LOGGED:
            bm25_name = (
                getattr(self.bm25_retriever, "index_name", None)
                if self.bm25_retriever is not None
                else None
            )
            logger.info(
                "HybridRetriever initialized",
                extra={
                    "embedding_profile": getattr(
                        self.embedding_settings, "profile", None
                    ),
                    "embedding_provider": getattr(
                        self.embedding_settings, "provider", None
                    ),
                    "embedding_model": getattr(
                        self.embedding_settings, "model_id", None
                    ),
                    "embedding_version": getattr(
                        self.embedding_settings, "version", None
                    ),
                    "embedding_namespace_mode": namespace_mode,
                    "bm25_index_name": bm25_name,
                    "qdrant_collection_name": qdrant_collection,
                },
            )
            HYBRID_INIT_LOGGED = True
        self.filter_allowlist: Dict[str, str] = {
            "doc_tag": "doc_tag",
            "snapshot_scope": "snapshot_scope",
            "document_id": "document_id",
            "embedding_version": "embedding_version",
            "tenant": "tenant",
            "lang": "lang",
        }

        # Validate collection schema against active profile
        effective_settings = self.embedding_settings or get_embedding_settings()
        self.embedding_settings = effective_settings
        qdrant_collection = getattr(
            getattr(search_config.vector, "qdrant", None), "collection_name", None
        )
        include_entity_vector = (
            os.getenv("QDRANT_INCLUDE_ENTITY_VECTOR", "true").lower() == "true"
        )
        strict_validation = bool(
            settings.embedding_strict_mode
            and getattr(settings, "env", "development").lower()
            not in ("development", "dev", "test")
        )
        payload_fields = [
            "embedding_version",
            "embedding_provider",
            "embedding_dimensions",
            "tenant",
            "document_id",
        ]
        if qdrant_collection and qdrant_client:
            validate_qdrant_schema(
                qdrant_client,
                qdrant_collection,
                self.embedding_settings,
                require_sparse=getattr(qdrant_vector_cfg, "enable_sparse", False),
                require_colbert=getattr(qdrant_vector_cfg, "enable_colbert", False),
                include_entity=include_entity_vector,
                require_payload_fields=payload_fields,
                strict=strict_validation,
            )

        timeout_ms = getattr(hybrid_config, "bm25_timeout_ms", None)
        self.bm25_timeout_seconds = (
            float(timeout_ms) / 1000.0 if timeout_ms is not None else 2.0
        )
        expansion_timeout_ms = getattr(hybrid_config, "expansion_timeout_ms", None)
        self.expansion_timeout_seconds = (
            float(expansion_timeout_ms) / 1000.0
            if expansion_timeout_ms is not None
            else 2.0
        )
        self.allow_index_migration = (
            os.getenv("HYBRID_ALLOW_INDEX_MIGRATION", "false").lower() == "true"
        )

        self.vector_field_weights = dict(
            getattr(hybrid_config, "vector_fields", {"content": 1.0})
        )
        self.max_sources_to_expand = getattr(
            getattr(hybrid_config, "expansion", {}), "max_sources", 5
        )
        self.vector_retriever = QdrantMultiVectorRetriever(
            qdrant_client,
            embedder,
            collection_name=qdrant_collection
            or config.search.vector.qdrant.collection_name,
            field_weights=self.vector_field_weights,
            rrf_k=getattr(hybrid_config, "rrf_k", 60),
            embedding_settings=self.embedding_settings,
            use_query_api=getattr(qdrant_vector_cfg, "use_query_api", False),
            query_api_dense_limit=getattr(
                qdrant_vector_cfg, "query_api_dense_limit", 200
            ),
            query_api_sparse_limit=getattr(
                qdrant_vector_cfg, "query_api_sparse_limit", 200
            ),
            query_api_candidate_limit=getattr(
                qdrant_vector_cfg, "query_api_candidate_limit", 200
            ),
            primary_vector_name=getattr(
                qdrant_vector_cfg, "query_vector_name", "content"
            ),
            schema_supports_sparse=getattr(qdrant_vector_cfg, "enable_sparse", False),
            schema_supports_colbert=getattr(qdrant_vector_cfg, "enable_colbert", False),
        )
        dense_active = True
        sparse_active = bool(
            self.vector_retriever.supports_sparse
            and self.vector_retriever.schema_supports_sparse
            and (
                not self.vector_retriever.sparse_field_name
                or self.vector_retriever.field_weights.get(
                    self.vector_retriever.sparse_field_name, 0
                )
                > 0
            )
        )
        colbert_active = bool(
            self.vector_retriever.supports_colbert
            and self.vector_retriever.schema_supports_colbert
            and self.vector_retriever.use_query_api
        )
        logger.info(
            "Retrieval modes resolved",
            extra={
                "dense_active": dense_active,
                "sparse_active": sparse_active,
                "colbert_active": colbert_active,
                "colbert_query_api": self.vector_retriever.use_query_api,
                "has_sparse_field": bool(self.vector_retriever.sparse_field_name),
            },
        )
        self.tokenizer = tokenizer or TokenizerService()
        self.reranker_config = getattr(hybrid_config, "reranker", None)
        self._reranker_enabled = bool(getattr(self.reranker_config, "enabled", False))
        self.rerank_top_n = int(getattr(self.reranker_config, "top_n", 0) or 0)
        self._reranker: Optional[RerankProvider] = None
        if self._reranker_enabled:
            logger.info("HybridRetriever reranker enabled via configuration")
        self.context_group_cap = getattr(
            search_config.response, "max_sections_per_parent", 3
        )

        # Load configuration
        # (already assigned above)
        # Fusion configuration
        self.fusion_method = FusionMethod(getattr(hybrid_config, "method", "rrf"))
        self.rrf_k = getattr(hybrid_config, "rrf_k", 60)
        self.fusion_alpha = getattr(hybrid_config, "fusion_alpha", 0.6)
        self.graph_propagation_decay = getattr(
            hybrid_config, "graph_propagation_decay", 0.85
        )

        # Expansion configuration
        expansion_config = getattr(hybrid_config, "expansion", {})
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

        # Graph enrichment configuration (Phase 2.3 parity)
        graph_config = getattr(search_config, "graph", None)
        self.graph_max_depth = (
            getattr(graph_config, "max_depth", 3) if graph_config else 3
        )
        self.graph_max_related = (
            getattr(graph_config, "max_related_per_seed", 20) if graph_config else 20
        )
        self.graph_weight = getattr(hybrid_config, "graph_weight", 0.3)
        rels_env = os.getenv("GRAPH_REL_TYPES")
        if rels_env:
            self.graph_relationships = [
                rel.strip() for rel in rels_env.split(",") if rel.strip()
            ]
        else:
            self.graph_relationships = [
                "MENTIONS",
                "CONTAINS_STEP",
                "HAS_PARAMETER",
                "REQUIRES",
                "AFFECTS",
            ]
        self.graph_enabled = self.graph_max_related > 0 and self.graph_max_depth > 0

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
        if self.embedding_settings:
            logger.info(
                "HybridRetriever embedding profile",
                profile=self.embedding_settings.profile,
                provider=self.embedding_settings.provider,
                model=self.embedding_settings.model_id,
                dims=self.embedding_settings.dims,
                tokenizer_backend=self.embedding_settings.tokenizer_backend,
                supports_sparse=getattr(
                    self.embedding_settings.capabilities, "supports_sparse", None
                ),
                supports_colbert=getattr(
                    self.embedding_settings.capabilities, "supports_colbert", None
                ),
            )

    def _normalize_filter_value(self, value: Any) -> Optional[Any]:
        """Normalize filter values to scalar or non-empty list."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            items = [v for v in value if v is not None]
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            return list(items)
        return value

    def _normalize_filters(
        self, raw_filters: Optional[Dict[str, Any]], caller: str = "hybrid"
    ) -> Dict[str, Any]:
        """Apply allowlist, value normalization, and embedding/tenant gating."""
        filters = dict(raw_filters or {})
        normalized: Dict[str, Any] = {}

        # Enforce embedding_version from active settings
        active_version = getattr(self.embedding_settings, "version", None)
        incoming_version = filters.get("embedding_version")
        if active_version:
            if incoming_version and incoming_version != active_version:
                logger.error(
                    "Embedding version mismatch; overriding with active profile",
                    extra={
                        "caller": caller,
                        "incoming": incoming_version,
                        "active": active_version,
                    },
                )
            filters["embedding_version"] = active_version

        for key, value in filters.items():
            if key not in self.filter_allowlist:
                logger.warning(
                    "Ignoring unsupported filter key",
                    extra={"caller": caller, "key": key},
                )
                continue
            normalized_value = self._normalize_filter_value(value)
            if normalized_value is None:
                continue
            normalized[key] = normalized_value

        return normalized

    def _build_bm25_predicates(
        self, filters: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Build WHERE predicates and params for BM25 search."""
        where_clauses: List[str] = []
        params: Dict[str, Any] = {}

        for key, value in filters.items():
            param_name = f"filter_{key}"
            if isinstance(value, list):
                where_clauses.append(f"chunk.{key} IN ${param_name}")
                params[param_name] = value
            else:
                where_clauses.append(f"chunk.{key} = ${param_name}")
                params[param_name] = value
        return where_clauses, params

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
        metrics: Dict[str, Any] = {
            "namespace_mode": getattr(self, "namespace_mode", None),
            "bm25_index_name": getattr(self.bm25_retriever, "index_name", None),
            "qdrant_collection_name": getattr(
                getattr(self, "vector_retriever", None),
                "collection_name",
                getattr(self, "qdrant_collection_name", None),
            ),
        }
        if self.embedding_settings:
            metrics["embedding_profile"] = self.embedding_settings.profile
            metrics["embedding_provider"] = self.embedding_settings.provider
            metrics["embedding_model"] = self.embedding_settings.model_id
        normalized_filters = self._normalize_filters(
            filters or {}, caller="hybrid.retrieve"
        )
        raw_doc_tag = normalized_filters.get("doc_tag")
        doc_tag = (
            raw_doc_tag[0]
            if isinstance(raw_doc_tag, list) and raw_doc_tag
            else raw_doc_tag
        )

        # Step 1: Parallel BM25 and vector search
        # Retrieve more candidates for fusion (3x top_k)
        candidate_k = min(top_k * 3, 100)

        # Branch: vector-only (bge_reranker) vs legacy (BM25 + fusion)
        bm25_results: List[ChunkResult] = []
        vec_results: List[ChunkResult] = []

        if self.hybrid_mode != "bge_reranker" and self.bm25_retriever:
            # BM25 search
            bm25_start = time.time()
            bm25_results = self.bm25_retriever.search(
                query, candidate_k, normalized_filters
            )
            metrics["bm25_time_ms"] = (time.time() - bm25_start) * 1000
            metrics["bm25_count"] = len(bm25_results)
        else:
            metrics["bm25_time_ms"] = 0.0
            metrics["bm25_count"] = 0

        # Vector search (always)
        vec_start = time.time()
        vec_results = self.vector_retriever.search(
            query, candidate_k, normalized_filters
        )
        vector_stats = getattr(self.vector_retriever, "last_stats", {}) or {}
        metrics["vector_path"] = vector_stats.get("path", "legacy")
        metrics["vec_time_ms"] = vector_stats.get(
            "duration_ms", (time.time() - vec_start) * 1000
        )
        metrics["vec_count"] = len(vec_results)
        metrics["vector_fields"] = list(self.vector_field_weights.keys())
        if "prefetch_count" in vector_stats:
            metrics["vector_prefetch_count"] = vector_stats["prefetch_count"]
        if "colbert_used" in vector_stats:
            metrics["vector_colbert_used"] = vector_stats["colbert_used"]

        # Step 2: Fuse rankings
        fusion_start = time.time()
        if self.hybrid_mode == "bge_reranker":
            # Vector-only path: use vector scores directly
            fused_results = vec_results
            for r in fused_results:
                if r.fused_score is None:
                    r.fused_score = r.vector_score
                r.fusion_method = "weighted"
            metrics["fusion_method"] = "vector-only"
        else:
            if self.fusion_method == FusionMethod.RRF:
                fused_results = self._rrf_fusion(bm25_results, vec_results)
            else:
                fused_results = self._weighted_fusion(bm25_results, vec_results)
            metrics["fusion_method"] = self.fusion_method.value
        metrics["fusion_time_ms"] = (time.time() - fusion_start) * 1000

        fused_results = [r for r in fused_results if not r.is_microdoc_stub]

        # Step 3: Take fused results as seeds and optionally rerank
        fused_results.sort(key=lambda x: x.fused_score or 0, reverse=True)
        self._log_stage_snapshot("post-fusion", fused_results)

        reranker_active = False
        seeds: List[ChunkResult]

        if self._reranker_enabled and fused_results:
            pool_cap = self.rerank_top_n or top_k
            rerank_pool_size = min(pool_cap, len(fused_results))
            rerank_candidates = fused_results[:rerank_pool_size]
            ordered_candidates = self._apply_reranker(query, rerank_candidates, metrics)
            reranker_active = bool(metrics.get("reranker_applied"))
            if reranker_active:
                ordered_candidates.sort(
                    key=lambda chunk: (
                        chunk.rerank_score or float("-inf"),
                        chunk.fused_score or 0.0,
                    ),
                    reverse=True,
                )
                seeds = ordered_candidates[:top_k]
            else:
                seeds = fused_results[:top_k]
        else:
            seeds = fused_results[:top_k]

        # Hydrate vector-only winners with citation labels
        self._hydrate_missing_citations(seeds)

        # Prefer chunks that carry richer citation labels only when reranker is inactive
        if not reranker_active:
            seeds.sort(
                key=lambda chunk: (
                    len(chunk.citation_labels or []),
                    chunk.fused_score or 0.0,
                ),
                reverse=True,
            )

        if doc_tag:
            before = len(seeds)
            seeds = [c for c in seeds if getattr(c, "doc_tag", None) == doc_tag]
            logger.info(
                "Filtered seeds by doc_tag=%s: kept %d/%d", doc_tag, len(seeds), before
            )

        metrics["seed_count"] = len(seeds)
        seed_ids: Optional[Set[str]] = None
        seed_rank_map: Optional[Dict[str, int]] = None
        if reranker_active:
            seed_ids = {chunk.chunk_id for chunk in seeds}
            seed_rank_map = {chunk.chunk_id: idx for idx, chunk in enumerate(seeds)}

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
            if doc_tag:
                expanded_results = [
                    n
                    for n in expanded_results
                    if getattr(n, "doc_tag", None) == doc_tag
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
            metrics["expansion_cap_hit"] = int(
                len(seeds) > getattr(self, "max_sources_to_expand", 5)
            )
        else:
            metrics["expansion_count"] = 0
            metrics["expanded_source_count"] = 0
            metrics["expansion_cap_hit"] = 0

        # Include micro-doc extras prior to dedup
        if microdoc_extras:
            all_results.extend(microdoc_extras)

        # Step 6: Dedup, hydrate citations, and maintain deterministic ordering
        all_results = self._dedup_results(all_results)
        all_results = [
            c for c in all_results if not getattr(c, "is_microdoc_stub", False)
        ]
        self._hydrate_missing_citations(all_results)

        graph_chunks, graph_stats = self._apply_graph_enrichment(
            seeds, all_results, doc_tag
        )
        if graph_chunks:
            all_results.extend(graph_chunks)
            all_results = self._dedup_results(all_results)
            all_results = [
                c for c in all_results if not getattr(c, "is_microdoc_stub", False)
            ]
        metrics.update(graph_stats)

        self._annotate_coverage(all_results)

        if doc_tag:
            all_results = [
                c for c in all_results if getattr(c, "doc_tag", None) == doc_tag
            ]

        primaries = [c for c in all_results if not c.is_microdoc_extra]
        extras = [c for c in all_results if c.is_microdoc_extra]

        def _primary_score_key(chunk: ChunkResult) -> Tuple[int, float, float, float]:
            rerank_val = (
                float(chunk.rerank_score)
                if chunk.rerank_score is not None
                else float(chunk.fused_score or 0.0)
            )
            citation_weight = float(len(chunk.citation_labels or []))
            if reranker_active and seed_ids and chunk.chunk_id in seed_ids:
                rank_idx = seed_rank_map.get(chunk.chunk_id, 0) if seed_rank_map else 0
                return (1.0, -float(rank_idx), rerank_val, citation_weight)
            return (0.0, rerank_val, citation_weight, float(chunk.fused_score or 0.0))

        primaries.sort(key=_primary_score_key, reverse=True)
        if reranker_active and seed_ids:
            extras.sort(
                key=lambda chunk: (
                    0.0,
                    (
                        float(chunk.rerank_score)
                        if chunk.rerank_score is not None
                        else float(chunk.fused_score or 0.0)
                    ),
                ),
                reverse=True,
            )
        else:
            extras.sort(key=lambda chunk: float(chunk.fused_score or 0.0), reverse=True)

        primaries = self._apply_doc_continuity_boost(primaries, alpha=0.12)
        self._log_stage_snapshot("post-continuity", primaries)

        # Step 6: Context assembly with budget enforcement
        context_start = time.time()
        primary_budget, primary_tokens = self._enforce_context_budget(primaries)
        extra_budget, total_tokens = self._enforce_context_budget(
            extras, starting_tokens=primary_tokens
        )
        final_results = primary_budget + extra_budget
        self._log_stage_snapshot("post-budget", final_results)

        metrics["context_assembly_ms"] = (time.time() - context_start) * 1000
        metrics["primary_count"] = len(primary_budget)
        metrics["microdoc_used"] = len(extra_budget)
        metrics["microdoc_present"] = int(bool(extras))
        metrics["final_count"] = len(final_results)
        metrics["total_tokens"] = total_tokens

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
        # Build rank dictionaries and preserve best modality scores
        bm25_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        for i, r in enumerate(bm25_results):
            rank = i + 1
            best_rank = bm25_ranks.get(r.chunk_id)
            if best_rank is None or rank < best_rank:
                bm25_ranks[r.chunk_id] = rank
            if r.bm25_rank is None or r.bm25_rank > rank:
                r.bm25_rank = rank
            if r.bm25_score is not None:
                best_score = bm25_scores.get(r.chunk_id)
                if best_score is None or r.bm25_score > best_score:
                    bm25_scores[r.chunk_id] = r.bm25_score

        vec_ranks: Dict[str, int] = {}
        vec_scores: Dict[str, float] = {}
        vec_kinds: Dict[str, str] = {}
        for i, r in enumerate(vec_results):
            rank = i + 1
            best_rank = vec_ranks.get(r.chunk_id)
            if best_rank is None or rank < best_rank:
                vec_ranks[r.chunk_id] = rank
            if r.vector_rank is None or r.vector_rank > rank:
                r.vector_rank = rank
            if r.vector_score is not None:
                best_score = vec_scores.get(r.chunk_id)
                if best_score is None or r.vector_score > best_score:
                    vec_scores[r.chunk_id] = r.vector_score
            if r.vector_score_kind:
                current_kind = vec_kinds.get(r.chunk_id)
                if current_kind is None or (
                    r.vector_score is not None
                    and vec_scores.get(r.chunk_id, float("-inf")) == r.vector_score
                ):
                    vec_kinds[r.chunk_id] = r.vector_score_kind

        # Build combined result map (prefer BM25 ordering, then enrich with vector data)
        all_chunks: Dict[str, ChunkResult] = {}
        for r in bm25_results:
            all_chunks.setdefault(r.chunk_id, r)
        for r in vec_results:
            existing = all_chunks.get(r.chunk_id)
            if existing is None:
                all_chunks[r.chunk_id] = r
            else:
                if existing.vector_score is None or (
                    r.vector_score is not None
                    and existing.vector_score < r.vector_score
                ):
                    existing.vector_score = r.vector_score
                if existing.vector_rank is None or (
                    r.vector_rank is not None and existing.vector_rank > r.vector_rank
                ):
                    existing.vector_rank = r.vector_rank

        # Calculate RRF scores
        for chunk_id, chunk in all_chunks.items():
            bm25_rank = bm25_ranks.get(chunk_id)
            vec_rank = vec_ranks.get(chunk_id)

            # RRF formula
            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)
            if vec_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + vec_rank)

            chunk.fused_score = rrf_score
            chunk.fusion_method = "rrf"

            if bm25_rank is not None:
                chunk.bm25_rank = bm25_rank
            if vec_rank is not None:
                chunk.vector_rank = vec_rank

            best_bm25_score = bm25_scores.get(chunk_id)
            if best_bm25_score is not None and (
                chunk.bm25_score is None or chunk.bm25_score < best_bm25_score
            ):
                chunk.bm25_score = best_bm25_score

            best_vec_score = vec_scores.get(chunk_id)
            if best_vec_score is not None and (
                chunk.vector_score is None or chunk.vector_score < best_vec_score
            ):
                chunk.vector_score = best_vec_score
            best_vec_kind = vec_kinds.get(chunk_id)
            if best_vec_kind is not None:
                chunk.vector_score_kind = best_vec_kind

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
                for r in results:
                    if getattr(r, score_attr, 0):
                        setattr(r, f"norm_{score_attr}", 1.0)
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
        """Favor documents that consistently appear at the top of the list."""
        if not chunks:
            return chunks

        from collections import Counter

        counts = Counter(getattr(c, "document_id", None) for c in chunks)
        total = max(1, len(chunks))

        boosted: List[ChunkResult] = []
        for c in chunks:
            metric = float(c.fused_score or c.vector_score or c.bm25_score or 0.0)
            if metric == 0.0:
                boosted.append(c)
                continue
            doc_id = getattr(c, "document_id", None)
            share = counts.get(doc_id, 0) / total if doc_id else 0.0
            multiplier = 1.0 + alpha * share
            adjusted = metric * multiplier
            # Clamp to keep fusion contract in [0, 1]
            c.fused_score = min(adjusted, 1.0)
            boosted.append(c)

        boosted.sort(
            key=lambda x: (
                float(x.rerank_score)
                if x.rerank_score is not None
                else float(x.fused_score or 0.0)
            ),
            reverse=True,
        )
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

    def _apply_reranker(
        self, query: str, seeds: List[ChunkResult], metrics: Dict[str, Any]
    ) -> List[ChunkResult]:
        cfg = self.reranker_config
        if not cfg or not getattr(cfg, "enabled", False):
            metrics["reranker_applied"] = False
            metrics["reranker_reason"] = "disabled"
            return seeds

        reranker = self._get_reranker()
        if reranker is None:
            metrics["reranker_applied"] = False
            metrics["reranker_reason"] = "not_available"
            return seeds

        def _clean_text(text: str) -> str:
            # Remove simple HTML tags and known markup artifacts, collapse whitespace.
            text = re.sub(r"<[^>]+>", " ", text)
            text = text.replace("[CODE]", " ").replace("[/CODE]", " ")
            text = text.replace("</details>", " ").replace("<details>", " ")
            text = re.sub(
                r"\b[A-Z]{1,4}\]\s*", " ", text
            )  # drop stray tokens like 'DE]'
            text = text.replace("####", " ")
            text = text.replace("DE]", " ")
            text = re.sub(r"^[^A-Za-z0-9]+", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        candidates: List[Dict[str, Any]] = []
        for chunk in seeds:
            text_body = (chunk.text or "").strip()
            heading = (chunk.heading or "").strip()
            if heading and text_body:
                text = f"{heading}\n\n{text_body}"
            else:
                text = text_body or heading

            text = _clean_text(text)
            if not text:
                has_tokens = (chunk.token_count or 0) > 0
                if not heading or not has_tokens:
                    continue

            candidates.append(
                {
                    "id": chunk.chunk_id,
                    "text": text,
                    "original_result": chunk,
                }
            )

        if not candidates:
            metrics["reranker_applied"] = False
            metrics["reranker_reason"] = "no_text"
            return seeds

        top_n = getattr(cfg, "top_n", None)
        if not top_n or top_n <= 0:
            top_n = len(candidates)
        top_k = min(top_n, len(candidates))

        # Batch candidates to respect reranker service limits
        service_max_batch = 32
        service_max_batch_tokens = 1024  # user-requested cap

        def _approx_tokens(text: str) -> int:
            # Simple word-count approximation
            return max(1, len(text.split()))

        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_tokens = 0

        for idx, cand in enumerate(candidates):
            tcount = _approx_tokens(cand.get("text", ""))
            if current_batch and (
                len(current_batch) + 1 > service_max_batch
                or current_tokens + tcount > service_max_batch_tokens
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            cand["orig_index"] = idx
            current_batch.append(cand)
            current_tokens += tcount

        if current_batch:
            batches.append(current_batch)

        start_time = time.time()
        reranked_payload: List[Dict[str, Any]] = []
        try:
            for batch in batches:
                batch_top_k = min(top_k, len(batch))
                reranked_batch = reranker.rerank(
                    query=query, candidates=batch, top_k=batch_top_k
                )
                reranked_payload.extend(reranked_batch)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as exc:  # pragma: no cover - provider failure logging
            logger.warning("Reranker call failed; returning fused ordering: %s", exc)
            metrics["reranker_applied"] = False
            metrics["reranker_reason"] = "provider_error"
            return seeds

        reranked_payload = sorted(
            reranked_payload,
            key=lambda c: (c.get("rerank_score") or float("-inf")),
            reverse=True,
        )
        reranked_payload = reranked_payload[:top_k]

        reranked_chunks: List[ChunkResult] = []
        for payload in reranked_payload:
            chunk = payload.get("original_result")
            if not isinstance(chunk, ChunkResult):
                continue

            score = payload.get("rerank_score")
            if score is not None:
                chunk.rerank_score = score

            rank = payload.get("original_rank")
            chunk.rerank_original_rank = None
            if rank is not None:
                try:
                    chunk.rerank_original_rank = int(rank)
                except (TypeError, ValueError):
                    chunk.rerank_original_rank = None

            chunk.reranker = payload.get("reranker") or reranker.model_id
            chunk.fusion_method = "rerank"
            reranked_chunks.append(chunk)

        try:
            sample_logs = reranked_payload[: min(3, len(reranked_payload))]
            logger.debug(
                "Reranker payload samples",
                extra={
                    "sample_count": len(sample_logs),
                    "query_snippet": query[:200],
                    "doc_snippets": [(p.get("text") or "")[:200] for p in sample_logs],
                    "scores": [p.get("rerank_score") for p in sample_logs],
                },
            )
        except Exception:
            pass

        if not reranked_chunks:
            metrics["reranker_applied"] = False
            metrics["reranker_reason"] = "no_results"
            return seeds

        reranked_chunks.sort(
            key=lambda c: (c.rerank_score or float("-inf"), c.fused_score or 0.0),
            reverse=True,
        )
        for idx, chunk in enumerate(reranked_chunks, start=1):
            chunk.rerank_rank = idx

        metrics["reranker_applied"] = True
        metrics["reranker_reason"] = "ok"
        metrics["reranker_model"] = reranker.model_id
        metrics["reranker_time_ms"] = latency_ms
        return reranked_chunks

    def _get_reranker(self) -> Optional[RerankProvider]:
        if not self.reranker_config or not getattr(
            self.reranker_config, "enabled", False
        ):
            return None

        if self._reranker is not None:
            return self._reranker

        cfg_provider = getattr(self.reranker_config, "provider", None)
        cfg_model = getattr(self.reranker_config, "model", None)

        try:
            self._reranker = ProviderFactory.create_rerank_provider(
                provider=cfg_provider,
                model=cfg_model,
            )
            logger.info(
                "HybridRetriever reranker initialized: provider=%s, model=%s",
                self._reranker.provider_name,
                self._reranker.model_id,
            )
        except Exception as exc:  # pragma: no cover - provider init logging
            logger.warning("Unable to initialize reranker provider: %s", exc)
            self._reranker = None

        return self._reranker

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
        seed_lookup = {seed.chunk_id: seed for seed in seeds}

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
            neighbor.doc_is_microdoc AS doc_is_microdoc,
            neighbor.is_microdoc_stub AS is_microdoc_stub,
            neighbor.source_path AS source_path,
            neighbor.embedding_version AS embedding_version,
            neighbor.tenant AS tenant,
            chunk_id AS source_chunk
        """

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    expansion_query,
                    chunk_ids=chunk_ids_to_expand,
                    timeout=self.expansion_timeout_seconds,
                )

                for record in result:
                    neighbor_id = record["chunk_id"]
                    rid = ("chunk_id", neighbor_id)

                    if rid in seen_ids:
                        continue  # Skip duplicates

                    seen_ids.add(rid)

                    # Find source chunk to get its score
                    source_chunk_id = record["source_chunk"]
                    source_seed = seed_lookup.get(source_chunk_id)
                    source_fused = (
                        float(source_seed.fused_score or 0.0) if source_seed else 0.0
                    )
                    source_rerank = (
                        float(source_seed.rerank_score)
                        if source_seed and source_seed.rerank_score is not None
                        else None
                    )

                    neighbor_fused_score = self._neighbor_score(source_fused)
                    neighbor_rerank_score = (
                        self._neighbor_score(source_rerank)
                        if source_rerank is not None
                        else None
                    )

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
                            doc_is_microdoc=record.get("doc_is_microdoc", False),
                            is_microdoc_stub=record.get("is_microdoc_stub", False),
                            embedding_version=record.get("embedding_version"),
                            tenant=record.get("tenant"),
                            is_expanded=True,
                            expansion_source=source_chunk_id,
                            fused_score=neighbor_fused_score,
                            citation_labels=[],
                            graph_distance=1,
                            graph_score=1.0,
                            graph_path=[source_chunk_id, neighbor_id],
                        )
                    )
                    if neighbor_rerank_score is not None:
                        expanded[-1].rerank_score = neighbor_rerank_score

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

    def _apply_graph_enrichment(
        self,
        seeds: List[ChunkResult],
        current_results: List[ChunkResult],
        doc_tag: Optional[str],
    ) -> Tuple[List[ChunkResult], Dict[str, int]]:
        """Expand results with graph neighbors and annotate graph scores."""
        stats = {
            "graph_neighbors_considered": 0,
            "graph_neighbors_added": 0,
        }
        if not self.graph_enabled or not seeds:
            return [], stats

        neighbors = self._fetch_graph_neighbors(seeds, doc_tag=doc_tag)
        stats["graph_neighbors_considered"] = len(neighbors)
        if not neighbors:
            return [], stats

        existing = {chunk.chunk_id: chunk for chunk in current_results}
        added: List[ChunkResult] = []

        for neighbor in neighbors:
            current = existing.get(neighbor.chunk_id)
            if current:
                current.graph_score = max(current.graph_score, neighbor.graph_score)
                if not current.graph_distance or (
                    neighbor.graph_distance
                    and neighbor.graph_distance < current.graph_distance
                ):
                    current.graph_distance = neighbor.graph_distance
                    current.graph_path = neighbor.graph_path
                if not current.expansion_source:
                    current.expansion_source = neighbor.expansion_source
                continue
            added.append(neighbor)
            existing[neighbor.chunk_id] = neighbor

        stats["graph_neighbors_added"] = len(added)
        return added, stats

    def _fetch_graph_neighbors(
        self, seeds: List[ChunkResult], doc_tag: Optional[str]
    ) -> List[ChunkResult]:
        """Fetch graph neighbors for the given seed chunks."""
        if not self.graph_enabled:
            return []

        seed_lookup = {seed.chunk_id: seed for seed in seeds if seed.chunk_id}
        if not seed_lookup:
            return []

        limit_per_seed = max(1, min(self.graph_max_related, 200))
        rel_pattern = "|".join(self.graph_relationships)
        query = f"""
        UNWIND $seed_ids AS seed_id
        MATCH (seed:Chunk {{id: seed_id}})
        CALL {{
            WITH seed
            MATCH path=(seed)-[r:{rel_pattern}*1..{self.graph_max_depth}]-(target:Chunk)
            WHERE seed.id <> target.id
              AND ($doc_tag IS NULL OR target.doc_tag = $doc_tag)
            WITH target, path
            ORDER BY length(path) ASC
            LIMIT $per_seed
            RETURN target, path, length(path) AS dist
        }}
        RETURN seed.id AS seed_id,
               target {{
                   .id,
                   .document_id,
                   .parent_section_id,
                   .order,
                   .level,
                   .heading,
                   .text,
                   token_count: coalesce(target.token_count, target.tokens, 0),
                   .is_combined,
                   .is_split,
                   .original_section_ids,
                   .boundaries_json,
                   .doc_tag,
                   .document_total_tokens,
                   .source_path,
                   .is_microdoc,
                   .doc_is_microdoc,
                   .is_microdoc_stub,
                   .embedding_version,
                   .tenant
               }} AS props,
               dist,
               [node IN nodes(path) | node.id] AS path_nodes
        """

        best_by_id: Dict[str, ChunkResult] = {}
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    query,
                    seed_ids=list(seed_lookup.keys()),
                    per_seed=limit_per_seed,
                    doc_tag=doc_tag,
                    timeout=self.expansion_timeout_seconds,
                )
                for record in result:
                    seed_id = record["seed_id"]
                    source_chunk = seed_lookup.get(seed_id)
                    if not source_chunk:
                        continue

                    props = record["props"] or {}
                    if doc_tag and props.get("doc_tag") != doc_tag:
                        continue
                    if (
                        source_chunk.document_id
                        and props.get("document_id")
                        and props["document_id"] != source_chunk.document_id
                    ):
                        continue

                    chunk = self._chunk_from_props(props)
                    chunk.is_expanded = True
                    chunk.expansion_source = seed_id

                    distance = int(record.get("dist") or 1)
                    chunk.graph_distance = max(1, distance)
                    chunk.graph_score = 1.0 / (chunk.graph_distance + 1)
                    path_nodes = record.get("path_nodes") or []
                    chunk.graph_path = [seed_id] + [
                        str(node_id) for node_id in path_nodes if node_id != seed_id
                    ]

                    source_score = (
                        source_chunk.fused_score
                        or source_chunk.vector_score
                        or source_chunk.bm25_score
                        or 0.0
                    )
                    if source_chunk.rerank_score is not None:
                        try:
                            rerank_sem = 1.0 / (
                                1.0 + math.exp(-float(source_chunk.rerank_score))
                            )
                        except OverflowError:
                            rerank_sem = 0.0 if source_chunk.rerank_score < 0 else 1.0
                        source_score = max(source_score, rerank_sem)

                    propagated = source_score * self.graph_propagation_decay
                    chunk.inherited_score = propagated
                    chunk.fused_score = max(
                        chunk.fused_score or 0.0,
                        propagated,
                    )
                    chunk.vector_score = chunk.fused_score
                    chunk.vector_score_kind = (
                        chunk.vector_score_kind or "graph_propagated"
                    )

                    existing = best_by_id.get(chunk.chunk_id)
                    if existing and existing.graph_score >= chunk.graph_score:
                        continue
                    best_by_id[chunk.chunk_id] = chunk
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Graph enrichment failed (expansion)",
                extra={"error": str(exc)},
            )
            return []

        return list(best_by_id.values())

    def _annotate_coverage(self, chunks: List[ChunkResult]) -> None:
        """Attach connection/mention counts used by ranker coverage features."""
        ids = {chunk.chunk_id for chunk in chunks if chunk.chunk_id}
        if not ids:
            return

        coverage_query = """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id: cid})
        OPTIONAL MATCH (c)-[r]->()
        WITH c, count(DISTINCT r) AS conn_count
        OPTIONAL MATCH (c)-[:MENTIONS]->(e)
        RETURN c.id AS id,
               conn_count AS connection_count,
               count(DISTINCT e) AS mention_count
        """

        coverage = {}
        try:
            with self.neo4j_driver.session() as session:
                records = session.run(coverage_query, ids=list(ids))
                for record in records:
                    coverage[record["id"]] = {
                        "connection_count": record.get("connection_count", 0),
                        "mention_count": record.get("mention_count", 0),
                    }
        except Exception as exc:
            logger.warning("Coverage enrichment failed: %s", exc)
            return

        for chunk in chunks:
            data = coverage.get(chunk.chunk_id)
            if not data:
                continue
            chunk.connection_count = int(data.get("connection_count") or 0)
            chunk.mention_count = int(data.get("mention_count") or 0)

    def _neighbor_score(self, source_score: float) -> float:
        if source_score <= 0:
            return 0.0
        epsilon = 1e-4
        half = source_score * 0.5
        return max(0.0, min(half, source_score - epsilon))

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
                if base.rerank_score is not None:
                    patched.rerank_score = self._neighbor_score(
                        float(base.rerank_score)
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
            chunk.doc_is_microdoc or chunk.token_count < self.micro_min_tokens
        ) and (chunk.document_total_tokens or 0) <= self.micro_doc_max

    def _is_microdoc_source(self, chunk: ChunkResult) -> bool:
        total_tokens = chunk.document_total_tokens or 0
        return chunk.doc_is_microdoc or total_tokens <= self.micro_doc_max

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
        query_vector: Optional[List[float]] = None
        try:
            query_vector = self.vector_retriever.embedder.embed_query(text)
        except Exception as exc:
            logger.debug(
                "Microdoc query embedding failed; falling back to passage embedding",
                extra={"error": str(exc), "doc": base.document_id},
            )
            try:
                vectors = self.vector_retriever.embedder.embed_documents([text])
                if vectors:
                    query_vector = vectors[0]
            except Exception as inner_exc:
                logger.debug(
                    "Microdoc embedding failed",
                    extra={"error": str(inner_exc), "doc": base.document_id},
                )
                return []

        if not query_vector:
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
        snapshot_scope = filters.get("snapshot_scope")
        if snapshot_scope:
            must_conditions.append(
                FieldCondition(
                    key="snapshot_scope", match=MatchValue(value=snapshot_scope)
                )
            )

        qdrant_filter = None
        if must_conditions or must_not_conditions:
            qdrant_filter = Filter(
                must=must_conditions or None, must_not=must_not_conditions or None
            )

        extras: List[ChunkResult] = []
        try:
            hits = self.vector_retriever.search_named_vector(
                "content",
                query_vector,
                self.micro_knn_limit,
                query_filter=qdrant_filter,
                score_threshold=self.micro_sim_threshold,
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
                snapshot_scope=payload.get("snapshot_scope"),
                document_total_tokens=doc_total,
                source_path=payload.get("source_path"),
                is_microdoc=payload.get("is_microdoc", False),
                doc_is_microdoc=payload.get("doc_is_microdoc", False),
                is_microdoc_stub=payload.get("is_microdoc_stub", False),
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
        if getattr(self.tokenizer, "supports_decode", False):
            tokens = self.tokenizer.encode(text)
            truncated_tokens = tokens[:token_budget]
            truncated_text = self.tokenizer.decode_tokens(truncated_tokens)
            return truncated_text, len(truncated_tokens)

        logger.warning(
            "Tokenizer backend %s does not support decode; truncating text approximately",
            getattr(self.tokenizer, "backend_name", "unknown"),
        )
        ratio = token_budget / total if total else 0
        approx_chars = max(1, int(len(text) * ratio))
        truncated_text = text[:approx_chars]
        return truncated_text, min(token_budget, total)

    def _chunk_from_props(self, props: Dict[str, Any]) -> ChunkResult:
        boundaries = props.get("boundaries_json", "{}")
        if isinstance(boundaries, dict):
            boundaries_json = json.dumps(boundaries, separators=(",", ":"))
        else:
            boundaries_json = boundaries or "{}"
        path_nodes = props.get("graph_path")
        if path_nodes and not isinstance(path_nodes, list):
            path_nodes = [path_nodes]
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
            doc_is_microdoc=bool(props.get("doc_is_microdoc", False)),
            is_microdoc_stub=bool(props.get("is_microdoc_stub", False)),
            embedding_version=props.get("embedding_version"),
            tenant=props.get("tenant"),
            citation_labels=[],
            graph_distance=int(props.get("graph_distance", 0)),
            graph_score=float(props.get("graph_score", 0.0)),
            graph_path=[str(node) for node in path_nodes] if path_nodes else None,
            connection_count=int(props.get("connection_count", 0)),
            mention_count=int(props.get("mention_count", 0)),
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

    def _enforce_context_budget(
        self, results: List[ChunkResult], starting_tokens: int = 0
    ) -> Tuple[List[ChunkResult], int]:
        """
        Enforce context budget by limiting total tokens.

        Groups by parent_section_id, sorts by order, and trims from tail
        when exceeding budget.

        Reference: Phase 7E Canonical Spec - Context Budget: Max 4,500 tokens
        """
        if not results:
            return [], starting_tokens

        from collections import OrderedDict

        total_tokens = starting_tokens
        final_results: List[ChunkResult] = []
        grouped: "OrderedDict[str, List[ChunkResult]]" = OrderedDict()

        for chunk in results:
            parent = chunk.parent_section_id or chunk.chunk_id
            grouped.setdefault(parent, []).append(chunk)

        for parent, chunks in grouped.items():
            taken = 0
            for chunk in chunks:
                tokens = max(0, chunk.token_count or 0)
                if total_tokens + tokens > self.context_max_tokens:
                    self._log_context_budget(total_tokens, len(final_results))
                    return final_results, total_tokens
                final_results.append(chunk)
                total_tokens += tokens
                taken += 1
                if taken >= max(1, self.context_group_cap):
                    break

        return final_results, total_tokens

    def _log_context_budget(self, tokens: int, count: int) -> None:
        logger.info(
            "Context budget enforced",
            extra={
                "chunks": count,
                "tokens": tokens,
                "max_tokens": self.context_max_tokens,
            },
        )

    def _log_stage_snapshot(
        self, stage: str, chunks: List[ChunkResult], limit: int = 5
    ) -> None:
        underlying = getattr(logger, "_logger", None) or getattr(logger, "logger", None)
        if not underlying or not underlying.isEnabledFor(logging.DEBUG):
            return
        sample = [getattr(c, "chunk_id", None) for c in chunks[:limit]]
        logger.debug(
            "Ranking stage snapshot",
            extra={"stage": stage, "count": len(chunks), "sample": sample},
        )

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

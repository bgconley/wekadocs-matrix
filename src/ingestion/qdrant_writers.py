# =============================================================================
# @status: ACTIVE
# @called-by: atomic.py
# =============================================================================
"""
Qdrant vector store write operations extracted from atomic.py.

Provides QdrantWriter class with:
- Batched upsert with dimension validation and retry
- Compensating deletes for rollback
- Telemetry via Prometheus metrics
"""

from __future__ import annotations

import os
import random
import time
import uuid as uuid_mod
from datetime import datetime
from typing import Dict, List, Tuple

from src.shared.embedding_fields import (
    canonicalize_embedding_metadata,
    ensure_no_embedding_model_in_payload,
)
from src.shared.observability import get_logger

logger = get_logger(__name__)


# Lazy exception loader to handle optional qdrant-client dependency
# This allows the exception to be used in except clauses without import-time errors
_RESPONSE_HANDLING_EXCEPTION_CACHE: type = None


def _get_response_handling_exception() -> type:
    """
    Lazy loader for qdrant_client.http.exceptions.ResponseHandlingException.

    Returns the exception class for use in except clauses. Falls back to a
    placeholder exception if qdrant-client is not installed or doesn't have
    the exception class (older versions).

    This is needed because ResponseHandlingException is raised on HTTP timeouts
    (e.g., when large ColBERT payloads exceed the client timeout).
    """
    global _RESPONSE_HANDLING_EXCEPTION_CACHE
    if _RESPONSE_HANDLING_EXCEPTION_CACHE is not None:
        return _RESPONSE_HANDLING_EXCEPTION_CACHE

    try:
        from qdrant_client.http.exceptions import ResponseHandlingException

        _RESPONSE_HANDLING_EXCEPTION_CACHE = ResponseHandlingException
    except ImportError:
        # Fallback for older qdrant-client versions or missing dependency
        # Use a placeholder that will never match
        class _PlaceholderException(Exception):
            pass

        _RESPONSE_HANDLING_EXCEPTION_CACHE = _PlaceholderException

    return _RESPONSE_HANDLING_EXCEPTION_CACHE


class QdrantWriter:
    """
    Handles Qdrant vector store write operations.

    Provides batched upsert with dimension validation + retry, and compensating
    deletes for rollback. Used by AtomicIngestionCoordinator.
    """

    def __init__(self, config, qdrant_client, neo4j_writer=None):
        """
        Initialize QdrantWriter.

        Args:
            config: Application configuration (needs config.search.vector.qdrant)
            qdrant_client: Qdrant client instance
            neo4j_writer: Optional Neo4jWriter for hash computations
        """
        self.config = config
        self.qdrant_client = qdrant_client
        self.neo4j_writer = neo4j_writer

    def _qdrant_upsert_vectors(
        self,
        document: Dict,
        sections: List[Dict],
        embeddings: Dict,
        builder,
    ) -> int:
        """Upsert vectors to Qdrant."""
        from qdrant_client.http.models import SparseVector
        from qdrant_client.models import PointStruct

        collection = self.config.search.vector.qdrant.collection_name
        points = []

        for section in sections:
            section_id = section.get("id")
            if not section_id:
                continue

            section_embeddings = embeddings.get("sections", {}).get(section_id)
            if not section_embeddings:
                continue

            # Convert to UUID for Qdrant
            point_uuid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, section_id))

            # Build CANONICAL payload matching build_graph.py schema (42 fields)
            # All fields must match for: graph reranker, multi-tenancy, filtering,
            # provenance, deduplication, drift detection, and embedding tracking
            text_content = section.get("text", "")

            # Compute embedding metadata using canonical helper
            # CRITICAL: These must come from builder config, not hardcoded fallbacks
            content_embedding = section_embeddings.get("content", [])

            if (
                not hasattr(builder, "embedding_settings")
                or not builder.embedding_settings
            ):
                raise ValueError(
                    "GraphBuilder missing embedding_settings - cannot create "
                    "canonical payload. Ensure embedding config is set."
                )

            embedding_metadata = canonicalize_embedding_metadata(
                embedding_model=builder.embedding_settings.version,
                dimensions=(
                    len(content_embedding)
                    if content_embedding
                    else builder.embedding_settings.dims
                ),
                provider=(
                    getattr(builder.embedder, "provider_name", None)
                    if hasattr(builder, "embedder")
                    else None
                ),
                task=builder.embedding_settings.task,
                profile=getattr(builder.embedding_settings, "profile", None),
                timestamp=datetime.utcnow(),
            )

            payload = {
                # === Core identifiers (6 fields) ===
                "id": section_id,
                "node_id": section_id,
                "kg_id": section_id,
                "document_id": document.get("id"),
                "doc_id": document.get("doc_id"),
                "node_label": "Chunk",
                # === Provenance (3 fields) ===
                "document_uri": document.get("document_uri"),
                "source_uri": document.get("source_uri"),
                "source_path": document.get("source_path"),
                # === Filtering (3 fields) ===
                "doc_tag": section.get("doc_tag") or document.get("doc_tag"),
                "snapshot_scope": section.get("snapshot_scope")
                or document.get("snapshot_scope"),
                "tenant": section.get("tenant") or document.get("tenant"),
                # === Chunk hierarchy (4 fields) ===
                "parent_section_id": section.get("parent_section_id"),
                "parent_section_original_id": section.get("parent_section_original_id"),
                "parent_chunk_id": section.get("parent_chunk_id"),
                "level": section.get("level", 3),
                # === Content (4 fields) ===
                "heading": section.get("title") or section.get("heading", ""),
                "text": text_content,
                "title": section.get("title"),
                "anchor": section.get("anchor"),
                # === Structural (5 fields) ===
                "order": section.get("order", 0),
                "token_count": section.get("token_count") or section.get("tokens", 0),
                "document_total_tokens": document.get("total_tokens", 0),
                "document_original_total_tokens": document.get("original_total_tokens"),
                "document_total_tokens_chunk": section.get("document_total_tokens"),
                # === Truncation metadata (2 fields) ===
                "was_truncated": section.get("was_truncated", False),
                "original_token_count": section.get("original_token_count"),
                # === Chunking metadata (4 fields) ===
                "is_combined": section.get("is_combined", False),
                "is_split": section.get("is_split", False),
                "original_section_ids": section.get(
                    "original_section_ids", [section.get("id")]
                ),
                "boundaries_json": section.get("boundaries_json", "{}"),
                # === Microdoc flags (3 fields) ===
                "is_microdoc": section.get("is_microdoc"),
                "doc_is_microdoc": section.get("doc_is_microdoc", False),
                "is_microdoc_stub": section.get("is_microdoc_stub", False),
                # === Document title (1 field) ===
                "doc_title": document.get("title", ""),
                # === Versioning (2 fields) ===
                "lang": section.get("lang") or document.get("lang"),
                "version": section.get("version") or document.get("version"),
                # === Timestamps (1 field) ===
                "updated_at": datetime.utcnow().isoformat() + "Z",
                # === Hashes for drift detection (2 fields) ===
                "text_hash": section.get("text_hash")
                or self.neo4j_writer._compute_text_hash(text_content),
                "shingle_hash": section.get("shingle_hash")
                or self.neo4j_writer._compute_shingle_hash(text_content),
                # === Semantic metadata (1 field) ===
                "semantic_metadata": self.neo4j_writer._extract_semantic_metadata(
                    section
                ),
                # === GLiNER entity metadata (1 field, Phase 2) ===
                # Added by enrich_chunks_with_entities() for filtering/boosting
                "entity_metadata": section.get("entity_metadata"),
                # === Phase 2: markdown-it-py structural metadata (7 fields) ===
                # Enable query-time filtering by structural characteristics
                "line_start": section.get("line_start"),
                "line_end": section.get("line_end"),
                "parent_path": section.get("parent_path", ""),
                "block_types": section.get("block_types", []),
                "code_ratio": section.get("code_ratio", 0.0),
                "has_code": section.get("has_code", False),
                "has_table": section.get("has_table", False),
                # === Phase 5: Derived structural fields for query-type adaptive retrieval ===
                # Computed at ingestion time for efficient Qdrant filtering
                "parent_path_depth": section.get("parent_path_depth", 0),
                "block_type": section.get("block_type", "paragraph"),
                # === Embedding metadata (5+ fields via spread) ===
                **embedding_metadata,
            }

            # CRITICAL: Remove legacy embedding_model field that may have leaked
            payload = ensure_no_embedding_model_in_payload(payload)

            # Build vectors dict with dense vectors
            vectors = {
                "content": section_embeddings["content"],
                "title": section_embeddings["title"],
                "doc_title": section_embeddings["doc_title"],
            }

            # REMOVED: Dense entity vector was broken (duplicated content embedding)
            # Now using entity-sparse for lexical entity name matching instead
            # See: build_graph.py and qdrant_schema.py for details

            # Add sparse vector if available (matching build_graph.py pattern)
            sparse_vector = section_embeddings.get("sparse")
            if sparse_vector:
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
                    vectors["text-sparse"] = SparseVector(
                        indices=list(indices), values=list(values)
                    )

            # Add doc_title sparse vector if available (literal title matches)
            doc_title_sparse_vector = section_embeddings.get("doc_title_sparse")
            if doc_title_sparse_vector:
                indices = (
                    doc_title_sparse_vector.get("indices")
                    if isinstance(doc_title_sparse_vector, dict)
                    else None
                )
                values = (
                    doc_title_sparse_vector.get("values")
                    if isinstance(doc_title_sparse_vector, dict)
                    else None
                )
                if indices and values:
                    vectors["doc_title-sparse"] = SparseVector(
                        indices=list(indices), values=list(values)
                    )

            # Add title sparse vector if available (lexical section heading match)
            title_sparse_vector = section_embeddings.get("title_sparse")
            if title_sparse_vector:
                indices = (
                    title_sparse_vector.get("indices")
                    if isinstance(title_sparse_vector, dict)
                    else None
                )
                values = (
                    title_sparse_vector.get("values")
                    if isinstance(title_sparse_vector, dict)
                    else None
                )
                if indices and values:
                    vectors["title-sparse"] = SparseVector(
                        indices=list(indices), values=list(values)
                    )

            # Add entity sparse vector if available (lexical matching on entity names)
            entity_sparse_vector = section_embeddings.get("entity_sparse")
            if entity_sparse_vector:
                indices = (
                    entity_sparse_vector.get("indices")
                    if isinstance(entity_sparse_vector, dict)
                    else None
                )
                values = (
                    entity_sparse_vector.get("values")
                    if isinstance(entity_sparse_vector, dict)
                    else None
                )
                if indices and values:
                    vectors["entity-sparse"] = SparseVector(
                        indices=list(indices), values=list(values)
                    )

            # Add ColBERT late-interaction vectors if available
            colbert_vectors = section_embeddings.get("colbert")
            if colbert_vectors:
                vectors["late-interaction"] = [
                    list(vector) for vector in colbert_vectors
                ]
                payload["colbert_vector_count"] = len(colbert_vectors)

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=vectors,
                    payload=payload,
                )
            )

        if points:
            # Phase 4: Build expected dimensions dict for validation
            # All dense vectors use the same embedding model
            expected_dim = {
                "content": builder.embedding_settings.dims,
                "title": builder.embedding_settings.dims,
                "doc_title": builder.embedding_settings.dims,
            }
            # ColBERT late-interaction uses a different dimension (128 for ColBERTv2)
            if hasattr(builder, "colbert_settings") and builder.colbert_settings:
                expected_dim["late-interaction"] = builder.colbert_settings.dims
            elif hasattr(builder, "colbert_dims"):
                expected_dim["late-interaction"] = builder.colbert_dims
            else:
                # Fallback: read from embedding plan
                from src.shared.config import get_embedding_plan

                plan = get_embedding_plan()
                if plan and plan.colbert and plan.colbert.enabled:
                    expected_dim["late-interaction"] = plan.colbert.profile.dims

            # Phase 7F: Batch Qdrant upserts to prevent timeout on large ColBERT
            # Large docs can produce 30MB+ JSON payloads exceeding 30s timeout.
            # Batching reduces payload size. See: Debug 2025-12-01 fix.
            batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "5"))
            max_bytes_per_batch = int(
                os.getenv("QDRANT_UPSERT_MAX_BYTES", str(12 * 1024 * 1024))
            )  # ~12MB default target

            json_overhead_factor = 2.5  # conservative multiplier for JSON vs. binary
            # Dense vectors: content, title, doc_title (3 standard dense)
            # late-interaction is multivector with variable size, handled separately
            # entity dense removed (2025-12-06), replaced by entity-sparse
            dense_vector_count = 3  # content + title + doc_title
            dense_dim_bytes = int(
                builder.embedding_settings.dims
                * dense_vector_count
                * 4
                * json_overhead_factor
            )

            def estimate_point_bytes(point: PointStruct) -> int:
                payload = getattr(point, "payload", {}) or {}
                vectors = getattr(point, "vector", {}) or {}

                token_count = (
                    payload.get("token_count")
                    or payload.get("tokens")
                    or payload.get("document_total_tokens_chunk")
                    or 0
                )
                colbert_vectors = vectors.get("late-interaction")
                colbert_tokens = len(colbert_vectors) if colbert_vectors else 0
                estimated_tokens = colbert_tokens or token_count or 0

                colbert_bytes = int(
                    estimated_tokens
                    * builder.embedding_settings.dims
                    * 4
                    * json_overhead_factor
                )
                return colbert_bytes + dense_dim_bytes

            batches: List[Tuple[List[PointStruct], int]] = []
            current_batch: List[PointStruct] = []
            current_bytes = 0

            for point in points:
                point_bytes = estimate_point_bytes(point)
                if point_bytes > max_bytes_per_batch:
                    logger.warning(
                        "qdrant_point_exceeds_max_bytes",
                        point_id=getattr(point, "id", None),
                        estimated_mb=round(point_bytes / 1_000_000, 2),
                        max_mb=round(max_bytes_per_batch / 1_000_000, 2),
                    )

                should_flush = current_batch and (
                    len(current_batch) >= batch_size
                    or current_bytes + point_bytes > max_bytes_per_batch
                )
                if should_flush:
                    batches.append((current_batch, current_bytes))
                    current_batch = []
                    current_bytes = 0

                current_batch.append(point)
                current_bytes += point_bytes

            if current_batch:
                batches.append((current_batch, current_bytes))

            total_batches = len(batches)

            for batch_num, (batch, batch_bytes) in enumerate(batches, start=1):
                logger.debug(
                    "qdrant_upsert_batch",
                    batch_num=batch_num,
                    total_batches=total_batches,
                    batch_size=len(batch),
                    total_points=len(points),
                    batch_estimated_mb=round(batch_bytes / 1_000_000, 2),
                    max_batch_mb=round(max_bytes_per_batch / 1_000_000, 2),
                )

                # Use retry wrapper with dimension validation for transient errors
                self._qdrant_upsert_with_retry(collection, batch, expected_dim)

        return len(points)

    def _qdrant_upsert_with_retry(
        self,
        collection: str,
        points: List,
        expected_dim: Dict[str, int],
        max_retries: int = 3,
    ) -> None:
        """
        Upsert points to Qdrant with dimension validation and exponential backoff retry.

        Uses upsert_validated() which provides:
        - Pre-upsert dimension validation for all vectors
        - Prometheus metrics (qdrant_upsert_total, qdrant_operation_latency_ms)

        Handles transient network failures common in distributed systems.
        Non-retriable errors (schema/dimension mismatch) fail immediately.

        Args:
            collection: Qdrant collection name
            points: List of PointStruct to upsert
            expected_dim: Expected vector dims, e.g. {"content": 1024}
            max_retries: Maximum retry attempts

        Raises:
            ValueError: If any vector dimension doesn't match expected_dim
        """
        last_exception = None
        base_delay = 0.5

        for attempt in range(max_retries + 1):
            try:
                # Phase 4: Use upsert_validated for dimension validation + metrics
                self.qdrant_client.upsert_validated(
                    collection_name=collection,
                    points=points,
                    expected_dim=expected_dim,
                    wait=True,
                )
                return  # Success
            except (
                ConnectionError,
                TimeoutError,
                OSError,
                _get_response_handling_exception(),
            ) as e:
                # Phase 7F: Extended retry includes ResponseHandlingException
                # (raised by qdrant-client on HTTP timeouts, e.g., large ColBERT)
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 30.0)
                    delay = delay * (1 + random.random() * 0.25)  # Add jitter

                    logger.warning(
                        "qdrant_upsert_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=round(delay, 2),
                        points_count=len(points),
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "qdrant_upsert_exhausted",
                        attempts=max_retries + 1,
                        points_count=len(points),
                        error=str(e),
                    )
                    raise
            except ValueError:
                # Dimension mismatch - non-retriable, fail fast
                raise
            except Exception as e:
                # Phase 7F: Log non-retriable exceptions before re-raising
                # Preserves error context that was previously lost
                logger.error(
                    "qdrant_upsert_non_retriable_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    points_count=len(points),
                )
                raise

        if last_exception:
            raise last_exception

    def _compensate_qdrant(self, point_ids: List[str], builder):
        """Delete points from Qdrant as compensation with retry."""
        collection = self.config.search.vector.qdrant.collection_name

        # Convert to UUIDs
        uuids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, pid)) for pid in point_ids]

        # Retry compensation - critical for data consistency
        self._qdrant_delete_with_retry(collection, uuids)

        logger.info(
            "qdrant_compensation_completed",
            points_deleted=len(uuids),
        )

    def _qdrant_delete_with_retry(
        self,
        collection: str,
        point_ids: List[str],
        max_retries: int = 3,
    ) -> None:
        """
        Delete points from Qdrant with retry logic and Prometheus telemetry.

        Compensation is critical - we retry harder to ensure consistency.
        Phase 4: Records qdrant_delete_total and qdrant_operation_latency_ms.

        Args:
            collection: Qdrant collection name
            point_ids: List of point UUIDs to delete
            max_retries: Maximum retry attempts
        """
        # Phase 4: Import metrics for delete telemetry
        from src.shared.observability.metrics import (
            qdrant_delete_total,
            qdrant_operation_latency_ms,
        )

        last_exception = None
        base_delay = 1.0  # Longer base delay for compensation
        start_time = time.time()
        status = "success"

        for attempt in range(max_retries + 1):
            try:
                self.qdrant_client.delete(
                    collection_name=collection,
                    points_selector=point_ids,
                    wait=True,
                )
                # Record success metrics
                latency_ms = (time.time() - start_time) * 1000
                qdrant_delete_total.labels(
                    collection_name=collection, status=status
                ).inc()
                qdrant_operation_latency_ms.labels(
                    collection_name=collection, operation="delete"
                ).observe(latency_ms)
                return
            except (ConnectionError, TimeoutError, OSError) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 60.0)
                    delay = delay * (1 + random.random() * 0.25)

                    logger.warning(
                        "qdrant_compensation_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=round(delay, 2),
                        points_count=len(point_ids),
                        error=str(e),
                    )
                    time.sleep(delay)
                else:
                    # Record error metrics before raising
                    status = "error"
                    latency_ms = (time.time() - start_time) * 1000
                    qdrant_delete_total.labels(
                        collection_name=collection, status=status
                    ).inc()
                    qdrant_operation_latency_ms.labels(
                        collection_name=collection, operation="delete"
                    ).observe(latency_ms)
                    logger.error(
                        "qdrant_compensation_exhausted",
                        attempts=max_retries + 1,
                        points_count=len(point_ids),
                        error=str(e),
                    )
                    raise
            except Exception:
                # Record error metrics for non-retriable exceptions
                status = "error"
                latency_ms = (time.time() - start_time) * 1000
                qdrant_delete_total.labels(
                    collection_name=collection, status=status
                ).inc()
                qdrant_operation_latency_ms.labels(
                    collection_name=collection, operation="delete"
                ).observe(latency_ms)
                raise

        if last_exception:
            raise last_exception

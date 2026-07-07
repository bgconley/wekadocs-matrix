# =============================================================================
# @status: ACTIVE
# @called-by: worker.py
# =============================================================================
"""
Atomic Ingestion Coordinator for Neo4j + Qdrant Synchronization.

This module provides a transactional wrapper around the GraphBuilder that ensures
atomic commits to both Neo4j and Qdrant, with rollback on failure.

Key Pattern: Deferred Commit
1. Prepare Phase: Compute all embeddings and validate data BEFORE any writes
2. Write Phase: Start Neo4j transaction, write to Neo4j, write to Qdrant
3. Commit Phase: Only commit Neo4j AFTER Qdrant succeeds
4. Compensate Phase: If Qdrant fails, rollback Neo4j;
   if Neo4j committed, delete from Qdrant

This ensures that chunk IDs in Neo4j always have corresponding vectors in Qdrant.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.shared.observability import get_logger

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: Tuple[type, ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    ),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retry with exponential backoff.

    Production-hardened retry logic for transient failures.

    Args:
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 0.5)
        max_delay: Maximum delay cap in seconds (default: 30.0)
        exponential_base: Base for exponential growth (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        retriable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry behavior
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retriable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                        if jitter:
                            # Add up to 25% random jitter
                            delay = delay * (1 + random.random() * 0.25)

                        logger.warning(
                            "retry_attempt",
                            func=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay_seconds=round(delay, 2),
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_retries + 1,
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        raise
                except Exception:
                    # Non-retriable exception, raise immediately
                    raise
            raise last_exception  # Should not reach here, but for type safety

        return wrapper

    return decorator


# Backward-compatible re-exports for tests that import from atomic
from src.ingestion.neo4j_writers import ALLOWED_ENTITY_RELATIONSHIP_TYPES  # noqa: E402
from src.ingestion.neo4j_writers import Neo4jWriter  # noqa: E402
from src.ingestion.qdrant_writers import QdrantWriter  # noqa: E402
from src.ingestion.saga import (  # noqa: E402
    IngestionValidator,
    SagaContext,
    ValidationResult,
)
from src.ingestion.stages.chunk import assemble_chunks  # noqa: E402
from src.ingestion.stages.embed import compute_embeddings  # noqa: E402
from src.ingestion.stages.enrich import (  # noqa: E402
    enrich_chunks_with_gliner,
    extract_and_enrich,
    merge_section_mentions,
)
from src.ingestion.stages.link import (  # noqa: E402
    create_cross_doc_links,
    get_document_count,
)
from src.ingestion.stages.parse import parse_document  # noqa: E402
from src.ingestion.stages.write import execute_saga  # noqa: E402

__all__ = [
    "ALLOWED_ENTITY_RELATIONSHIP_TYPES",
    "AtomicIngestionCoordinator",
    "AtomicIngestionResult",
    "IngestionValidator",
    "Neo4jWriter",
    "QdrantWriter",
    "SagaContext",
    "ValidationResult",
    "ingest_document_atomic",
]

# LGTM Phase 4: OTEL tracing for ingestion pipeline observability
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore

logger = get_logger(__name__)

# LGTM Phase 4: Tracer for ingestion pipeline spans
_tracer = trace.get_tracer("nutanixdocs.ingestion") if OTEL_AVAILABLE else None


@dataclass
class AtomicIngestionResult:
    """Result of an atomic ingestion operation."""

    success: bool
    document_id: str
    saga_id: str
    stats: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[ValidationResult] = None
    error: Optional[str] = None
    duration_ms: int = 0
    neo4j_committed: bool = False
    qdrant_committed: bool = False
    compensated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "document_id": self.document_id,
            "saga_id": self.saga_id,
            "stats": self.stats,
            "validation": self.validation.to_dict() if self.validation else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "neo4j_committed": self.neo4j_committed,
            "qdrant_committed": self.qdrant_committed,
            "compensated": self.compensated,
        }


class AtomicIngestionCoordinator:
    """
    Coordinates atomic ingestion to Neo4j and Qdrant with rollback capability.

    Usage:
        coordinator = AtomicIngestionCoordinator(neo4j_driver, qdrant_client, config)
        result = coordinator.ingest_document_atomic(
            source_uri="file:///path/to/doc.md",
            content="# Document content...",
            format="markdown"
        )
        if not result.success:
            logger.error("Ingestion failed", error=result.error)
    """

    def __init__(
        self,
        neo4j_driver,
        qdrant_client,
        config,
        *,
        validate_before_commit: bool = True,
        strict_mode: Optional[bool] = None,
    ):
        """
        Initialize the atomic ingestion coordinator.

        Args:
            neo4j_driver: Neo4j driver instance
            qdrant_client: Qdrant client instance
            config: Application configuration
            validate_before_commit: Run pre-commit validation (recommended)
            strict_mode: Fail on validation warnings (not just errors).
                         If None, reads from VALIDATION_STRICT_MODE env var.
                         Explicit True/False overrides config for compat.
        """
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.config = config
        self.neo4j_writer = Neo4jWriter(config)
        self.qdrant_writer = QdrantWriter(config, qdrant_client, self.neo4j_writer)
        self.validate_before_commit = validate_before_commit

        # Phase 5.2: Unified strict mode config
        # Read from Settings if not explicitly passed, with backward compat override
        if strict_mode is None:
            from src.shared.config import get_settings

            settings = get_settings()
            self.strict_mode = settings.validation_strict_mode
        else:
            self.strict_mode = strict_mode

        # DEPRECATED: Dense entity vector has been removed (2025-12-06)
        # The entity dense vector was broken - it duplicated content embedding.
        # Replaced by entity-sparse for lexical entity name matching.
        # This flag is kept for backward compatibility but always evaluates to False.
        # TODO: Remove this flag entirely in a future cleanup.
        self.include_entity_vector = False  # Always False - entity dense removed

        self.validator = IngestionValidator(neo4j_driver, qdrant_client, config)

    # -------------------------------------------------------------------------
    # Cross-Document Linking (Phase 3.5)
    # -------------------------------------------------------------------------

    def _get_document_count(self) -> int:
        """Get total document count for corpus size check."""
        return get_document_count(self.neo4j_driver)

    def _create_cross_doc_links(
        self,
        document_id: str,
        document: Dict,
        sections: List[Dict],
        embeddings: Dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Create cross-document links for a newly ingested document.

        This is called AFTER the Neo4j commit to ensure the document exists.
        Failures here are logged but NEVER fail the ingestion.

        Args:
            document_id: The document ID
            document: Document dict with title, etc.
            sections: List of section dicts
            embeddings: Dict with section embeddings

        Returns:
            Dict with linking stats, or None if skipped/failed
        """
        return create_cross_doc_links(
            neo4j_driver=self.neo4j_driver,
            qdrant_client=self.qdrant_client,
            config=self.config,
            document_id=document_id,
            document=document,
            sections=sections,
            embeddings=embeddings,
        )

    def ingest_document_atomic(
        self,
        source_uri: str,
        content: str,
        format: str = "markdown",
        *,
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
    ) -> AtomicIngestionResult:
        """
        Atomically ingest a document to both Neo4j and Qdrant.

        This is the main entry point that replaces non-atomic ingest_document calls.

        LGTM Phase 4: Enhanced with verbose logging and OTEL spans for full
        observability of the ingestion pipeline.

        Args:
            source_uri: Document source URI
            content: Document content
            format: Content format (markdown, html)
            embedding_model: Optional embedding model override
            embedding_version: Optional embedding version override

        Returns:
            AtomicIngestionResult with success status and metadata
        """
        start_time = time.time()
        saga_id = str(uuid.uuid4())

        # LGTM Phase 4: Extract feature flags for observability
        feature_flags = {}
        if self.config:
            ff = getattr(self.config, "feature_flags", None)
            cross_doc = getattr(self.config, "cross_doc_linking", None)
            if ff:
                feature_flags = {
                    "graph_as_reranker": getattr(ff, "graph_as_reranker", False),
                    "structure_aware_expansion": getattr(
                        ff, "structure_aware_expansion", False
                    ),
                    "graph_garbage_filter": getattr(ff, "graph_garbage_filter", False),
                    "dedup_best_score": getattr(ff, "dedup_best_score", False),
                }
            if cross_doc:
                feature_flags["cross_doc_linking_enabled"] = getattr(
                    cross_doc, "enabled", False
                )
                feature_flags["colbert_rerank"] = getattr(
                    cross_doc, "colbert_rerank", False
                )

        # LGTM Phase 4: Verbose log event 1 - ingestion_started
        logger.info(
            "ingestion_started",
            saga_id=saga_id,
            doc_path=source_uri,
            format=format,
            content_length=len(content),
            feature_flags=feature_flags,
        )

        # LGTM Phase 4: Create root span for full ingestion trace
        span_ctx = None
        if OTEL_AVAILABLE and _tracer:
            span_ctx = _tracer.start_as_current_span(
                "ingest_document",
                attributes={
                    "document.source_uri": source_uri,
                    "document.format": format,
                    "document.content_length": len(content),
                    "saga.id": saga_id,
                },
            )
            span_ctx.__enter__()

        try:
            # Phase 1: Parse and prepare all data
            parse_start = time.time()
            prepared = self._prepare_ingestion(
                source_uri,
                content,
                format,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
            )
            parse_time_ms = (time.time() - parse_start) * 1000

            document = prepared["document"]
            sections = prepared["sections"]
            entities = prepared["entities"]
            mentions = prepared["mentions"]
            references = prepared.get("references", [])  # Phase 3: Cross-doc refs
            document_id = document["id"]

            merge_section_mentions(sections, entities, mentions)

            # LGTM Phase 4: Verbose log event 2 - document_parsed
            logger.info(
                "document_parsed",
                doc_id=document_id,
                saga_id=saga_id,
                sections_count=len(sections),
                entities_count=len(entities),
                mentions_count=len(mentions),
                references_count=len(references),
                total_chars=sum(
                    len(s.get("text", "") or s.get("content", "")) for s in sections
                ),
                parse_time_ms=round(parse_time_ms, 2),
            )

            # LGTM Phase 4: Verbose log event 3 - chunking_complete
            # (sections are the chunks in our architecture)
            # Note: Parser uses "tokens", assembler uses "token_count"
            def _get_tokens(s: Dict) -> int:
                return int(s.get("token_count") or s.get("tokens") or 0)

            total_tokens = sum(_get_tokens(s) for s in sections)
            avg_tokens = total_tokens / len(sections) if sections else 0
            sample_chunk = sections[0] if sections else {}
            logger.info(
                "chunking_complete",
                doc_id=document_id,
                saga_id=saga_id,
                chunks_count=len(sections),
                total_tokens=total_tokens,
                avg_tokens_per_chunk=round(avg_tokens, 1),
                max_tokens=max((_get_tokens(s) for s in sections), default=0),
                min_tokens=min((_get_tokens(s) for s in sections), default=0),
                sample_chunk_text=(
                    (sample_chunk.get("text", "") or sample_chunk.get("content", ""))[
                        :200
                    ]
                    if sample_chunk
                    else None
                ),
                sample_chunk_title=sample_chunk.get("title")
                or sample_chunk.get("heading"),
            )

            # Phase 2: Pre-commit validation
            if self.validate_before_commit:
                validation = self.validator.validate_pre_ingest(
                    document_id, sections, entities, mentions
                )

                if not validation.valid:
                    return AtomicIngestionResult(
                        success=False,
                        document_id=document_id,
                        saga_id=saga_id,
                        validation=validation,
                        error=f"Validation failed: {validation.errors}",
                        duration_ms=int((time.time() - start_time) * 1000),
                    )

                if self.strict_mode and validation.warnings:
                    return AtomicIngestionResult(
                        success=False,
                        document_id=document_id,
                        saga_id=saga_id,
                        validation=validation,
                        error=f"Strict mode validation warnings: {validation.warnings}",
                        duration_ms=int((time.time() - start_time) * 1000),
                    )
            else:
                validation = None

            # Phase 3: Compute embeddings BEFORE any writes
            embed_start = time.time()
            embeddings = self._compute_embeddings(
                document, sections, entities, prepared["builder"]
            )
            embed_time_ms = (time.time() - embed_start) * 1000

            # LGTM Phase 4: Verbose log event 4 - embeddings_generated
            section_embeddings = embeddings.get("sections") or {}
            embedding_count = (
                len(section_embeddings) if isinstance(section_embeddings, dict) else 0
            )
            sample_embedding = (
                next(iter(section_embeddings.values()), {})
                if isinstance(section_embeddings, dict)
                else {}
            )
            dense_dim = (
                len(sample_embedding.get("content", [])) if sample_embedding else 0
            )
            has_sparse = bool(sample_embedding.get("sparse"))
            has_colbert = bool(sample_embedding.get("colbert"))
            logger.info(
                "embeddings_generated",
                doc_id=document_id,
                saga_id=saga_id,
                embedding_count=embedding_count,
                dense_dim=dense_dim,
                has_sparse=has_sparse,
                has_colbert=has_colbert,
                embed_time_ms=round(embed_time_ms, 2),
                vector_types=["content", "title", "doc_title"]
                + (["late-interaction"] if has_colbert else []),
            )

            # Phase 3b: Recompute document token aggregates after truncation
            truncated_sections = [s for s in sections if s.get("was_truncated")]
            if truncated_sections:
                new_total = sum(int(s.get("token_count", 0)) for s in sections)
                old_total = document.get("total_tokens", 0)
                tokens_reduced = old_total - new_total

                document["total_tokens"] = new_total
                document["original_total_tokens"] = old_total

                for section in sections:
                    section["document_total_tokens"] = new_total

                logger.info(
                    "document_token_totals_recomputed",
                    document_id=document_id,
                    truncated_section_count=len(truncated_sections),
                    old_total=old_total,
                    new_total=new_total,
                    tokens_reduced=tokens_reduced,
                )

            # Phase 4: Execute atomic writes with saga coordination
            saga_start = time.time()
            saga_result = self._execute_atomic_saga(
                saga_id=saga_id,
                document=document,
                sections=sections,
                entities=entities,
                mentions=mentions,
                references=references,
                embeddings=embeddings,
                builder=prepared["builder"],
            )
            saga_time_ms = (time.time() - saga_start) * 1000

            # LGTM Phase 4: Verbose log events 5 & 6 emitted in _execute_atomic_saga
            # (neo4j_write_complete and qdrant_upsert_complete)

            duration_ms = int((time.time() - start_time) * 1000)

            if saga_result["success"]:
                # LGTM Phase 4: Enhanced completion logging
                stats = saga_result.get("stats", {})
                if stats is None:
                    stats = {}
                logger.info(
                    "ingestion_complete",
                    doc_id=document_id,
                    saga_id=saga_id,
                    duration_ms=duration_ms,
                    parse_time_ms=round(parse_time_ms, 2),
                    embed_time_ms=round(embed_time_ms, 2),
                    saga_time_ms=round(saga_time_ms, 2),
                    total_chunks=len(sections),
                    total_nodes=stats.get("nodes_created", 0),
                    total_edges=stats.get("relationships_created", 0),
                    cross_doc_edges=stats.get("cross_doc_edges", 0),
                )

                # Set span status if available
                if OTEL_AVAILABLE and span_ctx:
                    span = trace.get_current_span()
                    if span and span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("ingestion.success", True)
                        span.set_attribute("ingestion.duration_ms", duration_ms)
                        span.set_attribute("ingestion.chunks_count", len(sections))

                # Structural edges are now built atomically inside the saga
                # transaction (Step 2c in _execute_atomic_saga). No post-commit
                # best-effort building needed.

                return AtomicIngestionResult(
                    success=True,
                    document_id=document_id,
                    saga_id=saga_id,
                    stats=stats,
                    validation=validation,
                    duration_ms=duration_ms,
                    neo4j_committed=True,
                    qdrant_committed=True,
                )
            else:
                logger.error(
                    "atomic_ingestion_failed",
                    saga_id=saga_id,
                    document_id=document_id,
                    error=saga_result.get("error"),
                    compensated=saga_result.get("compensated", False),
                )

                # Set error span status
                if OTEL_AVAILABLE and span_ctx:
                    span = trace.get_current_span()
                    if span and span.is_recording():
                        span.set_status(
                            Status(
                                StatusCode.ERROR, saga_result.get("error", "unknown")
                            )
                        )

                return AtomicIngestionResult(
                    success=False,
                    document_id=document_id,
                    saga_id=saga_id,
                    stats=saga_result.get("stats", {}),
                    validation=validation,
                    error=saga_result.get("error"),
                    duration_ms=duration_ms,
                    neo4j_committed=saga_result.get("neo4j_committed", False),
                    qdrant_committed=saga_result.get("qdrant_committed", False),
                    compensated=saga_result.get("compensated", False),
                )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(
                "atomic_ingestion_exception",
                saga_id=saga_id,
                error=str(e),
            )

            # Set exception span status
            if OTEL_AVAILABLE and span_ctx:
                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)

            return AtomicIngestionResult(
                success=False,
                document_id="unknown",
                saga_id=saga_id,
                error=str(e),
                duration_ms=duration_ms,
            )
        finally:
            # LGTM Phase 4: Close span if opened
            if span_ctx:
                span_ctx.__exit__(None, None, None)

    def _prepare_ingestion(
        self,
        source_uri: str,
        content: str,
        format: str,
        *,
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare all data for ingestion without writing to any store.

        Returns:
            Dict with document, sections, entities, mentions, and builder
        """
        from src.ingestion.build_graph import GraphBuilder

        parsed = parse_document(
            source_uri,
            content,
            format,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )
        config = parsed["config"]
        document = parsed["document"]
        sections = parsed["sections"]

        enrichment = extract_and_enrich(
            document=document,
            sections=sections,
            content=content,
            format=format,
            config=config,
        )
        entities = enrichment["entities"]
        mentions = enrichment["mentions"]
        reference_edges = enrichment["references"]

        sections = assemble_chunks(document, sections, config)
        enrich_chunks_with_gliner(document=document, sections=sections, config=config)

        # Create builder (without writing)
        builder = GraphBuilder(self.neo4j_driver, config, self.qdrant_client)

        return {
            "document": document,
            "sections": sections,
            "entities": entities,
            "mentions": mentions,
            "references": reference_edges,  # Phase 3: Cross-document REFERENCES
            "builder": builder,
        }

    def _compute_embeddings(
        self,
        document: Dict,
        sections: List[Dict],
        entities: Dict,
        builder,
    ) -> Dict[str, Any]:
        """
        Compute all embeddings before any writes with production-grade batching.

        Ensures vector data is ready before the atomic transaction.
        Computes dense, sparse, and ColBERT embeddings (build_graph.py parity).
        """
        return compute_embeddings(
            document,
            sections,
            entities,
            builder,
            self.config,
        )

    def _execute_atomic_saga(
        self,
        saga_id: str,
        document: Dict,
        sections: List[Dict],
        entities: Dict,
        mentions: List[Dict],
        references: List[Dict],  # Phase 3: Cross-document REFERENCES
        embeddings: Dict,
        builder,
    ) -> Dict[str, Any]:
        """
        Execute the atomic saga with Neo4j and Qdrant writes.

        Order:
        1. Neo4j writes (in a transaction)
        2. Qdrant writes
        3. Commit Neo4j (only if Qdrant succeeds)

        Phase 3: Added references parameter for cross-document REFERENCES edges.
        """
        return execute_saga(
            saga_id=saga_id,
            document=document,
            sections=sections,
            entities=entities,
            mentions=mentions,
            references=references,
            embeddings=embeddings,
            builder=builder,
            neo4j_driver=self.neo4j_driver,
            qdrant_client=self.qdrant_client,
            neo4j_writer=self.neo4j_writer,
            qdrant_writer=self.qdrant_writer,
            config=self.config,
        )


# ============================================================================
# Convenience Function for Migration
# ============================================================================


def ingest_document_atomic(
    source_uri: str,
    content: str,
    format: str = "markdown",
    *,
    embedding_model: Optional[str] = None,
    embedding_version: Optional[str] = None,
    validate: bool = True,
    strict: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Top-level atomic ingestion function.

    This is a drop-in replacement for the non-atomic ingest_document function,
    with the same signature but atomic commit guarantees.

    Args:
        source_uri: Document source URI
        content: Document content
        format: Content format (markdown, html)
        embedding_model: Optional embedding model override
        embedding_version: Optional embedding version override
        validate: Run pre-commit validation
        strict: Fail on validation warnings. If None (default), reads from
                VALIDATION_STRICT_MODE env var. Pass True/False to override.

    Returns:
        Dict with ingestion stats (same format as ingest_document)
    """
    from neo4j import GraphDatabase

    from src.shared.config import get_config, get_settings
    from src.shared.connections import CompatQdrantClient

    config = get_config()
    settings = get_settings()

    neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=3600,
    )

    qdrant_client = None
    if config.search.vector.primary == "qdrant" or config.search.vector.dual_write:
        qdrant_client = CompatQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )

    try:
        coordinator = AtomicIngestionCoordinator(
            neo4j_driver,
            qdrant_client,
            config,
            validate_before_commit=validate,
            strict_mode=strict,
        )

        result = coordinator.ingest_document_atomic(
            source_uri,
            content,
            format,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )

        if result.success:
            return result.stats
        else:
            raise RuntimeError(f"Atomic ingestion failed: {result.error}")

    finally:
        neo4j_driver.close()
        if qdrant_client is not None:
            try:
                # Close Qdrant client if it has a close method
                if hasattr(qdrant_client, "close"):
                    qdrant_client.close()
            except Exception as e:
                logger.warning("qdrant_client_close_failed", error=str(e))

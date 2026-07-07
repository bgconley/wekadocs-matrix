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

import os
import random
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
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
from src.ingestion.stages.link import (  # noqa: E402
    create_cross_doc_links,
    get_document_count,
)
from src.providers.factory import ProviderFactory  # noqa: E402
from src.providers.tokenizer_service import TokenizerService  # noqa: E402
from src.shared.chunk_utils import validate_chunk_schema  # noqa: E402
from src.shared.embedding_fields import (  # noqa: E402
    canonicalize_embedding_metadata,
    validate_embedding_metadata,
)

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

            # Structural entity quality gate: filter noisy regex-extracted entities
            # at the merge point (upstream of entity-sparse vector generation and
            # Neo4j MENTIONS creation).
            from src.providers.ner.labels import is_excluded_structural_entity

            # Attach mentions to sections for entity-sparse embedding generation
            # Build section_id → mentions mapping (mirrors build_graph.py:454 logic)
            mentions_by_section: Dict[str, List[Dict]] = defaultdict(list)
            for mention in mentions:
                # Section→Entity mentions have section_id key
                section_id = mention.get("section_id")
                if section_id:
                    mentions_by_section[section_id].append(mention)

            # Attach _mentions to each section
            # Note: Chunk assembly creates new section IDs; original IDs are stored
            # in 'original_section_ids'. Check both current ID and originals.
            for section in sections:
                section_mentions = []
                # Check current section ID
                section_id = section.get("id")
                if section_id and section_id in mentions_by_section:
                    section_mentions.extend(mentions_by_section[section_id])
                # Check original section IDs (from chunk assembly)
                original_ids = section.get("original_section_ids", [])
                for orig_id in original_ids:
                    if orig_id in mentions_by_section:
                        section_mentions.extend(mentions_by_section[orig_id])
                # Merge structural mentions with any existing GLiNER mentions
                # GLiNER adds _mentions in _prepare_ingestion; preserve them here
                existing_gliner_mentions = section.get("_mentions", [])

                # Deduplicate by entity_id across both sources to avoid double-counting
                seen_entity_ids = set()
                merged_mentions = []

                # Add GLiNER mentions first (they're higher quality - model-extracted)
                for m in existing_gliner_mentions:
                    eid = m.get("entity_id")
                    if eid and eid not in seen_entity_ids:
                        seen_entity_ids.add(eid)
                        merged_mentions.append(m)

                # Then add structural mentions (regex-extracted)
                for m in section_mentions:
                    eid = m.get("entity_id")
                    if not eid or eid in seen_entity_ids:
                        continue

                    # Structural mention dicts do not carry a name; resolve via
                    # the entities dict produced by structural extractors.
                    entity_name = ""
                    entity_data = (
                        entities.get(eid) if isinstance(entities, dict) else None
                    )
                    if isinstance(entity_data, dict):
                        entity_name = entity_data.get("name", "") or ""

                    if is_excluded_structural_entity(entity_name):
                        continue

                    seen_entity_ids.add(eid)
                    merged_mentions.append(m)

                section["_mentions"] = merged_mentions

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
        from pathlib import Path

        from src.ingestion.build_graph import GraphBuilder
        from src.ingestion.chunk_assembler import get_chunk_assembler
        from src.ingestion.extract import extract_entities
        from src.ingestion.parsers import parse_markdown  # Router selects engine
        from src.ingestion.parsers.html import parse_html
        from src.shared.config import get_config, get_settings

        # Deep copy config to avoid mutating the global singleton when applying
        # per-request embedding overrides. This ensures thread safety and
        # prevents cross-request interference in concurrent workers.
        # See: Phase 1 bug fix for config singleton mutation
        config = get_config().model_copy(deep=True)
        _ = get_settings()  # Validates settings load; value unused

        # Apply optional overrides with explicit logging
        if embedding_model:
            try:
                config.embedding.embedding_model = embedding_model
                logger.debug("embedding_model_override_applied", model=embedding_model)
            except AttributeError as e:
                logger.warning(
                    "embedding_model_override_failed",
                    model=embedding_model,
                    error=str(e),
                )
        if embedding_version:
            try:
                config.embedding.version = embedding_version
                logger.debug(
                    "embedding_version_override_applied", version=embedding_version
                )
            except AttributeError as e:
                logger.warning(
                    "embedding_version_override_failed",
                    version=embedding_version,
                    error=str(e),
                )

        # Parse document
        if format == "markdown":
            result = parse_markdown(source_uri, content)
        elif format == "html":
            result = parse_html(source_uri, content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        document = result["Document"]
        sections = result["Sections"]

        # Extract doc_tag and snapshot_scope
        # Priority:
        # 1. Explicit DocTag: header in content
        # 2. First-level directory under data/ingest/ (category from path)
        # 3. Filename with __ separator (scope__slug pattern)
        # 4. Filename stem as fallback
        doc_tag = None
        snapshot_scope = None
        doc_category = None  # New: category from directory path

        m = re.search(r"DocTag:\s*([A-Za-z0-9_\-]+)", content or "", flags=re.I)
        if m:
            doc_tag = m.group(1)
        else:
            try:
                source_path = Path(
                    source_uri.replace("file://", "") if source_uri else ""
                )
                fname = source_path.name
                stem = Path(fname).stem

                # NEW: Extract category from directory path relative to data/ingest/
                # e.g., /app/data/ingest/nutanix-platform/overview.md -> category="nutanix-platform"
                # e.g., /app/data/ingest/aws-solutions/sagemaker/guide.md → category="aws-solutions"
                path_parts = source_path.parts
                for i, part in enumerate(path_parts):
                    if part == "ingest" or part.endswith("ingest"):
                        # First directory after "ingest" is the category
                        if i + 1 < len(path_parts) - 1:  # Not the filename itself
                            doc_category = path_parts[i + 1]
                        break

                if "__" in stem:
                    scope_part, slug_part = stem.split("__", 1)
                    snapshot_scope = scope_part
                    doc_tag = slug_part
                else:
                    # Use category from path if available, otherwise filename
                    doc_tag = doc_category if doc_category else stem
            except (ValueError, AttributeError) as e:
                logger.debug(
                    "doc_tag_extraction_fallback",
                    source_uri=source_uri,
                    error=str(e),
                )
                # doc_tag remains None, which is acceptable

        document["doc_tag"] = doc_tag
        document["doc_category"] = doc_category  # New: category from directory path
        document["snapshot_scope"] = snapshot_scope

        for section in sections:
            section["doc_tag"] = doc_tag
            section["doc_category"] = doc_category
            section["snapshot_scope"] = snapshot_scope

        # Extract entities
        entities, mentions = extract_entities(sections)

        # Phase 3: Extract cross-document references
        # Import here to avoid circular imports
        from src.ingestion.extract.references import (
            create_reference_edge,
            extract_chunk_references,
            extract_references,
        )

        # CRITICAL: Extract hyperlink references from RAW markdown content
        # The markdown parser converts [Title](file.md) to HTML, then BeautifulSoup
        # extracts only the display text, losing the link URL entirely.
        # We must extract markdown hyperlinks BEFORE HTML conversion.
        raw_content_refs = []
        if format == "markdown" and content:
            # Use document ID as synthetic chunk ID for document-level references
            # This associates hyperlink references with the document rather than
            # a specific section (since we can't map character positions to sections)
            doc_chunk_id = document["id"]

            # Extract references from raw markdown content
            raw_refs = extract_references(content, doc_chunk_id)

            # Convert to edge format
            for ref in raw_refs:
                # Only include hyperlink references from raw content
                # (other patterns like see_also/related work fine on plain text)
                if ref.reference_type == "hyperlink":
                    edge = create_reference_edge(
                        source_chunk_id=doc_chunk_id,
                        target_doc_id=None,  # Will be resolved in Neo4j transaction
                        target_hint=ref.target_hint,
                        reference_type=ref.reference_type,
                        reference_text=ref.reference_text,
                        confidence=ref.confidence,
                    )
                    raw_content_refs.append(edge)

            logger.debug(
                "hyperlinks_extracted_from_raw_markdown",
                hyperlink_count=len(raw_content_refs),
                document_id=document["id"],
            )

        # Respect feature flag
        references_cfg = getattr(config, "references", None)
        if references_cfg and getattr(references_cfg, "enabled", False):
            # Extract reference patterns from text (see_also, related, refer_to)
            # Works on plain text without needing markdown link syntax
            reference_edges, ref_resolved, ref_unresolved = extract_chunk_references(
                sections,
                known_doc_titles=None,  # Target resolution happens in Neo4j transaction
            )
        else:
            reference_edges, ref_resolved, ref_unresolved = [], 0, 0

        # Merge hyperlinks from raw content with other references from sections
        # Deduplicate by target_hint to avoid double-counting
        existing_hints = {e.get("target_hint", "").lower() for e in reference_edges}
        for edge in raw_content_refs:
            if edge.get("target_hint", "").lower() not in existing_hints:
                reference_edges.append(edge)
                existing_hints.add(edge.get("target_hint", "").lower())

        logger.debug(
            "references_extracted_from_sections",
            total_references=len(reference_edges),
            hyperlinks_from_raw=len(raw_content_refs),
            local_resolved=ref_resolved,
            pending_neo4j_resolution=ref_unresolved,
        )

        # Assemble chunks
        assembler = get_chunk_assembler(
            getattr(config.ingestion, "chunk_assembly", None)
        )
        sections = assembler.assemble(document["id"], sections)

        # Set document tokens
        doc_total_tokens = sum(int(s.get("token_count", 0)) for s in sections)
        document["total_tokens"] = doc_total_tokens
        document.setdefault("doc_id", document.get("id"))

        for section in sections:
            section.setdefault("document_id", document["id"])
            section.setdefault("doc_id", document.get("doc_id"))
            section["document_total_tokens"] = doc_total_tokens

        # Phase 2 GLiNER: Enrich chunks with named entities (gated by config)
        # This adds entity_metadata, _embedding_text, and _mentions to each chunk
        # Phase 3.5: GLiNER entities are now written to Neo4j (Entity nodes + MENTIONS)
        if getattr(config, "ner", None) and getattr(config.ner, "enabled", False):
            try:
                from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

                enrich_chunks_with_entities(sections)
            except Exception as e:
                # Non-blocking: GLiNER failure should not abort ingestion
                logger.warning(
                    "gliner_enrichment_failed_non_blocking",
                    error=str(e),
                    document_id=document.get("id"),
                    section_count=len(sections),
                )

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

        Phase 1: Token-budgeted batching (EMBED_BATCH_MAX_TOKENS=7000)
        Phase 2: Per-batch error isolation with None placeholders
        Phase 3: Validation layer (dimension, schema, metadata)

        Returns:
            Dict with:
            - sections: Dict[section_id -> {content, title, sparse?, colbert?}]
            - entities: Dict[entity_id -> [...]] (reserved for future)
            - stats: Dict with sparse coverage and batch metrics
        """
        # Build entity_id → name lookup for entity-sparse generation
        # Entities is a Dict[entity_id → entity_dict] with 'id' and 'name' fields
        entity_id_to_name: Dict[str, str] = {}
        entity_name_fallback_count = 0
        if entities:
            for eid, entity in entities.items():
                if not isinstance(entity, dict):
                    continue
                # Primary: use 'name' field (expected for all entity types)
                ename = entity.get("name", "")
                if not ename:
                    # Safety-net fallback with explicit logging:
                    # Try 'instruction' (Steps), 'description', or 'content'
                    fallback_field = None
                    if entity.get("instruction"):
                        ename = entity["instruction"][:80]
                        fallback_field = "instruction"
                    elif entity.get("description"):
                        ename = entity["description"][:80]
                        fallback_field = "description"
                    elif entity.get("content"):
                        ename = entity["content"][:80]
                        fallback_field = "content"
                    if fallback_field:
                        entity_name_fallback_count += 1
                        logger.warning(
                            "entity_missing_name_field_using_fallback",
                            entity_id=eid[:16] if eid else "unknown",
                            entity_label=entity.get("label", "unknown"),
                            fallback_field=fallback_field,
                            fallback_value_preview=ename[:40] if ename else "",
                        )
                if eid and ename:
                    entity_id_to_name[eid] = ename

        if entity_name_fallback_count > 0:
            logger.warning(
                "entity_name_fallback_summary",
                total_entities=len(entities) if entities else 0,
                fallback_count=entity_name_fallback_count,
                message="Entities missing 'name' - fix extraction code",
            )

        embeddings = {
            "sections": {},  # section_id -> {content, title, sparse, colbert}
            "entities": {},  # entity_id -> [...] (reserved)
            "stats": {
                # Sparse coverage tracking (Graph Channel Rehabilitation)
                "sparse_eligible": 0,  # Non-stub chunks eligible for sparse
                "sparse_success": 0,  # Chunks that got sparse vectors
                "sparse_failures": 0,  # Batches where embed_sparse failed
                "sparse_content_missing": 0,  # Non-stub w/o sparse (SLO metric)
                # Batch metrics
                "batch_count": 0,
                "total_tokens_processed": 0,
                # Truncation tracking (SLO monitoring)
                "content_truncated": 0,  # Sections exceeding max_embed_tokens
                "tokens_dropped": 0,  # Tokens lost to truncation
            },
        }
        stats = embeddings["stats"]

        # Ensure embedder is initialized; fail fast if unavailable
        if not getattr(builder, "embedder", None):
            if hasattr(builder, "ensure_embedder"):
                builder.ensure_embedder()
            if not getattr(builder, "embedder", None):
                raise RuntimeError(
                    "Embedding provider is not initialized; aborting ingestion."
                )

        embedding_plan = getattr(builder, "embedding_plan", None)
        dense_embedder = builder.embedder
        sparse_embedder = dense_embedder
        colbert_embedder = dense_embedder
        if embedding_plan:
            if (
                embedding_plan.sparse
                and embedding_plan.sparse.profile_name
                != embedding_plan.dense.profile_name
            ):
                sparse_embedder = ProviderFactory.create_embedding_provider_for_role(
                    embedding_plan.sparse
                )
            if embedding_plan.colbert:
                if (
                    embedding_plan.sparse
                    and embedding_plan.colbert.profile_name
                    == embedding_plan.sparse.profile_name
                ):
                    colbert_embedder = sparse_embedder
                elif (
                    embedding_plan.colbert.profile_name
                    != embedding_plan.dense.profile_name
                ):
                    colbert_embedder = (
                        ProviderFactory.create_embedding_provider_for_role(
                            embedding_plan.colbert
                        )
                    )

        try:
            tokenizer = TokenizerService()
        except Exception as e:
            logger.warning("tokenizer_init_failed", error=str(e))
            tokenizer = None
        # Per-input token limit for embedding requests.
        # Prefer BGE-M3 safe limit (e.g., 8000) if set; fall back to legacy EMBEDDING_MAX_TOKENS.
        max_embed_tokens = int(
            os.getenv("BGE_M3_SAFE_INPUT_TOKENS")
            or os.getenv("EMBEDDING_MAX_TOKENS")
            or "8000"
        )

        # Get expected embedding dimensions from builder config
        embedding_dims = getattr(builder, "embedding_dims", None)
        if embedding_dims is None and hasattr(builder, "embedding_settings"):
            embedding_dims = getattr(builder.embedding_settings, "dimensions", 1024)
        if embedding_dims is None:
            embedding_dims = 1024  # Default fallback

        # Check embedding capabilities
        qdrant_cfg = getattr(self.config.search.vector, "qdrant", None)
        enable_sparse = bool(getattr(qdrant_cfg, "enable_sparse", False))
        enable_colbert = bool(getattr(qdrant_cfg, "enable_colbert", False))
        if embedding_plan:
            supports_sparse = bool(
                enable_sparse
                and embedding_plan.sparse
                and embedding_plan.sparse.profile.capabilities.supports_sparse
            )
            supports_colbert = bool(
                enable_colbert
                and embedding_plan.colbert
                and embedding_plan.colbert.profile.capabilities.supports_colbert
            )
        else:
            supports_sparse = getattr(
                getattr(builder, "embedding_settings", None),
                "capabilities",
                None,
            )
            supports_sparse = (
                getattr(supports_sparse, "supports_sparse", False)
                if supports_sparse
                else False
            )
            supports_colbert = getattr(
                getattr(builder, "embedding_settings", None),
                "capabilities",
                None,
            )
            supports_colbert = (
                getattr(supports_colbert, "supports_colbert", False)
                if supports_colbert
                else False
            )

        # Check strict mode config for sparse embeddings
        sparse_strict_mode = getattr(
            getattr(self.config.search.vector, "qdrant", None),
            "sparse_strict_mode",
            False,
        )

        # Check if doc_title-sparse vectors should be generated
        # Allows disabling doc_title sparse independently of text-sparse
        # Default: True for backward compat
        enable_doc_title_sparse = getattr(
            getattr(self.config.search.vector, "qdrant", None),
            "enable_doc_title_sparse",
            True,
        )

        # Check if title-sparse vectors should be generated
        # (section heading lexical matching). Default: True
        enable_title_sparse = getattr(
            getattr(self.config.search.vector, "qdrant", None),
            "enable_title_sparse",
            True,
        )

        # Check if entity-sparse vectors should be generated
        # (entity name lexical matching). Default: True
        enable_entity_sparse = getattr(
            getattr(self.config.search.vector, "qdrant", None),
            "enable_entity_sparse",
            True,
        )

        # Prepare batch texts for efficient embedding
        section_data = []
        content_texts = []
        title_texts = []
        doc_title_texts = []

        # Get document title for doc_title vector (same for all sections in this doc)
        doc_title = document.get("title", "")
        if not doc_title:
            # Fallback: derive from doc_id if title is empty
            doc_id = document.get("doc_id", document.get("id", ""))
            if "/" in doc_id:
                doc_title = doc_id.split("/")[-1].replace("-", " ").replace("_", " ")
            else:
                doc_title = doc_id

        for section in sections:
            section_id = section.get("id")
            if not section_id:
                continue

            # Prefer transient GLiNER-enriched text if available (Phase 2 GLiNER)
            # _embedding_text contains entity context: "{title}\n\n{text}\n\n[Context: entities]"
            # Falls back to standard builder method when GLiNER is disabled or no entities
            content_text = section.get(
                "_embedding_text"
            ) or builder._build_section_text_for_embedding(section)

            # Skip sections with empty content to prevent HTTP 400 from embedding API
            # Handles microdoc stubs from GreedyCombinerV2 with stub["text"] = ""
            if not content_text or not content_text.strip():
                logger.debug(
                    "skipping_empty_section_embedding",
                    section_id=section_id,
                    has_title=bool(section.get("title")),
                    has_heading=bool(section.get("heading")),
                    is_microdoc_stub=section.get("is_microdoc_stub", False),
                    doc_is_microdoc=section.get("doc_is_microdoc", False),
                )
                continue

            content_tokens = (
                tokenizer.count_tokens(content_text)
                if tokenizer
                else len((content_text or "").split())
            )

            # Enforce per-input embedding token limit.
            # IMPORTANT: Do NOT truncate chunk text here. Chunking happens upstream
            # (e.g., semantic chunking with guard-splitting). If an oversize chunk
            # reaches this stage it indicates a chunking regression, and truncation
            # would silently drop content and desynchronize graph payload vs vectors.
            if tokenizer and content_tokens > max_embed_tokens:
                original_tokens = content_tokens
                original_source = (
                    "gliner_enriched" if section.get("_embedding_text") else "base"
                )

                # Local counters (non-breaking additions) for debugging/telemetry.
                stats.setdefault("embedding_input_adjusted", 0)
                stats.setdefault("oversize_sections_blocked", 0)

                def _strip_embedding_context(text: str) -> str:
                    """Strip an appended enrichment trailer like "\n\n[Context: ...]".

                    GLiNER enrichment uses a transient embedding text format:
                      "{title}\n\n{text}\n\n[Context: entities]"

                    When this enrichment pushes the embedding input over the model
                    limit, we prefer dropping ONLY the additive context before
                    blocking ingestion.
                    """
                    for marker in ("\n\n[Context:", "\n[Context:"):
                        pos = text.find(marker)
                        if pos != -1:
                            return text[:pos].rstrip()
                    return text

                # First attempt: drop appended context (if present) to fit.
                adjusted_text = (
                    _strip_embedding_context(content_text)
                    if section.get("_embedding_text")
                    else content_text
                )
                if adjusted_text != content_text:
                    adjusted_tokens = tokenizer.count_tokens(adjusted_text)
                    if adjusted_tokens <= max_embed_tokens:
                        logger.warning(
                            "embedding_input_adjusted_drop_context",
                            section_id=section_id,
                            source=original_source,
                            original_tokens=original_tokens,
                            adjusted_tokens=adjusted_tokens,
                            max_embed_tokens=max_embed_tokens,
                        )
                        stats["embedding_input_adjusted"] += 1
                        content_text = adjusted_text
                        content_tokens = adjusted_tokens

                # Second attempt: if we were using enriched text, fall back to base
                # embedding text without enrichment (still no truncation).
                if content_tokens > max_embed_tokens and section.get("_embedding_text"):
                    try:
                        base_section = dict(section)
                        base_section.pop("_embedding_text", None)
                        base_text = builder._build_section_text_for_embedding(
                            base_section
                        )
                    except Exception:
                        base_text = None

                    if base_text and base_text != content_text:
                        base_tokens = tokenizer.count_tokens(base_text)
                        if base_tokens <= max_embed_tokens:
                            logger.warning(
                                "embedding_input_adjusted_fallback_base_text",
                                section_id=section_id,
                                original_tokens=original_tokens,
                                adjusted_tokens=base_tokens,
                                max_embed_tokens=max_embed_tokens,
                            )
                            stats["embedding_input_adjusted"] += 1
                            content_text = base_text
                            content_tokens = base_tokens

                # Final: block ingestion rather than silently truncating/dropping.
                if content_tokens > max_embed_tokens:
                    stats["oversize_sections_blocked"] += 1
                    logger.error(
                        "embedding_input_oversize_blocked",
                        section_id=section_id,
                        heading=(section.get("title") or section.get("heading") or "")[
                            :80
                        ],
                        content_tokens=content_tokens,
                        max_embed_tokens=max_embed_tokens,
                        source=original_source,
                    )
                    raise ValueError(
                        f"Section {section_id} embedding input is {content_tokens} tokens "
                        f"(limit {max_embed_tokens}). Upstream chunking must guarantee "
                        "sections fit the embedding model context window; refusing to "
                        "truncate to avoid silent data loss."
                    )

            title_text = builder._build_title_text_for_embedding(section)

            section_data.append(
                {
                    "id": section_id,
                    "section": section,
                    "content_text": content_text,
                    "title_text": title_text,
                    "doc_title_text": doc_title,
                    "token_count": content_tokens,
                }
            )
            content_texts.append(content_text)
            title_texts.append(title_text)
            doc_title_texts.append(doc_title)

        if not section_data:
            return embeddings

        # =====================================================================
        # PHASE 1: Token-Budgeted Batching
        # Port from GraphBuilder._process_embeddings (build_graph.py:1557-1580)
        # Prevents HTTP 400 cascade by proactively limiting batch token size
        # =====================================================================
        batch_budget = int(os.getenv("EMBED_BATCH_MAX_TOKENS", "7000") or "7000")
        if batch_budget <= 0:
            batch_budget = 7000

        # Create batches based on cumulative token count
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0

        for idx, data in enumerate(section_data):
            tokens = data.get("token_count") or len(data["content_text"].split())

            if current_batch and current_tokens + tokens > batch_budget:
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = tokens
            else:
                current_batch.append(idx)
                current_tokens += tokens

        if current_batch:
            batches.append(current_batch)

        stats["batch_count"] = len(batches)
        stats["total_tokens_processed"] = sum(
            d.get("token_count", 0) for d in section_data
        )

        logger.info(
            "embedding_batches_prepared",
            total_sections=len(section_data),
            batch_count=len(batches),
            batch_budget=batch_budget,
            total_tokens=stats["total_tokens_processed"],
        )

        # Initialize embedding result lists
        content_embeddings: List[List[float]] = []
        title_embeddings: List[List[float]] = []
        doc_title_embeddings: List[List[float]] = []

        # Guard capability flags with runtime method detection to prevent
        # confusing "count mismatch" errors when config advertises capability
        # but embedder lacks the method (consensus-identified bug fix)
        has_sparse_method = hasattr(sparse_embedder, "embed_sparse")
        has_colbert_method = hasattr(colbert_embedder, "embed_colbert")

        if supports_sparse and not has_sparse_method:
            logger.warning(
                "sparse_capability_mismatch",
                reason="supports_sparse=True but embedder lacks embed_sparse method",
                embedder_type=type(builder.embedder).__name__,
            )
            supports_sparse = False

        if supports_colbert and not has_colbert_method:
            logger.warning(
                "colbert_capability_mismatch",
                reason="supports_colbert=True but embedder lacks embed_colbert method",
                embedder_type=type(builder.embedder).__name__,
            )
            supports_colbert = False

        sparse_embeddings: Optional[List[Optional[dict]]] = (
            [] if supports_sparse else None
        )
        # doc_title-sparse: BM25-style lexical matching for document titles
        # Only generate if BOTH: embedder supports sparse AND config flag is enabled
        doc_title_sparse_embeddings: Optional[List[Optional[dict]]] = (
            [] if (supports_sparse and enable_doc_title_sparse) else None
        )
        # title-sparse: BM25-style lexical matching for section headings
        # Enables exact term matching for heading-based queries
        title_sparse_embeddings: Optional[List[Optional[dict]]] = (
            [] if (supports_sparse and enable_title_sparse) else None
        )
        # entity-sparse: BM25-style lexical matching for entity names
        # Enables exact term matching for entity-based queries (e.g., "Nutanix", "NFS")
        entity_sparse_embeddings: Optional[List[Optional[dict]]] = (
            [] if (supports_sparse and enable_entity_sparse) else None
        )
        colbert_embeddings: Optional[List[Optional[List[List[float]]]]] = (
            [] if supports_colbert else None
        )

        use_contextual = bool(
            builder.embedding_plan
            and builder.embedding_plan.dense.profile.supports_contextualized_chunks
        )
        if use_contextual and not hasattr(
            dense_embedder, "embed_contextualized_documents"
        ):
            raise RuntimeError(
                "Dense profile requires contextualized chunks but embedder "
                "does not implement embed_contextualized_documents."
            )

        if use_contextual:
            contextual = dense_embedder.embed_contextualized_documents(
                [content_texts],
                input_type=builder.embedding_plan.dense.profile.document_task,
            )
            if not contextual or len(contextual) != 1:
                raise RuntimeError(
                    "Contextual embedding response missing document payload."
                )
            content_embeddings = contextual[0]
            if len(content_embeddings) != len(content_texts):
                raise RuntimeError(
                    "Contextual embedding count mismatch: "
                    f"expected {len(content_texts)}, got {len(content_embeddings)}."
                )

        # =====================================================================
        # Sparse sub-batching: SPLADE OOMs on >10 sequences per call.
        # Sub-batch to cap GPU memory usage per inference call.
        # =====================================================================
        sparse_max_items = int(os.getenv("SPARSE_BATCH_MAX_ITEMS", "10"))

        def _embed_sparse_safe(texts: List[str]) -> List:
            """Sub-batch sparse embedding to avoid SPLADE OOM on large batches."""
            if len(texts) <= sparse_max_items:
                return sparse_embedder.embed_sparse(texts)
            results = []
            for i in range(0, len(texts), sparse_max_items):
                sub = texts[i : i + sparse_max_items]
                results.extend(sparse_embedder.embed_sparse(sub))
            return results

        # =====================================================================
        # PHASE 1+2: Process batches with error isolation
        # =====================================================================
        for batch_idx, batch_indices in enumerate(batches):
            batch_content = [content_texts[i] for i in batch_indices]
            batch_title = [title_texts[i] for i in batch_indices]
            batch_doc_title = [doc_title_texts[i] for i in batch_indices]
            batch_tokens = sum(
                section_data[i].get("token_count", 0) for i in batch_indices
            )

            logger.debug(
                "processing_embedding_batch",
                batch_index=batch_idx,
                batch_size=len(batch_indices),
                batch_tokens=batch_tokens,
            )

            # Dense embeddings - required, fail-all on error
            try:
                if not use_contextual:
                    content_embeddings.extend(
                        dense_embedder.embed_documents(batch_content)
                    )
                title_embeddings.extend(dense_embedder.embed_documents(batch_title))
                doc_title_embeddings.extend(
                    dense_embedder.embed_documents(batch_doc_title)
                )
            except Exception as e:
                logger.error(
                    "dense_embedding_batch_failed",
                    error=str(e),
                    batch_index=batch_idx,
                    batch_size=len(batch_content),
                )
                raise RuntimeError(
                    f"Dense embedding failed for batch {batch_idx}: {e}"
                ) from e

            # =====================================================================
            # PHASE 2: Per-Batch Error Isolation for Sparse Embeddings
            # Port from GraphBuilder (build_graph.py:1596-1637)
            # On failure, insert None placeholders to maintain index alignment
            # =====================================================================
            if sparse_embeddings is not None and hasattr(
                sparse_embedder, "embed_sparse"
            ):
                try:
                    sparse_embeddings.extend(_embed_sparse_safe(batch_content))
                except Exception as exc:
                    stats["sparse_failures"] += 1

                    if sparse_strict_mode:
                        logger.error(
                            "sparse_embedding_batch_failed_strict_mode",
                            error=str(exc),
                            batch_index=batch_idx,
                            batch_size=len(batch_content),
                        )
                        raise RuntimeError(
                            f"Sparse embedding failed in strict mode: {exc}"
                        ) from exc
                    else:
                        logger.warning(
                            "sparse_embedding_batch_failed_inserting_placeholders",
                            error=str(exc),
                            batch_index=batch_idx,
                            batch_size=len(batch_content),
                        )
                        # Insert None placeholders to maintain index alignment
                        # Only this batch's chunks lose sparse; others continue normally
                        sparse_embeddings.extend([None] * len(batch_content))

            # doc_title-sparse: BM25-style lexical matching for document titles
            if doc_title_sparse_embeddings is not None and hasattr(
                sparse_embedder, "embed_sparse"
            ):
                try:
                    doc_title_sparse_embeddings.extend(
                        _embed_sparse_safe(batch_doc_title)
                    )
                except Exception as exc:
                    logger.warning(
                        "doc_title_sparse_embedding_batch_failed_inserting_placeholders",
                        error=str(exc),
                        batch_index=batch_idx,
                        batch_size=len(batch_doc_title),
                    )
                    # Insert None placeholders to maintain index alignment
                    doc_title_sparse_embeddings.extend([None] * len(batch_doc_title))

            # title-sparse: BM25-style lexical matching for section headings
            if title_sparse_embeddings is not None and hasattr(
                sparse_embedder, "embed_sparse"
            ):
                try:
                    title_sparse_embeddings.extend(_embed_sparse_safe(batch_title))
                except Exception as exc:
                    logger.warning(
                        "title_sparse_embedding_batch_failed_inserting_placeholders",
                        error=str(exc),
                        batch_index=batch_idx,
                        batch_size=len(batch_title),
                    )
                    # Insert None placeholders to maintain index alignment
                    title_sparse_embeddings.extend([None] * len(batch_title))

            # entity-sparse: BM25-style lexical matching for entity names
            # Build entity text from section mentions for each chunk in this batch
            if entity_sparse_embeddings is not None and hasattr(
                sparse_embedder, "embed_sparse"
            ):
                try:
                    # Build entity texts for this batch
                    batch_entity_texts = []
                    for i in batch_indices:
                        section = section_data[i]["section"]
                        # Get mentions attached to section (from ingest_document_atomic)
                        section_mentions = section.get("_mentions", [])
                        if section_mentions:
                            # Sort by confidence (desc), cap at 8 to prevent noisy tails
                            # from diluting the SPLADE encoding of entity-sparse vectors
                            sorted_mentions = sorted(
                                section_mentions,
                                key=lambda m: m.get("confidence", 0.0),
                                reverse=True,
                            )[:8]
                            # Collect entity names from top mentions
                            # GLiNER mentions have 'name' directly; structural use lookup
                            entity_names = []
                            for m in sorted_mentions:
                                # First: check for direct 'name' field (GLiNER mentions)
                                if m.get("name"):
                                    entity_names.append(m["name"])
                                # Fallback: lookup by entity_id (structural entities)
                                elif m.get("entity_id") in entity_id_to_name:
                                    entity_names.append(
                                        entity_id_to_name[m["entity_id"]]
                                    )
                            entity_text = " ".join(entity_names) if entity_names else ""
                        else:
                            entity_text = ""
                        batch_entity_texts.append(entity_text)

                    # Only embed non-empty entity texts; use None for empty
                    if any(t.strip() for t in batch_entity_texts):
                        # Embed non-empty texts, None for empty ones
                        embeddings_result = []
                        non_empty_texts = []
                        non_empty_indices = []
                        for idx, text in enumerate(batch_entity_texts):
                            if text.strip():
                                non_empty_texts.append(text)
                                non_empty_indices.append(idx)

                        if non_empty_texts:
                            sparse_results = _embed_sparse_safe(non_empty_texts)
                            result_iter = iter(sparse_results)
                            for idx in range(len(batch_entity_texts)):
                                if idx in non_empty_indices:
                                    embeddings_result.append(next(result_iter))
                                else:
                                    embeddings_result.append(None)
                            entity_sparse_embeddings.extend(embeddings_result)
                        else:
                            entity_sparse_embeddings.extend(
                                [None] * len(batch_entity_texts)
                            )
                    else:
                        # All empty - use None placeholders
                        entity_sparse_embeddings.extend(
                            [None] * len(batch_entity_texts)
                        )
                except Exception as exc:
                    logger.warning(
                        "entity_sparse_embedding_batch_failed_inserting_placeholders",
                        error=str(exc),
                        batch_index=batch_idx,
                        batch_size=len(batch_indices),
                    )
                    # Insert None placeholders to maintain index alignment
                    entity_sparse_embeddings.extend([None] * len(batch_indices))

            # Phase 2: Per-Batch Error Isolation for ColBERT Embeddings
            if colbert_embeddings is not None and hasattr(
                colbert_embedder, "embed_colbert"
            ):
                try:
                    colbert_embeddings.extend(
                        colbert_embedder.embed_colbert(batch_content)
                    )
                except Exception as exc:
                    logger.warning(
                        "colbert_embedding_batch_failed_inserting_placeholders",
                        error=str(exc),
                        batch_index=batch_idx,
                        batch_size=len(batch_content),
                    )
                    # Insert None placeholders to maintain index alignment
                    colbert_embeddings.extend([None] * len(batch_content))

        # Validate batch output alignment
        expected = len(section_data)
        if len(content_embeddings) != expected:
            raise RuntimeError(
                f"Content embedding mismatch: expected {expected}, "
                f"got {len(content_embeddings)}"
            )
        if len(title_embeddings) != expected:
            raise RuntimeError(
                f"Title embedding mismatch: expected {expected}, "
                f"got {len(title_embeddings)}"
            )
        if len(doc_title_embeddings) != expected:
            raise RuntimeError(
                f"Doc title embedding mismatch: expected {expected}, "
                f"got {len(doc_title_embeddings)}"
            )
        if sparse_embeddings is not None and len(sparse_embeddings) != expected:
            raise RuntimeError(
                f"Sparse embedding mismatch: expected {expected}, "
                f"got {len(sparse_embeddings)}"
            )
        if (
            doc_title_sparse_embeddings is not None
            and len(doc_title_sparse_embeddings) != expected
        ):
            raise RuntimeError(
                f"Doc title sparse mismatch: expected {expected}, "
                f"got {len(doc_title_sparse_embeddings)}"
            )
        if (
            title_sparse_embeddings is not None
            and len(title_sparse_embeddings) != expected
        ):
            raise RuntimeError(
                f"Title sparse mismatch: expected {expected}, "
                f"got {len(title_sparse_embeddings)}"
            )
        if (
            entity_sparse_embeddings is not None
            and len(entity_sparse_embeddings) != expected
        ):
            raise RuntimeError(
                f"Entity sparse mismatch: expected {expected}, "
                f"got {len(entity_sparse_embeddings)}"
            )
        if colbert_embeddings is not None and len(colbert_embeddings) != expected:
            raise RuntimeError(
                f"ColBERT embedding mismatch: expected {expected}, "
                f"got {len(colbert_embeddings)}"
            )

        # =====================================================================
        # PHASE 3: Validation Layer + Assemble embeddings per section
        # Port from GraphBuilder (build_graph.py:1676-1711)
        # =====================================================================
        for idx, data in enumerate(section_data):
            section_id = data["id"]
            section = data["section"]
            embedding = content_embeddings[idx]
            title_embedding = title_embeddings[idx]
            doc_title_embedding = doc_title_embeddings[idx]
            sparse_vector = (
                sparse_embeddings[idx]
                if sparse_embeddings is not None and idx < len(sparse_embeddings)
                else None
            )
            doc_title_sparse_vector = (
                doc_title_sparse_embeddings[idx]
                if doc_title_sparse_embeddings is not None
                and idx < len(doc_title_sparse_embeddings)
                else None
            )
            title_sparse_vector = (
                title_sparse_embeddings[idx]
                if title_sparse_embeddings is not None
                and idx < len(title_sparse_embeddings)
                else None
            )
            entity_sparse_vector = (
                entity_sparse_embeddings[idx]
                if entity_sparse_embeddings is not None
                and idx < len(entity_sparse_embeddings)
                else None
            )
            colbert_vector = (
                colbert_embeddings[idx]
                if colbert_embeddings is not None and idx < len(colbert_embeddings)
                else None
            )

            # -----------------------------------------------------------
            # Validation 1: Dimension check
            # -----------------------------------------------------------
            if len(embedding) != embedding_dims:
                raise ValueError(
                    f"Embedding dimension mismatch for section {section_id}: "
                    f"expected {embedding_dims}-D, got {len(embedding)}-D. "
                    "Ingestion blocked - dimension safety enforced."
                )

            # -----------------------------------------------------------
            # Validation 2: Non-empty embedding check
            # -----------------------------------------------------------
            if not embedding or len(embedding) == 0:
                raise ValueError(
                    f"Section {section_id} missing REQUIRED vector_embedding. "
                    "Ingestion blocked - embeddings are mandatory in hybrid system."
                )

            # -----------------------------------------------------------
            # Validation 3: Chunk schema completeness
            # -----------------------------------------------------------
            if not validate_chunk_schema(section):
                raise ValueError(
                    f"Section {section_id} missing required chunk fields. "
                    "Ingestion blocked - chunk schema validation failed."
                )

            # -----------------------------------------------------------
            # Validation 4: Embedding metadata completeness
            # -----------------------------------------------------------
            if hasattr(builder, "embedding_settings") and builder.embedding_settings:
                test_metadata = canonicalize_embedding_metadata(
                    embedding_model=builder.embedding_settings.version,
                    dimensions=len(embedding),
                    provider=(
                        getattr(builder.embedder, "provider_name", None)
                        if hasattr(builder, "embedder")
                        else None
                    ),
                    task=getattr(builder.embedder, "task", None)
                    or builder.embedding_settings.task,
                    profile=getattr(builder.embedding_settings, "profile", None),
                    timestamp=datetime.utcnow(),
                )

                if not validate_embedding_metadata(
                    test_metadata,
                    expected_dimensions=embedding_dims,
                    expected_provider=builder.embedding_settings.provider,
                    expected_version=builder.embedding_settings.version,
                ):
                    raise ValueError(
                        f"Section {section_id} has invalid embedding metadata. "
                        "Ingestion blocked - metadata validation failed."
                    )

            # -----------------------------------------------------------
            # Graph Channel Rehabilitation: Track sparse coverage
            # -----------------------------------------------------------
            is_stub = section.get("is_microdoc_stub", False)
            if not is_stub and sparse_embeddings is not None:
                stats["sparse_eligible"] += 1
                # Check if sparse_vector is valid (has indices)
                has_valid_sparse = (
                    sparse_vector is not None
                    and isinstance(sparse_vector, dict)
                    and sparse_vector.get("indices")
                )
                if has_valid_sparse:
                    stats["sparse_success"] += 1
                else:
                    # Non-stub content chunk missing sparse - this is the SLO metric
                    stats["sparse_content_missing"] += 1
                    logger.warning(
                        "non_stub_content_chunk_missing_sparse_vector",
                        section_id=section_id,
                        heading=section.get("heading"),
                        token_count=section.get("token_count", 0),
                    )

            # -----------------------------------------------------------
            # Assemble validated embeddings
            # -----------------------------------------------------------
            section_embedding = {
                "content": embedding,
                "title": title_embedding,
                "doc_title": doc_title_embedding,
            }

            # Add sparse if computed (may be None for failed batches)
            if sparse_embeddings is not None:
                section_embedding["sparse"] = sparse_vector

            # Add doc_title sparse if computed (may be None for failed batches)
            if doc_title_sparse_embeddings is not None:
                section_embedding["doc_title_sparse"] = doc_title_sparse_vector

            # Add title sparse if computed (may be None for failed batches)
            if title_sparse_embeddings is not None:
                section_embedding["title_sparse"] = title_sparse_vector

            # Add entity sparse if computed (may be None for failed batches)
            if entity_sparse_embeddings is not None:
                section_embedding["entity_sparse"] = entity_sparse_vector

            # Add ColBERT if computed (may be None for failed batches)
            if colbert_embeddings is not None:
                section_embedding["colbert"] = colbert_vector

            embeddings["sections"][section_id] = section_embedding

        # Log final stats
        sparse_coverage = (
            (stats["sparse_success"] / stats["sparse_eligible"] * 100)
            if stats["sparse_eligible"] > 0
            else 0.0
        )

        logger.info(
            "embeddings_computed",
            section_count=len(section_data),
            batch_count=stats["batch_count"],
            has_sparse=sparse_embeddings is not None,
            has_colbert=colbert_embeddings is not None,
            sparse_eligible=stats["sparse_eligible"],
            sparse_success=stats["sparse_success"],
            sparse_failures=stats["sparse_failures"],
            sparse_content_missing=stats["sparse_content_missing"],
            sparse_coverage_pct=round(sparse_coverage, 2),
            content_truncated=stats["content_truncated"],
            tokens_dropped=stats["tokens_dropped"],
        )

        return embeddings

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
        document_id = document["id"]
        context = SagaContext(saga_id=saga_id, document_id=document_id)

        # Track what we write for compensation
        written_neo4j_chunks: List[str] = []
        written_qdrant_points: List[str] = []
        neo4j_tx = None
        neo4j_session = None

        stats = {
            "sections_upserted": 0,
            "entities_upserted": 0,
            "vectors_upserted": 0,
        }

        try:
            # Sanity checks before starting writes
            qdrant_required = bool(
                self.qdrant_client
                and (
                    self.config.search.vector.primary == "qdrant"
                    or self.config.search.vector.dual_write
                )
            )
            if (
                self.config.search.vector.primary == "qdrant"
                or self.config.search.vector.dual_write
            ) and not self.qdrant_client:
                raise RuntimeError(
                    "Qdrant client is required for primary or dual-write modes."
                )
            if qdrant_required and not embeddings.get("sections"):
                raise RuntimeError(
                    "Embeddings are required for Qdrant write but are missing."
                )

            # Step 1: Start Neo4j transaction (explicit, not auto-commit)
            neo4j_session = self.neo4j_driver.session()
            neo4j_tx = neo4j_session.begin_transaction()

            logger.debug(
                "neo4j_transaction_started",
                saga_id=saga_id,
                document_id=document_id,
            )

            # Step 2: Execute Neo4j writes within the transaction
            # These are the core graph writes
            self.neo4j_writer._neo4j_upsert_document(neo4j_tx, document)
            chunk_count = self.neo4j_writer._neo4j_upsert_sections(
                neo4j_tx, document_id, sections
            )
            stats["sections_upserted"] = chunk_count
            written_neo4j_chunks = [s["id"] for s in sections if "id" in s]

            # Prune structural entities that have no surviving mentions after
            # the quality gate. This prevents orphan Entity nodes from being
            # created in Neo4j.
            if entities and isinstance(entities, dict):
                mentioned_entity_ids = set()
                for section in sections:
                    for m in section.get("_mentions", []):
                        eid = m.get("entity_id")
                        if eid and eid in entities:
                            mentioned_entity_ids.add(eid)

                if mentioned_entity_ids:
                    original_count = len(entities)
                    entities = {
                        eid: edata
                        for eid, edata in entities.items()
                        if eid in mentioned_entity_ids
                    }
                    pruned_count = original_count - len(entities)
                    if pruned_count:
                        logger.info(
                            "structural_entities_pruned_zero_mentions",
                            pruned_count=pruned_count,
                            remaining=len(entities),
                            document_id=document_id,
                        )
                else:
                    # No entity IDs survived mention filtering.
                    logger.info(
                        "structural_entities_pruned_zero_mentions",
                        pruned_count=len(entities),
                        remaining=0,
                        document_id=document_id,
                    )
                    entities = {}

            entity_count = self.neo4j_writer._neo4j_upsert_entities(neo4j_tx, entities)
            stats["entities_upserted"] = entity_count

            # Phase 3.5: Collect ALL mentions from sections for Neo4j
            # Use section["_mentions"] which has the correct chunk ID mapping
            # (prepared["mentions"] has original section IDs that don't match chunk nodes)
            all_mentions = []
            structural_count = 0
            gliner_count = 0
            for section in sections:
                section_id = section.get("id")
                if not section_id:
                    continue
                for m in section.get("_mentions", []):
                    entity_id = m.get("entity_id")
                    if not entity_id:
                        continue
                    # Create mention with current chunk ID
                    mention_dict = {
                        "section_id": section_id,
                        "entity_id": entity_id,
                        "name": m.get("name", ""),
                        "type": m.get("type", ""),
                        "confidence": m.get("confidence", 0.5),
                        "source": m.get("source", "structural"),
                    }
                    all_mentions.append(mention_dict)
                    if m.get("source") == "gliner":
                        gliner_count += 1
                    else:
                        structural_count += 1

            logger.debug(
                "mentions_collected_for_neo4j",
                structural_count=structural_count,
                gliner_count=gliner_count,
                total_count=len(all_mentions),
            )

            self.neo4j_writer._neo4j_create_mentions(neo4j_tx, all_mentions)

            # Phase 3: Create cross-document REFERENCES edges (Chunk → Document)
            references_count = self.neo4j_writer._neo4j_create_references(
                neo4j_tx, references
            )
            stats["references_created"] = references_count

            # Step 2b: Store embedding metadata on Chunk nodes
            # This ensures cross-store consistency between Neo4j and Qdrant
            embedding_meta_count = self.neo4j_writer._neo4j_upsert_embedding_metadata(
                neo4j_tx, sections, embeddings, builder
            )
            stats["embedding_metadata_upserted"] = embedding_meta_count

            # LGTM Phase 4: Verbose log event 5 - neo4j_write_complete
            logger.info(
                "neo4j_write_complete",
                doc_id=document_id,
                saga_id=saga_id,
                nodes_created=chunk_count + entity_count + 1,  # +1 for document
                relationships_created=references_count + len(mentions),
                node_types={
                    "Document": 1,
                    "Section": chunk_count,
                    "Entity": entity_count,
                },
                relationship_types={
                    "HAS_CHUNK": chunk_count,  # P0: HAS_SECTION deprecated
                    "MENTIONS": len(mentions),
                    "REFERENCES": references_count,
                },
                embedding_metadata=embedding_meta_count,
            )

            # Step 2c: Build structural edges (ATOMIC - inside transaction)
            # This ensures NEXT_CHUNK, PARENT_HEADING, etc. are committed
            # atomically with chunks. If this fails, transaction rolls back.
            from src.ingestion.structural_edges import build_structural_edges_in_tx

            structural_result = build_structural_edges_in_tx(
                neo4j_tx, document_id, skip_has_chunk=True
            )
            stats["structural_edges"] = structural_result

            logger.info(
                "structural_edges_built",
                doc_id=document_id,
                saga_id=saga_id,
                edges=structural_result.get("stats", {}),
                warnings=structural_result.get("warnings", []),
            )

            # Step 3: Execute Qdrant writes
            # If this fails, we can still rollback Neo4j
            qdrant_count = 0
            if self.qdrant_client and embeddings.get("sections"):
                qdrant_count = self.qdrant_writer._qdrant_upsert_vectors(
                    document, sections, embeddings, builder
                )
                stats["vectors_upserted"] = qdrant_count
                written_qdrant_points = [s["id"] for s in sections if "id" in s]

                # LGTM Phase 4: Verbose log event 6 - qdrant_upsert_complete
                collection_name = getattr(builder, "collection_name", None)
                if not collection_name:
                    collection_name = getattr(
                        self.config.search.vector.qdrant, "collection_name", None
                    )
                if not collection_name:
                    collection_name = getattr(
                        self.config.search.vector, "collection", "chunks_multi"
                    )
                logger.info(
                    "qdrant_upsert_complete",
                    doc_id=document_id,
                    saga_id=saga_id,
                    points_upserted=qdrant_count,
                    collection=collection_name,
                    vector_types=[
                        "content",
                        "title",
                        "doc_title",
                        "text-sparse",
                        "title-sparse",
                        "entity-sparse",
                        "late-interaction",
                    ],
                )

            if (
                self.config.search.vector.primary == "qdrant"
                or self.config.search.vector.dual_write
            ) and qdrant_count == 0:
                raise RuntimeError("Qdrant write required but produced zero vectors.")

            # Step 4: COMMIT Neo4j (only after Qdrant succeeds)
            neo4j_tx.commit()
            context.neo4j_chunk_ids = written_neo4j_chunks

            logger.info(
                "atomic_saga_committed",
                saga_id=saga_id,
                document_id=document_id,
                stats=stats,
            )

            # Step 5: Incremental cross-document linking (Phase 3.5)
            # This runs AFTER commit to ensure the document exists in Neo4j/Qdrant
            # Errors here should NOT fail the ingestion - they are logged only
            cross_doc_stats = self._create_cross_doc_links(
                document_id=document_id,
                document=document,
                sections=sections,
                embeddings=embeddings,
            )
            if cross_doc_stats:
                stats["cross_doc_linking"] = cross_doc_stats

            return {
                "success": True,
                "stats": stats,
                "neo4j_committed": True,
                "qdrant_committed": qdrant_count > 0,
            }

        except Exception as e:
            logger.error(
                "atomic_saga_failed",
                saga_id=saga_id,
                document_id=document_id,
                error=str(e),
            )

            # Track compensation state for both stores independently
            neo4j_rolled_back = False
            qdrant_cleaned_up = False

            # Rollback Neo4j if transaction is still open
            if neo4j_tx and not neo4j_tx.closed():
                try:
                    neo4j_tx.rollback()
                    logger.info(
                        "neo4j_transaction_rolled_back",
                        saga_id=saga_id,
                        document_id=document_id,
                    )
                    neo4j_rolled_back = True
                except Exception as rollback_err:
                    logger.error(
                        "neo4j_rollback_failed",
                        saga_id=saga_id,
                        error=str(rollback_err),
                    )

            # ALWAYS clean up Qdrant if writes succeeded, regardless of Neo4j rollback
            # This prevents orphan vectors when Neo4j rolls back but Qdrant persisted
            if written_qdrant_points:
                try:
                    self.qdrant_writer._compensate_qdrant(
                        written_qdrant_points, builder
                    )
                    qdrant_cleaned_up = True
                    logger.info(
                        "qdrant_compensation_completed",
                        saga_id=saga_id,
                        points_cleaned=len(written_qdrant_points),
                    )
                except Exception as qdrant_err:
                    logger.error(
                        "qdrant_compensation_failed",
                        saga_id=saga_id,
                        error=str(qdrant_err),
                    )

            compensated = neo4j_rolled_back or qdrant_cleaned_up

            return {
                "success": False,
                "stats": stats,
                "error": str(e),
                "neo4j_committed": False,
                "qdrant_committed": len(written_qdrant_points) > 0,
                "compensated": compensated,
            }

        finally:
            # Clean up Neo4j session
            if neo4j_session:
                try:
                    neo4j_session.close()
                except Exception as session_err:
                    logger.warning(
                        "neo4j_session_close_failed",
                        saga_id=saga_id,
                        error=str(session_err),
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

"""
Atomic Ingestion Coordinator for Neo4j + Qdrant Synchronization.

This module provides a transactional wrapper around the GraphBuilder that ensures
atomic commits to both Neo4j and Qdrant, with rollback on failure.

Key Pattern: Deferred Commit
1. Prepare Phase: Compute all embeddings and validate data BEFORE any writes
2. Write Phase: Start Neo4j transaction, write to Neo4j, write to Qdrant
3. Commit Phase: Only commit Neo4j AFTER Qdrant succeeds
4. Compensate Phase: If Qdrant fails, rollback Neo4j; if Neo4j committed, delete from Qdrant

This ensures that chunk IDs in Neo4j always have corresponding vectors in Qdrant.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import structlog

T = TypeVar("T")

# Phase 7F: Lazy exception loader to handle optional qdrant-client dependency
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


from src.ingestion.saga import (  # noqa: E402
    IngestionValidator,
    SagaContext,
    ValidationResult,
)
from src.providers.tokenizer_service import TokenizerService  # noqa: E402
from src.shared.chunk_utils import validate_chunk_schema  # noqa: E402
from src.shared.embedding_fields import (  # noqa: E402
    canonicalize_embedding_metadata,
    ensure_no_embedding_model_in_payload,
    validate_embedding_metadata,
)
from src.shared.observability.metrics import (  # noqa: E402
    entity_relationships_missing_total,
    entity_relationships_total,
)

logger = structlog.get_logger(__name__)

# Allowlist of valid Entity→Entity relationship types for Cypher injection defense
# Phase 1.2: These are the only relationship types allowed to be interpolated into Cypher
# Phase 2 Cleanup: Removed DEPENDS_ON and REQUIRES (never materialized by ingestion)
ALLOWED_ENTITY_RELATIONSHIP_TYPES = frozenset(
    {
        "CONTAINS_STEP",  # Procedure→Step ordering
        "REFERENCES",  # Cross-document references (Phase 3)
        "CONFIGURES",  # Config→Component
        "RESOLVES",  # Error→Procedure
    }
)


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
                         If None (default), reads from VALIDATION_STRICT_MODE env var.
                         Explicit True/False overrides config for backward compatibility.
        """
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.config = config
        self.validate_before_commit = validate_before_commit

        # Phase 5.2: Unified strict mode config
        # Read from Settings if not explicitly passed, with backward compat override
        if strict_mode is None:
            from src.shared.config import get_settings

            settings = get_settings()
            self.strict_mode = settings.validation_strict_mode
        else:
            self.strict_mode = strict_mode

        # Phase 5: Entity vector support (default ON for GraphBuilder schema parity)
        # Entity vector is a copy of content embedding under "entity" name slot.
        # Disable via QDRANT_INCLUDE_ENTITY_VECTOR=false to save ~33% vector storage.
        self.include_entity_vector = (
            os.getenv("QDRANT_INCLUDE_ENTITY_VECTOR", "true").lower() == "true"
        )

        self.validator = IngestionValidator(neo4j_driver, qdrant_client, config)

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

        logger.info(
            "atomic_ingestion_started",
            saga_id=saga_id,
            source_uri=source_uri,
            format=format,
        )

        try:
            # Phase 1: Parse and prepare all data
            prepared = self._prepare_ingestion(
                source_uri,
                content,
                format,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
            )

            document = prepared["document"]
            sections = prepared["sections"]
            entities = prepared["entities"]
            mentions = prepared["mentions"]
            references = prepared.get("references", [])  # Phase 3: Cross-doc refs
            document_id = document["id"]

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
            embeddings = self._compute_embeddings(
                document, sections, entities, prepared["builder"]
            )

            # Phase 3b: Recompute document token aggregates after truncation
            # Truncation in _compute_embeddings may reduce section token counts,
            # so we must update document-level totals to maintain the invariant:
            # document["total_tokens"] == sum(section["token_count"])
            truncated_sections = [s for s in sections if s.get("was_truncated")]
            if truncated_sections:
                new_total = sum(int(s.get("token_count", 0)) for s in sections)
                old_total = document.get("total_tokens", 0)
                tokens_reduced = old_total - new_total

                document["total_tokens"] = new_total
                document["original_total_tokens"] = old_total  # Preserve for auditing

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
            saga_result = self._execute_atomic_saga(
                saga_id=saga_id,
                document=document,
                sections=sections,
                entities=entities,
                mentions=mentions,
                references=references,  # Phase 3: Cross-document REFERENCES
                embeddings=embeddings,
                builder=prepared["builder"],
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if saga_result["success"]:
                logger.info(
                    "atomic_ingestion_completed",
                    saga_id=saga_id,
                    document_id=document_id,
                    duration_ms=duration_ms,
                )

                return AtomicIngestionResult(
                    success=True,
                    document_id=document_id,
                    saga_id=saga_id,
                    stats=saga_result.get("stats", {}),
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

            return AtomicIngestionResult(
                success=False,
                document_id="unknown",
                saga_id=saga_id,
                error=str(e),
                duration_ms=duration_ms,
            )

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
        import re
        from pathlib import Path

        from src.ingestion.build_graph import GraphBuilder
        from src.ingestion.chunk_assembler import get_chunk_assembler
        from src.ingestion.extract import extract_entities
        from src.ingestion.parsers.html import parse_html
        from src.ingestion.parsers.markdown import parse_markdown
        from src.shared.config import get_config, get_settings

        config = get_config()
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
        doc_tag = None
        snapshot_scope = None

        m = re.search(r"DocTag:\s*([A-Za-z0-9_\-]+)", content or "", flags=re.I)
        if m:
            doc_tag = m.group(1)
        else:
            try:
                fname = Path(source_uri or "").name
                stem = Path(fname).stem
                if "__" in stem:
                    scope_part, slug_part = stem.split("__", 1)
                    snapshot_scope = scope_part
                    doc_tag = slug_part
                else:
                    doc_tag = stem
            except (ValueError, AttributeError) as e:
                logger.debug(
                    "doc_tag_extraction_fallback",
                    source_uri=source_uri,
                    error=str(e),
                )
                # doc_tag remains None, which is acceptable

        document["doc_tag"] = doc_tag
        document["snapshot_scope"] = snapshot_scope

        for section in sections:
            section["doc_tag"] = doc_tag
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
            # Extract other reference patterns from section text (see_also, related, refer_to)
            # These patterns work on plain text without needing markdown link syntax
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

        This ensures we have all vector data ready before starting the atomic transaction.
        Computes dense, sparse, and ColBERT embeddings to match build_graph.py behavior.

        Phase 1 Feature Parity: Token-budgeted batching (EMBED_BATCH_MAX_TOKENS=7000)
        Phase 2 Feature Parity: Per-batch error isolation with None placeholders
        Phase 3 Feature Parity: Comprehensive validation layer (dimension, schema, metadata)

        Returns:
            Dict with:
            - sections: Dict[section_id -> {content, title, sparse?, colbert?}]
            - entities: Dict[entity_id -> [...]] (reserved for future)
            - stats: Dict with sparse coverage and batch metrics
        """
        embeddings = {
            "sections": {},  # section_id -> {content, title, sparse, colbert}
            "entities": {},  # entity_id -> [...] (reserved)
            "stats": {
                # Sparse coverage tracking (Graph Channel Rehabilitation)
                "sparse_eligible": 0,  # Non-stub chunks eligible for sparse
                "sparse_success": 0,  # Chunks that got sparse vectors
                "sparse_failures": 0,  # Batches where embed_sparse failed
                "sparse_content_missing": 0,  # Non-stub chunks without sparse (SLO metric)
                # Batch metrics
                "batch_count": 0,
                "total_tokens_processed": 0,
                # Truncation tracking (consensus-identified enhancement for SLO monitoring)
                "content_truncated": 0,  # Sections that exceeded max_embed_tokens
                "tokens_dropped": 0,  # Total tokens lost to truncation
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

        try:
            tokenizer = TokenizerService()
        except Exception as e:
            logger.warning("tokenizer_init_failed", error=str(e))
            tokenizer = None
        max_embed_tokens = int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))

        # Get expected embedding dimensions from builder config
        embedding_dims = getattr(builder, "embedding_dims", None)
        if embedding_dims is None and hasattr(builder, "embedding_settings"):
            embedding_dims = getattr(builder.embedding_settings, "dimensions", 1024)
        if embedding_dims is None:
            embedding_dims = 1024  # Default fallback

        # Check embedding capabilities
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

        # Prepare batch texts for efficient embedding
        section_data = []
        content_texts = []
        title_texts = []

        for section in sections:
            section_id = section.get("id")
            if not section_id:
                continue

            content_text = builder._build_section_text_for_embedding(section)

            # Skip sections with empty content to prevent HTTP 400 errors from embedding API
            # This handles microdoc stubs created by GreedyCombinerV2 with stub["text"] = ""
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
            if tokenizer and content_tokens > max_embed_tokens:
                chunks = tokenizer.split_to_chunks(content_text, section_id=section_id)
                if not chunks:
                    raise RuntimeError(
                        f"Failed to split over-length text for section {section_id}"
                    )
                content_text = chunks[0]["text"]
                truncated_tokens = tokenizer.count_tokens(content_text)

                # Track truncation for SLO monitoring (consensus-identified enhancement)
                tokens_lost = content_tokens - truncated_tokens
                stats["content_truncated"] += 1
                stats["tokens_dropped"] += tokens_lost

                logger.warning(
                    "embedding_text_truncated",
                    section_id=section_id,
                    original_tokens=content_tokens,
                    truncated_tokens=truncated_tokens,
                    tokens_dropped=tokens_lost,
                    total_chunks=len(chunks),
                )

                # =============================================================
                # CRITICAL FIX: Truncation Metadata Synchronization
                # Mutate section dict so downstream code (Qdrant upsert, hash
                # computation) sees truncated content matching the embeddings.
                # Without this, vectors represent truncated text but payload
                # contains original text - causing retrieval misses and drift
                # detection false negatives.
                # =============================================================
                section["text"] = content_text  # Sync text with what was embedded
                section["token_count"] = truncated_tokens  # Sync token count
                section["tokens"] = (
                    truncated_tokens  # Sync legacy field (used by build_graph.py)
                )
                section["was_truncated"] = True  # Flag for debugging/auditing
                section["original_token_count"] = content_tokens  # Preserve original

                content_tokens = truncated_tokens

            title_text = builder._build_title_text_for_embedding(section)

            section_data.append(
                {
                    "id": section_id,
                    "section": section,
                    "content_text": content_text,
                    "title_text": title_text,
                    "token_count": content_tokens,
                }
            )
            content_texts.append(content_text)
            title_texts.append(title_text)

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

        # Guard capability flags with runtime method detection to prevent
        # confusing "count mismatch" errors when config advertises capability
        # but embedder lacks the method (consensus-identified bug fix)
        has_sparse_method = hasattr(builder.embedder, "embed_sparse")
        has_colbert_method = hasattr(builder.embedder, "embed_colbert")

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
        colbert_embeddings: Optional[List[Optional[List[List[float]]]]] = (
            [] if supports_colbert else None
        )

        # =====================================================================
        # PHASE 1+2: Process batches with error isolation
        # =====================================================================
        for batch_idx, batch_indices in enumerate(batches):
            batch_content = [content_texts[i] for i in batch_indices]
            batch_title = [title_texts[i] for i in batch_indices]
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
                content_embeddings.extend(
                    builder.embedder.embed_documents(batch_content)
                )
                title_embeddings.extend(builder.embedder.embed_documents(batch_title))
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
                builder.embedder, "embed_sparse"
            ):
                try:
                    sparse_embeddings.extend(
                        builder.embedder.embed_sparse(batch_content)
                    )
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

            # Phase 2: Per-Batch Error Isolation for ColBERT Embeddings
            if colbert_embeddings is not None and hasattr(
                builder.embedder, "embed_colbert"
            ):
                try:
                    colbert_embeddings.extend(
                        builder.embedder.embed_colbert(batch_content)
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
        if len(content_embeddings) != len(section_data):
            raise RuntimeError(
                f"Content embedding count mismatch: expected {len(section_data)}, got {len(content_embeddings)}"
            )
        if len(title_embeddings) != len(section_data):
            raise RuntimeError(
                f"Title embedding count mismatch: expected {len(section_data)}, got {len(title_embeddings)}"
            )
        if sparse_embeddings is not None and len(sparse_embeddings) != len(
            section_data
        ):
            raise RuntimeError(
                f"Sparse embedding count mismatch: expected {len(section_data)}, got {len(sparse_embeddings)}"
            )
        if colbert_embeddings is not None and len(colbert_embeddings) != len(
            section_data
        ):
            raise RuntimeError(
                f"ColBERT embedding count mismatch: expected {len(section_data)}, got {len(colbert_embeddings)}"
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
            sparse_vector = (
                sparse_embeddings[idx]
                if sparse_embeddings is not None and idx < len(sparse_embeddings)
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
            }

            # Add sparse if computed (may be None for failed batches)
            if sparse_embeddings is not None:
                section_embedding["sparse"] = sparse_vector

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
            self._neo4j_upsert_document(neo4j_tx, document)
            chunk_count = self._neo4j_upsert_sections(neo4j_tx, document_id, sections)
            stats["sections_upserted"] = chunk_count
            written_neo4j_chunks = [s["id"] for s in sections if "id" in s]

            entity_count = self._neo4j_upsert_entities(neo4j_tx, entities)
            stats["entities_upserted"] = entity_count

            self._neo4j_create_mentions(neo4j_tx, mentions)

            # Phase 3: Create cross-document REFERENCES edges (Chunk → Document)
            references_count = self._neo4j_create_references(neo4j_tx, references)
            stats["references_created"] = references_count

            # Step 2b: Store embedding metadata on Chunk nodes
            # This ensures cross-store consistency between Neo4j and Qdrant
            embedding_meta_count = self._neo4j_upsert_embedding_metadata(
                neo4j_tx, sections, embeddings, builder
            )
            stats["embedding_metadata_upserted"] = embedding_meta_count

            logger.debug(
                "neo4j_writes_completed",
                saga_id=saga_id,
                chunks=chunk_count,
                entities=entity_count,
                references=references_count,
                embedding_metadata=embedding_meta_count,
            )

            # Step 3: Execute Qdrant writes
            # If this fails, we can still rollback Neo4j
            qdrant_count = 0
            if self.qdrant_client and embeddings.get("sections"):
                qdrant_count = self._qdrant_upsert_vectors(
                    document, sections, embeddings, builder
                )
                stats["vectors_upserted"] = qdrant_count
                written_qdrant_points = [s["id"] for s in sections if "id" in s]

                logger.debug(
                    "qdrant_writes_completed",
                    saga_id=saga_id,
                    vectors=qdrant_count,
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
                    self._compensate_qdrant(written_qdrant_points, builder)
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

    def _sanitize_for_neo4j(self, data: Dict) -> Dict:
        """
        Sanitize a dict for Neo4j property storage.

        Neo4j only accepts primitive types (str, int, float, bool, None)
        or arrays of primitives. Nested dicts/maps cause TypeError.

        Strategy:
        - Keep primitives and None
        - Keep lists of primitives
        - JSON-serialize nested dicts/lists with dicts
        - Skip fields that can't be safely serialized
        """
        import json

        sanitized = {}
        for key, value in data.items():
            if value is None:
                continue  # Skip None values entirely

            # Primitives are safe
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
                continue

            # Lists need inspection
            if isinstance(value, list):
                if not value:
                    # Empty list is safe
                    sanitized[key] = value
                elif all(isinstance(item, (str, int, float, bool)) for item in value):
                    # List of primitives is safe
                    sanitized[key] = value
                else:
                    # List contains complex types - JSON serialize
                    try:
                        sanitized[key] = json.dumps(value)
                    except (TypeError, ValueError):
                        logger.debug(
                            "neo4j_sanitize_skip_field",
                            field=key,
                            reason="list_not_serializable",
                        )
                continue

            # Dicts must be JSON serialized
            if isinstance(value, dict):
                try:
                    sanitized[key] = json.dumps(value)
                except (TypeError, ValueError):
                    logger.debug(
                        "neo4j_sanitize_skip_field",
                        field=key,
                        reason="dict_not_serializable",
                    )
                continue

            # Other types - try to convert to string
            try:
                sanitized[key] = str(value)
            except Exception:
                logger.debug(
                    "neo4j_sanitize_skip_field",
                    field=key,
                    reason="unconvertible_type",
                    value_type=type(value).__name__,
                )

        return sanitized

    def _neo4j_upsert_document(self, tx, document: Dict):
        """Upsert document node within a transaction.

        Phase 3 Enhancement: Also resolves Ghost Documents and PendingReferences
        that match the newly ingested document's title.
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d += $props
        """
        props = self._sanitize_for_neo4j(document)
        tx.run(query, id=document["id"], props=props)

        # Phase 3: Resolve Ghost Documents with matching title
        # When a real document is ingested, redirect REFERENCES edges from any
        # Ghost Document that was created as a forward reference placeholder
        title = document.get("title", "")
        if not title:
            # Fallback: derive title from document ID (usually contains filename)
            doc_id = document.get("id", "")
            if doc_id:
                from src.ingestion.extract.references import normalize_filename_to_title

                filename = doc_id.split("/")[-1]
                title = normalize_filename_to_title(filename)
                document["title"] = title
                logger.warning(
                    "empty_title_fallback",
                    document_id=document.get("id"),
                    derived_title=title,
                    reason="document_missing_title",
                )
        if title:
            title_cf = title.casefold()
            # Fix #3 (Phase 3): Atomic ghost resolution with explicit locking
            # The lock prevents race conditions when multiple processes try to:
            # 1. Resolve the same ghost simultaneously
            # 2. Create new edges to a ghost that's being deleted
            #
            # The lock pattern works by:
            # 1. SET ghost._resolve_lock = $doc_id (atomic claim)
            # 2. Only proceed if ghost._resolve_lock = $doc_id (verify we won)
            # 3. Delete ghost atomically with edge cleanup
            #
            # If another process wins the lock, we gracefully skip (the ghost
            # will either be resolved or still exist for our next attempt).
            resolve_ghost_query = """
            // Find Ghost Documents with matching title (case-insensitive)
            // Fix #3: Atomically claim lock before proceeding
            MATCH (ghost:GhostDocument)
            WHERE toLower(ghost.title) = toLower($title_cf)
              AND ghost._resolve_lock IS NULL
            // Atomic lock acquisition - only claim if currently unlocked
            SET ghost._resolve_lock = $doc_id
            WITH ghost
            // Verify we successfully acquired the lock (another tx may have won)
            WHERE ghost._resolve_lock = $doc_id
            // Find all REFERENCES edges pointing to the ghost
            OPTIONAL MATCH (src)-[old_r:REFERENCES]->(ghost)
            // Get the real document we just upserted
            MATCH (real:Document {id: $doc_id})
            WHERE NOT real:GhostDocument
            // Use FOREACH to conditionally create edges only if old edges exist
            // This avoids issues when there are no edges to redirect
            WITH ghost, real, collect({src: src, old_r: old_r}) AS edges
            UNWIND CASE WHEN size(edges) > 0 AND edges[0].old_r IS NOT NULL
                        THEN edges ELSE [] END AS edge
            WITH ghost, real, edge.src AS src, edge.old_r AS old_r
            // Create new edge to real document with same properties
            CREATE (src)-[new_r:REFERENCES]->(real)
            SET new_r = properties(old_r),
                new_r.is_ghost_target = null,
                new_r.resolved_from_ghost = ghost.id,
                new_r.resolved_at = datetime({timezone: 'UTC'})
            WITH ghost, old_r, count(new_r) AS redirected
            // Delete old edge
            DELETE old_r
            WITH ghost, sum(redirected) AS total_redirected
            // Atomically delete ghost - at this point we hold the lock
            // so no other process can add edges
            DETACH DELETE ghost
            RETURN total_redirected AS redirected
            """
            result = tx.run(
                resolve_ghost_query,
                title=title,
                title_cf=title_cf,
                doc_id=document["id"],
            )
            record = result.single()
            if record and record["redirected"] > 0:
                logger.info(
                    "ghost_references_resolved",
                    document_id=document["id"],
                    title=title,
                    redirected_count=record["redirected"],
                )

            # Phase 3: Resolve PendingReferences with matching hint
            # When a real document is ingested, convert pending references that
            # match the document's title into real REFERENCES edges
            # Fix #3: Apply same atomic locking pattern as ghost resolution
            resolve_pending_query = """
            // Find PendingReferences where hint matches document title
            // Fix #3: Atomically claim lock before proceeding
            MATCH (pending:PendingReference)
            WHERE (toLower($title_cf) CONTAINS toLower(pending.hint)
               OR toLower(pending.hint) CONTAINS toLower($title_cf))
              AND pending._resolve_lock IS NULL
            // Atomic lock acquisition
            SET pending._resolve_lock = $doc_id
            WITH pending
            // Verify we successfully acquired the lock
            WHERE pending._resolve_lock = $doc_id
            // Find all PENDING_REF edges
            OPTIONAL MATCH (src)-[old_r:PENDING_REF]->(pending)
            // Get the real document
            MATCH (real:Document {id: $doc_id})
            // Collect edges to handle empty case gracefully
            WITH pending, real, collect({src: src, old_r: old_r}) AS edges
            UNWIND CASE WHEN size(edges) > 0 AND edges[0].old_r IS NOT NULL
                        THEN edges ELSE [] END AS edge
            WITH pending, real, edge.src AS src, edge.old_r AS old_r
            // Create real REFERENCES edge
            CREATE (src)-[new_r:REFERENCES]->(real)
            SET new_r.type = old_r.type,
                new_r.reference_text = old_r.reference_text,
                new_r.confidence = old_r.confidence,
                new_r.source_type = old_r.source_type,
                new_r.target_hint = pending.hint,
                new_r.resolved_from_pending = true,
                new_r.created_at = old_r.created_at,
                new_r.resolved_at = datetime({timezone: 'UTC'})
            WITH pending, old_r, count(new_r) AS resolved
            // Delete old PENDING_REF edge
            DELETE old_r
            WITH pending, sum(resolved) AS total_resolved
            // Atomically delete pending - we hold the lock
            DELETE pending
            RETURN total_resolved AS resolved
            """
            result = tx.run(
                resolve_pending_query,
                title=title,
                title_cf=title_cf,
                doc_id=document["id"],
            )
            record = result.single()
            if record and record["resolved"] > 0:
                logger.info(
                    "pending_references_resolved",
                    document_id=document["id"],
                    title=title,
                    resolved_count=record["resolved"],
                )

    def _neo4j_upsert_sections(self, tx, document_id: str, sections: List[Dict]) -> int:
        """Upsert section/chunk nodes within a transaction."""
        query = """
        UNWIND $sections AS section
        MERGE (c:Chunk {id: section.id})
        SET c += section
        WITH c, section
        MATCH (d:Document {id: $document_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """

        # Prepare sections for Cypher - sanitize to avoid Map{} errors
        section_data = []
        for s in sections:
            # Filter out internal fields
            data = {k: v for k, v in s.items() if k != "_citation_units"}
            # Ensure required fields
            data.setdefault("document_id", document_id)
            # Sanitize for Neo4j (convert dicts to JSON strings, filter non-primitives)
            sanitized = self._sanitize_for_neo4j(data)
            section_data.append(sanitized)

        tx.run(query, sections=section_data, document_id=document_id)
        return len(section_data)

    # Allowlist of valid entity labels to prevent Cypher injection
    # These map to the canonical plan's Entity subtypes
    ENTITY_LABEL_ALLOWLIST = frozenset(
        {
            "Entity",  # Base/fallback label
            "Command",  # CLI commands (weka fs snapshot, etc.)
            "Configuration",  # Config parameters
            "Procedure",  # Multi-step procedures
            "Step",  # Individual steps within procedures
            "Error",  # Error codes/messages
            "Concept",  # Abstract concepts
        }
    )

    def _neo4j_upsert_entities(self, tx, entities: Dict[str, Dict]) -> int:
        """Upsert entity nodes within a transaction.

        Phase 1.2 Enhancement: Entities now get proper subtype labels (Procedure, Step,
        Command, Configuration) in addition to the base Entity label. This enables
        type-specific queries like MATCH (p:Procedure)-[:CONTAINS_STEP]->(s:Step).

        Labels are validated against ENTITY_LABEL_ALLOWLIST to prevent Cypher injection.
        """
        if not entities:
            return 0

        # Group entities by their label for type-specific MERGE queries
        from collections import defaultdict

        by_label = defaultdict(list)

        for eid, edata in entities.items():
            data = dict(edata)  # Copy to avoid mutation
            data["id"] = eid

            # Extract and validate label (default to "Entity" if missing or invalid)
            label = data.get("label", "Entity")
            if label not in self.ENTITY_LABEL_ALLOWLIST:
                logger.warning(
                    "invalid_entity_label_rejected",
                    label=label,
                    entity_id=eid,
                    allowed=list(self.ENTITY_LABEL_ALLOWLIST),
                )
                label = "Entity"

            # Sanitize for Neo4j (convert dicts to JSON strings, filter non-primitives)
            sanitized = self._sanitize_for_neo4j(data)
            by_label[label].append(sanitized)

        # Run separate MERGE queries for each label type
        # This ensures proper Neo4j labels are applied, not just properties
        total = 0
        for label, entity_list in by_label.items():
            # Use f-string for label (safe due to allowlist validation above)
            query = f"""
            UNWIND $entities AS entity
            MERGE (e:Entity:{label} {{id: entity.id}})
            SET e += entity
            """
            tx.run(query, entities=entity_list)
            total += len(entity_list)

        return total

    def _neo4j_create_mentions(self, tx, mentions: List[Dict]):
        """Create MENTIONS and Entity→Entity relationships within a transaction.

        Phase 1.2 Fix: This method now routes relationships based on their key structure:
        - Section→Entity mentions have 'section_id' and 'entity_id' keys
        - Entity→Entity relationships have 'from_id' and 'to_id' keys

        The Entity→Entity relationships (e.g., CONTAINS_STEP from Procedure to Step)
        were previously being silently dropped because only the Section→Entity
        pattern was handled.
        """
        if not mentions:
            return

        # Phase 1.2: Separate mentions by type based on key structure
        section_entity_mentions = []
        entity_entity_relationships = []

        for mention in mentions:
            if "from_id" in mention and "to_id" in mention:
                # Entity→Entity relationship (e.g., Procedure→Step via CONTAINS_STEP)
                entity_entity_relationships.append(mention)
            elif "section_id" in mention and "entity_id" in mention:
                # Section→Entity mention (MENTIONED_IN)
                section_entity_mentions.append(mention)
            else:
                # Log unexpected mention structure for debugging
                logger.warning(
                    "unroutable_mention_structure",
                    keys=list(mention.keys()),
                    mention_type=mention.get("relationship", "unknown"),
                )

        # Process Section→Entity mentions (existing behavior)
        if section_entity_mentions:
            query = """
            UNWIND $mentions AS mention
            MATCH (e:Entity {id: mention.entity_id})
            MATCH (c:Chunk {id: mention.section_id})
            MERGE (e)-[r:MENTIONED_IN]->(c)
            SET r.count = coalesce(r.count, 0) + 1
            """
            tx.run(query, mentions=section_entity_mentions)

        # Phase 1.2: Process Entity→Entity relationships
        if entity_entity_relationships:
            self._neo4j_create_entity_relationships(tx, entity_entity_relationships)

    def _neo4j_create_references(self, tx, references: List[Dict]) -> int:
        """Create cross-document REFERENCES edges within a transaction.

        Phase 3: Implements (Chunk)-[:REFERENCES]->(Document) edge pattern.

        Fix #2 (Phase 3): Batched UNWIND implementation replaces O(N) per-reference
        queries with 4 batched operations:
        1. Batch resolve titles for hyperlinks
        2. Batch resolve fuzzy hints for non-hyperlinks
        3. Batch create ghost/pending/resolved references

        This method handles the consensus-approved hybrid edge pattern where:
        - Source is Chunk (preserves WHERE the reference occurred for RAG citations)
        - Target is Document (reliable resolution without brittle anchor matching)

        Reference dict structure (from extract/references.py):
        - source_chunk_id: The chunk where the reference was found
        - target_doc_id: Pre-resolved document ID (may be None)
        - target_hint: Original hint for Neo4j resolution (filename, title, phrase)
        - reference_type: Type of reference (hyperlink, see_also, related, refer_to)
        - reference_text: Display text of the reference
        - confidence: Extraction confidence score

        Returns:
            Number of REFERENCES edges created
        """
        # Delegate to streaming implementation (bug 11 hardening)
        return self._neo4j_create_references_streaming(tx, references)

    def _neo4j_create_references_streaming(self, tx, references: List[Dict]) -> int:
        """Streaming, rollback-safe creation of REFERENCES edges (production-grade fix for bug 11)."""

        if not references:
            return 0

        from src.ingestion.extract.references import (
            normalize_filename_to_title,
            slugify_for_id,
        )

        refs_cfg = getattr(getattr(self, "config", None), "references", None)
        res_cfg = getattr(refs_cfg, "resolution", None) if refs_cfg else None
        batch_size = getattr(res_cfg, "batch_size", 100)
        min_hint_len = getattr(res_cfg, "min_hint_length", 3)
        penalty_cfg = getattr(
            getattr(res_cfg, "__dict__", res_cfg), "fuzzy_penalty", None
        )
        FUZZY_RESOLUTION_PENALTY = penalty_cfg if penalty_cfg is not None else 0.25

        created_count = 0
        unresolved_count = 0
        created_ghost_ids: List[str] = []

        resolved_buffer: List[Dict] = []
        ghost_buffer: List[Dict] = []
        pending_buffer: List[Dict] = []

        def flush_resolved(buf: List[Dict]):
            nonlocal created_count
            if not buf:
                return
            query = """
            UNWIND $refs AS ref
            MATCH (src:Chunk {id: ref.source_chunk_id})
            MATCH (d:Document {id: ref.target_doc_id})
            MERGE (src)-[r:REFERENCES]->(d)
            ON CREATE SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.final_confidence,
                r.target_hint = ref.target_hint,
                r.source_type = 'chunk',
                r.created_at = datetime({timezone: 'UTC'})
            ON MATCH SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.final_confidence,
                r.source_type = 'chunk',
                r.updated_at = datetime({timezone: 'UTC'})
            RETURN count(r) AS created
            """
            rec = tx.run(query, refs=buf).single()
            if rec:
                created_count += rec["created"]
            buf.clear()

        def flush_ghost(buf: List[Dict]):
            nonlocal created_count, created_ghost_ids
            if not buf:
                return
            query = """
            UNWIND $refs AS ref
            MERGE (ghost:GhostDocument {id: ref.ghost_id})
            ON CREATE SET
                ghost.title = ref.expected_title,
                ghost.stub = true,
                ghost.source_hint = ref.target_hint,
                ghost.created_at = datetime({timezone: 'UTC'})
            WITH ghost, ref
            MATCH (src:Chunk {id: ref.source_chunk_id})
            MERGE (src)-[r:REFERENCES]->(ghost)
            ON CREATE SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.target_hint = ref.target_hint,
                r.source_type = 'chunk',
                r.is_ghost_target = true,
                r.created_at = datetime({timezone: 'UTC'})
            ON MATCH SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.source_type = 'chunk',
                r.updated_at = datetime({timezone: 'UTC'})
            RETURN collect(DISTINCT ghost.id) AS ghost_ids, count(r) AS created
            """
            rec = tx.run(query, refs=buf).single()
            if rec:
                created_count += rec["created"]
                created_ghost_ids.extend(rec.get("ghost_ids", []))
            buf.clear()

        def flush_pending(buf: List[Dict]):
            nonlocal unresolved_count
            if not buf:
                return
            query = """
            UNWIND $refs AS ref
            MERGE (pending:PendingReference {hint: ref.target_hint})
            ON CREATE SET
                pending.created_at = datetime({timezone: 'UTC'}),
                pending.reference_count = 1
            ON MATCH SET
                pending.reference_count = coalesce(pending.reference_count, 0) + 1,
                pending.updated_at = datetime({timezone: 'UTC'})
            WITH pending, ref
            MATCH (src:Chunk {id: ref.source_chunk_id})
            MERGE (src)-[r:PENDING_REF]->(pending)
            ON CREATE SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.source_type = 'chunk',
                r.created_at = datetime({timezone: 'UTC'})
            ON MATCH SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.source_type = 'chunk',
                r.updated_at = datetime({timezone: 'UTC'})
            RETURN count(r) AS created
            """
            rec = tx.run(query, refs=buf).single()
            if rec:
                unresolved_count += rec["created"]
            buf.clear()

        def stage_resolved(ref_obj: Dict):
            resolved_buffer.append(ref_obj)
            if len(resolved_buffer) >= batch_size:
                flush_resolved(resolved_buffer)

        def stage_ghost(ref_obj: Dict):
            ghost_buffer.append(ref_obj)
            if len(ghost_buffer) >= batch_size:
                flush_ghost(ghost_buffer)

        def stage_pending(ref_obj: Dict):
            pending_buffer.append(ref_obj)
            if len(pending_buffer) >= batch_size:
                flush_pending(pending_buffer)

        def process_title_batch(batch: List[Dict]):
            if not batch:
                return
            titles_to_resolve = list({r["possible_title"] for r in batch})
            title_to_doc = {}
            if titles_to_resolve:
                query = """
                UNWIND $titles AS title
                OPTIONAL MATCH (d:Document)
                WHERE toLower(d.title) = toLower(title)
                RETURN title, d.id AS doc_id
                """
                title_to_doc = {
                    rec["title"]: rec["doc_id"]
                    for rec in tx.run(query, titles=titles_to_resolve)
                    if rec["doc_id"]
                }

            for ref_data in batch:
                resolved_id = title_to_doc.get(ref_data["possible_title"])
                if resolved_id:
                    ref_data["target_doc_id"] = resolved_id
                    ref_data["final_confidence"] = ref_data["confidence"]
                    stage_resolved(ref_data)
                else:
                    expected_title = normalize_filename_to_title(
                        ref_data["target_hint"]
                    )
                    ref_data["ghost_id"] = f"ghost::{slugify_for_id(expected_title)}"
                    ref_data["expected_title"] = expected_title
                    stage_ghost(ref_data)
            batch.clear()

        def process_fuzzy_batch(batch: List[Dict]):
            if not batch:
                return
            hints = {r["target_hint"] for r in batch if r.get("target_hint")}
            hint_to_doc = {}
            if hints:
                query = """
                UNWIND $hints AS hint
                CALL db.index.fulltext.queryNodes('document_title_ft', hint) YIELD node AS d, score
                WHERE score > 0.5
                WITH hint, d, score
                ORDER BY score DESC, size(d.title) ASC
                WITH hint, collect(d.id)[0] AS doc_id
                RETURN hint, doc_id
                """
                hint_to_doc = {
                    rec["hint"]: rec["doc_id"]
                    for rec in tx.run(query, hints=list(hints))
                    if rec["doc_id"]
                }

            for ref_data in batch:
                target_hint = ref_data.get("target_hint", "")
                if len(target_hint.strip()) < min_hint_len:
                    stage_pending(ref_data)
                    continue
                resolved_id = hint_to_doc.get(target_hint)
                if resolved_id:
                    final_conf = ref_data["confidence"]
                    final_conf = max(0.1, final_conf - FUZZY_RESOLUTION_PENALTY)
                    ref_data["target_doc_id"] = resolved_id
                    ref_data["final_confidence"] = final_conf
                    ref_data["is_fuzzy_match"] = True
                    stage_resolved(ref_data)
                else:
                    stage_pending(ref_data)
            batch.clear()

        title_batch: List[Dict] = []
        fuzzy_batch: List[Dict] = []

        try:
            for ref in references:
                source_chunk_id = ref.get("source_chunk_id")
                if not source_chunk_id:
                    logger.warning(
                        "reference_missing_source_chunk",
                        target_hint=ref.get("target_hint", ""),
                        reference_type=ref.get("reference_type", "unknown"),
                    )
                    continue

                ref_data = {
                    "source_chunk_id": source_chunk_id,
                    "target_doc_id": ref.get("target_doc_id"),
                    "target_hint": ref.get("target_hint", ""),
                    "reference_type": ref.get("reference_type", "unknown"),
                    "reference_text": ref.get("reference_text", ""),
                    "confidence": ref.get("confidence", 0.5),
                    "is_fuzzy_match": False,
                }

                if ref_data["target_doc_id"]:
                    ref_data["final_confidence"] = ref_data["confidence"]
                    stage_resolved(ref_data)
                    continue

                if ref_data["reference_type"] == "hyperlink" and ref_data[
                    "target_hint"
                ].endswith(".md"):
                    ref_data["possible_title"] = normalize_filename_to_title(
                        ref_data["target_hint"]
                    )
                    title_batch.append(ref_data)
                    if len(title_batch) >= batch_size * 2:
                        process_title_batch(title_batch)
                    continue

                fuzzy_batch.append(ref_data)
                if len(fuzzy_batch) >= batch_size * 2:
                    process_fuzzy_batch(fuzzy_batch)

            process_title_batch(title_batch)
            process_fuzzy_batch(fuzzy_batch)
            flush_resolved(resolved_buffer)
            flush_ghost(ghost_buffer)
            flush_pending(pending_buffer)

        except Exception as exc:
            if created_ghost_ids:
                tx.run(
                    """
                    UNWIND $ids AS gid
                    MATCH (g:GhostDocument {id: gid})
                    DETACH DELETE g
                    """,
                    ids=created_ghost_ids,
                )
            logger.warning(
                "references_rollback_ghosts",
                ghost_ids=created_ghost_ids,
                error=str(exc),
            )
            raise

        if created_count > 0 or unresolved_count > 0:

            def _safe_log_value(value: str, max_length: int = 200) -> str:
                if not value:
                    return ""
                sanitized = re.sub(r"[\\x00-\\x1f\\x7f-\\x9f]", "", value)
                return (
                    (sanitized[:max_length] + "...")
                    if len(sanitized) > max_length
                    else sanitized
                )

            logger.info(
                "references_created",
                created=created_count,
                unresolved=unresolved_count,
                total_attempted=len(references),
                sample_hint=_safe_log_value(
                    (resolved_buffer or ghost_buffer or pending_buffer or [{}])[0].get(
                        "target_hint", ""
                    )
                ),
            )

        return created_count

    # Override earlier definition to delegate to streaming, production-grade path (bug 11)
    def _neo4j_create_references(self, tx, references: List[Dict]) -> int:
        return self._neo4j_create_references_streaming(tx, references)

    def _neo4j_create_entity_relationships(self, tx, relationships: List[Dict]):
        """Create Entity→Entity relationships within a transaction.

        Phase 1.2: Handles relationships like CONTAINS_STEP (Procedure→Step)
        that were previously being dropped by _neo4j_create_mentions.

        Relationship dict structure:
        - from_id: Source entity ID
        - from_label: Source entity label (e.g., "Procedure")
        - to_id: Target entity ID
        - to_label: Target entity label (e.g., "Step")
        - relationship: Relationship type (e.g., "CONTAINS_STEP")
        - order: Optional ordering field for sequential relationships
        - confidence: Extraction confidence score
        - source_section_id: Section where relationship was extracted
        """
        if not relationships:
            return

        # Group relationships by type for efficient batch processing
        by_type: Dict[str, List[Dict]] = {}
        for rel in relationships:
            rel_type = rel.get("relationship")
            if not rel_type:
                logger.warning(
                    "entity_relationship_missing_type",
                    from_id=rel.get("from_id"),
                    to_id=rel.get("to_id"),
                    reason="relationship_type_required",
                )
                continue  # Skip relationships without explicit type
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        # Process each relationship type with a dedicated query
        rejected_count = 0
        for rel_type, rels in by_type.items():
            # Defense-in-depth: Validate relationship type against allowlist
            # This prevents Cypher injection via malformed/malicious extractor output
            if rel_type not in ALLOWED_ENTITY_RELATIONSHIP_TYPES:
                logger.warning(
                    "entity_relationship_type_rejected",
                    relationship_type=rel_type,
                    count=len(rels),
                    reason="not_in_allowlist",
                    allowed_types=list(ALLOWED_ENTITY_RELATIONSHIP_TYPES),
                )
                rejected_count += len(rels)
                continue

            # Since relationship types must be known at query compile time in Cypher,
            # we use separate queries per type (validated above against allowlist)
            #
            # Issue #7 Fix: Use OPTIONAL MATCH and return counts to detect missing entities
            # Issue #18 Fix: Use datetime({timezone: 'UTC'}) for explicit UTC timezone
            #
            # Multi-model validation fix: Aggregate missing counts BEFORE filtering
            # The collect/unwind pattern preserves missing counts through the WHERE clause
            query = f"""
            UNWIND $rels AS rel
            OPTIONAL MATCH (from:Entity {{id: rel.from_id}})
            OPTIONAL MATCH (to:Entity {{id: rel.to_id}})
            WITH rel, from, to,
                 CASE WHEN from IS NULL THEN 1 ELSE 0 END AS mf,
                 CASE WHEN to IS NULL THEN 1 ELSE 0 END AS mt
            WITH collect({{r: rel, f: from, t: to}}) AS pairs,
                 sum(mf) AS total_missing_from,
                 sum(mt) AS total_missing_to
            UNWIND pairs AS p
            WITH p.r AS rel, p.f AS from, p.t AS to,
                 total_missing_from, total_missing_to
            WHERE from IS NOT NULL AND to IS NOT NULL
            MERGE (from)-[r:{rel_type}]->(to)
            SET r.order = rel.order,
                r.confidence = rel.confidence,
                r.source_section_id = rel.source_section_id,
                r.created_at = datetime({{timezone: 'UTC'}})
            // N1 Fix: COALESCE prevents NULL when all rows filtered by WHERE clause
            RETURN count(r) AS created_count,
                   COALESCE(max(total_missing_from), 0) AS missing_from_count,
                   COALESCE(max(total_missing_to), 0) AS missing_to_count
            """
            result = tx.run(query, rels=rels)
            record = result.single()

            # N1 Fix: Defensive None handling (belt-and-suspenders with Cypher COALESCE)
            created_count = record["created_count"] if record else 0
            missing_from = (record.get("missing_from_count") or 0) if record else 0
            missing_to = (record.get("missing_to_count") or 0) if record else 0

            # Issue #7: Log when entities are not found
            if missing_from > 0 or missing_to > 0:
                logger.warning(
                    "entity_relationships_missing_entities",
                    relationship_type=rel_type,
                    attempted=len(rels),
                    created=created_count,
                    missing_from_entities=missing_from,
                    missing_to_entities=missing_to,
                )
                # Issue #8: Prometheus metric for missing entities
                entity_relationships_missing_total.labels(
                    relationship_type=rel_type
                ).inc(missing_from + missing_to)

            # Issue #8: Prometheus metrics for created relationships
            if created_count > 0:
                entity_relationships_total.labels(
                    relationship_type=rel_type, status="created"
                ).inc(created_count)

            logger.debug(
                "entity_relationships_created",
                relationship_type=rel_type,
                attempted=len(rels),
                created=created_count,
            )

        # Issue #8: Prometheus metric for rejected relationships
        if rejected_count > 0:
            entity_relationships_total.labels(
                relationship_type="rejected", status="rejected"
            ).inc(rejected_count)
            logger.info(
                "entity_relationships_rejected_total",
                rejected_count=rejected_count,
            )

    def _neo4j_upsert_embedding_metadata(
        self,
        tx,
        sections: List[Dict],
        embeddings: Dict,
        builder,
    ) -> int:
        """
        Store embedding metadata on Chunk nodes in Neo4j.

        This ensures cross-store consistency between Neo4j and Qdrant by storing
        the same embedding metadata in both stores. Matches build_graph.py behavior.

        Args:
            tx: Neo4j transaction
            sections: List of section dicts
            embeddings: Pre-computed embeddings dict
            builder: GraphBuilder instance with embedding settings

        Returns:
            Number of chunks updated with embedding metadata
        """
        if not embeddings.get("sections"):
            return 0

        if not hasattr(builder, "embedding_settings") or not builder.embedding_settings:
            logger.warning(
                "neo4j_embedding_metadata_skipped",
                reason="No embedding_settings in builder",
            )
            return 0

        # Build batch update data
        updates = []
        for section in sections:
            section_id = section.get("id")
            if not section_id:
                continue

            section_embeddings = embeddings.get("sections", {}).get(section_id)
            if not section_embeddings:
                continue

            content_embedding = section_embeddings.get("content", [])

            updates.append(
                {
                    "id": section_id,
                    "embedding_version": builder.embedding_settings.version,
                    "embedding_provider": (
                        getattr(builder.embedder, "provider_name", None)
                        if hasattr(builder, "embedder")
                        else None
                    ),
                    "embedding_dimensions": (
                        len(content_embedding)
                        if content_embedding
                        else builder.embedding_settings.dims
                    ),
                    "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
                    "embedding_task": builder.embedding_settings.task,
                    # Note: We don't store vector_embedding in Neo4j for atomic path
                    # to reduce write amplification. Qdrant is the primary vector store.
                }
            )

        if not updates:
            return 0

        # Batch update embedding metadata on Chunk nodes
        query = """
        UNWIND $updates AS update
        MATCH (c:Chunk {id: update.id})
        SET c.embedding_version = update.embedding_version,
            c.embedding_provider = update.embedding_provider,
            c.embedding_dimensions = update.embedding_dimensions,
            c.embedding_timestamp = update.embedding_timestamp,
            c.embedding_task = update.embedding_task
        """

        tx.run(query, updates=updates)

        logger.debug(
            "neo4j_embedding_metadata_upserted",
            chunks_updated=len(updates),
            provider=updates[0]["embedding_provider"] if updates else None,
        )

        return len(updates)

    # =========================================================================
    # Helper methods for canonical payload fields (matching build_graph.py)
    # =========================================================================

    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA256 hash of text content for drift detection."""
        value = text or ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _compute_shingle_hash(self, text: str, n: int = 8) -> str:
        """Compute shingle hash for deduplication."""
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""
        shingles = []
        limit = 64
        for i in range(0, max(0, len(tokens) - n + 1)):
            shingles.append(" ".join(tokens[i : i + n]))
            if len(shingles) >= limit:
                break
        if not shingles:
            return ""
        combined = "|".join(sorted(set(shingles)))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _extract_semantic_metadata(self, section: Dict) -> Dict[str, Any]:
        """Extract semantic metadata from section (NER, topics, etc.)."""
        metadata = section.get("semantic_metadata")
        if metadata:
            return metadata
        return {"entities": [], "topics": []}

    def _qdrant_upsert_vectors(
        self,
        document: Dict,
        sections: List[Dict],
        embeddings: Dict,
        builder,
    ) -> int:
        """Upsert vectors to Qdrant."""
        import uuid as uuid_mod

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
                    "GraphBuilder missing embedding_settings - cannot create canonical payload. "
                    "Ensure embedding configuration is properly set in config."
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
                # === Versioning (2 fields) ===
                "lang": section.get("lang") or document.get("lang"),
                "version": section.get("version") or document.get("version"),
                # === Timestamps (1 field) ===
                "updated_at": datetime.utcnow().isoformat() + "Z",
                # === Hashes for drift detection (2 fields) ===
                "text_hash": section.get("text_hash")
                or self._compute_text_hash(text_content),
                "shingle_hash": section.get("shingle_hash")
                or self._compute_shingle_hash(text_content),
                # === Semantic metadata (1 field) ===
                "semantic_metadata": self._extract_semantic_metadata(section),
                # === Embedding metadata (5+ fields via spread) ===
                **embedding_metadata,
            }

            # CRITICAL: Remove any legacy embedding_model field that might have leaked in
            payload = ensure_no_embedding_model_in_payload(payload)

            # Build vectors dict with dense vectors
            vectors = {
                "content": section_embeddings["content"],
                "title": section_embeddings["title"],
            }

            # Phase 5: Add entity vector if enabled (copy of content embedding)
            # This maintains schema parity with GraphBuilder when QDRANT_INCLUDE_ENTITY_VECTOR=true
            if self.include_entity_vector:
                vectors["entity"] = section_embeddings["content"]

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
            # Both content and title vectors use the same embedding model
            expected_dim = {
                "content": builder.embedding_settings.dims,
                "title": builder.embedding_settings.dims,
            }
            # Phase 5: Include entity dimension if entity vector is enabled
            if self.include_entity_vector:
                expected_dim["entity"] = builder.embedding_settings.dims

            # Phase 7F: Batch Qdrant upserts to prevent timeout for large ColBERT payloads
            # Large documents with many chunks can produce 30MB+ JSON payloads that exceed
            # the 30-second client timeout. Batching reduces payload size.
            # See: Debug investigation 2025-12-01 - ResponseHandlingException timeout fix
            batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "5"))
            max_bytes_per_batch = int(
                os.getenv("QDRANT_UPSERT_MAX_BYTES", str(12 * 1024 * 1024))
            )  # ~12MB default target

            json_overhead_factor = 2.5  # conservative multiplier for JSON vs. binary
            dense_vector_dims = 2 + (1 if self.include_entity_vector else 0)
            dense_dim_bytes = int(
                builder.embedding_settings.dims
                * dense_vector_dims
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

                # Use retry wrapper with dimension validation for transient network failures
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

        Handles transient network failures that are common in distributed systems.
        Non-retriable errors (e.g., schema mismatch, dimension mismatch) fail immediately.

        Args:
            collection: Qdrant collection name
            points: List of PointStruct to upsert
            expected_dim: Expected vector dimensions, e.g. {"content": 1024, "title": 1024}
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
                # Phase 7F: Extended retry to include ResponseHandlingException
                # which is raised by qdrant-client on HTTP timeouts (e.g., large ColBERT payloads)
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
                # This preserves error context that was previously lost in silent re-raise
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
        import uuid as uuid_mod

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

        Compensation is critical - we retry harder to ensure data consistency.
        Phase 4: Records qdrant_delete_total counter and qdrant_operation_latency_ms histogram.

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

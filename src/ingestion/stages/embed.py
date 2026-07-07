"""Embedding computation stage for atomic ingestion."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.providers.factory import ProviderFactory
from src.providers.tokenizer_service import TokenizerService
from src.shared.chunk_utils import validate_chunk_schema
from src.shared.embedding_fields import (
    canonicalize_embedding_metadata,
    validate_embedding_metadata,
)
from src.shared.observability import get_logger

logger = get_logger(__name__)


def compute_embeddings(
    document: Dict[str, Any],
    sections: List[Dict[str, Any]],
    entities: Dict[str, Any],
    builder,
    config,
    *,
    trace=None,
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
    _ = trace

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
            and embedding_plan.sparse.profile_name != embedding_plan.dense.profile_name
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
                embedding_plan.colbert.profile_name != embedding_plan.dense.profile_name
            ):
                colbert_embedder = ProviderFactory.create_embedding_provider_for_role(
                    embedding_plan.colbert
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
    qdrant_cfg = getattr(config.search.vector, "qdrant", None)
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
        getattr(config.search.vector, "qdrant", None),
        "sparse_strict_mode",
        False,
    )

    # Check if doc_title-sparse vectors should be generated
    # Allows disabling doc_title sparse independently of text-sparse
    # Default: True for backward compat
    enable_doc_title_sparse = getattr(
        getattr(config.search.vector, "qdrant", None),
        "enable_doc_title_sparse",
        True,
    )

    # Check if title-sparse vectors should be generated
    # (section heading lexical matching). Default: True
    enable_title_sparse = getattr(
        getattr(config.search.vector, "qdrant", None),
        "enable_title_sparse",
        True,
    )

    # Check if entity-sparse vectors should be generated
    # (entity name lexical matching). Default: True
    enable_entity_sparse = getattr(
        getattr(config.search.vector, "qdrant", None),
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
                    base_text = builder._build_section_text_for_embedding(base_section)
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
                    heading=(section.get("title") or section.get("heading") or "")[:80],
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
    stats["total_tokens_processed"] = sum(d.get("token_count", 0) for d in section_data)

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

    sparse_embeddings: Optional[List[Optional[dict]]] = [] if supports_sparse else None
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
    if use_contextual and not hasattr(dense_embedder, "embed_contextualized_documents"):
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
        batch_tokens = sum(section_data[i].get("token_count", 0) for i in batch_indices)

        logger.debug(
            "processing_embedding_batch",
            batch_index=batch_idx,
            batch_size=len(batch_indices),
            batch_tokens=batch_tokens,
        )

        # Dense embeddings - required, fail-all on error
        try:
            if not use_contextual:
                content_embeddings.extend(dense_embedder.embed_documents(batch_content))
            title_embeddings.extend(dense_embedder.embed_documents(batch_title))
            doc_title_embeddings.extend(dense_embedder.embed_documents(batch_doc_title))
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
        if sparse_embeddings is not None and hasattr(sparse_embedder, "embed_sparse"):
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
                doc_title_sparse_embeddings.extend(_embed_sparse_safe(batch_doc_title))
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
                                entity_names.append(entity_id_to_name[m["entity_id"]])
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
                    entity_sparse_embeddings.extend([None] * len(batch_entity_texts))
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
                colbert_embeddings.extend(colbert_embedder.embed_colbert(batch_content))
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
    if title_sparse_embeddings is not None and len(title_sparse_embeddings) != expected:
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

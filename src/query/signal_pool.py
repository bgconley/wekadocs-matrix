# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (signal-diverse rerank pool)
# =============================================================================
"""
Signal-diverse rerank pool builder.

Replaces the flat "top N by fused score" rerank input with a pool that
ensures the cross-encoder evaluates chunks from every retrieval signal
source. Pure functions — no IO, no Qdrant/Neo4j calls. Independently
testable.

The algorithm fills slots in priority order:
1. Consensus (top by fused_score — multi-signal agreement)
2. Per-signal unique slots (chunks that scored well on one signal but
   were buried by RRF fusion)
3. Structural graph expansion (NEXT_CHUNK + sibling neighbors)
4. Per-document depth (ensure top documents have deep representation)
5. Backfill (fill remaining capacity from fused_score order)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Set

if TYPE_CHECKING:
    from src.query.retrieval_types import ChunkResult
    from src.shared.config import SignalPoolConfig

logger = logging.getLogger(__name__)


@dataclass
class SignalPoolResult:
    """Output of the signal pool builder."""

    pool: List[ChunkResult]  # The selected candidates for reranking
    slot_fills: Dict[str, int] = field(default_factory=dict)  # Count per slot
    overflow_count: int = 0  # Candidates that didn't fit
    degraded: bool = False  # True if per-field scores were unavailable


def build_signal_pool(
    fused_results: List[ChunkResult],
    structural_candidates: List[ChunkResult],
    config: SignalPoolConfig,
) -> SignalPoolResult:
    """
    Build a signal-diverse rerank pool.

    Args:
        fused_results: All chunks from fusion, sorted by fused_score descending.
        structural_candidates: Chunks from pre-rerank graph expansion.
        config: Signal pool configuration with slot allocations.

    Returns:
        SignalPoolResult with the selected pool and fill statistics.
    """
    pool_ids: Set[str] = set()
    pool: List[ChunkResult] = []
    slot_fills: Dict[str, int] = {}

    def _add(chunk: ChunkResult, slot_name: str) -> bool:
        """Add a chunk to the pool if not already present. Returns True if added."""
        if chunk.chunk_id in pool_ids:
            return False
        if len(pool) >= config.pool_size:
            return False
        pool_ids.add(chunk.chunk_id)
        pool.append(chunk)
        slot_fills[slot_name] = slot_fills.get(slot_name, 0) + 1
        return True

    # Detect whether per-field scores are available (weighted fusion was enabled)
    has_per_field = _has_per_field_scores(fused_results)

    # --- Slot 1: Consensus (top by fused_score) ---
    sorted_by_fused = sorted(
        fused_results, key=lambda c: c.fused_score or 0, reverse=True
    )
    for c in sorted_by_fused:
        if slot_fills.get("consensus", 0) >= config.consensus_slots:
            break
        _add(c, "consensus")

    # --- Slots 2-7: Per-signal unique slots ---
    if has_per_field:
        _fill_per_field_slots(fused_results, pool_ids, _add, config)
    elif config.fallback_to_provenance:
        _fill_provenance_slots(fused_results, pool_ids, _add, config)

    # --- Slot 8: Structural expansion ---
    for c in structural_candidates:
        if slot_fills.get("structural", 0) >= config.structural_slots:
            break
        _add(c, "structural")

    # --- Slot 9: Per-document depth ---
    _fill_per_doc_depth(fused_results, pool_ids, _add, config)

    # --- Backfill: fill remaining capacity from fused_score order ---
    remaining = config.pool_size - len(pool)
    if remaining > 0:
        for c in sorted_by_fused:
            if remaining <= 0:
                break
            if _add(c, "backfill"):
                remaining -= 1

    result = SignalPoolResult(
        pool=pool,
        slot_fills=slot_fills,
        overflow_count=max(0, len(fused_results) - len(pool)),
        degraded=not has_per_field,
    )

    logger.info(
        "signal_pool_built",
        extra={
            "pool_size": len(pool),
            "slot_fills": slot_fills,
            "degraded": result.degraded,
            "fused_input_count": len(fused_results),
            "structural_input_count": len(structural_candidates),
        },
    )

    return result


def _has_per_field_scores(fused_results: List[ChunkResult]) -> bool:
    """Check if per-field scores are populated (weighted fusion was active)."""
    # Sample the first 50 results — if any have per-field scores, they're available
    for c in fused_results[:50]:
        if (
            c.title_vec_score is not None
            or c.entity_vec_score is not None
            or c.doc_title_vec_score is not None
        ):
            return True
    return False


def _fill_signal_slot(
    candidates: List[ChunkResult],
    pool_ids: Set[str],
    add_fn: Callable,
    score_fn: Callable[[ChunkResult], float],
    slot_name: str,
    limit: int,
) -> None:
    """Fill a signal slot with top candidates by that signal, excluding pool members."""
    ranked = sorted(candidates, key=score_fn, reverse=True)
    filled = 0
    for c in ranked:
        if filled >= limit:
            break
        if c.chunk_id not in pool_ids and score_fn(c) > 0:
            if add_fn(c, slot_name):
                filled += 1


def _fill_per_field_slots(
    fused_results: List[ChunkResult],
    pool_ids: Set[str],
    add_fn: Callable,
    config: SignalPoolConfig,
) -> None:
    """Fill all per-signal slots using individual field scores."""
    signal_slots = [
        (lambda c: c.vector_score or 0, "content_dense", config.content_dense_slots),
        (lambda c: c.title_vec_score or 0, "title_dense", config.title_dense_slots),
        (
            lambda c: c.doc_title_vec_score or 0,
            "doc_title_dense",
            config.doc_title_dense_slots,
        ),
        (
            lambda c: max(c.lexical_vec_score or 0, c.bm25_score or 0),
            "text_sparse",
            config.text_sparse_slots,
        ),
        (
            lambda c: c.entity_vec_score or 0,
            "entity_sparse",
            config.entity_sparse_slots,
        ),
        (
            lambda c: c.title_sparse_score or 0,
            "title_sparse",
            config.title_sparse_slots,
        ),
        (
            lambda c: c.related_to_score or 0,
            "related_to",
            config.related_to_slots,
        ),
    ]

    for score_fn, slot_name, limit in signal_slots:
        _fill_signal_slot(fused_results, pool_ids, add_fn, score_fn, slot_name, limit)


def _fill_provenance_slots(
    fused_results: List[ChunkResult],
    pool_ids: Set[str],
    add_fn: Callable,
    config: SignalPoolConfig,
) -> None:
    """Fallback: fill slots using coarse BM25 vs vector provenance only.

    Used when per-field scores are unavailable (weighted fusion not enabled).
    """
    # BM25-strong chunks (high bm25_score)
    bm25_budget = (
        config.text_sparse_slots
        + config.entity_sparse_slots
        + config.title_sparse_slots
    )
    _fill_signal_slot(
        fused_results,
        pool_ids,
        add_fn,
        score_fn=lambda c: c.bm25_score or 0,
        slot_name="bm25_provenance",
        limit=bm25_budget,
    )

    # Vector-strong chunks (high vector_score)
    vector_budget = (
        config.content_dense_slots
        + config.title_dense_slots
        + config.doc_title_dense_slots
    )
    _fill_signal_slot(
        fused_results,
        pool_ids,
        add_fn,
        score_fn=lambda c: c.vector_score or 0,
        slot_name="vector_provenance",
        limit=vector_budget,
    )


def _fill_per_doc_depth(
    fused_results: List[ChunkResult],
    pool_ids: Set[str],
    add_fn: Callable,
    config: SignalPoolConfig,
) -> None:
    """Ensure top documents have deep chunk representation in the pool.

    For each of the top K documents (by best chunk score), select up to M
    chunks not already in the pool. Prefers chunks that contribute different
    signal types within the same document.
    """
    # Group chunks by document and find best score per document
    doc_best: Dict[str, float] = {}
    doc_chunks: Dict[str, List[ChunkResult]] = defaultdict(list)
    for c in fused_results:
        score = c.fused_score or 0
        if score > doc_best.get(c.document_id, 0):
            doc_best[c.document_id] = score
        doc_chunks[c.document_id].append(c)

    # Select top K documents by their best chunk score
    top_docs = sorted(doc_best.keys(), key=lambda d: doc_best[d], reverse=True)
    top_docs = top_docs[: config.per_doc_k]

    filled = 0
    for doc_id in top_docs:
        if filled >= config.per_doc_depth_slots:
            break

        # Get chunks from this doc not yet in pool
        available = [c for c in doc_chunks[doc_id] if c.chunk_id not in pool_ids]
        if not available:
            continue

        # Sort by a diversity key: prefer chunks with high scores on ANY signal
        # This surfaces chunks that are strong on one dimension even if fused_score is low
        available.sort(
            key=lambda c: max(
                c.fused_score or 0,
                c.bm25_score or 0,
                c.vector_score or 0,
                c.entity_vec_score or 0,
                c.title_vec_score or 0,
                c.doc_title_vec_score or 0,
                c.doc_title_sparse_score or 0,
                c.lexical_vec_score or 0,
            ),
            reverse=True,
        )

        for c in available[: config.per_doc_m]:
            if filled >= config.per_doc_depth_slots:
                break
            if add_fn(c, "per_doc_depth"):
                filled += 1

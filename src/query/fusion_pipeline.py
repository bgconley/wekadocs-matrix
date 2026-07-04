# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade wrappers)
# =============================================================================
"""
Fusion pipeline: RRF fusion, weighted fusion, and document continuity boost.

Extracted from hybrid_retrieval.py to isolate score fusion logic behind
a stable interface.
"""

from collections import Counter
from typing import Dict, List

from src.query.retrieval_types import ChunkResult
from src.shared.observability import get_logger

logger = get_logger(__name__)


def rrf_fusion(
    owner, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
) -> List[ChunkResult]:
    """
    Reciprocal Rank Fusion (RRF) - robust fusion without parameter tuning.

    RRF formula: score = Σ(1 / (k + rank_i))
    where k is a constant (default 60) and rank_i is the rank in result list i

    Reference: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods"
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
                r.vector_score is not None and existing.vector_score < r.vector_score
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
            rrf_score += 1.0 / (owner.rrf_k + bm25_rank)
        if vec_rank is not None:
            rrf_score += 1.0 / (owner.rrf_k + vec_rank)

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


def weighted_fusion(
    owner, bm25_results: List[ChunkResult], vec_results: List[ChunkResult]
) -> List[ChunkResult]:
    """
    Weighted linear combination fusion.

    score = α * vector_score + (1-α) * bm25_score
    where α is the vector weight (default 0.6)

    Note: Requires score normalization since BM25 and vector scores
    have different ranges.
    """

    def normalize_scores(results: List[ChunkResult], score_attr: str):
        scores = [getattr(r, score_attr, 0) for r in results if getattr(r, score_attr)]
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
            owner.fusion_alpha * norm_vec + (1 - owner.fusion_alpha) * norm_bm25
        )
        chunk.fusion_method = "weighted"

    return list(all_chunks.values())


def apply_doc_continuity_boost(
    chunks: List[ChunkResult], alpha: float = 0.12
) -> List[ChunkResult]:
    """Favor documents that consistently appear at the top of the list."""
    if not chunks:
        return chunks

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

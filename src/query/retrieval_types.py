# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (facade re-export), query_service.py,
#             mcp_app.py, context_assembly.py, signal_pool.py
# =============================================================================
"""
Shared retrieval types and pure helpers.

Extracted from hybrid_retrieval.py to reduce coupling and provide a stable
import surface for ChunkResult and related types.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
    context_source: Optional[str] = (
        None  # Type: "sequential", "sibling", "parent_section", "shared_entities"
    )

    # Scoring metadata
    fusion_method: Optional[str] = None  # Method used for fusion
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None  # BM25/keyword score
    vector_rank: Optional[int] = None
    vector_score: Optional[float] = None  # Vector similarity score
    vector_score_kind: Optional[str] = None  # Similarity metric (cosine, dot, etc.)
    title_vec_score: Optional[float] = None
    doc_title_vec_score: Optional[float] = None  # Document-level title dense score
    doc_title_sparse_score: Optional[float] = (
        None  # Document-level title sparse/BM25 score
    )
    title_sparse_score: Optional[float] = None  # Section heading sparse/SPLADE score
    entity_vec_score: Optional[float] = None
    lexical_vec_score: Optional[float] = None

    # RELATED_TO graph signal scores (separate from entity graph_score)
    related_to_score: Optional[float] = None  # Blended RELATED_TO contribution
    related_to_edge_score: Optional[float] = (
        None  # Raw edge score (score_final/colbert/score)
    )
    related_to_prior_score: Optional[float] = None  # Combined prior signal
    related_to_source_doc: Optional[str] = (
        None  # Which related doc this chunk came from
    )
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

    # GLiNER entity metadata (Phase 4: Entity-aware retrieval)
    entity_metadata: Optional[Dict[str, Any]] = None
    entity_boost_applied: bool = False  # Whether entity boosting was applied

    # Phase 5: Structural metadata for query-type adaptive boosting
    # Fields: has_code, has_table, parent_path_depth, block_type, code_ratio
    structural_metadata: Optional[Dict[str, Any]] = None
    structural_boost_applied: bool = False

    # Full heading hierarchy path from Neo4j (e.g., "Configuration > S3 Backend > Buckets")
    # Hydrated from Neo4j for rerank candidates; used for reranker context enrichment
    parent_path_norm: Optional[str] = None

    # RRF debug: per-field contributions to fused_score (when rrf_debug_logging=true)
    # Structure: {"field_name": {"rank": int, "weight": float, "contribution": float}, ...}
    rrf_field_contributions: Optional[Dict[str, Dict[str, float]]] = None

    # Citation labels (order, title, level) derived from CitationUnits
    citation_labels: List[Tuple[int, str, int]] = field(default_factory=list)
    # Graph enrichment (Phase 2.3 legacy parity)
    graph_distance: int = 0
    graph_score: float = 0.0
    graph_path: Optional[List[str]] = None
    colbert_vector: Optional[List[List[float]]] = None
    connection_count: int = 0
    mention_count: int = 0

    def __post_init__(self):
        """Ensure required fields are populated."""
        if self.original_section_ids is None:
            self.original_section_ids = []


def _snapshot_top(chunks: List["ChunkResult"], n: int = 20) -> List[Dict[str, Any]]:
    """Capture a lightweight snapshot of the top N candidates for diagnostics."""
    return [
        {
            "chunk_id": c.chunk_id,
            "fused_score": round(c.fused_score or 0.0, 5),
            "rerank_score": (
                round(c.rerank_score, 5) if c.rerank_score is not None else None
            ),
            "reranker": getattr(c, "reranker", None),
            "is_expanded": c.is_expanded,
            "doc_tag": c.doc_tag,
        }
        for c in chunks[:n]
    ]


def _deduplicate_entity_metadata(
    entity_metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Deduplicate entity_values and entity_values_normalized in entity_metadata.

    GLiNER may extract the same entity multiple times from different spans.
    This cleans up the payload by deduplicating while preserving insertion order.

    Args:
        entity_metadata: Raw entity_metadata dict from Qdrant payload (or None)

    Returns:
        Deduplicated entity_metadata dict (or None if input was None)
    """
    if not entity_metadata:
        return entity_metadata

    # Deduplicate lists while preserving order (dict.fromkeys is order-preserving in Python 3.7+)
    entity_values = entity_metadata.get("entity_values") or []
    entity_values_normalized = entity_metadata.get("entity_values_normalized") or []

    return {
        "entity_types": entity_metadata.get("entity_types") or [],
        "entity_values": list(dict.fromkeys(entity_values)),
        "entity_values_normalized": list(dict.fromkeys(entity_values_normalized)),
        "entity_count": len(
            list(dict.fromkeys(entity_values))
        ),  # Update count to reflect deduped
    }


def dedup_chunk_results(
    results: List[ChunkResult],
    vector_weight: float = 0.7,
    graph_weight: float = 0.3,
    id_fn=None,
) -> List[ChunkResult]:
    """
    Standalone dedup function that merges duplicate ChunkResults.

    For each unique chunk, keeps the highest-scoring instance and merges
    the best signals from all duplicates, then recomputes fused_score
    using the provided weights.

    Args:
        results: List of ChunkResults (may contain duplicates)
        vector_weight: Weight for vector_score in fused calculation
        graph_weight: Weight for graph_score in fused calculation
        id_fn: Optional function to compute identity key; defaults to chunk_id

    Returns:
        Deduplicated list with merged scores
    """
    if not results:
        return []

    if id_fn is None:

        def id_fn(r):
            return r.chunk_id

    by_id: Dict[Any, List[ChunkResult]] = defaultdict(list)
    for r in results:
        by_id[id_fn(r)].append(r)

    deduped: List[ChunkResult] = []
    for rid, dups in by_id.items():
        if len(dups) == 1:
            deduped.append(dups[0])
            continue

        # Pick winner: prefer reranked chunks, then highest fused_score
        # This ensures rerank_score is preserved during dedup
        def _winner_key(x: ChunkResult) -> tuple:
            has_rerank = 1 if x.rerank_score is not None else 0
            fused = (
                x.fused_score if x.fused_score is not None else x.vector_score or 0.0
            )
            return (has_rerank, fused)

        winner = max(dups, key=_winner_key)

        # Merge best signals from all duplicates
        vector_scores = [w.vector_score for w in dups if w.vector_score is not None]
        graph_scores = [w.graph_score for w in dups if w.graph_score is not None]
        bm25_scores = [w.bm25_score for w in dups if w.bm25_score is not None]
        rerank_scores = [w.rerank_score for w in dups if w.rerank_score is not None]

        winner.vector_score = (
            max(vector_scores) if vector_scores else (winner.vector_score or 0.0)
        )
        winner.graph_score = (
            max(graph_scores) if graph_scores else (winner.graph_score or 0.0)
        )
        winner.bm25_score = (
            max(bm25_scores) if bm25_scores else (winner.bm25_score or 0.0)
        )
        # Preserve rerank_score from any duplicate (cross-encoder is authoritative)
        if rerank_scores:
            winner.rerank_score = max(rerank_scores)

        # Recompute fused_score with provided weights
        vscore = winner.vector_score or 0.0
        gscore = winner.graph_score or 0.0
        winner.fused_score = (vector_weight * vscore) + (graph_weight * gscore)

        deduped.append(winner)

    return deduped

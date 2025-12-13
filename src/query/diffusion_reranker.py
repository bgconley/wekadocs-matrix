"""
Graph-based diffusion reranking for retrieval candidates.

This module implements a lightweight PPR-style (Personalized PageRank) diffusion
over the candidate subgraph to enhance ranking using graph structure.

The diffusion propagates relevance scores through:
- Shared entity edges (chunks mentioning the same entities)
- NEXT_CHUNK edges (sequential adjacency)
- Hierarchy edges (parent-child relationships)

Key design principles:
1. Bounded: Only operates within the candidate pool (no graph blowup)
2. Seed-preserving: Blends diffused scores with original, not replacing
3. Lightweight: Pure Python/NumPy, no external graph libraries required
4. Tunable: Configurable damping, iterations, and blend weight

Usage:
    from src.query.diffusion_reranker import DiffusionReranker
    from src.query.graph_features import extract_candidate_features

    # Get subgraph features
    subgraph = extract_candidate_features(session, candidates, query_entities)

    # Apply diffusion reranking
    reranker = DiffusionReranker()
    reranked = reranker.rerank(subgraph)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import structlog

# Optional numpy import - fallback to pure Python if not available
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

logger = structlog.get_logger(__name__)


@dataclass
class RankedCandidate:
    """A candidate with original and diffused scores."""

    chunk_id: str
    original_score: float
    diffused_score: float
    final_score: float
    rank: int = 0


class DiffusionReranker:
    """
    Applies PPR-style diffusion reranking to candidate chunks.

    The algorithm:
    1. Build adjacency matrix from subgraph edges
    2. Initialize scores from vector search (personalization vector)
    3. Iterate: score = alpha * adj @ score + (1-alpha) * personalization
    4. Blend diffused scores with originals

    This is a lightweight implementation that:
    - Works on small candidate sets (10-100 chunks)
    - Uses sparse operations where possible
    - Falls back to pure Python if NumPy isn't available
    """

    def __init__(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
        blend_weight: float = 0.3,
        edge_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the diffusion reranker.

        Args:
            damping: PPR damping factor (0.85 is standard)
            max_iterations: Maximum diffusion iterations
            convergence_threshold: Stop when score change < threshold
            blend_weight: How much to blend diffused scores (0=ignore, 1=full)
            edge_weights: Weights by edge type (default: shared_entity=1.0, next_chunk=0.5)
        """
        self.damping = damping
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.blend_weight = blend_weight
        self.edge_weights = edge_weights or {
            "shared_entity": 1.0,
            "next_chunk": 0.5,
        }

    def rerank(
        self,
        subgraph,  # CandidateSubgraph from graph_features
        *,
        preserve_order: bool = False,
    ) -> List[RankedCandidate]:
        """
        Apply diffusion reranking to a candidate subgraph.

        Args:
            subgraph: CandidateSubgraph from graph_features module
            preserve_order: If True, keep original order, just update scores

        Returns:
            List of RankedCandidate sorted by final_score (descending)
        """
        if not subgraph.node_features:
            return []

        # Extract chunk IDs and build index
        chunk_ids = [f.chunk_id for f in subgraph.node_features]
        id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        n = len(chunk_ids)

        # Get original scores
        original_scores = [f.base_score for f in subgraph.node_features]

        # If no edges, skip diffusion
        if not subgraph.edges:
            return self._build_results(
                chunk_ids, original_scores, original_scores, preserve_order
            )

        # Build adjacency and run diffusion
        if HAS_NUMPY:
            diffused_scores = self._diffuse_numpy(
                n, id_to_idx, original_scores, subgraph.edges
            )
        else:
            diffused_scores = self._diffuse_pure_python(
                n, id_to_idx, original_scores, subgraph.edges
            )

        # Blend scores
        final_scores = [
            (1 - self.blend_weight) * orig + self.blend_weight * diff
            for orig, diff in zip(original_scores, diffused_scores)
        ]

        return self._build_results(
            chunk_ids, original_scores, final_scores, preserve_order
        )

    def _diffuse_numpy(
        self,
        n: int,
        id_to_idx: Dict[str, int],
        original_scores: List[float],
        edges: List,  # List[CandidateEdge]
    ) -> List[float]:
        """Run PPR diffusion using NumPy for efficiency."""
        # Build weighted adjacency matrix
        adj = np.zeros((n, n), dtype=np.float64)
        for edge in edges:
            src_idx = id_to_idx.get(edge.src)
            dst_idx = id_to_idx.get(edge.dst)
            if src_idx is not None and dst_idx is not None:
                weight = edge.weight * self.edge_weights.get(edge.edge_type, 1.0)
                adj[src_idx, dst_idx] += weight
                adj[dst_idx, src_idx] += weight  # Undirected

        # Normalize by row (make transition matrix)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid div by zero
        adj = adj / row_sums

        # Initialize personalization vector (normalized original scores)
        personalization = np.array(original_scores, dtype=np.float64)
        p_sum = personalization.sum()
        if p_sum > 0:
            personalization = personalization / p_sum
        else:
            personalization = np.ones(n, dtype=np.float64) / n

        # Run PPR iterations
        scores = personalization.copy()
        for iteration in range(self.max_iterations):
            old_scores = scores.copy()
            scores = (
                self.damping * (adj @ scores) + (1 - self.damping) * personalization
            )

            # Check convergence
            diff = np.abs(scores - old_scores).sum()
            if diff < self.convergence_threshold:
                logger.debug("diffusion_converged", iteration=iteration + 1, diff=diff)
                break

        # Scale back to original score range
        if scores.sum() > 0:
            scale = sum(original_scores) / scores.sum()
            scores = scores * scale

        return scores.tolist()

    def _diffuse_pure_python(
        self,
        n: int,
        id_to_idx: Dict[str, int],
        original_scores: List[float],
        edges: List,  # List[CandidateEdge]
    ) -> List[float]:
        """Pure Python fallback for PPR diffusion."""
        # Build adjacency as dict of dicts for sparse representation
        adj: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
        for edge in edges:
            src_idx = id_to_idx.get(edge.src)
            dst_idx = id_to_idx.get(edge.dst)
            if src_idx is not None and dst_idx is not None:
                weight = edge.weight * self.edge_weights.get(edge.edge_type, 1.0)
                adj[src_idx][dst_idx] = adj[src_idx].get(dst_idx, 0) + weight
                adj[dst_idx][src_idx] = adj[dst_idx].get(src_idx, 0) + weight

        # Normalize by row
        for i in range(n):
            row_sum = sum(adj[i].values())
            if row_sum > 0:
                for j in adj[i]:
                    adj[i][j] /= row_sum

        # Initialize personalization
        p_sum = sum(original_scores)
        if p_sum > 0:
            personalization = [s / p_sum for s in original_scores]
        else:
            personalization = [1.0 / n] * n

        # Run PPR iterations
        scores = personalization.copy()
        for iteration in range(self.max_iterations):
            old_scores = scores.copy()

            # Compute new scores
            new_scores = [0.0] * n
            for i in range(n):
                # Diffusion from neighbors
                for j, weight in adj[i].items():
                    new_scores[i] += weight * old_scores[j]
                # Apply damping and personalization
                new_scores[i] = (
                    self.damping * new_scores[i]
                    + (1 - self.damping) * personalization[i]
                )

            scores = new_scores

            # Check convergence
            diff = sum(abs(s - o) for s, o in zip(scores, old_scores))
            if diff < self.convergence_threshold:
                break

        # Scale back to original score range
        score_sum = sum(scores)
        if score_sum > 0:
            scale = sum(original_scores) / score_sum
            scores = [s * scale for s in scores]

        return scores

    def _build_results(
        self,
        chunk_ids: List[str],
        original_scores: List[float],
        final_scores: List[float],
        preserve_order: bool,
    ) -> List[RankedCandidate]:
        """Build sorted result list."""
        results = [
            RankedCandidate(
                chunk_id=cid,
                original_score=orig,
                diffused_score=final,
                final_score=final,
            )
            for cid, orig, final in zip(chunk_ids, original_scores, final_scores)
        ]

        if not preserve_order:
            results.sort(key=lambda x: x.final_score, reverse=True)

        for i, r in enumerate(results):
            r.rank = i + 1

        return results


def apply_diffusion_rerank(
    subgraph,  # CandidateSubgraph from graph_features
    *,
    damping: float = 0.85,
    blend_weight: float = 0.3,
) -> List[RankedCandidate]:
    """
    Convenience function to apply diffusion reranking.

    Args:
        subgraph: CandidateSubgraph from graph_features module
        damping: PPR damping factor
        blend_weight: How much to blend diffused scores

    Returns:
        List of RankedCandidate sorted by final_score
    """
    reranker = DiffusionReranker(damping=damping, blend_weight=blend_weight)
    return reranker.rerank(subgraph)


def rerank_with_graph_features(
    session,
    candidates: List[Dict],
    query_entities: Optional[List[str]] = None,
    *,
    damping: float = 0.85,
    blend_weight: float = 0.3,
) -> Tuple[List[RankedCandidate], Dict]:
    """
    End-to-end function: extract features and apply diffusion.

    This combines graph_features extraction with diffusion reranking
    for convenience.

    Args:
        session: Neo4j session
        candidates: List of dicts with "chunk_id" and "score"
        query_entities: Entity names from query
        damping: PPR damping factor
        blend_weight: Blend weight for diffused scores

    Returns:
        Tuple of (reranked candidates, subgraph stats)
    """
    from src.query.graph_features import extract_candidate_features

    # Extract features and edges
    subgraph = extract_candidate_features(
        session,
        candidates,
        query_entities=query_entities,
        return_edges=True,
    )

    # Apply diffusion
    reranker = DiffusionReranker(damping=damping, blend_weight=blend_weight)
    reranked = reranker.rerank(subgraph)

    return reranked, subgraph.stats

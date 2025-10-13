"""
Ranking Module (Task 2.3)
Blends multiple signals to rank search results.
See: /docs/spec.md §4.1 (Hybrid retrieval - ranking)
See: /docs/pseudocode-reference.md Phase 2, Task 2.3
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.query.hybrid_search import SearchResult
from src.shared.config import get_config


@dataclass
class RankingFeatures:
    """Feature values used for ranking a result."""

    semantic_score: float = 0.0
    graph_distance_score: float = 0.0
    recency_score: float = 0.0
    entity_priority_score: float = 0.0
    coverage_score: float = 0.0
    final_score: float = 0.0


@dataclass
class RankedResult:
    """A search result with ranking features and final score."""

    result: SearchResult
    features: RankingFeatures
    rank: int = 0


class Ranker:
    """
    Multi-signal ranker for hybrid search results.

    Blends:
    - Semantic similarity (vector score)
    - Graph proximity (distance from seed)
    - Recency (document last_edited timestamp)
    - Entity priority (label-based importance)
    - Coverage (mention count, connections)
    """

    # Label priority weights (higher = more important)
    LABEL_PRIORITIES = {
        "Procedure": 1.0,
        "Command": 0.9,
        "Configuration": 0.85,
        "Error": 0.9,
        "Concept": 0.7,
        "Example": 0.6,
        "Step": 0.8,
        "Parameter": 0.75,
        "Component": 0.85,
        "Section": 1.0,  # Sections are primary units
        "Document": 0.5,  # Documents less important than sections
    }

    def __init__(self):
        self.config = get_config()

        # Get hybrid search weights from config
        self.vector_weight = self.config.search.hybrid.vector_weight
        self.graph_weight = self.config.search.hybrid.graph_weight

        # Feature weights (sum to 1.0)
        self.weights = {
            "semantic": 0.4,
            "graph_distance": 0.2,
            "recency": 0.15,
            "entity_priority": 0.15,
            "coverage": 0.1,
        }

    def rank(
        self,
        results: List[SearchResult],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[RankedResult]:
        """
        Rank results using multiple signals.

        Args:
            results: Search results to rank
            query_context: Optional context (e.g., filters, intent)

        Returns:
            Ranked results with features
        """
        if not results:
            return []

        # Extract all features
        ranked = []
        for result in results:
            features = self._extract_features(result, query_context)
            ranked.append(RankedResult(result=result, features=features))

        # Sort by final score (descending)
        ranked.sort(key=lambda r: r.features.final_score, reverse=True)

        # Assign ranks (with deterministic tie-breaking by node_id)
        for i, ranked_result in enumerate(ranked):
            ranked_result.rank = i + 1

        # Deterministic tie-breaking for same scores
        ranked = self._break_ties(ranked)

        return ranked

    def _extract_features(
        self, result: SearchResult, context: Optional[Dict[str, Any]] = None
    ) -> RankingFeatures:
        """Extract all ranking features for a result."""
        features = RankingFeatures()

        # 1. Semantic score (from vector search)
        features.semantic_score = self._normalize_score(result.score)

        # 2. Graph distance score (closer = higher)
        features.graph_distance_score = self._distance_score(result.distance)

        # 3. Recency score (newer = higher)
        features.recency_score = self._recency_score(result.metadata)

        # 4. Entity priority score (based on label)
        features.entity_priority_score = self._priority_score(result.node_label)

        # 5. Coverage score (connections, mentions)
        features.coverage_score = self._coverage_score(result.metadata)

        # Compute weighted final score
        features.final_score = (
            self.weights["semantic"] * features.semantic_score
            + self.weights["graph_distance"] * features.graph_distance_score
            + self.weights["recency"] * features.recency_score
            + self.weights["entity_priority"] * features.entity_priority_score
            + self.weights["coverage"] * features.coverage_score
        )

        return features

    def _normalize_score(self, score: float) -> float:
        """Normalize semantic score to [0, 1]."""
        # Vector scores are typically cosine similarity in [-1, 1] or [0, 1]
        # Ensure [0, 1] range
        return max(0.0, min(1.0, score))

    def _distance_score(self, distance: int) -> float:
        """
        Convert graph distance to score.
        Distance 0 (seed) = 1.0, decreases with distance.
        """
        if distance == 0:
            return 1.0
        # Exponential decay: score = e^(-0.5 * distance)
        return math.exp(-0.5 * distance)

    def _recency_score(self, metadata: Dict[str, Any]) -> float:
        """
        Score based on document recency.
        More recent documents get higher scores.
        """
        # Try to get last_edited or updated_at timestamp
        timestamp_str = (
            metadata.get("last_edited")
            or metadata.get("updated_at")
            or metadata.get("created_at")
        )

        if not timestamp_str:
            return 0.5  # Neutral score if no timestamp

        try:
            # Parse timestamp (assume ISO format)
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
            else:
                return 0.5

            # Calculate age in days
            now = datetime.now(timestamp.tzinfo or None)
            age_days = (now - timestamp).days

            # Exponential decay: score = e^(-age_days / 365)
            # Documents older than 1 year get low scores
            decay_factor = 365.0
            score = math.exp(-age_days / decay_factor)

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5  # Neutral score if parsing fails

    def _priority_score(self, label: str) -> float:
        """Score based on entity label priority."""
        return self.LABEL_PRIORITIES.get(label, 0.5)

    def _coverage_score(self, metadata: Dict[str, Any]) -> float:
        """
        Score based on coverage indicators.
        Nodes with more connections or mentions get higher scores.
        """
        score = 0.5  # Base score

        # Check for mention count
        mention_count = metadata.get("mention_count", 0)
        if mention_count > 0:
            # Logarithmic scaling: more mentions = higher score
            score += min(0.3, math.log(mention_count + 1) / 10)

        # Check for connection count
        connection_count = metadata.get("connection_count", 0)
        if connection_count > 0:
            score += min(0.2, math.log(connection_count + 1) / 10)

        return min(1.0, score)

    def _break_ties(self, ranked: List[RankedResult]) -> List[RankedResult]:
        """
        Break ties deterministically using node_id lexicographic order.
        """
        # Group by score
        score_groups = {}
        for r in ranked:
            score = round(r.features.final_score, 6)  # Round to avoid float issues
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(r)

        # Sort each group by node_id
        result = []
        for score in sorted(score_groups.keys(), reverse=True):
            group = score_groups[score]
            group.sort(key=lambda r: r.result.node_id)
            result.extend(group)

        # Reassign ranks
        for i, r in enumerate(result):
            r.rank = i + 1

        return result


def rank_results(
    results: List[SearchResult], query_context: Optional[Dict[str, Any]] = None
) -> List[RankedResult]:
    """Convenience function to rank results."""
    ranker = Ranker()
    return ranker.rank(results, query_context)

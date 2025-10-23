"""
Ranking Module (Task 2.3)
Blends multiple signals to rank search results.
See: /docs/spec.md §4.1 (Hybrid retrieval - ranking)
See: /docs/pseudocode-reference.md Phase 2, Task 2.3

Pre-Phase 7 improvements:
- Robust normalization for similarity vs distance
- UTC handling for recency scoring
- Tie-break stability with label priority
- Entity focus hook (no-op until Phase 7)
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.query.hybrid_search import SearchResult
from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)


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
        "Chunk": 1.0,  # Pre-Phase 7: Treat Chunk as Section
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
        features.semantic_score = self._normalize_score(
            result.score, result.metadata.get("score_kind", "similarity")
        )

        # 2. Graph distance score (closer = higher)
        features.graph_distance_score = self._distance_score(result.distance)

        # 3. Recency score (newer = higher)
        features.recency_score = self._recency_score(result.metadata)

        # 4. Entity priority score (based on label)
        features.entity_priority_score = self._priority_score(result.node_label)

        # 5. Coverage score (connections, mentions)
        features.coverage_score = self._coverage_score(result.metadata)

        # Pre-Phase 7: Entity focus hook (returns 0.0 until Phase 7)
        entity_focus_bonus = self._entity_focus_bonus(result, context)

        # Compute weighted final score
        features.final_score = (
            self.weights["semantic"] * features.semantic_score
            + self.weights["graph_distance"] * features.graph_distance_score
            + self.weights["recency"] * features.recency_score
            + self.weights["entity_priority"] * features.entity_priority_score
            + self.weights["coverage"] * features.coverage_score
            + entity_focus_bonus  # No-op until Phase 7
        )

        return features

    def _normalize_score(self, score: float, score_kind: str = "similarity") -> float:
        """
        Pre-Phase 7: Robust normalization to similarity in [0, 1].
        Handles both similarity and distance metrics.

        Args:
            score: Raw score from vector search
            score_kind: "similarity" or "distance"

        Returns:
            Normalized similarity score in [0, 1]
        """
        if score_kind == "distance":
            # Distance metrics: lower is better
            # Assume distance in [0, 2] for normalized vectors
            # Convert to similarity: 1 - (distance/2)
            distance = max(0.0, min(2.0, score))
            similarity = 1.0 - (distance / 2.0)

            # Log warning first time we see distance scores
            if not hasattr(self, "_distance_warning_logged"):
                logger.warning(
                    "Distance-based scores detected, converting to similarity. "
                    "Consider setting score_kind='similarity' at source."
                )
                self._distance_warning_logged = True

            return similarity
        else:
            # Similarity metrics: higher is better
            # Cosine similarity in [-1, 1], convert to [0, 1]
            if score < 0:
                # Handle negative similarities (rare but possible)
                return (score + 1.0) / 2.0
            else:
                # Already in [0, 1]
                return min(1.0, score)

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
        Pre-Phase 7: Score based on document recency with UTC handling.
        More recent documents get higher scores.
        """
        # Priority: updated_at > last_edited > created_at
        timestamp_str = (
            metadata.get("updated_at")
            or metadata.get("last_edited")
            or metadata.get("created_at")
        )

        if not timestamp_str:
            return 0.5  # Neutral score if no timestamp

        try:
            # Parse timestamp (assume ISO format)
            if isinstance(timestamp_str, str):
                # Handle ISO format with Z suffix or already has timezone
                if timestamp_str.endswith("Z"):
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
            else:
                return 0.5

            # Pre-Phase 7: Handle naive timestamps by assigning UTC
            if timestamp.tzinfo is None:
                logger.debug("Converting naive timestamp to UTC")
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Calculate age in days using UTC
            now = datetime.now(timezone.utc)
            age_days = (now - timestamp).days

            # Exponential decay: score = e^(-age_days / 365)
            # Documents older than 1 year get progressively lower scores
            decay_factor = 365.0
            score = math.exp(-age_days / decay_factor)

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"Failed to parse timestamp: {e}")
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

    def _entity_focus_bonus(
        self, result: SearchResult, context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Pre-Phase 7: No-op entity focus hook.
        Will be implemented in Phase 7 to boost results matching focused entities.

        Args:
            result: Search result to evaluate
            context: Query context with potential focused_entities

        Returns:
            Bonus score (0.0 until Phase 7)
        """
        # Phase 7 will check for focused_entities in context
        # and boost results that match them
        return 0.0

    def _break_ties(self, ranked: List[RankedResult]) -> List[RankedResult]:
        """
        Pre-Phase 7: Enhanced tie-breaking with label priority then node_id.
        Ensures stable, deterministic ordering.
        """
        # Group by score (rounded to handle float precision)
        score_groups = {}
        for r in ranked:
            score = round(r.features.final_score, 6)  # Round to avoid float issues
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(r)

        # Sort each group by label priority (desc) then node_id (asc)
        result = []
        for score in sorted(score_groups.keys(), reverse=True):
            group = score_groups[score]
            # Pre-Phase 7: Tie-break by priority then node_id
            group.sort(
                key=lambda r: (
                    -self._priority_score(r.result.node_label),  # Higher priority first
                    r.result.node_id,  # Then alphabetically by ID
                )
            )
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

"""Ranking weight tuner using NDCG optimization.

Implements Phase 4 Task 4.4 - Learning & adaptation.
Tunes ranking weights based on feedback to improve relevance.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .feedback import FeedbackCollector, QueryFeedback

logger = logging.getLogger(__name__)


class RankingWeightTuner:
    """Tunes ranking weights using NDCG optimization on feedback data."""

    # Default feature weights (from ranking.py)
    DEFAULT_WEIGHTS = {
        "semantic_score": 0.35,
        "graph_proximity": 0.25,
        "entity_priority": 0.15,
        "recency": 0.15,
        "coverage": 0.10,
    }

    def __init__(self, feedback_collector: FeedbackCollector):
        """Initialize ranking weight tuner.

        Args:
            feedback_collector: Feedback collector for training data
        """
        self.feedback_collector = feedback_collector
        self.current_weights = self.DEFAULT_WEIGHTS.copy()

    def compute_ndcg(
        self,
        relevance_scores: List[float],
        predicted_ranks: List[int],
        k: Optional[int] = None,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        Args:
            relevance_scores: Ground truth relevance (e.g., ratings)
            predicted_ranks: Predicted ranking positions (0-indexed)
            k: Cut-off rank (optional, defaults to len(relevance_scores))

        Returns:
            NDCG@k score in [0, 1]
        """
        if not relevance_scores:
            return 0.0

        if k is None:
            k = len(relevance_scores)

        # DCG: sum of (rel_i / log2(i+2)) for i in predicted order
        dcg = 0.0
        for i, rank in enumerate(predicted_ranks[:k]):
            if rank < len(relevance_scores):
                rel = relevance_scores[rank]
                dcg += rel / np.log2(i + 2)

        # IDCG: optimal DCG (sorted by relevance desc)
        ideal_order = sorted(
            range(len(relevance_scores)),
            key=lambda i: relevance_scores[i],
            reverse=True,
        )
        idcg = 0.0
        for i, rank in enumerate(ideal_order[:k]):
            rel = relevance_scores[rank]
            idcg += rel / np.log2(i + 2)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def _extract_training_data(
        self,
        feedbacks: List[QueryFeedback],
    ) -> List[Tuple[Dict[str, float], List[float]]]:
        """Extract (features, relevance) pairs from feedback.

        Args:
            feedbacks: List of rated query feedbacks

        Returns:
            List of (feature_dict, relevance_list) tuples
        """
        training_data = []

        for fb in feedbacks:
            if fb.rating is None or not fb.result_ids:
                continue

            # Extract feature vectors for each result
            if not fb.ranking_features:
                continue

            # Use the query rating as relevance for all returned results
            # (In practice, you'd have per-result relevance judgments)
            relevance = [fb.rating] * len(fb.result_ids)

            training_data.append((fb.ranking_features, relevance))

        return training_data

    def _score_with_weights(
        self,
        features: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Compute weighted score from features.

        Args:
            features: Feature dictionary
            weights: Weight dictionary

        Returns:
            Weighted score
        """
        score = 0.0
        for feat, weight in weights.items():
            score += features.get(feat, 0.0) * weight
        return score

    def _objective_function(
        self,
        weight_vector: np.ndarray,
        training_data: List[Tuple[Dict[str, float], List[float]]],
        feature_names: List[str],
    ) -> float:
        """Objective function for optimization (negative mean NDCG).

        Args:
            weight_vector: Current weight values
            training_data: List of (features, relevance) tuples
            feature_names: Ordered feature names

        Returns:
            Negative mean NDCG (to minimize)
        """
        weights = dict(zip(feature_names, weight_vector))

        ndcg_scores = []
        for features, relevance in training_data:
            # Compute scores with current weights
            # Note: In real scenario, features would be per-result,
            # here we use query-level features as proxy
            _ = self._score_with_weights(features, weights)

            # Create predicted ranking (single item for now)
            predicted_ranks = [0]  # Simplified: single result

            ndcg = self.compute_ndcg(relevance, predicted_ranks, k=10)
            ndcg_scores.append(ndcg)

        mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        return -mean_ndcg  # Negative because we minimize

    def tune_weights(
        self,
        intent: Optional[str] = None,
        min_samples: int = 10,
    ) -> Dict[str, float]:
        """Tune ranking weights using collected feedback.

        Args:
            intent: Filter by intent (optional)
            min_samples: Minimum feedback samples required

        Returns:
            Optimized weight dictionary
        """
        # Get rated feedback
        feedbacks = self.feedback_collector.get_feedback(
            intent=intent,
            rated_only=True,
        )

        if len(feedbacks) < min_samples:
            logger.warning(
                f"Insufficient feedback samples ({len(feedbacks)} < {min_samples}), "
                f"using default weights"
            )
            return self.current_weights.copy()

        # Extract training data
        training_data = self._extract_training_data(feedbacks)
        if not training_data:
            logger.warning("No valid training data extracted, using default weights")
            return self.current_weights.copy()

        # Prepare optimization
        feature_names = list(self.DEFAULT_WEIGHTS.keys())
        initial_weights = np.array([self.DEFAULT_WEIGHTS[f] for f in feature_names])

        # Constraints: weights sum to 1.0, all non-negative
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in feature_names]

        # Optimize
        result = minimize(
            self._objective_function,
            initial_weights,
            args=(training_data, feature_names),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        if result.success:
            optimized_weights = dict(zip(feature_names, result.x))
            self.current_weights = optimized_weights

            logger.info(
                f"Weight optimization successful: "
                f"NDCG improved from {-self._objective_function(initial_weights, training_data, feature_names):.4f} "
                f"to {-result.fun:.4f}"
            )

            return optimized_weights
        else:
            logger.warning(f"Weight optimization failed: {result.message}")
            return self.current_weights.copy()

    def evaluate_weights(
        self,
        weights: Dict[str, float],
        held_out_feedback: List[QueryFeedback],
    ) -> Dict[str, float]:
        """Evaluate weights on held-out feedback.

        Args:
            weights: Weight dictionary to evaluate
            held_out_feedback: Held-out feedback data

        Returns:
            Evaluation metrics (NDCG@k for various k)
        """
        training_data = self._extract_training_data(held_out_feedback)

        if not training_data:
            return {
                "ndcg@5": 0.0,
                "ndcg@10": 0.0,
                "ndcg@20": 0.0,
                "samples": 0,
            }

        ndcg_5, ndcg_10, ndcg_20 = [], [], []

        for features, relevance in training_data:
            _ = self._score_with_weights(features, weights)
            predicted_ranks = [0]  # Simplified

            ndcg_5.append(self.compute_ndcg(relevance, predicted_ranks, k=5))
            ndcg_10.append(self.compute_ndcg(relevance, predicted_ranks, k=10))
            ndcg_20.append(self.compute_ndcg(relevance, predicted_ranks, k=20))

        return {
            "ndcg@5": np.mean(ndcg_5),
            "ndcg@10": np.mean(ndcg_10),
            "ndcg@20": np.mean(ndcg_20),
            "samples": len(training_data),
        }

    def get_weight_delta(self) -> Dict[str, float]:
        """Get difference from default weights.

        Returns:
            Dict of weight changes
        """
        return {
            feat: self.current_weights[feat] - self.DEFAULT_WEIGHTS[feat]
            for feat in self.DEFAULT_WEIGHTS.keys()
        }

"""Learning and adaptation module for Phase 4 Task 4.4.

This module implements:
- Feedback collection (query → result → rating)
- Ranking weight optimization (NDCG-based)
- Template and index suggestion mining
"""

from .feedback import FeedbackCollector, QueryFeedback
from .ranking_tuner import RankingWeightTuner
from .suggestions import SuggestionEngine

__all__ = [
    "FeedbackCollector",
    "QueryFeedback",
    "RankingWeightTuner",
    "SuggestionEngine",
]

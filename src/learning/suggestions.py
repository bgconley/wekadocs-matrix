"""Template and index suggestion engine.

Implements Phase 4 Task 4.4 - Learning & adaptation.
Mines query patterns to suggest new templates and performance indexes.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from neo4j import Driver

from .feedback import FeedbackCollector

logger = logging.getLogger(__name__)


@dataclass
class TemplateSuggestion:
    """Suggested query template."""

    intent: str
    pattern: str
    frequency: int
    avg_rating: Optional[float]
    example_queries: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "intent": self.intent,
            "pattern": self.pattern,
            "frequency": self.frequency,
            "avg_rating": self.avg_rating,
            "example_queries": self.example_queries,
            "confidence": self.confidence,
        }


@dataclass
class IndexSuggestion:
    """Suggested database index."""

    label: str
    property: str
    frequency: int
    avg_query_time_ms: float
    estimated_benefit: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "label": self.label,
            "property": self.property,
            "frequency": self.frequency,
            "avg_query_time_ms": self.avg_query_time_ms,
            "estimated_benefit": self.estimated_benefit,
        }


class SuggestionEngine:
    """Mines usage patterns to suggest templates and indexes."""

    def __init__(
        self,
        driver: Driver,
        feedback_collector: FeedbackCollector,
    ):
        """Initialize suggestion engine.

        Args:
            driver: Neo4j driver for index analysis
            feedback_collector: Feedback collector for pattern mining
        """
        self.driver = driver
        self.feedback_collector = feedback_collector

    def _extract_cypher_patterns(
        self,
        cypher_queries: List[str],
    ) -> List[str]:
        """Extract structural patterns from Cypher queries.

        Args:
            cypher_queries: List of Cypher query strings

        Returns:
            List of normalized patterns
        """
        patterns = []

        for query in cypher_queries:
            # Normalize: replace literals with placeholders
            pattern = query

            # Replace string literals
            pattern = re.sub(r"'[^']*'", "$STR", pattern)
            pattern = re.sub(r'"[^"]*"', "$STR", pattern)

            # Replace numbers
            pattern = re.sub(r"\b\d+\b", "$NUM", pattern)

            # Replace variable names but keep labels
            # Match (var:Label) -> (VAR:Label)
            pattern = re.sub(r"\((\w+):", r"($VAR:", pattern)
            pattern = re.sub(r"-\[(\w+):", r"-[$VAR:", pattern)

            # Normalize whitespace
            pattern = " ".join(pattern.split())

            patterns.append(pattern)

        return patterns

    def suggest_templates(
        self,
        min_frequency: int = 5,
        min_rating: float = 0.6,
        max_suggestions: int = 10,
    ) -> List[TemplateSuggestion]:
        """Suggest new query templates based on common patterns.

        Args:
            min_frequency: Minimum pattern frequency
            min_rating: Minimum average rating
            max_suggestions: Maximum suggestions to return

        Returns:
            List of template suggestions
        """
        # Get all feedback
        feedbacks = self.feedback_collector.get_feedback(limit=10000)

        # Group by intent
        intent_queries = defaultdict(list)
        for fb in feedbacks:
            if fb.cypher_query:
                intent_queries[fb.intent].append(fb)

        suggestions = []

        for intent, fbs in intent_queries.items():
            # Extract patterns
            cypher_queries = [fb.cypher_query for fb in fbs]
            patterns = self._extract_cypher_patterns(cypher_queries)

            # Count patterns
            pattern_counts = Counter(patterns)

            # Analyze each pattern
            for pattern, freq in pattern_counts.most_common():
                if freq < min_frequency:
                    continue

                # Get feedback for this pattern
                pattern_fbs = [fb for fb, p in zip(fbs, patterns) if p == pattern]

                # Compute metrics
                ratings = [fb.rating for fb in pattern_fbs if fb.rating is not None]
                avg_rating = sum(ratings) / len(ratings) if ratings else None

                if avg_rating is not None and avg_rating < min_rating:
                    continue

                # Collect example queries
                examples = [fb.query_text for fb in pattern_fbs[:3]]

                # Compute confidence (based on frequency and rating)
                confidence = min(1.0, freq / (min_frequency * 2))
                if avg_rating is not None:
                    confidence *= avg_rating

                suggestions.append(
                    TemplateSuggestion(
                        intent=intent,
                        pattern=pattern,
                        frequency=freq,
                        avg_rating=avg_rating,
                        example_queries=examples,
                        confidence=confidence,
                    )
                )

        # Sort by confidence and frequency
        suggestions.sort(key=lambda s: (s.confidence, s.frequency), reverse=True)

        return suggestions[:max_suggestions]

    def _extract_property_accesses(
        self,
        cypher_query: str,
    ) -> List[tuple[str, str]]:
        """Extract (label, property) pairs from Cypher query.

        Args:
            cypher_query: Cypher query string

        Returns:
            List of (label, property) tuples
        """
        accesses = []

        # Match patterns like: (n:Label {prop: ...}) or (n:Label) WHERE n.prop

        # Pattern 1: (var:Label {prop: ...})
        matches = re.findall(r"\((\w+):(\w+)\s*\{([^}]+)\}", cypher_query)
        for var, label, props in matches:
            # Extract property names
            prop_names = re.findall(r"(\w+)\s*:", props)
            for prop in prop_names:
                accesses.append((label, prop))

        # Pattern 2: WHERE n.prop or n.prop = ...
        matches = re.findall(r"WHERE\s+(\w+)\.(\w+)", cypher_query)
        for var, prop in matches:
            # Try to find the label for this variable
            label_match = re.search(rf"\({var}:(\w+)", cypher_query)
            if label_match:
                label = label_match.group(1)
                accesses.append((label, prop))

        # Pattern 3: RETURN n.prop
        matches = re.findall(r"RETURN.*?(\w+)\.(\w+)", cypher_query, re.IGNORECASE)
        for var, prop in matches:
            label_match = re.search(rf"\({var}:(\w+)", cypher_query)
            if label_match:
                label = label_match.group(1)
                accesses.append((label, prop))

        return accesses

    def suggest_indexes(
        self,
        min_frequency: int = 10,
        min_query_time_ms: float = 100.0,
        max_suggestions: int = 10,
    ) -> List[IndexSuggestion]:
        """Suggest database indexes based on slow queries.

        Args:
            min_frequency: Minimum property access frequency
            min_query_time_ms: Minimum query time to consider
            max_suggestions: Maximum suggestions to return

        Returns:
            List of index suggestions
        """
        # Get feedback with slow queries
        feedbacks = self.feedback_collector.get_feedback(limit=10000)
        slow_feedbacks = [
            fb
            for fb in feedbacks
            if fb.execution_time_ms and fb.execution_time_ms >= min_query_time_ms
        ]

        if not slow_feedbacks:
            logger.info("No slow queries found for index analysis")
            return []

        # Extract property accesses
        property_usage = defaultdict(lambda: {"count": 0, "total_time_ms": 0.0})

        for fb in slow_feedbacks:
            accesses = self._extract_property_accesses(fb.cypher_query)
            for label, prop in accesses:
                key = (label, prop)
                property_usage[key]["count"] += 1
                property_usage[key]["total_time_ms"] += fb.execution_time_ms

        # Get existing indexes
        existing_indexes = self._get_existing_indexes()

        # Generate suggestions
        suggestions = []
        for (label, prop), stats in property_usage.items():
            freq = stats["count"]
            if freq < min_frequency:
                continue

            # Skip if index already exists
            if (label, prop) in existing_indexes:
                continue

            avg_time = stats["total_time_ms"] / freq

            # Estimate benefit (heuristic: higher frequency + longer time = more benefit)
            estimated_benefit = (freq / min_frequency) * (avg_time / min_query_time_ms)

            suggestions.append(
                IndexSuggestion(
                    label=label,
                    property=prop,
                    frequency=freq,
                    avg_query_time_ms=avg_time,
                    estimated_benefit=estimated_benefit,
                )
            )

        # Sort by estimated benefit
        suggestions.sort(key=lambda s: s.estimated_benefit, reverse=True)

        return suggestions[:max_suggestions]

    def _get_existing_indexes(self) -> Set[tuple[str, str]]:
        """Get existing (label, property) indexes from database.

        Returns:
            Set of (label, property) tuples
        """
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES YIELD labelsOrTypes, properties")

            indexes = set()
            for record in result:
                labels = record["labelsOrTypes"] or []
                properties = record["properties"] or []

                # Create (label, property) pairs
                for label in labels:
                    for prop in properties:
                        indexes.add((label, prop))

            return indexes

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive suggestion report.

        Returns:
            Dict with template and index suggestions
        """
        template_suggestions = self.suggest_templates()
        index_suggestions = self.suggest_indexes()

        stats = self.feedback_collector.get_statistics()

        return {
            "statistics": stats,
            "template_suggestions": [s.to_dict() for s in template_suggestions],
            "index_suggestions": [s.to_dict() for s in index_suggestions],
            "summary": {
                "total_templates": len(template_suggestions),
                "total_indexes": len(index_suggestions),
                "high_confidence_templates": len(
                    [s for s in template_suggestions if s.confidence > 0.7]
                ),
            },
        }

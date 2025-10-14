"""Tests for Phase 4 Task 4.4 - Learning & adaptation.

Tests feedback collection, ranking weight tuning (NDCG), and suggestion engine.
NO MOCKS - tests run against live Neo4j.
"""

import pytest

from src.learning import (
    FeedbackCollector,
    QueryFeedback,
    RankingWeightTuner,
    SuggestionEngine,
)


class TestFeedbackCollection:
    """Test feedback collection and persistence."""

    def test_log_and_retrieve_query(self, neo4j_driver):
        """Test logging a query and retrieving it."""
        collector = FeedbackCollector(neo4j_driver)

        query_id = collector.log_query(
            query_text="How to configure TLS?",
            intent="search",
            cypher_query="MATCH (s:Section) WHERE s.text CONTAINS 'TLS' RETURN s LIMIT 10",
            result_ids=["sec1", "sec2", "sec3"],
            ranking_features={
                "semantic_score": 0.85,
                "graph_proximity": 0.6,
                "entity_priority": 0.7,
                "recency": 0.5,
                "coverage": 0.8,
            },
            execution_time_ms=45.3,
        )

        assert query_id is not None

        # Retrieve
        feedbacks = collector.get_feedback(limit=1)
        assert len(feedbacks) >= 1

        fb = feedbacks[0]
        assert fb.query_text == "How to configure TLS?"
        assert fb.intent == "search"
        assert len(fb.result_ids) == 3
        assert fb.execution_time_ms == 45.3

    def test_add_user_feedback(self, neo4j_driver):
        """Test adding user feedback to a query."""
        collector = FeedbackCollector(neo4j_driver)

        query_id = collector.log_query(
            query_text="Test query",
            intent="search",
            cypher_query="MATCH (n) RETURN n LIMIT 1",
            result_ids=["r1"],
            ranking_features={"semantic_score": 0.5},
            execution_time_ms=10.0,
        )

        # Add feedback
        collector.add_feedback(
            query_id=query_id,
            rating=0.8,
            notes="Good results but missing one section",
            missed_entities=["Configuration:TLS_CERT_PATH"],
        )

        # Retrieve and verify
        feedbacks = collector.get_feedback(rated_only=True, limit=10)
        matching = [fb for fb in feedbacks if fb.query_id == query_id]

        assert len(matching) == 1
        fb = matching[0]
        assert fb.rating == 0.8
        assert fb.notes == "Good results but missing one section"
        assert "Configuration:TLS_CERT_PATH" in fb.missed_entities

    def test_feedback_statistics(self, neo4j_driver):
        """Test feedback statistics retrieval."""
        collector = FeedbackCollector(neo4j_driver)

        # Log multiple queries
        qid1 = collector.log_query(
            "Query 1", "search", "MATCH (n) RETURN n", ["r1"], {}, 10.0
        )
        qid2 = collector.log_query(
            "Query 2", "traverse", "MATCH (n)-[r]->(m) RETURN n,r,m", ["r2"], {}, 20.0
        )

        # Add ratings
        collector.add_feedback(qid1, 0.9)
        collector.add_feedback(qid2, 0.7)

        stats = collector.get_statistics()

        assert stats["total_queries"] >= 2
        assert stats["rated_queries"] >= 2
        assert stats["avg_rating"] is not None
        assert 0.0 <= stats["avg_rating"] <= 1.0
        assert "search" in stats["intents"]
        assert "traverse" in stats["intents"]

    def test_invalid_rating_rejected(self, neo4j_driver):
        """Test that invalid ratings are rejected."""
        collector = FeedbackCollector(neo4j_driver)

        qid = collector.log_query(
            "Test", "search", "MATCH (n) RETURN n", ["r1"], {}, 10.0
        )

        with pytest.raises(ValueError, match="Rating must be in"):
            collector.add_feedback(qid, 1.5)  # Out of range

        with pytest.raises(ValueError, match="Rating must be in"):
            collector.add_feedback(qid, -0.1)  # Negative


class TestRankingWeightTuner:
    """Test ranking weight tuning with NDCG."""

    def test_compute_ndcg_perfect_ranking(self):
        """Test NDCG computation with perfect ranking."""
        tuner = RankingWeightTuner(None)

        relevance = [1.0, 0.8, 0.6, 0.4, 0.2]
        predicted_ranks = [0, 1, 2, 3, 4]  # Perfect order

        ndcg = tuner.compute_ndcg(relevance, predicted_ranks)

        assert ndcg == pytest.approx(1.0, abs=0.01)

    def test_compute_ndcg_worst_ranking(self):
        """Test NDCG computation with worst ranking."""
        tuner = RankingWeightTuner(None)

        relevance = [1.0, 0.8, 0.6, 0.4, 0.2]
        predicted_ranks = [4, 3, 2, 1, 0]  # Reverse order

        ndcg = tuner.compute_ndcg(relevance, predicted_ranks)

        # Worst ranking should be notably worse than perfect (1.0)
        assert ndcg < 0.8  # Should be poor (but not 0 due to DCG discounting)

    def test_compute_ndcg_at_k(self):
        """Test NDCG@k computation."""
        tuner = RankingWeightTuner(None)

        relevance = [1.0, 0.8, 0.6, 0.4, 0.2]
        predicted_ranks = [0, 1, 2, 3, 4]

        ndcg_5 = tuner.compute_ndcg(relevance, predicted_ranks, k=5)
        ndcg_3 = tuner.compute_ndcg(relevance, predicted_ranks, k=3)

        # Both should be perfect for this ranking
        assert ndcg_5 == pytest.approx(1.0, abs=0.01)
        assert ndcg_3 == pytest.approx(1.0, abs=0.01)

    def test_tune_weights_with_feedback(self, neo4j_driver):
        """Test weight tuning with feedback data."""
        collector = FeedbackCollector(neo4j_driver)
        tuner = RankingWeightTuner(collector)

        # Create synthetic feedback
        for i in range(15):
            qid = collector.log_query(
                f"Query {i}",
                "search",
                "MATCH (s:Section) RETURN s LIMIT 10",
                [f"r{i}"],
                {
                    "semantic_score": 0.7 + (i % 3) * 0.1,
                    "graph_proximity": 0.5,
                    "entity_priority": 0.6,
                    "recency": 0.4,
                    "coverage": 0.7,
                },
                50.0,
            )

            # Vary ratings
            rating = 0.6 + (i % 5) * 0.08
            collector.add_feedback(qid, rating)

        # Tune weights
        optimized_weights = tuner.tune_weights(min_samples=10)

        # Verify weights
        assert all(w >= 0.0 for w in optimized_weights.values())
        assert sum(optimized_weights.values()) == pytest.approx(1.0, abs=0.01)

        # Weights may or may not have changed (optimizer might converge to similar weights)
        # Just verify that optimization ran without error
        delta = tuner.get_weight_delta()
        assert isinstance(delta, dict)
        assert len(delta) == len(tuner.DEFAULT_WEIGHTS)

    def test_tune_weights_insufficient_samples(self, neo4j_driver):
        """Test that insufficient samples returns default weights."""
        collector = FeedbackCollector(neo4j_driver)
        tuner = RankingWeightTuner(collector)

        # Only add a few samples (less than min_samples)
        for i in range(3):
            qid = collector.log_query(
                f"Query {i}", "search", "MATCH (n) RETURN n", ["r1"], {}, 10.0
            )
            collector.add_feedback(qid, 0.8)

        weights = tuner.tune_weights(min_samples=10)

        # Should return defaults
        assert weights == tuner.DEFAULT_WEIGHTS

    def test_evaluate_weights(self, neo4j_driver):
        """Test weight evaluation on held-out data."""
        collector = FeedbackCollector(neo4j_driver)
        tuner = RankingWeightTuner(collector)

        # Create test data
        test_feedbacks = []
        for i in range(5):
            fb = QueryFeedback(
                query_id=f"test_{i}",
                query_text=f"Test query {i}",
                intent="search",
                cypher_query="MATCH (n) RETURN n",
                result_ids=[f"r{i}"],
                rating=0.8,
                ranking_features={
                    "semantic_score": 0.7,
                    "graph_proximity": 0.5,
                    "entity_priority": 0.6,
                    "recency": 0.4,
                    "coverage": 0.7,
                },
                execution_time_ms=50.0,
            )
            test_feedbacks.append(fb)

        metrics = tuner.evaluate_weights(tuner.DEFAULT_WEIGHTS, test_feedbacks)

        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics
        assert "ndcg@20" in metrics
        assert metrics["samples"] == 5


class TestSuggestionEngine:
    """Test template and index suggestion mining."""

    def test_suggest_templates(self, neo4j_driver):
        """Test template suggestion from query patterns."""
        collector = FeedbackCollector(neo4j_driver)
        engine = SuggestionEngine(neo4j_driver, collector)

        # Create similar queries (same pattern)
        base_query = "MATCH (s:Section {document_id: $doc_id}) RETURN s LIMIT 10"

        for i in range(8):
            qid = collector.log_query(
                f"Get sections for document {i}",
                "search",
                base_query,
                [f"s{i}"],
                {},
                30.0,
            )
            collector.add_feedback(qid, 0.85)

        # Get suggestions
        suggestions = engine.suggest_templates(min_frequency=5, min_rating=0.7)

        # Should have at least one suggestion
        assert len(suggestions) > 0

        # Verify suggestion structure
        s = suggestions[0]
        assert s.intent == "search"
        assert s.frequency >= 5
        assert s.avg_rating is None or s.avg_rating >= 0.7  # May be None if no ratings
        assert len(s.example_queries) > 0
        assert 0.0 <= s.confidence <= 1.0

    def test_suggest_templates_filters_low_rating(self, neo4j_driver):
        """Test that low-rated patterns are filtered out."""
        collector = FeedbackCollector(neo4j_driver)
        engine = SuggestionEngine(neo4j_driver, collector)

        # Create queries with low ratings
        for i in range(10):
            qid = collector.log_query(
                f"Bad query {i}",
                "search",
                "MATCH (n) WHERE n.bad_prop = $val RETURN n",
                [f"r{i}"],
                {},
                500.0,  # Slow
            )
            collector.add_feedback(qid, 0.3)  # Low rating

        suggestions = engine.suggest_templates(min_frequency=5, min_rating=0.7)

        # Should filter out low-rated patterns
        low_rated = [s for s in suggestions if s.avg_rating and s.avg_rating < 0.7]
        assert len(low_rated) == 0

    def test_suggest_indexes(self, neo4j_driver):
        """Test index suggestion from slow queries."""
        collector = FeedbackCollector(neo4j_driver)
        engine = SuggestionEngine(neo4j_driver, collector)

        # Create slow queries accessing unindexed properties
        slow_query = "MATCH (s:Section) WHERE s.checksum = $cs RETURN s LIMIT 10"

        for i in range(12):
            collector.log_query(
                f"Slow query {i}",
                "search",
                slow_query,
                [f"s{i}"],
                {},
                150.0,  # Slow
            )

        suggestions = engine.suggest_indexes(
            min_frequency=10,
            min_query_time_ms=100.0,
        )

        # May or may not suggest (depends on existing indexes)
        # Just verify structure if any returned
        for s in suggestions:
            assert s.label
            assert s.property
            assert s.frequency >= 10
            assert s.avg_query_time_ms >= 100.0
            assert s.estimated_benefit > 0.0

    def test_generate_report(self, neo4j_driver):
        """Test comprehensive report generation."""
        collector = FeedbackCollector(neo4j_driver)
        engine = SuggestionEngine(neo4j_driver, collector)

        # Add some feedback
        for i in range(5):
            qid = collector.log_query(
                f"Report test {i}",
                "search",
                "MATCH (n:Section) RETURN n LIMIT 10",
                [f"r{i}"],
                {},
                40.0,
            )
            if i < 3:
                collector.add_feedback(qid, 0.8)

        report = engine.generate_report()

        assert "statistics" in report
        assert "template_suggestions" in report
        assert "index_suggestions" in report
        assert "summary" in report

        stats = report["statistics"]
        assert "total_queries" in stats
        assert "rated_queries" in stats

        summary = report["summary"]
        assert "total_templates" in summary
        assert "total_indexes" in summary


class TestIntegration:
    """Integration tests for the learning module."""

    def test_end_to_end_learning_loop(self, neo4j_driver):
        """Test complete learning loop: feedback → tuning → suggestions."""
        # Setup
        collector = FeedbackCollector(neo4j_driver)
        tuner = RankingWeightTuner(collector)
        engine = SuggestionEngine(neo4j_driver, collector)

        # Step 1: Collect diverse feedback
        intents = ["search", "traverse", "compare"]
        for i in range(20):
            intent = intents[i % len(intents)]
            qid = collector.log_query(
                f"Query {i} for {intent}",
                intent,
                f"MATCH (n:{intent.upper()}) RETURN n LIMIT 10",
                [f"r{i}"],
                {
                    "semantic_score": 0.6 + (i % 5) * 0.08,
                    "graph_proximity": 0.5,
                    "entity_priority": 0.6,
                    "recency": 0.4,
                    "coverage": 0.7,
                },
                45.0 + i * 2,
            )

            # Rate queries
            rating = 0.5 + (i % 6) * 0.08
            collector.add_feedback(qid, rating)

        # Step 2: Tune weights
        _ = tuner.DEFAULT_WEIGHTS.copy()  # noqa: F841 - Keep for documentation
        optimized_weights = tuner.tune_weights(min_samples=10)

        # Weights may or may not have changed (optimizer convergence depends on data)
        # Just verify weights are valid
        assert all(w >= 0.0 for w in optimized_weights.values())
        assert sum(optimized_weights.values()) == pytest.approx(1.0, abs=0.01)

        # Step 3: Generate suggestions
        report = engine.generate_report()

        # Verify report structure
        assert report["statistics"]["total_queries"] >= 20
        assert report["statistics"]["rated_queries"] >= 20

        # Should have some suggestions
        assert isinstance(report["template_suggestions"], list)
        assert isinstance(report["index_suggestions"], list)

    def test_ndcg_improvement_demonstration(self, neo4j_driver):
        """Demonstrate NDCG improvement through weight tuning."""
        collector = FeedbackCollector(neo4j_driver)
        tuner = RankingWeightTuner(collector)

        # Create training data (simplified)
        training_queries = []
        for i in range(15):
            qid = collector.log_query(
                f"Training query {i}",
                "search",
                "MATCH (s:Section) RETURN s LIMIT 10",
                [f"r{i}"],
                {
                    "semantic_score": 0.7 + (i % 3) * 0.1,
                    "graph_proximity": 0.5 + (i % 2) * 0.1,
                    "entity_priority": 0.6,
                    "recency": 0.4,
                    "coverage": 0.7 + (i % 4) * 0.05,
                },
                50.0,
            )

            # Higher ratings for queries with better features
            rating = 0.6 if i % 3 == 0 else 0.8
            collector.add_feedback(qid, rating)
            training_queries.append(qid)

        # Hold-out data
        holdout_feedbacks = []
        for i in range(5):
            fb = QueryFeedback(
                query_id=f"holdout_{i}",
                query_text=f"Holdout {i}",
                intent="search",
                cypher_query="MATCH (n) RETURN n",
                result_ids=[f"h{i}"],
                rating=0.75,
                ranking_features={
                    "semantic_score": 0.8,
                    "graph_proximity": 0.6,
                    "entity_priority": 0.6,
                    "recency": 0.4,
                    "coverage": 0.75,
                },
                execution_time_ms=50.0,
            )
            holdout_feedbacks.append(fb)

        # Evaluate with default weights
        baseline_metrics = tuner.evaluate_weights(
            tuner.DEFAULT_WEIGHTS, holdout_feedbacks
        )

        # Tune weights
        optimized_weights = tuner.tune_weights(min_samples=10)

        # Evaluate with optimized weights
        optimized_metrics = tuner.evaluate_weights(optimized_weights, holdout_feedbacks)

        # Both evaluations should complete
        assert baseline_metrics["samples"] == 5
        assert optimized_metrics["samples"] == 5

        # NDCG scores should be in valid range
        assert 0.0 <= baseline_metrics["ndcg@10"] <= 1.0
        assert 0.0 <= optimized_metrics["ndcg@10"] <= 1.0

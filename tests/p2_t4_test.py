"""
Phase 2, Task 2.4 Tests: Response Builder
Tests Markdown + JSON response generation with NO MOCKS.
"""

import json

import pytest

from src.query.hybrid_search import SearchResult
from src.query.ranking import RankedResult, RankingFeatures
from src.query.response_builder import (Diagnostics, Evidence, Response,
                                        ResponseBuilder, build_response)


@pytest.fixture
def sample_ranked_results():
    """Create sample ranked results for testing."""
    results = [
        SearchResult(
            node_id="section-1",
            node_label="Section",
            score=0.95,
            distance=0,
            metadata={
                "title": "Installation Prerequisites",
                "text": "Before installing, ensure your system meets minimum requirements.",
                "document_id": "doc-1",
                "updated_at": "2024-01-15T00:00:00Z",
            },
        ),
        SearchResult(
            node_id="command-1",
            node_label="Command",
            score=0.85,
            distance=1,
            metadata={
                "name": "weka cluster create",
                "description": "Create a new cluster",
                "updated_at": "2024-01-10T00:00:00Z",
            },
            path=["section-1", "command-1"],
        ),
        SearchResult(
            node_id="error-1",
            node_label="Error",
            score=0.75,
            distance=2,
            metadata={
                "code": "E1001",
                "name": "Network Error",
                "description": "Cannot reach cluster nodes",
            },
        ),
    ]

    # Create RankedResults with features
    ranked = []
    for i, result in enumerate(results):
        features = RankingFeatures(
            semantic_score=result.score,
            graph_distance_score=1.0 / (result.distance + 1),
            recency_score=0.8,
            entity_priority_score=0.9,
            coverage_score=0.7,
            final_score=result.score * 0.8,  # Simplified
        )
        ranked.append(RankedResult(result=result, features=features, rank=i + 1))

    return ranked


class TestEvidence:
    """Test evidence extraction."""

    def test_extract_evidence(self, sample_ranked_results):
        """Test that evidence is extracted from results."""
        builder = ResponseBuilder()
        evidence = builder._extract_evidence(sample_ranked_results[:2])

        assert len(evidence) == 2
        assert all(isinstance(e, Evidence) for e in evidence)
        assert all(e.node_id is not None for e in evidence)
        assert all(e.node_label is not None for e in evidence)

    def test_evidence_has_section_id(self, sample_ranked_results):
        """Test that section evidence includes section_id."""
        builder = ResponseBuilder()
        evidence = builder._extract_evidence(sample_ranked_results[:1])

        # First result is a Section
        assert evidence[0].section_id == "section-1"
        assert evidence[0].node_label == "Section"

    def test_evidence_has_snippet(self, sample_ranked_results):
        """Test that evidence includes text snippet."""
        builder = ResponseBuilder()
        evidence = builder._extract_evidence(sample_ranked_results)

        # Should have snippets from metadata
        assert any(e.snippet for e in evidence)
        # Snippets should be limited length
        assert all(len(e.snippet or "") <= 250 for e in evidence)

    def test_evidence_has_path(self, sample_ranked_results):
        """Test that evidence includes paths when available."""
        builder = ResponseBuilder()
        evidence = builder._extract_evidence(sample_ranked_results)

        # Second result has a path
        path_evidence = [e for e in evidence if e.path]
        assert len(path_evidence) > 0
        assert isinstance(path_evidence[0].path, list)


class TestConfidence:
    """Test confidence estimation."""

    def test_estimate_confidence_in_range(self, sample_ranked_results):
        """Test that confidence is in [0, 1]."""
        builder = ResponseBuilder()
        evidence = builder._extract_evidence(sample_ranked_results)
        confidence = builder._estimate_confidence(sample_ranked_results, evidence)

        assert 0.0 <= confidence <= 1.0

    def test_high_confidence_for_strong_results(self):
        """Test that strong results give high confidence."""
        # Create strong results
        result = SearchResult(
            "id1", "Section", score=0.95, distance=0, metadata={}, path=["a", "b"]
        )
        features = RankingFeatures(semantic_score=0.95, final_score=0.9)
        ranked = [RankedResult(result=result, features=features, rank=1)]

        builder = ResponseBuilder()
        evidence = builder._extract_evidence(ranked)
        confidence = builder._estimate_confidence(ranked, evidence)

        # Should be high confidence
        assert confidence > 0.7

    def test_low_confidence_for_weak_results(self):
        """Test that weak results give lower confidence."""
        # Create weak results
        result = SearchResult("id1", "Section", score=0.3, distance=5, metadata={})
        features = RankingFeatures(semantic_score=0.3, final_score=0.2)
        ranked = [RankedResult(result=result, features=features, rank=1)]

        builder = ResponseBuilder()
        evidence = builder._extract_evidence(ranked)
        confidence = builder._estimate_confidence(ranked, evidence)

        # Should be lower confidence
        assert confidence < 0.7


class TestDiagnostics:
    """Test diagnostics generation."""

    def test_diagnostics_has_ranking_features(self, sample_ranked_results):
        """Test that diagnostics include ranking features."""
        builder = ResponseBuilder()
        timing = {"vector_time_ms": 50, "graph_time_ms": 100}
        diagnostics = builder._build_diagnostics(sample_ranked_results, timing)

        assert diagnostics.ranking_features is not None
        assert "semantic_score" in diagnostics.ranking_features
        assert "final_score" in diagnostics.ranking_features

    def test_diagnostics_has_timing(self, sample_ranked_results):
        """Test that diagnostics include timing info."""
        builder = ResponseBuilder()
        timing = {"vector_time_ms": 50, "graph_time_ms": 100, "total_time_ms": 150}
        diagnostics = builder._build_diagnostics(sample_ranked_results, timing)

        assert diagnostics.timing == timing

    def test_diagnostics_has_candidate_count(self, sample_ranked_results):
        """Test that diagnostics include total candidates."""
        builder = ResponseBuilder()
        diagnostics = builder._build_diagnostics(sample_ranked_results, {})

        assert diagnostics.total_candidates == len(sample_ranked_results)


class TestStructuredResponse:
    """Test structured JSON response."""

    def test_structured_response_schema(self, sample_ranked_results):
        """Test that structured response has required fields."""
        builder = ResponseBuilder()
        timing = {"vector_time_ms": 50, "graph_time_ms": 100}
        response = builder.build_response(
            "test query", "search", sample_ranked_results, timing
        )

        json_response = response.answer_json
        assert json_response.answer is not None
        assert isinstance(json_response.evidence, list)
        assert isinstance(json_response.confidence, float)
        assert isinstance(json_response.diagnostics, Diagnostics)

    def test_structured_response_json_serializable(self, sample_ranked_results):
        """Test that response can be serialized to JSON."""
        builder = ResponseBuilder()
        timing = {"vector_time_ms": 50}
        response = builder.build_response(
            "test query", "search", sample_ranked_results, timing
        )

        # Should be JSON serializable
        json_dict = response.answer_json.to_dict()
        json_str = json.dumps(json_dict)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "answer" in parsed
        assert "evidence" in parsed
        assert "confidence" in parsed
        assert "diagnostics" in parsed

    def test_evidence_array_structure(self, sample_ranked_results):
        """Test that evidence array has correct structure."""
        builder = ResponseBuilder()
        response = builder.build_response("test", "search", sample_ranked_results, {})

        evidence = response.answer_json.evidence
        assert len(evidence) > 0

        for ev in evidence:
            # Required fields
            assert ev.node_id is not None
            assert ev.node_label is not None
            assert isinstance(ev.confidence, float)
            assert 0.0 <= ev.confidence <= 1.0


class TestMarkdownResponse:
    """Test Markdown response generation."""

    def test_markdown_contains_query(self, sample_ranked_results):
        """Test that Markdown includes the original query."""
        builder = ResponseBuilder()
        response = builder.build_response(
            "how to install weka", "search", sample_ranked_results, {}
        )

        assert "how to install weka" in response.answer_markdown

    def test_markdown_contains_intent(self, sample_ranked_results):
        """Test that Markdown includes intent."""
        builder = ResponseBuilder()
        response = builder.build_response(
            "test query", "troubleshoot", sample_ranked_results, {}
        )

        assert "troubleshoot" in response.answer_markdown.lower()

    def test_markdown_contains_confidence(self, sample_ranked_results):
        """Test that Markdown shows confidence."""
        builder = ResponseBuilder()
        response = builder.build_response("test", "search", sample_ranked_results, {})

        assert (
            "Confidence" in response.answer_markdown
            or "confidence" in response.answer_markdown
        )

    def test_markdown_contains_evidence(self, sample_ranked_results):
        """Test that Markdown lists evidence."""
        builder = ResponseBuilder()
        response = builder.build_response("test", "search", sample_ranked_results, {})

        md = response.answer_markdown
        assert "Evidence" in md or "evidence" in md
        # Should mention at least one node ID
        assert any(r.result.node_id in md for r in sample_ranked_results[:2])

    def test_markdown_contains_why_these_results(self, sample_ranked_results):
        """Test that Markdown includes 'Why these results?' section."""
        builder = ResponseBuilder()
        response = builder.build_response("test", "search", sample_ranked_results, {})

        md = response.answer_markdown
        assert "Why" in md and "result" in md.lower()
        # Should show ranking features
        assert "semantic" in md.lower() or "similarity" in md.lower()

    def test_markdown_is_formatted(self, sample_ranked_results):
        """Test that Markdown has proper formatting."""
        builder = ResponseBuilder()
        response = builder.build_response("test", "search", sample_ranked_results, {})

        md = response.answer_markdown
        # Should have headers
        assert "#" in md
        # Should have bold or italics
        assert "**" in md or "*" in md


class TestIntentSpecificAnswers:
    """Test intent-specific answer generation."""

    def test_search_intent_answer(self):
        """Test answer for search intent."""
        result = SearchResult("id1", "Section", score=0.9, distance=0, metadata={})
        features = RankingFeatures(semantic_score=0.9, final_score=0.8)
        ranked = [RankedResult(result=result, features=features, rank=1)]

        builder = ResponseBuilder()
        answer = builder._extract_answer_text("search", ranked)

        assert "found" in answer.lower() or "result" in answer.lower()

    def test_troubleshoot_intent_answer(self):
        """Test answer for troubleshoot intent."""
        result = SearchResult("id1", "Procedure", score=0.9, distance=0, metadata={})
        features = RankingFeatures(semantic_score=0.9, final_score=0.8)
        ranked = [RankedResult(result=result, features=features, rank=1)]

        builder = ResponseBuilder()
        answer = builder._extract_answer_text("troubleshoot", ranked)

        assert (
            "procedure" in answer.lower()
            or "resolution" in answer.lower()
            or "steps" in answer.lower()
        )

    def test_compare_intent_answer(self):
        """Test answer for compare intent."""
        result = SearchResult(
            "id1", "Configuration", score=0.9, distance=0, metadata={}
        )
        features = RankingFeatures(semantic_score=0.9, final_score=0.8)
        ranked = [RankedResult(result=result, features=features, rank=1)]

        builder = ResponseBuilder()
        answer = builder._extract_answer_text("compare", ranked)

        assert "compar" in answer.lower() or "differ" in answer.lower()


class TestEndToEnd:
    """End-to-end response builder tests."""

    def test_complete_response_generation(self, sample_ranked_results):
        """Test complete response generation."""
        response = build_response(
            query="how to install weka",
            intent="search",
            ranked_results=sample_ranked_results,
            timing={"vector_time_ms": 50, "graph_time_ms": 100, "total_time_ms": 150},
        )

        assert isinstance(response, Response)
        assert response.answer_markdown is not None
        assert len(response.answer_markdown) > 0
        assert response.answer_json is not None

    def test_response_has_both_formats(self, sample_ranked_results):
        """Test that response includes both Markdown and JSON."""
        response = build_response(
            query="test",
            intent="search",
            ranked_results=sample_ranked_results,
            timing={},
        )

        # Both formats should exist
        assert response.answer_markdown
        assert response.answer_json

        # JSON should be convertible to dict
        json_dict = response.answer_json.to_dict()
        assert isinstance(json_dict, dict)
        assert "answer" in json_dict

    def test_empty_results_handled_gracefully(self):
        """Test that empty results are handled."""
        response = build_response(
            query="test", intent="search", ranked_results=[], timing={}
        )

        assert response.answer_json.confidence >= 0.0
        assert len(response.answer_json.evidence) == 0
        assert (
            "no" in response.answer_json.answer.lower()
            or "not found" in response.answer_json.answer.lower()
        )

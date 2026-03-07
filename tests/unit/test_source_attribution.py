"""Tests for source attribution: _infer_source() and _infer_source_tags()."""

from __future__ import annotations

from src.mcp_server.mcp_app import _infer_source, _infer_source_tags
from src.query.hybrid_retrieval import ChunkResult


def _make_chunk(**kwargs) -> ChunkResult:
    """Helper to create a minimal ChunkResult with overrides."""
    defaults = dict(
        chunk_id="chunk-001",
        document_id="doc1",
        parent_section_id="s1",
        order=0,
        level=1,
        heading="Test heading",
        text="Test text",
        token_count=50,
    )
    defaults.update(kwargs)
    return ChunkResult(**defaults)


class TestInferSource:
    """_infer_source() returns a single exclusive source string."""

    def test_reranked_source(self):
        chunk = _make_chunk(rerank_score=0.95)
        assert _infer_source(chunk) == "reranked"

    def test_graph_expanded_by_distance(self):
        chunk = _make_chunk(graph_distance=2)
        assert _infer_source(chunk) == "graph_expanded"

    def test_graph_expanded_by_score(self):
        chunk = _make_chunk(graph_score=0.8)
        assert _infer_source(chunk) == "graph_expanded"

    def test_rrf_fusion_source(self):
        chunk = _make_chunk(fusion_method="rrf")
        assert _infer_source(chunk) == "rrf_fusion"

    def test_vector_only_source(self):
        chunk = _make_chunk(vector_score=0.7, bm25_score=None)
        assert _infer_source(chunk) == "vector"

    def test_bm25_only_source(self):
        chunk = _make_chunk(bm25_score=0.6, vector_score=None)
        assert _infer_source(chunk) == "bm25"

    def test_hybrid_fallback(self):
        chunk = _make_chunk()
        assert _infer_source(chunk) == "hybrid"

    def test_reranked_takes_priority_over_graph(self):
        """rerank_score should win even if graph signals are present."""
        chunk = _make_chunk(rerank_score=0.9, graph_distance=1, graph_score=0.5)
        assert _infer_source(chunk) == "reranked"


class TestInferSourceTags:
    """_infer_source_tags() returns a list of all applicable tags."""

    def test_reranked_tag(self):
        chunk = _make_chunk(rerank_score=0.95)
        tags = _infer_source_tags(chunk)
        assert "reranked" in tags

    def test_graph_expanded_tag(self):
        chunk = _make_chunk(graph_distance=2)
        tags = _infer_source_tags(chunk)
        assert "graph_expanded" in tags

    def test_rrf_fusion_tag(self):
        chunk = _make_chunk(fusion_method="rrf")
        tags = _infer_source_tags(chunk)
        assert "rrf_fusion" in tags

    def test_vector_tag(self):
        chunk = _make_chunk(vector_score=0.7)
        tags = _infer_source_tags(chunk)
        assert "vector" in tags

    def test_bm25_tag(self):
        chunk = _make_chunk(bm25_score=0.6)
        tags = _infer_source_tags(chunk)
        assert "bm25" in tags

    def test_hybrid_fallback_when_no_signals(self):
        chunk = _make_chunk()
        tags = _infer_source_tags(chunk)
        assert tags == ["hybrid"]

    def test_multi_origin_chunk_has_multiple_tags(self):
        """A reranked+graph chunk should have both tags."""
        chunk = _make_chunk(
            rerank_score=0.9,
            graph_distance=1,
            graph_score=0.5,
            fusion_method="rrf",
            vector_score=0.7,
            bm25_score=0.4,
        )
        tags = _infer_source_tags(chunk)
        assert "reranked" in tags
        assert "graph_expanded" in tags
        assert "rrf_fusion" in tags
        assert "vector" in tags
        assert "bm25" in tags
        assert len(tags) == 5

    def test_reranked_graph_chunk_source_vs_tags(self):
        """source is 'reranked' (exclusive), source_tags includes both."""
        chunk = _make_chunk(rerank_score=0.9, graph_distance=2, graph_score=0.5)
        source = _infer_source(chunk)
        tags = _infer_source_tags(chunk)
        assert source == "reranked"
        assert "reranked" in tags
        assert "graph_expanded" in tags

"""
Unit tests for Phase A (Query API weighted fusion) and Phase C.0.3 (dedup best score).
"""

import math

from src.query.hybrid_retrieval import (
    ChunkResult,
    QdrantMultiVectorRetriever,
    dedup_chunk_results,
)


def _make_chunk(
    chunk_id: str,
    vector_score: float = None,
    graph_score: float = None,
    fused_score: float = None,
) -> ChunkResult:
    """Helper to create minimal ChunkResult for testing."""
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc1",
        parent_section_id="sec1",
        order=0,
        level=1,
        heading="Test",
        text="Test content",
        token_count=10,
        vector_score=vector_score,
        graph_score=graph_score,
        fused_score=fused_score,
    )


class DummyPoint:
    def __init__(self, pid, score):
        self.id = pid
        self.score = score
        self.payload = {}


class MockEmbedder:
    """Mock embedder that supports sparse embeddings for testing."""

    def embed_sparse(self, text):
        return {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]}


class DummyClient:
    """Minimal client stub for weighted Query API tests."""

    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        call_idx = len(self.calls) - 1
        using = kwargs.get("using")

        # Stage 1: candidate recall – return three dummy points with payloads
        if call_idx == 0:
            return type(
                "DummyResponse",
                (),
                {
                    "points": [
                        DummyPoint("a", 0.1),
                        DummyPoint("b", 0.1),
                        DummyPoint("c", 0.1),
                    ]
                },
            )()

        # Stage 2: per-field scoring over the same candidate IDs
        field_scores = {
            "content": {"a": 2.0, "b": 1.0},
            "title": {"a": 1.0, "c": 0.5},
            "entity": {"b": 0.5},
            "text-sparse": {"c": 0.4},
        }
        scores = field_scores.get(using, {})
        points = [DummyPoint(pid, score) for pid, score in scores.items()]
        return type("DummyResponse", (), {"points": points})()


# =============================================================================
# Tests for _fuse_rankings
# =============================================================================


def test_fuse_rankings_applies_weights_and_normalization():
    """Test that _fuse_rankings correctly applies weights and per-field normalization."""
    # Use MockEmbedder with schema_supports_sparse=True to prevent
    # the sparse field weight from being zeroed out (lines 719-720 in hybrid_retrieval.py)
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=DummyClient(),
        embedder=MockEmbedder(),
        field_weights={"content": 0.5, "title": 0.2, "entity": 0.1, "text-sparse": 0.2},
        schema_supports_sparse=True,
    )
    rankings = {
        "content": [("a", 2.0), ("b", 1.0)],
        "title": [("a", 1.0), ("c", 0.5)],
        "entity": [("b", 0.5)],
        "text-sparse": [("c", 0.4)],
    }
    fused = retriever._fuse_rankings(rankings)

    # Normalize per field:
    # content: max=2.0 -> a:1.0*0.5=0.5, b:0.5*0.5=0.25
    # title: max=1.0 -> a:1.0*0.2=0.2, c:0.5*0.2=0.1
    # entity: max=0.5 -> b:1.0*0.1=0.1
    # text-sparse: max=0.4 -> c:1.0*0.2=0.2
    # Results: a=0.7, b=0.35, c=0.3

    assert math.isclose(fused["a"], 0.7), f"Expected a=0.7, got {fused['a']}"
    assert math.isclose(fused["b"], 0.35), f"Expected b=0.35, got {fused['b']}"
    assert math.isclose(fused["c"], 0.3), f"Expected c=0.3, got {fused['c']}"


# =============================================================================
# Tests for dedup_chunk_results (standalone function)
# =============================================================================


def test_dedup_chunk_results_single_items_unchanged():
    """Non-duplicate items pass through unchanged."""
    r1 = _make_chunk("x", vector_score=0.5, fused_score=0.5)
    r2 = _make_chunk("y", vector_score=0.3, fused_score=0.3)

    result = dedup_chunk_results([r1, r2])

    assert len(result) == 2
    assert {r.chunk_id for r in result} == {"x", "y"}


def test_dedup_chunk_results_merges_best_scores():
    """Duplicate chunks get best scores merged and fused_score recomputed."""
    r1 = _make_chunk("x", vector_score=0.5, graph_score=0.2, fused_score=0.5)
    r2 = _make_chunk("x", vector_score=0.4, graph_score=0.8, fused_score=1.0)

    # Use weights: vector=0.6, graph=0.4
    result = dedup_chunk_results([r1, r2], vector_weight=0.6, graph_weight=0.4)

    assert len(result) == 1
    winner = result[0]

    # Merged: max vector_score = 0.5, max graph_score = 0.8
    assert math.isclose(winner.vector_score, 0.5)
    assert math.isclose(winner.graph_score, 0.8)

    # Fused = 0.6*0.5 + 0.4*0.8 = 0.3 + 0.32 = 0.62
    assert math.isclose(
        winner.fused_score, 0.62
    ), f"Expected 0.62, got {winner.fused_score}"


def test_dedup_chunk_results_default_weights():
    """Default weights (0.7, 0.3) are used when not specified."""
    r1 = _make_chunk("x", vector_score=1.0, graph_score=0.0, fused_score=1.0)
    r2 = _make_chunk("x", vector_score=0.0, graph_score=1.0, fused_score=1.0)

    result = dedup_chunk_results([r1, r2])  # defaults: 0.7, 0.3

    assert len(result) == 1
    winner = result[0]

    # Merged: vector=1.0, graph=1.0
    # Fused = 0.7*1.0 + 0.3*1.0 = 1.0
    assert math.isclose(winner.fused_score, 1.0)


def test_dedup_chunk_results_empty_input():
    """Empty input returns empty output."""
    result = dedup_chunk_results([])
    assert result == []


def test_dedup_chunk_results_preserves_bm25_score():
    """BM25 scores are also merged (max)."""
    r1 = _make_chunk("x", vector_score=0.5, fused_score=0.5)
    r1.bm25_score = 0.3
    r2 = _make_chunk("x", vector_score=0.4, fused_score=0.4)
    r2.bm25_score = 0.7

    result = dedup_chunk_results([r1, r2])

    assert len(result) == 1
    assert math.isclose(result[0].bm25_score, 0.7)


# =============================================================================
# Tests for Query API weighted path
# =============================================================================


def test_query_api_weighted_path_sets_weighted_fusion_and_sparse_metrics():
    """End-to-end unit test for _search_via_query_api_weighted using DummyClient."""

    class DummyBundleSparse:
        def __init__(self):
            self.indices = [0, 1]
            self.values = [0.5, 0.5]

    class DummyBundle:
        def __init__(self):
            self.dense = [0.1, 0.2]
            self.sparse = DummyBundleSparse()

    client = DummyClient()
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=client,
        embedder=None,
        field_weights={
            "content": 0.5,
            "title": 0.2,
            "entity": 0.1,
            "text-sparse": 0.2,
        },
        schema_supports_sparse=True,
        query_api_candidate_limit=10,
    )
    # Ensure sparse field is enabled with non-zero weight
    retriever.supports_sparse = True
    retriever.field_weights["text-sparse"] = 0.2

    bundle = DummyBundle()
    results = retriever._search_via_query_api_weighted(
        bundle=bundle,
        top_k=2,
        filters=None,
    )

    # We expect two results, ordered by fused score: a, then b
    assert [r.chunk_id for r in results] == ["a", "b"]
    # Fused scores mirror _fuse_rankings test expectations
    assert math.isclose(results[0].fused_score, 0.7)
    assert math.isclose(results[1].fused_score, 0.35)
    # Vector score kind should mark weighted fusion
    assert all(r.vector_score_kind == "weighted_fusion" for r in results)
    # Sparse metrics should be present in last_stats
    stats = retriever.last_stats
    assert "sparse_scored_ratio" in stats
    assert "sparse_topk_ratio" in stats

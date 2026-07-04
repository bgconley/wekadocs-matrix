"""Tests for ColBERT observability in RetrievalTraceBuilder."""

from __future__ import annotations

from src.mcp_server.retrieval_trace import RetrievalTraceBuilder


class TestRecordColbertApplied:
    """When ColBERT was applied, trace output should reflect that."""

    def test_colbert_applied_renders_in_format(self):
        trace = RetrievalTraceBuilder(trace_id="test-001", session_id="sess-001")
        trace.record_colbert(
            applied=True,
            runtime_available=True,
            query_embedding_ok=True,
            rank_deltas=[0, -2, 1, 3, -1, 0, 2, -3, 1, 0],
            candidates=50,
            hydrated=48,
            latency_ms=42.5,
        )
        output = trace.format()
        assert "COLBERT" in output
        assert "Applied:            True" in output
        assert "Runtime available:  True" in output
        assert "Query embedding OK: True" in output
        assert "Candidates:         50" in output
        assert "Hydrated:           48" in output
        assert "42.5ms" in output
        assert "Rank deltas" in output

    def test_colbert_not_applied_renders_in_format(self):
        trace = RetrievalTraceBuilder(trace_id="test-002", session_id="sess-002")
        trace.record_colbert(
            applied=False,
            runtime_available=True,
            query_embedding_ok=False,
            candidates=0,
            hydrated=0,
            latency_ms=0.0,
        )
        output = trace.format()
        assert "COLBERT" in output
        assert "Applied:            False" in output
        assert "Query embedding OK: False" in output

    def test_colbert_appears_in_to_dict(self):
        trace = RetrievalTraceBuilder(trace_id="test-003", session_id="sess-003")
        trace.record_colbert(
            applied=True,
            runtime_available=True,
            query_embedding_ok=True,
            rank_deltas=[1, -1, 0],
            candidates=30,
            hydrated=28,
            latency_ms=55.3,
        )
        d = trace.to_dict()
        assert "colbert" in d
        cb = d["colbert"]
        assert cb["applied"] is True
        assert cb["runtime_available"] is True
        assert cb["query_embedding_ok"] is True
        assert cb["rank_deltas_top10"] == [1, -1, 0]
        assert cb["candidates"] == 30
        assert cb["hydrated"] == 28
        assert cb["latency_ms"] == 55.3

    def test_colbert_none_when_not_recorded(self):
        trace = RetrievalTraceBuilder(trace_id="test-004", session_id="sess-004")
        d = trace.to_dict()
        assert d["colbert"] is None


class TestRecordStageSnapshots:
    """Stage snapshots should appear in format() and to_dict()."""

    def test_stage_snapshots_render_in_format(self):
        trace = RetrievalTraceBuilder(trace_id="test-005", session_id="sess-005")
        trace.record_stage_snapshots(
            {
                "post_fusion": [
                    {
                        "chunk_id": "c1",
                        "fused_score": 0.95,
                        "rerank_score": None,
                        "doc_tag": "d1",
                    },
                    {
                        "chunk_id": "c2",
                        "fused_score": 0.88,
                        "rerank_score": None,
                        "doc_tag": "d1",
                    },
                ],
                "post_reranker": [
                    {
                        "chunk_id": "c2",
                        "fused_score": 0.88,
                        "rerank_score": 0.99,
                        "doc_tag": "d1",
                    },
                    {
                        "chunk_id": "c1",
                        "fused_score": 0.95,
                        "rerank_score": 0.85,
                        "doc_tag": "d1",
                    },
                ],
            }
        )
        output = trace.format()
        assert "STAGE SNAPSHOTS" in output
        assert "post_fusion" in output
        assert "post_reranker" in output
        assert "c1" in output
        assert "c2" in output

    def test_stage_snapshots_appear_in_to_dict(self):
        trace = RetrievalTraceBuilder(trace_id="test-006", session_id="sess-006")
        snapshots = {
            "post_fusion": [
                {
                    "chunk_id": "c1",
                    "fused_score": 0.95,
                    "rerank_score": None,
                    "doc_tag": "d1",
                },
            ],
        }
        trace.record_stage_snapshots(snapshots)
        d = trace.to_dict()
        assert "stage_snapshots" in d
        assert "post_fusion" in d["stage_snapshots"]
        assert len(d["stage_snapshots"]["post_fusion"]) == 1

    def test_empty_stage_snapshots_in_to_dict(self):
        trace = RetrievalTraceBuilder(trace_id="test-007", session_id="sess-007")
        d = trace.to_dict()
        assert d["stage_snapshots"] == {}


def test_snapshot_top_preserves_zero_rerank_score():
    """_snapshot_top must show rerank_score=0.0, not hide it as None."""
    from src.query.hybrid_retrieval import ChunkResult, _snapshot_top

    chunk = ChunkResult(
        chunk_id="c1",
        document_id="d1",
        parent_section_id="s1",
        order=1,
        level=1,
        heading="H",
        text="T",
        token_count=5,
    )
    chunk.rerank_score = 0.0
    chunk.reranker = "circuit_open"
    snap = _snapshot_top([chunk])
    assert snap[0]["rerank_score"] == 0.0  # Must be 0.0, not None
    assert snap[0]["reranker"] == "circuit_open"
    assert snap[0]["is_expanded"] is False


def test_snapshot_top_none_rerank_score():
    """When rerank_score was never set, snapshot shows None."""
    from src.query.hybrid_retrieval import ChunkResult, _snapshot_top

    chunk = ChunkResult(
        chunk_id="c1",
        document_id="d1",
        parent_section_id="s1",
        order=1,
        level=1,
        heading="H",
        text="T",
        token_count=5,
    )
    snap = _snapshot_top([chunk])
    assert snap[0]["rerank_score"] is None

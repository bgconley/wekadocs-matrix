import types

from src.query.hybrid_retrieval import ChunkResult, HybridRetriever


class FakeRerankProvider:
    def __init__(self):
        self.model_id = "fake-v3"
        self.provider_name = "fake"

    def rerank(self, query, candidates, top_k=10):
        sliced = list(candidates[:top_k])
        for idx, cand in enumerate(sliced, start=1):
            cand["original_rank"] = idx
        sliced.reverse()
        for idx, cand in enumerate(sliced, start=1):
            cand["rerank_score"] = 100.0 - idx
            cand["reranker"] = self.model_id
        return sliced


class ErrorRerankProvider(FakeRerankProvider):
    def rerank(self, query, candidates, top_k=10):
        raise RuntimeError("provider down")


class DummyTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _bootstrap_retriever(reranker_enabled=True):
    hr = object.__new__(HybridRetriever)
    hr.tokenizer = DummyTokenizer()
    hr.reranker_config = types.SimpleNamespace(enabled=reranker_enabled, top_n=2)
    hr._reranker_enabled = reranker_enabled
    hr.rerank_top_n = 2 if reranker_enabled else 0
    hr._reranker = None
    return hr


def _chunk(chunk_id: str, text: str, score: float) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc",
        parent_section_id="parent",
        order=1,
        level=1,
        heading="Heading",
        text=text,
        token_count=10,
        fused_score=score,
    )


def test_apply_reranker_reorders_and_sets_metadata(monkeypatch):
    hr = _bootstrap_retriever(reranker_enabled=True)
    fake_reranker = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake_reranker)

    seeds = [_chunk("a", "first chunk", 0.1), _chunk("b", "second chunk", 0.2)]
    original_scores = {chunk.chunk_id: chunk.fused_score for chunk in seeds}
    metrics = {}

    reranked = hr._apply_reranker("query", seeds, metrics)

    assert reranked[0].chunk_id == "b"
    assert reranked[0].rerank_score is not None
    assert reranked[0].fusion_method == "rerank"
    assert reranked[0].rerank_rank == 1
    assert reranked[0].rerank_original_rank == 2
    assert reranked[1].rerank_rank == 2
    assert reranked[1].rerank_original_rank == 1
    assert reranked[0].fused_score == original_scores["b"]
    assert reranked[1].fused_score == original_scores["a"]
    assert metrics["reranker_applied"] is True
    assert metrics["reranker_reason"] == "ok"
    assert metrics["reranker_model"] == fake_reranker.model_id
    assert metrics["reranker_time_ms"] >= 0


def test_apply_reranker_handles_provider_error(monkeypatch):
    hr = _bootstrap_retriever(reranker_enabled=True)
    monkeypatch.setattr(hr, "_reranker", ErrorRerankProvider())

    seeds = [_chunk("a", "chunk text", 0.1)]
    metrics = {}
    reranked = hr._apply_reranker("query", seeds, metrics)

    assert reranked == seeds
    assert metrics["reranker_applied"] is False
    assert metrics["reranker_reason"] == "provider_error"


def test_apply_reranker_skips_when_no_text(monkeypatch):
    hr = _bootstrap_retriever(reranker_enabled=True)
    monkeypatch.setattr(hr, "_reranker", FakeRerankProvider())

    chunk = _chunk("a", "", 0.1)
    chunk.heading = ""
    metrics = {}

    reranked = hr._apply_reranker("query", [chunk], metrics)
    assert reranked == [chunk]
    assert metrics["reranker_reason"] == "no_text"


def test_apply_reranker_skips_zero_token_headings(monkeypatch):
    hr = _bootstrap_retriever(reranker_enabled=True)
    monkeypatch.setattr(hr, "_reranker", FakeRerankProvider())

    chunk = _chunk("a", "", 0.1)
    chunk.heading = "Heading only"
    chunk.token_count = 0
    metrics = {}

    reranked = hr._apply_reranker("query", [chunk], metrics)
    assert reranked == [chunk]
    assert metrics["reranker_reason"] == "no_text"

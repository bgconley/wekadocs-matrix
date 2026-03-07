import types

from src.query.hybrid_retrieval import ChunkResult, HybridRetriever


class FakeRerankProvider:
    def __init__(self):
        self.model_id = "fake-v3"
        self.provider_name = "fake"
        self.last_candidates = None  # Capture for test inspection

    def rerank(self, query, candidates, top_k=10, *, instruction=None):
        self.last_candidates = list(candidates)  # Capture input
        sliced = list(candidates[:top_k])
        for idx, cand in enumerate(sliced, start=1):
            cand["original_rank"] = idx
        sliced.reverse()
        for idx, cand in enumerate(sliced, start=1):
            cand["rerank_score"] = 100.0 - idx
            cand["reranker"] = self.model_id
        return sliced


class ErrorRerankProvider(FakeRerankProvider):
    def rerank(self, query, candidates, top_k=10, *, instruction=None):
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
    hr._reranker_available = True
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


def test_apply_reranker_uses_heading_when_text_empty(monkeypatch):
    """A chunk with empty text but a valid heading is still reranked using the heading."""
    hr = _bootstrap_retriever(reranker_enabled=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    chunk = _chunk("a", "", 0.1)
    chunk.heading = "Heading only"
    chunk.token_count = 0
    metrics = {}

    hr._apply_reranker("query", [chunk], metrics)
    # Heading-only chunks ARE sent to reranker (heading is meaningful content)
    assert metrics["reranker_applied"] is True
    assert fake.last_candidates is not None
    assert fake.last_candidates[0]["text"] == "Heading only"


def test_apply_reranker_prepends_parent_path_norm(monkeypatch):
    """Verify reranker text includes parent_path_norm > heading > body."""
    hr = _bootstrap_retriever(reranker_enabled=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    chunk = _chunk("a", "Body content here", 0.5)
    chunk.heading = "Bucket Settings"
    chunk.parent_path_norm = "Configuration > S3 Backend"
    metrics = {}

    hr._apply_reranker("how to configure S3", [chunk], metrics)

    # Verify the reranker received text with parent_path prepended
    assert fake.last_candidates is not None
    text = fake.last_candidates[0]["text"]
    assert text.startswith("Configuration > S3 Backend")
    assert "Bucket Settings" in text
    assert "Body content here" in text


def test_apply_reranker_without_parent_path_norm(monkeypatch):
    """Without parent_path_norm, text is heading + body (legacy behavior)."""
    hr = _bootstrap_retriever(reranker_enabled=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    chunk = _chunk("a", "Body content here", 0.5)
    chunk.heading = "Bucket Settings"
    # parent_path_norm is None (default)
    metrics = {}

    hr._apply_reranker("query", [chunk], metrics)

    text = fake.last_candidates[0]["text"]
    assert text.startswith("Bucket Settings")
    assert "Body content here" in text
    assert ">" not in text.split("\n")[0]  # No breadcrumb path


def test_apply_reranker_passes_per_type_instruction(monkeypatch):
    """Per-query-type instruction is passed to reranker.rerank()."""
    import types

    hr = _bootstrap_retriever(reranker_enabled=True)
    hr.reranker_config = types.SimpleNamespace(
        enabled=True,
        top_n=2,
        instruction="default instruction",
        instructions_by_type={
            "subsystem_architecture": "Special subsystem instruction"
        },
    )
    fake = FakeRerankProvider()
    captured = {}
    _orig_rerank = fake.rerank

    def capturing_rerank(query, candidates, top_k=10, *, instruction=None):
        captured["instruction"] = instruction
        return _orig_rerank(query, candidates, top_k)

    fake.rerank = capturing_rerank
    monkeypatch.setattr(hr, "_reranker", fake)

    metrics = {}
    hr._apply_reranker(
        "query",
        [_chunk("a", "text", 0.5)],
        metrics,
        query_type="subsystem_architecture",
    )
    assert captured["instruction"] == "Special subsystem instruction"
    assert metrics["reranker_instruction"] == "Special subsystem instruction"


def test_apply_reranker_falls_back_to_default_instruction(monkeypatch):
    """When no per-type instruction, falls back to default config instruction."""
    import types

    hr = _bootstrap_retriever(reranker_enabled=True)
    hr.reranker_config = types.SimpleNamespace(
        enabled=True,
        top_n=2,
        instruction="default instruction",
        instructions_by_type={"subsystem_architecture": "Special instruction"},
    )
    fake = FakeRerankProvider()
    captured = {}
    _orig_rerank = fake.rerank

    def capturing_rerank(query, candidates, top_k=10, *, instruction=None):
        captured["instruction"] = instruction
        return _orig_rerank(query, candidates, top_k)

    fake.rerank = capturing_rerank
    monkeypatch.setattr(hr, "_reranker", fake)

    metrics = {}
    hr._apply_reranker(
        "query",
        [_chunk("a", "text", 0.5)],
        metrics,
        query_type="procedural",  # No per-type instruction for this type
    )
    assert captured["instruction"] is None  # No per-call override
    assert metrics["reranker_instruction"] == "default instruction"  # Falls back


def test_apply_reranker_detects_circuit_open_fallback(monkeypatch):
    """When reranker returns circuit_open markers, reranker_applied should be False."""
    import types

    hr = _bootstrap_retriever(reranker_enabled=True)
    hr.reranker_config = types.SimpleNamespace(
        enabled=True,
        top_n=5,
        instruction="default instruction",
        instructions_by_type=None,
    )

    class CircuitOpenReranker:
        model_id = "test-model"
        provider_name = "test"

        def rerank(self, query, candidates, top_k=10, *, instruction=None):
            return [
                {**c, "rerank_score": 0.0, "reranker": "circuit_open"}
                for c in candidates[:top_k]
            ]

    monkeypatch.setattr(hr, "_reranker", CircuitOpenReranker())

    seeds = [_chunk("a", "text a", 0.5), _chunk("b", "text b", 0.3)]
    metrics = {}
    hr._apply_reranker("query", seeds, metrics)

    assert metrics["reranker_applied"] is False
    assert metrics["reranker_reason"] == "fallback_zero_scores"
    assert metrics["reranker_real_scores_count"] == 0


def test_apply_reranker_detects_real_scores(monkeypatch):
    """When reranker returns real scores, reranker_applied should be True."""
    import types

    hr = _bootstrap_retriever(reranker_enabled=True)
    hr.reranker_config = types.SimpleNamespace(
        enabled=True,
        top_n=5,
        instruction="default instruction",
        instructions_by_type=None,
    )

    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    seeds = [_chunk("a", "text a", 0.5), _chunk("b", "text b", 0.3)]
    metrics = {}
    hr._apply_reranker("query", seeds, metrics)

    assert metrics["reranker_applied"] is True
    assert metrics["reranker_reason"] == "ok"
    assert metrics["reranker_real_scores_count"] > 0

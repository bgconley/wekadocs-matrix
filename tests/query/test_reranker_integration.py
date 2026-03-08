import types

import pytest

from src.query.hybrid_retrieval import ChunkResult, FusionMethod, HybridRetriever
from src.query.query_intent import QueryIntent
from src.query.retrieval_plan import ResolvedRetrievalPlan, RetrievalProfile
from src.shared.config import SignalPoolConfig


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


def _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=False):
    hr = object.__new__(HybridRetriever)
    hr.tokenizer = DummyTokenizer()
    hr.reranker_config = types.SimpleNamespace(enabled=reranker_enabled, top_n=2)
    hr._reranker_enabled = reranker_enabled
    hr.rerank_top_n = 2 if reranker_enabled else 0
    hr._reranker = None
    hr._reranker_available = True
    hr.config = types.SimpleNamespace(
        feature_flags=types.SimpleNamespace(
            precision_focused_rerank_text=focused_rerank_text,
        )
    )
    hr._plan = ResolvedRetrievalPlan(
        profile=RetrievalProfile.PRECISION_VECTOR,
        use_focused_rerank_text=focused_rerank_text,
        use_specificity_adjustment=False,
        use_structure_expansion=True,
        graph_garbage_filter_on=False,
        graph_score_normalized_on=False,
    )
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


# ---------------------------------------------------------------------------
# Provider-level tests for instruction mode and health check
# ---------------------------------------------------------------------------


def test_native_instruction_mode_sends_separate_field(monkeypatch):
    """In native mode, instruction should be a separate JSON field, not in query."""
    from src.providers.rerank.local_reranker_service import LocalRerankerServiceProvider

    provider = LocalRerankerServiceProvider(
        model="test-model",
        base_url="http://fake:9006",
        instruction="Test instruction",
        instruction_mode="native",
        batch_size=2,
    )

    captured_payloads = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"results": [{"index": 0, "score": 0.9}]}

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        return FakeResponse()

    monkeypatch.setattr(provider._client, "post", fake_post)

    provider.rerank("my query", [{"text": "doc text", "id": "1"}], top_k=1)

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["instruction"] == "Test instruction"
    assert "Test instruction" not in payload["query"]  # NOT prepended


def test_prepend_instruction_mode_puts_in_query(monkeypatch):
    """In prepend mode, instruction should be prepended to query, not a separate field."""
    from src.providers.rerank.local_reranker_service import LocalRerankerServiceProvider

    provider = LocalRerankerServiceProvider(
        model="test-model",
        base_url="http://fake:9006",
        instruction="Test instruction",
        instruction_mode="prepend",
        batch_size=2,
    )

    captured_payloads = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"results": [{"index": 0, "score": 0.9}]}

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        return FakeResponse()

    monkeypatch.setattr(provider._client, "post", fake_post)

    provider.rerank("my query", [{"text": "doc text", "id": "1"}], top_k=1)

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "instruction" not in payload  # NOT a separate field
    assert "Test instruction" in payload["query"]  # Prepended to query


def test_health_check_returns_false_on_connection_error(monkeypatch):
    """health_check() must return False when the service is unreachable."""
    import httpx

    from src.providers.rerank.local_reranker_service import LocalRerankerServiceProvider

    provider = LocalRerankerServiceProvider(
        model="test-model",
        base_url="http://unreachable:9999",
    )

    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(provider._client, "get", fake_get)
    assert provider.health_check() is False


# ---------------------------------------------------------------------------
# Phase A3: Focused reranker text tests
# ---------------------------------------------------------------------------


def _precision_intent(anchors=("metadata",)):
    """Create a precision QueryIntent for testing."""
    return QueryIntent(
        query_type="subsystem_architecture",
        precision_mode=True,
        subsystem_terms=anchors,
        primary_anchors=anchors,
        generic_modifiers=("managed",),
    )


def test_focused_text_extracts_anchor_lines(monkeypatch):
    """Focused text should only include lines with anchor terms + ±1 context."""
    hr = _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    body = (
        "Line 0: general overview\n"
        "Line 1: cluster architecture\n"
        "Line 2: metadata management details\n"
        "Line 3: more metadata info\n"
        "Line 4: unrelated content\n"
        "Line 5: deployment guide"
    )
    chunk = _chunk("a", body, 0.5)
    chunk.heading = "Architecture Overview"
    chunk.parent_path_norm = "Internals > WEKA"
    metrics = {}

    hr._apply_reranker(
        "metadata query",
        [chunk],
        metrics,
        query_type="subsystem_architecture",
        intent=_precision_intent(("metadata",)),
    )

    text = fake.last_candidates[0]["text"]
    # Should include lines 1-4 (lines 2,3 match + ±1 context) but not line 0 or 5
    assert "metadata management details" in text
    assert "more metadata info" in text
    assert "cluster architecture" in text  # ±1 context for line 2
    assert "unrelated content" in text  # ±1 context for line 3
    assert "deployment guide" not in text  # line 5, outside window
    # Structural context preserved
    assert "Internals > WEKA" in text
    assert "Architecture Overview" in text
    assert metrics["precision_focused_rerank_text"] is True


def test_focused_text_fallback_when_no_anchor_match(monkeypatch):
    """When no anchor matches in body, fallback to heading + first 500 chars."""
    hr = _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    body = "No relevant terms here. Just general content about clusters and deployment."
    chunk = _chunk("a", body, 0.5)
    chunk.heading = "Generic Heading"
    chunk.parent_path_norm = "Path > To > Section"
    metrics = {}

    hr._apply_reranker(
        "metadata query",
        [chunk],
        metrics,
        query_type="subsystem_architecture",
        intent=_precision_intent(("metadata",)),
    )

    text = fake.last_candidates[0]["text"]
    assert "Generic Heading" in text
    assert "No relevant terms here" in text
    assert metrics["focused_rerank_text_fallback_count"] == 1


def test_non_precision_intent_gets_full_text(monkeypatch):
    """Non-precision intents should get the full text, not focused windows."""
    hr = _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    body = (
        "Line 0: general overview\n"
        "Line 1: cluster architecture\n"
        "Line 2: deployment guide"
    )
    chunk = _chunk("a", body, 0.5)
    chunk.heading = "Heading"
    chunk.parent_path_norm = "Path"
    metrics = {}

    # Non-precision intent
    non_precision = QueryIntent(query_type="conceptual")
    hr._apply_reranker(
        "query",
        [chunk],
        metrics,
        query_type="conceptual",
        intent=non_precision,
    )

    text = fake.last_candidates[0]["text"]
    # All lines should be present (full text mode)
    assert "general overview" in text
    assert "cluster architecture" in text
    assert "deployment guide" in text
    assert metrics["precision_focused_rerank_text"] is False


def test_focused_text_flag_off_uses_full_text(monkeypatch):
    """When flag is False, precision intents still get full text."""
    hr = _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=False)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    body = (
        "Line 0: general overview\n"
        "Line 1: metadata content\n"
        "Line 2: deployment guide"
    )
    chunk = _chunk("a", body, 0.5)
    chunk.heading = "Heading"
    chunk.parent_path_norm = "Path"
    metrics = {}

    hr._apply_reranker(
        "metadata query",
        [chunk],
        metrics,
        query_type="subsystem_architecture",
        intent=_precision_intent(("metadata",)),
    )

    text = fake.last_candidates[0]["text"]
    # Full text should be present because flag is off
    assert "general overview" in text
    assert "metadata content" in text
    assert "deployment guide" in text
    assert metrics["precision_focused_rerank_text"] is False


def test_focused_text_preserves_table_lines(monkeypatch):
    """Table lines (with pipes) should be preserved unchanged in focused windows."""
    hr = _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    body = (
        "| Column A | Column B |\n"
        "|----------|----------|\n"
        "| metadata | 128 KB   |\n"
        "| other    | 256 KB   |\n"
        "Unrelated paragraph."
    )
    chunk = _chunk("a", body, 0.5)
    chunk.heading = "Resource Table"
    chunk.parent_path_norm = ""
    metrics = {}

    hr._apply_reranker(
        "metadata query",
        [chunk],
        metrics,
        query_type="subsystem_architecture",
        intent=_precision_intent(("metadata",)),
    )

    text = fake.last_candidates[0]["text"]
    # _clean_text collapses whitespace, so extra spaces in table cells are reduced
    assert "| metadata | 128 KB |" in text
    assert "|----------|----------|" in text  # context line


def test_focused_text_no_anchors_uses_full_text(monkeypatch):
    """When intent has no primary_anchors, full text is used even with flag on."""
    hr = _bootstrap_retriever(reranker_enabled=True, focused_rerank_text=True)
    fake = FakeRerankProvider()
    monkeypatch.setattr(hr, "_reranker", fake)

    body = "Some content about architecture and internals."
    chunk = _chunk("a", body, 0.5)
    chunk.heading = "Heading"
    chunk.parent_path_norm = "Path"
    metrics = {}

    no_anchor_intent = QueryIntent(
        query_type="subsystem_architecture",
        precision_mode=True,
        subsystem_terms=("architecture", "internals"),
        primary_anchors=(),  # No anchors
        generic_modifiers=("architecture", "internals"),
    )

    hr._apply_reranker(
        "query",
        [chunk],
        metrics,
        query_type="subsystem_architecture",
        intent=no_anchor_intent,
    )

    text = fake.last_candidates[0]["text"]
    assert "Some content about architecture and internals" in text
    assert (
        metrics["precision_focused_rerank_text"] is False
    )  # Disabled due to empty anchors


# ---------------------------------------------------------------------------
# Fix 2 regression: pool_before_colbert + reranker disabled must use pool output
# ---------------------------------------------------------------------------


def test_pool_before_colbert_reranker_disabled_uses_pool_output(monkeypatch):
    """When signal_pool_before_colbert=True and the cross-encoder reranker is
    disabled, the final seeds must come from the signal-pool output (via
    best_available), not from raw fused_results.

    Before the fix, retrieve() discarded pool + ColBERT work and fell back
    to fused_results[:top_k].
    """
    hr = object.__new__(HybridRetriever)

    # --- Config ---
    hr.config = types.SimpleNamespace(
        feature_flags=types.SimpleNamespace(
            signal_pool_before_colbert=True,
            precision_focused_rerank_text=False,
            structure_aware_expansion=False,
            graph_as_reranker=False,
            query_api_weighted_fusion=False,
            dedup_best_score=False,
            signal_diverse_rerank_pool=True,
        ),
        ner=types.SimpleNamespace(enabled=False),
        monitoring=types.SimpleNamespace(
            metrics_aggregation_enabled=False,
            slo_monitoring_enabled=False,
        ),
    )
    hr.embedding_settings = None

    # --- Retrieval plan ---
    hr._plan = ResolvedRetrievalPlan(
        profile=RetrievalProfile.PRECISION_VECTOR,
        use_signal_pool=True,
        signal_pool_before_colbert=True,
        use_colbert=False,
        use_weighted_fusion=False,
        use_related_to_expansion=False,
        use_related_to_blending=False,
        use_focused_rerank_text=False,
        use_specificity_adjustment=False,
        use_structure_expansion=False,
        use_entity_graph_channel=False,
        use_graph_enrichment=False,
        graph_garbage_filter_on=False,
        graph_score_normalized_on=False,
    )

    # --- Signal pool: consensus_slots=3 + text_sparse_slots=2 ---
    # The pool will pick 3 by fused_score + 2 by sparse score,
    # creating output that DIFFERS from raw fused_results[:5].
    # All other slot types zeroed to prevent them filling before sparse.
    hr._signal_pool_enabled = True
    hr._signal_pool_config = SignalPoolConfig(
        enabled=True,
        pool_size=5,
        consensus_slots=3,
        content_dense_slots=0,
        title_dense_slots=0,
        doc_title_dense_slots=0,
        text_sparse_slots=2,
        entity_sparse_slots=0,
        title_sparse_slots=0,
        structural_slots=0,
        related_to_slots=0,
        per_doc_depth_slots=0,
    )

    # --- Reranker: DISABLED (the scenario under test) ---
    hr._reranker_enabled = False
    hr.reranker_config = types.SimpleNamespace(enabled=False, top_n=0)
    hr._reranker = None
    hr._reranker_available = False
    hr.rerank_top_n = 0

    # --- ColBERT: disabled (simplifies path; pool alone is enough) ---
    hr.colbert_rerank_enabled = False

    # --- Skip everything we don't need ---
    hr.hybrid_mode = "bge_reranker"  # skip BM25
    hr.bm25_retriever = None
    hr.fusion_method = FusionMethod.RRF
    hr.rrf_k = 60
    hr.fusion_alpha = 0.6
    hr.graph_channel_enabled = False
    hr.neo4j_disabled = True
    hr.expansion_enabled = False
    hr.expansion_query_min_tokens = 12
    hr.expansion_score_delta_max = 0.02
    hr.expansion_max_neighbors = 1
    hr.expansion_sparse_threshold = 0.0
    hr.expansion_rescoring_enabled = False
    hr.max_sources_to_expand = 5
    hr.microdoc_enabled = False
    hr.namespace_mode = "none"
    hr.context_max_tokens = 4000
    hr.context_group_cap = 3
    hr.graph_enabled = False
    hr.graph_enrichment_enabled = False
    hr._last_query_text = ""

    # Create 10 chunks. Chunks 7 and 8 have HIGH sparse scores but LOW
    # fused scores. The signal pool (consensus=3, text_sparse=2) will
    # rescue them, producing a pool that differs from raw fused[:5].
    #
    # Pool output:  fused_0, fused_1, fused_2 (consensus) + fused_7, fused_8 (sparse)
    # Raw fused[:5]: fused_0, fused_1, fused_2, fused_3, fused_4
    #
    # If the fix works, seeds contain fused_7/fused_8.
    # If the fix is reverted, seeds contain fused_3/fused_4 instead.
    fused_chunks = []
    for i in range(10):
        c = ChunkResult(
            chunk_id=f"fused_{i}",
            document_id="doc",
            parent_section_id="sec",
            order=i,
            level=1,
            heading=f"Heading {i}",
            text=f"Body text for chunk {i}",
            token_count=20,
            fused_score=1.0 - i * 0.05,
            vector_score=1.0 - i * 0.05,
            title_vec_score=0.1,  # needed for per-field detection
            lexical_vec_score=0.95 if i in (7, 8) else 0.01,
        )
        fused_chunks.append(c)

    # --- Mock vector retriever ---
    class FakeVecRetriever:
        supports_colbert = False
        schema_supports_colbert = False
        rrf_field_weights = {"content": 1.0}
        last_stats = {"path": "test", "duration_ms": 1.0}

        def search(self, query, k, filters, **kwargs):
            return list(fused_chunks)

        def get_queried_vector_fields(self):
            return ["content"]

    hr.vector_retriever = FakeVecRetriever()

    # --- Mock internal methods that we don't need ---
    hr._normalize_filters = lambda filters, caller=None: filters or {}
    hr._apply_structural_boost = lambda results, qt: 0
    hr._hydrate_missing_citations = lambda results: None
    hr._hydrate_parent_paths = lambda results: None
    hr._dedup_results = lambda results: results
    hr._annotate_coverage = lambda results: None
    hr._apply_doc_continuity_boost = lambda results, alpha=0.12: results
    hr._log_stage_snapshot = lambda label, results: None
    hr._expand_with_structure = lambda q, seeds, tag, force=False: []
    hr.tokenizer = DummyTokenizer()

    def _passthrough_budget(results, starting_tokens=0):
        tokens = starting_tokens + sum(c.token_count or 0 for c in results)
        return results, tokens

    hr._enforce_context_budget = _passthrough_budget

    # --- Call retrieve() ---
    results, metrics = hr.retrieve("general overview query", top_k=5)

    # --- Assertions ---
    # The signal pool was used
    assert metrics.get("signal_pool_used") is True
    assert metrics.get("signal_pool_before_colbert") is True
    assert metrics.get("colbert_input_from_pool") is True

    # Reranker was NOT applied
    assert metrics.get("final_reranker_applied") is False

    # The key regression check: seeds should come from the pool output,
    # NOT from raw fused_results[:5].
    #
    # Pool output: fused_0-2 (consensus) + fused_7, fused_8 (sparse rescue)
    # Raw fused[:5]: fused_0-4
    #
    # If fix works: fused_7 and/or fused_8 appear in results
    # If fix reverted: only fused_0-4 appear (sparse chunks discarded)
    result_ids = [r.chunk_id for r in results]
    assert len(result_ids) == 5

    # Consensus chunks should always be present
    for i in range(3):
        assert (
            f"fused_{i}" in result_ids
        ), f"fused_{i} should be in results (consensus slot)"

    # THE CRITICAL CHECK: sparse-rescued chunks must survive into seeds.
    # This fails if the fix is reverted (seeds would be fused_0-4, no fused_7/8).
    sparse_rescued = [cid for cid in result_ids if cid in ("fused_7", "fused_8")]
    assert len(sparse_rescued) > 0, (
        f"Sparse-rescued chunks (fused_7, fused_8) missing from seeds. "
        f"Got: {result_ids}. This indicates the pool output was discarded "
        f"and raw fused_results were used instead."
    )


# ---------------------------------------------------------------------------
# Phase B1: Post-rerank specificity adjustment tests
# ---------------------------------------------------------------------------


def test_specificity_anchor_bonus_breaks_tie():
    """Chunks with anchor in heading should get a score bump."""
    hr = _bootstrap_retriever(reranker_enabled=True)

    # Two chunks tied at 9.375
    c1 = _chunk("generic", "cluster architecture overview", 0.5)
    c1.heading = "Converged configuration"
    c1.parent_path_norm = "WEKA system overview"
    c1.rerank_score = 9.375

    c2 = _chunk("specific", "metadata management details", 0.4)
    c2.heading = "Metadata management"
    c2.parent_path_norm = "WEKA client and mount modes"
    c2.rerank_score = 9.375

    intent = _precision_intent(("metadata",))
    metrics: dict = {}

    applied, count = hr._apply_specificity_adjustment([c1, c2], intent, metrics)

    assert applied is True
    assert count >= 1
    # c2 should now score higher than c1
    assert c2.rerank_score > c1.rerank_score
    # c1 unchanged (no anchor, no deploy cue)
    assert c1.rerank_score == 9.375


def test_specificity_deploy_penalty():
    """Chunks with cloud/deploy cues in heading/path get penalized."""
    hr = _bootstrap_retriever(reranker_enabled=True)

    c1 = _chunk("cloud", "AWS deployment guide", 0.5)
    c1.heading = "Slurm based architecture"
    c1.parent_path_norm = "AWS ParallelCluster and WEKA"
    c1.rerank_score = 9.375

    c2 = _chunk("local", "metadata internals", 0.4)
    c2.heading = "Metadata management"
    c2.parent_path_norm = "WEKA system overview"
    c2.rerank_score = 9.375

    intent = _precision_intent(("metadata",))
    metrics: dict = {}

    hr._apply_specificity_adjustment([c1, c2], intent, metrics)

    # c1 penalized (deploy cues: aws, parallelcluster, slurm)
    assert c1.rerank_score < 9.375
    # c2 boosted (anchor match)
    assert c2.rerank_score > 9.375
    # Gap should be anchor_bonus + deploy_penalty = 0.25
    assert c2.rerank_score - c1.rerank_score == pytest.approx(0.25)


def test_specificity_skipped_for_non_precision():
    """Adjustment does nothing without precision_mode."""
    hr = _bootstrap_retriever(reranker_enabled=True)

    c1 = _chunk("a", "text", 0.5)
    c1.heading = "Metadata stuff"
    c1.rerank_score = 9.0

    non_precision = QueryIntent(query_type="conceptual")
    metrics: dict = {}

    applied, count = hr._apply_specificity_adjustment([c1], non_precision, metrics)

    # No anchors in non-precision intent → no adjustment
    assert applied is False
    assert c1.rerank_score == 9.0


def test_specificity_skipped_when_no_anchors():
    """No adjustment when intent has empty primary_anchors."""
    hr = _bootstrap_retriever(reranker_enabled=True)

    c1 = _chunk("a", "text", 0.5)
    c1.heading = "Architecture overview"
    c1.rerank_score = 9.0

    intent = QueryIntent(
        query_type="subsystem_architecture",
        precision_mode=True,
        primary_anchors=(),
    )
    metrics: dict = {}

    applied, _ = hr._apply_specificity_adjustment([c1], intent, metrics)
    assert applied is False
    assert c1.rerank_score == 9.0


def test_specificity_bounded_adjustment():
    """Adjustment should be bounded — not flip scores by more than ±0.25."""
    hr = _bootstrap_retriever(reranker_enabled=True)

    # Chunk with both anchor match AND deploy cue (net: +0.15 - 0.10 = +0.05)
    c1 = _chunk("mixed", "metadata on AWS", 0.5)
    c1.heading = "Metadata on AWS deployment"
    c1.parent_path_norm = ""
    c1.rerank_score = 9.5

    intent = _precision_intent(("metadata",))
    metrics: dict = {}

    hr._apply_specificity_adjustment([c1], intent, metrics)

    # Net adjustment = +0.15 (anchor) - 0.10 (deploy) = +0.05
    assert c1.rerank_score == pytest.approx(9.55)


# ---------------------------------------------------------------------------
# Phase 3: RELATED_TO blending tests
# ---------------------------------------------------------------------------


def test_blend_related_to_scores_standalone():
    """Chunks with related_to_score get blended into fused_score."""
    hr = _bootstrap_retriever(reranker_enabled=False)
    hr.config = types.SimpleNamespace(
        references=types.SimpleNamespace(
            query=types.SimpleNamespace(related_to_weight_ratio=0.15)
        ),
        feature_flags=types.SimpleNamespace(),
    )

    c1 = _chunk("a", "text a", 0.8)
    c1.related_to_score = 0.5
    c2 = _chunk("b", "text b", 0.6)
    c2.related_to_score = None  # No RELATED_TO signal
    metrics: dict = {}

    hr._blend_related_to_scores([c1, c2], "conceptual", metrics)

    # c1 should be blended: (1-0.15)*0.8 + 0.15*0.5 = 0.68 + 0.075 = 0.755
    assert c1.fused_score == pytest.approx(0.755)
    # c2 unchanged
    assert c2.fused_score == 0.6
    assert metrics["related_to_blend_count"] == 1
    assert metrics["related_to_blend_lambda"] == pytest.approx(0.15)


def test_blend_cli_disabled():
    """CLI query type gets lambda=0.0, no blending."""
    hr = _bootstrap_retriever(reranker_enabled=False)
    hr.config = types.SimpleNamespace(
        references=types.SimpleNamespace(
            query=types.SimpleNamespace(related_to_weight_ratio=0.15)
        ),
        feature_flags=types.SimpleNamespace(),
    )

    c1 = _chunk("a", "text", 0.8)
    c1.related_to_score = 0.5
    metrics: dict = {}

    hr._blend_related_to_scores([c1], "cli", metrics)

    assert c1.fused_score == 0.8  # Unchanged
    assert metrics["related_to_blend_count"] == 0
    assert metrics["related_to_blend_lambda"] == 0.0


def test_graph_channel_decoupled_from_enrichment():
    """Graph channel should run when use_entity_graph_channel=True,
    even when use_graph_enrichment=False. Before decoupling,
    _graph_retrieval_channel() required both graph_channel_enabled
    AND graph_enabled (derived from enrichment)."""
    hr = _bootstrap_retriever(reranker_enabled=False)
    hr._plan = ResolvedRetrievalPlan(
        profile=RetrievalProfile.GRAPH_ASSISTED,
        use_entity_graph_channel=True,
        use_graph_enrichment=False,  # enrichment OFF
        graph_garbage_filter_on=True,
        graph_score_normalized_on=True,
    )
    hr.neo4j_disabled = False
    hr.graph_channel_enabled = False  # legacy flag is OFF
    hr.graph_enabled = False  # legacy derived flag is OFF
    hr.graph_adaptive_enabled = True
    hr.graph_relationships = ["MENTIONS"]
    hr.graph_max_related = 20
    hr.graph_max_depth = 3
    hr.config = types.SimpleNamespace(
        feature_flags=types.SimpleNamespace(),
        ner=types.SimpleNamespace(enabled=False),
        search=types.SimpleNamespace(
            hybrid=types.SimpleNamespace(
                query_type_relationships={
                    "conceptual": ["MENTIONS"],
                    "subsystem_architecture": ["MENTIONS"],
                },
            ),
        ),
    )

    # Mock entity extractor to return entities
    class FakeExtractor:
        def extract_entities(self, query):
            return ["metadata", "tiering"]

    hr._entity_extractor = FakeExtractor()

    # Mock neo4j driver to return empty results (we just need to verify
    # the method doesn't short-circuit at the gate)
    class FakeSession:
        def run(self, cypher, **kwargs):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeDriver:
        def session(self):
            return FakeSession()

    hr.neo4j_driver = FakeDriver()

    result_chunks, stats = hr._graph_retrieval_channel("metadata query", None)

    # The key assertion: method did NOT short-circuit at the gate.
    # It reached entity extraction and the Neo4j query (returning empty).
    # With the old coupled gate (graph_channel_enabled AND graph_enabled),
    # both were False, so it would have returned immediately with
    # graph_channel_entities=0 before ever calling extract_entities().
    #
    # After decoupling, the gate only checks self._plan.use_entity_graph_channel
    # (True) and self.neo4j_disabled (False), so it passes through.
    assert stats["graph_channel_entities"] == 2  # "metadata" + "tiering"


def test_graph_channel_injects_precision_anchors():
    """When GLiNER returns nothing useful, primary_anchors from QueryIntent
    should still provide graph anchors so the channel has terms to query."""
    hr = _bootstrap_retriever(reranker_enabled=False)
    hr._plan = ResolvedRetrievalPlan(
        profile=RetrievalProfile.GRAPH_ASSISTED,
        use_entity_graph_channel=True,
        use_graph_enrichment=False,
        graph_garbage_filter_on=True,
        graph_score_normalized_on=True,
    )
    hr.neo4j_disabled = False
    hr.graph_channel_enabled = False
    hr.graph_enabled = False
    hr.graph_adaptive_enabled = True
    hr.graph_relationships = ["MENTIONS"]
    hr.graph_max_related = 20
    hr.graph_max_depth = 3
    hr.config = types.SimpleNamespace(
        feature_flags=types.SimpleNamespace(),
        ner=types.SimpleNamespace(enabled=False),
        search=types.SimpleNamespace(
            hybrid=types.SimpleNamespace(
                query_type_relationships={
                    "subsystem_architecture": ["MENTIONS"],
                },
            ),
        ),
    )

    # GLiNER returns only "weka" (garbage-filtered away at len < 4)
    class WeakExtractor:
        def extract_entities(self, query):
            return ["we"]  # too short, filtered by garbage filter

    hr._entity_extractor = WeakExtractor()

    class FakeSession:
        def run(self, cypher, **kwargs):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeDriver:
        def session(self):
            return FakeSession()

    hr.neo4j_driver = FakeDriver()

    # Intent with primary_anchors — these should be injected
    intent = QueryIntent(
        query_type="subsystem_architecture",
        precision_mode=True,
        subsystem_terms=("metadata",),
        primary_anchors=("metadata",),
        generic_modifiers=("managed",),
    )

    result_chunks, stats = hr._graph_retrieval_channel(
        "how is metadata managed", None, intent=intent
    )

    # "we" filtered by garbage filter (< 4 chars), but "metadata" injected from anchors
    assert stats["graph_channel_entities"] >= 1
    sources = stats.get("graph_anchor_sources", {})
    assert "metadata" in sources.get("precision_anchors", [])
    assert sources["merged"] >= 1

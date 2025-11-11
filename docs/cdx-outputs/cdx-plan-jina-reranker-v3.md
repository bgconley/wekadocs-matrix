# Plan: Jina Reranker v3 Integration (Option 1 – Drop‑In Hook)

## Goal
Wire Jina Reranker v3 into the Phase 7E retrieval path by implementing a drop-in hook in `HybridRetriever._apply_reranker`, reusing the existing provider factory, config knobs, and metrics. Keep the rest of the retrieval pipeline unchanged.

## Scope
- Implement only the reranker handoff in the Phase 7E retriever (no refactor to Phase 7C engine, no new async stage).
- Ensure provider and factory support v3 defaults and response shape.
- Add unit tests and optional integration test behind `JINA_API_KEY`.
- Update config samples and docs.

## Assumptions
- The Jina API key is provided via `JINA_API_KEY`.
- Reranker config is controlled at `config.search.hybrid.reranker` (enabled, model, top_n).
- Existing Prometheus counters/histograms for reranking are already declared and used in the provider.

## Deliverables
- Functional `_apply_reranker` in the 7E retriever that calls Jina Reranker v3 via the provider factory.
- Updated provider/factory defaults for v3.
- Tests proving reordering and error/fallback behavior.
- Documentation and sample config updates.

## Plan

### 1) Provider and Factory Readiness (v3)
- Files:
  - `src/providers/rerank/jina.py`
  - `src/providers/factory.py`
- Tasks:
  - Extend `JinaRerankProvider` to robustly parse v3 responses. Detect fields at runtime:
    - Keep current handling for v2 (e.g., `{"results": [{"index": ..., "relevance_score": ...}]}`).
    - Add v3 path (if schema differs): gracefully map to `{"rerank_score", "original_rank", "reranker"}`.
  - Update factory default model:
    - In `ProviderFactory.create_rerank_provider()`, set `RERANK_MODEL` default to `jina-reranker-v3` (still overrideable by env).
  - Validate `top_n` behavior: if v3 enforces different limits, clamp to `min(top_n, len(candidates))`.
- Acceptance:
  - Provider returns a list of candidates each with `rerank_score`, `original_rank`, and preserves the original candidate content.
  - Factory defaults to v3 when `RERANK_MODEL` not specified.

### 2) Implement HybridRetriever Hook
- Files:
  - `src/query/hybrid_retrieval.py`
- Tasks:
  - In `HybridRetriever.__init__`:
    - Keep current behavior; do not require a reranker instance to be passed from callers.
    - Record `self.reranker_config = config.search.hybrid.reranker`.
    - Add a lazy field `self._reranker = None`.
  - Implement `_get_reranker()`:
    - If `self._reranker` is None and `self.reranker_config.enabled` is True, instantiate via `ProviderFactory.create_rerank_provider()` (reads env/config), catching and logging initialization errors. On error, leave `self._reranker=None`.
  - Implement `_apply_reranker(query, seeds, metrics)`:
    - Early exit when no seeds or reranker disabled/unavailable:
      - `metrics["reranker_applied"] = False`
      - `metrics["reranker_reason"] = "disabled" | "not_available"`
      - Return original seeds.
    - Candidate packaging:
      - For each `ChunkResult`, build `{ "id": chunk.chunk_id, "text": chunk.text or chunk.heading, "original_result": chunk }`. Skip candidates with empty text; log a warning once if all skipped.
    - Call reranker:
      - Compute `top_k = min(self.reranker_config.top_n or len(candidates), len(candidates))`.
      - Time the call; on success:
        - Update each referenced `ChunkResult`:
          - `chunk.fused_score = reranked_cand["rerank_score"]`
          - `chunk.metadata["rerank_score"] = ...`
          - `chunk.metadata["original_rank"] = reranked_cand.get("original_rank", 0)`
          - `chunk.metadata["reranker"] = reranked_cand.get("reranker", "jina-reranker-v3")`
          - `chunk.fusion_method = "rerank"`
        - Sort by `rerank_score` descending, truncate to `top_k`.
        - Metrics:
          - `metrics["reranker_applied"] = True`
          - `metrics["reranker_reason"] = "ok"`
          - `metrics["reranker_model"] = self._reranker.model_id`
          - `metrics["reranker_time_ms"] = elapsed_ms`
      - On exception:
        - Log structured warning with error type and message.
        - Set `metrics["reranker_applied"] = False`
        - `metrics["reranker_reason"] = "provider_error"`
        - Return original seeds unchanged.
- Acceptance:
  - With a fake provider, results reorder deterministically and metrics reflect application.
  - On provider failure, original ordering returns and metrics indicate a failure reason.

### 3) Minimal Service Wiring (Optional)
- Files:
  - `src/mcp_server/query_service.py`
- Tasks:
  - No change required since retriever lazily creates the provider. Optionally:
    - If `search.hybrid.reranker.enabled=True`, add an info log at service startup noting reranker activation for visibility.
- Acceptance:
  - No regressions in the service. Reranker activation controlled by config/env only.

### 4) Tests
- Files:
  - `tests/query/test_reranker_integration.py` (new)
  - `tests/query/test_vector_store.py` (unchanged)
- Unit tests:
  - Fake provider rerank:
    - Inject a fake `RerankProvider` into `HybridRetriever` or monkeypatch the factory; have it return reversed scores.
    - Assert `_apply_reranker` reorders seeds, sets `fused_score`, writes metadata (`rerank_score`, `original_rank`, `reranker`), and marks metrics `reranker_applied=True` with `reranker_time_ms > 0`.
  - Failure path:
    - Fake provider raises `RuntimeError`; ensure original order returns and `metrics["reranker_reason"] == "provider_error"`.
  - Candidate packaging:
    - Seeds with empty text/heading result in early exit; metrics set to `reranker_applied=False`, `reranker_reason="no_text"`.
- (Optional) Integration test:
  - Skip unless `JINA_API_KEY` present.
  - Run a small query, assert metrics show reranker applied and chunk metadata contains rerank fields.
- Acceptance:
  - New unit tests pass locally and in CI. Integration test runs only when API key present.

### 5) Config and Docs
- Files:
  - `docs/hybrid-rag-v2_2-testing.md`
  - `.env.example`
  - `docs/hybrid-rag-v2_2-changelog.md`
- Tasks:
  - Add env vars in `.env.example`:
    - `RERANK_PROVIDER=jina-ai`
    - `RERANK_MODEL=jina-reranker-v3`
    - `JINA_API_KEY=...`
    - Optional: `RERANK_TOP_N=20`
  - Document how to enable reranking (`search.hybrid.reranker.enabled: true`).
  - Update change log’s “Open Follow-Ups” to mark reranker integration as completed once merged.
- Acceptance:
  - Clear operator guidance on enabling reranking in dev/staging/prod.

### 6) Performance/Operational Readiness
- Tasks:
  - Verify p95/p99 latency impact when enabling reranking (baseline a few queries).
  - Confirm metrics visibility:
    - Rerank request total/success/error counts
    - Rerank latency histogram
    - Service logs include reranker model, timing, and reasons on failure/skip
  - Set conservative defaults:
    - `enabled: false` by default in config, gated rollout per environment.
- Acceptance:
  - No unexpected latency regressions or rate-limit spikes. Observability shows reranker activity.

## Risks & Mitigations
- **Rate limits / 429s**: Provider’s rate limiter and circuit breaker already handle backoff; we still log and fallback gracefully.
- **Candidate text quality**: If many chunks lack meaningful text, reranking adds little. Ensure chunk assembly keeps content intact; fallback to `heading` when `text` is empty.
- **API changes in v3**: Provider maps both v2/v3 shapes; add a unit test with a v3-style mock payload to lock behavior.

## Rollback Plan
- Config-only: Disable feature with `search.hybrid.reranker.enabled=false` and redeploy. No code revert needed.

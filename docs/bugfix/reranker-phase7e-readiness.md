# Phase 7E HybridRetriever – Reranker Implementation Guide

## Why this exists
- `config.search.hybrid.reranker` is exposed, and metrics surfaces expect rerank status.
- The Phase 7C `HybridSearchEngine` already supports reranking, but the Phase 7E `HybridRetriever` only logs a warning and returns seeds unchanged.
- This note captures the attachment points so a future agent can implement the capability without re-discovering context.

## Current stub behavior
1. `HybridRetriever.__init__` stores `reranker_config` and emits a warning when `enabled=True`.
2. `_apply_reranker` is invoked right after fusion (pre-expansion) and now records metrics:
   - `metrics["reranker_applied"] = False`
   - `metrics["reranker_reason"] = "disabled" | "not_implemented"`
3. `_get_7e_retriever` in `QueryService` warns whenever reranking is enabled so operators know the knob is a no-op.

## Target architecture for real reranking
1. **Provider plumbing**
   - Accept an optional `RerankProvider` instance in `HybridRetriever.__init__`.
   - Allow dependency injection (tests) and fallback to `ProviderFactory.create_rerank_provider()` when config enables reranking.
2. **Candidate construction**
   - `_apply_reranker` should convert each `ChunkResult` seed to the canonical dict expected by `RerankProvider.rerank`:
     ```python
     candidate = {
         "id": chunk.chunk_id,
         "text": chunk.text or chunk.heading,
         "metadata": {... optional extras ...},
         "original_result": chunk,
     }
     ```
   - Filter non-empty text; consider truncating long chunks to stay within provider token budgets.
3. **Reranker call**
   - Call `self.reranker.rerank(query, candidates, top_k)` where `top_k` defaults to `min(len(seeds), reranker_config.top_n or 20)`.
   - Handle provider exceptions by logging and falling back to the original seeds (maintaining fusion order).
4. **Result adaptation**
   - Update each `ChunkResult` referenced by reranked candidates:
     - `chunk.fused_score = reranked_score`
     - annotate metadata, e.g. `chunk.rerank_score`, `chunk.fusion_method = "rerank"`.
   - Sort and truncate to `top_k` before returning to the caller.
5. **Metrics & monitoring**
   - Record `metrics["reranker_applied"] = True`, `metrics["reranker_reason"] = "ok"`, `metrics["reranker_model"]`, and `metrics["reranker_time_ms"]`.
   - Increment relevant counters in `src/shared/observability/metrics.py` (mirroring the Phase 7C implementation).

## Testing checklist
1. **Unit tests**
   - Inject a fake reranker that reverses scores, assert `_apply_reranker` reorders seeds and metrics flip to `True`.
   - Verify fallbacks when the provider raises.
2. **Integration / e2e**
   - Use the Jina reranker via `ProviderFactory` with a mock or live key.
   - Ensure metrics dashboards show reranker activity and that latency budgets remain acceptable.
3. **Config validation**
   - Add a regression test that enabling reranking without a provider raises a clear error, preventing silent failure.

## Status (2025-11-23)
- Reranker wiring remains but `config/search.hybrid.reranker.enabled=false` so the feature is off by default (same as pre-activation).
- `_apply_reranker` still guards empty chunks and preserves fused scores, but with the flag disabled the hybrid pipeline behaves exactly as before (metrics show `reranker_applied=False`).
- QueryService continues to surface rerank metadata for future experimentation, yet the default MCP/server flow is the original hybrid-only ordering.

## Suggested rollout steps
1. Implement provider injection + `_apply_reranker` logic described above.
2. Add feature flag / config entry to allow gradual rollout (start with `enabled=false`).
3. Validate in staging with both noop and real Jina reranker, monitoring latency and accuracy.
4. Update documentation (Phase 7E integration plan) once reranking is live.

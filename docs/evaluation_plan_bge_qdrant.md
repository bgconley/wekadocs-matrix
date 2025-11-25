# BGE-M3 + Qdrant Hybrid Retrieval Evaluation Plan

## Goals
1. Validate that the Query API + ColBERT rerank path outperforms or matches the legacy multi-search pipeline.
2. Provide repeatable, configuration-driven measurements (latency, recall@K, fusion metrics) for each embedding profile (initial focus: `bge_m3`, baseline `jina_v3`).
3. Gate rollout via feature flags (`enable_sparse`, `enable_colbert`, `use_query_api`) with clear success criteria and regression alarms.

## Prerequisites
- Local services running via `docker-compose` (`qdrant`, `neo4j`, `redis`), optional remote equivalents.
- Strict-mode ingestion completed for the targeted embedding profile(s) so Qdrant contains native `text-sparse` + `late-interaction` vectors.
- `PROFILE_MATRIX` smoke job green for the profile under test (ensures ingestion/regression parity).

## Gold Set Construction
1. **Query harvesting:** Collect ~30–50 real support/engineering queries covering:
   - CLI flags / error codes (lexical emphasis)
   - Workflow/how-to questions (dense/semantic)
   - Graph-sensitive prompts (multi-hop adjacency)
2. **Labeling:** For each query, annotate:
   - `expected_section_ids` (top 5 relevant sections)
   - `must_contain_terms` or `doc_tag` if scoped
3. Store annotations under `reports/eval/gold_sets/<profile>/queries.yaml`.

## Metrics & Logging
**Automatic captures (via `HybridRetriever`):**
- `vector_path` (`legacy` / `query_api`)
- `vec_time_ms`, `bm25_time_ms`, `fusion_time_ms`, `context_assembly_ms`, `total_time_ms`
- `vector_prefetch_count`, `vector_colbert_used` (when Query API on)
- Candidate counts per stage (BM25, vector, fused, final)

**Derived metrics computed by evaluation harness:**
- Recall@K (K ∈ {5,10,20})
- MRR@K
- Latency percentiles per channel (P50/P95)
- Query API win/loss breakdown vs legacy (by recall and latency)

## Evaluation Procedure
1. Ensure feature flags reflect desired comparison:
   - Baseline run: `use_query_api=false` (legacy)
   - Experimental run: `use_query_api=true`, `enable_sparse=true`, `enable_colbert=true`
2. Run the harness (`python scripts/eval/run_eval.py --profile bge_m3 --gold reports/eval/gold_sets/bge_m3/queries.yaml --flags use_query_api=true`) which:
   - Hits the MCP query endpoint or directly instantiates `HybridRetriever`.
   - Logs metrics JSON lines to `reports/eval/run_<timestamp>.jsonl`.
   - Outputs summary tables (recall/latency) and diff vs baseline.
3. Upload summary to `reports/eval/summary_<date>.md` with sections:
   - Dataset overview
   - Metrics table (legacy vs query API)
   - Observations / regressions / action items

## Observability Hooks
- Prometheus metrics already include `retrieval_expansion_*` and total latency; add dashboard panels for:
  - `vector_path` distribution (counts of legacy vs query_api)
  - `vec_time_ms` P95 by path
  - Prefetch counts vs candidate limits
- SLO alarms:
  - Vector path P95 > target (e.g., 500 ms)
  - Query API fallback rate > 5%

## Rollout Checklist
1. **Dev/staging:** Run evaluation harness; confirm query API path meets or exceeds recall and respects latency budgets.
2. **Shadow mode:** Enable `use_query_api=true` in non-prod, collect metrics for 24h, ensure fallback rate remains <5%.
3. **Prod rollout:** Toggle feature flag per profile; monitor dashboards + SLO alarms; revert if regressions observed.

## Next Steps
- Implement `scripts/eval/run_eval.py` harness (CLI to run gold set through `HybridRetriever`).
- Add GitHub Actions workflow to run evaluation harness nightly (non-blocking) and publish summary artifacts.
- Extend metrics aggregator to bucket stats by `vector_path` for longer-term analysis.

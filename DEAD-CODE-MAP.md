# Dead Code Map — Quick Reference

> ⚠️ **CORRECTION (see AUDIT-VERIFICATION.md):** Two earlier "dead" calls were WRONG.
> `src/query/planner.py` is ACTIVE (called by query_service.py:827). `src/query/hybrid_search.py`
> is a runtime-reachable legacy path in active query_service.py — de-couple before deleting, not a Phase-0 delete.

> **Key:** 🟢 = ACTIVE | 🟡 = MIXED | 🔴 = DEAD | ⚪ = DORMANT (config-gated)

---

## 🔴 ENTIRELY DEAD PACKAGES (delete the directory)

| Package | LOC | Last Use | Tag |
|---------|-----|----------|-----|
| `src/learning/` | 983 | Phase 4 (never integrated) | `@safe-to-delete: Yes` |
| `src/registry/` | 282 | Phase 7C (never integrated) | `@safe-to-delete: Yes` |
| `src/ops/optimizer.py` | 499 | Phase 4 (test-only) | |
| `src/ops/warmers/` | 162 | Phase 4 (test-only) | |
| `src/mcp_server/validation.py` | 407 | Written, never wired | `@safe-to-delete: Yes` |
| `src/mcp_server/security/` | 355 | JWTAuth + RateLimiter, never wired | `@safe-to-delete: Yes` |

## 🔴 INDIVIDUALLY DEAD FILES (delete the file)

| File | LOC | Reason |
|------|-----|--------|
| `src/ingestion/api.py` | 35 | Dead test wrapper |
| `src/ingestion/reconcile.py` | 525 | Never called in production |
| `src/ingestion/incremental.py` | 363 | Never instantiated |
| `src/ingestion/parsers/notion.py` | 256 | Notion not supported |
| `src/ingestion/auto/watcher.py` | 54 | Deprecated, zero imports |
| `src/ingestion/auto/orchestrator.py` | 1,134 | Superseded |
| `src/ingestion/auto/verification.py` | 277 | Only used by dead orchestrator |
| `src/ingestion/auto/report.py` | 296 | Only used by dead orchestrator |
| `src/ingestion/auto/backpressure.py` | 282 | Never instantiated |
| `src/query/diffusion_reranker.py` | 362 | `@status: DEAD` |
| `src/query/graph_features.py` | 339 | `@status: DEAD` |
| `src/query/graph_expansion.py` | 302 | `@status: DEAD` |
| ~~`src/query/planner.py`~~ | 358 | ❌ **NOT DEAD — see AUDIT-VERIFICATION.md.** Imported & called by `query_service.py:827` (`planner.plan()`). DO NOT DELETE. |
| `src/query/templates/advanced/schemas.py` | 150 | `@status: TEST_ONLY` |
| `src/neo/structural_builder.py` | 520 | Superseded |
| `src/neo/graph_enhancements.py` | 440 | No external imports |
| `src/neo/entity_normalization.py` | 274 | No external imports |
| `src/neo/explain_guard.py` | 276 | `@status: DEAD` |
| `src/neo/defensive_query.py` | 112 | Never called |
| `src/neo/health.py` | 117 | Superseded |
| `src/shared/feature_flags.py` | 173 | Superseded by config.py |
| `src/shared/audit/logger.py` | 207 | Never called |

## 🟡 MIXED-STATUS FILES (strip dead methods only)

| File | Active LOC | Dead LOC | What to remove |
|------|-----------|----------|----------------|
| `src/ingestion/build_graph.py` | ~300 | ~2,900 | All write methods (upsert_document + 35 helpers) |
| `src/ingestion/saga.py` | ~300 | ~380 | SagaCoordinator, IngestionSagaBuilder, SagaStep |
| `src/neo/contract_checks.py` | ~30 | ~425 | 6 dead check methods |
| `src/query/hybrid_search.py` | ~0 | ~916 | Entire file is legacy (but still imported by ranking.py) |
| `src/query/ranking.py` | ~100 | ~400 | `Ranker` class — never called by active retrieval path |
| `src/query/fusion_pipeline.py` | ~100 | ~117 | Weighted fusion path (HybridRetriever hard-codes RRF) |

## ⚪ DORMANT FILES (config-gated, not loaded normally)

| File | LOC | Gate | Risk of deletion |
|------|-----|------|------------------|
| `src/ingestion/parsers/markdown.py` | 389 | `config.parser.engine="legacy"` | Low — markdown-it-py is default |
| `src/ingestion/parsers/shadow_comparison.py` | 257 | `config.parser.shadow_mode=true` | Medium — useful diagnostic |
| `src/ingestion/extract/ner_gliner.py` | 226 | `config.ner.enabled` | Low — feature rarely used |

---

## Dead Code Heatmap

```
src/
├── ingestion/     ████████░░  21% dead
│   ├── build_graph.py    ██████████  90% dead
│   ├── saga.py           ██████░░░░  55% dead
│   └── auto/             ██████░░░░  55% dead
├── query/         █░░░░░░░░░  ~7% confirmed dead (1,003 LOC) + hybrid_search 916 once de-coupled
│   └── hybrid_search.py  ██████████  legacy but LIVE (query_service.py imports+uses it)
├── providers/     ██░░░░░░░░  7% dead (mostly healthy)
├── mcp_server/    ███░░░░░░░  10% dead
│   └── security/         ██████████  100% dead
├── services/      █░░░░░░░░░  <5% dead (healthy)
├── neo/           ██████████  90% dead (massive)
├── shared/        █░░░░░░░░░  5% dead (healthy)
├── learning/      ██████████  100% dead (delete)
├── registry/      ██████████  100% dead (delete)
└── ops/           ██████████  100% dead (delete)
```

---

## Quick Kill List (copy-paste for `rm`)

```bash
# Entirely dead packages
rm -rf src/learning/
rm -rf src/registry/
rm -rf src/ops/optimizer.py src/ops/warmers/
rm -rf src/mcp_server/validation.py src/mcp_server/security/
rm -rf src/shared/feature_flags.py src/shared/audit/

# Individually dead files
rm src/ingestion/api.py
rm src/ingestion/reconcile.py
rm src/ingestion/incremental.py
rm src/ingestion/parsers/notion.py
rm src/ingestion/auto/watcher.py
rm src/ingestion/auto/orchestrator.py
rm src/ingestion/auto/verification.py
rm src/ingestion/auto/report.py
rm src/ingestion/auto/backpressure.py
rm src/query/diffusion_reranker.py
rm src/query/graph_features.py
rm src/query/graph_expansion.py
# rm src/query/planner.py   # ❌ REMOVED — planner.py is ACTIVE (query_service.py:827). See AUDIT-VERIFICATION.md
rm src/query/templates/advanced/schemas.py
rm src/neo/structural_builder.py
rm src/neo/graph_enhancements.py
rm src/neo/entity_normalization.py
rm src/neo/explain_guard.py
rm src/neo/defensive_query.py
rm src/neo/health.py
```

Then update `__init__.py` files to remove dead imports.

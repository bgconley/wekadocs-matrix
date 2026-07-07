# Audit Verification — Accuracy Review of Session Artifacts

> Independent verification of claims in REPO-MAP.md, DEAD-CODE-MAP.md, REFACTOR-PLAN.md, ARCHITECTURE.md, DEV-HISTORY.md
> Method: grep/wc against actual source, not subagent self-reports.
>
> **STATUS: ALL ERRORS BELOW HAVE BEEN CORRECTED IN THE ARTIFACTS (2026-06-03).** Each error is marked ✅ RESOLVED.

## Verdict Summary

The artifacts were **directionally correct but contained material errors** that would cause breakage if acted on literally. They were built from subagent self-reports that were never independently verified. Two recommended deletions were WRONG and would have broken the running system. All have now been corrected.

---

## CONFIRMED ACCURATE

| Claim | Verification |
|-------|--------------|
| `atomic.py` = 4,052 LOC | ✅ exact |
| `build_graph.py` = 3,200 LOC | ✅ exact |
| `mcp_app.py` = 3,834 LOC | ✅ exact |
| `hybrid_retrieval.py` = 2,095 LOC | ✅ exact |
| `config.py` = 2,015 LOC | ✅ exact |
| `src/learning/` is dead | ✅ 0 imports in src/ |
| `src/registry/` is dead | ✅ only self-import in its own __init__ |
| `src/ops/optimizer.py`, `ops/warmers/` dead | ✅ 0 imports |
| `mcp_server/security/` dead | ✅ 0 imports (confirmed via 2 greps) |
| `mcp_server/validation.py` dead | ✅ 0 imports |
| `shared/feature_flags.py`, `shared/audit/` dead | ✅ 0 imports |
| `neo/` mostly dead (explain_guard, entity_normalization, graph_enhancements, structural_builder, defensive_query, health) | ✅ apparent import-refs are all docstring "Usage:" examples, not real imports |
| `auto/orchestrator.py`, `verification.py`, `report.py`, `backpressure.py`, `watcher.py` dead | ✅ verification only imported by orchestrator; orchestrator only referenced by dead reconcile.py comment |
| `query/diffusion_reranker.py`, `graph_features.py`, `graph_expansion.py` dead | ✅ only self/mutual references within the dead cluster |
| build_graph.py: atomic.py uses only 8 members (ensure_embedder, embedder, embedding_plan/settings, colbert_*, 2 text builders) | ✅ confirmed — rest of GraphBuilder is unused by active path |

---

## ERRORS FOUND (must fix before acting)

### ERROR 1 — `src/query/planner.py` is NOT dead (CRITICAL) — ✅ RESOLVED
- **Artifact claim:** DEAD-CODE-MAP and REFACTOR-PLAN list `planner.py` (358 LOC) for deletion; REPO-MAP marked it "UNUSED / no active production imports."
- **Reality:** `mcp_server/query_service.py:26` imports `QueryPlanner` at top level. Line 474 instantiates it. **Line 827 calls `planner.plan(query, filters=filters)` on the live search path** to classify query intent.
- **Impact if deleted:** ImportError on MCP server startup → entire query service broken.
- **✅ RESOLVED:** planner.py re-labeled ACTIVE in REPO-MAP; removed from kill lists in DEAD-CODE-MAP and REFACTOR-PLAN (commented out in the bash kill-list).

### ERROR 2 — `src/query/hybrid_search.py` is NOT purely legacy/dead (CRITICAL) — ✅ RESOLVED
- **Artifact claim:** "100% legacy," recommended for deletion after de-coupling ranking.py.
- **Reality:** `query_service.py:25` imports `HybridSearchEngine, QdrantVectorStore, SearchResult` at top level; line 246 instantiates `HybridSearchEngine(...)`; line 1035 comment confirms a live "legacy HybridSearchEngine already orders results" path. Also `query_service.py:233` lazily imports `Neo4jVectorStore` from it. Additionally `ranking.py:23` imports `SearchResult` from it, and `ranking.py`'s `RankedResult`/`RankingFeatures` ARE used by `response_builder.py` and `query_service.py` — so the dependency chain is live.
- **Impact if deleted:** ImportError on query service + loss of a runtime-reachable retrieval path.
- **✅ RESOLVED:** Re-labeled "LEGACY (live)" in REPO-MAP; REFACTOR-PLAN now states it is runtime-reachable and NOT a simple delete (de-couple first).

### ERROR 3 — Summary rollup LOC was wrong (the per-module data was right) — ✅ RESOLVED
- **Artifact claim:** REPO-MAP Quick-Stats said "~35,000 active + ~9,000 dead."
- **Reality:** `src/` is **68,551 LOC across 169 files** (measured). The per-module LOC numbers in the detailed tables were actually CORRECT and sum exactly to 68,551. The error was confined to the **headline summary table** — it under-reported active LOC by ~20,000 and the rollup conflated full-file dead with in-file dead methods.
- **Correction of my own earlier note:** My first-pass claim that "~50 files were never analyzed" was itself imprecise. Coverage was near-complete (per-module sums reconcile exactly). `src/providers/__init__.py` was named in analysis but does NOT exist (no top-level providers package init); subdirectory `__init__.py` files were the only thin gaps.
- **✅ RESOLVED:** REPO-MAP Quick-Stats replaced with measured 68,551 total, a per-module LOC table, and active≈55,000 / full-file-dead=9,038 split. Dead-Code Summary table rebuilt with measured numbers separating full-file dead from in-file dead methods.

### ERROR 4 — Test/quality numbers are unverified (RIGOR) — ⚠️ FLAGGED (not yet re-measured)
- **Artifact claim:** DEV-HISTORY cites "411 tests, 93.4% pass rate," "44/44," "21/21," etc.
- **Reality:** These came from context-file narratives (historical Oct 2025 claims), NOT from running the suite this session. The test suite was never executed or even collected here (the runtime-environment probe was blocked).
- **✅ RESOLVED (labeling):** DEV-HISTORY now carries a header stating all metrics are historical claims, not current measurements. **Still outstanding:** actually running the suite — requires the Python/venv environment.

---

## Minor Corrections (also applied)

- **CircuitBreaker count:** artifacts said "2 CircuitBreaker implementations." Actual = **3** (`connectors/circuit_breaker.py`, `providers/embeddings/jina.py`, `shared/resilience/circuit_breaker.py`), all active. Fixed in REPO-MAP.
- **`ranking.py` status:** partially active — `Ranker.rank()` is bypassed, but its `RankedResult`/`RankingFeatures` dataclasses are imported by `response_builder.py:25` and `query_service.py:27,346,445`. Keep the file; only the Ranker path is dead.
- **`ops/session_cleanup_job.py`:** NOT dead — it is a standalone CLI with `__main__` (line 126); excluded from ops dead-LOC.
- **`src/providers/__init__.py`:** does not exist (no top-level providers package init) — earlier analysis named it in error.

---

## Reliability Assessment by Artifact (after corrections)

| Artifact | Reliability (now) | Notes |
|----------|-------------------|-------|
| REPO-MAP.md | High | LOC now measured & reconciled to 68,551; planner/hybrid_search status fixed; dead-code table rebuilt |
| DEAD-CODE-MAP.md | High | 2 false-positive kills removed; correction banner added; bash kill-list patched |
| ARCHITECTURE.md | High | Diagrams reflect real import structure; consistent with verification (no changes needed) |
| DEV-HISTORY.md | High (intent) / labeled (numbers) | Narrative accurate; metrics now flagged as historical |
| REFACTOR-PLAN.md | High | Phasing sound; dangerous deletions removed/re-scoped |

---

## Required Corrections — STATUS

1. ✅ Remove `src/query/planner.py` from ALL kill lists — DONE (DEAD-CODE-MAP + REFACTOR-PLAN).
2. ✅ Re-scope `src/query/hybrid_search.py` from "Phase 0 delete" to "de-couple first" — DONE.
3. ✅ Reconcile LOC totals — DONE (measured per-module table added; summary fixed).
4. ✅ Label test/pass-rate numbers as historical — DONE (DEV-HISTORY header). ⚠️ Re-measurement still pending (needs venv).
5. ✅ Documented the 8 GraphBuilder members atomic.py depends on — preserved-list recorded for build_graph trim.

## Bottom Line

The mapping effort was a solid ~80%-correct foundation that inherited the classic risk of trusting subagent self-reports. Two of its dead-code recommendations were actively wrong and would have broken the system. **All four errors have now been corrected in the artifacts**, with measured LOC reconciled to 68,551 and the two dangerous deletions removed/re-scoped. The one remaining gap is empirical: the test suite has been labeled historical but not yet re-run (blocked on the Python/venv environment). No deletion should proceed until that safety net is actually executed.

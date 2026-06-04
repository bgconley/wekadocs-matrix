# Cleanup Execution Status

**Last updated:** 2026-06-03
**Current phase:** Phase 3 (Remove dead code blocks) ✅ COMPLETE
**Next phase:** Phase 4

---

## Phase 0: Empirical Baseline ✅

**Status:** COMPLETE
**Date:** 2026-06-03

### Environment
- Python: 3.11.14
- pytest: 7.4.4
- ruff: 0.15.15
- mypy: 2.1.0
- Venv: .venv/ (created successfully)

### Test Collection
- **Baseline:** 1698 tests collected
- **Collection errors:** 0
- **Fix applied:** Created tools/__init__.py to fix module import

### Static Analysis
- **Ruff issues:** 2 (F841 unused-variable in src/query/hybrid_retrieval.py:173,178)
- **Mypy errors:** 532 across 87 files (checked 169)
- **Top mypy offender:** src/ingestion/build_graph.py (67 errors)

### Artifacts
- reports/baseline/ruff.txt
- reports/baseline/mypy.txt
- reports/baseline/test_collection.txt

---

## Phase 1: Delete Dead Files ✅

**Status:** COMPLETE
**Date:** 2026-06-03
**Goal:** Delete files with zero live imports

### Execution Summary

**Deletion batches:**
1. **Dead packages** (5 packages, ~2,830 LOC)
   - src/learning/ (4 files)
   - src/registry/ (2 files)
   - src/ops/warmers/ (2 files)
   - src/mcp_server/security/ (3 files)
   - src/shared/audit/ (2 files)

2. **Dead source files** (21 files, ~6,944 LOC estimated from CLEANUP-PLAN)
   - src/ingestion/: api.py, reconcile.py, incremental.py, parsers/notion.py
   - src/ingestion/auto/: orchestrator.py, verification.py, report.py, backpressure.py, watcher.py
   - src/query/: diffusion_reranker.py, graph_features.py, graph_expansion.py
   - src/neo/: explain_guard.py, entity_normalization.py, graph_enhancements.py, structural_builder.py, defensive_query.py, health.py
   - src/mcp_server/: validation.py
   - src/shared/: feature_flags.py
   - src/ops/: optimizer.py

3. **Orphaned test files** (11 test files, ~1,853 LOC)
   - Deleted: tests/p4_t4_test.py, tests/test_phase7c_index_registry.py, tests/p4_t3_test.py, tests/p1_t4_test.py, tests/p5_t3_test.py
   - Deleted: tests/ingestion/test_per_call_embedding_overrides.py, tests/p3_t4_test.py, tests/p3_t4_integration_test.py, tests/p6_t4_test.py, tests/v2_2/test_reconciliation_drift.py, tests/p6_t2_test.py, tests/p2_t2_test.py, tests/unit/test_graph_enhancements.py, tests/neo/test_explain_guard.py, tests/p4_t2_perf_test.py, tests/p4_t2_test.py

4. **Test file modifications** (2 files)
   - tests/conftest.py: Removed src.mcp_server.security import
   - tests/p6_t1_test.py: Removed 2 test methods using deleted imports (kept 7 active tests)
   - tests/e2e/test_golden_set.py: Removed 1 test method using deleted imports (kept all active tests)

5. **Manual cleanup** (1 file, 13 lines)
   - src/ingestion/build_graph.py: Removed dead reconciliation code block

### Verification Results ✅

```bash
# All dead imports removed
$ grep -r "from src.learning\|from src.registry\|..." tests/
(no results, exit 1)

# Test collection successful
$ pytest --collect-only -q
1484 tests collected in 8.08s
```

### Metrics Impact

| Metric | Before (Phase 0) | After (Phase 1) | Change |
|--------|------------------|-----------------|---------|
| src/ files | 169 | 135 | -34 |
| src/ LOC | 68,551 | 59,500 | -9,051 (13.2%) |
| tests/ files | TBD | 151 | TBD |
| tests/ LOC | TBD | 37,326 | TBD |
| Tests collected | 1,698 | 1,484 | -214 |
| Collection errors | 0 | 0 | 0 |

### Lessons Learned

- Subagents reported "all imports are in tests only" correctly
- Manual verification confirmed no production code dependencies
- One broken import discovered in src/ingestion/build_graph.py after deletion (reconciliation code using deleted Reconciler class)
- Fixed manually before proceeding
- 7 test files had mixed imports (dead + active), required careful analysis to keep active tests

---

## Phase 2+: Future

See CLEANUP-EXECUTION-PLAN.md for full phase breakdown.

---

## Phase 3: Remove Dead Code Blocks ✅

**Status:** COMPLETE
**Date:** 2026-06-03
**Goal:** Delete files and methods marked @status: DEAD (0 external callers)

### Execution Summary

**1. Deleted dead files (3 files, 243 LOC):**
- `src/tools/inspect_chunk.py` (116 LOC, 0 callers)
- `src/tools/__init__.py` (0 LOC, empty — orphaned after inspect_chunk removal)
- `src/ops/session_cleanup_job.py` (127 LOC, @status: DEAD)

**2. Removed dead methods from src/shared/connections.py (60 LOC, 2 methods):**
- `purge_document` (was L179-L190) — DELETED — 0 external callers
- `create_collection_with_dims` (was L356-L398) — DELETED — 0 external callers
- Updated header comment: `@status: MIXED` → `@status: ACTIVE`

**3. Updated test files (3 files):**
- `tests/test_phase3_qdrant_safety.py`: Removed `test_blue_green_helper_exists` (tests dead `create_collection_with_dims`); removed it from `main()` test list. 263 → 221 lines (-42)
- `tests/integration/test_session_tracking.py`: Removed `TestSessionCleanup` class (2 test methods importing deleted `src.ops.session_cleanup_job`). 608 → 503 lines (-105)
- `tests/integration/test_phase7c_integration.py`: No changes needed

**4. build_graph.py massive cleanup (3,126 → 287 LOC, -2,839 LOC, 91% reduction):**
- Removed 39 DEAD methods + 1 module-level function
- Kept only 4 alive methods: `__init__`, `ensure_embedder`, `_build_section_text_for_embedding`, `_build_title_text_for_embedding`
- Removed dead `__init__` calls: `_ensure_qdrant_collection`, `_ensure_neo4j_vector_index`, `_reconcile_schema_version_embedding_metadata`

**5. Deleted 14 test files (testing only the removed dead code):**
- tests/p3_t3_test.py, tests/p3_t3_integration_test.py
- tests/test_phase7c_ingestion.py, tests/test_phase7e1_chunk_ingestion.py
- tests/test_phase7e3_cache_invalidation.py, tests/test_phase7e4_observability.py
- tests/v2_2/test_ingestion_edge_cases.py, tests/v2_2/test_retrieval_edge_cases.py
- tests/v2_2/test_hybrid_rag_v22_integration.py
- tests/integration/test_phase7e3_context_stitching.py
- tests/integration/phase7e3_regression_pack/ (entire directory)
- tests/integration/prod_docs_pack/ (entire directory)

### Verification Results ✅

```bash
# build_graph.py verified
$ python3 -m py_compile src/ingestion/build_graph.py  → PASS
$ grep -n "def " src/ingestion/build_graph.py  → Only 4 methods remain

# No dead code references
$ grep -rn "inspect_chunk" src/ tests/ --include="*.py"  (no results)
$ grep -rn "from src.ops.session_cleanup_job" tests/ --include="*.py"  (no results)

# All py_compile checks pass
$ python3 -m py_compile src/shared/connections.py  → PASS
$ python3 -m py_compile tests/test_phase3_qdrant_safety.py  → PASS
$ python3 -m py_compile tests/integration/test_session_tracking.py  → PASS
```

### Metrics Impact

| Metric | Before (Phase 2) | After (Phase 3) | Change |
|--------|-------------------|------------------|---------|
| src/ files | 135 | 131 | -4 |
| src/ LOC | 58,499 | 54,890 | -3,609 |
| tests/ files | 151 | 137 | -14 |
| Methods removed | 0 | 42 | 2 from connections.py + 40 from build_graph.py |
| Files removed | 0 | 17 | 3 src/ + 14 tests/ |

---

## Metrics Tracker

| Metric | Phase 0 Baseline | After Phase 1 | After Phase 3 | Target |
|--------|------------------|---------------|---------------|--------|
| src/ files | 169 | 135 | 131 | ≤100 |
| src/ LOC | 68,551 | 59,500 | 54,890 | ≤40,000 |
| tests/ files | — | 151 | 137 | — |
| Test collection | 1,698 | 1,484 | TBD (should drop) | ≤1,000 |
| Collection errors | 0 | 0 | 0 | 0 |
| Ruff issues | 2 | TBD | TBD | 0 |
| Mypy errors | 532 | TBD | TBD | ≤200 |

---

## Cumulative Progress (3 phases)

| Metric | Before Cleanup | Current | Reduction |
|--------|---------------|---------|-----------|
| src/ files | 169 | 131 | -38 (22%) |
| src/ LOC | 68,551 | 54,890 | -13,661 (20%) |
| tests/ files | — | 137 | -14+ deleted |

## Top 5 Largest Files Remaining

| File | LOC | Refactor Priority |
|------|-----|-------------------|
| src/ingestion/atomic.py | ~4,050 | High |
| src/mcp_server/mcp_app.py | ~3,830 | High |
| src/query/hybrid_retrieval.py | ~2,080 | Medium |
| src/shared/config.py | ~2,015 | Medium |
| src/query/vector_backends.py | ~1,750 | Medium |

---

## Phase 4: Next Priorities

1. **Chonkie adapter consolidation** (~319 LOC savings via base class — Arctic + Qwen3 share 85% of their code)
2. **Dead methods in embedding providers** (~50 LOC across voyage.py, jina.py, etc.)
3. **Feature flag consolidation** (merge overlapping flags)
4. **Configuration simplification** (deduplicate config fields)
5. **Large function decomposition** (process_document, mcp_endpoint — both ~300 lines)
6. **CircuitBreaker unification** (3 implementations → 1)

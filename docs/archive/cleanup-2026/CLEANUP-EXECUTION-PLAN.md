# CLEANUP EXECUTION PLAN

## Overview

This document defines the **execution strategy** for the CLEANUP-PLAN.md phases. It answers:
- How do we execute the 13 phases?
- How do we coordinate subagents?
- How do we verify work without trusting summary claims?
- How do we iterate when things break?
- What are the success metrics beyond LOC reduction?

**Relationship to CLEANUP-PLAN.md:**
- CLEANUP-PLAN.md defines **WHAT** to do (13 phases with exit criteria)
- CLEANUP-EXECUTION-PLAN.md defines **HOW** to do it (process, coordination, verification)

---

## Guiding Principles

### 1. Subagents Do Labor, Parent Verifies Claims

**Pattern:**
```
Parent → Subagent: "Delete these 5 files, verify with grep, report LOC removed"
Subagent → Parent: "Deleted files A, B, C, D, E. LOC: 1,234. Verified no imports."
Parent → grep: "grep -rn 'from A\|from B\|from C\|from D\|from E' src/ tests/"
Parent decision: "Claims check out. Commit."
```

**Never trust:**
- "All tests pass" without junit.xml evidence
- "No imports found" without grep output
- "Phase X complete" without exit criteria verification

**Always verify:**
- File deletions (ls -la <path>)
- Import cleanup (grep -rn)
- Test collection (pytest --collect-only output)
- Smoke tests (actual ingestion + query)

### 2. Parallel Execution for Independent Work

**Parallel-safe:**
- Deleting unrelated modules (learning/ vs registry/ vs ops/)
- Running ruff + mypy + pytest --collect-only simultaneously
- Cleaning up tests for different deleted modules

**Sequential (dependencies):**
- Phase 0 must complete before Phase 1
- Phase 1 (delete files) before Phase 2 (delete methods)
- Phase 3 (delete hybrid_search.py) before Phase 8 (unify retrieval types)
- Phase 4 (split monoliths) before Phase 5 (clean up providers)

### 3. Iteration Cadence

**Per-phase iteration:**
1. Parent defines task with exit criteria
2. Subagent(s) execute in parallel
3. Parent spot-checks all claims
4. If claims fail → re-run subagent with corrections
5. If claims pass → commit, move to next phase
6. If phase breaks smoke tests → revert, analyze, adjust plan

**Cross-phase iteration:**
- After Phase 1-3 (deletion phases): Re-run test collection, adjust test counts
- After Phase 4 (monolith split): Run full test suite, measure pass/fail
- After Phase 5 (provider cleanup): Verify all 6 providers still work
- After Phase 9 (test hygiene): Measure coverage, adjust threshold

### 4. Risk Management

**Breaking working features is unacceptable.**

**Mitigation:**
- Phase 0 establishes baseline (what works now)
- Smoke tests after every phase (ingest + query)
- Git commits per phase (easy revert if broken)
- Never delete something with runtime callers (verify with grep first)

**High-risk files (require extra verification):**
- `src/neo/contract_checks.py` - marked DEAD but imported by worker.py
- `src/ingestion/build_graph.py` - marked DEPRECATED but lazily imported by atomic.py
- `src/shared/connections.py` - contains DEAD methods but module is ACTIVE
- `src/mcp_server/security/*` - marked DEAD but represents security gap

**Low-risk files (safe to delete):**
- `src/learning/*` - zero imports, test-only
- `src/registry/*` - zero imports
- `src/ops/optimizer.py` - test-only
- Dead test files (p1_t*, p2_t*, etc. that test dead modules)

### 5. Success Metrics

**Quantitative (from CLEANUP-PLAN.md):**
- LOC: 68,551 → ≤40,000 (target: 41% reduction)
- Dead code: 9,038 + 4,600 → 0 (100% reduction)
- Monolithic files (>1000 LOC): 10 → 0
- Env vars: 85+ → ≤20
- Feature flags: 18 → ≤6

**Qualitative:**
- **Comprehensibility:** New contributor can understand the architecture in <30 minutes
- **Navigability:** Every module has a clear, single responsibility
- **Maintainability:** Changes to one concern don't require touching 5 files
- **Testability:** Tests are organized by module, not by historical phase
- **Deployability:** One config file, clear env vars, documented profiles

**Operational:**
- Smoke tests pass (ingest + query)
- No runtime errors in logs
- MCP server responds to all 16 canonical tools
- All 6 embedding providers can be loaded

---

## Phase 0: Empirical Baseline

### Goal
Establish what **actually works** today. No claims, only measurements.

### Why This Phase is Critical
- We can't delete code without knowing what's tested
- We can't split monoliths without knowing what breaks
- We can't iterate without a baseline to compare against

### Execution Strategy

**Subagent 1: Python Environment Setup**
```
Task:
  - Verify Python 3.11 available at /opt/homebrew/bin/python3.11
  - Create .venv with Python 3.11
  - Install requirements.txt
  - Install dev tools (pytest, ruff, mypy)

Exit Criteria:
  - .venv exists and is Python 3.11
  - `source .venv/bin/activate && python --version` returns 3.11.x
  - `pip list | grep -E "pytest|ruff|mypy"` shows all installed
  - Write results to reports/baseline/environment.md

Verification:
  - Parent runs: cat .venv/pyvenv.cfg | grep "version = 3.11"
  - Parent runs: .venv/bin/pip list | grep -E "pytest|ruff|mypy"
```

**Subagent 2: Test Collection**
```
Task:
  - Activate .venv
  - Run: pytest --collect-only -q > reports/baseline/tests_collected.txt
  - Count total tests collected
  - Categorize by directory (unit/, integration/, e2e/, p*_t*, etc.)
  - Identify tests for dead modules (learning/, registry/, ops/)

Exit Criteria:
  - reports/baseline/tests_collected.txt exists
  - reports/baseline/test_summary.md with:
    - Total tests: X
    - By directory: unit (X), integration (X), e2e (X), p1_t* (X), etc.
    - Tests for dead modules: X (list them)

Verification:
  - Parent runs: wc -l reports/baseline/tests_collected.txt
  - Parent runs: grep -c "tests/learning" reports/baseline/tests_collected.txt
  - Parent runs: grep -c "tests/registry" reports/baseline/tests_collected.txt
```

**Subagent 3: Static Analysis**
```
Task:
  - Activate .venv
  - Run: ruff check src/ > reports/baseline/ruff.txt
  - Count total issues, categorize by rule (F841, E501, etc.)
  - Run: mypy src/ --ignore-missing-imports > reports/baseline/mypy.txt
  - Count total errors

Exit Criteria:
  - reports/baseline/ruff.txt exists
  - reports/baseline/ruff_summary.md with:
    - Total issues: X
    - By rule: F841 (X), E501 (X), etc.
    - Top 10 files with most issues
  - reports/baseline/mypy.txt exists
  - reports/baseline/mypy_summary.md with:
    - Total errors: X
    - Top 10 files with most errors

Verification:
  - Parent runs: wc -l reports/baseline/ruff.txt
  - Parent runs: wc -l reports/baseline/mypy.txt
  - Parent runs: tail -1 reports/baseline/ruff.txt (should show "Found X errors")
```

### Phase 0 Deliverables
```
reports/baseline/
├── environment.md          # Python version, venv location
├── test_summary.md         # Test count by directory
├── tests_collected.txt     # Full pytest output
├── ruff_summary.md         # Ruff issues by rule
├── ruff.txt                # Full ruff output
├── mypy_summary.md         # Mypy errors by file
└── mypy.txt                # Full mypy output
```

### Phase 0 Exit Criteria
- [ ] Python 3.11 environment verified
- [ ] Test collection successful (X tests collected)
- [ ] Ruff baseline captured (X issues)
- [ ] Mypy baseline captured (X errors)
- [ ] All artifacts in reports/baseline/
- [ ] README.md updated with baseline numbers

### Phase 0 Risks
**Risk 1: Tests fail to collect**
- Mitigation: Identify which tests fail, categorize as "dated" vs "broken"
- Action: If >50% fail, skip test execution, just count collection errors

**Risk 2: Python 3.11 not available**
- Mitigation: Try alternatives (pyenv, conda)
- Action: If no 3.11, use 3.10 and document version

**Risk 3: Ruff/mypy not installed**
- Mitigation: Install in venv, document in environment.md
- Action: If install failed, use system versions and document differences

## Phase 0 Results (Actual Baseline)

### Environment ✅
- **Python:** 3.11.14 (from /opt/homebrew/bin/python3.11)
- **Venv:** .venv/ created successfully
- **pytest:** 7.4.4
- **ruff:** 0.15.15
- **mypy:** 2.1.0

### Test Collection ✅
- **Total tests:** 1698
- **Collection errors:** 0
- **Test files:** 30 files matching p*_t* pattern (dated, pre-Dec 2025)
- **Note:** Created tools/__init__.py to fix collection error about tools.redis_epoch_bump

### Static Analysis ✅
- **Ruff issues:** 2 (both F841 unused-variable in src/query/hybrid_retrieval.py:173,178)
- **Mypy errors:** 532 across 87 files (of 169 checked)
  - Top file: src/ingestion/build_graph.py (67 errors)
  - Top error types: arg-type (154), assignment (70), attr-defined (66)

### Artifacts ✅
- reports/baseline/ruff.txt (26 lines)
- reports/baseline/mypy.txt (663 lines)
- reports/baseline/test_collection.txt (created during verification)

---

## Phase 1: Full-File Dead Code Removal

### Goal
Delete 9,038 LOC of files with zero live imports.

### Execution Strategy

**Step 1: Parent verifies kill list**
```bash
# Parent runs this before delegating
grep -rn "from src.learning\|from src.registry\|from src.ops.optimizer" src/ tests/
# Expected: zero hits

grep -rn "from src.neo.explain_guard\|from src.neo.entity_normalization" src/ tests/
# Expected: zero hits

# Parent confirms every file on kill list has no active imports
```

**Step 2: Parallel deletion subagents**

**Subagent A: Delete learning/ + registry/ + ops/**
```
Task:
  - Delete: src/learning/ (4 files, 983 LOC)
  - Delete: src/registry/ (2 files, 282 LOC)
  - Delete: src/ops/optimizer.py (499 LOC)
  - Delete: src/ops/warmers/ (2 files, 162 LOC)
  - Keep: src/ops/session_cleanup_job.py (standalone CLI)
  - Verify: grep -rn "from src.learning\|from src.registry\|from src.ops.optimizer\|from src.ops.warmers" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 8
  - LOC removed: 1,926
  - Verification grep: zero hits

Reporting:
  - "Deleted: learning/ (983 LOC), registry/ (282 LOC), ops/optimizer.py (499 LOC), ops/warmers/ (162 LOC)"
  - "Total: 1,926 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent B: Delete mcp_server dead code**
```
Task:
  - Delete: src/mcp_server/validation.py (407 LOC)
  - Delete: src/mcp_server/security/ (3 files, 355 LOC)
  - Verify: grep -rn "from src.mcp_server.validation\|from src.mcp_server.security" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 4
  - LOC removed: 762
  - Verification grep: zero hits

Reporting:
  - "Deleted: mcp_server/validation.py (407 LOC), mcp_server/security/ (355 LOC)"
  - "Total: 762 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent C: Delete shared dead code**
```
Task:
  - Delete: src/shared/feature_flags.py (173 LOC)
  - Delete: src/shared/audit/ (2 files, 218 LOC)
  - Verify: grep -rn "from src.shared.feature_flags\|from src.shared.audit" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 3
  - LOC removed: 391
  - Verification grep: zero hits

Reporting:
  - "Deleted: shared/feature_flags.py (173 LOC), shared/audit/ (218 LOC)"
  - "Total: 391 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent D: Delete neo dead code**
```
Task:
  - Delete: src/neo/explain_guard.py (276 LOC)
  - Delete: src/neo/entity_normalization.py (274 LOC)
  - Delete: src/neo/graph_enhancements.py (440 LOC)
  - Delete: src/neo/structural_builder.py (520 LOC)
  - Delete: src/neo/defensive_query.py (112 LOC)
  - Delete: src/neo/health.py (117 LOC)
  - Keep: src/neo/contract_checks.py (DEAD but imported by worker.py - DO NOT TOUCH)
  - Verify: grep -rn "from src.neo.explain_guard\|from src.neo.entity_normalization\|from src.neo.graph_enhancements\|from src.neo.structural_builder\|from src.neo.defensive_query\|from src.neo.health" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 5
  - LOC removed: 1,379
  - Verification grep: zero hits

Reporting:
  - "Deleted: neo/explain_guard.py (276 LOC), neo/entity_normalization.py (274 LOC), neo/graph_enhancements.py (440 LOC), neo/structural_builder.py (520 LOC), neo/defensive_query.py (112 LOC), neo/health.py (117 LOC)"
  - "Total: 1,739 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent E: Delete query dead code**
```
Task:
  - Delete: src/query/diffusion_reranker.py (362 LOC)
  - Delete: src/query/graph_features.py (339 LOC)
  - Delete: src/query/graph_expansion.py (302 LOC)
  - Keep: src/query/planner.py (ACTIVE - query_service.py:827 calls it)
  - Keep: src/query/hybrid_search.py (ACTIVE - runtime-reachable legacy path)
  - Verify: grep -rn "from src.query.diffusion_reranker\|from src.query.graph_features\|from src.query.graph_expansion" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 3
  - LOC removed: 1,003
  - Verification grep: zero hits

Reporting:
  - "Deleted: query/diffusion_reranker.py (362 LOC), query/graph_features.py (339 LOC), query/graph_expansion.py (302 LOC)"
  - "Total: 1,003 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent F: Delete ingestion dead code**
```
Task:
  - Delete: src/ingestion/api.py (35 LOC)
  - Delete: src/ingestion/reconcile.py (525 LOC)
  - Delete: src/ingestion/incremental.py (362 LOC)
  - Delete: src/ingestion/parsers/notion.py (256 LOC)
  - Verify: grep -rn "from src.ingestion.api\|from src.ingestion.reconcile\|from src.ingestion.incremental\|from src.ingestion.parsers.notion" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 4
  - LOC removed: 1,178
  - Verification grep: zero hits

Reporting:
  - "Deleted: ingestion/api.py (35 LOC), ingestion/reconcile.py (525 LOC), ingestion/incremental.py (362 LOC), ingestion/parsers/notion.py (256 LOC)"
  - "Total: 1,178 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent G: Delete ingestion/auto dead code**
```
Task:
  - Delete: src/ingestion/auto/watcher.py (54 LOC) - singular, superseded by watchers.py
  - Delete: src/ingestion/auto/orchestrator.py (1,134 LOC)
  - Delete: src/ingestion/auto/verification.py (277 LOC)
  - Delete: src/ingestion/auto/report.py (296 LOC)
  - Delete: src/ingestion/auto/backpressure.py (282 LOC)
  - Keep: src/ingestion/auto/watchers.py (ACTIVE - plural)
  - Verify: grep -rn "from src.ingestion.auto.watcher\b\|from src.ingestion.auto.orchestrator\|from src.ingestion.auto.verification\|from src.ingestion.auto.report\|from src.ingestion.auto.backpressure" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - Files deleted: 5
  - LOC removed: 2,043
  - Verification grep: zero hits

Reporting:
  - "Deleted: ingestion/auto/watcher.py (54 LOC), ingestion/auto/orchestrator.py (1,134 LOC), ingestion/auto/verification.py (277 LOC), ingestion/auto/report.py (296 LOC), ingestion/auto/backpressure.py (282 LOC)"
  - "Total: 2,043 LOC"
  - "Verification: grep returned X hits (expected 0)"
```

**Step 3: Parent verification**
```bash
# Parent runs after all subagents complete
grep -rn "from src.learning\|from src.registry\|from src.ops.optimizer\|from src.ops.warmers\|from src.mcp_server.validation\|from src.mcp_server.security\|from src.shared.feature_flags\|from src.shared.audit\|from src.neo.explain_guard\|from src.neo.entity_normalization\|from src.neo.graph_enhancements\|from src.neo.structural_builder\|from src.neo.defensive_query\|from src.neo.health\|from src.query.diffusion_reranker\|from src.query.graph_features\|from src.query.graph_expansion\|from src.ingestion.api\|from src.ingestion.reconcile\|from src.ingestion.incremental\|from src.ingestion.parsers.notion\|from src.ingestion.auto.watcher\b\|from src.ingestion.auto.orchestrator\|from src.ingestion.auto.verification\|from src.ingestion.auto.report\|from src.ingestion.auto.backpressure" src/ tests/
# Expected: zero hits

# Parent counts LOC removed
git diff --stat HEAD
# Expected: ~26 files, ~9,038 LOC removed
```

**Step 4: Delete orphaned tests**
```bash
# Parent identifies tests for deleted modules
grep -rln "from src.learning\|from src.registry\|from src.ops.optimizer" tests/
# If hits: delete those test files

grep -rln "from src.neo.explain_guard\|from src.neo.entity_normalization" tests/
# If hits: delete those test files

# etc. for all deleted modules
```

**Step 5: Test collection verification**
```bash
# Parent runs after deletions
pytest --collect-only -q > reports/phase1/tests_collected.txt
# Expected: fewer tests than baseline (orphaned tests deleted)
# Expected: zero collection errors (no broken imports)
```

### Phase 1 Deliverables
```
reports/phase1/
├── deletion_summary.md     # Files deleted, LOC removed per subagent
├── verification_grep.txt   # Grep output (should be zero hits)
├── tests_collected.txt     # Pytest collection after deletions
└── git_diff_stat.txt       # Git diff --stat HEAD
```

### Phase 1 Exit Criteria
- [ ] 9,038 LOC removed (verified by git diff --stat)
- [ ] All deleted files have zero imports (verified by grep)
- [ ] Orphaned tests deleted
- [ ] pytest --collect-only succeeds (zero collection errors)
- [ ] Smoke test still passes (ingest + query)
- [ ] All artifacts in reports/phase1/

### Phase 1 Risks
**Risk 1: Subagent deletes wrong file**
- Mitigation: Parent provides explicit file list
- Action: If wrong file deleted, git revert, re-run subagent

**Risk 2: Hidden import not caught by grep**
- Mitigation: Run pytest --collect-only after deletions
- Action: If collection error, identify missing import, restore file

**Risk 3: Smoke test breaks**
- Mitigation: Run smoke test after each subagent
- Action: If breaks, git revert that subagent's work, analyze

---

## Phase 2: Dead Method Removal

### Goal
Delete ~4,600 LOC of dead methods in mixed-status files.

### Execution Strategy

**Subagent 1: build_graph.py cleanup**
```
Task:
  - Read src/ingestion/build_graph.py (3,200 LOC)
  - Identify dead methods (grep for each method name in src/ tests/)
  - Delete methods with zero callers:
    - upsert_document
    - _upsert_document_node
    - _upsert_sections
    - _upsert_entities
    - _create_mentions
    - _process_embeddings
    - _ensure_qdrant_collection
    - _build_entity_text_for_embedding
    - _compute_text_hash
    - _compute_shingle_hash
    - _extract_semantic_metadata
    - ingest_document (standalone function at bottom)
  - KEEP methods called by atomic.py:
    - __init__
    - ensure_embedder
    - embedder (property)
    - _build_section_text_for_embedding
    - _build_title_text_for_embedding
  - Verify: grep -rn "build_graph.upsert_document\|build_graph._upsert" src/ tests/
  - Expected: zero hits for deleted methods

Exit Criteria:
  - File reduced from 3,200 LOC to ~350 LOC
  - All kept methods still imported by atomic.py
  - Verification grep: zero hits for deleted methods

Reporting:
  - "build_graph.py: 3,200 → X LOC"
  - "Methods deleted: 12"
  - "Methods kept: 5"
  - "Verification: grep returned X hits for deleted methods (expected 0)"
```

**Subagent 2: saga.py cleanup**
```
Task:
  - Read src/ingestion/saga.py (688 LOC)
  - Identify dead classes/methods:
    - SagaContext (KEEP - used by atomic.py)
    - ValidationResult (KEEP - used by atomic.py)
    - IngestionValidator (KEEP - used by atomic.py)
    - SagaStatus (DEAD)
    - StepStatus (DEAD)
    - SagaStepResult (DEAD)
    - SagaStep (DEAD)
    - SagaCoordinator (DEAD)
    - IngestionSagaBuilder (DEAD)
  - Delete all DEAD classes/methods
  - Verify: grep -rn "saga.SagaCoordinator\|saga.SagaStep\|saga.SagaStatus" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - File reduced from 688 LOC to ~280 LOC
  - All kept classes still imported
  - Verification grep: zero hits for deleted classes

Reporting:
  - "saga.py: 688 → X LOC"
  - "Classes/methods deleted: 6"
  - "Classes/methods kept: 3"
  - "Verification: grep returned X hits (expected 0)"
```

**Subagent 3: contract_checks.py cleanup**
```
Task:
  - Read src/neo/contract_checks.py (455 LOC)
  - KEEP:
    - GraphContractChecker.__init__
    - find_documents_needing_repair
  - DELETE all other methods:
    - run_all_checks
    - check_chunk_document_membership
    - check_next_chunk_no_branching
    - check_next_chunk_no_cycles
    - check_hierarchy_coverage
    - check_entity_normalization
    - check_chunk_id_uniqueness
    - run_contract_checks
  - Verify: grep -rn "contract_checks.run_all_checks\|contract_checks.check_chunk" src/ tests/
  - Expected: zero hits

Exit Criteria:
  - File reduced from 455 LOC to ~80 LOC
  - Kept methods still called by worker.py
  - Verification grep: zero hits

Reporting:
  - "contract_checks.py: 455 → X LOC"
  - "Methods deleted: 8"
  - "Methods kept: 2"
  - "Verification: grep returned X hits (expected 0)"
```

### Phase 2 Deliverables
```
reports/phase2/
├── build_graph_cleanup.md
├── saga_cleanup.md
├── contract_checks_cleanup.md
├── verification_grep.txt
└── tests_collected.txt
```

### Phase 2 Exit Criteria
- [ ] ~4,600 LOC removed (verified by git diff --stat)
- [ ] All deleted methods have zero callers (verified by grep)
- [ ] pytest --collect-only succeeds
- [ ] Smoke test still passes
- [ ] All artifacts in reports/phase2/

---

## Phase 3-13: Summary

For brevity, phases 3-13 follow the same pattern:
1. Parent defines task with explicit file/method lists
2. Subagents execute in parallel where safe
3. Parent verifies all claims with grep + pytest
4. Parent commits when exit criteria met
5. Parent runs smoke test
6. If breaks, revert and analyze

**Key differences by phase:**

**Phase 3 (Delete hybrid_search.py):** Sequential, high-risk, requires query_service.py refactoring first.

**Phase 4 (Split monoliths):** Sequential, very high-risk, requires extensive test verification after each split.

**Phase 5 (Provider cleanup):** Low-risk, can be parallel (Chonkie adapters, HTTP clients, CircuitBreakers).

**Phase 6 (Feature flag reduction):** Low-risk, can be parallel (YAML cleanup, env var audit).

**Phase 7 (Parser unification):** Medium-risk, requires smoke test verification.

**Phase 8 (Retrieval unification):** High-risk, requires eval harness verification.

**Phase 9 (Test reorganization):** Low-risk, parallel (move tests by directory, delete dead tests).

**Phase 10 (MCP cleanup):** Medium-risk, requires HTTP + STDIO verification.

**Phase 11 (Observability hardening):** Low-risk, parallel (circuit breaker unification, metrics cleanup).

**Phase 12 (CLI polish):** Low-risk, parallel (new CLI commands, delete root-level scripts).

**Phase 13 (Final gate):** Sequential, requires full measurement and comparison to baseline.

---

## Iteration Strategy

### Per-Phase Iteration

**Iteration 1: Execute**
- Subagents complete work
- Parent spot-checks claims
- If claims fail → re-run subagent with corrections
- If claims pass → commit

**Iteration 2: Verify**
- Run pytest --collect-only
- Run smoke test
- If breaks → git revert, analyze, adjust plan
- If passes → move to next phase

**Iteration 3: Adjust**
- Update CLEANUP-PLAN.md with empirical results
- Adjust LOC targets if needed
- Adjust exit criteria if needed

### Cross-Phase Iteration

**After Phase 1-3 (All deletions):**
- Re-run test collection
- Measure: new test count = baseline - orphaned tests
- Adjust Phase 9 (test hygiene) targets

**After Phase 4 (Monolith splits):**
- Run full test suite (pytest)
- Measure: new pass/fail count
- Adjust Phase 9 (test hygiene) targets

**After Phase 5 (Provider cleanup):**
- Verify all 6 providers still work
- Measure: provider load time
- Adjust Phase 12 (CLI polish) if needed

**After Phase 9 (Test hygiene):**
- Run pytest --cov=src
- Measure: coverage %
- Adjust target coverage if needed

**After Phase 13 (Final gate):**
- Re-run Phase 0 measurements
- Compare to baseline
- If targets not met → iterate on specific phases

---

## Success Criteria

### Must-Have (from CLEANUP-PLAN.md)

**Quantitative:**
- [ ] LOC: 68,551 → ≤40,000 (41% reduction)
- [ ] Dead code: 9,038 + 4,600 → 0 (100% reduction)
- [ ] Monolithic files (>1000 LOC): 10 → 0
- [ ] Env vars: 85+ → ≤20
- [ ] Feature flags: 18 → ≤6

**Qualitative:**
- [ ] Comprehensibility: New contributor understanding in <30 min
- [ ] Navigability: Single responsibility per module
- [ ] Maintainability: Changes don't require touching 5 files
- [ ] Testability: Tests organized by module
- [ ] Deployability: One config file, clear env vars

**Operational:**
- [ ] Smoke tests pass (ingest + query)
- [ ] No runtime errors in logs
- [ ] MCP server responds to all 16 tools
- [ ] All 6 providers load successfully

### Nice-to-Have

**Quality Improvements:**
- [ ] Type coverage: mypy errors reduced by 50%
- [ ] Lint compliance: ruff issues reduced by 50%
- [ ] Documentation: Each module has README.md
- [ ] Examples: Each MCP tool has example usage

**Process Improvements:**
- [ ] CI/CD: Automated cleanup verification in CI
- [ ] Monitoring: Alerts for LOC growth
- [ ] Governance: PR checklist for new code

---

## Conclusion

This execution plan transforms CLEANUP-PLAN.md from a static document into an actionable, verifiable process. By:
- Using subagents for labor but parent for verification
- Establishing clear spot-check methodology
- Defining iteration cadence and risk management
- Setting success metrics beyond LOC reduction

...we can execute the 13 phases with confidence that we're improving the codebase without breaking working features.

**Next step:** Execute Phase 0 (empirical baseline) with 3 parallel subagents.

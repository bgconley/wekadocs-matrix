# Test Suite Analysis: Dated vs Current Tests

**Analysis Date:** January 28, 2026
**Repository:** wekadocs-matrix
**Total Test Files:** 159 (excluding `__init__.py` and `conftest.py`)
**Methodology:** Git log last-modified dates, source import verification, endpoint existence checks, label schema analysis

---

## 1. Assessment of Test Suite Reliability

### Verdict: Pass/Fail CANNOT be trusted as a quality signal

The test suite has **severe structural staleness**. The majority of tests were written for an architecture that has been **explicitly refactored and deprecated**. Test failures primarily reflect **architectural drift**, not code quality regression.

### Key Findings

#### A. Legacy REST Endpoints Are Disabled (affects p1_t2_test.py directly)

The three MCP REST endpoints tested by `p1_t2_test.py` are gated behind `MCP_HTTP_LEGACY_REST_ENABLED`, which defaults to `"false"`:

```python
# src/mcp_server/main.py line 74-75
MCP_HTTP_LEGACY_REST_ENABLED = os.getenv(
    "MCP_HTTP_LEGACY_REST_ENABLED", "false"  # Default: OFF (was "true")
)
```

All three endpoints return HTTP 404 when the flag is off:
- `POST /mcp/initialize` → line 519: `if not MCP_HTTP_LEGACY_REST_ENABLED: raise HTTPException(status_code=404, ...)`
- `GET /mcp/tools/list` → line 541: same gate
- `POST /mcp/tools/call` → line 590: same gate

The system has migrated to `mcp_app.py` (138KB, Streamable HTTP MCP protocol). The old REST endpoints were intentionally deprecated. **Tests for them must fail** — this is by design, not a bug.

#### B. `:Section` Label Fully Replaced by `:Chunk` (affects ~30 test files)

Verified via source code grep:
- **Source code**: ZERO references to `(:Section {` in any `.py` source file
- **Source code**: 10+ references to `(:Chunk {` in `build_graph.py`, `orchestrator.py`
- **Refactor commit**: `7555231` (Dec 12, 2025): "refactor: deprecate :Section label in favor of :Chunk across codebase"

Tests still referencing `:Section` labels:
| File | Last Modified | References |
|------|--------------|------------|
| `tests/p1_t3_test.py` | 2025-10-12 | `MERGE (s:Section {id: $id})`, `section_id_unique` constraint |
| `tests/p2_t1_test.py` | 2025-11-11 | `assert "Section" in template` |
| `tests/p3_t3_integration_test.py` | 2025-10-13 | Section-based queries |
| `tests/p3_t4_test.py` | 2025-10-13 | Section-based ingestion |
| `tests/p4_t2_perf_test.py` | 2025-10-14 | Section-based perf tests |
| `tests/p4_t4_test.py` | 2025-10-14 | Section-based caching |
| `tests/p5_t2_test.py` | 2025-12-12 | Section embeddings |
| `tests/p5_t3_test.py` | 2025-10-15 | Section references |
| `tests/p6_t1_test.py` | 2025-11-28 | Section-based |
| `tests/p6_t2_test.py` | 2025-11-28 | Section-based |
| `tests/p6_t3_test.py` | 2025-10-15 | Section-based |

Total: **~30 test files** still reference `Section` or `section_embeddings` (which was renamed to chunk-oriented storage).

#### C. Phase 1 Failing Tests Analysis (p1_t2_test.py, p1_t3_test.py)

**p1_t2_test.py** (10 tests, last modified: 2025-10-12):

| Test | Status | Reason |
|------|--------|--------|
| `test_mcp_initialize_endpoint` | WILL FAIL | Endpoint returns 404 (legacy disabled) |
| `test_mcp_tools_list_endpoint` | WILL FAIL | Endpoint returns 404 (legacy disabled) |
| `test_mcp_tools_call_endpoint` | WILL FAIL | Endpoint returns 404 (legacy disabled) |
| `test_correlation_id_header` | Likely passes | `/health` endpoint + middleware still active |
| `test_metrics_endpoint` | Uncertain | `/metrics` exists but response schema may have changed |
| `test_connection_pools_created` | Depends on Docker | Requires live Neo4j+Qdrant+Redis |
| `test_async_connection_pools` | Depends on Docker | Requires live Redis |
| `test_graceful_shutdown_handlers_exist` | Likely passes | `app` exists and has handlers |
| `test_structured_logging_configured` | Likely passes | `observability` package exists (as directory) |
| `test_opentelemetry_tracing_setup` | Likely passes | Standard OTel setup |

**p1_t3_test.py** (8 tests, last modified: 2025-10-12):

| Test | Status | Reason |
|------|--------|--------|
| `test_schema_creation` | WILL FAIL | References `src.shared.schema.create_schema` with Section schema |
| `test_constraints_exist` | WILL FAIL | Checks for `section_id_unique` — this constraint was renamed for Chunk |
| `test_indexes_exist` | WILL FAIL | Checks for `section_document_id` index — renamed for Chunk |
| `test_vector_indexes_exist` | WILL FAIL | Checks for `section_embeddings` vector index — renamed |
| `test_schema_version_node` | May pass | Generic schema version check |
| `test_schema_idempotence` | WILL FAIL | Depends on same schema.create_schema path |
| `test_can_create_document_node` | Likely passes | `:Document` label still valid |
| `test_can_create_section_node` | WILL FAIL | `MERGE (s:Section {id: $id})` — `:Section` label no longer in schema |

**Summary:** Of the 18 tests in these two failing files, **at least 10 will fail due to architectural changes**, not code bugs. The remaining tests either pass or depend on Docker services.

---

## 2. List of Clearly Dated Tests

These tests target dead code, deprecated endpoints, or pre-refactor architecture. All were last modified **before** the Dec 12, 2025 Section→Chunk refactor and have NOT been updated since.

### Era 1: Original Phase Tests (Oct 12–15, 2025) — 19 files
These were written for the initial architecture and **never updated after any pivot**.

| File | Last Modified | Key Issue |
|------|--------------|-----------|
| `tests/p1_t2_test.py` | 2025-10-12 | Tests legacy REST endpoints now gated behind `MCP_HTTP_LEGACY_REST_ENABLED=false` |
| `tests/p1_t3_test.py` | 2025-10-12 | Tests `:Section` schema, constraint names, vector indexes — all renamed to `:Chunk` |
| `tests/p1_t4_test.py` | 2025-10-12 | Observability/correlation tests, references old observability imports |
| `tests/p2_t2_test.py` | 2025-10-12 | Cypher validator tests, references `:Section` nodes |
| `tests/p3_t1_test.py` | 2025-10-13 | Parser tests with `:Section` references |
| `tests/p3_t2_test.py` | 2025-10-13 | Graph builder tests for `:Section` |
| `tests/p3_t3_test.py` | 2025-10-13 | Embedding tests referencing `section_embeddings` |
| `tests/p3_t3_integration_test.py` | 2025-10-13 | Integration tests with `:Section` queries |
| `tests/p3_t4_test.py` | 2025-10-13 | Reconciliation tests with `:Section` |
| `tests/p4_t1_test.py` | 2025-10-14 | Complex patterns with `:Section` |
| `tests/p4_t1_complex_patterns_test.py` | 2025-10-14 | Test patterns using `:Section` |
| `tests/p4_t2_optimizer_test.py` | 2025-10-14 | Query optimizer for `:Section` queries |
| `tests/p4_t2_perf_test.py` | 2025-10-14 | Performance tests on `:Section` |
| `tests/p4_t3_test.py` | 2025-10-14 | Cache tests for `:Section` |
| `tests/p4_t3_cache_perf_test.py` | 2025-10-14 | Cache perf for `:Section` |
| `tests/p4_t4_test.py` | 2025-10-14 | Security tests with `:Section` |
| `tests/p5_t3_test.py` | 2025-10-15 | Monitoring with Section references |
| `tests/p6_t3_test.py` | 2025-10-15 | Auto-ingest with `:Section` |
| `tests/p6_t4_test.py` | 2025-10-15 | Scheduling with Section references |

### Era 2: Mid-Evolution Tests (Oct 23 – Dec 11, 2025) — ~86 files
Many of these were written during active development but **before the major Section→Chunk migration**. They may partially work (unit tests that don't touch graph schema) but integration/schema-dependent ones are suspect.

Notable broken/suspect files in this era:
| File | Last Modified | Key Issue |
|------|--------------|-----------|
| `tests/test_phase7c_schema_v2_1.py` | 2025-10-25 | Tests old schema v2.1 with `section_id_unique` |
| `tests/test_phase7c_ingestion.py` | 2025-11-16 | References `:Section` in ingestion |
| `tests/test_phase7c_reranking.py` | 2025-11-16 | References `:Section` in reranking |
| `tests/test_phase7c_dual_write.py` | 2025-11-16 | Dual-write assumes Section+Chunk parity (transitional) |
| `tests/integration/test_phase7c_integration.py` | 2025-11-29 | Integration with Section labels |
| `tests/integration/test_phase7e3_context_stitching.py` | 2025-11-01 | Context stitching on Sections |
| `tests/test_integration_prephase7.py` | 2025-11-28 | Pre-Phase 7 integration with Sections |
| `tests/test_phase1_foundation.py` | 2025-10-23 | Foundation tests, pre-refactor |
| `tests/test_phase2_provider_wiring.py` | 2025-10-23 | Provider wiring may still be valid |
| `tests/test_phase3_qdrant_safety.py` | 2025-10-23 | Qdrant safety likely independent of labels |
| `tests/test_phase4_ranking_coverage.py` | 2025-11-21 | Ranking on Section model |
| `tests/test_phase7e_phase0.py` | 2025-10-29 | Section-based phase 0 |
| `tests/test_phase7e3_cache_invalidation.py` | 2025-10-29 | Cache invalidation for Section |
| `tests/test_phase7e4_observability.py` | 2025-10-29 | `section_embeddings` references |

---

## 3. List of Likely Current Tests

These tests target active code, were last modified **after** the Dec 12, 2025 refactor, and align with the current architecture.

### Era 3: Post-Refactor / Transitional (Dec 12, 2025 – Jan 19, 2026) — 12 files

| File | Last Modified | Focus |
|------|--------------|-------|
| `tests/p4_t2_test.py` | 2025-12-12 | Updated during Section→Chunk refactor |
| `tests/p5_t2_test.py` | 2025-12-12 | Updated during Section→Chunk refactor |
| `tests/p5_t4_test.py` | 2025-12-12 | Batch processing tests (updated) |
| `tests/test_graph_contract.py` | 2025-12-13 | Graph contracts for Chunk-based schema |
| `tests/shared/test_config_reload.py` | 2025-12-28 | Config reload (provider-agnostic) |
| `tests/shared/test_embedding_plan_fingerprint.py` | 2025-12-28 | Embedding plan fingerprints |
| `tests/shared/test_embedding_profiles.py` | 2025-12-28 | Embedding profile tests |
| `tests/shared/test_qdrant_schema_plan_dims.py` | 2025-12-28 | Qdrant schema dimensions |
| `tests/providers/test_profile_matrix.py` | 2025-12-28 | Provider matrix (Voyage, multi-embedder) |
| `tests/providers/test_voyage_provider.py` | 2025-12-28 | Voyage embedding provider |
| `tests/test_tokenizer_service.py` | 2025-12-28 | Tokenizer service |
| `tests/test_embedding_field_canonicalization.py` | 2025-12-28 | Field canonicalization |
| `tests/clients/test_snowflake_embedding_client.py` | 2026-01-19 | Snowflake Arctic embedding |
| `tests/providers/test_arctic_chonkie_adapter.py` | 2026-01-19 | Arctic + Chonkie adapter |
| `tests/providers/test_snowflake_arctic_provider.py` | 2026-01-19 | Snowflake Arctic provider |

### Era 4: Modern Architecture (Mar 3–8, 2026) — 30 files

| File | Last Modified | Focus |
|------|--------------|-------|
| `tests/providers/test_bge_m3_service_provider.py` | 2026-03-03 | BGE-M3 provider (import fixes applied) |
| `tests/query/test_qdrant_multivector_sparse.py` | 2026-03-03 | Multivector sparse retrieval |
| `tests/query/test_query_api_payload.py` | 2026-03-03 | Query API payload validation |
| `tests/test_phase7c_provider_factory.py` | 2026-03-03 | Provider factory (import fixes) |
| `tests/test_phase7e2_hybrid_retrieval.py` | 2026-03-03 | Hybrid retrieval (import fixes) |
| `tests/integration/test_phase1_entity_edges.py` | 2026-03-03 | Entity edge integration |
| `tests/contracts/test_mcp_streamable_contracts.py` | 2026-03-04 | MCP Streamable HTTP contracts |
| `tests/integration/test_gds_readiness.py` | 2026-03-04 | GDS readiness for RELATED_TO |
| `tests/integration/test_related_to_integration.py` | 2026-03-04 | RELATED_TO retrieval integration |
| `tests/mcp_server_tests/test_tool_profiles.py` | 2026-03-04 | MCP tool profiles |
| `tests/scripts/test_backfill_edge_parity.py` | 2026-03-04 | RELATED_TO edge backfill |
| `tests/unit/test_cross_doc_edge_model.py` | 2026-03-04 | RELATED_TO v2 edge model |
| `tests/unit/test_cross_doc_linking_v2.py` | 2026-03-04 | Cross-document linking v2 |
| `tests/unit/test_reciprocity.py` | 2026-03-04 | RELATED_TO reciprocity |
| `tests/unit/test_related_to_retrieval.py` | 2026-03-04 | RELATED_TO retrieval |
| `tests/unit/test_related_to_schema.py` | 2026-03-04 | RELATED_TO schema |
| `tests/unit/test_related_to_signal_pool.py` | 2026-03-04 | RELATED_TO signal pool |
| `tests/unit/test_structural_priors.py` | 2026-03-04 | Structural priors for GDS |
| `tests/mcp_server_tests/test_evidence_pack.py` | 2026-03-05 | Evidence pack extraction |
| `tests/unit/test_entity_quality_gating.py` | 2026-03-05 | Entity quality gating |
| `tests/integration/test_precision_retrieval_live.py` | 2026-03-07 | Precision retrieval (live) |
| `tests/query/test_query_intent.py` | 2026-03-07 | Query intent classification |
| `tests/query/test_retrieval_plan.py` | 2026-03-07 | Retrieval planning |
| `tests/query/test_signal_pool.py` | 2026-03-07 | Signal pool ordering |
| `tests/services/test_mxbai_reranker_service.py` | 2026-03-07 | mxbai reranker service |
| `tests/unit/test_colbert_observability.py` | 2026-03-07 | ColBERT observability |
| `tests/unit/test_phase1_reranker_batching.py` | 2026-03-07 | Reranker batching |
| `tests/unit/test_source_attribution.py` | 2026-03-07 | Source attribution |
| `tests/unit/test_structural_precision.py` | 2026-03-07 | Structural precision retrieval |
| `tests/query/test_reranker_integration.py` | 2026-03-08 | Reranker integration |
| `tests/test_structure_aware_expansion.py` | 2026-03-08 | Structure-aware expansion |

---

## 4. Estimate: Dated vs Current

### By File Count

| Category | Files | Percentage |
|----------|-------|-----------|
| **Clearly dated** (Oct 12 – Dec 11, 2025, not updated post-refactor) | ~105 | **66%** |
| **Likely current** (Dec 12, 2025+, updated for Chunk architecture) | ~47 | **30%** |
| **No git history** (eval helpers, misc) | ~7 | **4%** |
| **Total** | 159 | 100% |

### By Severity of Staleness

| Severity | Files | Description |
|----------|-------|-------------|
| **Guaranteed broken** (test disabled endpoints / removed labels) | ~25 | p1_t2 (3 MCP endpoints → 404), p1_t3 (Section schema), and ~20 tests with `:Section` Cypher |
| **Likely broken** (reference deprecated patterns but may pass accidentally) | ~50 | Tests referencing `section_embeddings`, old index names, or pre-refactor provider wiring |
| **Potentially valid** (unit tests agnostic to graph labels) | ~30 | Circuit breaker tests, pure unit tests, tokenizer tests from Era 2 |
| **Current** (actively maintained, aligned with architecture) | ~47 | Mar 2026 tests, Dec 2025+ updated tests |

### Architecture Pivot Timeline vs Test Coverage

```
Oct 2025:  Original Phase 1-6 architecture (19 tests written)
           ↓
Nov 2025:  Phase 7 development (74 more tests, still Section-based)
           ↓
Dec 12:    **PIVOT: Section → Chunk refactor** (commit 7555231)
           Only 3 old test files updated (p4_t2, p5_t2, p5_t4)
           ↓
Jan 2026:  Snowflake Arctic integration (3 new tests)
           ↓
Mar 2026:  RELATED_TO, evidence packs, precision retrieval (30 new tests)
```

**Key insight:** 66% of tests were written BEFORE the Dec 12 architectural pivot and **have never been updated**. The test suite is a living fossil record of the Nov 2025 architecture.

---

## 5. Recommendation: Treat Test Failures as Phase 9 Hygiene, Not Bugs

### Primary Recommendation

**Do NOT treat the p1_t2/p1_t3 test failures as bugs to fix.** They should be classified as **Phase 9 test hygiene** — tests that need to be updated, removed, or rewritten to match the current architecture.

### Specific Actions Recommended

#### Immediate (Phase 9 Triage)
1. **Mark p1_t2_test.py as stale** — The 3 MCP REST endpoint tests test deliberately-disabled legacy endpoints. Either:
   - Delete these 3 tests (the functionality has moved to `mcp_app.py` which has its own contract tests in `tests/contracts/test_mcp_streamable_contracts.py`)
   - OR rewrite them to test the Streamable HTTP MCP protocol instead

2. **Mark p1_t3_test.py as stale** — Tests verify constraints/indexes with `:Section`-era names that no longer exist. Either:
   - Rewrite to verify `:Chunk`-era constraints (e.g., `chunk_id_unique`, `chunk_document_id`)
   - OR delete if equivalent coverage exists in newer tests

#### Short-term (Phase 9 Sprint 1)
3. **Audit all Era 1 tests** (19 files, Oct 12-15, 2025) — Most can be deleted since they target the original architecture. Verify whether modern replacement tests exist in `tests/unit/`, `tests/integration/`, or the `v2_2/` suite.

4. **Audit Era 2 tests** (86 files, Oct 23 – Dec 11, 2025) — Categorize each as:
   - **Still valid** (unit tests that don't touch graph schema): Keep
   - **Broken by refactor** (tests that query/write `:Section` nodes): Delete or rewrite
   - **Superseded** (tests replaced by newer equivalents): Delete

#### Medium-term (Phase 9 Sprint 2)
5. **Establish test freshness hygiene** — Any test not modified in 6+ months should be reviewed. Consider adding a CI check that flags stale test files.

6. **Align test suite with actual architecture** — The current architecture (Chunk-based graph, Streamable HTTP MCP, multi-provider embeddings, RELATED_TO edges, evidence packs) should have dedicated, well-maintained test suites. The 47 current tests (Era 3-4) are a solid foundation.

### Why NOT to "fix" the failing tests

Fixing p1_t2/p1_t3 to pass against the current codebase would mean either:
- **(A)** Re-enabling the legacy REST endpoints → Reverses an intentional deprecation
- **(B)** Rewriting the tests to match current architecture → This IS Phase 9 hygiene, not a bug fix
- **(C)** Mocking the old behavior → Tests that mock the thing they're testing have zero value

Option (B) is correct, and it's properly scoped as **test modernization work**, not regression repair.

### Trust Level for Different Test Groups

| Test Group | Trust Level | Rationale |
|-----------|-------------|-----------|
| `tests/unit/test_*.py` (Mar 2026) | **HIGH** | Recent, aligned with current architecture |
| `tests/mcp_server_tests/` | **HIGH** | Tests current MCP app (Streamable HTTP) |
| `tests/contracts/` | **HIGH** | Contract tests for current APIs |
| `tests/providers/` (Jan+ 2026) | **HIGH** | Current provider implementations |
| `tests/query/` (Mar 2026) | **HIGH** | Current retrieval pipeline |
| `tests/v2_2/` (Nov 2025) | **MEDIUM** | May predate Chunk migration |
| `tests/integration/` (mixed) | **MIXED** | Check individual file dates |
| `tests/p*_t*_test.py` (Oct 2025) | **LOW/DEAD** | Pre-refactor, most will fail |
| `tests/test_phase{1-5}_*.py` | **LOW/DEAD** | Pre-refactor, most will fail |
| `tests/e2e_v22_prod/` | **UNVERIFIED** | Conftest says "tests not executed yet" |

---

## Appendix: Data Sources

- **Git history**: `git log --format="%h %ad %s" --date=short` for each test file
- **Endpoint verification**: `grep` of `/mcp/initialize`, `/mcp/tools/*` in `src/mcp_server/main.py`
- **Label verification**: `grep "(:Section {" src/` (0 results) vs `grep "(:Chunk {" src/` (10+ results)
- **Legacy flag**: `MCP_HTTP_LEGACY_REST_ENABLED` defaults to `"false"` at line 74-75 of `main.py`
- **Import verification**: `src/shared/observability/__init__.py` exports `get_logger` (package exists)
- **File counts**: `find tests -name "*.py" -not -name "__init__.py" -not -name "conftest.py"` = 159 files

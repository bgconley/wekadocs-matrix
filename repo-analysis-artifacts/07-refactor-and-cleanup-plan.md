# 07 — Refactor and Cleanup Plan

## Guiding Principles

1. **Never break production** — verify each change against running services before proceeding.
2. **Dead code removal requires import verification** — confirm zero callers before deleting.
3. **Phase-gated** — each step should be independently verifiable and reversible.
4. **Low-risk wins first** — build confidence with easy deletions before tackling coupling points.

## Phase 1: Low-Risk Wins (No Code Changes, No Service Impact)

### 1.1 Delete Generated/Archived Artifacts
**Risk:** None (read-only cleanup)
**Verification:** `git status` shows only deletions

| Action | Files |
|--------|-------|
| Delete AI session artifacts | `claude-raw/` (39 files) |
| Delete session context files | `context-*.md` (27 files) |
| Delete old archives | `Archive.zip`, `Archive 2.zip` |
| Delete backup file | `src/mcp_server/mcp_app.py.bak` |
| Delete generated reports | `reports/` (355+ files) — or move to external storage |
| Add to `.gitignore` | `claude-raw/`, `context-*.md`, `reports/` |

**Test gate:** `git diff --stat` shows only deletions. No source files modified.

### 1.2 Add Generated Directories to `.gitignore`
**Risk:** None
**Verification:** `.gitignore` diff reviewed

| Action | Files |
|--------|-------|
| Add `claude-raw/` | Already partially covered |
| Add `reports/` | Not currently gitignored |
| Add `context-*.md` | Already gitignored per AGENT_CONTEXT.md rule |

## Phase 2: Dead Code Removal (No Runtime Impact)

### 2.1 Delete Entirely Dead Packages
**Risk:** Low — all files marked `@status: DEAD`, no imports found
**Verification:** `grep -r "from src.learning\|from src.registry\|from src.ops.warmers\|from src.shared.audit\|from src.shared.feature_flags\|from src.query.graph_expansion\|from src.query.graph_features\|from src.query.diffusion_reranker" src/ tests/` returns zero results

| Action | Files |
|--------|-------|
| Delete learning package | `src/learning/` (4 files: `__init__.py`, `feedback.py`, `ranking_tuner.py`, `suggestions.py`) |
| Delete registry package | `src/registry/` (2 files: `__init__.py`, `index_registry.py`) |
| Delete ops/warmers package | `src/ops/warmers/` (2 files: `__init__.py`, `query_warmer.py`) |
| Delete dead shared modules | `src/shared/audit/logger.py`, `src/shared/feature_flags.py` |
| Delete dead query modules | `src/query/graph_expansion.py`, `src/query/graph_features.py`, `src/query/diffusion_reranker.py` |

**Test gate:** `pytest` passes. No import errors.

### 2.2 Delete Dead Ingestion Modules
**Risk:** Low — all files marked `@status: DEAD`, no imports from active code
**Verification:** Same grep pattern as above

| Action | Files |
|--------|-------|
| Delete dead auto modules | `src/ingestion/auto/orchestrator.py`, `src/ingestion/auto/backpressure.py`, `src/ingestion/auto/report.py`, `src/ingestion/auto/verification.py`, `src/ingestion/auto/watcher.py` |
| Delete dead api.py | `src/ingestion/api.py` |
| Delete dead parsers | `src/ingestion/parsers/notion.py` |
| Delete dead ingestion modules | `src/ingestion/saga.py`, `src/ingestion/reconcile.py`, `src/ingestion/incremental.py` |

**Test gate:** `pytest` passes. Ingestion worker starts successfully.

### 2.3 Delete Dead MCP Server Modules
**Risk:** **MEDIUM** — these are security modules. See R1 in risk register.
**Verification:** Confirm `main.py` does not import from `src.mcp_server.security`

| Action | Files |
|--------|-------|
| Delete dead security modules | `src/mcp_server/security/__init__.py`, `auth.py`, `rate_limiter.py` |
| Delete dead validation module | `src/mcp_server/validation.py` |

**⚠️ WARNING:** Before deleting security modules, either:
- Wire them into `main.py` (add JWT auth middleware and rate limiter), OR
- Document the security gap in `main.py` comments and add a TODO

**Test gate:** MCP server starts successfully. `curl http://localhost:8000/health` returns 200.

### 2.4 Delete Dead Neo4j Modules
**Risk:** **MEDIUM** — `contract_checks.py` is imported by worker.py (see R2)
**Verification:** `grep -n "contract_checks" src/ingestion/worker.py`

| Action | Files |
|--------|-------|
| Delete safe dead modules | `src/neo/entity_normalization.py`, `src/neo/explain_guard.py`, `src/neo/defensive_query.py`, `src/neo/graph_enhancements.py`, `src/neo/health.py` |
| **DO NOT YET DELETE** | `src/neo/contract_checks.py` — imported by worker.py |
| **DO NOT YET DELETE** | `src/neo/structural_builder.py` — circular import, ambiguous |

**Test gate:** Ingestion worker starts successfully.

## Phase 3: Resolve Active-Dead Coupling (Requires Code Changes)

### 3.1 Fix `src/neo/contract_checks.py` Import
**Risk:** High — changing worker.py import path
**Verification:** Run ingestion worker against live services

**Option A — Make contract_checks.py ACTIVE:**
1. Move `src/neo/contract_checks.py` to `src/ingestion/contract_checks.py`
2. Update import in `worker.py` line 260
3. Update `@status:` annotation to ACTIVE
4. Wire into worker with clear ownership

**Option B — Remove the import:**
1. Remove `from src.neo.contract_checks import GraphContractChecker` from `worker.py` line 260
2. Verify `GraphContractChecker` is not used elsewhere in worker.py
3. Delete `src/neo/contract_checks.py`

**Test gate:** `pytest` passes. Ingestion worker processes a test job successfully.

### 3.2 Migrate `GraphBuilder` from `build_graph.py`
**Risk:** High — changes active ingestion path
**Verification:** Run full ingestion pipeline against live services

1. Read `src/ingestion/build_graph.py` to understand `GraphBuilder` dependencies
2. Move `GraphBuilder` class (and dependencies) into `src/ingestion/atomic.py` or a new `src/ingestion/graph_builder.py`
3. Update the lazy import in `atomic.py` line 865
4. Update `@status:` annotation on `build_graph.py` to DEAD
5. Delete `src/ingestion/build_graph.py`

**Test gate:** `pytest` passes. Ingestion worker processes a test document successfully. Verify Neo4j and Qdrant contain expected data.

### 3.3 Resolve `src/shared/connections.py` DEAD Methods
**Risk:** Medium — changes shared foundation module
**Verification:** `grep -rn "CompatQdrantClient.delete_sections_v1\|CompatQdrantClient.delete_sections_v2" src/ tests/`

1. Verify `delete_sections_v1` and `delete_sections_v2` have zero callers
2. Remove the DEAD methods from `CompatQdrantClient`
3. Update `@status:` annotations in the file header

**Test gate:** `pytest` passes. Qdrant operations (delete, upsert) work correctly.

## Phase 4: Schema and Config Cleanup

### 4.1 Consolidate Neo4j Schema Files
**Risk:** Medium — affects schema management
**Verification:** Compare DDL across files

1. Identify canonical schema: likely `create_schema_v2_2_complete__phase7E.cypher`
2. Archive older versions with clear version labels in a `scripts/neo4j/archived/` directory
3. Fix or delete `scripts/apply_complete_schema_v2_1.py`
4. Document which schema file is canonical in `scripts/neo4j/README.md`

**Test gate:** `scripts/init_schema.py` applies schema successfully. `neo4j_schema_dump.cypher` matches expected v2.2 schema.

### 4.2 Differentiate or Delete `production.yaml`
**Risk:** Low
**Verification:** Compare `config/development.yaml` vs `config/production.yaml`

1. If production needs differ: update `production.yaml` with production-appropriate values
2. If production uses the same config: delete `production.yaml` and document this

**Test gate:** `ENV=production python -c "from src.shared.config import get_config; print(get_config())"` works correctly.

## Phase 5: Documentation and Hardening

### 5.1 Update `@status:` Annotations
**Risk:** Low
**Verification:** `scripts/ci/check_dead_imports.py` passes

1. After all deletions, run `scripts/ci/check_dead_imports.py` to verify no dead imports remain
2. Update any `@status:` annotations that became stale during cleanup
3. Ensure all ACTIVE modules have clear `@called-by:` annotations

### 5.2 Wire or Document Security Gap
**Risk:** High — security impact
**Verification:** Security audit

1. If wiring in: add JWT auth middleware and rate limiter to `main.py`
2. If not wiring: add explicit TODO and security documentation in `main.py` header
3. Update `docs/` with security architecture documentation

### 5.3 Create `scripts/neo4j/README.md`
**Risk:** Low
**Verification:** File exists and is readable

Document:
- Canonical schema file
- How to apply schema (`scripts/init_schema.py`)
- How to backup schema (`inventory_neo4j.py`)
- How to reset databases (`scripts/reset_datastores.py`)
- Archived schema versions and their purposes

## Migration Strategy for Ambiguous Paths

### `src/ingestion/saga.py`
1. Compare `saga.py` with `atomic.py`'s inline saga logic
2. If `saga.py` contains the canonical implementation: move it to `src/ingestion/saga_coordinator.py` and update imports
3. If `atomic.py` has the canonical implementation: delete `saga.py`

### `src/neo/structural_builder.py`
1. Investigate the circular import
2. If it's truly dead: delete
3. If it's needed: move to `src/ingestion/` and fix the import

### `config/production.yaml`
1. Verify with deployment team whether production uses different config
2. If yes: differentiate the file
3. If no: delete and document

## Low-Risk Wins vs High-Risk Changes

| Category | Actions | Risk | Effort |
|----------|---------|------|--------|
| **Low-risk wins** | Delete archives, reports, AI artifacts, dead packages | None | 1-2 hours |
| **Low-medium** | Delete dead ingestion/MCP/neo modules (except contract_checks) | Low | 2-4 hours |
| **Medium** | Fix contract_checks.py import, delete dead security modules | Medium | 4-8 hours |
| **High** | Migrate GraphBuilder, consolidate schema files | High | 1-2 days |
| **High** | Wire security modules or document gap | High | 1-2 days |

## Suggested Test Gates Per Phase

| Phase | Test Command | Success Criteria |
|-------|-------------|------------------|
| 1 | `git diff --stat` | Only deletions, no source changes |
| 2 | `pytest` | All tests pass, no import errors |
| 2 | `docker compose up -d mcp-server` | MCP server starts, `/health` returns 200 |
| 2 | `docker compose up -d ingestion-worker` | Worker starts, no import errors in logs |
| 3 | `pytest` | All tests pass |
| 3 | Ingest test document | Neo4j and Qdrant contain expected data |
| 4 | `scripts/init_schema.py` | Schema applied successfully |
| 4 | `ENV=production python -c "..."` | Config loads correctly |
| 5 | `scripts/ci/check_dead_imports.py` | Zero dead import violations |

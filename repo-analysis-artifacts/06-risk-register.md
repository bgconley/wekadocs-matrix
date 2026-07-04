# 06 — Risk Register

## Severity Key

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Will cause production failure, data loss, or security breach |
| **HIGH** | Will cause significant degradation, incorrect behavior, or maintainability crisis |
| **MEDIUM** | Will cause confusion, technical debt accumulation, or minor failures |
| **LOW** | Cosmetic, noise, or minor inefficiency |

## Risk Table

### R1 — No Authentication or Rate Limiting on MCP Server
- **Severity:** CRITICAL
- **Classification:** Proven issue
- **Evidence:** `src/mcp_server/security/auth.py` and `src/mcp_server/security/rate_limiter.py` exist with `@status: DEAD`. Neither is imported by `src/mcp_server/main.py`. The FastAPI app has no auth middleware or rate limiting configured.
- **Why it matters:** The MCP server is publicly accessible (NGINX ingress, TLS). Any client can call MCP tools without authentication or rate limits. This enables abuse, resource exhaustion, and unauthorized access to the knowledge base.
- **Suggested mitigation:** Either wire `src/mcp_server/security/auth.py` JWT middleware into `main.py` and configure rate limiting, or explicitly document the security gap and add a network-level access control (IP whitelist, API gateway).
- **Confidence:** High — the security modules exist but are definitively not wired in.

### R2 — DEAD Module Imported at Runtime (`src/neo/contract_checks.py`)
- **Severity:** HIGH
- **Classification:** Proven issue
- **Evidence:** `src/neo/contract_checks.py` has `@status: DEAD` but is imported by `src/ingestion/worker.py` at line 260: `from src.neo.contract_checks import GraphContractChecker`. This import happens inside the worker's main processing loop.
- **Why it matters:** If this module has import-time errors, missing dependencies, or runtime bugs, the ingestion worker will fail to process jobs. The `@status: DEAD` annotation creates a false sense of safety — developers may modify or remove "dead" code without realizing it's actively imported.
- **Suggested mitigation:** Either make `contract_checks.py` properly ACTIVE (wire it into the worker with clear ownership) or remove the import and delete the module. If it's needed for contract validation, move it to `src/ingestion/` or `src/shared/`.
- **Confidence:** High — the import is explicit and in the worker's hot path.

### R3 — Legacy `build_graph.py` Coupling
- **Severity:** HIGH
- **Classification:** Proven issue
- **Evidence:** `src/ingestion/build_graph.py` has `@status: DEPRECATED` but is lazily imported by `src/ingestion/atomic.py` at line 865: `from src.ingestion.build_graph import GraphBuilder`. The `GraphBuilder` class is used by the active ingestion path.
- **Why it matters:** The production ingestion path depends on a deprecated module. If `build_graph.py` is modified or removed by a developer who sees the `@status: DEPRECATED` annotation, ingestion will break. This creates a hidden coupling.
- **Suggested mitigation:** Migrate `GraphBuilder` usage into `atomic.py` or a new module in `src/ingestion/`, then delete `build_graph.py`.
- **Confidence:** High — the lazy import is explicit in atomic.py.

### R4 — Multiple Neo4j Schema DDL Files — No Canonical
- **Severity:** MEDIUM
- **Classification:** Proven issue
- **Evidence:** `scripts/neo4j/` contains at least 6 different schema DDL files spanning v2.0 through v2.2:
  - `create_schema.cypher` (base)
  - `create_schema_v2_1.cypher`
  - `create_schema_v2_1_complete.cypher`
  - `create_schema_v2_1_complete__v3.cypher`
  - `create_graphrag_schema_v2_2_20251105.cypher`
  - `create_graphrag_schema_v2_2_20251105_guard.cypher`
  - `create_schema_v2_2_complete__phase7E.cypher`
  - `schema_backup_*.cypher` (2 archived backups)
  - `schema_migration_phase3_markdown_it_py.cypher`
- **Why it matters:** Developers may apply the wrong schema version, causing schema mismatches between the code (`src.shared.schema.create_schema()`) and the database. The broken script `apply_complete_schema_v2_1.py` references a non-existent Cypher file, suggesting drift between scripts and actual files.
- **Suggested mitigation:** Pick one canonical DDL file (likely `create_schema_v2_2_complete__phase7E.cypher`), archive the rest with clear version labels, and update `apply_complete_schema_v2_1.py` or delete it.
- **Confidence:** High — multiple files exist with different version numbers.

### R5 — `scripts/apply_complete_schema_v2_1.py` is Broken
- **Severity:** MEDIUM
- **Classification:** Proven issue
- **Evidence:** The script references `scripts/neo4j/create_schema_v2_1_complete.cypher` which exists, but the script's import path and execution context may be wrong. The `scripts/neo4j/` directory is mounted as read-only into the Neo4j container (`./scripts/neo4j:/scripts:ro`), but the script tries to read from the filesystem.
- **Why it matters:** If someone tries to use this script to apply the schema, it will fail. This could cause confusion during deployment or recovery.
- **Suggested mitigation:** Fix the file path or delete the script if `init_schema.py` (which uses `src.shared.schema.create_schema()`) is the canonical path.
- **Confidence:** Medium — need to verify the exact error by running the script.

### R6 — Config Duplication (production.yaml = development.yaml)
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:** `config/production.yaml` is identical to `config/development.yaml` (both 566 lines, same content).
- **Why it matters:** If production needs differ from development (different embedding profiles, different cache sizes, different feature flags), those differences are not reflected. This creates a risk of deploying with incorrect production configuration.
- **Suggested mitigation:** Differentiate `production.yaml` from `development.yaml` with production-appropriate values, or delete `production.yaml` and document that the same config is used for all environments.
- **Confidence:** High — confirmed identical content.

### R7 — Hardcoded New Relic Key in `.env.production`
- **Severity:** HIGH
- **Classification:** Proven issue
- **Evidence:** `.env.production` contains `NEW_RELIC_LICENSE_KEY=c71e093ba36af151d238c1e443093261FFFFNRAL`. While `.env.production` is gitignored, it exists in the working tree and could be accidentally committed. The key format suggests it's a real (or test) New Relic license key.
- **Why it matters:** If committed to git, API keys are exposed. Even if gitignored, the file exists in the repo and could be added to version control.
- **Suggested mitigation:** Remove the hardcoded key from `.env.production`. Use a secrets manager or environment variable injection for production. Add a pre-commit hook to detect hardcoded keys.
- **Confidence:** High — the key is explicitly in the file.

### R8 — Repo Bloat (400+ non-source files)
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:**
  - `claude-raw/`: 39 AI session artifacts
  - `context-*.md`: 27 session context files
  - `reports/`: 355+ generated test/ingestion artifacts
  - `Archive.zip`, `Archive 2.zip`: 2 old code snapshots
  - `mcp_app.py.bak`: 1 backup file
  - `hf-cache/`: HuggingFace model cache
  - `.npm-cache/`, `.pytest_cache/`, `.ruff_cache/`, `.uv-cache/`: Build caches
- **Why it matters:** Consumes significant disk space, adds noise to git operations, increases clone time, and makes it harder to find actual source files. The `reports/` directory alone has 355+ files.
- **Suggested mitigation:** Move generated artifacts to external storage (S3, artifact server). Add `.claude-raw/`, `context-*.md`, `reports/` to `.gitignore`. Delete `Archive.zip`, `Archive 2.zip`, `mcp_app.py.bak`.
- **Confidence:** High — file counts are verifiable.

### R9 — Empty K8s Overlays
- **Severity:** LOW
- **Classification:** Hypothesis
- **Evidence:** `deploy/k8s/overlays/staging/` and `deploy/k8s/overlays/production/` exist as empty directories. The CI workflow references `kubectl apply -k deploy/k8s/overlays/staging` for staging deployment.
- **Why it matters:** If the overlays are empty, the staging deployment uses the base config, which may not have staging-appropriate settings (different DB URIs, different replica counts, etc.).
- **Suggested mitigation:** Create proper Kustomize overlays for staging and production, or remove the empty directories and update CI to use base config for all environments.
- **Confidence:** Medium — directories exist but are empty.

### R10 — Entire `src/learning/` Package is Dead
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:** All 4 files in `src/learning/` have `@status: DEAD`. No module imports from this package.
- **Why it matters:** Dead code increases cognitive load and maintenance burden. Developers may waste time understanding unused code.
- **Suggested mitigation:** Delete the entire `src/learning/` package.
- **Confidence:** High — all files marked DEAD, no imports found.

### R11 — Multiple Dead Query Modules
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:** `src/query/graph_expansion.py`, `src/query/graph_features.py`, and `src/query/diffusion_reranker.py` all have `@status: DEAD`. No callers found.
- **Why it matters:** These modules may contain useful logic that was superseded but not removed. They add noise to the codebase.
- **Suggested mitigation:** Review for reusable logic, then delete.
- **Confidence:** High — all marked DEAD, no imports found.

### R12 — `src/registry/` Package is Dead
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:** Both files in `src/registry/` have `@status: DEAD`. No callers found.
- **Why it matters:** Dead code.
- **Suggested mitigation:** Delete the entire `src/registry/` package.
- **Confidence:** High — all files marked DEAD, no imports found.

### R13 — `src/ops/warmers/` Package is Dead
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:** Both files have `@status: DEAD`. No callers found.
- **Why it matters:** Dead code.
- **Suggested mitigation:** Delete the entire `src/ops/warmers/` package.
- **Confidence:** High — all files marked DEAD, no imports found.

### R14 — `src/shared/audit/logger.py` and `src/shared/feature_flags.py` are Dead
- **Severity:** LOW
- **Classification:** Proven issue
- **Evidence:** Both files have `@status: DEAD`. No callers found.
- **Why it matters:** Dead code in the shared foundation layer.
- **Suggested mitigation:** Delete both files.
- **Confidence:** High — both marked DEAD, no imports found.

### R15 — `src/ingestion/saga.py` is Dead but May Contain Reusable Logic
- **Severity:** MEDIUM
- **Classification:** Hypothesis
- **Evidence:** `src/ingestion/saga.py` has `@status: DEAD` but contains `IngestionValidator` and `SagaContext` classes. The `atomic.py` module has its own inline saga logic. It's unclear if `saga.py` was refactored into `atomic.py` or if both coexist.
- **Why it matters:** If `saga.py` contains the canonical saga implementation and `atomic.py` has duplicated logic, this is a code duplication issue. If `saga.py` is truly dead, it should be deleted.
- **Suggested mitigation:** Compare `saga.py` with `atomic.py`'s inline saga logic. If duplicated, keep one and delete the other.
- **Confidence:** Medium — need to compare the actual implementations.

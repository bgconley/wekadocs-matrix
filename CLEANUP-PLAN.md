# CLEANUP-PLAN — WekaDocs Matrix Full-Repo Overhaul

> This document is the single source of truth for the cleanup effort.
> It **supersedes** `REFACTOR-PLAN.md` (which is now marked deprecated).
> Every phase here is grounded in the LOC numbers verified in `AUDIT-VERIFICATION.md`.
>
> ## Related artifacts (in this repo)
> | File | Role |
> | --- | --- |
> | `REPO-MAP.md` | Every `src/` file, LOC, and status. Source of truth for "what exists." |
> | `ARCHITECTURE.md` | Mermaid diagrams of the current system (data flow, schemas, tools). |
> | `DEV-HISTORY.md` | Why the codebase looks the way it does (six architectural pivots). |
> | `AUDIT-VERIFICATION.md` | Proof of every LOC/status claim; the error log from auditing this effort. |
> | `DEAD-CODE-MAP.md` | Quick-reference kill list (with corrections for planner.py, hybrid_search.py). |
> | `REFACTOR-PLAN.md` | **Deprecated.** Superseded by this file. |

---

## 0. Goal

Transform `wekadocs-matrix` from its current state into a repo with these properties:

| Property | Current (verified) | Target |
|---|---|---|
| `src/` LOC | 68,551 across 169 files | ≤ 40,000 across ≤ 100 files |
| Full-file dead code | 9,038 LOC | 0 |
| Dead methods inside mixed files | ~4,600 LOC | 0 |
| Monolithic files (>1,000 LOC) | 5 — atomic.py (4052), mcp_app.py (3834), build_graph.py (3200), hybrid_retrieval.py (2095), config.py (2015), plus vector_backends.py (1750), expansion_pipeline.py (1165), tokenizer_service.py (1168), cross_doc_linking.py (1419), mcp_server/query_service.py (1157) | 0 file over 1,000 LOC; top-level orchestrators under 750 |
| Embedding providers wired | 6 | 6 (modular design for experimentation/deployment scenarios) |
| Chonkie adapters | 3 near-identical | 1 generic class |
| HTTP client classes | 3 near-identical | 1 generic class |
| CircuitBreaker classes | 3 | 1 (lives in `shared/resilience/`) |
| Markdown parsers | 2 + shadow comparison | 1 |
| Chunk assemblers | 3 | 1 (semantic) |
| Retrieval engines | 2 | 1 (`HybridRetriever`) |
| MCP transports | 3 (streamable + legacy REST + stdio) | 2 (drop legacy REST; keep stdio for Claude Desktop) |
| MCP tools (canon + aliases) | 16 + 16 aliases + 1 legacy | 16 canonical, zero aliases |
| Env-var knobs | 85+ | ≤ 20 (documented in one place) |
| Feature-flag knobs | ~18 in `development.yaml` + JSON + env | Single YAML file, ≤ 6 flags |
| Test suite pass rate (live) | UNMEASURED (historical 93% claimed) | 100% green before any merge |
| Standalone CLI tools scattered | 8 (`bootstrap_schema.py`, `inventory_*.py`, `db-check.py`, `tools/fusion_ab.py`, `tools/redis_*.py`, `src/ingestion/auto/cli.py`, `src/tools/inspect_chunk.py`) | 1 unified `wekadocs` CLI |
| Branch state | `multi-embedder-reranker` unmerged with Qwen3 + evidence pack redesign | Branch merged, master up to date |

**"Done" means**: every phase's exit criteria green, the test suite passes end-to-end, a single doc can be ingested and a single query can be answered against a fresh Qdrant + Neo4j, and the code passes `ruff check` + `mypy` with no suppressions.

---

## 1. Rules

1. **A safety net before every destructive phase.** The test suite must be collectable and runnable before any Phase that deletes or restructures. If Phase 0 (below) can't be completed, nothing else moves.
2. **One PR per phase.** Each phase's exit criteria must be green on its own before the next phase starts. No mega-PRs.
3. **No "move to a branch for later."** Every deletion is final. Every refactor is reviewed. Every stub is documented as intentional.
4. **Subagent work is always independently spot-checked.** The audit (AUDIT-VERIFICATION.md) taught this lesson. After any delegated deletion/refactor, a human or a verifiable command (`grep -rn ... src/` returning zero hits, or `python -c "import ..."`) must confirm the result.
5. **Historical test numbers remain labeled historical** until re-measured.

---

## 2. Phase 0 — Empirical Baseline (prerequisite for all else)

**Goal:** Establish what *actually* works today. No claims, only measurements.

### Actions
1. Set up the Python/venv environment. Document exact Python version in `.python-version`.
2. `pytest --collect-only -q > /tmp/collected_tests.txt` — record the count.
3. `pytest -q --junitxml=reports/baseline/junit.xml > reports/baseline/pytest.log 2>&1`. Record:
   - total collected, passed, failed, errors, skipped, xfail
   - top-10 slowest tests
4. `ruff check src/ > reports/baseline/ruff.log 2>&1`. Record: total issues by category.
5. `mypy src/ --ignore-missing-imports > reports/baseline/mypy.log 2>&1`. Record: error count.
6. Smoke test: drop one real Weka doc into `data/documents/spool/`, run `make up` + worker, verify Qdrant has points and Neo4j has Document/Section/Chunk nodes. Record pass/fail.
7. Smoke test: call `kb.retrieve_evidence` via the MCP streamable HTTP transport with a real question. Verify you get ≥1 chunk back with provenance. Record pass/fail.

### Exit criteria
- `reports/baseline/` directory committed as the measurement baseline.
- README updated with the actual numbers.
- Smoke tests pass. If either smoke test fails, **Phase 0 is not done**; the failure must be understood and a fix must land before moving on.

### LOC impact
0 (measurement only).

### Rollback
N/A — phase is read-only.

---

## 3. Phase 1 — Full-File Dead Code Removal

**Goal:** Delete the 9,038 LOC of files that have zero live imports.

### Actions — safe to delete now
```
src/learning/              # 4 files,  983 LOC
src/registry/              # 2 files,  282 LOC
src/ops/optimizer.py       #           499 LOC
src/ops/warmers/           # 2 files,  162 LOC
src/mcp_server/validation.py           407 LOC
src/mcp_server/security/   # 3 files,  355 LOC
src/shared/feature_flags.py            173 LOC
src/shared/audit/          # 2 files,  218 LOC
src/neo/explain_guard.py               276 LOC
src/neo/entity_normalization.py        274 LOC
src/neo/graph_enhancements.py          440 LOC
src/neo/structural_builder.py          520 LOC
src/neo/defensive_query.py             112 LOC
src/neo/health.py                      117 LOC
src/query/diffusion_reranker.py        362 LOC
src/query/graph_features.py            339 LOC
src/query/graph_expansion.py           302 LOC
src/query/templates/advanced/schemas.py 150 LOC
src/ingestion/api.py                    35 LOC
src/ingestion/reconcile.py             525 LOC
src/ingestion/incremental.py           362 LOC
src/ingestion/parsers/notion.py        256 LOC
src/ingestion/auto/watcher.py           54 LOC
src/ingestion/auto/orchestrator.py   1,134 LOC
src/ingestion/auto/verification.py     277 LOC
src/ingestion/auto/report.py           296 LOC
src/ingestion/auto/backpressure.py     282 LOC
                          ─────
              TOTAL        9,038 LOC
```

### Actions — also delete the now-orphan tests
For each package above, find matching `tests/**/test_*.py` that `import from src.<deleted>`, delete those tests too. Run `grep -rn "src.learning\|src.registry\|src.ops.optimizer\|src.ops.warmers\|src.neo.explain_guard\|..." tests/` to find them. Don't guess.

### Exit criteria
- `pytest --collect-only` runs cleanly: zero import errors, zero collection errors.
- `pytest` pass count ≥ Phase 0 pass count − (deleted test count). Any NEW failure is a bug — fix before proceeding.
- `grep -rn "src.learning\|src.registry\|src.ops.optimizer\|src.ops.warmers\|src.neo.explain_guard\|src.neo.entity_normalization\|src.neo.graph_enhancements\|src.neo.structural_builder\|src.neo.defensive_query\|src.neo.health\|src.query.diffusion\|src.query.graph_features\|src.query.graph_expansion\|src.query.planner\|src.ingestion.api\|src.ingestion.reconcile\|src.ingestion.incremental\|src.ingestion.parsers.notion\|src.ingestion.auto.watcher\b\|src.ingestion.auto.orchestrator\|src.ingestion.auto.verification\|src.ingestion.auto.report\|src.ingestion.auto.backpressure\|src.mcp_server.validation\|src.mcp_server.security\|src.shared.feature_flags\|src.shared.audit" src/ tests/` returns zero hits.
- ⚠️ **planner.py is NOT in this list** — verified active (query_service.py:827 calls `planner.plan()`).
- ⚠️ **hybrid_search.py is NOT in this list** — see Phase 2.
- ⚠️ **ops/session_cleanup_job.py is NOT in this list** — verified standalone CLI with `__main__`.

### LOC impact
−9,038 (full files) plus whatever orphaned tests are removed.

### Rollback
Git revert of a single commit.

---

## 4. Phase 2 — Strip Dead Methods from Mixed-Status Files

**Goal:** Inside files that are kept, remove methods/functions that the audit confirmed are not called.

### Targets

#### `src/ingestion/build_graph.py` (3,200 → ~350 LOC)
**KEEP these 8 members (used by `atomic.py`):**
- `__init__`, `ensure_embedder`, `embedder` (property)
- `embedding_plan`, `embedding_settings`, `colbert_dims`, `colbert_settings`
- `_build_section_text_for_embedding`, `_build_title_text_for_embedding`

**DELETE everything else:** `upsert_document`, `_upsert_document_node`, `_upsert_sections`, `_upsert_entities`, `_create_mentions`, `_process_embeddings`, `_ensure_qdrant_collection`, `_build_entity_text_for_embedding`, `_compute_text_hash`, `_compute_shingle_hash`, `_extract_semantic_metadata`, and the standalone `ingest_document()` function at the bottom. Every deletion must be verified with `grep -rn "build_graph.<name>" src/`.

#### `src/ingestion/saga.py` (688 → ~280 LOC)
**KEEP:** `SagaContext`, `ValidationResult`, `IngestionValidator` (and its helpers).
**DELETE:** `SagaStatus`, `StepStatus`, `SagaStepResult`, `SagaStep`, `SagaCoordinator`, `IngestionSagaBuilder`, `SagaStepFailure`, `SagaCompensationFailure`.

#### `src/neo/contract_checks.py` (455 → ~80 LOC)
**KEEP:** `GraphContractChecker.__init__` and `find_documents_needing_repair`.
**DELETE:** `run_all_checks`, `check_chunk_document_membership`, `check_next_chunk_no_branching`, `check_next_chunk_no_cycles`, `check_hierarchy_coverage`, `check_entity_normalization`, `check_chunk_id_uniqueness`, `run_contract_checks`.

### Exit criteria
- `grep` confirms each deleted name has zero non-comment, non-docstring references in `src/` and `tests/`.
- Test suite: same pass rate ± (tests of deleted methods, which are also deleted in Phase 2).
- Smoke test (ingest a doc + query) still passes.

### LOC impact
~−3,705.

### Rollback
Git revert.

---

## 5. Phase 3 — De-couple and Delete `hybrid_search.py`

**Goal:** Remove the second retrieval engine that `query_service.py` still leans on.

### Actions (strict order)
1. **Read `src/mcp_server/query_service.py` and map every call into `HybridSearchEngine`.** The audit found: import at line 25 (`HybridSearchEngine, QdrantVectorStore, SearchResult`), lazy import at line 233 (`Neo4jVectorStore`), instantiation at line 246, ordering reference at line 1035, plus `_get_search_engine()` cached accessor and `_wrap_chunks_as_ranked` / `_wrap_search_results_as_ranked` adapters.
2. **Delete the legacy fallback path.** `HybridRetriever` is the only engine that should be used. Replace the legacy call chain with calls to `HybridRetriever`. The two `_wrap_*` adapter methods become dead and should be deleted.
3. **Retire the `ranker` in `ranking.py`.** `RankedResult` and `RankingFeatures` are used by `response_builder.py` and `query_service.py` — **keep them**. The `Ranker.rank()` method is bypassed by `HybridRetriever` and is dead — delete it.
4. **Remove the `SearchResult` import from `ranking.py:23`.** `ranking.py` should use `ChunkResult` only.
5. **Delete `src/query/hybrid_search.py` (916 LOC)** and every related type: `SearchResult`, `HybridSearchResults`, `QdrantVectorStore`, `Neo4jVectorStore`, `VectorStore` base, `HybridSearchEngine`.
6. Final verification: `grep -rn hybrid_search src/ tests/` returns zero non-comment hits. `grep -rn "HybridSearchEngine\|Neo4jVectorStore\b\|class VectorStore" src/ tests/` returns zero.

### Exit criteria
- Smoke test passes against a freshly populated store.
- Response shape identical for the same query (use the `golden_query_set.yaml` fixture).

### LOC impact
−916 + adapter removals in query_service + Ranker.rank removal in ranking.

---

## 6. Phase 4 — Monolith Decomposition

**Goal:** No file over 1,000 LOC. Each god module becomes a package of focused modules.

### 4a. `atomic.py` (4,052) → `src/ingestion/pipeline/`
New package layout:
```
src/ingestion/pipeline/
    __init__.py                # re-exports AtomicIngestionCoordinator
    coordinator.py             # orchestration (~400 LOC)
    parser.py                  # parse+assemble (~200 LOC, thin glue to parsers/ + chunk_assembler)
    entity_extractor.py        # entities + GLiNER (~150 LOC)
    embedder.py                # dense/sparse/ColBERT computation (~600 LOC)
    neo4j_writer.py            # graph writes + structural edges (~500 LOC)
    qdrant_writer.py           # vector writes + compensation (~400 LOC)
    saga.py                    # transaction coordination (replaces inline _execute_atomic_saga) (~200 LOC)
    cross_doc_linker.py        # post-commit cross-doc linking (~200 LOC)
```
`atomic.py` is deleted; the thin facade it used to be becomes `coordinator.py`.

### 4b. `mcp_app.py` (3,834) → `src/mcp_server/tools/`
```
src/mcp_server/
    factory.py                 # build_mcp_server() — thin, delegates (~200 LOC)
    deps.py                    # Deps container + lifespan
    tools/
        __init__.py
        kb/                    # 6 kb.* tools, one file per tool
            search.py          # kb.search → _kb_search_candidates (~200)
            read_excerpt.py
            expand_excerpt.py
            extract_evidence.py # kb.extract_evidence → the evidence pipeline (~400)
            retrieve_evidence.py # kb.retrieve_evidence → full evidence pack
            search_sections.py
            get_section_text.py
        graph/                 # 10 graph.* tools
            describe.py, expand.py, paths.py, parents.py, children.py,
            entities_for_sections.py, sections_for_entities.py,
            traverse.py, summarize.py, context_bundle.py
        registry.py            # canonical-name → handler mapping + profile filter
```
The 16 underscore-aliases are permanently deleted in this phase (they're deprecated).

### 4c. `build_graph.py` (already trimmed in Phase 2, ~350 LOC) → inline into `src/ingestion/settings.py`
After Phase 2 it's only a settings container + 2 text helpers. These become `src/ingestion/settings.py: IngestionSettings` dataclass and two static methods. The `GraphBuilder` class is deleted.

### 4d. `hybrid_retrieval.py` (2,095) → `src/query/retrieval/`
```
src/query/retrieval/
    __init__.py
    retriever.py               # orchestrator (~500 LOC)
    vector_search.py           # multi-vector retrieval (~300)
    fusion.py                  # merge with existing fusion_pipeline.py (~200)
    rerank.py                  # merge with existing rerank_pipeline.py (~400)
    expansion.py               # merge with existing expansion_pipeline.py (~400)
    graph_channel.py           # merge with existing graph_pipeline.py (~400)
    context.py                 # merge with existing context_assembly.py (~300)
```
Result: each concern ≤500 LOC.

### 4e. `config.py` (2,015) → `src/shared/config/`
Pydantic config models are currently one god class. Split by concern:
```
src/shared/config/
    __init__.py
    app.py                     # AppConfig (tiny)
    embedding.py               # EmbeddingConfig, EmbeddingPlan, EmbeddingRolePlan
    search.py                  # VectorSearchConfig, QdrantVectorConfig, Neo4jVectorConfig
    retrieval.py               # BM25Config, ExpansionConfig, SignalPoolConfig, RerankerConfig, StructuralRetrievalConfig
    ingestion.py               # IngestionConfig, ParserConfig, ChunkAssemblyConfig, CrossDocLinkingConfig
    databases.py               # Neo4j/Qdrant/Redis connection configs
    observability.py           # MonitoringConfig
    loading.py                 # YAML loading + env override (currently inline)
```
### 4f. Other over-1000 files
- `vector_backends.py` (1,750) → split `BM25Retriever`, `QdrantMultiVectorRetriever`, `VectorRetriever` into separate files.
- `expansion_pipeline.py` (1,165) → absorbed into 4d above.
- `tokenizer_service.py` (1,168) → split into `tokenizer/hf_backend.py`, `tokenizer/jina_backend.py`, `tokenizer/service.py`.
- `cross_doc_linking.py` (1,419) → split into `linking/dense_strategy.py`, `linking/rrf_strategy.py`, `linking/colbert_strategy.py`, `linking/orchestrator.py`.
- `query_service.py` (1,157) → after Phase 3 drops the legacy engine, split into `rewriter.py`, `service.py`, `adapters.py`.

### Exit criteria
- `wc -l` on every file in `src/`, sorted, shows none over 1,000.
- Every public name previously exported by these god modules is still reachable via `src.<package>.<module>`. Backward import paths are explicitly deleted — no shim layers preserved.
- Test pass rate unchanged from Phase 3.
- Smoke tests still green.

### LOC impact
Net neutral (restructure), but cognitive complexity drops massively.

---

## 7. Phase 5 — Provider Consolidation

**Goal:** One implementation per concern. No duplicated retry, HTTP, or adapter logic.

### 5a. Unify Chonkie adapters (3 → 1 generic, keep 2 specialized)
Replace `BgeM3ChonkieAdapter` + `ArcticChonkieAdapter` + `Qwen3ChonkieAdapter` with a generic `ChonkieAdapter(http_client, tokenizer_service, settings)` base class that handles common logic (batching, retries, health checks). Keep specialized subclasses **only** where providers genuinely differ in API shape or capabilities (~400 LOC total):
- `BgeM3ChonkieAdapter` (for BGE-M3 sparse + ColBERTv2)
- `Qwen3ChonkieAdapter` (for Qwen3 dense semantic chunking)
- Delete `ArcticChonkieAdapter` if it's identical to the generic base (verify by reading the actual code)

The factory pattern and provider abstraction are **intentional modularity** (see reports/baseline/provider_architecture_analysis.md), not redundancy. The goal is to clean up the adapter layer, not delete providers.

### 5b. Unify HTTP clients (3 → 1 generic)
Replace `EmbeddingClient` + `Qwen3EmbeddingClient` + `SnowflakeEmbeddingClient` with one generic `EmbeddingClient(base_url, model, dims, api_key=None, headers=None, retry_policy=None)` (~300 LOC total). The client handles common logic: retry with exponential backoff, connection pooling, health checks. Provider-specific overrides go into config, not subclasses.

All 6 providers remain in the factory — the goal is to clean up the HTTP client abstraction, not delete providers.

### 5c. Unify CircuitBreakers (3 → 1)
Keep `src/shared/resilience/circuit_breaker.py` (the thread-safe one). Delete `connectors/circuit_breaker.py` and `providers/embeddings/jina.py`'s inline `CircuitBreaker`. Wire `connectors` and `jina.py` to import from `shared/resilience`.

### 5d. Audit provider cleanup (keep all 6 providers, clean up factory/adapter layer)
The provider architecture is **intentional modular design** (see `reports/baseline/provider_architecture_analysis.md`), not redundancy. All 6 providers serve distinct deployment scenarios:
- BGE-M3 (default, dense+sparse+ColBERT)
- Qwen3-Embedding-0.6B (dense, active runtime profile)
- Snowflake Arctic (dense, local service)
- Jina (cloud API)
- Voyage (cloud API)
- SentenceTransformers (local offline)

**Actions:**
1. **Remove unused aliases** in `factory.py` — keep aliases that are actively used in config files or documentation
2. **Document profile-to-provider mapping** — which profile activates which provider, what capabilities each has
3. **Clean up adapter layer** (Phase 5a) and HTTP client layer (Phase 5b)
4. **DO NOT delete providers** — they're features for experimentation and deployment flexibility

### 5e. Reduce rerankers (3 → 2)
Keep: Local (Qwen3-Reranker-4B) as default, Noop for tests.
Delete: JinaRerankProvider (the remote one).

### Exit criteria
- 3 Chonkie adapters → 1 file. 3 HTTP clients → 1. 3 CircuitBreakers → 1.
- `ls src/providers/embeddings/` shows: `base.py`, `contracts.py`, `bge_m3.py`, `jina.py`, `service.py`, `__init__.py`. Six files, not 12.
- `ls src/providers/rerank/`: `base.py`, `local.py`, `noop.py`, `__init__.py`. Four files, not 5.
- `EMBEDDINGS_PROVIDER` env var: only two valid values.
- Smoke test (ingest + query) passes against both providers.

### LOC impact
−1,500 to −2,000 (estimated).

---

## 8. Phase 6 — Feature-Flag and Env-Var Reduction

**Goal:** ≤ 20 env vars, ≤ 6 feature flags, documented in a single place.

### Actions
1. **Delete `config/feature_flags.json` (orphaned).**
2. **Collapse env-var duplication.** Current state has multiple env var names pointing to the same setting (e.g. `EMBEDDINGS_PROVIDER` vs `EMBEDDINGS_PROFILE`, various `BGE_M3_*` vs `CHONKIE_*` aliases). Pick one canonical name per setting, alias the rest in a documented deprecation list, log a warning on use of deprecated names.
3. **Move all feature flags to `config/development.yaml`** under a single `feature_flags:` section. The legacy inference path that reads 18+ individual bools stays only until Phase 7 removes those flags.
4. **Retire flags whose conditions are permanently on.** Grep each flag; if it's set `true` in every environment and no path reads it as `false`, delete the flag and the off-path code.
5. **Retire flags whose conditions are permanently off.** Same treatment.
6. **Document the remaining ≤ 6 flags with a one-line rationale each.**

### Exit criteria
- `grep -rnE "os\.environ\[?\w+" src/ | sort -u` shows ≤ 20 distinct env-var names in non-test code.
- `config/feature_flags.json` deleted.
- One YAML file has ≤ 6 feature flags.
- No code reads an env var by more than one name.

### LOC impact
Small (config simplification) but huge in cognitive load.

---

## 9. Phase 7 — Parser Unification

**Goal:** One markdown parser.

### Actions
1. Keep `parsers/markdown_it_parser.py` (default, AST-based).
2. Delete `parsers/markdown.py` (legacy).
3. Delete `parsers/shadow_comparison.py` (migration is long complete).
4. Simplify `parsers/__init__.py` to a direct import — eliminate the router and the `engine` config option.
5. Delete `html.py` if the audit confirms it's unused by production ingestion (verify with `grep`). If it's used, keep it but document its consumer.

### Exit criteria
- `config.ingestion.parser.engine` is no longer a recognized config key.
- `config.ingestion.parser.shadow_mode` and `.fail_on_mismatch` are gone.
- Smoke test on a markdown doc still produces the expected Section/Chunk graph.

### LOC impact
~−850 (389 + 257 + router simplification).

---

## 10. Phase 8 — Retrieval Unification

**Goal:** One chunk result type, one retrieval pipeline, no dual-type confusion.

### Actions (after Phase 3 has deleted `hybrid_search.py`)
1. `ChunkResult` is the canonical type. Retire any lingering `SearchResult`, `RankedResult`-as-old-shape usage.
2. `RankingFeatures` is kept (it's on the live output path). `Ranker` class is deleted (it was only used by the legacy engine).
3. Signal pool (`signal_pool.py`) is a real concern, not an opt-in experiment. If it measurably improves quality (use the eval harness), make it always-on; otherwise delete.
4. 4 retrieval profiles (VECTOR_ONLY, PRECISION_VECTOR, GRAPH_ASSISTED, GRAPH_FULL) → 2 profiles (`basic`, `graph`). Delete the legacy bool-inference path.
5. Delete `fusion_pipeline.py`'s dead `weighted_fusion()` once its callers go through RRF only.

### Exit criteria
- One `ChunkResult` type, imported everywhere retrieval happens.
- The eval-harness run on the golden query set has equal or better MRR@10 vs. baseline.

### LOC impact
−1,000 to −1,500.

---

## 11. Phase 9 — Test Suite Hygiene

**Goal:** A clean, green, measured test suite that enforces quality.

### Actions
1. **Re-organize by module.** Current tests are 50% `p1_*` through `p6_*` (phase artifacts). Move into `tests/unit/<module>/`, `tests/integration/`, `tests/smoke/`. Delete the phase-prefix tests that have been restructured, keep ones that are unique.
2. **Delete or rewrite tests for dead modules.** Phase 4 (learning/registry/ops) had dedicated tests that now fail to import — delete them.
3. **Add hard coverage gate.** `pytest-cov --cov=src --cov-fail-under=70` in CI. The number is a floor, not a target — bump once coverage improves.
4. **Add real smoke tests.** `tests/smoke/test_ingest_and_query.py`: ingest one real doc, query it, assert ≥1 chunk returned with correct section.
5. **Remove e2e test artifacts from the repo.** `tests/e2e_v22_prod/artifacts/` is huge log noise — move to Git LFS or delete.
6. **Delete `tmpfile`, `Archive.zip`, `Archive 2.zip`.** These have no business in the repo.

### Exit criteria
- `pytest` green at ≥ X% of Phase 0 count minus (deleted tests that were genuinely testing dead code).
- Coverage ≥ 70%.
- Smoke tests green.
- CI passes on push.

---

## 12. Phase 10 — MCP Server Cleanup

**Goal:** One transport for LLM agents, one for humans (CLI).

### Actions
1. **Delete the legacy REST endpoints** in `main.py`: `mcp_initialize`, `mcp_tools_list`, `mcp_tools_call`. Delete the gate `MCP_HTTP_LEGACY_REST_ENABLED`.
2. **Delete `src/mcp_server/models.py`** (only used by those legacy endpoints).
3. **Delete the 16 underscore aliases** (already planned in Phase 4b).
4. **Delete `search_documentation`** (legacy v1 tool).
5. **Drop the `analyst` and `full` tool profiles.** Single `production` profile with 16 canonical tools. (If operator debugging needs all-16 unconditionally, add a one-time env override `MCP_DISABLE_TOOL_FILTER=1`.)
6. Keep stdio transport for Claude Desktop direct connect.

### Exit criteria
- `main.py` has 2 transports: HTTP Streamable at `/_mcp` and STDIO.
- `GET /mcp/tools/list` returns 404.
- Smoke test passes via the Streamable transport.

### LOC impact
~−300 in main.py + ~−80 in models.py + aliases.

---

## 13. Phase 11 — Observability Hardening

**Goal:** One CircuitBreaker (Phase 5c), one metrics module, traceable retrieval.

### Actions
1. `shared/observability/metrics.py` (553 LOC) with 50+ metric definitions — prune to metrics that are actually emitted (search the codebase for each one; delete orphans).
2. Verify every circuit breaker path has a half-open recovery test.
3. Verify every retrieval trace (`retrieval_trace.py`) includes the full pipeline (already claimed 10 stages; verify).
4. Consolidate `_safe_parse_int` / `_safe_parse_float` helpers that appear duplicated in `circuit_breaker.py` and `retrieval_diagnostics.py`.

### Exit criteria
- No metric defined that isn't emitted.
- Trace output for a single query is a complete, human-readable narrative.

---

## 14. Phase 12 — UX Polish

**Goal:** One CLI, one status page, one way to do everything from a terminal.

### Actions
1. Build `wekadocs` CLI (new package `src/cli/`) unifying:
   - `ingest <path>` — wraps `src/ingestion/auto/cli.py`
   - `status` — wraps `tests/../db-check.py`
   - `query "..."` — wraps `tools/fusion_ab.py`-style query
   - `cache invalidate <doc_id>` — wraps `tools/redis_epoch_bump.py`
   - `inventory` — wraps `inventory_neo4j.py` + `inventory_qdrant.py`
   - `schema bootstrap` — wraps `bootstrap_schema.py`
2. Delete the 8 scattered root-level scripts (keep git history).
3. Add `wekadocs --version`, `wekadocs doctor` (validates config + connectivity).

### Exit criteria
- `wekadocs --help` lists all 6 subcommands.
- Each subcommand is covered by a CLI smoke test.
- All 8 root scripts deleted.

---

## 15. Phase 13 — Final Gate

**Goal:** Confirm the cleanup is real.

### Actions
1. Re-run Phase 0 measurement against the cleaned repo. Compare:
   - LOC: should be ≤ 40,000
   - Test pass rate: ≥ Phase 0
   - `ruff`/`mypy` issues: ≤ Phase 0
2. Re-ingest a fresh doc set into empty stores, re-run the eval harness. MRR@10 ≥ baseline.
3. Merge the feature branch (`multi-embedder-reranker`) with the cleaned master. The Qwen3 stack lives on master now.
4. Delete `REFACTOR-PLAN.md` (this plan supersedes it).

### Exit criteria
- Every number in section 0 ("Goal" table) is green.
- Branch `multi-embedder-reranker` closed (merged or dropped).
- The repo root contains: `REPO-MAP.md`, `ARCHITECTURE.md`, `DEV-HISTORY.md`, `AUDIT-VERIFICATION.md`, `CLEANUP-PLAN.md`. Everything else (`DEAD-CODE-MAP.md`, `REFACTOR-PLAN.md`, the scattered `context-*.md`, `repo_files_chunk_*`) is archived to `claude-raw/` or removed.

---

## 16. Iteration Discipline

After each phase, run the full measurement:

```
pytest -q --cov=src --cov-report=term-missing \
       --cov-fail-under=<<previous>> \
       --junitxml=reports/phase-N/junit.xml
ruff check src/
mypy src/ --ignore-missing-imports
./scripts/smoke.sh           # ingest + query
```

Any regression blocks the next phase. **Phases may overlap only if they touch disjoint files**; never start a monolith refactor while a deletion PR is open against it.

---

## 17. Risk Register

| Phase | Risk | Mitigation |
|---|---|---|
| 1 (dead deletion) | Accidentally deleting an active module | Grep-verify before `rm`; commit per-module so a single revert is cheap |
| 3 (hybrid_search de-couple) | Subtle behavior change in query path | Freeze eval-harness baseline first; A/B the change |
| 4 (monolith split) | Import churn across many files | Split one god module per PR; each PR green alone |
| 5 (provider drop) | Some deployment uses Voyage/Arctic/Qwen3 | Grep `.env.*` and `docker-compose*` for provider references before removing |
| 6 (env-var removal) | Operator docs reference deprecated names | Deprecation-warning + README section |
| 12 (unified CLI) | New CLI may miss a feature of the old scripts | Mirror each old script's `--help` 1:1 |

---

## 18. What Is Explicitly Out of Scope

- Re-designing the graph schema (Neo4j relationship types). 22 types is a lot but not the mess we're fixing.
- Replacing Neo4j or Qdrant with different stores.
- Multi-tenancy or per-customer isolation — the current single-tenant setup is sufficient.
- LLM-based retrieval planning (Phase 4 `QueryPlanner` is already live — keeping it, not replacing it).
- Merging the BGE-M3 and Qwen3 embedding stacks — picking one is an architecture choice, not a cleanup.

---

## 19. Deliverables

After all phases:
- `CLEANUP-PLAN.md` (this file): kept as operational history.
- `ARCHITECTURE.md`: updated to reflect the post-cleanup architecture.
- `REPO-MAP.md`: regenerated from fresh source scan.
- `DEAD-CODE-MAP.md`: deleted (all dead code is gone).
- `DEV-HISTORY.md`: kept as-is (historical truth doesn't change).
- `AUDIT-VERIFICATION.md`: kept as-is (lesson learned about trusting subagents).
- `REFACTOR-PLAN.md`: deleted (superseded).
- All `context-*.md`, `repo_files_chunk_*`, `ARCHIVE*.zip`, `*.env.local`, `tmpfile` moved to `claude-raw/` (archived, not committed to master).

---

## 20. Ready-to-Execute Order

```
Phase 0 (baseline)            ← block on green; no destructive work without this
  └┬─ Phase 1 (full-file dead)
   ├─ Phase 2 (dead methods)
   ├─ Phase 3 (hybrid_search de-couple)
   └─ Phase 7 (parser unification)     ← independent, parallel-safe
      ↓
   Phase 4a (atomic.py split)          ← big; do alone
   Phase 4b (mcp_app.py split)         ← big; do alone
   Phase 4c-f (remaining monoliths)    ← parallel-friendly
      ↓
   Phase 5 (providers)                 ← parallel-safe with Phase 6
   Phase 6 (env vars / flags)          ← parallel-safe with Phase 5
      ↓
   Phase 8 (retrieval unification)     ← after 3, 5, 6
   Phase 10 (MCP cleanup)              ← independent
   Phase 11 (obs hardening)            ← independent
      ↓
   Phase 9 (tests)                     ← after all structural phases
   Phase 12 (unified CLI)              ← independent
      ↓
   Phase 13 (final gate)               ← merge + measurement
```

Each phase is a distinct PR, reviewed, green, merged. None is optional. None is bundled.

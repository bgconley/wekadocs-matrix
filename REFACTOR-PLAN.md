# WekaDocs Matrix — Streamlining & Refactoring Plan

> ⚠️ **SUPERSEDED (2026-06-03).** This document is deprecated in favor of `CLEANUP-PLAN.md`,
> which is grounded in verified LOC numbers, separates full-file dead from in-file dead,
> explicitly excludes planner.py and hybrid_search.py from deletion, and adds iteration
> discipline + exit criteria per phase. Keep this file only as historical reference;
> do not execute against it.

> **Current state (VERIFIED):** 68,551 LOC in `src/` (169 files), of which ~9,038 is full-file dead
> code + ~4,600 dead methods inside mixed files. 85+ env var knobs, ~3 mutually exclusive ways to do almost everything.
> **Target state:** ~40,000 clean LOC, ~20 env vars, 1 way to do each thing, fully tested, great UX.
>
> ⚠️ planner.py is NOT a deletion target (it is active). hybrid_search.py requires de-coupling before removal.
> See AUDIT-VERIFICATION.md.

---

## Phase 0 — Dead Code Removal (1-2 days, zero risk)

**The simplest, highest-value cleanup.** Delete files that are provably dead (0 runtime imports).

### Remove entire packages (DELETE directories)

| Package | LOC | Reason |
|---------|-----|--------|
| `src/learning/` | 983 | Orphaned Phase 4. Tagged `@safe-to-delete: Yes`. Zero runtime imports. |
| `src/registry/` | 282 | Orphaned Phase 7C index registry. Zero runtime imports. |
| `src/ops/optimizer.py` | 499 | Phase 4 query optimizer, never invoked. Test-only. |
| `src/ops/warmers/` | 162 | Cache warming, never invoked. Test-only. |

### Remove dead files (DELETE)

| File | LOC | Reason |
|------|-----|--------|
| `src/ingestion/api.py` | 35 | Dead test wrapper. Already superseded. |
| `src/ingestion/reconcile.py` | 525 | Qdrant-Graph reconciliation. Never called in production. |
| `src/ingestion/incremental.py` | 363 | Incremental update. Never instantiated. |
| `src/ingestion/parsers/notion.py` | 256 | Notion format not supported. |
| `src/ingestion/auto/watcher.py` | 54 | Deprecated. Zero imports anywhere. |
| `src/ingestion/auto/orchestrator.py` | 1,134 | Superseded by worker.py + atomic.py. |
| `src/ingestion/auto/verification.py` | 277 | Only used by dead orchestrator. |
| `src/ingestion/auto/report.py` | 296 | Only used by dead orchestrator. |
| `src/ingestion/auto/backpressure.py` | 282 | Never instantiated. |
| `src/query/diffusion_reranker.py` | 362 | Tagged `@status: DEAD`. |
| `src/query/graph_features.py` | 339 | Tagged `@status: DEAD`. Only imported by dead diffuser. |
| `src/query/graph_expansion.py` | 302 | Tagged `@status: DEAD`. Superseded. |
| ~~`src/query/planner.py`~~ | 358 | ❌ **NOT DEAD (verified).** Imported & called by `query_service.py:827` via `planner.plan()`. Keep. |
| `src/query/templates/advanced/schemas.py` | 150 | Tagged `@status: TEST_ONLY`. |
| `src/neo/structural_builder.py` | 520 | Superseded by `ingestion/structural_edges.py`. |
| `src/neo/graph_enhancements.py` | 440 | Zero external imports. |
| `src/neo/entity_normalization.py` | 274 | Zero external imports. |
| `src/neo/explain_guard.py` | 276 | Tagged DEAD. Use server-side Cypher safety instead. |
| `src/neo/defensive_query.py` | 112 | Never called by active code. |
| `src/neo/health.py` | 117 | Superseded by `src/monitoring/health.py`. |
| `src/mcp_server/validation.py` | 407 | CypherValidator, never wired. |
| `src/mcp_server/security/` | 355 | JWTAuth + RateLimiter, never wired to any endpoint. |
| `src/shared/feature_flags.py` | 173 | Superseded by `config.py` + `development.yaml`. |
| `src/shared/audit/` | 207 | AuditLogger, never called. |

**Total full-file removal: 9,038 LOC (measured, excludes planner.py which is active).**

### Strip dead methods from mixed-status files

| File | Dead LOC | What to remove |
|------|----------|----------------|
| `src/ingestion/build_graph.py` | ~2,900 | All write methods (upsert_document + 35 helpers). Keep: `__init__`, `ensure_embedder`, 2 text builders. |
| `src/ingestion/saga.py` | ~380 | `SagaCoordinator`, `IngestionSagaBuilder`, `SagaStep`, `SagaStepResult`. Keep: `SagaContext`, `IngestionValidator`, `ValidationResult`. |
| `src/neo/contract_checks.py` | ~425 | 6 dead check methods. Keep: `__init__` + `find_documents_needing_repair`. |

**Extra removal: ~3,705 LOC of dead methods** (build_graph ~2,900 + saga ~380 + contract_checks ~425).
When trimming `build_graph.py`, PRESERVE the 8 members `atomic.py` calls: `ensure_embedder`, `embedder`,
`embedding_plan`, `embedding_settings`, `colbert_dims`, `colbert_settings`, `_build_section_text_for_embedding`,
`_build_title_text_for_embedding`.

**Phase 0 total: ~12,743 LOC removed (9,038 full-file + ~3,705 dead methods). Shrinks `src/` from 68,551 → ~55,800 LOC.**
(`hybrid_search.py`'s 916 LOC is NOT in this total — it requires Phase 1 de-coupling first.)

---

## Phase 1 — Consolidate Competing Implementations (3-5 days)

### 1a. One Markdown Parser

**Current:** 2 parsers (820 LOC markdown-it-py + 389 LOC legacy) + router (246 LOC) + shadow comparison (257 LOC) = 1,712 LOC

**Target:** 1 parser. Rip out:
- `parsers/markdown.py` (legacy) — keep only for reference
- `parsers/shadow_comparison.py` — migration complete
- Router complexity in `parsers/__init__.py` — strip to single import

**Savings: ~900 LOC, 2 feature flags (parser.engine, parser.shadow_mode)**

### 1b. One Ingestion Pipeline

**Current:** 3 ingestion implementations
- `AtomicIngestionCoordinator` in `atomic.py` (4,052 LOC) — ACTIVE
- `GraphBuilder` in `build_graph.py` (~300 ACTIVE LOC, 2,900 DEAD) — settings container
- `Orchestrator` in `auto/orchestrator.py` — DELETE in Phase 0

**Target:** Refactor `atomic.py` (4,052 LOC) into a modular pipeline with composition:
```
atomic.py → split into:
  src/ingestion/pipeline/
    coordinator.py      (top-level orchestration)
    parser.py           (parse → sections)
    chunker.py          (assemble chunks)
    entity_extractor.py (extract entities)
    embedder.py         (compute embeddings)
    neo4j_writer.py     (graph writes)
    qdrant_writer.py    (vector writes)
    saga.py             (transaction coordination)
```

**This eliminates the 4,052-line monolith.** `atomic.py` becomes a thin facade.

### 1c. One Retrieval Engine

**Current:** 2 retrieval engines
- `HybridRetriever` (hybrid_retrieval.py, 2,095 LOC) — ACTIVE
- `HybridSearchEngine` (hybrid_search.py, 916 LOC) — LEGACY but RUNTIME-REACHABLE: imported top-level by `query_service.py:25`, instantiated at line 246, with a live path at line 1035. NOT a simple delete.

**Target (ordered — hybrid_search.py has THREE live consumers):**
1. Remove the legacy `HybridSearchEngine` path from `query_service.py` (top-level import line 25, instantiation line 246, lazy `Neo4jVectorStore` import line 233, ordering path line 1035) — route everything to `HybridRetriever`.
2. Update `ranking.py:23` to stop importing `SearchResult` from hybrid_search (use `ChunkResult`). Note `ranking.py`'s `RankedResult`/`RankingFeatures` ARE used by `response_builder.py` and `query_service.py`, so keep those — only the `Ranker.rank()` path is bypassed.
3. ONLY after steps 1-2, delete `hybrid_search.py`. Verify with `grep -rn hybrid_search src/` returning zero non-comment hits first.

### 1d. One Chunk Assembler (default)

**Current:** 3 chunking strategies controlled by env var
- Greedy (`chunk_assembler.py` GreedyCombinerV2, ~800 LOC)
- Semantic (`semantic_chunker.py` SemanticChunkerAssembler, 1,026 LOC) — config default
- Structured (mentioned, defined elsewhere)

**Target:** Make semantic chunking the ONLY option. Remove greedy fallback and env var gating. If a simpler chunker is needed for tests, make a tiny one.

### 1e. Unify Chonkie Adapters

**Current:** 3 near-identical Chonkie adapters (1,313 LOC total)
- `BgeM3ChonkieAdapter` (784 LOC)
- `ArcticChonkieAdapter` (279 LOC)
- `Qwen3ChonkieAdapter` (250 LOC)

**Target:** Single `ChonkieAdapter` class parameterized by provider. ~300 LOC.

### 1f. Unify HTTP Clients

**Current:** 3 near-identical HTTP client classes (554 LOC total)
- `EmbeddingClient` (134 LOC)
- `Qwen3EmbeddingClient` (171 LOC)
- `SnowflakeEmbeddingClient` (249 LOC)

**Target:** Single `EmbeddingClient` with provider-specific config. ~200 LOC.

---

## Phase 2 — Provider Consolidation (2-3 days)

### 2a. Reduce to 3 Embedding Providers

**Current:** 6 embedding providers, ~65 env vars, 18 aliases

| Provider | Status | Recommendation |
|----------|--------|----------------|
| BGE-M3 (embedding-service) | Default, production | **KEEP** — primary provider |
| Jina | Active, remote | Keep if needed; consider removing |
| Voyage | Active, remote | Remove — overlapping with BGE-M3 |
| Snowflake Arctic | Active, local | Remove — overlapping |
| Qwen3 Triton | Active, local | Remove — overlapping |
| SentenceTransformers | Active, local | Remove — overlapping |

**Recommendation:** Keep BGE-M3 + Jina (for fallback/alternate embedding). Remove 4 providers, ~30 env vars, and all their Chonkie adapters.

### 2b. Reduce to 2 Rerankers

**Current:** 3 rerankers in sequence (cross-encoder + ColBERT + signal pool) + 1 dead

| Reranker | Role | Recommendation |
|----------|------|----------------|
| Local (Qwen3-Reranker-4B) | Final scoring | **KEEP** — production |
| ColBERT MaxSim | Pre-rerank | Keep if measurably improves quality |
| Signal Pool | Diversity pre-filter | Keep if measurably improves recall |
| Jina Reranker | Remote option | Remove unless used |
| Noop | Test fallback | Keep for testing |

### 2c. Single Tokenizer Backend

**Current:** 3 tokenizer backends + 1 count-only backend, 9 env vars

Keep: HuggingFace tokenizer (primary). Remove: Jina Segmenter, VoyageTokenCounter. Tokens are tokens.

---

## Phase 3 — Configuration Simplification (2-3 days)

### Target: ~20 env vars (from 85+)

| Category | Now | Target | Rationale |
|----------|-----|--------|-----------|
| Embedding | 15+ | 3 | `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_DIM`, `EMBEDDINGS_MODEL` |
| Reranker | 8+ | 2 | `RERANKER_MODEL`, `RERANKER_BATCH_SIZE` |
| Chunking | 10+ | 2 | `CHUNK_TARGET_TOKENS`, `CHUNK_SIMILARITY_THRESHOLD` |
| Retrieval | 20+ | 4 | `RETRIEVAL_PROFILE`, `TOP_K`, `GRAPH_CHANNEL`, `EXPAND_NEIGHBORS` |
| Tokenizer | 9 | 1 | `TOKENIZER_MODEL` |
| MCP Server | 15+ | 3 | `MCP_TOOL_PROFILE`, `MCP_SCRATCH_TTL`, `LOG_LEVEL` |
| Database | 12+ | 6 | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `QDRANT_URL`, `REDIS_URL` |
| Caches | 6+ | 1 | `CACHE_ENABLED` |

### Simplify config.yaml

- Collapse scattered subsections into flat, obvious keys
- Remove unused config fields (RerankerConfig.max_pairs, RankingConfig.recency, etc.)
- Consolidate `development.yaml` and `production.yaml` overrides
- Delete `feature_flags.json` (it's already unused)

---

## Phase 4 — Neo4j Layer Simplification (1 day)

**Current:** `src/neo/` is 90% dead code (2,194 / 2,434 LOC)

**After Phase 0:** Only 3 files remain: `schema.py`, `schema_validator.py`, and ~30 active LOC from `contract_checks.py`.

**Target:** Move `schema.py` (22 relationship types) into `shared/models.py`. Move `schema_validator.py` into `monitoring/`. Delete `contract_checks.py` entirely (its one used method can migrate to `graph_service.py`).

**Result:** `src/neo/` directory disappears. ~240 LOC absorbed into existing modules.

---

## Phase 5 — Testing Overhaul (3-5 days)

### Current test structure (167 files, ~25,000 LOC)

**Problems:**
1. 40+ phase-based test files (`p1_t1_test.py` through `p6_t4_test.py`) — these are sequential development artifacts, not organized by module
2. Dead module tests exist for `learning/`, `registry/`, `ops/` — testing code that never ran
3. E2E test artifacts (logs, traces) stored in repo
4. Many tests skip/mock everything — not actually verifying behavior
5. No test coverage measurement config visible

### Target structure

```
tests/
  unit/
    test_ingestion/
    test_query/
    test_services/
    test_providers/
    test_mcp/
  integration/
    test_full_ingestion.py
    test_full_retrieval.py
    test_evidence_pipeline.py
  fixtures/
    sample_docs/
    test_configs/
```

### Actions

1. **Map existing tests to modules** — tag each test with what it covers
2. **Delete orphaned tests** — tests for deleted modules
3. **Keep good tests, rewrite weak ones** — consolidate phase tests into module-aligned unit tests
4. **Add coverage measurement** — pytest-cov with threshold gates
5. **Add smoke tests** — one doc ingest + one query per provider
6. **Clean test artifacts** — remove `/tests/e2e_v22_prod/artifacts/` log directories

---

## Phase 6 — MCP Server Streamlining (1-2 days)

After Phase 0 deletes `validation.py` and `security/`:

### Simplify `mcp_app.py` (3,834 LOC → ~2,000 LOC)

1. **Remove 16 deprecated underscore aliases** — only expose canonical names
2. **Remove legacy `search_documentation` tool** — keep `kb.search`
3. **De-duplicate response handling** — 4 helper functions with identical patterns
4. **Extract diagnostics into `retrieval_trace.py`** — it's already there, move the wiring
5. **Single tool profile: production** — remove `analyst` and `full` profiles

### Simplify `main.py` (754 LOC → ~300 LOC)

1. Remove legacy REST endpoints (`/mcp/*`) gated by `MCP_HTTP_LEGACY_REST_ENABLED`
2. Remove `models.py` (only used by legacy endpoints)
3. Single transport: HTTP Streamable (`/_mcp`)

---

## Phase 7 — UX Improvements (2-3 days)

### 7a. CLI Tooling

Replace scattered standalone scripts with a single CLI:

```bash
wekadocs ingest <file>          # Ingest a document
wekadocs query "how to..."      # Search documentation
wekadocs health                 # System health check
wekadocs inventory              # Database inventory
wekadocs cache invalidate <doc> # Cache management
```

Consolidate: `bootstrap_schema.py`, `inventory_neo4j.py`, `inventory_qdrant.py`, `db-check.py`, `tools/fusion_ab.py`, `tools/redis_epoch_bump.py`, `tools/redis_invalidation.py`, `src/ingestion/auto/cli.py`

### 7b. Configuration

```bash
wekadocs config show             # Current effective config
wekadocs config validate         # Validate config against schema
```

### 7c. Observability Dashboard

- Single Grafana dashboard showing: ingestion throughput, query latency P50/P95/P99, cache hit rates, error rates, SLO compliance
- Structured logs in JSON for easy parsing

---

## Phase 8 — Reliability & Polish (3-5 days)

### 8a. Error Handling Audit

- Every try/except in the codebase should either handle or propagate
- No bare `except:` blocks
- Circuit breakers should be consistently applied (one implementation, not two)

### 8b. Graceful Degradation

- If Neo4j is down, vector search should still work
- If Qdrant is down, graph-only retrieval should still work
- If the reranker is down, raw scores should still be returned

### 8c. Retry Policies

- Standardize retries across HTTP clients (currently 3 different implementations)
- Exponential backoff with jitter everywhere

### 8d. Concurrency Hardening

- Audit async/await for proper usage (no blocking calls in async context)
- Connection pooling review

---

## Implementation Order & Dependencies

```
Phase 0: Dead Code Removal     (0 dependencies)
    ↓
Phase 1: Consolidation         (after dead code gone, clear boundaries)
    ↓
Phase 2: Provider Cleanup       (after consolidation, fewer providers to manage)
    ↓
Phase 3: Config Simplification  (after providers reduced, fewer things to configure)
    ↓
Phase 4: Neo4j Cleanup          (minor, after Phase 1)
    ↓
Phase 5: Testing Overhaul       (after code is clean, test the final shape)
    ↓
Phase 6: MCP Streamlining       (after tests pass, refactor API surface)
    ↓
Phase 7: UX Improvements        (after code is solid, build the UX)
    ↓
Phase 8: Polish & Reliability   (after everything works, make it bulletproof)
```

---

## Final State Comparison

| Metric | Current | Target |
|--------|---------|--------|
| Total `src/` LOC | 68,551 (measured) | ~40,000 (-42%) |
| Dead code | ~20% (9,038 full-file + ~4,600 in-file, of 68,551) | 0% |
| Env var knobs | 85+ | ~20 (-76%) |
| Embedding providers | 6 | 2 (-67%) |
| Chunking strategies | 3 | 1 (-67%) |
| Retrieval engines | 2 | 1 (-50%) |
| MCP transports | 3 | 1 (-67%) |
| MCP tools exposed | 16 (+16 aliases) | 16 (no aliases) |
| Independent CLI tools | 8 | 1 unified CLI |
| Chonkie adapters | 3 | 1 (-67%) |
| HTTP clients | 3 | 1 (-67%) |
| Test files | 167 | ~80 (-52%) |
| Monolithic files >1000 LOC | 5 | 0 |
| Feature flag configs | 2 files | 1 file |
| Docker images | 4 | 3 (merge service+worker) |

---

## Risk Mitigation

**Every phase is independently shippable and testable.** After each phase:
1. All existing tests pass (or are updated/removed with justification)
2. A single doc can be ingested end-to-end
3. A single query returns results

**Rollback:** Each phase produces a mergeable PR. If Phase 2 breaks something, Phase 0/1 are not affected.

**Prioritization:** Phase 0 is the highest value/zero risk step. Do it first, ship it, gain confidence.

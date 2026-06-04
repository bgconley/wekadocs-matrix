# WekaDocs Matrix — Development History

> Reconstructed from 36 context files (Oct 2025 – Mar 2026)
> This explains WHY the codebase looks the way it does.
>
> ⚠️ **ALL TEST COUNTS AND PASS RATES IN THIS DOCUMENT (e.g. "411 tests, 93.4%", "44/44")
> ARE HISTORICAL CLAIMS COPIED FROM THE CONTEXT FILES (Oct 2025).** They were NOT measured
> this session and may be stale. The current test suite has not been run/collected.
> See AUDIT-VERIFICATION.md, ERROR 4.

---

## Timeline Overview

```
2025-10-13  Phases 1-3: Foundation, Query Engine, Ingestion Pipeline
                 Key crisis: Async→Sync pivot, Qdrant UUID compatibility
2025-10-14  Phase 4: Advanced Query Features (templates, cache, optimizer, learning)
2025-10-15  Phase 5: Integration & Deployment (connectors, monitoring, testing)
2025-10-16  Phase 6: Auto-Ingestion (watchers, queue, orchestrator, CLI)
2025-10-17  THE TURNING POINT: Massive simplification (context-18)
                 Queue 604→104 lines, watchers 370→24, service 123→36
                 Orchestrator bypassed in favor of direct Phase 3 calls
2025-10-18  Post-Phase 6 stabilization: 411 tests, 93.4% pass rate
2025-10-20  Graph traversal bugs fixed, Enhanced Responses spec (Phase 7E)
2026-01-19  Multi-embedder pivot: Snowflake Arctic replaces Voyage as dense embedder
                 BGE-M3 demoted to sparse/ColBERT only. New branch created.
2026-03-07  Evidence pack redesign: retrieval profiles, signal pool, MCP modernization
                 Qwen3 stack replaces BGE-M3 entirely. Branch still unmerged.
```

---

## The Six Architectural Pivots

### Pivot 1: Async → Sync (Phase 3, Oct 13)

**What happened:** The ingestion pipeline (parsers, extractors, graph builder) was built with async Neo4j patterns. Tests used synchronous drivers. Rather than fixing tests, the team converted the ENTIRE implementation to synchronous.

**Lasting impact:**
- The entire ingestion pipeline is locked into synchronous patterns
- `CompatQdrantClient` wrapper class born to normalize multiple call signatures
- 64-character hex-to-UUID conversion heuristic for Qdrant point IDs
- Codebase can never easily go async again

### Pivot 2: The Great Simplification (Phase 6, Oct 17 — context-18)

**What happened:** After building a full-featured Phase 6 (Redis Streams queue, multi-source watchers, metrics-heavy service, state-machine orchestrator), the ingestion-worker container was discovered running 4-day-old Phase 1 stub code. During the rebuild, the team scrapped nearly everything:

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Queue | 604 lines (Streams) | 104 lines (Lists) | 83% |
| Watchers | 370 lines (S3+HTTP+spool) | 24 lines (watchdog only) | 94% |
| Service | 123 lines (metrics+backpressure) | 36 lines (health+enqueue) | 71% |
| Worker | Orchestrator-driven | Direct Phase 3 calls | — |

**Lasting impact:**
- **Two parallel watcher implementations** — `watcher.py` (24L, active) and `watchers.py` (370L, preserved for future) — the singular/plural naming confusion
- **Two queue paradigms** — module-level functions vs. legacy `JobQueue` class
- **Orchestrator abandoned** but not deleted — `auto/orchestrator.py` (1,134 LOC) still exists
- Pydantic v2 backwards-compat layer (`WekaBaseModel`, `graph_schema` alias, `populate_by_name=True`)

### Pivot 3: Redis Streams → Lists (Oct 17 — contexts 17-18)

**What happened:** Original Phase 6 specs called for Redis Streams with consumer groups. Lists were chosen during simplification for `brpoplpush` atomicity. But legacy `JobQueue` class had already created Streams — causing WRONGTYPE errors when the new List-based code tried to push to the same keys.

**Lasting impact:**
- `ingest:jobs` and `ingest:processing` keys needed manual DEL to purge Stream types
- Dual Redis hash structures (`ingest:status` + `ingest:state:{job_id}`) — created because JobQueue and Orchestrator were developed separately
- `ensure_key_types()` function quarantines wrong-typed keys
- 5 root causes discovered in a single diagnostic session

### Pivot 4: Multi-Embedder Architecture (Jan 2026)

**What happened:** Snowflake Arctic v2 was chosen to replace Voyage AI as the dense embedder. BGE-M3 was demoted to sparse + ColBERT only. A new `multi-embedder-reranker` branch was created — **still unmerged to master as of March 2026.**

**Design:** Arctic → dense (1024-D), BGE-M3 → sparse + ColBERT. Arctic doesn't support sparse or ColBERT — `NotImplementedError` raised by design.

**Lasting impact:**
- Provider factory with static dictionary of creator functions — manual registration required per provider
- `EMBEDDINGS_PROFILE` env var confusion: YAML says `qwen3_0_6b`, env says `bge_m3`
- Embedding plan was WRONG at creation time (`voyage_context_3` instead of `snowflake_arctic_v2l`), documented in `EMBEDDER_CORRECTIONS.md`
- Tokenizer cache path defaults were Docker-only (`/opt/hf-cache`), causing wrong tokenizer (Jina Segmenter) to be used in local dev
- Offline mode defaulted to `true` — another tokenizer bug source

### Pivot 5: Qwen3 Stack Migration (Mar 2026)

**What happened:** The evidence pack redesign session (March 7, 2026) moved the entire embedding stack to Qwen3:
- Qwen3-Embedding-0.6B (dense)
- SPLADEv3 (sparse)
- ColBERTv2 (ColBERT)
- Qwen3-Reranker-4B (cross-encoder)
- GLiNER medium (NER)
- Qwen2.5-1.5B-Instruct (query reformulation)

**Lasting impact:**
- Full re-ingestion required — SPLADEv3 vocabulary incompatible with BGE-M3
- `chunks_multi_qwen3_0_6b` Qdrant collection created with 0 points
- Signal pool (`signal_diverse_rerank_pool`) fully implemented but requires 3 feature flags set simultaneously to activate
- 4 retrieval profiles replace 18+ individual boolean flags — but legacy flag inference preserved
- Dual MCP transports: streamable HTTP (default ON) + legacy REST (default OFF, frozen Phase 1 stubs)

### Pivot 6: Evidence Pack Redesign (Mar 2026)

**What happened:** Shift from LLM-driven multi-step retrieval planning to single `kb.retrieve_evidence` call with deep retrieval (60-150 candidates) + cross-encoder + always-on graph enrichment. 24-tool surface → 17 unique → 3 in production.

**Lasting impact:**
- `mcp_app.py` at 3,834 LOC — all 16 canonical tools + 16 deprecated aliases in one file
- `scratch_store.py` uses single global session ID because MCP SDK session objects are unreliable as dict keys
- `structlog` vs `stdlib logging` incompatibility — kwargs vs f-strings
- 7 pre-existing test failures in `tests/query/` (require live infrastructure)
- Codebase pruning (25 dead files, ~3,800 dead methods) was PLANNED but never executed

---

## What Was Planned But Never Built

| Feature | Where Spec'd | Status |
|---------|-------------|--------|
| Phase 7 Jina providers for embeddings/reranker | `phase-7-integration-plan.md` | Never implemented |
| DocRAG v3 lean schema bridge | Canonical v3 spec | Never implemented |
| EXPLAIN-plan guard for Cypher | Phase 4 spec | Deferred |
| Frontier-gated graph traversal (MMR-ish) | Pseudocode reference | Deferred |
| `related_sections` in response builder | TODO at `response_builder.py:302` | Never prioritized |
| `compare_systems`, `troubleshoot_error`, `explain_architecture` tools | `spec.md` | Never implemented |
| Signal pool activation | `config.feature_flags` | Implemented but gated OFF |
| Notion/Confluence connectors | `webhooks.py` | Stubs returning "not_implemented" |
| S3Watcher | `auto/watchers.py` | Stub raising NotImplementedError |
| CLI watch mode | `auto/cli.py` | Returns "not implemented" |
| Phase 4 learning loop | `src/learning/` | 983 LOC built, never integrated |
| Phase 7C index registry | `src/registry/` | 282 LOC built, never integrated |
| Query optimizer | `src/ops/optimizer.py` | 499 LOC built, never invoked |
| Cache warmer | `src/ops/warmers/` | 162 LOC built, never invoked |

---

## The Simplification Pattern

The development history reveals a recurring pattern:

```
1. Spec a complex feature (Streams, S3 watchers, metrics service, orchestrator)
2. Build it over multiple sessions (500+ lines, dozens of tests)
3. Discover infrastructure incompatibilities or bugs
4. SCRAP the complex version and replace with minimal implementation
5. Keep the complex version's files in the repo (never deleted)
```

This is why the codebase has:
- 3 ingestion pipelines (atomic.py vs. build_graph.py vs. orchestrator.py)
- 2 retrieval engines (HybridRetriever vs. HybridSearchEngine)
- 2 markdown parsers (markdown-it-py vs. legacy markdown)
- 2 watcher implementations (watcher.py vs. watchers.py)
- 2 queue paradigms (List functions vs. Streams JobQueue class)
- 2 saga implementations (inline vs. SagaCoordinator class)

**Each simplification was correct.** The problem is that the old code was never removed.

---

## The Branch Problem

The most critical insight from the context files: **the multi-embedder-reranker branch has been unmerged since January 2026.** This means:
- Master branch is running the OLD BGE-M3-only stack
- The branch has Arctic v2, Qwen3, evidence pack redesign, MCP modernization — all unshipped
- The branch has 25 dead files + 3,800 dead methods that were "planned for pruning" but never removed
- The `EMBEDDINGS_PROFILE` env var mismatch between YAML and .env files
- Full re-ingestion required to switch to Qwen3

---

## Key Technical Debt Patterns

1. **Pydantic v2 migration tax:** `model_name` → `embedding_model` aliasing, `schema` → `graph_schema` renaming, `WekaBaseModel` with `protected_namespaces=()`, `populate_by_name=True` — 3 separate sessions spent on this
2. **Neo4j limitations:** No parameters in variable-length patterns → hard-coded depths. No nested maps → JSON serialization. Dual driver format handling.
3. **Docker/volume masking:** `./hf-cache` bind-mount replaces container's `/opt/hf-cache`, making Dockerfile's tokenizer prefetch useless
4. **structlog vs stdlib:** Mixed logging styles (kwargs vs f-strings) — Docker routes to stdlib, kwargs-style calls raise TypeError
5. **NO-MOCKS testing philosophy:** Every test against live Docker. Tests break when Docker isn't running, when APIs change, or when driver versions shift.

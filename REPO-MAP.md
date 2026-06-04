# WekaDocs Matrix — Complete Repository Map

> Generated: 2026-06-03 | Updated: 2026-06-03 (post-verification)
> Repository: `wekadocs-matrix` — GraphRAG pipeline for Weka documentation
>
> ✅ **VERIFIED:** All LOC figures below confirmed via `wc -l` against actual source.
> See AUDIT-VERIFICATION.md for the verification method and corrections.

---

## Quick Stats (VERIFIED)

`src/` total: **68,551 LOC across 169 Python files** (excludes `__pycache__`).

| Category | Count | LOC |
|----------|-------|-----|
| **`src/` Python (total)** | 169 files | **68,551** (measured) |
| ↳ Active production | ~140 files | **~55,000** |
| ↳ Confirmed dead (full files) | 26 files | **9,038** (measured) |
| ↳ Dead methods inside mixed files | — | **~4,600** (build_graph ~2,900, saga ~380, contract_checks ~425, hybrid_search ~916*) |
| **Standalone tools** (`tools/`, root `*.py`) | ~7 files | ~2,500 |
| **Test files** | 167 files | ~25,000+ |
| **Config/YAML/JSON** | 25+ files | ~1,500 |
| **Docker** | 4 Dockerfiles | — |
| **Terraform / K8s** | 30+ files | — |
| **Markdown docs/plans** | 50+ files | — |

\* `hybrid_search.py` is dead-weight but NOT independently deletable — it is imported and
instantiated at runtime by the active `query_service.py`. Counts as dead only after de-coupling.

### Per-module LOC (measured)

| Module | Files | LOC |
|--------|-------|-----|
| ingestion | 37 | 20,782 |
| query | 29 | 14,425 |
| mcp_server | 12 | 7,619 |
| providers | 23 | 7,611 |
| shared | 23 | 7,360 |
| services | 8 | 2,879 |
| neo | 10 | 2,434 |
| monitoring | 4 | 1,562 |
| connectors | 6 | 1,147 |
| learning | 4 | 983 |
| ops | 4 | 788 |
| clients | 4 | 558 |
| registry | 2 | 282 |
| tools | 2 | 116 |
| (src root) | 1 | 5 |
| **TOTAL** | **169** | **68,551** |


---

## Complete File Inventory: `src/`

### `src/ingestion/` — Document Ingestion Pipeline (37 files, ~20,000 LOC)

| # | File | LOC | Status | Purpose |
|---|------|-----|--------|---------|
| 1 | `__init__.py` | 6 | ACTIVE | Package init |
| 2 | `worker.py` | 470 | **ACTIVE—ENTRY** | Main worker process (Docker CMD), Redis queue loop |
| 3 | `atomic.py` | 4,052 | **ACTIVE—CORE** | Atomic saga-coordinated ingestion (Neo4j+Qdrant). Largest file. |
| 4 | `build_graph.py` | 3,200 | **MIXED** | 2,900 LOC dead, 300 LOC active (settings container only) |
| 5 | `saga.py` | 689 | **MIXED** | ~300 LOC active (validation), ~380 LOC dead (SagaCoordinator) |
| 6 | `semantic_chunker.py` | 1,026 | ACTIVE | Semantic/Chonkie chunking via BGE-M3 |
| 7 | `chunk_assembler.py` | 1,052 | ACTIVE | Chunk assembler factory (greedy/semantic/structured/pipeline) |
| 8 | `structural_edges.py` | 421 | ACTIVE | Transaction-aware Neo4j edge building |
| 9 | `run_stats.py` | 261 | ACTIVE | Batch stats accumulator |
| 10 | `api.py` | 35 | **DEAD** | Thin test wrapper; superseded by atomic.py |
| 11 | `semantic.py` | 50 | ACTIVE | Semantic enricher stub (no-op) |
| 12 | `reconcile.py` | 525 | **DEAD** | Qdrant-Graph reconciliation |
| 13 | `incremental.py` | 363 | **DEAD** | Incremental update |
| 14-17 | `parsers/__init__.py` | 246 | ACTIVE | Parser routing (legacy vs. markdown-it-py) |
| 15 | `parsers/markdown.py` | 389 | DORMANT | Legacy markdown parser |
| 16 | `parsers/markdown_it_parser.py` | 820 | ACTIVE | AST-based parser (default) |
| 17 | `parsers/html.py` | 273 | ACTIVE | HTML parser |
| 18 | `parsers/notion.py` | 256 | **DEAD** | Notion parser |
| 19 | `parsers/shadow_comparison.py` | 257 | DORMANT | Parser comparison (shadow mode) |
| 20-25 | `extract/` | ~1,573 | ACTIVE | entity extractors: commands, configs, procedures, ner_gliner, references |
| 26-37 | `auto/` | ~4,590 | MIXED | Auto-ingestion: service (ACTIVE), watchers (ACTIVE), queue (ACTIVE), reaper (ACTIVE), watcher.py (DEAD), orchestrator (DEAD), verification (DEAD), report (DEAD), backpressure (DEAD), cli (STANDALONE), progress (STANDALONE) |

**Dead code in ingestion: ~4,300 LOC (21.5%)**

### `src/query/` — Retrieval & Search (29 files, ~14,425 LOC)

| # | File | LOC | Status | Purpose |
|---|------|-----|--------|---------|
| 1 | `hybrid_retrieval.py` | 2,095 | **ACTIVE—CORE** | Main `HybridRetriever` orchestrator (40+ methods) |
| 2 | `vector_backends.py` | 1,750 | ACTIVE | Multi-vector Qdrant + BM25 retrievers |
| 3 | `graph_pipeline.py` | 1,228 | ACTIVE | Neo4j graph signals (RELATED_TO, entity channel) |
| 4 | `expansion_pipeline.py` | 1,165 | ACTIVE | Bounded adjacency + structure-aware expansion |
| 5 | `response_builder.py` | 862 | ACTIVE | Markdown+JSON response assembly |
| 6 | `session_tracker.py` | 700 | ACTIVE | Neo4j session/query/answer tracking |
| 7 | `rerank_pipeline.py` | 547 | ACTIVE | Cross-encoder + ColBERT + specificity adjustment |
| 8 | `context_assembly.py` | 513 | ACTIVE | Sorted context assembly with token budget |
| 9 | `ranking.py` | 501 | ACTIVE | Legacy `Ranker` class (uses dead `SearchResult` type) |
| 10 | `structural_retrieval.py` | 452 | ACTIVE | Query-type adaptive RRF weights |
| 11 | `retrieval_plan.py` | 345 | ACTIVE | 4 named profiles replacing 18+ boolean flags |
| 12 | `traversal.py` | 333 | ACTIVE | MCP graph traversal |
| 13 | `query_intent.py` | 330 | ACTIVE | 8 intent types classifier |
| 14 | `signal_pool.py` | 303 | ACTIVE | 9-slot signal-diverse rerank pool |
| 15 | `retrieval_types.py` | 267 | ACTIVE | Data types (30+ field `ChunkResult`) |
| 16 | `fusion_pipeline.py` | 217 | ACTIVE | RRF fusion (weighted dead) |
| 17 | `processing/disambiguation.py` | 182 | ACTIVE | Query entity extraction |
| 18 | `entity_extraction.py` | 120 | ACTIVE | Trie-based Neo4j entity extraction |
| 19 | `retrieval_observability.py` | 31 | ACTIVE | Stage snapshot logging |
| 20-22 | `templates/` | ~193 | ACTIVE | Guardrails + schemas |
| 23 | `hybrid_search.py` | 916 | **LEGACY (live)** | Old `HybridSearchEngine` — imported & instantiated by active `query_service.py:246`; runtime path at line 1035. NOT independently deletable. |
| 24 | `planner.py` | 358 | 🟢 **ACTIVE** | `QueryPlanner` — imported by `query_service.py:26`, called at line 827 (`planner.plan()`) for intent classification. (Earlier "UNUSED" label was WRONG — see AUDIT-VERIFICATION.md) |
| 25 | `graph_features.py` | 339 | **DEAD** | Only imported by dead diffuser |
| 26 | `diffusion_reranker.py` | 362 | **DEAD** | PPR graph diffusion reranker |
| 27 | `graph_expansion.py` | 302 | **DEAD** | Superseded by expansion_pipeline.py |

**Dead code in query:** ~1,003 LOC confirmed full-file dead (diffusion_reranker 362 + graph_features 339 + graph_expansion 302). `hybrid_search.py` (916) is dead-weight but live-imported — only counts once de-coupled. `planner.py` is NOT dead (corrected).

### `src/providers/` — ML/AI Backend Abstraction (24 files, ~8,500 LOC)

| # | File | LOC | Status | Purpose |
|---|------|-----|--------|---------|
| 1 | `factory.py` | 610 | **ACTIVE** | Provider creation hub (6 embedding, 3 rerank, 18 aliases) |
| 2 | `settings.py` | 47 | ACTIVE | Capabilities/settings data model |
| 3 | `tokenizer_service.py` | 1,168 | ACTIVE | Dual-backend tokenizer (HF + Jina segmenter) |
| 4-15 | `embeddings/` | ~3,332 | ACTIVE | 6 providers + 3 Chonkie adapters |
| 16-18 | `ner/` | ~971 | ACTIVE | GLiNER dual-mode NER |
| 19-23 | `rerank/` | ~1,158 | ACTIVE | 3 rerankers (Jina, Local, Noop) |
| 24-27 | `clients/` | ~558 | ACTIVE | 3 HTTP clients for remote embedding services |

**Key issues:** 3 Chonkie adapters are structurally ~80% identical. **3** CircuitBreaker implementations (connectors, jina, shared/resilience). 3 near-identical HTTP client classes. ~65 env var knobs.

### `src/mcp_server/` — MCP API Server (12 files, ~7,619 LOC)

| # | File | LOC | Status | Purpose |
|---|------|-----|--------|---------|
| 1 | `mcp_app.py` | 3,834 | **ACTIVE—CORE** | ALL 16 canonical + 16 deprecated tool definitions |
| 2 | `query_service.py` | 1,157 | ACTIVE | Retrieval orchestration for MCP tools |
| 3 | `main.py` | 754 | ACTIVE | FastAPI HTTP entry (health/metrics + 2 transports) |
| 4 | `retrieval_trace.py` | 599 | ACTIVE | Pipeline observability traces |
| 5 | `scratch_store.py` | 168 | ACTIVE | In-memory passage cache |
| 6 | `models.py` | 79 | ACTIVE | Pydantic models (legacy REST only) |
| 7 | `webhooks.py` | 133 | ACTIVE | GitHub/Notion/Confluence webhooks |
| 8 | `stdio_server.py` | 133 | STANDALONE | STDIO transport for Claude Desktop |
| 9 | `validation.py` | 407 | **DEAD** | Cypher validator (never wired) |
| 10-12 | `security/` | 355 | **DEAD** | JWTAuth + RateLimiter (never wired) |

**Dead code in mcp_server: 762 LOC (10%)**

### `src/services/` — Application Services (8 files, ~2,879 LOC)

| # | File | LOC | Status | Purpose |
|---|------|-----|--------|---------|
| 1 | `cross_doc_linking.py` | 1,419 | ACTIVE | Cross-document RELATED_TO edge generation |
| 2 | `cross_doc_edge_model.py` | 359 | ACTIVE | No-deps data model for linking |
| 3 | `graph_service.py` | 638 | ACTIVE | Neo4j graph operations for MCP tools |
| 4 | `context_assembler.py` | 78 | ACTIVE | Graph+text context bundles |
| 5 | `context_budget_manager.py` | 137 | ACTIVE | Token budget enforcement |
| 6 | `delta_cache.py` | 89 | ACTIVE | Session dedup cache |
| 7 | `text_service.py` | 127 | ACTIVE | Text fetching for MCP tools |
| 8 | `__init__.py` | 32 | ACTIVE | Package init |

### `src/neo/` — Neo4j Schema & Utilities (9 files, ~2,434 LOC)

| # | File | LOC | Status | Purpose |
|---|------|-----|--------|---------|
| 1 | `schema.py` | 65 | **ACTIVE** | Canonical 22 relationship types |
| 2 | `schema_validator.py` | 163 | **ACTIVE** | Startup schema validation |
| 3 | `contract_checks.py` | 455 | **MIXED** | ~30 LOC active (needs_repair), rest dead |
| 4 | `explain_guard.py` | 276 | **DEAD** | Cypher EXPLAIN plan validation |
| 5 | `entity_normalization.py` | 274 | **DEAD** | Entity name normalization |
| 6 | `graph_enhancements.py` | 440 | **DEAD** | markdown-it-py graph enhancers |
| 7 | `health.py` | 117 | **DEAD** | (superseded by src/monitoring/health.py) |
| 8 | `structural_builder.py` | 520 | **DEAD** | (superseded by src/ingestion/structural_edges.py) |
| 9 | `defensive_query.py` | 112 | **DEAD** | Defensive query wrapper |

**Dead code in neo: ~2,194 LOC (90%)** — nearly the whole directory is dead

### `src/shared/` — Shared Infrastructure (23 files, ~7,360 LOC)

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `config.py` | 2,015 | **ACTIVE** | Massive Pydantic config model (all knobs) |
| `connections.py` | 543 | ACTIVE | Neo4j+Qdrant+Redis connection management |
| `qdrant_schema.py` | 410 | ACTIVE | Qdrant collection schema generation |
| `schema.py` | 462 | ACTIVE | Neo4j schema DDL (v2.1/v2.2 versions) |
| `chunk_utils.py` | 318 | ACTIVE | Chunk ID/metadata generation |
| `section_metadata.py` | 327 | ACTIVE | markdown-it-py metadata extraction |
| `embedding_fields.py` | 228 | ACTIVE | Embedding field canonicalization |
| `cache.py` | 659 | ACTIVE | L1+L2 tiered cache |
| `observability/*` | ~1,550 | ACTIVE | Logging, metrics (50+), tracing, exemplars, diagnostics |
| `resilience/circuit_breaker.py` | 331 | ACTIVE | Thread-safe circuit breaker |
| `vector_utils.py` | 65 | ACTIVE | Vector dim detection |
| `logging.py` | 14 | ACTIVE | Bridge to observability.logging |
| `models.py` | 12 | ACTIVE | Pydantic base model |
| `feature_flags.py` | 173 | **DEAD** | (superseded by config.yaml + config.py) |
| `audit/logger.py` | 207 | **DEAD** | AuditLogger (never called) |
| `__init__.py` | — | ACTIVE | Package init |

**Dead code in shared: 380 LOC (5%)**

### Other Modules

| Module | Files | LOC | Status |
|--------|-------|-----|--------|
| `src/connectors/` | 6 | ~1,149 | **ALL ACTIVE** — GitHub/webhook ingestion connectors |
| `src/monitoring/` | 4 | ~1,562 | **ALL ACTIVE** — health, metrics, SLOs |
| `src/learning/` | 4 | ~983 | **ALL DEAD** — orphaned Phase 4 feedback/learning module |
| `src/registry/` | 2 | ~282 | **ALL DEAD** — orphaned Phase 7C index registry |
| `src/ops/` | 4 | ~778 | **ALL DEAD** — optimizer, warmers, session cleanup wrapper |
| `src/tools/` | 1 | 116 | Standalone — `inspect_chunk.py` CLI |

---

## Dead Code Summary (VERIFIED — measured via `wc -l`)

Split into **full-file dead** (safe to delete the file) and **in-file dead methods** (strip methods, keep file).

| Module | Full-file dead LOC | In-file dead methods | Notes |
|--------|-------------------|----------------------|-------|
| ingestion | 3,217 | ~3,280 (build_graph ~2,900 + saga ~380) | build_graph/saga keep their active parts |
| query | 1,003 | ~916 (hybrid_search, only after de-couple) | planner.py is NOT dead (corrected) |
| neo | 1,739 | ~425 (contract_checks) | keep schema.py, schema_validator.py, 2 contract methods |
| mcp_server | 762 | 0 | validation.py + security/ |
| shared | 391 | 0 | feature_flags.py + audit/ |
| ops | 661 | 0 | optimizer + warmers (session_cleanup_job.py is a live CLI, NOT dead) |
| learning | 983 | 0 | entire package |
| registry | 282 | 0 | entire package |
| providers | 0 confirmed | (3 Chonkie adapters & 3 HTTP clients are DUPLICATIVE, not dead) | consolidation, not deletion |
| **TOTAL** | **9,038** (full-file) | **~8,900** (in-file/conditional) | |

**Headline:** ~9,038 LOC deletable immediately (whole files). Another ~4,600 LOC of dead methods
inside mixed files (build_graph, saga, contract_checks) once those files are trimmed.
`hybrid_search.py` (916) deletable only after de-coupling from `query_service.py`.
Earlier "~12,800 / 22%" rollup conflated these categories and double-counted planner.py.


---

## External Files

| Path | Purpose |
|------|---------|
| `config/development.yaml` | 565-line single-source config (dozens of feature flags) |
| `config/production.yaml` | Production overrides |
| `config/feature_flags.json` | Orphaned: 6 flags defined, but `config.py` has actual flags |
| `config/embedding_profiles.yaml` | Embedding profile definitions |
| `config/alloy/config.alloy` | Grafana Alloy agent config |
| `config/grafana/` | 8 dashboard JSON files |
| `docker/` | 4 Dockerfiles (mcp-server, ingestion-service, ingestion-worker, mxbai-reranker) |
| `infra/terraform/` | GCP VM provisioning (LGTM stack: Loki+Grafana+Tempo+Mimir) |
| `deploy/k8s/` | Kubernetes manifests (namespace, configmap, secrets, statefulsets, deployments, ingress) |
| `deploy/scripts/` | DR runbook, blue-green switch, canary rollout, backup/restore |
| `deploy/monitoring/` | Grafana dashboards + Prometheus alerts + runbook |
| `.github/workflows/ci.yml` | CI/CD: test→smoke→eval→build→deploy pipeline |
| `Makefile` | Docker compose aliases + neo4j-cypher-mcp controls |
| `requirements.txt` | 42+ direct dependencies |
| `tools/` | Standalone: `fusion_ab.py`, `redis_epoch_bump.py`, `redis_invalidation.py` |
| `data/ingest/` | ~200+ markdown files of Weka product documentation |

---

## Key Redundant/Duplicate Implementations

| Concern | Where | Impact |
|---------|-------|--------|
| **3 ingestion pipelines** | `atomic.py` (active), `build_graph.py` (90% dead), `auto/orchestrator.py` (dead) | 3 ways to do the same thing |
| **2 retrieval engines** | `hybrid_retrieval.py` (active), `hybrid_search.py` (legacy but live-imported by `query_service.py:246`) | Parallel `ChunkResult` vs `SearchResult` types |
| **2 markdown parsers** | `markdown_it_parser.py` (active), `markdown.py` (legacy, shadow mode) | Full duplicate contract |
| **3 Chonkie adapters** | `BgeM3ChonkieAdapter`, `ArcticChonkieAdapter`, `Qwen3ChonkieAdapter` | ~80% identical code |
| **3 CircuitBreaker** | `connectors/circuit_breaker.py` + `providers/embeddings/jina.py` + `shared/resilience/circuit_breaker.py` | Three separate impls, all active (verified) |
| **3 HTTP clients** | `EmbeddingClient`, `Qwen3EmbeddingClient`, `SnowflakeEmbeddingClient` | Near-identical retry+backoff |
| **2 saga implementations** | `_execute_atomic_saga()` inline + `SagaCoordinator` class (dead) | Generic was never used |
| **2 file watchers** | `auto/watchers.py` (active), `auto/watcher.py` (deprecated) | Singular vs plural |
| **6 embedding providers** | Jina, Voyage, BGE-M3, Snowflake Arctic, Qwen3, SentenceTransformers | Only 1 active at a time; ~65 env vars |
| **4 rerankers** | cross-encoder, ColBERT, signal pool (pre-rerank), dead diffusion reranker | 3 active in sequence |

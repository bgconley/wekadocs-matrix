# 09 — Evidence Index

## Important Files Inspected

### Source Code (src/)

| File | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `src/mcp_server/main.py` | Primary HTTP entrypoint | FastAPI app, StreamableHTTP mount, health checks, connector manager startup. Imports mcp_app, webhooks, connections, config. No security middleware. |
| `src/mcp_server/mcp_app.py` | MCP server factory + tool implementations | 3835 lines. `build_mcp_server()` registers all MCP tools. Imported by main.py and stdio_server.py. |
| `src/mcp_server/stdio_server.py` | STDIO transport for Claude Desktop | Standalone entrypoint. Not in Docker. Imports mcp_app.py. |
| `src/mcp_server/security/auth.py` | JWT auth middleware (DEAD) | `@status: DEAD`. Not imported by main.py. |
| `src/mcp_server/security/rate_limiter.py` | Rate limiter (DEAD) | `@status: DEAD`. Not imported by main.py. |
| `src/mcp_server/validation.py` | Validation module (DEAD) | `@status: DEAD`. Never wired in. |
| `src/mcp_server/query_service.py` | QueryService class | Integrates hybrid search, ranking, response building. Imported by mcp_app.py. |
| `src/ingestion/worker.py` | Ingestion worker entrypoint | Async Redis-polling worker. Imports atomic.py, contract_checks.py (DEAD module). |
| `src/ingestion/atomic.py` | Production ingestion path | `AtomicIngestionCoordinator`. Saga-coordinated Neo4j+Qdrant writes. Lazily imports build_graph.py (DEPRECATED). |
| `src/ingestion/build_graph.py` | Legacy GraphBuilder (DEPRECATED) | `@status: DEPRECATED`. Still imported by atomic.py line 865. |
| `src/ingestion/api.py` | Test facade (DEAD) | `@status: DEAD`. Delegates to build_graph. Not imported by any src/ module. |
| `src/ingestion/auto/orchestrator.py` | Old orchestrator (DEAD) | `@status: DEAD`. Never instantiated. Superseded by worker.py + atomic.py. |
| `src/ingestion/auto/service.py` | Auto-ingest HTTP service | FastAPI on port 9108. File watcher + enqueue + metrics. |
| `src/ingestion/auto/queue.py` | Redis job queue | `IngestJob`, `JobQueue`, `JobStatus`. Imported by worker.py, service.py, cli.py. |
| `src/ingestion/auto/reaper.py` | Stale job recovery | `JobReaper`. Imported by worker.py. |
| `src/ingestion/auto/cli.py` | ingestctl CLI (STANDALONE) | `@status: STANDALONE`. Commands: ingest, status, tail, cancel, clean, report. |
| `src/ingestion/auto/watcher.py` | Old watcher (DEAD) | `@status: DEAD`. Superseded by watchers.py (plural). |
| `src/ingestion/auto/watchers.py` | Active file watcher | `FileSystemWatcher`. Imported by service.py. |
| `src/ingestion/saga.py` | Saga implementation (DEAD) | `@status: DEAD`. Superseded by inline saga logic in atomic.py. |
| `src/ingestion/reconcile.py` | Reconciliation (DEAD) | `@status: DEAD`. No runtime callers. |
| `src/ingestion/incremental.py` | Incremental updates (DEAD) | `@status: DEAD`. No runtime callers. |
| `src/ingestion/chunk_assembler.py` | Chunk assembly | `StructuredChunker`, `SemanticChunkerAssembler`. Imported by atomic.py. |
| `src/ingestion/structural_edges.py` | Structural edge building | `build_structural_edges_in_tx()`. Imported by atomic.py. |
| `src/ingestion/parsers/markdown.py` | Legacy markdown parser | `@status: DEPRECATED`. Custom parser. |
| `src/ingestion/parsers/markdown_it_parser.py` | Active markdown parser | `@status: ACTIVE`. markdown-it-py AST parser. |
| `src/ingestion/parsers/notion.py` | Notion parser (DEAD) | `@status: DEAD`. No runtime callers. |
| `src/ingestion/extract/__init__.py` | Entity extraction router | Imported by atomic.py. Routes to commands, configs, procedures, NER. |
| `src/ingestion/extract/ner_gliner.py` | GLiNER NER enrichment | `enrich_chunks_with_entities()`. Imported by atomic.py. |
| `src/providers/factory.py` | Provider factory | `ProviderFactory`. Creates embedding/rerank providers from ENV config. |
| `src/providers/settings.py` | Embedding settings | `EmbeddingSettings`, `EmbeddingCapabilities` frozen dataclasses. |
| `src/query/hybrid_retrieval.py` | Hybrid retriever | `HybridRetriever`. Core retrieval engine. Dense + sparse + BM25 + graph. |
| `src/query/fusion_pipeline.py` | Score fusion | RRF and weighted fusion. |
| `src/query/rerank_pipeline.py` | Reranking pipeline | Cross-encoder reranking. |
| `src/query/response_builder.py` | Response assembly | `Response`, `StructuredResponse`, `build_response()`. |
| `src/query/traversal.py` | Graph traversal | `TraversalService`. Imported by main.py. |
| `src/query/graph_expansion.py` | Graph expansion (DEAD) | `@status: DEAD`. No callers. |
| `src/query/graph_features.py` | Graph features (DEAD) | `@status: DEAD`. No callers. |
| `src/query/diffusion_reranker.py` | Diffusion reranker (DEAD) | `@status: DEAD`. No callers. |
| `src/services/graph_service.py` | Graph service | `GraphService`. Imported by mcp_app.py. |
| `src/services/text_service.py` | Text service | `TextService`. Imported by mcp_app.py. |
| `src/services/context_budget_manager.py` | Budget management | `ContextBudgetManager`, `BudgetExceeded`. |
| `src/services/cross_doc_linking.py` | Cross-doc linking | `CrossDocLinker`. Imported by atomic.py. |
| `src/shared/config.py` | Central config loader | `Config`, `Settings`, `EmbeddingConfig`, `EmbeddingPlan`. Imported by 20+ modules. |
| `src/shared/connections.py` | Connection pool manager | `ConnectionManager`, `CompatQdrantClient`. Imported by 15+ modules. Contains DEAD methods. |
| `src/shared/schema.py` | Neo4j schema DDL | `create_schema()`, `create_vector_indexes()`, `verify_schema()`. |
| `src/shared/observability/metrics.py` | Prometheus metrics | `PrometheusMiddleware`, `get_metrics()`, `setup_metrics()`. |
| `src/shared/observability/tracing.py` | OpenTelemetry tracing | `setup_tracing()`, `init_tracing()`. |
| `src/shared/audit/logger.py` | Audit logger (DEAD) | `@status: DEAD`. No callers. |
| `src/shared/feature_flags.py` | Feature flags (DEAD) | `@status: DEAD`. No callers. |
| `src/neo/schema.py` | Neo4j schema constants | `RELATIONSHIP_TYPES` — canonical 24 relationship types. |
| `src/neo/contract_checks.py` | Contract checks (DEAD but imported) | `@status: DEAD` but imported by worker.py line 260. Highest-risk classification. |
| `src/neo/structural_builder.py` | Structural builder (DEAD, circular) | `@status: DEAD` but imports itself. Circular import. |
| `src/neo/entity_normalization.py` | Entity normalization (DEAD) | `@status: DEAD`. No callers. |
| `src/neo/explain_guard.py` | Explain guard (DEAD) | `@status: DEAD`. No callers. |
| `src/monitoring/health.py` | Health checker | `HealthChecker`. `run_startup_health_checks()` in main.py. |
| `src/connectors/manager.py` | Connector manager | `ConnectorManager`. Manages GitHub connector polling. |
| `src/connectors/github.py` | GitHub connector | `GitHubConnector`. Syncs GitHub repos. |
| `src/learning/__init__.py` | Learning package (DEAD) | `@status: DEAD`. Entire package unused. |
| `src/registry/__init__.py` | Registry package (DEAD) | `@status: DEAD`. No callers. |
| `src/ops/warmers/__init__.py` | Query warmers (DEAD) | `@status: DEAD`. No callers. |
| `src/tools/inspect_chunk.py` | Chunk inspection CLI | `@status: ACTIVE`. Standalone CLI tool. |

### Configuration and Infrastructure

| File | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `docker-compose.yml` | All service definitions | 497 lines. 12+ services. Defines mcp-server, ingestion-worker, ingestion-service, neo4j, qdrant, redis, mxbai-reranker. |
| `docker/mcp-server.Dockerfile` | MCP server container | CMD: `uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000` |
| `docker/ingestion-worker.Dockerfile` | Worker container | CMD: `python -m src.ingestion.worker` |
| `docker/ingestion-service.Dockerfile` | Auto-ingest container | CMD: `uvicorn src.ingestion.auto.service:app --host 0.0.0.0 --port 9108` |
| `docker/mxbai-reranker.Dockerfile` | Reranker container | CUDA 12.6, runs mxbai-rerank-large-v2 |
| `.github/workflows/ci.yml` | CI/CD pipeline | 5 jobs: test → profile-matrix-smoke → eval-harness → build → deploy-staging → deploy-production |
| `deploy/k8s/base/kustomization.yaml` | K8s deployment | 12 resources in wekadocs namespace. mcp-server-blue (3 replicas), ingestion-worker (2 replicas), neo4j, qdrant, redis |
| `deploy/k8s/overlays/staging/` | Staging overlay (EMPTY) | Directory exists but no kustomization files |
| `deploy/k8s/overlays/production/` | Production overlay (EMPTY) | Directory exists but no kustomization files |
| `config/development.yaml` | Master config | 566 lines. Embedding profiles, search config, chunk assembly, GLiNER, cache, feature flags |
| `config/production.yaml` | Production config | Identical to development.yaml (566 lines, same content) |
| `config/embedding_profiles.yaml` | Embedding profiles | Named profile definitions |
| `config/feature_flags.json` | Feature flags | Feature flag definitions |
| `config/alloy/config.alloy` | Grafana Alloy config | Docker log discovery, Loki processing, OTEL collection, Prometheus scraping |
| `.env.example` | Environment variables | MCP port 8000, Neo4j credentials, Redis password, JWT secret, GPU gateway 10.25.0.50:8080 |
| `.env.production` | Production env | Hardcoded New Relic key. Docker service names for DB hosts |
| `.env.docker` | Docker dev env | LAN IP 10.25.0.50:8080, MCP_HTTP_STREAMABLE_ENABLED=true |
| `.gitignore` | Ignored paths | .env files, __pycache__, reports/cleanup, logs, hf-cache, .serena, *.bak, agent context files |
| `.pre-commit-config.yaml` | Pre-commit hooks | black, ruff, isort, detect-secrets, gitlint |
| `requirements.txt` | Dependencies | ~80 deps: FastAPI, Neo4j, Qdrant, Redis, sentence-transformers, GLiNER, MCP SDK, OpenTelemetry |
| `Makefile` | Build commands | make up/down, make test-phase-N, Neo4j Cypher MCP management |
| `pytest.ini` | Test config | Markers: order, slow, integration, external, unit, live, xfail, chaos. Asyncio=auto. Strict markers. |
| `.coveragerc` | Coverage config | Branch coverage, source=src/, omits generated files |
| `.coveragerc.phase-3` | Phase-3 coverage | 80% threshold for ingestion+shared only |

### Scripts and Tools

| File | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `scripts/init_schema.py` | Schema initialization | Uses `src.shared.schema.create_schema()`. Active schema path. |
| `scripts/apply_complete_schema_v2_1.py` | Broken schema script | References non-existent Cypher file. |
| `scripts/reset_datastores.py` | Database reset | Clears Qdrant/Neo4j/Redis while preserving schema |
| `scripts/cleanup-databases.py` | Surgical data deletion | Multi-embedder reset with metadata preservation |
| `scripts/monitor_ingestion.py` | Ingestion monitor | Real-time monitoring of file drops, queue status, DB counts |
| `scripts/ingestctl` | ingestctl CLI | Delegates to `src.ingestion.auto.cli.main()` |
| `scripts/backfill_cross_doc_edges.py` | Cross-doc edge backfill | Creates RELATED_TO edges between similar documents |
| `scripts/evaluate_retrieval.py` | Evaluation wrapper | Wrapper around `scripts/eval/run_eval.py` |
| `scripts/eval/run_eval.py` | Evaluation harness | Loads gold YAML, runs HybridRetriever, computes recall/MRR/latency |
| `scripts/ci/check_dead_imports.py` | Dead import detection | Scans @status: annotations, detects ACTIVE/MIXED importing DEAD |
| `scripts/ci/check_phase_gate.py` | Phase gate check | Detects phase from PR, verifies reports/phase-N/junit.xml exists |
| `scripts/dev/seed_minimal_graph.py` | Test data seeding | Seeds deterministic Neo4j+Qdrant test data |
| `scripts/neo4j_structural_migration.py` | Retired migration | Has `--force-retired-script` guard. Dated 2026-03-02 |
| `tools/fusion_ab.py` | Fusion A/B testing | Compares RRF vs Weighted fusion |
| `tools/redis_epoch_bump.py` | Cache invalidation | Epoch-based cache invalidation tool |
| `tools/redis_invalidation.py` | Pattern-scan invalidation | Fallback cache invalidation |
| `bootstrap_schema.py` | Schema bootstrap | One-shot Qdrant schema bootstrap for bge_m3 |
| `db-check.py` | Database checker | Neo4j/Qdrant/Redis parity check |
| `inventory_neo4j.py` | Neo4j inventory | Dumps schema to neo4j_schema_dump.cypher |
| `inventory_qdrant.py` | Qdrant inventory | Dumps schema to qdrant_schema_inventory.json |

### Data and Schema Files

| File | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `neo4j_schema_dump.cypher` | Live Neo4j schema | 16 constraints, 4 vector indexes (1024-D), 3 fulltext indexes. SchemaVersion v2.2 |
| `neo4j_full_migration.cypher` | Migration script | Same schema + optional relationship builders (commented out) |
| `qdrant_schema_inventory.json` | Live Qdrant schema | chunks_multi_bge_m3 collection. 4 dense + 4 sparse vectors. HNSW m=48, ef_construct=256 |
| `scripts/neo4j/create_schema.cypher` | Base schema DDL | |
| `scripts/neo4j/create_schema_v2_1.cypher` | v2.1 schema DDL | |
| `scripts/neo4j/create_schema_v2_1_complete.cypher` | v2.1 complete DDL | Referenced by broken script |
| `scripts/neo4j/create_schema_v2_1_complete__v3.cypher` | v2.1 v3 variant | |
| `scripts/neo4j/create_graphrag_schema_v2_2_20251105.cypher` | v2.2 schema DDL | |
| `scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher` | v2.2 guard DDL | |
| `scripts/neo4j/create_schema_v2_2_complete__phase7E.cypher` | v2.2 Phase 7E DDL | Most recent schema |
| `scripts/neo4j/schema_backup_20251029_deployed_v2.1.cypher` | Archived v2.1 backup | |
| `scripts/neo4j/schema_backup_20251105_clean_pre_chunking_reform.cypher` | Archived pre-chunking backup | |
| `scripts/neo4j/schema_migration_phase3_markdown_it_py.cypher` | Phase 3 migration DDL | |
| `scripts/neo4j/schema_ddl_complete_20251125.py` | v2.1 complete DDL (Python) | |
| `scripts/neo4j/neo4j_schema_snapshot.py` | Neo4j snapshot tool | |
| `scripts/neo4j/qdrant_setup_chunks_multi.py` | Qdrant setup script | |
| `scripts/neo4j/fix_boundaries_headings.py` | Boundary fix script | |

### Tests

| File | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `tests/conftest.py` | Test fixtures | Session-scoped: Docker services, FastAPI TestClient, Neo4j/Redis/Qdrant clients, JWT tokens, OTel tracing |
| `tests/e2e/test_golden_set.py` | Golden set E2E | 20-query set, no mocks, runs against live Docker stack |
| `tests/e2e_v22_prod/` | v2.2 prod validation | 6 test files, spec-only, not yet executed |
| `tests/integration/` | Integration tests | 22 tests: GDS readiness, GLiNER flow, Jina batching, sparse ColBERT, session tracking |
| `tests/unit/` | Unit tests | 26 tests: tokenizer, structural retrieval, GLiNER, cross-doc linking, markdown parser |
| `tests/contracts/` | Contract tests | Cypher policy, graph v2, MCP streamable |
| `tests/p*_t*_test.py` | Phase tests | 24 files (P1-P6, T1-T4 each) |
| `tests/test_phase*.py` | Phase-specific tests | 15 files covering phase1 through phase7E |

### Documentation

| File | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `docs/architecture/2026-03-04-end-to-end-architecture.md` | Architecture docs | Schema v4.0, two pipelines (ingestion + retrieval), six AI models, VRAM breakdown |
| `docs/hybrid-rag-v2_2-spec.md` | Hybrid RAG spec | Neo4j v2.2 schema + Qdrant multi-vector architecture |
| `docs/implementation-plan.md` | Implementation plan | Phase-gated plan (Phases 1-9c), no-mocks tests required |
| `docs/configuration.md` | Config docs | Hierarchical config: YAML + env vars |
| `docs/api-contracts.md` | API contracts | Frozen schema v2025-10-20, Neo4j node labels |
| `AGENT_CONTEXT.md` | Session context | Generated 2026-01-19. Embedding architecture corrections, Phase 7E status |
| `TASK_BACKLOG.md` | Task backlog | 52 items, P0-P3. Embedding architecture corrections as P0 |
| `SESSION_CONTEXT_20260119_*.md` | Session context | Arctic Embedder integration session notes |
| `SESSION-SUMMARY.md` | Session summary | |
| `EMBEDDER_CORRECTIONS.md` | Embedder corrections | |
| `rrf-fusion-no-neo4j-20251205.md` | RRF fusion research | |
| `ARCHITECTURE.md` | (Not found at root) | Architecture docs are in docs/ subdirs |

### AI Artifacts and Generated Files (Not Source Code)

| Path | Why It Matters | Key Evidence |
|------|---------------|--------------|
| `claude-raw/` | AI session artifacts | 39 files from Claude/GPT-5 Pro sessions. Implementation plans, Cypher scripts. Not source code. |
| `context-*.md` (27 files) | Session context snapshots | Historical AI session context. Gitignored. |
| `reports/` (355+ files) | Generated artifacts | Test reports, ingestion reports, cleanup reports. Generated by CI and runtime. |
| `Archive.zip`, `Archive 2.zip` | Old code snapshots | No references in code or docs. |
| `mcp_app.py.bak` | Backup file | Gitignored. Pre-modification backup from Jan 19 session. Never imported. |
| `hf-cache/` | HuggingFace cache | Cached models. Gitignored. |
| `.npm-cache/`, `.pytest_cache/`, `.ruff_cache/`, `.uv-cache/` | Build caches | Generated by tooling. Gitignored. |
| `.serena/` | IDE state | Serena IDE state. Gitignored. |
| `qdrant_sample.json` | Sample data | Sample Qdrant data file. |
| `phase4-kickoff-fixtures.patch` | Test fixtures | Patch file for test fixtures. |

## Key Evidence Snippets

### E1: DEAD module imported by ACTIVE worker
```
# src/ingestion/worker.py:260
from src.neo.contract_checks import GraphContractChecker
# src/neo/contract_checks.py:2
# @status: DEAD
```

### E2: DEPRECATED module lazily imported by ACTIVE atomic
```
# src/ingestion/atomic.py:865
from src.ingestion.build_graph import GraphBuilder
# src/ingestion/build_graph.py:2
# @status: DEPRECATED
```

### E3: Security modules exist but not wired
```
# src/mcp_server/main.py — no import of src.mcp_server.security
# src/mcp_server/security/auth.py:2
# @status: DEAD
# src/mcp_server/security/rate_limiter.py:2
# @status: DEAD
```

### E4: Production config identical to development
```
# config/production.yaml — 566 lines, identical to config/development.yaml
# Verified by content comparison
```

### E5: Hardcoded New Relic key
```
# .env.production:NEW_RELIC_LICENSE_KEY=c71e093ba36af151d238c1e443093261FFFFNRAL
```

### E6: Docker CMD confirms entrypoints
```
# docker/mcp-server.Dockerfile:CMD ["python", "-m", "uvicorn", "src.mcp_server.main:app", ...]
# docker/ingestion-worker.Dockerfile:CMD ["python", "-m", "src.ingestion.worker"]
# docker/ingestion-service.Dockerfile:CMD ["uvicorn", "src.ingestion.auto.service:app", ...]
```

### E7: K8s overlays are empty
```
# deploy/k8s/overlays/staging/ — directory exists, no files
# deploy/k8s/overlays/production/ — directory exists, no files
```

### E8: Broken schema script
```
# scripts/apply_complete_schema_v2_1.py references scripts/neo4j/create_schema_v2_1_complete.cypher
# The file exists but the script's import path and execution context may be wrong
```

### E9: Multiple schema versions
```
# scripts/neo4j/ contains 6+ schema DDL files:
# create_schema.cypher, create_schema_v2_1.cypher, create_schema_v2_1_complete.cypher,
# create_schema_v2_1_complete__v3.cypher, create_graphrag_schema_v2_2_20251105.cypher,
# create_graphrag_schema_v2_2_20251105_guard.cypher, create_schema_v2_2_complete__phase7E.cypher
```

### E10: Status annotation system
```
# 38 files marked @status: DEAD
# 115 files marked @status: ACTIVE
# 3 files marked @status: STANDALONE
# Many files mention deprecated/superseded/legacy in comments
```

# 02 — Active vs Deprecated Map

## Classification Key

| Classification | Meaning |
|---------------|---------|
| **ACTIVE** | Currently imported and used at runtime. Part of production execution path. |
| **STANDALONE** | Executable independently but not imported by other src/ modules. Used via CLI or direct invocation. |
| **LIKELY ACTIVE** | No explicit `@status:` annotation. Not currently imported by any other module. May be a library module or future integration point. |
| **AMBIGUOUS** | Conflicting evidence: marked one way but imported another way, or imported by both active and dead code. |
| **DEPRECATED** | Explicitly marked as deprecated/superseded. Still referenced by active code (creates coupling risk). |
| **DEAD** | Explicitly marked DEAD. Not imported by any active code path. Safe to remove. |
| **GENERATED** | Auto-generated artifacts (reports, caches, backups). Not source code. |
| **ARCHIVED** | Historical snapshots, old versions, backup files. Not part of active codebase. |

## Path Classification Table

### Core Runtime (src/)

| Path | Classification | Confidence | Evidence |
|------|---------------|------------|----------|
| `src/mcp_server/main.py` | ACTIVE | High | `@status: ACTIVE -- ENTRY POINT`. Docker CMD. FastAPI app. Imports mcp_app, webhooks, connections, config. |
| `src/mcp_server/mcp_app.py` | ACTIVE | High | `@status: ACTIVE`. Imported by main.py, stdio_server.py, 4 test files. 3835 lines. MCP server factory. |
| `src/mcp_server/query_service.py` | ACTIVE | High | `@status: ACTIVE`. Imported by mcp_app.py. QueryService class integrates hybrid search, ranking, response building. |
| `src/mcp_server/models.py` | ACTIVE | High | `@status: ACTIVE`. Imported by main.py. Pydantic models for MCP protocol. |
| `src/mcp_server/scratch_store.py` | ACTIVE | High | `@status: ACTIVE`. Imported by mcp_app.py. In-memory scratch storage. |
| `src/mcp_server/retrieval_trace.py` | ACTIVE | High | `@status: ACTIVE`. Imported by mcp_app.py. Retrieval trace builder. |
| `src/mcp_server/webhooks.py` | ACTIVE | High | `@status: ACTIVE`. Imported by main.py. GitHub/Notion/Confluence webhook handlers. |
| `src/mcp_server/stdio_server.py` | STANDALONE | High | `@status: STANDALONE`. Not in Docker. Claude Desktop STDIO transport. Imports mcp_app.py. |
| `src/mcp_server/validation.py` | DEAD | High | `@status: DEAD`. Never imported. Never wired in. |
| `src/mcp_server/security/__init__.py` | DEAD | High | `@status: DEAD`. Not imported by main.py. Auth module exists but not wired. |
| `src/mcp_server/security/auth.py` | DEAD | High | `@status: DEAD`. JWT auth middleware. Not used in main.py. |
| `src/mcp_server/security/rate_limiter.py` | DEAD | High | `@status: DEAD`. Rate limiter. Not used in main.py. |
| `src/ingestion/worker.py` | ACTIVE | High | `@status: ACTIVE`. Docker CMD. Async Redis-polling worker. Imports atomic, queue, reaper, contract_checks. |
| `src/ingestion/atomic.py` | ACTIVE | High | `@status: ACTIVE`. `@called-by: worker.py`. AtomicIngestionCoordinator. Saga-coordinated Neo4j+Qdrant writes. |
| `src/ingestion/build_graph.py` | DEPRECATED | High | `@status: DEPRECATED`. Still lazily imported by atomic.py (line 865) and api.py. Legacy GraphBuilder. |
| `src/ingestion/api.py` | DEAD | High | `@status: DEAD`. Test facade. Delegates to build_graph. Not imported by any src/ module. |
| `src/ingestion/saga.py` | DEAD | High | `@status: DEAD`. Superseded by inline saga logic in atomic.py. |
| `src/ingestion/reconcile.py` | DEAD | High | `@status: DEAD`. No runtime callers. |
| `src/ingestion/incremental.py` | DEAD | High | `@status: DEAD`. No runtime callers. |
| `src/ingestion/chunk_assembler.py` | ACTIVE | High | `@status: ACTIVE`. Imported by atomic.py. StructuredChunker, SemanticChunkerAssembler. |
| `src/ingestion/structural_edges.py` | ACTIVE | High | `@status: ACTIVE`. Imported by atomic.py. build_structural_edges_in_tx(). |
| `src/ingestion/run_stats.py` | ACTIVE | High | `@status: ACTIVE`. Imported by worker.py. IngestionRunStats. |
| `src/ingestion/semantic_chunker.py` | ACTIVE | High | `@status: ACTIVE`. Imported by atomic.py. |
| `src/ingestion/semantic.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/ingestion/parsers/__init__.py` | ACTIVE | High | `@status: ACTIVE`. Imported by atomic.py. Parser router. |
| `src/ingestion/parsers/markdown.py` | DEPRECATED | High | `@status: DEPRECATED`. Legacy parser. markdown_it_parser.py is preferred. |
| `src/ingestion/parsers/markdown_it_parser.py` | ACTIVE | High | `@status: ACTIVE`. Primary markdown parser. |
| `src/ingestion/parsers/html.py` | ACTIVE | High | `@status: ACTIVE`. HTML parser. |
| `src/ingestion/parsers/notion.py` | DEAD | High | `@status: DEAD`. No runtime callers. |
| `src/ingestion/parsers/shadow_comparison.py` | DEPRECATED | High | `@status: DEPRECATED`. Comparison utility. |
| `src/ingestion/extract/__init__.py` | ACTIVE | High | `@status: ACTIVE`. Imported by atomic.py. Entity extraction router. |
| `src/ingestion/extract/commands.py` | ACTIVE | High | `@status: ACTIVE`. extract_commands(). |
| `src/ingestion/extract/configs.py` | ACTIVE | High | `@status: ACTIVE`. extract_configurations(). |
| `src/ingestion/extract/procedures.py` | ACTIVE | High | `@status: ACTIVE`. extract_procedures(). |
| `src/ingestion/extract/ner_gliner.py` | ACTIVE | High | `@status: ACTIVE`. enrich_chunks_with_entities(). |
| `src/ingestion/extract/references.py` | ACTIVE | High | `@status: ACTIVE`. extract_references(), create_reference_edge(). |
| `src/ingestion/auto/__init__.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/ingestion/auto/queue.py` | ACTIVE | High | `@status: ACTIVE`. Imported by worker.py, service.py, cli.py. Redis job queue. |
| `src/ingestion/auto/reaper.py` | ACTIVE | High | `@status: ACTIVE`. Imported by worker.py. JobReaper. |
| `src/ingestion/auto/service.py` | ACTIVE | High | `@status: ACTIVE`. Docker CMD. FastAPI on port 9108. |
| `src/ingestion/auto/watchers.py` | ACTIVE | High | `@status: ACTIVE`. FileSystemWatcher. |
| `src/ingestion/auto/cli.py` | STANDALONE | High | `@status: STANDALONE`. ingestctl CLI. |
| `src/ingestion/auto/progress.py` | STANDALONE | High | `@status: STANDALONE`. ProgressTracker/ProgressReader. |
| `src/ingestion/auto/orchestrator.py` | DEAD | High | `@status: DEAD`. Superseded by worker.py + atomic.py. Never instantiated. |
| `src/ingestion/auto/backpressure.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/ingestion/auto/report.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/ingestion/auto/verification.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/ingestion/auto/watcher.py` | DEAD | High | `@status: DEAD`. Superseded by watchers.py (plural). |
| `src/query/hybrid_retrieval.py` | ACTIVE | High | `@status: ACTIVE`. HybridRetriever. Core retrieval engine. |
| `src/query/hybrid_search.py` | ACTIVE | High | `@status: ACTIVE`. HybridSearchEngine, QdrantVectorStore. |
| `src/query/fusion_pipeline.py` | ACTIVE | High | `@status: ACTIVE`. RRF and weighted fusion. |
| `src/query/ranking.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/rerank_pipeline.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/response_builder.py` | ACTIVE | High | `@status: ACTIVE`. Response, StructuredResponse. |
| `src/query/context_assembly.py` | ACTIVE | High | `@status: ACTIVE`. ContextAssembler. |
| `src/query/traversal.py` | ACTIVE | High | `@status: ACTIVE`. TraversalService. Imported by main.py. |
| `src/query/graph_pipeline.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/graph_expansion.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/query/graph_features.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/query/diffusion_reranker.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/query/entity_extraction.py` | ACTIVE | High | `@status: ACTIVE`. EntityExtractor. |
| `src/query/expansion_pipeline.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/planner.py` | ACTIVE | High | `@status: ACTIVE`. QueryPlanner. |
| `src/query/session_tracker.py` | ACTIVE | High | `@status: ACTIVE`. SessionTracker. |
| `src/query/signal_pool.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/structural_retrieval.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/vector_backends.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/retrieval_observability.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/retrieval_plan.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/retrieval_types.py` | ACTIVE | High | `@status: ACTIVE`. ChunkResult. |
| `src/query/processing/disambiguation.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/query/templates/__init__.py` | ACTIVE | High | `@status: ACTIVE`. Cypher templates. |
| `src/query/templates/advanced/__init__.py` | ACTIVE | High | `@status: ACTIVE`. Advanced Cypher templates. |
| `src/query/templates/advanced/schemas.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/services/graph_service.py` | ACTIVE | High | `@status: ACTIVE`. GraphService. |
| `src/services/text_service.py` | ACTIVE | High | `@status: ACTIVE`. TextService. |
| `src/services/context_assembler.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/services/context_budget_manager.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/services/cross_doc_linking.py` | ACTIVE | High | `@status: ACTIVE`. CrossDocLinker. |
| `src/services/cross_doc_edge_model.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/services/delta_cache.py` | ACTIVE | High | `@status: ACTIVE`. SessionDeltaCache. |
| `src/shared/config.py` | ACTIVE | High | `@status: ACTIVE`. Core config loader. Imported by everything. |
| `src/shared/connections.py` | AMBIGUOUS | High | `@status: ACTIVE` but contains DEAD methods. Imported by main.py, worker.py, atomic.py, query_service.py. Some methods marked DEAD (CompatQdrantClient.delete_sections_v1, CompatQdrantClient.delete_sections_v2). |
| `src/shared/schema.py` | ACTIVE | High | `@status: ACTIVE`. Neo4j schema DDL. |
| `src/shared/models.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/logging.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/cache.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/chunk_utils.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/embedding_fields.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/section_metadata.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/vector_utils.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/observability/__init__.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/observability/logging.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/observability/metrics.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/observability/tracing.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/observability/retrieval_diagnostics.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/observability/exemplars.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/resilience/__init__.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/resilience/circuit_breaker.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/shared/audit/logger.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/shared/feature_flags.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/providers/factory.py` | ACTIVE | High | `@status: ACTIVE`. ProviderFactory. |
| `src/providers/settings.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/tokenizer_service.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/base.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/embedding_service.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/jina.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/sentence_transformers.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/snowflake_arctic.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/qwen3_triton.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/voyage.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/chonkie_adapter.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/arctic_chonkie_adapter.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/qwen3_chonkie_adapter.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/embeddings/contracts.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/rerank/base.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/rerank/jina.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/rerank/local_reranker_service.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/rerank/noop.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/ner/gliner_service.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/providers/ner/labels.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/clients/embedding_client.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/clients/qwen3_embedding_client.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/clients/snowflake_embedding_client.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/connectors/base.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/connectors/github.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/connectors/manager.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/connectors/queue.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/connectors/circuit_breaker.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/monitoring/health.py` | ACTIVE | High | `@status: ACTIVE`. HealthChecker. |
| `src/monitoring/metrics.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/monitoring/slos.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/neo/schema.py` | ACTIVE | High | `@status: ACTIVE`. RELATIONSHIP_TYPES canonical list. |
| `src/neo/schema_validator.py` | ACTIVE | High | `@status: ACTIVE`. |
| `src/neo/structural_builder.py` | AMBIGUOUS | High | `@status: DEAD` but imports itself (circular). Imported by neo/structural_builder.py only. |
| `src/neo/contract_checks.py` | AMBIGUOUS | High | `@status: DEAD` but imported by worker.py at runtime (line 260). This is the highest-risk classification. |
| `src/neo/entity_normalization.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/neo/explain_guard.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/neo/defensive_query.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/neo/graph_enhancements.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/neo/health.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/ops/optimizer.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/ops/session_cleanup_job.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/ops/warmers/__init__.py` | DEAD | High | `@status: DEAD`. |
| `src/ops/warmers/query_warmer.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/registry/__init__.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/registry/index_registry.py` | DEAD | High | `@status: DEAD`. No callers. |
| `src/learning/__init__.py` | DEAD | High | `@status: DEAD`. Entire package unused. |
| `src/learning/feedback.py` | DEAD | High | `@status: DEAD`. |
| `src/learning/ranking_tuner.py` | DEAD | High | `@status: DEAD`. |
| `src/learning/suggestions.py` | DEAD | High | `@status: DEAD`. |
| `src/tools/inspect_chunk.py` | ACTIVE | High | `@status: ACTIVE`. CLI tool. |

### Scripts and Tools

| Path | Classification | Confidence | Evidence |
|------|---------------|------------|----------|
| `scripts/init_schema.py` | ACTIVE | High | Uses src.shared.schema.create_schema(). Active schema initialization. |
| `scripts/apply_complete_schema_v2_1.py` | BROKEN | High | References non-existent `scripts/neo4j/create_schema_v2_1_complete.cypher`. |
| `scripts/reset_datastores.py` | ACTIVE | High | Clears Qdrant/Neo4j/Redis. |
| `scripts/cleanup-databases.py` | ACTIVE | High | Surgical data deletion. |
| `scripts/monitor_ingestion.py` | ACTIVE | High | Real-time ingestion monitor. |
| `scripts/neo4j_structural_migration.py` | RETIRED | High | Has `--force-retired-script` guard. Dated 2026-03-02. |
| `scripts/migrate_section_to_chunk.py` | ACTIVE | High | Removes :Section label from dual-labeled nodes. |
| `scripts/backfill_cross_doc_edges.py` | ACTIVE | High | Creates RELATED_TO edges. |
| `scripts/backfill_document_tokens.py` | ACTIVE | High | Backfills Document.token_count. |
| `scripts/backfill_doc_title_vectors.py` | ACTIVE | High | Backfills doc_title vectors. |
| `scripts/evaluate_retrieval.py` | ACTIVE | High | Wrapper around scripts/eval/run_eval.py. |
| `scripts/run_canonical_retrieval_benchmark.py` | ACTIVE | High | Runs frozen 10-query benchmark. |
| `scripts/verify_providers.py` | ACTIVE | High | Verifies embedding/rerank providers. |
| `scripts/validate_gds_readiness.py` | ACTIVE | High | 8-gate GDS readiness validation. |
| `scripts/phase7e_preflight.py` | ACTIVE | High | Phase 7E preflight validation. |
| `scripts/ingestctl` | ACTIVE | High | Delegates to src.ingestion.auto.cli.main(). |
| `scripts/ci/check_phase_gate.py` | ACTIVE | High | CI phase gate check. |
| `scripts/ci/check_dead_imports.py` | ACTIVE | High | Dead import detection. |
| `scripts/eval/run_eval.py` | ACTIVE | High | Main evaluation harness. |
| `scripts/dev/seed_minimal_graph.py` | ACTIVE | High | Seeds deterministic test data. |
| `tools/fusion_ab.py` | ACTIVE | High | A/B testing for fusion methods. |
| `tools/redis_epoch_bump.py` | ACTIVE | High | Epoch-based cache invalidation. |
| `tools/redis_invalidation.py` | ACTIVE | High | Pattern-scan cache invalidation. |

### Infrastructure and Config

| Path | Classification | Confidence | Evidence |
|------|---------------|------------|----------|
| `docker-compose.yml` | ACTIVE | High | 497 lines. Defines all services. |
| `docker/mcp-server.Dockerfile` | ACTIVE | High | CMD: uvicorn src.mcp_server.main:app |
| `docker/ingestion-worker.Dockerfile` | ACTIVE | High | CMD: python -m src.ingestion.worker |
| `docker/ingestion-service.Dockerfile` | ACTIVE | High | CMD: uvicorn src.ingestion.auto.service:app |
| `docker/mxbai-reranker.Dockerfile` | ACTIVE | High | Standalone reranker service. |
| `deploy/k8s/base/` | ACTIVE | High | Kustomize base with 12 resources. |
| `deploy/k8s/overlays/staging/` | AMBIGUOUS | High | Empty directory. Exists but no kustomization files. |
| `deploy/k8s/overlays/production/` | AMBIGUOUS | High | Empty directory. Exists but no kustomization files. |
| `deploy/scripts/canary-rollout.sh` | ACTIVE | High | Progressive canary deployment. |
| `deploy/scripts/blue-green-switch.sh` | ACTIVE | High | Blue/green traffic switch. |
| `deploy/scripts/backup-all.sh` | ACTIVE | High | Full backup script. |
| `deploy/scripts/restore-all.sh` | ACTIVE | High | Full restore script. |
| `deploy/scripts/dr-drill.sh` | ACTIVE | High | DR drill script. |
| `infra/terraform/` | ACTIVE | High | Terraform for LGTM VM provisioning. |
| `config/development.yaml` | ACTIVE | High | Master config (566 lines). |
| `config/production.yaml` | DUPLICATE | High | Identical to development.yaml. |
| `config/embedding_profiles.yaml` | ACTIVE | High | Named embedding profile definitions. |
| `config/feature_flags.json` | ACTIVE | High | Feature flag definitions. |
| `config/alloy/config.alloy` | ACTIVE | High | Grafana Alloy telemetry config. |
| `.github/workflows/ci.yml` | ACTIVE | High | 5-job CI pipeline. |

### Standalone Services

| Path | Classification | Confidence | Evidence |
|------|---------------|------------|----------|
| `services/gliner-ner/server.py` | ACTIVE | High | FastAPI on port 9002. GLiNER NER. Self-contained. |
| `services/mxbai-reranker/server.py` | ACTIVE | High | FastAPI on port 9006. Cross-encoder reranker. Self-contained. |

### Deprecated/Empty/Generated

| Path | Classification | Confidence | Evidence |
|------|---------------|------------|----------|
| `ingest/` | EMPTY/DEPRECATED | High | Only contains empty `watch/` subdirectory and .DS_Store. |
| `migration/` | RETIRED | High | One-time-use migration scripts for embedding field canonicalization. |
| `mcp_app.py.bak` | BACKUP | High | Gitignored. Never imported. Pre-modification backup from Jan 19. |
| `claude-raw/` | AI ARTIFACTS | High | 39 files from Claude/GPT-5 Pro sessions. Not source code. |
| `context-*.md` (27 files) | AI ARTIFACTS | High | Session context snapshots. Gitignored. |
| `reports/` (355+ files) | GENERATED | High | Test/ingestion artifacts. Generated by CI and runtime. |
| `Archive.zip` | ARCHIVED | High | Old code snapshot. No references. |
| `Archive 2.zip` | ARCHIVED | High | Old code snapshot. No references. |
| `hf-cache/` | GENERATED | High | HuggingFace model cache. Gitignored. |
| `.npm-cache/` | GENERATED | High | npm cache. |
| `.pytest_cache/` | GENERATED | High | pytest cache. |
| `.ruff_cache/` | GENERATED | High | ruff cache. |
| `.uv-cache/` | GENERATED | High | uv package cache. |
| `.serena/` | GENERATED | High | Serena IDE state. |
| `scripts/neo4j/recovery-20251030/` | ARCHIVED | High | Recovery runbook and backups from Oct 30. |
| `scripts/neo4j/backup_phase7e4_20251030_152731/` | ARCHIVED | High | Backup from Phase 7E4 deployment. |
| `scripts/neo4j/schema_backup_*.cypher` | ARCHIVED | High | Schema backups. |
| `scripts/neo4j/neo4j_snapshots/` | ARCHIVED | High | Stored Neo4j schema snapshots. |
| `scripts/qdrant_snapshots_20251206_canonical/qdrant_snapshots/` | ARCHIVED | High | Stored Qdrant schema snapshots. |
| `scripts/qdrant and helpers/` | ARCHIVED | High | Versioned copies from PRs 1535/1546. |

## Multiple Plausible Active Paths

### 1. Markdown Parser: `markdown.py` vs `markdown_it_parser.py`

| Aspect | `markdown.py` | `markdown_it_parser.py` |
|--------|-------------|------------------------|
| Status | DEPRECATED | ACTIVE |
| Parser | Custom markdown parser | markdown-it-py AST parser |
| Imported by | atomic.py (via parsers/__init__.py router) | atomic.py (via parsers/__init__.py router) |
| Evidence | `@status: DEPRECATED` | `@status: ACTIVE` |
| Risk | Still in codebase, may be fallback path | Preferred path |

**Verdict:** `markdown_it_parser.py` is canonical. `markdown.py` is deprecated but may serve as fallback. The router in `parsers/__init__.py` determines which is used.

### 2. Watcher: `watcher.py` vs `watchers.py`

| Aspect | `watcher.py` | `watchers.py` |
|--------|-------------|---------------|
| Status | DEAD | ACTIVE |
| Class | FileSystemWatcher (singular) | FileSystemWatcher (plural, updated) |
| Imported by | Nothing | service.py, auto/__init__.py |
| Evidence | `@status: DEAD` | `@status: ACTIVE` |

**Verdict:** `watchers.py` (plural) is canonical. `watcher.py` (singular) is legacy.

### 3. Neo4j Schema: Multiple DDL Files

| File | Version | Status |
|------|---------|--------|
| `scripts/neo4j/create_schema.cypher` | Base | Active reference |
| `scripts/neo4j/create_schema_v2_1.cypher` | v2.1 | Active |
| `scripts/neo4j/create_schema_v2_1_complete.cypher` | v2.1 complete | Active (referenced by broken script) |
| `scripts/neo4j/create_schema_v2_1_complete__v3.cypher` | v2.1 v3 | Variant |
| `scripts/neo4j/create_graphrag_schema_v2_2_20251105.cypher` | v2.2 | Active |
| `scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher` | v2.2 guard | Active |
| `scripts/neo4j/create_schema_v2_2_complete__phase7E.cypher` | v2.2 Phase 7E | Active |
| `scripts/neo4j/schema_backup_20251029_deployed_v2.1.cypher` | v2.1 backup | Archived |
| `scripts/neo4j/schema_backup_20251105_clean_pre_chunking_reform.cypher` | Pre-chunking | Archived |
| `scripts/neo4j/schema_migration_phase3_markdown_it_py.cypher` | Phase 3 migration | Active |
| `scripts/neo4j/schema_ddl_complete_20251125.py` | v2.1 complete | Active |

**Verdict:** No clear canonical. The runtime schema is managed by `src.shared.schema.create_schema()` which is config-driven. The Cypher files serve as backups, migration scripts, and reference. The most recent is `create_schema_v2_2_complete__phase7E.cypher`.

## Do Not Touch Unless Confirmed

| Path | Reason |
|------|--------|
| `src/neo/contract_checks.py` | Marked DEAD but imported by worker.py. Removing the import without confirming GraphContractChecker is not needed will break ingestion worker startup. |
| `src/ingestion/build_graph.py` | Marked DEPRECATED but lazily imported by atomic.py. Removing without migrating GraphBuilder usage will break ingestion. |
| `src/shared/connections.py` | Contains DEAD methods but the module itself is ACTIVE. Removing the DEAD methods requires confirming no external callers exist. |
| `src/mcp_server/security/*` | Marked DEAD but represents a security gap. Either wire into main.py or explicitly document the lack of auth/rate-limiting. |
| `config/production.yaml` | If production truly differs from development, the differences are lost. Verify before deleting. |
| `deploy/k8s/overlays/staging/` and `production/` | Empty directories suggest planned but unimplemented environment-specific configs. |
| `src/ingestion/saga.py` | Marked DEAD but may contain reusable saga logic. Verify before deletion. |
| `scripts/apply_complete_schema_v2_1.py` | Broken script. May need the referenced Cypher file to be restored or the script to be updated. |

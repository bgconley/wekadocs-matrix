# Entrypoint Map

## Runtime Entrypoints

| Entrypoint | File | How it starts | Status | Notes |
|---|---|---|---|---|
| FastAPI MCP server | `src/mcp_server/main.py` | `docker/mcp-server.Dockerfile` CMD runs `python -m uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000` | Active | Mounts streamable MCP at `/_mcp`; legacy REST can be disabled/enabled by env. |
| Low-level MCP server | `src/mcp_server/mcp_app.py` | Built by `startup_event` in FastAPI and by stdio server | Active | Tool registry/instructions are WEKA-specific; `MCP_TOOL_PROFILE` filters listed tools. |
| STDIO MCP server | `src/mcp_server/stdio_server.py` | `python -m src.mcp_server.stdio_server` | Standalone/alternative | Shares `build_mcp_server()` with FastAPI path; not the Docker compose default. |
| Auto-ingestion service | `src/ingestion/auto/service.py` | `docker/ingestion-service.Dockerfile`, compose command `python -m src.ingestion.auto.service` | Active | Watches `/app/data/ingest`, enqueues Redis jobs, exposes health/metrics. |
| Ingestion worker | `src/ingestion/worker.py` | `docker/ingestion-worker.Dockerfile`, compose command `python -m src.ingestion.worker` | Active | Consumes Redis queue and calls `AtomicIngestionCoordinator.ingest_document_atomic()`. |
| GLiNER service | `services/gliner-ner/server.py` | Service-specific Docker/run path | Supporting | External service for NER enrichment when enabled. |
| Reranker service | `services/mxbai-reranker/server.py` | Service-specific Docker/run path | Supporting/alternative | Repo includes local reranker service; compose currently points to external gateway by env. |

## Manual/Operator Entrypoints

- `scripts/batch_crossdoc_link.py` and `scripts/backfill_cross_doc_edges.py` can run cross-document linking outside the ingestion worker.
- `scripts/eval/run_eval.py`, `scripts/run_canonical_retrieval_benchmark.py`, and `scripts/smoke_test_golden_queries.py` exercise retrieval but were not treated as authoritative tests.
- `scripts/neo4j/*.cypher` and schema helpers are migration/admin entrypoints.
- `src/ingestion/atomic.py` has a convenience `ingest_document_atomic()` function for direct use, but compose routes through the worker.

## Environment Entrypoints

- `CONFIG_PATH=/app/config/${ENV:-development}.yaml` in compose makes `config/development.yaml` the default app config.
- `EMBEDDINGS_PROFILE`, `EMBEDDING_NAMESPACE_MODE`, `RERANK_PROVIDER`, `RERANK_MODEL`, `RERANKER_BASE_URL`, `CHUNK_ASSEMBLER`, `MCP_TOOL_PROFILE`, and DB/queue envs materially alter active behavior.
- `.env.local`, `.env.docker`, `.env.production`, `docker-compose.yml`, `config/development.yaml`, and `config/embedding_profiles.yaml` disagree in several places, so runtime truth requires inspecting the launched environment, not just repo defaults.

## GitNexus Note

`npx gitnexus status` reports the index is current for commit `7745c92`. `gitnexus context build_mcp_server` confirmed callers from `src/mcp_server/main.py` and `src/mcp_server/stdio_server.py`.

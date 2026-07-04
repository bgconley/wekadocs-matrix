# 01 — Repo Inventory

## Directory-by-Directory Inventory

### Root Level

| Path | Type | Purpose |
|------|------|---------|
| `.env*` | Config | Multiple env files: `.env.example`, `.env.production`, `.env.docker`, `.env.local`, `.env.apply-schema`. All production env files gitignored. |
| `.github/workflows/ci.yml` | CI | 5-job pipeline: test → profile-matrix-smoke → eval-harness → build → deploy-staging → deploy-production |
| `.gitignore` | Config | Ignores .env files, __pycache__, reports/cleanup, logs, hf-cache, .serena, *.bak, agent context files |
| `.pre-commit-config.yaml` | Config | black, ruff, isort, detect-secrets, gitlint |
| `.coveragerc` / `.coveragerc.phase-3` | Config | General coverage + phase-gated 80% threshold for ingestion/shared |
| `pytest.ini` | Config | Markers: order, slow, integration, external, unit, live, xfail, chaos. Asyncio=auto. Strict markers. |
| `requirements.txt` | Manifest | ~80 deps: FastAPI, Neo4j, Qdrant, Redis, sentence-transformers, GLiNER, MCP SDK, OpenTelemetry, PyJWT |
| `Makefile` | Build | `make up/down`, `make test-phase-N`, Neo4j Cypher MCP management |
| `docker-compose.yml` | Infra | 497 lines. 12+ services: mcp-server, ingestion-worker, ingestion-service, neo4j, qdrant, redis, mxbai-reranker, plus dev tools |
| `bootstrap_schema.py` | Script | One-shot Qdrant schema bootstrap for bge_m3 profile |
| `db-check.py` | Script | Database status checker (Neo4j/Qdrant/Redis parity) |
| `inventory_neo4j.py` | Script | Neo4j schema dump to `.cypher` |
| `inventory_qdrant.py` | Script | Qdrant schema inventory to JSON |
| `neo4j_schema_dump.cypher` | Data | Live Neo4j schema dump (v2.2, 16 constraints, 4 vector indexes, 3 fulltext indexes) |
| `neo4j_full_migration.cypher` | Data | Complete migration script (same schema + optional relationship builders) |
| `qdrant_schema_inventory.json` | Data | Live Qdrant schema (chunks_multi_bge_m3, 4 dense + 4 sparse vectors) |
| `ARCHITECTURE.md` | Docs | (Not found at root — docs in `docs/` subdirs) |
| `AGENT_CONTEXT.md` | AI Artifact | Generated session context (Jan 19, 2026). Gitignored. |
| `TASK_BACKLOG.md` | AI Artifact | 52 items, P0-P3. Generated Jan 19, 2026. |
| `SESSION_CONTEXT_*.md` | AI Artifacts | 2 session context files. Gitignored. |
| `SESSION_PROGRESS_*.md` | AI Artifact | Phase progress notes. |
| `SESSION-SUMMARY.md` | AI Artifact | Session summary. |
| `EMBEDDER_CORRECTIONS.md` | AI Artifact | Embedder fix notes. |
| `rrf-fusion-no-neo4j-20251205.md` | Docs | RRF fusion research notes. |
| `architecture_diagram.svg` | Asset | SVG architecture diagram. |
| `qdrant_sample.json` | Data | Sample Qdrant data. |
| `phase4-kickoff-fixtures.patch` | Data | Test fixture patch. |
| `Archive.zip`, `Archive 2.zip` | Archive | Old code snapshots. No references. |

### `src/` — Source Code (137 Python files, 26 packages)

| Package | Files | Purpose | Status |
|---------|-------|---------|--------|
| `src/__init__.py` | 1 | Package root | ACTIVE |
| `src/clients/` | 4 | Embedding HTTP clients (unified, Qwen3, Snowflake Arctic) | ACTIVE |
| `src/connectors/` | 7 | External system connectors (GitHub, base, circuit breaker, Redis queue) | ACTIVE |
| `src/ingestion/` | 14 | Document ingestion pipeline (atomic saga, chunking, parsing, entity extraction) | ACTIVE |
| `src/ingestion/auto/` | 12 | Auto-ingestion subsystem (queue, watcher, CLI, reaper, service) | MIXED |
| `src/ingestion/extract/` | 6 | Entity extraction (commands, configs, procedures, GLiNER NER, references) | ACTIVE |
| `src/ingestion/parsers/` | 6 | Document parsers (markdown, markdown-it-py, HTML, Notion, shadow comparison) | MIXED |
| `src/learning/` | 4 | Learning system (feedback, ranking tuner, suggestions) | DEAD |
| `src/mcp_server/` | 11 | MCP protocol server (FastAPI, tool definitions, query service, security) | MIXED |
| `src/mcp_server/security/` | 3 | Auth and rate limiting | DEAD |
| `src/monitoring/` | 4 | Health checks, Prometheus metrics, SLO tracking | ACTIVE |
| `src/neo/` | 10 | Neo4j-specific (schema, structural edges, entity normalization, explain guard) | MIXED |
| `src/ops/` | 3 | Operations (query optimizer, session cleanup, query warmer) | MIXED |
| `src/providers/` | 15 | Provider abstraction (factory, settings, tokenizer, embeddings, NER, rerank) | ACTIVE |
| `src/providers/embeddings/` | 10 | Embedding providers (Jina, SentenceTransformers, Snowflake Arctic, Qwen3, Voyage, Chonkie adapters) | ACTIVE |
| `src/providers/ner/` | 3 | NER provider (GLiNER) | ACTIVE |
| `src/providers/rerank/` | 5 | Rerank providers (Jina, local, noop) | ACTIVE |
| `src/query/` | 22 | Query/retrieval pipeline (hybrid search, fusion, graph expansion, ranking, response building) | MIXED |
| `src/query/templates/` | 7 | Cypher query templates (search, traverse, explain, troubleshoot, compare, advanced) | ACTIVE |
| `src/registry/` | 2 | Vector index registry | DEAD |
| `src/services/` | 7 | Service layer (graph, text, context assembler, budget manager, cross-doc linking) | ACTIVE |
| `src/shared/` | 16 | Core infrastructure (config, connections, schema, models, logging, observability, resilience) | MIXED |
| `src/shared/observability/` | 5 | Tracing, metrics, logging, retrieval diagnostics, exemplars | ACTIVE |
| `src/shared/resilience/` | 2 | Circuit breaker pattern | ACTIVE |
| `src/tools/` | 1 | CLI tools (chunk inspection) | ACTIVE |

### `tests/` — Test Suite (~170 Python files, 19 directories)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `tests/conftest.py` | 1 | Session-scoped fixtures: Docker services, FastAPI TestClient, Neo4j/Redis/Qdrant clients, JWT tokens, OTel tracing |
| `tests/unit/` | 26 | Unit tests (tokenizer, structural retrieval, GLiNER, cross-doc linking, markdown parser, etc.) |
| `tests/integration/` | 22 | Integration tests (GDS readiness, GLiNER flow, Jina batching, sparse ColBERT, session tracking) |
| `tests/e2e/` | 1 | Golden set baseline (20-query set, no mocks) |
| `tests/e2e_v22_prod/` | 6 | v2.2 production validation (spec-only, not yet executed) |
| `tests/contracts/` | 3 | Contract tests (Cypher policy, graph v2, MCP streamable) |
| `tests/ingestion/` | 3 | Ingestion tests (graph building, namespace enforcement, per-call overrides) |
| `tests/mcp_server_tests/` | 2 | MCP server tests (evidence pack, tool profiles) |
| `tests/neo/` | 1 | Neo4j tests (explain guard) |
| `tests/providers/` | 5 | Provider tests (Arctic Chonkie, BGE-M3, profile matrix, Snowflake Arctic, Voyage) |
| `tests/query/` | 16 | Query engine tests (context assembly, guardrails, hybrid bridge, RRF, reranking, etc.) |
| `tests/services/` | 1 | Service tests (mxbai reranker) |
| `tests/shared/` | 5 | Shared tests (config reload, embedding plan fingerprint, profiles, namespace suffix, schema) |
| `tests/fixtures/` | 8 | YAML query sets, sample docs with references |
| `tests/eval/` | 3 | Evaluation harness (id resolver, metrics, queries.yaml) |
| `tests/baselines/` | 1 | Baseline data |
| `tests/clients/` | 1 | Client tests (Snowflake embedding) |
| `tests/scripts/` | 2 | Script tests (backfill edge parity, verify providers) |
| `tests/p*_t*_test.py` | 24 | Phase-organized standalone tests (P1-P6, T1-T4) |
| `tests/test_phase*.py` | 15 | Phase-specific tests (phase1-7E) |

### `scripts/` — CLI Scripts (~60 files)

| Subdirectory | Purpose |
|--------------|---------|
| `scripts/` (root) | Schema init, database reset, cleanup, ingestion monitoring, backfill scripts, evaluation, validation, smoke tests |
| `scripts/ci/` | CI scripts (phase gate check, dead import detection) |
| `scripts/eval/` | Evaluation harness (run_eval, gold eval, go/no-go, lexical comparison) |
| `scripts/dev/` | Development scripts (seed minimal graph) |
| `scripts/migration/` | Migration Cypher (phase2 edge cleanup) |
| `scripts/neo4j/` | Neo4j schema DDL files, snapshots, recovery runbooks, setup scripts |
| `scripts/perf/` | Performance tests (traversal latency, verbosity latency) |
| `scripts/phase0/` | Phase 0 scripts (baseline capture) |
| `scripts/test/` | Test scripts (run phase, check metrics, summarize) |
| `scripts/retrieval_diagnostics/` | Retrieval diagnostics (show) |
| `scripts/qdrant and helpers/` | Versioned Qdrant helper scripts (archived) |
| `scripts/qdrant_snapshots_20251206_canonical/` | Qdrant schema snapshots |

### `docker/` — Dockerfiles

| File | Service | Port | Base |
|------|---------|------|------|
| `mcp-server.Dockerfile` | MCP Server | 8000 | python:3.11-slim |
| `ingestion-worker.Dockerfile` | Ingestion Worker | — | python:3.11-slim |
| `ingestion-service.Dockerfile` | Auto-Ingest Service | 9108 | python:3.11-slim |
| `mxbai-reranker.Dockerfile` | Cross-Encoder Reranker | 9006 | pytorch:2.6.0-cuda12.6 |

### `deploy/` — Deployment

| Path | Purpose |
|------|---------|
| `deploy/k8s/base/` | Kustomize base: 12 resources (mcp-server-blue/green/canary, ingestion-worker, neo4j, qdrant, redis) |
| `deploy/k8s/overlays/staging/` | Staging overlay (empty directory) |
| `deploy/k8s/overlays/production/` | Production overlay (empty directory) |
| `deploy/helm/` | Helm charts (if any) |
| `deploy/scripts/` | Deploy scripts (canary-rollout, blue-green-switch, backup-all, restore-all, dr-drill) |
| `deploy/monitoring/` | Prometheus alerts, Grafana dashboards |
| `deploy/DR-RUNBOOK.md` | Disaster recovery runbook |

### `infra/` — Infrastructure as Code

| Path | Purpose |
|------|---------|
| `infra/terraform/` | Terraform: GCP project, LGTM VM module (Loki, Tempo, Mimir, Grafana) |
| `infra/scripts/` | Terraform deploy/destroy scripts |

### `monitoring/` — Monitoring Config

| Path | Purpose |
|------|---------|
| `monitoring/alerts/` | Phase 7E SLO alerts (oversized chunks, integrity, expansion rate, latency, etc.) |
| `monitoring/dashboards/` | Phase 7E ingestion, retrieval, SLO dashboards |

### `config/` — Application Config

| Path | Purpose |
|------|---------|
| `config/development.yaml` | Master config (566 lines): embedding profiles, search config, chunk assembly, GLiNER, cache, feature flags |
| `config/production.yaml` | Production config (identical to development.yaml) |
| `config/embedding_profiles.yaml` | Named embedding profile definitions |
| `config/feature_flags.json` | Feature flag definitions |
| `config/alloy/config.alloy` | Grafana Alloy telemetry config |
| `config/grafana/` | Grafana dashboard provisioning and JSON dashboards |

### `services/` — Standalone ML Services

| Path | Purpose |
|------|---------|
| `services/gliner-ner/` | GLiNER NER service (FastAPI, port 9002, MPS/CUDA/CPU) |
| `services/mxbai-reranker/` | Cross-encoder reranker (FastAPI, port 9006, CUDA) |

### `data/` — Data Directories

| Path | Purpose |
|------|---------|
| `data/documents/` | Ingested documents |
| `data/ingest/` | Input directory for file watcher |
| `data/samples/` | Sample data |
| `data/test/` | Test data |

### `hf-cache/` — HuggingFace Cache

Cached models: BAAI/bge-m3, Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Reranker-0.6B/4B, Snowflake/snowflake-arctic-embed-l-v2.0, voyageai/voyage-context-3

### `reports/` — Generated Artifacts (355+ files)

| Subdirectory | Content |
|--------------|---------|
| `reports/phase-{1-7}/` | Phase-specific test reports (junit.xml, summary.json) |
| `reports/ingest/<uuid>/` | Per-ingestion-job reports |
| `reports/cleanup/` | Cleanup operation reports |
| `reports/community_detection/` | Community detection analysis |
| `reports/retrieval_diagnostics/` | RRF parameter research, diagnostics |
| `reports/retrieval_benchmarks/` | Benchmark results |
| `reports/full-suite-*/` | Full suite test reports |
| `reports/test-runs/` | Test run artifacts |

### `docs/` — Documentation

| Subdirectory | Content |
|--------------|---------|
| `docs/architecture/` | End-to-end architecture docs |
| `docs/decisions/` | Architecture decision records |
| `docs/guides/` | User/developer guides |
| `docs/plans/` | Implementation plans |
| `docs/session-notes/` | Session notes |
| `docs/bugfix/` | Bug fix documentation |
| `docs/mcp/` | MCP protocol docs |

### `claude-raw/` — AI Session Artifacts (39 files)

Historical planning documents, implementation plans, Cypher scripts from Claude/GPT-5 Pro sessions. Not part of active codebase.

### `context-*.md` — AI Session Context (27 files)

Historical AI session context snapshots. Not part of active codebase.

## Frameworks and Tools Detected

| Category | Tool | Version |
|----------|------|---------|
| Web Framework | FastAPI + uvicorn | Latest |
| API Protocol | MCP SDK | 1.16.0 |
| Graph Database | Neo4j | 5.15-community |
| Vector Database | Qdrant | 1.7.4 |
| Cache/Queue | Redis | 7.2-alpine |
| ORM/Driver | neo4j driver | 5.26.0 |
| Embeddings | sentence-transformers | 2.7.0 |
| NER | GLiNER | 0.2.24 |
| Chunking | Chonkie | Latest |
| Auth | PyJWT | 2.8.0 |
| Observability | OpenTelemetry + Prometheus | Latest |
| Logging | structlog + python-json-logger | Latest |
| Testing | pytest + pytest-asyncio + pytest-cov | 7.4.4 + 0.23.3 + 4.1.0 |
| Linting | ruff | 0.14.1 |
| Formatting | black | 25.9.0 |
| Import sorting | isort | 7.0.0 |
| Security | detect-secrets | 1.5.0 |
| Commit lint | gitlint | 0.19.1 |
| Containerization | Docker + Docker Compose | — |
| Orchestration | Kubernetes (Kustomize) | — |
| CI/CD | GitHub Actions | — |
| IaC | Terraform | >= 1.5.0 |
| Telemetry | Grafana Alloy + LGTM stack | — |

## Entrypoints and Runtime Commands

### Docker Compose Services

```bash
docker compose up -d                    # Start all services
docker compose up mcp-server            # Start MCP server only
docker compose down -v                  # Stop and remove volumes
```

### Application Commands

```bash
# MCP Server (HTTP)
uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000

# MCP Server (STDIO — Claude Desktop)
python -m src.mcp_server.stdio_server

# Ingestion Worker
python -m src.ingestion.worker

# Auto-Ingest Service
uvicorn src.ingestion.auto.service:app --host 0.0.0.0 --port 9108

# Ingest CLI
python -m src.ingestion.auto.cli ingest <file>
python -m src.ingestion.auto.cli status
python -m src.ingestion.auto.cli report

# Schema
python scripts/init_schema.py
python bootstrap_schema.py

# Database checks
python db-check.py
python inventory_neo4j.py
python inventory_qdrant.py

# Backfill
python scripts/backfill_cross_doc_edges.py
python scripts/backfill_document_tokens.py
python scripts/backfill_doc_title_vectors.py
```

### Test Commands

```bash
# All tests
pytest

# Phase-specific
make test-phase-1    # through test-phase-6
pytest tests/p*_t*_test.py

# By marker
pytest -m unit
pytest -m integration
pytest -m e2e

# Coverage
pytest --cov=src --cov-config=.coveragerc
pytest --cov=src --cov-config=.coveragerc.phase-3  # ingestion+shared only, 80% threshold
```

### Deploy Commands

```bash
# Kubernetes
kubectl apply -k deploy/k8s/overlays/staging
bash deploy/scripts/canary-rollout.sh --auto-rollback
bash deploy/scripts/blue-green-switch.sh

# Backup/Restore
bash deploy/scripts/backup-all.sh
bash deploy/scripts/restore-all.sh
bash deploy/scripts/dr-drill.sh

# Terraform
bash infra/scripts/deploy-lgtm.sh
bash infra/scripts/destroy-lgtm.sh
```

## Test/Build/Deploy Commands Summary

| Command | What It Does |
|---------|-------------|
| `make up` | Docker Compose up with volume cleanup |
| `make down` | Docker Compose down with volume cleanup |
| `make test-phase-N` | Runs phase N tests via `scripts/test/run_phase.sh` |
| `pytest` | Runs all tests (asyncio=auto, strict markers) |
| `pytest -m integration` | Runs integration tests only |
| `pytest --cov=src` | Runs tests with coverage |
| `.github/workflows/ci.yml` | CI: test → profile-matrix-smoke → eval-harness → build → deploy-staging → deploy-production |
| `docker build -f docker/mcp-server.Dockerfile` | Build MCP server image |
| `kubectl apply -k deploy/k8s/base` | Deploy to K8s |
| `bash deploy/scripts/canary-rollout.sh` | Progressive canary deployment |

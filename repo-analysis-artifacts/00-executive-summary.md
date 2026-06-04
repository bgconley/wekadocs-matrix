# 00 — Executive Summary

## What This Repo Is

**WekaDocs-Matrix** is a GraphRAG (Retrieval-Augmented Generation) documentation intelligence system for WEKA distributed file system documentation. It ingests Markdown/HTML documentation, builds a knowledge graph in Neo4j, creates multi-vector embeddings in Qdrant (dense + sparse + ColBERT), and exposes retrieval via the Model Context Protocol (MCP) to AI coding assistants.

**Tech stack:** Python 3.11, FastAPI, Neo4j 5.15, Qdrant 1.7.4, Redis 7.2, MCP SDK 1.16.0, 6 AI models via unified GPU gateway.

**Current state:** Phase 7E+. Multi-embedder architecture in progress. ~137 source files, ~170 test files, 497-line docker-compose with 12+ services.

## Main Active Path

The system runs as **three independent processes** (all containerized via Docker/K8s):

| Process | Entrypoint | Purpose |
|---------|-----------|---------|
| MCP HTTP Server | `src/mcp_server/main.py` | FastAPI + MCP Streamable HTTP on port 8000. Handles all query/retrieval requests. |
| Ingestion Worker | `src/ingestion/worker.py` | Async Redis-polling worker. Saga-coordinated Neo4j + Qdrant writes via `AtomicIngestionCoordinator`. |
| Auto-Ingest Service | `src/ingestion/auto/service.py` | FastAPI on port 9108. File watcher + enqueue + Prometheus metrics. |

**Supporting infrastructure:** Neo4j (graph), Qdrant (vectors), Redis (cache/queue), mxbai-reranker (GPU cross-encoder), GLiNER NER service.

## Main Ambiguous/Deprecated Paths

| Path | Classification | Risk |
|------|---------------|------|
| `src/ingestion/auto/orchestrator.py` | **DEAD** — superseded by `atomic.py` | Low (not imported) |
| `src/ingestion/api.py` | **DEAD** — test facade only | Low (not imported) |
| `src/ingestion/build_graph.py` | **DEPRECATED** — legacy GraphBuilder, still lazily imported by `atomic.py` | **Medium** — creates coupling to dead code |
| `src/ingestion/saga.py` | **DEAD** — but `atomic.py` has its own inline saga logic | **Medium** — potential confusion |
| `src/ingestion/reconcile.py` | **DEAD** — no runtime callers | Low |
| `src/ingestion/incremental.py` | **DEAD** — no runtime callers | Low |
| `src/ingestion/parsers/notion.py` | **DEAD** — no runtime callers | Low |
| `src/learning/*` | **DEAD** — entire package unused | Low |
| `src/mcp_server/security/*` | **DEAD** — auth/rate-limiting not wired into main.py | **Medium** — security gap |
| `src/mcp_server/validation.py` | **DEAD** — never wired in | Low |
| `src/neo/contract_checks.py` | **DEAD** — but imported by `worker.py` at runtime | **High** — ACTIVE code imports DEAD module |
| `src/neo/structural_builder.py` | **DEAD** — but imported by itself (circular) | Low |
| `src/ops/warmers/*` | **DEAD** — no callers | Low |
| `src/registry/*` | **DEAD** — no callers | Low |
| `src/shared/feature_flags.py` | **DEAD** — no callers | Low |
| `src/shared/audit/logger.py` | **DEAD** — no callers | Low |
| `src/query/diffusion_reranker.py` | **DEAD** — no callers | Low |
| `src/query/graph_expansion.py` | **DEAD** — no callers | Low |
| `src/query/graph_features.py` | **DEAD** — no callers | Low |
| `scripts/apply_complete_schema_v2_1.py` | **BROKEN** — references non-existent Cypher file | Low |
| `config/production.yaml` | **DUPLICATE** — identical to `development.yaml` | Low |
| `mcp_app.py.bak` | **BACKUP** — gitignored, never imported | Low |
| `claude-raw/` | **AI ARTIFACTS** — 39 files from prior sessions | Low |
| `context-*.md` | **AI ARTIFACTS** — 27 session context files | Low |
| `reports/` | **GENERATED** — 355+ test/ingestion artifacts | Low |
| `Archive.zip`, `Archive 2.zip` | **OLD SNAPSHOTS** — no references | Low |
| `ingest/` | **EMPTY** — deprecated, active code in `src/ingestion/` | Low |
| `migration/` | **RETIRED** — one-time-use migration scripts | Low |

## Biggest Risks

1. **Security modules marked DEAD but not removed** (`src/mcp_server/security/*`) — JWT auth and rate limiting exist but are never wired into the main server. The MCP server has no authentication or rate limiting in production.

2. **DEAD module still imported at runtime** (`src/neo/contract_checks.py` imported by `worker.py`) — This is the highest-risk classification: code marked DEAD but actively imported. If this module has bugs or dependency issues, the ingestion worker will fail to start.

3. **Legacy `build_graph.py` still referenced** — `atomic.py` lazily imports `GraphBuilder` from the deprecated `build_graph.py`. This creates a coupling to dead code that should be migrated.

4. **Multiple schema versions in `scripts/neo4j/`** — At least 6 different schema DDL files spanning v2.0 through v2.2. No clear canonical. `apply_complete_schema_v2_1.py` is broken.

5. **Config duplication** — `production.yaml` is identical to `development.yaml`. If production needs differ, they are not reflected.

6. **Repo bloat** — ~400+ files of generated artifacts, AI session contexts, old archives, and backup files consume significant space and add noise.

## Recommended Next Steps

1. **Immediate (low risk):** Remove `mcp_app.py.bak`, `claude-raw/`, `context-*.md`, `Archive.zip`, `Archive 2.zip`, `reports/` (or move to external storage).
2. **High priority:** Wire up or remove `src/mcp_server/security/*` — either integrate JWT auth and rate limiting into `main.py`, or remove the dead code and document the gap.
3. **Medium priority:** Resolve `src/neo/contract_checks.py` — either make it ACTIVE (wire into worker.py properly) or remove the import and delete the module.
4. **Medium priority:** Migrate `GraphBuilder` usage from `build_graph.py` into `atomic.py` or a new module, then delete `build_graph.py`.
5. **Low priority:** Consolidate Neo4j schema files — pick one canonical DDL, archive the rest.
6. **Low priority:** Differentiate `production.yaml` from `development.yaml` or delete the duplicate.

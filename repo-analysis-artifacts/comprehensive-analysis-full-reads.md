# Comprehensive Repository Analysis - Full File Read Results

**Date:** 2026-05-09
**Branch:** multi-embedder-reranker
**Analysis Method:** 14 parallel agents, full line-by-line file reads
**Files Analyzed:** 1,130 files across 10 chunks (113 files each)

---

## Executive Summary

This is a **WekaDocs GraphRAG MCP system** — a hybrid Qdrant (vector) + Neo4j (graph) retrieval-augmented generation pipeline for WEKA documentation. The system implements an 8-step retrieval pipeline: embed → multi-vector Qdrant search → BM25 → RRF fusion → entity/structural boost → cross-encoder reranking → expansion → evidence extraction.

### Key Metrics
- **Neo4j:** 3,621 nodes, 3,747 relationships, 268 sections
- **Qdrant:** 268 vectors (384 dims, cosine similarity)
- **Redis:** Queue operational with reaper process
- **MCP Server:** P95 = 70ms, graph mode default
- **Docker:** All 7 services healthy

### Architecture: 3-Process Model
1. **MCP Server** — Native SDK (no FastMCP), supports STDIO and Streamable HTTP transports, tool profiles (production/analyst)
2. **Ingestion Worker** — Semantic chunking, atomic saga pattern, Redis Streams job queue
3. **Auto-Ingest Service** — Background document ingestion with reconciliation

---

## Module Status (from Import Graph Analysis)

| Status | Count | Description |
|--------|-------|-------------|
| **ACTIVE** | 115 | Currently imported and used |
| **DEAD** | 38 | No longer imported, code rot risk |
| **STANDALONE** | 3 | Self-contained, no dependencies |

### Critical Issues from Module Analysis

1. **`src/neo/contract_checks.py`** — Marked DEAD but lazily imported by `worker.py`
2. **`src/ingestion/build_graph.py`** — Marked DEPRECATED but lazily imported by `atomic.py` behind config flag
3. **Auth/Rate-limiting modules exist but NOT wired in** — Security gap

---

## Retrieval Pipeline Architecture

### 4-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Recall Layer (Parallel Retrieval)                        │
│    Dense Vector │ Sparse Vector │ ColBERT │ BM25 (Legacy)   │
│    └────────────┴───────────────┴────────┴─────────┘        │
│                    ↓ Score Fusion (RRF)                      │
├─────────────────────────────────────────────────────────────┤
│ 2. Precision Layer (Cross-Encoder)                          │
│    Input: Top 50 Candidate Chunks → Reranker Model          │
├─────────────────────────────────────────────────────────────┤
│ 3. Expansion Layer (Context Stitching)                      │
│    Micro-Doc Stitching │ Adjacency Expansion                │
├─────────────────────────────────────────────────────────────┤
│ 4. Final Ranking (Feature Blending)                         │
│    Ranker.rank() = Semantic + Graph + Recency               │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Vector Qdrant Schema
- **3 Dense vectors:** content, title, doc_title
- **4 Sparse vectors:** text-sparse, doc_title-sparse, title-sparse, entity-sparse
- **1 ColBERT multivector:** late-interaction (128-dim per-token)

### Embedding Stack
| Model | Type | Dimensions | Purpose |
|-------|------|-----------|---------|
| Qwen3-Embedding-0.6B | Dense | 1024 | Primary dense embeddings |
| SPLADEv3 | Sparse | Variable | Lexical matching |
| ColBERTv2 | Late-interaction | 128/token | Token-level MaxSim |
| BGE-M3 | Multi-head | 1024 | Fallback/alternative |
| Snowflake Arctic Embed L v2.0 | Dense | 1024 | Chonkie boundary detection |
| Voyage Context-3 | Dense (contextual) | 1024 | Contextual chunk embeddings |
| Jina v3 | Dense (multilingual) | 1024 | Multilingual support |

### Reranker Stack
- **Primary:** Qwen3-Reranker-4B (8K context), domain-tuned instructions
- **Fallback:** mxbai-reranker-small-v1, Jina v3
- **Gateway:** Unified at 10.25.0.50:8080

### Signal Pool Architecture
- 200-candidate signal-diverse rerank pool
- 9 slots: consensus, per-signal unique, structural, per-document depth
- Ensures cross-encoder evaluates chunks from every retrieval signal source

---

## Graph Model

### Node Types
- `:Chunk` — Primary unit of retrieval (replaced `:Section`)
- `:Document` — Top-level document container
- `:Entity` — Extracted entities (GLiNER)
- `:Heading` — Document heading hierarchy

### Relationship Types
| Relationship | Direction | Description |
|-------------|-----------|-------------|
| `HAS_CHUNK` | Document→Chunk | Document contains chunks |
| `NEXT_CHUNK` | Chunk→Chunk | Sequential chunk ordering |
| `PARENT_HEADING` | Chunk→Heading | Chunk belongs to heading |
| `CHILD_OF` | Heading→Heading | Heading hierarchy |
| `MENTIONS` | Chunk→Entity | Chunk mentions entity |
| `RELATED_TO` v2 | Chunk↔Chunk | Cross-document semantic similarity (signal-provenance edges) |

### Cross-Document Linking (Phase 3.5)
- 6-stage pipeline for RELATED_TO edge creation
- Uses dense/sparse/ColBERT signals with RRF fusion
- Signal-provenance edge model with reciprocity and structural priors
- Backfill script: `scripts/backfill_cross_doc_edges.py` (1,356 lines)

---

## Entity Extraction

### GLiNER Medium v2.1
- Singleton pattern with HTTP/local mode fallback
- Circuit breaker + LRU caching for short texts
- Quality gating: threshold 0.55, per-label confidence floors, 8-entity cap per chunk
- Deterministic entity_id generation
- WEKA entity exclusions

### Entity Labels
- STORAGE_CONCEPT (confidence floor ≥ 0.70)
- COMMAND (confidence floor ≥ 0.55)
- Plus domain-specific labels

---

## Ingestion Pipeline

### Atomic Saga Pattern
- Ensures Neo4j and Qdrant stay in sync
- Compensating transactions on failure
- Drift detection and reconciliation

### Semantic Chunking
- Uses Chonkie with Snowflake Arctic Embed L v2.0 for boundary detection
- Line-level granularity with parent_path tracking
- block_types, code_ratio tracking

### Redis Streams Job Queue
- Priority queuing (LPUSH/RPUSH)
- Backpressure monitoring
- Graceful degradation
- Stale job reaper process

### Key Ingestion Modules
| Module | Status | Description |
|--------|--------|-------------|
| `src/ingestion/atomic.py` | ACTIVE | Atomic ingestion pipeline |
| `src/ingestion/semantic_chunker.py` | ACTIVE | Semantic chunking |
| `src/ingestion/worker.py` | ACTIVE | Worker process |
| `src/ingestion/reconcile.py` | DEAD | Sync Qdrant with graph (dead code) |
| `src/ingestion/build_graph.py` | DEPRECATED | Legacy graph builder (superseded) |
| `src/ingestion/saga.py` | ACTIVE | Saga pattern for atomicity |

---

## Query Pipeline

### Retrieval Plan System
Consolidates 18+ scattered boolean flags into 4 named profiles:
1. **vector_only** — Basic vector search
2. **precision_vector** — Vector with precision tuning
3. **graph_assisted** — Vector + graph enhancement
4. **graph_full** — Full graph + vector pipeline

### Query Processing Steps
1. Embed query
2. Multi-vector Qdrant search
3. BM25 retrieval
4. RRF fusion (k=60, recommended reduction to 30-40)
5. Entity/structural boost
6. Cross-encoder reranking
7. Expansion (micro-doc stitching, adjacency)
8. Evidence extraction

### Graph Expansion
- `ContextExpander` with parent/adjacent expansion
- NOT fully integrated into main pipeline
- Query-type-aware relationship selection

---

## Services Layer

| Service | Description |
|---------|-------------|
| `ContextAssemblerService` | Assembles final context from retrieved chunks |
| `SummarizationService` | Generates summaries |
| `ContextBudgetManager` | Phase-aware token/byte budget enforcement |
| `CrossDocLinker` | Phase 3.5 semantic similarity edges (1,420 lines) |
| `GraphService` | Projection-only Cypher queries with cursor-aware paging |
| `TextService` | Section text fetching with truncation and byte accounting |
| `DeltaCache` | In-memory session-scoped deduplication cache |

---

## MCP Server

### Transports
- **STDIO** — Local development
- **Streamable HTTP** — Production deployment

### Tool Profiles
- **production** — Default, safety guardrails active
- **analyst** — Extended analysis capabilities

### Enhanced Responses (Contexts E1-E7)
- Bi-directional graph traversal
- Graph mode default
- Verbosity modes
- traverse_relationships tool

### Performance
- P95 = 70ms (7x better than 500ms requirement)

---

## Security Analysis

### Current State
- **Auth module exists:** `src/mcp_server/security/auth.py`
- **Rate limiter exists:** `src/mcp_server/security/rate_limiter.py`
- **Circuit breaker:** `src/connectors/circuit_breaker.py` (150 lines)
- **Webhook verification:** HMAC-SHA256 signature verification
- **detect-secrets:** Pre-commit hook with baseline

### Security Gaps
1. **Auth NOT wired in** — Modules exist but not integrated into MCP server
2. **Rate-limiting NOT wired in** — Same issue
3. **Hardcoded New Relic key in `.env.production`** — Credential exposure risk
4. **Fallback passwords in deployment scripts** — Weak security posture

---

## Deployment Infrastructure

### Docker Compose (7 Services)
1. MCP Server
2. Ingestion Worker
3. Ingestion Service
4. Neo4j
5. Qdrant
6. Redis
7. Reranker (mxbai)

### Kubernetes (Blue/Green + Canary)
- **Namespace:** wekadocs-matrix
- **Deployments:** mcp-server, mcp-server-canary, mcp-server-green, ingestion-worker, neo4j-statefulset, qdrant-statefulset, redis-statefulset
- **ConfigMap:** Centralized configuration
- **Ingress:** External access
- **Secrets:** Kubernetes secrets (currently empty placeholder)

### DR Runbook
- **RTO:** 1 hour
- **RPO:** 15 minutes
- **Scripts:** backup-all.sh, restore-all.sh, dr-drill.sh

### Monitoring
- **Prometheus:** 30+ metrics
- **Grafana:** 5 dashboards (overview, ingestion, retrieval, infrastructure, cross-doc-linking)
- **OTEL:** Full tracing (FastAPI, Neo4j, Qdrant, Redis)
- **Alloy:** Telemetry collector for GCP LGTM stack

---

## Testing Infrastructure

### Test Coverage by Phase
| Phase | Tests | Coverage |
|-------|-------|----------|
| Phase 1 | 100% | MCP server |
| Phase 2 | 100% | Graph operations |
| Phase 3 | 100% | Cross-doc linking |
| Phase 4 | 100% | Advanced query features |
| Phase 5 | 293/297 | 98.65% |
| Phase 6 | ~95%+ | Auto-ingestion |

### Test Categories
- Provider wiring
- Dimension safety
- Ranking stability
- Schema v2.1 constraints
- Cache invalidation
- Observability/SLOs
- Cross-document linking
- Entity quality
- Parser compatibility

### Test Markers
order, slow, integration, external, unit, live, xfail, chaos

---

## Known Issues & Anomalies

### Critical
1. **"Island of graphs" anti-pattern** — No cross-document edges (partially addressed by Phase 3.5)
2. **BGE reranker batching bug** — Causes 300-500ms latency (1 HTTP call per document)
3. **Atomic pipeline drops Entity-to-Entity relationships** — CONTAINS_STEP edges missing
4. **Dead schema declarations** — 5 relationship types never created
5. **Truncation metadata mismatch** — In atomic ingestion

### Medium
1. **ColBERT disable** — Redundant with downstream cross-encoder (unanimous agreement)
2. **Query classification underutilized** — Returns types but not used for signal weight adjustment
3. **Entity vector field underutilized** — Qdrant entity field exists but may be empty/wrong type
4. **RRF k parameter** — Currently 60, recommended 30-40 for sharper rank separation

### Low
1. **400+ non-source files** — AI artifacts, reports, archives cluttering repo
2. **100+ cdx-outputs context files** — Development session logs, not production code
3. **Multiple Neo4j schema DDL files** — No clear canonical (v2.0, v2.1, v2.2)
4. **K8s Qdrant image version mismatch** — With docker-compose

---

## Development History (Phased)

### Phase 1-3: Core GraphRAG
- Graph schema, entity extraction, cross-doc linking
- Multi-model consensus reviews (o3, gpt-5.1-codex, gemini-3-pro)

### Phase 4: Advanced Query Features
- ColBERT integration, graph expansion, advanced Cypher templates

### Phase 5: Integration & Deployment
- Docker, K8s, monitoring, CI/CD

### Phase 6: Auto-Ingestion
- Redis Streams queue, saga pattern, reconciliation
- First production document ingested (2.8MB → 3,621 nodes, 268 vectors)

### Phase 7: Enhancement
- 7E: Qwen3 reranker integration, hybrid BM25/RRF/rerank
- 7D: Schema v2.2 migration
- Epoch-based cache invalidation (Redis)

---

## Provider Abstraction

### ProviderFactory Pattern
Central dispatch for embedding and rerank providers, ENV-selectable:
- Docker-compose friendly configuration without code changes
- Extensive aliasing support

### Supported Embedding Providers
- qwen3-embedding-0.6b (local)
- bge-m3-service (local)
- snowflake-arctic-service (local)
- voyage-ai (API)
- jina-ai (API)

### Supported Rerank Providers
- local-reranker-service (Qwen3-Reranker-4B)
- mxbai-reranker (local)
- jina-ai (API)

---

## Configuration Files

### Key Config Files
- `config/development.yaml` — Development settings
- `config/production.yaml` — Production settings
- `config/embedding_profiles.yaml` — All embedding model profiles
- `config/feature_flags.json` — Feature flag toggles
- `config/alloy/config.alloy` — Telemetry collector config

### Environment Files
- `.env` — Local development
- `.env.production` — Production (contains hardcoded New Relic key)
- `.env.docker` — Docker Compose
- `.env.example` — Template

---

## CI/CD Pipeline

### GitHub Actions
- CI workflow: `.github/workflows/ci.yml`
- Phase-specific test runs via Makefile

### Pre-commit Hooks
- Black 25.9.0 (formatting)
- Ruff v0.14.1 (linting)
- isort 7.0.0 (import sorting)
- detect-secrets (security scanning)
- gitlint (commit message validation)

### Makefile Targets
- `make up/down` — Docker Compose
- `make test-phase-N` — Phase-specific tests
- Neo4j Cypher MCP server controls

---

## Documentation Artifacts

### Architecture Docs
- `docs/architecture/2026-03-04-end-to-end-architecture.md` — Full E2E architecture
- `docs/architecture/retrieval-path-030826.md` — Retrieval path analysis
- `architecture_diagram.svg` — Visual pipeline diagram

### Bugfix Docs
- `docs/bugfix/` — 5 bugfix reports (tokenizer, neo4j planner, explainguard, reranker)

### CDX Outputs (100+ files)
- Development session contexts
- Multi-model consensus analyses
- Phase planning documents
- Implementation diffs

### API Contracts
- `docs/api-contracts.md` — API specifications
- `docs/app-spec-phase6.md` — Phase 6 app spec

---

## Scripts Directory

| Script | Lines | Purpose |
|--------|-------|---------|
| `apply_complete_schema_v2_1.py` | 110 | Apply Neo4j schema v2.1 |
| `attribution_test.py` | 140 | 3-run attribution test (legacy vs graph_assisted) |
| `backfill_cross_doc_edges.py` | 1,356 | Phase 3.5 cross-document linking |
| `backfill_doc_title_vectors.py` | 180 | Backfill doc_title vectors |
| `backfill_document_tokens.py` | 230 | Backfill Document.token_count |

---

## Sample Ingest Data

55 git-ignored files in `docs/sample_ingest/`:
- WEKA S3 protocol documentation
- SMB/NFS support docs
- Monitoring and alerting guides
- Operation guides (background tasks, alerts, events)
- Golden test datasets (sparse_vs_bm25.yaml)

---

## Recommendations

### Immediate (Phase 0)
1. **Wire in auth and rate-limiting** — Modules exist but not integrated
2. **Remove hardcoded credentials** from `.env.production`
3. **Disable ColBERT** — Redundant with cross-encoder
4. **Reduce RRF k from 60 to 30-40** — Sharper rank separation

### Short-term (Phase 1)
1. **Fix BGE reranker batching** — 300-500ms latency reduction
2. **Fix atomic pipeline Entity-to-Entity relationships** — CONTAINS_STEP edges
3. **Implement query-type-aware signal weights** — Use classification output
4. **Clean up 400+ non-source files** — Move to separate archive repo

### Medium-term (Phase 2)
1. **Populate Qdrant entity vector field** — With BGE-M3 sparse vectors
2. **Resolve schema DDL ambiguity** — Establish canonical version
3. **Fix K8s/docker-compose Qdrant version mismatch**

### Long-term (Phase 3+)
1. **Implement graph channel as reranker** — Not independent channel
2. **Add multi-hop graph semantics** — Currently shallow (Document→Section→Entity)
3. **Implement per-channel quotas** — Prevent rank hijacking

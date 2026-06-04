# 04 — ASCII Architecture Diagrams

## Diagram 1: Top-Level Repo/Module Map

```
wekadocs-matrix/
│
├── src/                              [SOURCE CODE — 137 Python files]
│   ├── shared/                       [FOUNDATION — no internal deps]
│   │   ├── config.py                 Central config loader (YAML + env)
│   │   ├── connections.py            Connection pools (Neo4j, Qdrant, Redis)
│   │   ├── schema.py                 Neo4j schema DDL
│   │   ├── models.py                 Core Pydantic models
│   │   ├── observability/            Tracing, metrics, logging, diagnostics
│   │   └── resilience/               Circuit breaker
│   │
│   ├── providers/                    [PROVIDER ABSTRACTION — no internal deps]
│   │   ├── factory.py                ProviderFactory (ENV-selectable)
│   │   ├── settings.py               EmbeddingSettings dataclass
│   │   ├── embeddings/               Jina, ST, Arctic, Qwen3, Voyage, Chonkie
│   │   ├── rerank/                   Jina, local, noop
│   │   └── ner/                      GLiNER
│   │
│   ├── ingestion/                    [INGESTION PIPELINE]
│   │   ├── atomic.py                 AtomicIngestionCoordinator (ACTIVE)
│   │   ├── worker.py                 Async Redis-polling worker (ACTIVE)
│   │   ├── build_graph.py            Legacy GraphBuilder (DEPRECATED)
│   │   ├── chunk_assembler.py        StructuredChunker, SemanticChunkerAssembler
│   │   ├── structural_edges.py       NEXT_CHUNK, PARENT_HEADING edges
│   │   ├── parsers/                  markdown-it-py (active), markdown (deprecated), HTML
│   │   ├── extract/                  Commands, configs, procedures, GLiNER NER
│   │   └── auto/                     Queue, watchers, reaper, CLI, service
│   │
│   ├── query/                        [RETRIEVAL PIPELINE]
│   │   ├── hybrid_retrieval.py       HybridRetriever (dense+sparse+BM25+graph)
│   │   ├── fusion_pipeline.py        RRF + weighted fusion
│   │   ├── rerank_pipeline.py        Cross-encoder reranking
│   │   ├── response_builder.py       Response assembly
│   │   ├── context_assembly.py       Context stitching
│   │   ├── graph_pipeline.py         Graph-aware retrieval
│   │   ├── traversal.py              Neo4j graph traversal
│   │   ├── templates/                Cypher query templates
│   │   └── processing/               Query disambiguation
│   │
│   ├── mcp_server/                   [MCP PROTOCOL SERVER]
│   │   ├── main.py                   FastAPI HTTP server (port 8000)
│   │   ├── mcp_app.py                MCP server factory + tool implementations
│   │   ├── query_service.py          QueryService (hybrid search integration)
│   │   ├── stdio_server.py           STDIO transport (Claude Desktop)
│   │   ├── scratch_store.py          In-memory scratch storage
│   │   ├── webhooks.py               GitHub/Notion/Confluence webhooks
│   │   └── security/                 JWT auth + rate limiter (DEAD — not wired)
│   │
│   ├── services/                     [SERVICE LAYER]
│   │   ├── graph_service.py          Graph query execution
│   │   ├── text_service.py           Text retrieval
│   │   ├── context_assembler.py      Context assembly + summarization
│   │   ├── context_budget_manager.py Token/byte budgeting
│   │   ├── cross_doc_linking.py      Cross-document relationship discovery
│   │   └── delta_cache.py            Session delta caching
│   │
│   ├── neo/                          [NEO4J-SPECIFIC]
│   │   ├── schema.py                 RELATIONSHIP_TYPES (24 types)
│   │   ├── schema_validator.py       Schema validation
│   │   └── [dead modules: contract_checks, entity_normalization, explain_guard, etc.]
│   │
│   ├── connectors/                   [EXTERNAL SYSTEM CONNECTORS]
│   │   ├── github.py                 GitHub repo sync
│   │   ├── manager.py                ConnectorManager
│   │   └── circuit_breaker.py        Circuit breaker pattern
│   │
│   ├── monitoring/                   [OBSERVABILITY]
│   │   ├── health.py                 HealthChecker
│   │   ├── metrics.py                MetricsCollector
│   │   └── slos.py                   SLO tracking
│   │
│   ├── clients/                      [EMBEDDING HTTP CLIENTS]
│   │   ├── embedding_client.py       Unified gateway client
│   │   ├── qwen3_embedding_client.py Qwen3 client
│   │   └── snowflake_embedding_client.py Snowflake Arctic client
│   │
│   ├── ops/                          [OPERATIONS]
│   │   ├── optimizer.py              Query optimizer (DEAD)
│   │   └── warmers/                  Query warmer (DEAD)
│   │
│   ├── registry/                     [VECTOR INDEX REGISTRY] (DEAD)
│   ├── learning/                     [LEARNING SYSTEM] (DEAD)
│   └── tools/                        CLI tools
│
├── tests/                            [TEST SUITE — ~170 Python files]
│   ├── conftest.py                   Session-scoped fixtures
│   ├── unit/                         26 unit tests
│   ├── integration/                  22 integration tests
│   ├── e2e/                          Golden set baseline
│   ├── e2e_v22_prod/                 v2.2 production validation (spec-only)
│   ├── contracts/                    Contract tests
│   ├── query/                        Query engine tests
│   ├── providers/                    Provider tests
│   ├── ingestion/                    Ingestion tests
│   ├── mcp_server_tests/             MCP server tests
│   └── p*_t*_test.py                 Phase-organized tests (24 files)
│
├── scripts/                          [CLI SCRIPTS — 60+ files]
│   ├── init_schema.py                Schema initialization
│   ├── ingestctl                     ingestctl CLI wrapper
│   ├── eval/                         Evaluation harness
│   ├── ci/                           CI scripts (phase gate, dead imports)
│   ├── neo4j/                        Neo4j schema DDL files, snapshots
│   ├── dev/                          Development scripts
│   └── migration/                    Migration Cypher
│
├── docker/                           [DOCKERFILES — 4 files]
│   ├── mcp-server.Dockerfile         MCP server (port 8000)
│   ├── ingestion-worker.Dockerfile   Ingestion worker
│   ├── ingestion-service.Dockerfile  Auto-ingest service (port 9108)
│   └── mxbai-reranker.Dockerfile     Cross-encoder reranker (CUDA)
│
├── deploy/                           [DEPLOYMENT]
│   ├── k8s/base/                     Kustomize base (12 resources)
│   ├── scripts/                      Canary, blue-green, backup, DR scripts
│   ├── monitoring/                   Prometheus alerts, Grafana dashboards
│   └── DR-RUNBOOK.md                 Disaster recovery runbook
│
├── infra/                            [INFRASTRUCTURE AS CODE]
│   └── terraform/                    GCP LGTM VM (Loki, Tempo, Mimir, Grafana)
│
├── config/                           [APPLICATION CONFIG]
│   ├── development.yaml              Master config (566 lines)
│   ├── production.yaml               Duplicate of development.yaml
│   ├── embedding_profiles.yaml       Named embedding profiles
│   ├── feature_flags.json            Feature flags
│   ├── alloy/config.alloy            Grafana Alloy telemetry
│   └── grafana/                      Dashboard provisioning
│
├── services/                         [STANDALONE ML SERVICES]
│   ├── gliner-ner/                   GLiNER NER (port 9002)
│   └── mxbai-reranker/               Cross-encoder reranker (port 9006)
│
├── monitoring/                       [MONITORING CONFIG]
│   ├── alerts/                       Phase 7E SLO alerts
│   └── dashboards/                   Phase 7E ingestion/retrieval/SLO dashboards
│
├── reports/                          [GENERATED ARTIFACTS — 355+ files]
│   ├── phase-{1-7}/                  Phase test reports
│   ├── ingest/<uuid>/                Per-job reports
│   └── cleanup/                      Cleanup reports
│
├── data/                             [DATA DIRECTORIES]
│   ├── documents/                    Ingested documents
│   ├── ingest/                       Input directory for file watcher
│   └── samples/                      Sample data
│
├── docker-compose.yml                [DOCKER COMPOSE — 497 lines]
├── Makefile                          Build/test commands
├── requirements.txt                  ~80 Python dependencies
├── pytest.ini                        Test configuration
├── .pre-commit-config.yaml           Pre-commit hooks
└── .github/workflows/ci.yml          CI/CD pipeline
```

## Diagram 2: Runtime/Service Topology

```
                    +-----------------------+
                    |   AI Clients          |
                    | (Claude Desktop, etc) |
                    +-----------+-----------+
                                |
                    MCP Streamable HTTP (HTTPS)
                                |
                    +-----------v-----------+
                    |   NGINX Ingress       |
                    |   TLS termination     |
                    |   wekadocs.example.com|
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------v---------+               +---------v---------+
    |  MCP Server Blue  |               |  MCP Server Green |
    |  (3 replicas)     |               |  (0 replicas)     |
    |  FastAPI + MCP    |               |  (blue/green      |
    |  :8000            |               |   standby)        |
    +---------+---------+               +---------+---------+
              |                                   |
              +-----------------+-----------------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------v---------+               +---------v---------+
    |  Ingestion Worker  |               | Ingestion Service  |
    |  (2 replicas)      |               |  (1 replica)       |
    |  Async Redis poller|               |  FastAPI :9108     |
    |  :no public port   |               |  File watcher      |
    +---------+---------+               +---------+---------+
              |                                   |
    +---------v---------+               +---------v---------+
    |  Neo4j 5.15        |               |   Qdrant 1.7.4    |
    |  Community Edition  |               |  Vector DB        |
    |  bolt://:7687       |               |  :6333/:6334      |
    |  20Gi PVC           |               |  10Gi PVC         |
    |  APOC + GDS enabled |               |  4 dense + 4      |
    +---------+---------+               |    sparse vectors   |
              |                         +---------+---------+
    +---------v---------+                       |
    |     Redis 7.2      |                       |
    |  :6379              |                       |
    |  5Gi PVC            |                       |
    |  db=0: jobs         |                       |
    |  db=1: tests        |                       |
    +---------------------+                       |
                                                  |
                    +-----------------------------+
                    |
          +---------v---------+
          |  mxbai-reranker   |
          |  CUDA 12.6        |
          |  :9006            |
          |  mxbai-rerank-    |
          |  large-v2         |
          +-------------------+
                    |
          +---------v---------+
          |  GPU Gateway      |
          |  10.25.0.50:8080  |
          |  or               |
          |  host.docker.     |
          |  internal:8080    |
          |                   |
          |  Qwen3-Embedding  |
          |  SPLADEv3         |
          |  ColBERTv2        |
          |  Qwen3-Reranker   |
          |  GLiNER           |
          |  Qwen2.5-1.5B     |
          +-------------------+

    External (optional):
    +---------+---------+
    |  Jina AI API      |
    |  Voyage AI        |
    |  HuggingFace Hub  |
    +-------------------+

    Observability (separate GCE VM via Tailscale):
    +---------+---------+
    |  Grafana :3000    |
    |  Loki    :3100    |
    |  Tempo   :3200    |
    |  Mimir   :9009    |
    |  Alloy   :4317    |
    +-------------------+
```

## Diagram 3: Primary Request/Data Flow

```
  INGESTION FLOW (Async)                    RETRIEVAL FLOW (Per-Request)
  =====================                     ==========================

  File drop / Webhook                       MCP Client request
        |                                         |
        v                                         v
  +-------------+                         +-------------+
  | File Watcher|                         | MCP Server  |
  | (service.py)|                         | (main.py)   |
  +------+------+                         +------+------+
         |                                         |
         v                                         v
  +-------------+                         +-------------+
  | Redis Queue |                         | QueryService|
  | (brpoplpush)|                         | .search_    |
  +------+------+                         | sections_   |
         |                                | light()     |
         v                                +------+------+
  +-------------+                                         |
  | Worker      |                                         v
  | (worker.py) |                          +--------------+---------------+
  |             |                          |  HybridRetriever              |
  | process_    |                          |  .retrieve()                  |
  | job(job)    |                          +--------------+---------------+
  |             |                                         |
  | parse_file  |                    +-------------------+-------------------+
  | + extract   |                    |                   |                   |
  | + chunk     |                    v                   v                   v
  | + embed     |          +--------+-------+   +--------+-------+   +------+------_
  | + write     |          | Qdrant Multi-  |   | Neo4j BM25     |   | Entity/     |
  |             |          | Vector Search  |   | Full-Text      |   | Structural  |
  v             v          | (dense+sparse  |   | Search         |   | Boost       |
  +-------------+          |  +ColBERT)     |   +--------+-------+   +------+------_
  | AtomicIngestion     |           |                   |                   |
  | Coordinator         |           v                   v                   v
  | (atomic.py)         |      +----+-----+     +------+------+   +------+------_
  |                     |      | RRF Fusion|     | Graph       |   | Signal Pool   |
  | _execute_atomic_    |      | (k=60)    |     | Expansion   |   | (diversity)   |
  | saga()              |      +----+-----+     +------+------+   +------+------_
  |                     |           |                   |                   |
  | 1. Neo4j upsert     |           v                   v                   v
  | 2. Qdrant upsert    |      +----+-----+     +------+------+   +------+------_
  | 3. Commit (if Qdrant|      | Rerank  |     | Cross-    |   | Context       |
  |    succeeds)         |      | Pipeline|     | Doc Link  |   | Assembly      |
  | 4. Cross-doc links   |      |(Qwen3/  |     | Edges     |   | + Budget Mgr  |
  +-------------+        |      | mxbai)  |     +-----------+   +------+------_
                         +--------------+                     |
                              |                                 v
                              v                          +------+------+
                        +-----+-----+                     | Response  |
                        | Ack/Nack  |                     | Builder   |
                        +-----------+                     +-----------+
```

## Diagram 4: Dependency/Coupling Map

```
  +---------------------------------------------------+
  |              NO INTERNAL DEPS (Foundation)         |
  |                                                   |
  |  +-------------+    +------------------------+    |
  |  | shared/     |    | providers/             |    |
  |  | config.py   |    | factory.py             |    |
  |  | connections |    | settings.py            |    |
  |  | schema.py   |    | embeddings/*.py        |    |
  |  | models.py   |    | rerank/*.py            |    |
  |  | observability|   | ner/*.py               |    |
  |  | resilience/ |    | tokenizer_service.py   |    |
  |  +-------------+    +------------------------+    |
  +----------------|----------------|----------------+
                   |                |
                   v                v
  +---------------------------------------------------+
  |              MID LAYER (Services/Query)           |
  |                                                   |
  |  +-------------+    +------------------------+    |
  |  | services/   |    | query/                 |    |
  |  | graph_svc   |    | hybrid_retrieval.py    |    |
  |  | text_svc    |    | fusion_pipeline.py     |    |
  |  | context_    |    | rerank_pipeline.py     |    |
  |  | assembler   |    | response_builder.py    |    |
  |  | budget_mgr  |    | context_assembly.py    |    |
  |  | cross_doc   |    | graph_pipeline.py      |    |
  |  +-------------+    | traversal.py           |    |
  |                     | templates/*.cypher     |    |
  |                     +------------------------+    |
  +----------------|----------------|----------------+
                   |                |
                   v                v
  +---------------------------------------------------+
  |              TOP LAYER (Entry Points)             |
  |                                                   |
  |  +-------------+    +------------------------+    |
  |  | mcp_server/ |    | ingestion/             |    |
  |  | main.py     |    | atomic.py              |    |
  |  | mcp_app.py  |    | worker.py              |    |
  |  | query_svc   |    | chunk_assembler.py     |    |
  |  | webhooks    |    | parsers/               |    |
  |  | scratch     |    | extract/               |    |
  |  +-------------+    | auto/                  |    |
  |                     +------------------------+    |
  +----------------|----------------|----------------+
                   |                |
                   v                v
  +---------------------------------------------------+
  |              EXTERNAL SYSTEMS                      |
  |                                                   |
  |  Neo4j  |  Qdrant  |  Redis  |  GPU Gateway  |
  |  :7687  |  :6333   |  :6379  |  :8080        |
  +---------------------------------------------------+

  COUPLING POINTS (potential risk areas):
  - shared/connections.py: imported by 15+ modules
  - shared/config.py: imported by 20+ modules
  - providers/factory.py: imported by ingestion + query + mcp_server
  - src/ingestion/build_graph.py: DEPRECATED but imported by ACTIVE atomic.py
  - src/neo/contract_checks.py: DEAD but imported by ACTIVE worker.py
  - src/mcp_server/security/*: DEAD but represents security gap
```

## Diagram 5: Active vs Legacy Path Map

```
  INGESTION PATHS:

  OLD (DEPRECATED):                          NEW (ACTIVE):
  +------------------+                       +-------------------+
  | build_graph.py   |                       | atomic.py         |
  | GraphBuilder     |---DEPRECATED--\       | AtomicIngestion   |
  | ingest_document()|                |       | Coordinator       |
  +------------------+                |       +-------------------+
                                      |             |
  +------------------+                |             v
  | orchestrator.py  |---DEAD-------/       +-------------------+
  | (never used)     |                       | worker.py         |
  +------------------+                       | Async Redis poller|
                                              +-------------------+
  +------------------+
  | api.py           |---DEAD
  | (test facade)    |
  +------------------+

  LEGACY FALLBACK:                    ACTIVE PREFERRED:
  +------------------+                +-------------------+
  | parsers/         |                | parsers/          |
  | markdown.py      |---DEPRECATED-->| markdown_it_      |
  | (custom parser)  |                | parser.py         |
  +------------------+                +-------------------+
                                       | parsers/          |
                                       | html.py           |
                                       +-------------------+

  WATCHER:
  +------------------+                +-------------------+
  | watcher.py       |---DEAD-------->| watchers.py       |
  | (singular)       |                | (plural, updated) |
  +------------------+                +-------------------+

  MCP SERVER PATHS:

  PRODUCTION (ACTIVE):                  DEVELOPMENT (STANDALONE):
  +------------------+                  +-------------------+
  | main.py          |                  | stdio_server.py   |
  | FastAPI :8000    |                  | STDIO transport   |
  | Streamable HTTP  |                  | (Claude Desktop)  |
  +------------------+                  +-------------------+
          |
          v
  +------------------+
  | mcp_app.py       |
  | build_mcp_server |
  | Tool definitions |
  +------------------+

  SECURITY (GAP):
  +------------------+
  | security/        |---DEAD — not wired into main.py
  | auth.py          |   JWT auth exists but unused
  | rate_limiter.py  |   Rate limiting exists but unused
  +------------------+
```

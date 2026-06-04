# 03 — Architecture

## Narrative Architecture Explanation

WekaDocs-Matrix is a **GraphRAG documentation intelligence system** that transforms WEKA distributed file system documentation into a searchable, queryable knowledge base. It operates through two distinct pipelines:

### Ingestion Pipeline (Offline/Async)

Documents (Markdown/HTML files) are dropped into an ingest directory or pushed via webhook. A file system watcher detects new files and enqueues them to a Redis job queue. An async worker polls Redis, parses documents, extracts entities, chunks content, computes embeddings, and writes to both Neo4j (graph structure) and Qdrant (vector embeddings) in a saga-coordinated atomic transaction. Cross-document relationships are discovered and stored as RELATED_TO edges.

### Retrieval Pipeline (Online/Per-Request)

AI coding assistants connect via the Model Context Protocol (MCP) over Streamable HTTP. When a user asks a question, the system reformulates the query, computes dense/sparse/ColBERT embeddings, performs multi-vector similarity search in Qdrant and BM25 search in Neo4j, fuses results using RRF (Reciprocal Rank Fusion), applies graph-based signal boosting, reranks with a cross-encoder, expands via graph traversal, and assembles a context-bounded response.

## Major Modules and Responsibilities

### `src.shared/` — Foundation Layer (No Internal Dependencies)

**`config.py`** — Central configuration loader. Loads YAML config + environment variables via Pydantic Settings. Defines Config, Settings, EmbeddingConfig, EmbeddingProfileDefinition, EmbeddingPlan, HybridSearchConfig, QdrantVectorConfig, RerankerConfig, CrossDocLinkingConfig, ValidatorConfig, ReconciliationConfig. Single source of truth for all runtime configuration.

**`connections.py`** — Connection pool manager. Manages Neo4j driver, Qdrant client (CompatQdrantClient wrapper), and Redis client. Singleton pattern with global `_connection_manager` instance. Provides `initialize_connections()`, `close_connections()`, `get_connection_manager()`.

**`schema.py`** — Neo4j schema DDL. Defines `create_schema()`, `create_vector_indexes()`, `verify_schema()`, `drop_schema()`. Creates 16 unique constraints, 4 vector indexes, 3 fulltext indexes across 15+ node labels.

**`models.py`** — Core Pydantic data models shared across the system.

**`chunk_utils.py`** — Chunk ID generation, schema validation, metadata creation.

**`embedding_fields.py`** — Embedding field canonicalization and validation.

**`observability/`** — OpenTelemetry tracing, Prometheus metrics, structlog logging, retrieval diagnostics, exemplar collection.

**`resilience/`** — Circuit breaker pattern implementation.

### `src.providers/` — Provider Abstraction Layer (No Internal Dependencies)

**`factory.py`** — `ProviderFactory` creates embedding and rerank providers from environment configuration. Supports jina-ai, sentence-transformers, embedding-service, voyage-ai, snowflake-arctic-service, qwen3-triton-service for embeddings. Supports jina-ai, local-reranker-service, noop for reranking.

**`settings.py`** — `EmbeddingCapabilities` and `EmbeddingSettings` (frozen dataclass). Defines model dimensions, task types, tokenizer requirements.

**`tokenizer_service.py`** — Token counting service for input budget management.

**`embeddings/`** — Individual embedding provider implementations:
- `jina.py` — Jina AI embeddings
- `sentence_transformers.py` — Local sentence-transformers
- `snowflake_arctic.py` — Snowflake Arctic embeddings
- `qwen3_triton.py` — Qwen3 via Triton inference server
- `voyage.py` — Voyage AI embeddings
- `chonkie_adapter.py` / `arctic_chonkie_adapter.py` / `qwen3_chonkie_adapter.py` — Chonkie semantic chunking adapters
- `embedding_service.py` — Unified embedding service wrapper

**`rerank/`** — Individual rerank providers:
- `jina.py` — Jina AI reranker
- `local_reranker_service.py` — Local cross-encoder (mxbai-rerank-large-v2)
- `noop.py` — No-op reranker (disabled)

**`ner/`** — GLiNER named entity recognition service.

### `src.ingestion/` — Document Ingestion Pipeline

**`atomic.py`** — `AtomicIngestionCoordinator`. The production ingestion path. Saga-coordinated Neo4j + Qdrant writes. Guarantees: Neo4j commits only if Qdrant succeeds; compensates on failure. Methods: `ingest_document_atomic()`, `_prepare_ingestion()`, `_compute_embeddings()`, `_execute_atomic_saga()`.

**`worker.py`** — Async ingestion worker. Polls Redis for jobs. Runs saga-coordinated ingestion. Background job reaper for stale job recovery.

**`chunk_assembler.py`** — `get_chunk_assembler()`, `StructuredChunker`, `SemanticChunkerAssembler`, `GreedyCombiner`. Merges/splits sections into chunks.

**`structural_edges.py`** — `build_structural_edges_in_tx()`. Creates NEXT_CHUNK, PARENT_HEADING, CHILD_OF, PARENT_OF, NEXT relationships atomically.

**`parsers/`** — Document parsers:
- `markdown_it_parser.py` — Primary markdown parser (markdown-it-py AST)
- `markdown.py` — Legacy markdown parser (deprecated)
- `html.py` — HTML parser
- `notion.py` — Notion parser (dead)

**`extract/`** — Entity extraction:
- `commands.py` — Extracts command entities
- `configs.py` — Extracts configuration entities
- `procedures.py` — Extracts procedure entities
- `ner_gliner.py` — GLiNER-based named entity recognition
- `references.py` — Hyperlink and cross-document reference extraction

**`auto/`** — Auto-ingestion subsystem:
- `queue.py` — Redis-backed job queue with DLQ support
- `watchers.py` — File system watcher
- `reaper.py` — Stale job recovery
- `progress.py` — Redis-stream progress tracking
- `service.py` — FastAPI HTTP service (port 9108)
- `cli.py` — ingestctl CLI tool

### `src.query/` — Retrieval Pipeline

**`hybrid_retrieval.py`** — `HybridRetriever`. Phase 7E main retrieval engine. Combines dense vector, sparse lexical, BM25, graph signals. Methods: `retrieve()`, `rrf_fusion()`, `weighted_fusion()`.

**`hybrid_search.py`** — `HybridSearchEngine`, `QdrantVectorStore`, `Neo4jVectorStore`. Legacy search engine.

**`fusion_pipeline.py`** — Score fusion (RRF with k=60, weighted).

**`graph_pipeline.py`** — Graph-aware retrieval integration.

**`graph_expansion.py`** — Graph expansion (dead — no callers).

**`ranking.py`** — Result ranking.

**`rerank_pipeline.py`** — Cross-encoder reranking pipeline.

**`response_builder.py`** — `Response`, `StructuredResponse`, `build_response()`. Assembles final answers.

**`context_assembly.py`** — `ContextAssembler`. Stitches retrieved chunks into context.

**`traversal.py`** — `TraversalService`. Neo4j graph traversal.

**`planner.py`** — `QueryPlanner`. Query planning.

**`session_tracker.py`** — `SessionTracker`. Multi-turn conversation sessions.

**`signal_pool.py`** — Signal-diverse candidate selection.

**`structural_retrieval.py`** — Structural boosting based on query type.

**`entity_extraction.py`** — `EntityExtractor`. Query entity extraction.

**`expansion_pipeline.py`** — Bounded adjacency expansion.

**`templates/`** — Cypher query templates:
- `search.cypher` — Section search
- `traverse.cypher` — Graph traversal
- `explain.cypher` — Explanation queries
- `troubleshoot.cypher` — Troubleshooting path
- `compare.cypher` — Comparison queries
- `advanced/` — Dependency chain, temporal, troubleshooting path, impact assessment, comparison

### `src.mcp_server/` — MCP Protocol Server

**`main.py`** — FastAPI HTTP server. Mounts `/_mcp` as StreamableHTTPSessionManager. Routes: `/health`, `/ready`, `/metrics`, `/_mcp`, `/webhooks/*`.

**`mcp_app.py`** — Shared MCP server factory. `build_mcp_server()`. Registers tools: `kb_search`, `kb_read_excerpt`, `kb_read_section`, `kb_read_document`, `kb_traverse`, `kb_search_documentation`, `kb_evidence_pack`, `kb_followup`, `kb_scratch_read/write/delete`, `kb_session_info`, `kb_diagnostic_trace`, `kb_diagnostic_context`.

**`query_service.py`** — `QueryService`. Integrates hybrid search, ranking, response building. Handles query rewriting, embedder/reranker caching, session tracking.

**`scratch_store.py`** — In-memory scratch storage for multi-turn conversations.

**`retrieval_trace.py`** — Retrieval trace builder for observability.

**`webhooks.py`** — GitHub/Notion/Confluence webhook handlers.

**`security/`** — JWT auth and rate limiting (dead — not wired in).

### `src.services/` — Service Layer

Thin wrappers that MCP handlers call:
- **`graph_service.py`** — `GraphService`. Graph query execution.
- **`text_service.py`** — `TextService`. Text retrieval.
- **`context_assembler.py`** — `ContextAssemblerService`, `SummarizationService`.
- **`context_budget_manager.py`** — `ContextBudgetManager`, `BudgetExceeded`. Token/byte budgeting.
- **`cross_doc_linking.py`** — `CrossDocLinker`. Cross-document relationship discovery.
- **`cross_doc_edge_model.py`** — Cross-document edge scoring model.
- **`delta_cache.py`** — `SessionDeltaCache`. Delta caching for sessions.

### `src.neo/` — Neo4j-Specific Logic

- **`schema.py`** — `RELATIONSHIP_TYPES`. Canonical relationship allow-list (24 types).
- **`structural_builder.py`** — Structural edge building (dead — circular import).
- **`entity_normalization.py`** — Entity name normalization (dead).
- **`explain_guard.py`** — Cypher query safety guard (dead).
- **`defensive_query.py`** — Defensive query construction (dead).
- **`contract_checks.py`** — `GraphContractChecker` (dead but imported by worker.py).
- **`schema_validator.py`** — Schema validation.
- **`graph_enhancements.py`** — Graph enhancement utilities (dead).
- **`health.py`** — Neo4j health checks (dead).

### `src.connectors/` — External System Connectors

- **`base.py`** — `BaseConnector`, `ConnectorConfig`, `ConnectorStatus`.
- **`github.py`** — `GitHubConnector`. GitHub repo sync via API.
- **`manager.py`** — `ConnectorManager`. Manages multiple connectors, starts/stops polling.
- **`queue.py`** — `IngestionQueue`. Redis-based ingestion queue.
- **`circuit_breaker.py`** — `CircuitBreaker`, `CircuitBreakerState`.

### `src.monitoring/` — Observability

- **`health.py`** — `HealthChecker`. Comprehensive startup health checks (Neo4j schema, Qdrant dimensions, embedding config).
- **`metrics.py`** — `MetricsCollector`, `MetricsAggregator`. Chunk/retrieval/ingestion metrics.
- **`slos.py`** — SLO definitions and tracking.

## Runtime/Deployment Topology

```
                    +-------------------+
                    |   AI Clients      |
                    | (Claude, etc.)    |
                    +--------+----------+
                             |
                    MCP Streamable HTTP
                             |
                    +--------v----------+
                    |   NGINX Ingress   |
                    | (TLS termination) |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                              |
    +---------v---------+          +---------v---------+
    |  MCP Server Blue  |          |  MCP Server Green |
    |  (3 replicas)     |          |  (0 replicas)     |
    |  Port 8000        |          |  (standby)        |
    +---------+---------+          +---------+---------+
              |                              |
              +--------------+---------------+
                             |
              +--------------+--------------+
              |                              |
    +---------v---------+          +---------v---------+
    |  Ingestion Worker  |          |  Ingestion Service |
    |  (2 replicas)      |          |  (1 replica)       |
    |  Port (none)       |          |  Port 9108         |
    +---------+---------+          +---------+---------+
              |                              |
    +---------v---------+          +---------v---------+
    |   Neo4j           |          |     Qdrant         |
    |   5.15-community   |          |   1.7.4            |
    |   20Gi PVC         |          |   10Gi PVC         |
    +---------+---------+          +---------+---------+
              |                              |
    +---------v---------+          +---------v---------+
    |     Redis         |          |   mxbai-reranker   |
    |   7.2-alpine      |          |   CUDA 12.6        |
    |   5Gi PVC         |          |   GPU required     |
    +-------------------+          +-------------------+

    External GPU Gateway (10.25.0.50:8080):
    - Qwen3-Embedding-0.6B (dense, 1024-D)
    - SPLADEv3 (sparse)
    - ColBERTv2 (late-interaction)
    - Qwen3-Reranker-4B (cross-encoder)
    - GLiNER Medium v2.1 (NER)
    - Qwen2.5-1.5B-Instruct (query reformulation, not deployed)

    LGTM VM (GCE via Terraform):
    - Grafana (3000), Loki (3100), Tempo (3200), Mimir (9009)
    - Grafana Alloy (telemetry collector)
    - Accessed via Tailscale mesh
```

## Persistence and External Systems

### Databases

| System | Type | Role | Key Details |
|--------|------|------|-------------|
| Neo4j | Graph DB | Document structure, entity relationships, structural edges | 15+ labels, 24 relationship types, 16 constraints, 4 vector indexes (1024-D), 3 fulltext indexes |
| Qdrant | Vector DB | Multi-vector semantic search | Collection: `chunks_multi_bge_m3`. 4 dense vectors (content, title, doc_title, late-interaction/ColBERT). 4 sparse vectors (text-sparse, doc_title-sparse, title-sparse, entity-sparse). HNSW: m=48, ef_construct=256 |
| Redis | Cache/Queue | L1/L2 caching, ingestion job queue, progress tracking | db=0 for jobs, db=1 for tests. Keys: `{ns}:search:doc:*`, `{ns}:fusion:doc:*`, `{ns}:answer:doc:*`, `{ns}:vector:chunk:*`, `ingest:jobs:*`, `ingest:processing:*` |

### External Services

| System | Role | Access |
|--------|------|--------|
| Unified GPU Gateway | Serves all 6 AI models | `10.25.0.50:8080` (LAN) or `host.docker.internal:8080` (Docker) |
| Jina AI API | Alternative embedding/reranker provider | External API (requires JINA_API_KEY) |
| Voyage AI | Alternative embedding provider | External API (requires VOYAGE_API_KEY) |
| HuggingFace Hub | Model downloads | Cached in `hf-cache/` |
| GitHub API | Webhook-based doc sync | External API (requires GITHUB_WEBHOOK_SECRET) |

### Observability Stack

| System | Role | Port |
|--------|------|------|
| Prometheus | Metrics storage | 9090 |
| Loki | Log aggregation | 3100 |
| Tempo | Distributed tracing | 3200/4317/4318 |
| Mimir | Time-series metrics | 9009 |
| Grafana | Dashboards and alerts | 3000 |
| Grafana Alloy | Log/trace/metrics collection | 4317/4318 |

## Config/Auth/Secrets Handling

### Configuration Loading

1. **YAML config file** (single source of truth): `config/${ENV}.yaml` where ENV defaults to "development"
2. **Environment variables** (sensitive/runtime overrides): Loaded via Pydantic Settings
3. **Config path override**: `CONFIG_PATH` env var can override default location

### Environment Selection

```python
# config.py loads:
config_file = f"config/{settings.env}.yaml"  # e.g., config/development.yaml
```

### Secrets

| Secret | Source | Notes |
|--------|--------|-------|
| Neo4j password | `.env` / K8s secrets | Default: `change-me-in-production` |
| Redis password | `.env` / K8s secrets | Default: `testredis123` |
| JWT secret | `.env` / K8s secrets | HS256, 60-minute expiry |
| GitHub webhook secret | `.env` / K8s secrets | For webhook verification |
| Jina API key | Env var | Required for Jina embeddings |
| New Relic key | `.env.production` | **Hardcoded in file** (security concern) |

### Authentication

**Current state:** No authentication or rate limiting is wired into the MCP server. The `src/mcp_server/security/` module exists with JWT auth middleware and rate limiter, but both are marked DEAD and never imported by `main.py`.

**Risk:** The MCP server is publicly accessible without authentication or rate limiting.

### Feature Flags

Defined in `config/feature_flags.json`. Controls optional features like:
- Sparse vector enablement
- ColBERT vector enablement
- GLiNER NER enablement
- Cross-document linking enablement
- Reference extraction enablement
- Query rewriting enablement

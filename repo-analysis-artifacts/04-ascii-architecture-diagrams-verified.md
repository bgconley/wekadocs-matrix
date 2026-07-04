# 04 — ASCII Architecture Diagrams (Verified from Source)

> All diagrams derived from reading actual source files: `main.py`, `mcp_app.py`, `worker.py`, `atomic.py`, `hybrid_retrieval.py`, `connections.py`, `docker-compose.yml`.

---

## Diagram 1: Service Topology (from docker-compose.yml)

```
                    +-----------------------+
                    |   AI Clients          |
                    | (Claude Desktop, etc) |
                    +-----------+-----------+
                                |
                    Streamable HTTP (STDIO or HTTP)
                                |
                    +-----------v-----------+
                    |   MCP Server          |
                    |   FastAPI :8000       |
                    |   /_mcp (Streamable)  |
                    |   /mcp/* (legacy, off)|
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------v---------+               +---------v---------+
    |  Neo4j Enterprise  |               |   Qdrant 1.16     |
    |  :7474/:7687       |               |  :6333/:6334      |
    |  APOC + GDS        |               |  7 vector fields  |
    |  2560m heap        |               |  1536m mem        |
    +---------+---------+               +---------+---------+
              |                                   |
    +---------v---------+                         |
    |     Redis 7.2      |                         |
    |  :6379              |                         |
    |  db=0: jobs+cache   |                         |
    |  768mb maxmem       |                         |
    +---------------------+                         |
              |                                     |
              v                                     v
    +---------+---------+               +-----------+-----------+
    | Ingestion Worker  |               | Ingestion Service     |
    | Async brpoplpush  |               | File watcher :8081    |
    | AtomicIngestion   |               | (drops to data/ingest)|
    | Coordinator       |               +-----------------------+
    +-------------------+

  External ML Gateway (10.25.0.50:8080):
    Qwen3-Embedding  |  SPLADEv3  |  ColBERTv2  |  Qwen3-Reranker
    GLiNER NER (:9002)  |  mxbai-reranker (:9006)

  Observability (GCP LGTM via Tailscale):
    Grafana Alloy :4317 (OTLP) → Loki + Tempo + Mimir + Grafana
```

---

## Diagram 2: MCP Server Internal Architecture (from main.py + mcp_app.py)

```
  uvicorn → FastAPI (main.py)
      │
      ├─ /health              → HealthResponse (embedding capabilities)
      ├─ /ready               → ReadinessResponse (neo4j/qdrant/redis checks)
      ├─ /metrics             → JSON metrics (latency percentiles)
      ├─ /metrics/prometheus  → Prometheus text format
      ├─ /_mcp                → StreamableHTTPSessionManager (MCP protocol)
      ├─ /mcp/initialize      → deprecated, disabled by default
      ├─ /mcp/tools/list      → deprecated, disabled by default
      ├─ /mcp/tools/call      → deprecated, disabled by default
      └─ /webhooks            → GitHub/Notion/Confluence webhook router

  /_mcp → mcp_app.py (build_mcp_server)
      │
      ├─ MCP Server (mcp.server.lowlevel.server.Server)
      │   │
      │   ├─ kb_search              → QueryService.search_sections_light()
      │   │   ├─ _normalize_scope()
      │   │   ├─ _dedupe_by_doc()
      │   │   └─ Returns: passages with scores, cursor pagination
      │   │
      │   ├─ kb_read_excerpt        → ScratchStore.get() + evidence extraction
      │   │   ├─ Blended scoring: 70% retrieval + 30% lexical
      │   │   └─ Returns: quotes with confidence, parent_path, doc_tag
      │   │
      │   ├─ kb_search_evidence     → Deeper fetch (up to 150) + evidence pack
      │   │   └─ Graph expansion via MCP_EVIDENCE_GRAPH_EXPANSION_ENABLED
      │   │
      │   ├─ kb_traverse            → TraversalService.traverse()
      │   │   ├─ Neo4j Cypher traversal
      │   │   └─ Bi-directional, configurable depth
      │   │
      │   ├─ kb_create_scratch      → ScratchStore.put() (session-scoped)
      │   ├─ kb_delete_scratch      → ScratchStore.delete()
      │   ├─ kb_list_scratch        → ScratchStore.list()
      │   │
      │   └─ kb_write_trace         → RetrievalTraceBuilder + append_followup
      │
      └─ Services (imported by mcp_app.py)
          ├─ QueryService           → hybrid_retrieval.HybridRetriever
          ├─ ContextAssemblerService → context_assembly + SummarizationService
          ├─ ContextBudgetManager   → token/byte budget enforcement
          ├─ GraphService           → projection-only Cypher queries
          ├─ TextService            → section text fetching
          └─ TraversalService       → Neo4j graph traversal
```

---

## Diagram 3: Ingestion Data Flow (from worker.py + atomic.py)

```
  File Watcher (ingestion-service)          Ingestion Worker (worker.py)
  ─────────────────────────                 ───────────────────────
  drops file → data/ingest/                 async main():
      │                                         │
      ▼                                         ▼
  HTTP POST to Redis queue                      while not shutdown:
  (or file watch)                                   │
      │                                            ▼
      ▼                                     item = brpoplpush(timeout=1)
  ingestion-service                              │
  pushes job to                                  ▼
  Redis (LIST)                              process_job(job):
      │                                        ├─ read file content
      │                                        ├─ detect format (.md/.html)
      │                                        ▼
      │                              AtomicIngestionCoordinator.ingest_document_atomic()
      │                                        │
      ▼                                        ▼
  (optional)                                   Phase 1: Parse & Prepare
  HTTP callback                                    ├─ parse_file()
      │                                           ├─ extract entities (GLiNER + structural)
      │                                           ├─ chunk (semantic/structured)
      │                                           └─ embed (multi-vector via ProviderFactory)
      │                                              │
      ▼                                            ▼
  JobReaper (background task)              Phase 2: Pre-commit Validation
  reaps stale jobs (>600s)                       ├─ validate_pre_ingestion()
      │                                          ├─ dimension checks
      │                                          └─ schema v2.1 compliance
      │                                             │
      ▼                                            ▼
  stale_action: requeue/fail                   Phase 3: Saga Write
                                                   ├─ START saga
                                                   ├─ Neo4j tx: upsert Document/Chunk/Entity
                                                   │   + MENTIONS, NEXT_CHUNK, PARENT_HEADING
                                                   │   + RELATED_TO v2 (cross-doc links)
                                                   ▼
                                              Qdrant upsert (7 vector fields)
                                                   │
                                                   ├─ if Qdrant fails → rollback Neo4j
                                                   └─ if Neo4j committed but Qdrant fails → compensate
                                                   │
                                                   ▼
                                              Phase 4: Post-commit
                                                   ├─ _create_cross_doc_links()
                                                   │   (if corpus >= min_corpus_size)
                                                   ├─ ACK job
                                                   └─ IngestionRunStats tracking

  Structural Health Check (idle period):
      GraphContractChecker.find_documents_needing_repair()
      → ERROR if any docs found (atomic ingestion should prevent this)
```

---

## Diagram 4: Retrieval Pipeline (from hybrid_retrieval.py + mcp_app.py)

```
  kb_search() [mcp_app.py]
      │
      ▼
  QueryService.search_sections_light()
      │
      ▼
  HybridRetriever.retrieve() [hybrid_retrieval.py]
      │
      ├─ Step 1: Embed query (ProviderFactory → BGE-M3 / Qwen3 / Voyage)
      │   → QueryEmbeddingBundle (dense + sparse + colbert)
      │
      ├─ Step 2: Parallel Recall (QdrantMultiVectorRetriever)
      │   ├─ dense search (content vector)
      │   ├─ sparse search (text-sparse vector)
      │   ├─ colbert search (late-interaction)
      │   ├─ title-sparse search
      │   └─ entity-sparse search
      │
      ├─ Step 3: BM25 (Neo4j full-text index)
      │   └─ chunk_text_index_v2
      │
      ├─ Step 4: RRF Fusion [fusion_pipeline.py]
      │   ├─ RRF k=60 (recommended: reduce to 30-40)
      │   └─ Returns: fused scores + fusion_method metadata
      │
      ├─ Step 5: Entity/Structural Boost [structural_retrieval.py]
      │   ├─ MENTION boost (GLiNER entities)
      │   ├─ PARENT_HEADING boost (heading hierarchy)
      │   └─ Query-type-specific RRF weights
      │
      ├─ Step 6: Signal Pool [signal_pool.py]
      │   └─ 200-candidate diversity: consensus + per-signal unique + structural
      │
      ├─ Step 7: Reranking [rerank_pipeline.py]
      │   └─ Cross-encoder (Qwen3-Reranker-4B / mxbai / Jina)
      │       Input: top 50 candidates → re-ranked output
      │
      ├─ Step 8: Expansion [expansion_pipeline.py]
      │   ├─ Adjacency expansion (prev/next chunks)
      │   └─ Graph expansion (Neo4j neighbor chunks)
      │       └─ gated by: expand=True + neo4j available
      │
      ▼
  ChunkResult[] → back to mcp_app.py
      │
      ├─ _dedupe_by_doc() (max 1-5 per doc)
      ├─ Cursor pagination (base64 encoded offset)
      └─ Returns: results with scores, source, preview, scratch_uri
```

---

## Diagram 5: Module Dependency Graph (from actual imports)

```
  FOUNDATION (no internal deps)
  ─────────────────────────────
  shared/config.py          ← imported by 20+ modules
  shared/connections.py     ← imported by 15+ modules
  shared/models.py
  shared/schema.py
  shared/chunk_utils.py
  shared/embedding_fields.py
  shared/vector_utils.py
  shared/qdrant_schema.py
  shared/observability/     ← tracing, metrics, logging
  shared/resilience/        ← circuit breaker

  providers/factory.py      ← ProviderFactory (ENV-selectable dispatch)
  providers/settings.py     ← EmbeddingSettings
  providers/tokenizer_service.py
  providers/embeddings/     ← jina, st, arctic, qwen3, voyage, chonkie
  providers/rerank/         ← jina, local, noop
  providers/ner/            ← gliner_service (singleton + circuit breaker)

  │
  ▼
  MID LAYER (services + query)
  ────────────────────────────
  services/graph_service.py     ← Neo4j projection queries
  services/text_service.py      ← section text fetching
  services/context_assembler.py ← ContextAssembler + Summarization
  services/context_budget_manager.py ← token/byte budgeting
  services/cross_doc_linking.py ← Phase 3.5 RELATED_TO edges (1420 lines)
  services/delta_cache.py       ← session dedup cache

  query/hybrid_retrieval.py     ← HybridRetriever (main entry point)
  query/fusion_pipeline.py      ← RRF + weighted fusion
  query/rerank_pipeline.py      ← cross-encoder reranking
  query/expansion_pipeline.py   ← adjacency + graph expansion
  query/graph_pipeline.py       ← graph-aware retrieval
  query/traversal.py            ← TraversalService (Neo4j Cypher)
  query/retrieval_types.py      ← ChunkResult, FusionMethod, ExpandWhen
  query/vector_backends.py      ← QdrantMultiVectorRetriever
  query/signal_pool.py          ← diversity pool builder
  query/structural_retrieval.py ← entity/heading boost
  query/entity_extraction.py    ← entity extraction for queries
  query/processing/disambiguation.py ← query analysis
  query/query_intent.py         ← intent classification
  query/retrieval_plan.py       ← 4 named profiles (vector_only, etc.)
  query/templates/*.cypher      ← search, traverse, troubleshoot, etc.

  │
  ▼
  INGESTION
  ─────────
  ingestion/atomic.py           ← AtomicIngestionCoordinator (ACTIVE)
  ingestion/worker.py           ← async worker loop (ACTIVE)
  ingestion/saga.py             ← SagaContext, IngestionValidator (ACTIVE)
  ingestion/run_stats.py        ← IngestionRunStats (ACTIVE)
  ingestion/semantic_chunker.py ← semantic chunking (ACTIVE)
  ingestion/structural_edges.py ← NEXT_CHUNK, PARENT_HEADING (ACTIVE)
  ingestion/auto/               ← queue, reaper, service, CLI
  ingestion/parsers/            ← markdown_it_py (active), markdown (deprecated)
  ingestion/extract/            ← commands, configs, procedures, GLiNER

  ingestion/build_graph.py      ← LEGACY (DEPRECATED, lazily imported by atomic.py)
  ingestion/orchestrator.py     ← DEAD (never loaded)
  ingestion/api.py              ← DEAD (test facade only)
  ingestion/reconcile.py        ← DEAD (sync Qdrant, superseded by atomic)

  │
  ▼
  ENTRY POINTS
  ─────────────
  mcp_server/main.py            ← FastAPI entry point (ACTIVE)
  mcp_server/mcp_app.py         ← MCP server factory + tools (ACTIVE)
  mcp_server/query_service.py   ← QueryService wrapper (ACTIVE)
  mcp_server/stdio_server.py    ← STDIO transport (STANDALONE)
  mcp_server/webhooks.py        ← GitHub/Notion/Confluence (ACTIVE)
  mcp_server/scratch_store.py   ← in-memory scratch (ACTIVE)
  mcp_server/security/          ← auth.py, rate_limiter.py (DEAD — not wired)

  │
  ▼
  EXTERNAL SYSTEMS
  ─────────────────
  Neo4j :7687  │  Qdrant :6333  │  Redis :6379  │  GPU Gateway :8080
```

---

## Diagram 6: Active vs Legacy Paths (verified from @status annotations)

```
  INGESTION:

  ACTIVE:
  ┌─────────────────────────────────────────────────────────────┐
  │ atomic.py → AtomicIngestionCoordinator.ingest_document_atomic()│
  │   ├─ _prepare_ingestion() (parse + chunk + embed + extract)  │
  │   ├─ validate_pre_ingestion() (dimension + schema checks)    │
  │   ├─ Neo4j tx: upsert Document/Chunk/Entity/Relationships    │
  │   ├─ Qdrant upsert (7 vector fields)                         │
  │   ├─ _create_cross_doc_links() (Phase 3.5 RELATED_TO)        │
  │   └─ saga commit/compensate                                  │
  └─────────────────────────────────────────────────────────────┘
        ↑ called by
  ┌─────────────────────────────────────────────────────────────┐
  │ worker.py → process_job() → coordinator.ingest_document_atomic()│
  └─────────────────────────────────────────────────────────────┘

  DEPRECATED/DEAD:
  build_graph.py  ── DEPRECATED ── lazily imported by atomic.py behind config flag
  orchestrator.py ── DEAD ──────── never loaded (auto/__init__.py cleanup)
  api.py          ── DEAD ──────── test facade, not called from production
  reconcile.py    ── DEAD ──────── superseded by atomic.py saga pattern

  PARSERS:
  parsers/markdown_it_parser.py  ── ACTIVE (preferred)
  parsers/markdown_parser.py     ── DEPRECATED (legacy custom parser)

  MCP SERVER:

  ACTIVE:
  ┌─────────────────────────────────────────────────────────────┐
  │ main.py → FastAPI + StreamableHTTPSessionManager at /_mcp    │
  │   ├─ health, ready, metrics endpoints                        │
  │   ├─ OTEL tracing + Prometheus metrics                       │
  │   └─ correlation_id_middleware + metrics_middleware           │
  └─────────────────────────────────────────────────────────────┘
        ↑
  ┌─────────────────────────────────────────────────────────────┐
  │ mcp_app.py → build_mcp_server() (mcp.server.lowlevel)        │
  │   ├─ kb_search, kb_read_excerpt, kb_search_evidence          │
  │   ├─ kb_traverse, kb_create_scratch, kb_write_trace          │
  │   └─ ContextBudgetManager (14k tokens / 512KB per turn)      │
  └─────────────────────────────────────────────────────────────┘

  DEPRECATED (disabled by default):
  /mcp/initialize    ── Deprecation header + 404 when disabled
  /mcp/tools/list    ── Same
  /mcp/tools/call    ── Same

  DEAD (not wired in):
  security/auth.py       ── JWT auth exists but never mounted
  security/rate_limiter.py ── Rate limiter exists but never mounted

  NEO4J MODULES:

  ACTIVE:
  neo/schema.py            ← RELATIONSHIP_TYPES (24 types)
  neo/defensive_query.py   ← query safety guards
  neo/graph_enhancements.py ← parent heading + structural labels
  neo/health.py            ← health checks
  neo/schema_validator.py  ← schema validation

  DEAD (imported but not used):
  neo/contract_checks.py   ── DEAD but imported by worker.py (GraphContractChecker)
  neo/entity_normalization.py ── DEAD
  neo/explain_guard.py     ── DEAD
  neo/structural_builder.py ── DEAD (superseded by atomic.py)
```

---

## Diagram 7: Qdrant Vector Schema (from docker-compose + config)

```
  Collection: weka_sections (namespaced by profile)

  Named Vectors:
  ┌──────────────────┬──────────┬──────────┬──────────────────────────┐
  │ Vector Name       │ Type     │ Dim      │ Purpose                  │
  ├──────────────────┼──────────┼──────────┼──────────────────────────┤
  │ content           │ dense    │ 1024     │ Main chunk embedding     │
  │ title             │ dense    │ 1024     │ Chunk title embedding    │
  │ doc_title         │ dense    │ 1024     │ Document-level embedding │
  │ late-interaction  │ multi    │ 128/tok  │ ColBERT token-level      │
  ├──────────────────┼──────────┼──────────┼──────────────────────────┤
  │ text-sparse       │ sparse   │ var      │ SPLADEv3 lexical         │
  │ doc_title-sparse  │ sparse   │ var      │ Doc title lexical        │
  │ title-sparse      │ sparse   │ var      │ Chunk title lexical      │
  │ entity-sparse     │ sparse   │ var      │ GLiNER entity names      │
  └──────────────────┴──────────┴──────────┴──────────────────────────┘

  Payload Fields:
  - chunk_id (string, unique)
  - document_id (string)
  - doc_tag (string)
  - heading (string)
  - text (string)
  - parent_path (string)
  - parent_path_norm (string)
  - source_path (string)
  - token_count (int)
  - chunk_index (int)
  - created_at (datetime)
  - embedding_version (string)
  - embedding_profile (string)
  - entities (array of {name, type, confidence})
  - mentions (array of entity mentions)
```

---

## Diagram 8: Neo4j Graph Model (from atomic.py + schema)

```
  (d:Document {id, title, source_uri, doc_tag, created_at, token_count})
       │
       HAS_CHUNK (ordered)
       │
       v
  (c:Chunk {id, text, heading, parent_path, token_count, chunk_index})
       │
       ├─ NEXT_CHUNK → (c:Chunk) [sequential ordering]
       ├─ PARENT_HEADING → (h:Heading {text, level, path})
       │
       ├─ MENTIONS → (e:Entity {id, name, type, confidence})
       │   [from GLiNER + structural extractors]
       │
       └─ RELATED_TO → (c:Chunk) [cross-document, signal-provenance]
           {score, colbert_score, method, reciprocity_count}

  (h:Heading)
       │
       CHILD_OF → (h:Heading) [hierarchy]

  Entity→Entity relationships (ALLOWED_ENTITY_RELATIONSHIP_TYPES):
  - CONTAINS_STEP  [procedure → step ordering]
  - REFERENCES     [cross-document references]
  - CONFIGURES     [config → component]
  - RESOLVES       [error → procedure]

  Node Labels: Document, Chunk, Heading, Entity
  Relationship Types: 24 declared (some never materialized)
```

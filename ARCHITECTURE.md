# WekaDocs Matrix — System Architecture

> GraphRAG pipeline for Weka product documentation
> Multi-vector hybrid search + Neo4j knowledge graph + MCP API server

---

## System Overview

```mermaid
graph TB
    subgraph "External Surface"
        LLM["AI Agent (Claude, Cursor, etc.)"]
        GH["GitHub Webhooks"]
        FS["File System (spool/)"]
    end

    subgraph "MCP Server (HTTP + STDIO)"
        MCPApp["mcp_app.py<br/>16 canonical tools"]
        QSrv["query_service.py<br/>Retrieval Orchestrator"]
        Webhooks["webhooks.py<br/>GitHub/Notion"]
        Health["main.py<br/>/health /ready /metrics"]
    end

    subgraph "Query Engine"
        HR["HybridRetriever<br/>RRF fusion → signal pool →<br/>ColBERT → cross-encoder"]
        QDB["QdrantMultiVectorRetriever<br/>7 named vectors per chunk"]
        GP["GraphPipeline<br/>RELATED_TO / ENTITY / REFERENCES"]
        EP["ExpansionPipeline<br/>adjacency + structure-aware"]
        RP["RerankPipeline<br/>cross-encoder + ColBERT"]
        CA["ContextAssembler<br/>token-budgeted assembly"]
    end

    subgraph "Ingestion Pipeline"
        W["Worker<br/>Redis queue loop"]
        AIC["AtomicIngestionCoordinator<br/>saga-coordinated writes"]
        ParserTag["Parser<br/>markdown-it-py (AST)"]
        ChunkA["Chunk Assembler<br/>semantic/greedy"]
        Extract["Entity Extractors<br/>commands/configs/procedures"]
        SE["Structural Edges<br/>NEXT_CHUNK/PARENT_HEADING"]
    end

    subgraph "Storage"
        Neo4j["Neo4j<br/>Document/Section/Chunk<br/>Entity/Procedure/Command<br/>22 relationship types"]
        Qdrant["Qdrant<br/>multi-vector chunks<br/>7 named vectors"]
        Redis["Redis<br/>job queue + cache<br/>epoch invalidation"]
    end

    subgraph "Providers (ML Backends)"
        EmbProvider["Embedding Provider<br/>BGE-M3 / Jina / Voyage<br/>Snowflake Arctic / Qwen3"]
        RerankProvider["Reranker<br/>Local (Qwen3-Reranker-4B)<br/>Jina Reranker"]
        NER["GLiNER<br/>zero-shot NER"]
        Tokenizer["TokenizerService<br/>HF + Jina segmenter"]
    end

    LLM -->|MCP Streamable HTTP / STDIO| MCPApp
    GH --> Webhooks
    FS --> W

    MCPApp --> QSrv
    QSrv --> HR
    HR --> QDB
    HR --> GP
    HR --> EP
    HR --> RP
    HR --> CA
    CA --> QSrv

    W --> AIC
    AIC --> ParserTag
    ParserTag --> ChunkA
    AIC --> Extract
    AIC --> SE
    AIC -->|saga write| Neo4j
    AIC -->|saga write| Qdrant
    W --> Redis

    QDB --> Qdrant
    GP --> Neo4j
    HR --> EmbProvider
    RP --> RerankProvider
    AIC --> EmbProvider
    AIC --> NER
    EmbProvider --> Tokenizer

    style MCPApp fill:#4a9,stroke:#333
    style HR fill:#49a,stroke:#333
    style AIC fill:#49a,stroke:#333
    style Neo4j fill:#944,stroke:#333
    style Qdrant fill:#944,stroke:#333
    style Redis fill:#944,stroke:#333
```

---

## Parser Routing (2 Parsers, 1 Router)

```mermaid
graph LR
    Markdown["parse_markdown()"]
    Router["parsers/__init__.py<br/>config.parser.engine"]
    Legacy["parsers/markdown.py<br/>(markdown + BeautifulSoup)<br/>DORMANT"]
    MITPy["parsers/markdown_it_parser.py<br/>(AST-based, default)<br/>ACTIVE"]
    Shadow["parsers/shadow_comparison.py<br/>config.parser.shadow_mode"]
    HTML["parsers/html.py"]
    Notion["parsers/notion.py<br/>DEAD"]

    Markdown --> Router
    Router -->|"engine=markdown-it-py"| MITPy
    Router -->|"engine=legacy"| Legacy
    Router -->|"shadow_mode=true"| Shadow
    HTML --> Router
    Notion -.->|"never called"| Router
```

---

## Retrieval Architecture (Active Path)

```mermaid
graph TB
    Query["User Query"]
    Intent["classify_query_intent()<br/>8 query types"]
    Plan["resolve_retrieval_plan()<br/>4 named profiles"]
    MVS["Multi-Vector Search<br/>7 signal sources"]
    RRF["RRF Fusion<br/>6 fields weighted"]
    SignalPool["Signal Pool<br/>9-slot diversity<br/>(optional)"]
    ColBERT["ColBERT MaxSim<br/>pre-rerank<br/>(optional)"]
    GraphChan["Graph Channels<br/>RELATED_TO + ENTITY<br/>+ MENTIONS"]
    Expansion["Bounded Expansion<br/>NEXT_CHUNK + siblings<br/>+ shared entities"]
    Reranker["Cross-Encoder<br/>Qwen3-Reranker-4B"]
    Specificity["Specificity Adjust<br/>post-rerank tie-break"]
    Assembly["Context Assembler<br/>4500 token budget<br/>grouped by section"]
    Evidence["Evidence Extraction<br/>span-level quotes<br/>+ structure expansion"]

    Query --> Intent
    Intent --> Plan
    Plan --> MVS
    MVS --> RRF
    RRF --> SignalPool
    SignalPool --> ColBERT
    SignalPool --> GraphChan
    ColBERT --> Reranker
    Reranker --> Specificity
    GraphChan --> Expansion
    Specificity --> Assembly
    Expansion --> Assembly
    Assembly --> Evidence

    style MVS fill:#49a,stroke:#333
    style RRF fill:#49a,stroke:#333
    style Reranker fill:#49a,stroke:#333
```

---

## Ingestion Pipeline (Active Path)

```mermaid
graph TB
    File["Markdown File<br/>in spool/ dir"]
    Queue["Redis Queue<br/>BRPOPLPUSH<br/>duplicate detection"]
    Worker["worker.py<br/>async queue loop"]
    Parse["markdown-it-py<br/>AST parsing<br/>frontmatter + headings"]
    Assemble["Chunk Assembler<br/>semantic (Chonkie)<br/>or greedy"]
    Entities["Entity Extraction<br/>commands/configs/procedures"]
    Embed["Embedding<br/>dense+sparse+ColBERT<br/>BGE-M3"]
    SagaBegin["SAGA: Begin<br/>Neo4j + Qdrant"]
    NeoWrite["Neo4j Write<br/>Document/Section/Chunk/Entity<br/>MENTIONS/REFERENCES/HAS_CHUNK"]
    QdrantWrite["Qdrant Write<br/>7 named vectors per chunk<br/>retry on failure"]
    Structural["Structural Edges<br/>NEXT_CHUNK/PARENT_HEADING<br/>CHILD_OF/PARENT_OF"]
    CrossDoc["Cross-Doc Linking<br/>RELATED_TO edges<br/>RRF + ColBERT"]
    SagaEnd["SAGA: Commit<br/>or Compensate"]

    File --> Queue
    Queue --> Worker
    Worker --> Parse
    Parse --> Assemble
    Assemble --> Entities
    Entities --> Embed
    Embed --> SagaBegin
    SagaBegin --> NeoWrite
    SagaBegin --> QdrantWrite
    NeoWrite --> Structural
    QdrantWrite --> SagaEnd
    Structural --> SagaEnd
    SagaEnd --> CrossDoc
    CrossDoc -.->|"async post-commit"| SagaEnd
```

---

## MCP Tool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  mcp_app.py — build_mcp_server() → mcp.server.Server       │
│                                                             │
│  3 tool profiles (MCP_TOOL_PROFILE env):                    │
│  ├── production (default): kb.retrieve_evidence,            │
│  │                         kb.read_excerpt,                │
│  │                         graph.expand                     │
│  ├── analyst: all 16 canonical tools                       │
│  └── full: no filtering                                    │
│                                                             │
│  CANONICAL TOOLS (16):                                     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ kb.search         — Hybrid search + dedup + scratch   │ │
│  │ kb.read_excerpt    — Scratch retrieval                │ │
│  │ kb.expand_excerpt  — Structural neighbor expansion     │ │
│  │ kb.extract_evidence — Span-level quotes                │ │
│  │ kb.retrieve_evidence — Full evidence pack pipeline     │ │
│  │ kb.search_sections — Section-level search (light)      │ │
│  │ kb.get_section_text — Section text fetch               │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │ graph.describe    — Node description                  │ │
│  │ graph.expand      — Neighbor expansion                │ │
│  │ graph.paths       — Inter-node paths                  │ │
│  │ graph.parents     — Parent nodes                      │ │
│  │ graph.children    — Child nodes                       │ │
│  │ graph.entities_for_sections — Entity lookup            │ │
│  │ graph.sections_for_entities — Reverse entity lookup    │ │
│  │ graph.traverse    — Relationship traversal             │ │
│  │ graph.summarize   — Neighborhood summary               │ │
│  │ graph.context_bundle — Graph+text context bundle       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  LEGACY ALIASES (16, deprecated):                           │
│  kb_search → kb.search    search_documentation (v1)         │
│  graph_describe → graph.describe  etc.                      │
│                                                             │
│  TRANSPORTS:                                                │
│  ├── HTTP Streamable: main.py → /_mcp (MCP SDK SessionMgr) │
│  ├── HTTP Legacy REST: main.py → /mcp/* (gated)            │
│  └── STDIO: stdio_server.py → Claude Desktop direct          │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Model (Neo4j Graph Schema)

```mermaid
graph TB
    subgraph "Document Structure"
        Doc["Document<br/>(id, url, title)"]
        Sect["Section<br/>(id, title, level, parent_path, block_type)"]
        Chunk["Chunk<br/>(id, order, parent_section_id, text, has_code, has_table)"]
        Doc -->|HAS_CHUNK| Chunk
        Chunk -->|IN_SECTION| Sect
        Chunk1["Chunk N"] -->|NEXT_CHUNK| Chunk2["Chunk N+1"]
        Sect -->|PARENT_HEADING| ParentSect["Parent Section"]
        Sect -->|CHILD_OF| Doc
    end

    subgraph "Entity Graph"
        Entity["Entity<br/>(name, type, normalized_name)"]
        Command["Command<br/>(name, pattern)"]
        Config["Config<br/>(name, file, param_type)"]
        Procedure["Procedure<br/>(name, steps)"]
        Step["Step<br/>(number, description)"]
        Chunk -->|MENTIONS| Entity
        Chunk -->|MENTIONS| Command
        Chunk -->|MENTIONS| Config
        Sect -->|CONTAINS_STEP| Step
        Step -->|MEMBER_OF| Procedure
        Command -->|HAS_PARAMETER| Param["Parameter"]
        Config -->|DEFINES| ConfigParam["Config Parameter"]
    end

    subgraph "Cross-Document"
        DocA["Document A"] -->|REFERENCES| DocB["Document B"]
        DocA -->|RELATED_TO| DocC["Document C"]
    end

    subgraph "Session Tracking"
        Session["Session<br/>(id, start_time)"]
        Query["Query<br/>(id, raw_text, intent)"]
        Answer["Answer<br/>(id, entity_focus)"]
        Session -->|HAS_QUERY| Query
        Query -->|ANSWERED_AS| Answer
        Chunk -->|RETRIEVED| Query
    end
```

---

## 22 Relationship Types (from `neo/schema.py`)

```
ANSWERED_AS    CHILD_OF        CONTAINS_STEP   DEFINES
EXECUTES       FOCUSED_ON      HAS_CITATION     HAS_PARAMETER
HAS_QUERY      HAS_CHUNK       IN_CHUNK         IN_SECTION
MENTIONED_IN   MENTIONS        NEXT             NEXT_CHUNK
PARENT_HEADING PARENT_OF       REFERENCES       RELATED_TO
RESOLVES       RETRIEVED       SUPPORTED_BY
```

---

## Feature Flags & Configuration Knobs

### Retrieval Profile System (4 profiles replace 18+ individual flags)

| Profile | dense | sparse | colbert | signal_pool | graph | expansion | focused_rerank |
|---------|-------|--------|---------|-------------|-------|-----------|----------------|
| `vector_only` | ✓ | — | ✓ | — | — | — | — |
| `precision_vector` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| `graph_assisted` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `graph_full` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Key Toggle Points

| Category | Count | Examples |
|----------|-------|----------|
| Embedding providers | 6 | BGE-M3 (default), Jina, Voyage, Arctic, Qwen3, ST |
| Rerank providers | 3 | Local (Qwen3-Reranker-4B), Jina, Noop |
| Chunk assemblers | 3 | Greedy (default), Semantic (Chonkie), Structured |
| Markdown parsers | 2 | markdown-it-py (default), legacy markdown |
| Retrieval engines | 2 | HybridRetriever (active), HybridSearchEngine (legacy) |
| Sparse vector types | 4 | text-sparse, title-sparse, entity-sparse, doc_title-sparse |
| Ingestion watchers | 3 | FileSystem, S3 (stub), HTTP |
| MCP transports | 3 | HTTP Streamable, HTTP Legacy REST, STDIO |
| MCP tool profiles | 3 | production (3 tools), analyst (16), full (unfiltered) |

### Environment Variable Knobs: ~85 total
- ~15 in mcp_server (MCP_TOOL_PROFILE, MCP_EVIDENCE_*, MCP_SCRATCH_*)
- ~15 in providers (EMBEDDINGS_PROVIDER, BGE_M3_*, QWEN3_*, CHONKIE_*)
- ~12 in tokenizer (TOKENIZER_BACKEND, HF_*, TRANSFORMERS_*)
- ~8 in reranker (RERANKER_*)
- ~20 in ingestion (INGEST_WATCH_*, EMBED_*, SPARSE_*)
- ~15 in retrieval (hybrid.*, expansion.*, signal_pool.*, cache.*)

---

## Docker Architecture

```
┌──────────────────────────────────────────────────────┐
│  docker-compose (implied)                             │
│                                                       │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │ mcp-server           │  │ ingestion-service     │  │
│  │ (FastAPI:8000,       │  │ (FastAPI:9108,        │  │
│  │  MCP Streamable HTTP)│  │  file watcher)         │  │
│  │ → query_service      │  │ → auto/service.py      │  │
│  └─────────────────────┘  └──────────────────────┘  │
│                                                       │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │ ingestion-worker     │  │ mxbai-reranker        │  │
│  │ (Redis queue loop)   │  │ (Qwen3-Reranker-4B    │  │
│  │ → worker.py          │  │  HTTP service)         │  │
│  └─────────────────────┘  └──────────────────────┘  │
│                                                       │
│  ┌──────────┐ ┌────────────┐ ┌──────────────────┐   │
│  │ Neo4j    │ │ Qdrant     │ │ Redis             │   │
│  │ (graph)  │ │ (vectors)  │ │ (queue + cache)    │   │
│  └──────────┘ └────────────┘ └──────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## Key Architectural Patterns

### 1. Saga-Coordinated Dual Write
`AtomicIngestionCoordinator` writes to both Neo4j and Qdrant within a logical saga. If either write fails, the other is compensated (Qdrant points deleted, Neo4j transaction rolled back).

### 2. Multi-Vector Qdrant Strategy
Each chunk has 7 named vectors in a single Qdrant collection:
- **content** (dense, 1024d) — BGE-M3 semantic
- **title** (dense, 1024d) — section heading vector
- **text-sparse** (SPLADE, 250002d) — lexical matching
- **title-sparse** (SPLADE, 250002d) — heading lexical
- **entity-sparse** (SPLADE, 250002d) — entity lexical
- **doc_title-sparse** (SPLADE, 250002d) — document-level lexical
- **colbert** (multi-vector) — late-interaction

### 3. RRF Fusion with Per-Field Weights
Multi-vector results are fused via Reciprocal Rank Fusion with configurable field weights. Content=1.2, text-sparse=2.0 (SPLADE prioritized for domain terms), rest at 0.3-0.5.

### 4. Evidence Extraction Pipeline
`kb.retrieve_evidence`: Search → dedup → signal pool → ColBERT → cross-encoder → graph expansion → span-level evidence extraction → score blending (70% retrieval / 30% lexical overlap).

### 5. Epoch-Based Cache Invalidation
Redis stores document-level and chunk-level epoch counters. Cache keys include epoch versions for O(1) invalidation.

---

## Dependency Map (Production Import Chain)

```
main.py / stdio_server.py
  └── mcp_app.py
        ├── query_service.py
        │     ├── HybridRetriever (hybrid_retrieval.py)
        │     │     ├── QdrantMultiVectorRetriever (vector_backends.py)
        │     │     │     └── EmbeddingProvider (provider factory)
        │     │     ├── classify_query_intent (query_intent.py)
        │     │     ├── rrf_fusion (fusion_pipeline.py)
        │     │     ├── bounded_expansion (expansion_pipeline.py)
        │     │     ├── get_reranker + colbert_rerank (rerank_pipeline.py)
        │     │     ├── graph_pipeline.py
        │     │     ├── signal_pool.py
        │     │     ├── resolve_retrieval_plan (retrieval_plan.py)
        │     │     └── ContextAssembler (context_assembly.py)
        │     └── SessionTracker (session_tracker.py)
        ├── GraphService (services/graph_service.py)
        │     ├── ContextBudgetManager
        │     └── SessionDeltaCache
        ├── TextService (services/text_service.py)
        ├── ContextAssemblerService (services/context_assembler.py)
        └── ScratchStore (scratch_store.py)

worker.py
  └── AtomicIngestionCoordinator (atomic.py)
        ├── parse_markdown (ingestion/parsers)
        ├── chunk_assembler / semantic_chunker
        ├── extract_entities (extract/)
        ├── compute_embeddings (providers)
        ├── build_structural_edges (structural_edges.py)
        └── cross_doc_linking (services/cross_doc_linking.py)

Everything: shared/config.py, shared/connections.py, shared/observability/*
```

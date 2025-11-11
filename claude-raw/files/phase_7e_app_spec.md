# Phase-7E GraphRAG v2.1 Complete Application Specification

## Executive Summary

Phase-7E implements a production-ready GraphRAG (Graph Retrieval-Augmented Generation) system combining Neo4j graph database, Qdrant vector database, and Jina v3 embeddings (1024-dimensional). The system preserves 100% of document content through intelligent chunking, provides hybrid retrieval with rank fusion, and supports multi-turn conversational interactions with full provenance tracking.

**Core Principles:**
- Zero content loss - every character preserved and retrievable
- Idempotent operations - safe to re-run at any time
- Full provenance tracking - complete citation chains
- Production stability - no incomplete features or placeholders

## System Architecture

### High-Level Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     MCP Server Interface                      │
│                    (Model Context Protocol)                   │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────┴─────────────────────────────────────────────┐
│                    GraphRAG Core Engine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Document   │  │   Chunking   │  │    Embedding     │  │
│  │    Parser    │→ │   Pipeline   │→ │     Service      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                           │                     │            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Storage & Persistence Layer              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │  Neo4j   │  │  Qdrant  │  │   Redis Cache    │  │  │
│  │  │  Graph   │  │  Vector  │  │   Write-Through  │  │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Retrieval & Ranking Pipeline               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │  Vector  │  │   BM25   │  │  Rank Fusion    │  │  │
│  │  │  Search  │  │  Sparse  │  │   & Reranking   │  │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Document Input → Parser → Sections → Combiner → Chunks → Embedder → Storage
                                         ↓                      ↓
                                    Splitter              Neo4j + Qdrant
                                  (if needed)                   ↓
                                                          Retrieval Engine
                                                                ↓
Query → Embedding → Hybrid Search → Rank Fusion → Context Assembly → Response
```

## Core Components and Features

### Component 1: Document Parser
**Purpose:** Extract structured content from documents while preserving all metadata and relationships.

**Features:**
- Multi-format support (Markdown, HTML, PDF via extraction)
- Hierarchical section detection (H1-H6 levels)
- Metadata preservation (timestamps, versions, URIs)
- Boundary tracking for perfect reconstruction
- Token counting using exact Jina v3 tokenizer

**Specifications:**
- Input: Raw documents with source URIs
- Output: Structured sections with hierarchy and metadata
- Constraints: Must preserve byte-perfect content

### Component 2: Intelligent Chunking Pipeline
**Purpose:** Create optimally-sized chunks for embedding while preserving semantic coherence.

**Features:**
- Hierarchical combination within semantic boundaries
- Target chunk size: 800-1500 tokens (configurable)
- Absolute maximum: 7900 tokens (safety margin for 8192 limit)
- Fallback splitting for oversized content
- Deterministic chunk ID generation

**Specifications:**
- Combine adjacent micro-sections within same H2
- Split only when single unit exceeds 7900 tokens
- Track original section IDs for provenance
- Generate stable IDs using SHA256 hash

### Component 3: Embedding Service
**Purpose:** Generate high-quality vector representations using Jina v3.

**Features:**
- 1024-dimensional embeddings
- Batch processing with adaptive sizing
- Dual-backend support (HuggingFace local, Jina API)
- Token counting with exact XLM-RoBERTa tokenizer
- Embedding metadata tracking

**Specifications:**
- Model: jina-embeddings-v3
- Provider: jina-ai
- Dimensions: 1024
- Similarity: Cosine
- Max tokens: 8192

### Component 4: Graph Storage (Neo4j)
**Purpose:** Store document structure, relationships, and metadata in graph format.

**Node Types:**
- Document: Source documents with metadata
- Section/Chunk: Dual-labeled content nodes
- Session: Multi-turn conversation tracking
- Query: User queries with timestamps
- Answer: Generated responses with feedback
- Domain entities: Command, Configuration, Procedure, Error, Concept, Example, Step, Parameter, Component

**Relationships:**
- (:Document)-[:HAS_SECTION]->(:Section)
- (:Section)-[:NEXT_CHUNK]->(:Section)
- (:Session)-[:HAS_QUERY]->(:Query)
- (:Query)-[:HAS_ANSWER]->(:Answer)
- (:Answer)-[:CITES]->(:Section)

**Properties (Section/Chunk nodes):**
- id: Unique identifier (SHA256-based)
- document_id: Foreign key to Document
- parent_section_id: Logical parent anchor
- level: Heading depth (1-6)
- order: Position within parent
- heading: Section title
- text: Content text
- is_combined: Boolean flag
- is_split: Boolean flag
- original_section_ids: Array of source sections
- boundaries_json: Serialized boundary metadata
- token_count: Integer count
- vector_embedding: 1024-D float array
- embedding_version: Model identifier
- embedding_provider: Provider name
- embedding_dimensions: Vector size
- embedding_timestamp: UTC timestamp
- updated_at: Last modification time

### Component 5: Vector Storage (Qdrant)
**Purpose:** Enable fast semantic search through vector similarity.

**Collection Configuration:**
- Name: chunks
- Vector size: 1024
- Distance metric: Cosine
- Payload indexes: document_id, parent_section_id, order, updated_at

**Point Structure:**
- ID: Stable UUID from chunk ID
- Vector: 1024-dimensional embedding
- Payload: Complete chunk metadata and text

### Component 6: Hybrid Retrieval Pipeline
**Purpose:** Combine vector and keyword search for optimal retrieval.

**Features:**
- Vector similarity search (semantic)
- BM25 sparse retrieval (keyword)
- Reciprocal rank fusion
- Context-aware expansion
- Page-aware grouping

**Retrieval Strategy:**
1. Parallel search: Vector + BM25
2. Score normalization
3. Reciprocal rank fusion
4. Group by parent_section_id
5. Optional adjacency expansion (±1 chunks)
6. Context assembly by order

### Component 7: Cache Layer (Redis)
**Purpose:** Accelerate retrieval and reduce computation.

**Caching Strategy:**
- Write-through for embeddings
- Query result caching with TTL
- Invalidation on document update
- Lookup join optimization

### Component 8: Session Management
**Purpose:** Track multi-turn conversations with context preservation.

**Features:**
- Session lifecycle management
- Query-answer relationship tracking
- Context accumulation
- User feedback capture
- Citation chain preservation

## Implementation Phases and Dependencies

### Phase 7E.1: Foundation Layer
**Purpose:** Establish core infrastructure and dependencies.

**Sub-tasks:**
1. **7E.1.1** - Environment Setup
   - Install Neo4j Community Edition
   - Install Qdrant with Docker
   - Install Redis for caching
   - Configure Python 3.11+ environment

2. **7E.1.2** - Dependency Installation
   - Core: neo4j-python-driver, qdrant-client, redis-py
   - Embeddings: transformers, sentencepiece, huggingface_hub
   - Processing: numpy, pydantic, python-dotenv
   - Utilities: hashlib, datetime, json, logging

3. **7E.1.3** - Configuration Management
   - Environment variable setup
   - Connection string configuration
   - Logging infrastructure
   - Error handling framework

**Acceptance Criteria:**
- All services running and accessible
- Dependencies installed and importable
- Configuration validated
- Basic connectivity tests pass

### Phase 7E.2: Schema Implementation
**Purpose:** Create database schemas and indexes.

**Dependencies:** Phase 7E.1 complete

**Sub-tasks:**
1. **7E.2.1** - Neo4j Schema Creation
   - Apply create_schema_v2_1_complete__v3.cypher
   - Create constraints and indexes
   - Add dual-labeling for Section/Chunk
   - Implement SchemaVersion marker

2. **7E.2.2** - Qdrant Collection Setup
   - Create chunks collection
   - Configure vector parameters (1024-D, cosine)
   - Create payload indexes
   - Set optimization parameters

3. **7E.2.3** - Schema Validation
   - Verify all constraints created
   - Test index performance
   - Validate vector dimensions
   - Check dual-label functionality

**Acceptance Criteria:**
- All constraints and indexes created
- Vector indexes operational
- Schema version recorded
- Validation queries pass

### Phase 7E.3: Document Processing Pipeline
**Purpose:** Implement document parsing and chunking.

**Dependencies:** Phase 7E.2 complete

**Sub-tasks:**
1. **7E.3.1** - Document Parser Implementation
   - Markdown/HTML parsing logic
   - Section hierarchy detection
   - Metadata extraction
   - Boundary tracking

2. **7E.3.2** - Token Counter Integration
   - HuggingFace tokenizer setup
   - Jina Segmenter API fallback
   - Token counting optimization
   - Integrity hash computation

3. **7E.3.3** - Intelligent Chunking Pipeline
   - Section combiner algorithm
   - Splitting fallback mechanism
   - Chunk ID generation
   - Metadata preservation

**Acceptance Criteria:**
- Documents parsed with hierarchy preserved
- Token counts accurate to model
- Chunks within size constraints
- Zero content loss verified

### Phase 7E.4: Embedding Service
**Purpose:** Generate and manage vector embeddings.

**Dependencies:** Phase 7E.3 complete

**Sub-tasks:**
1. **7E.4.1** - Embedding Provider Setup
   - Jina API configuration
   - Batch processing implementation
   - Error handling and retries
   - Rate limiting

2. **7E.4.2** - Embedding Pipeline
   - Adaptive batch creation
   - Vector generation
   - Metadata attachment
   - Quality validation

3. **7E.4.3** - Embedding Storage
   - Neo4j vector property storage
   - Qdrant point creation
   - Cache warming
   - Integrity verification

**Acceptance Criteria:**
- Embeddings generated for all chunks
- Vectors stored in both databases
- Metadata correctly attached
- Dimension validation passes

### Phase 7E.5: Storage Integration
**Purpose:** Implement unified storage layer.

**Dependencies:** Phase 7E.4 complete

**Sub-tasks:**
1. **7E.5.1** - Neo4j Integration
   - Node creation with properties
   - Relationship establishment
   - Transaction management
   - Batch operations

2. **7E.5.2** - Qdrant Integration
   - Point upsert implementation
   - Payload management
   - Collection optimization
   - Batch operations

3. **7E.5.3** - Redis Cache Layer
   - Cache strategy implementation
   - TTL configuration
   - Invalidation logic
   - Performance monitoring

**Acceptance Criteria:**
- Data consistently stored across systems
- Relationships properly established
- Cache functioning correctly
- Performance metrics acceptable

### Phase 7E.6: Retrieval Pipeline
**Purpose:** Implement hybrid search and ranking.

**Dependencies:** Phase 7E.5 complete

**Sub-tasks:**
1. **7E.6.1** - Vector Search Implementation
   - Query embedding generation
   - Similarity search
   - Top-k retrieval
   - Score normalization

2. **7E.6.2** - BM25 Sparse Search
   - Index creation
   - Query processing
   - Term frequency calculation
   - Score computation

3. **7E.6.3** - Rank Fusion
   - Reciprocal rank fusion algorithm
   - Score combination
   - Reranking logic
   - Context assembly

**Acceptance Criteria:**
- Both search methods functional
- Fusion produces better results
- Context properly assembled
- Performance within targets

### Phase 7E.7: Session Management
**Purpose:** Enable multi-turn conversations.

**Dependencies:** Phase 7E.6 complete

**Sub-tasks:**
1. **7E.7.1** - Session Lifecycle
   - Session creation and expiry
   - State management
   - Context accumulation
   - Cleanup procedures

2. **7E.7.2** - Query-Answer Tracking
   - Relationship creation
   - Turn management
   - Citation tracking
   - Feedback collection

3. **7E.7.3** - Context Preservation
   - Historical context retrieval
   - Relevance filtering
   - Context windowing
   - Memory optimization

**Acceptance Criteria:**
- Sessions properly managed
- Context preserved across turns
- Citations accurately tracked
- Performance acceptable

### Phase 7E.8: Quality Assurance
**Purpose:** Ensure production readiness.

**Dependencies:** Phase 7E.7 complete

**Sub-tasks:**
1. **7E.8.1** - Integrity Testing
   - Zero-loss verification
   - Chunk reassembly tests
   - Boundary validation
   - Hash verification

2. **7E.8.2** - Performance Testing
   - Ingestion throughput
   - Query latency
   - Concurrent user load
   - Cache effectiveness

3. **7E.8.3** - Retrieval Quality
   - Relevance metrics
   - Precision/recall
   - A/B testing
   - User feedback analysis

**Acceptance Criteria:**
- All integrity tests pass
- Performance meets SLAs
- Quality metrics acceptable
- System stable under load

### Phase 7E.9: Monitoring and Operations
**Purpose:** Enable production operations.

**Dependencies:** Phase 7E.8 complete

**Sub-tasks:**
1. **7E.9.1** - Metrics Collection
   - System metrics
   - Application metrics
   - Business metrics
   - Custom dashboards

2. **7E.9.2** - Health Checks
   - Service availability
   - Index health
   - Cache status
   - Queue depths

3. **7E.9.3** - Operational Procedures
   - Backup and recovery
   - Scaling procedures
   - Incident response
   - Maintenance windows

**Acceptance Criteria:**
- Monitoring fully operational
- Alerts properly configured
- Procedures documented
- Team trained

## System Specifications

### Performance Requirements
- Document ingestion: <100ms per section
- Embedding generation: <500ms per chunk
- Query response: <2 seconds p95
- Concurrent users: 100+
- Cache hit ratio: >80%

### Storage Requirements
- Neo4j: ~10KB per chunk (metadata + relationships)
- Qdrant: ~8KB per chunk (vector + payload)
- Redis: ~50MB for hot cache
- Total: ~20KB per chunk across all systems

### Quality Requirements
- Zero content loss (100% preservation)
- Retrieval precision: >0.85
- Retrieval recall: >0.90
- Embedding quality: >0.95 cosine similarity for duplicates

### Operational Requirements
- Availability: 99.9% uptime
- Recovery: <1 hour RTO, <1 minute RPO
- Scalability: Linear with document count
- Maintainability: Full observability

## Configuration Parameters

### Core Parameters
```
# Embedding Configuration
EMBED_DIM=1024
EMBED_PROVIDER=jina-ai
EMBED_MODEL_ID=jina-embeddings-v3
EMBED_MAX_TOKENS=8192
EMBED_TARGET_TOKENS=7900

# Chunking Configuration
TARGET_MIN_TOKENS=800
TARGET_MAX_TOKENS=1500
ABSOLUTE_MAX_TOKENS=7900
OVERLAP_TOKENS=200
SPLIT_MIN_TOKENS=1000

# Retrieval Configuration
VECTOR_TOP_K=20
BM25_TOP_K=20
FINAL_TOP_K=10
CONTEXT_WINDOW=3

# Cache Configuration
CACHE_TTL_SECONDS=3600
CACHE_MAX_ENTRIES=10000

# System Configuration
BATCH_SIZE=32
MAX_WORKERS=4
REQUEST_TIMEOUT=30
RETRY_COUNT=3
```

## Acceptance Criteria Summary

### Phase Completion Criteria
Each phase must meet 100% of its acceptance criteria before proceeding to the next phase. No partial implementations or "TODO" markers are acceptable.

### System-Wide Criteria
1. **Data Integrity:** Zero content loss verified through hash comparison
2. **Performance:** All operations within specified latency targets
3. **Scalability:** Linear scaling demonstrated up to 1M documents
4. **Reliability:** 99.9% uptime achieved in testing
5. **Quality:** Retrieval metrics exceed thresholds
6. **Observability:** Full monitoring and alerting operational
7. **Documentation:** Complete API and operational docs
8. **Testing:** >90% code coverage with integration tests

## Risk Mitigation

### Technical Risks
1. **Token limit violations:** Mitigated by conservative limits and validation
2. **Embedding API failures:** Mitigated by dual-backend support
3. **Database corruption:** Mitigated by transactional operations and backups
4. **Performance degradation:** Mitigated by caching and optimization

### Operational Risks
1. **Service dependencies:** Mitigated by health checks and circuit breakers
2. **Data loss:** Mitigated by write-through patterns and verification
3. **Scaling issues:** Mitigated by horizontal scaling design
4. **Security:** Mitigated by authentication and encryption

## Conclusion

Phase-7E delivers a production-ready GraphRAG system that preserves all content, provides high-quality retrieval, and supports conversational interactions. The phased implementation ensures each component is fully functional before integration, resulting in a stable, scalable, and maintainable system.

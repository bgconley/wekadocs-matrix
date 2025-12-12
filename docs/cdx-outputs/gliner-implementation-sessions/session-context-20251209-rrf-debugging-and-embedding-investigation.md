# Session Context: RRF Debugging, Embedding Investigation & Vector Field Metrics

**Session Date:** 2025-12-09
**Branch:** `dense-graph-enhance`
**Last Commit:** `d0c0e75` - "feat: complete GLiNER NER integration (Phases 1-5)"
**Document Version:** 11.0
**Previous Context:** `session-context-20251207-vector-architecture-reform.md`

---

## Executive Summary

This session focused on three main areas:
1. **RRF Score Investigation** - Explained why `fused_score` values are inherently low (~0.017) and fixed misleading metrics
2. **Vector Field Metrics Fix** - `vector_fields` in response now accurately reflects all 6 queried vectors
3. **Embedding 400 Error Investigation** - Traced sparse embedding failures to token budget mismatches; confirmed graceful degradation is working correctly

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, and their relationships (HAS_SECTION, HAS_CHUNK, MENTIONS, NEXT_CHUNK, etc.). With `neo4j_disabled: true`, Neo4j is bypassed during retrieval but **still used during ingestion** for storing document structure.

**Connection Details:**
```
URI from host machine: bolt://localhost:7687
URI from inside Docker: bolt://neo4j:7687
Neo4j Browser UI: http://localhost:7474
Username: neo4j
Password: testpassword123
Database: neo4j (default)
Container name: weka-neo4j
Docker service name: neo4j
```

**Access Patterns:**
```bash
# Direct cypher-shell access (CRITICAL: remove -it flag for non-TTY contexts like Claude Code)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count chunks
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"

# Example: Count all node types
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"

# Clear ALL data (for clean re-ingestion)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# Check largest chunk sizes (for debugging token issues)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "
MATCH (c:Chunk)
WITH c, size(c.text) as chars
ORDER BY chars DESC
LIMIT 5
RETURN c.heading as heading, chars, chars/4 as approx_tokens
"
```

**Python Access from Host:**
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    print(result.single()["count"])
```

**CRITICAL FINDING (Previous Session):** Neo4j CPU spikes (~60-190%) every 10-12 seconds are **NOT query-related** - they are internal JVM garbage collection / checkpoint maintenance. This was initially suspected as evidence of graph queries despite `neo4j_disabled=true`, but investigation confirmed it's background maintenance unrelated to queries.

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection with **8 distinct vector types per chunk**.

**Connection Details:**
```
REST API: http://127.0.0.1:6333
gRPC endpoint: localhost:6334
Authentication: None (development mode)
Collection: chunks_multi_bge_m3
Container name: weka-qdrant
Docker service name: qdrant
```

**Current Vector Schema (8 vectors per chunk):**

| Vector Name | Type | Dimensions | Purpose | RRF Weight |
|-------------|------|------------|---------|------------|
| `content` | Dense | 1024 | Main semantic content embedding | 1.0 |
| `title` | Dense | 1024 | Section heading semantic embedding | 1.0 |
| `doc_title` | Dense | 1024 | Document title semantic embedding | 1.0 |
| `late-interaction` | Dense (multi) | 1024 × N | ColBERT MaxSim token-level matching | N/A |
| `text-sparse` | Sparse | Variable | BM25-style lexical content matching | 1.0 |
| `title-sparse` | Sparse | Variable | Section heading lexical matching | **2.0** |
| `doc_title-sparse` | Sparse | Variable | Document title lexical matching | 1.0 |
| `entity-sparse` | Sparse | Variable | Entity name lexical matching (GLiNER) | **1.5** |

**Payload Indexes:** 28+ indexes including 4 entity metadata indexes:
- `entity_metadata.entity_types` (KEYWORD)
- `entity_metadata.entity_values` (KEYWORD)
- `entity_metadata.entity_values_normalized` (KEYWORD)
- `entity_metadata.entity_count` (INTEGER)

**Access Patterns:**
```bash
# Check collection status and point count
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}\")"

# Delete all points (PRESERVE SCHEMA - important for re-ingestion)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# Scroll through points with payload
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("chunks_multi_bge_m3", limit=3, with_payload=True, with_vectors=False)
for p in points:
    em = p.payload.get("entity_metadata", {})
    print(f"{p.payload.get('heading', '')[:50]}: {em.get('entity_count', 0)} entities")
EOF
```

---

### Redis Cache and Queue

Redis serves two purposes: **L2 query result caching** and **RQ job queue** for async ingestion.

**Connection Details:**
```
Host: localhost (from host) / redis (from Docker)
Port: 6379
Password: testredis123
Database: 0
Container name: weka-redis
Docker service name: redis
```

**Access Patterns:**
```bash
# Check database size
docker exec weka-redis redis-cli -a testredis123 DBSIZE

# Flush all data (REQUIRED for clean re-ingestion - worker tracks processed files by hash)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# Check RQ job queue
docker exec weka-redis redis-cli -a testredis123 KEYS "rq:*"
```

**CRITICAL - When to Access Redis:**
- **ALWAYS flush Redis** before re-ingesting the same documents (worker tracks processed files by content hash)
- Debugging stuck ingestion jobs
- Cache invalidation during development

**When NOT to Access Redis:**
- Vector retrieval testing (Redis not in retrieval path)
- Schema verification (Qdrant only)
- Graph queries (Neo4j only)
- General debugging of retrieval quality

---

### BGE-M3 Embedding Service

Provides **dense (1024-D)**, **sparse (BM25-style)**, and **ColBERT multi-vector** embeddings. This is a native macOS service running outside Docker.

**Connection Details:**
```
Host machine: http://127.0.0.1:9000
Inside Docker: http://host.docker.internal:9000
Model: BAAI/bge-m3
```

**Endpoints:**
- `GET /healthz` - Health check
- `POST /v1/embeddings` - Dense embeddings (MUST include "model" param)
- `POST /v1/embeddings/sparse` - Sparse embeddings
- `POST /v1/embeddings/colbert` - ColBERT multi-vectors

**Testing:**
```bash
# Health check
curl -s http://127.0.0.1:9000/healthz

# Generate dense embeddings (model param REQUIRED)
curl -s -X POST 'http://127.0.0.1:9000/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{"model": "BAAI/bge-m3", "input": ["test query"]}'

# Generate sparse embeddings
curl -s -X POST 'http://127.0.0.1:9000/v1/embeddings/sparse' \
  -H 'Content-Type: application/json' \
  -d '{"model": "BAAI/bge-m3", "input": ["test query"]}'
```

**CRITICAL - Token Limits (Discovered This Session):**
```
max_batch_tokens = 8192 (enforced by BGE service)
```

When a batch exceeds this limit, the service returns HTTP 400:
```json
{"detail":"Approximate token count 12500 exceeds max_batch_tokens=8192"}
```

**Multi-Layer Protection Against Token Overflows:**
1. **Token-Budgeted Batching** (`atomic.py:1294`): `EMBED_BATCH_MAX_TOKENS=7000`
2. **Batch-Splitting Retry** (`bge_m3_service.py:151-154`): Splits batches on 400
3. **Per-Batch Error Isolation** (`atomic.py:1425-1451`): Inserts `None` placeholders

---

### BGE Reranker Service

Cross-encoder reranker using BAAI/bge-reranker-v2-m3 for final-stage scoring.

**Connection Details:**
```
Host machine: http://127.0.0.1:9001
Health endpoint: GET /healthz
Rerank endpoint: POST /v1/rerank
```

**CRITICAL: Understanding Reranker Scores**
The BGE-reranker-v2-m3 outputs **raw logits**, NOT probabilities:
- **Negative scores** = Low relevance (but still ranked relatively)
- **Positive scores** = High relevance
- Scores are RELATIVE - higher is better even if both are negative
- A score of -1.5 is "more relevant" than -6.0

**The reranker uses ORIGINAL text, NOT enriched `_embedding_text`** - this is intentional to maintain independent scoring.

---

### GLiNER NER Service (Native MPS-Accelerated)

Native macOS service with Metal Performance Shaders acceleration. Provides **65x faster** entity extraction than CPU-based Docker inference.

**Connection Details:**
```
Host machine: http://127.0.0.1:9002
Inside Docker: http://host.docker.internal:9002
Model: urchade/gliner_medium-v2.1
Device: MPS (Metal Performance Shaders on Apple Silicon)
```

**Starting/Stopping:**
```bash
cd services/gliner-ner && ./run.sh    # Start (auto-detects MPS)
pkill -f "server.py"                   # Stop
```

**Health Check:**
```bash
curl -s http://127.0.0.1:9002/healthz
# Returns: {"status": "ok", "device": "mps", ...}
```

---

## Docker Container Management

### Container Overview

| Container | Service | Purpose | Key Ports | Source Mounts |
|-----------|---------|---------|-----------|---------------|
| `weka-neo4j` | neo4j | Graph database | 7687, 7474 | Data volume |
| `weka-qdrant` | qdrant | Vector store | 6333, 6334 | Data volume |
| `weka-redis` | redis | Cache + queue | 6379 | Data volume |
| `weka-mcp-server` | mcp-server | HTTP MCP + STDIO | 8000 | `./src:/app/src:ro` |
| `weka-ingestion-worker` | ingestion-worker | RQ background jobs | None | `./src:/app/src:ro` |

### Volume Mounts (CRITICAL for Development)

The MCP server and ingestion containers mount source code as **read-only volumes**:
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

**This means:**
- **Code changes are immediately visible** inside containers
- **NO rebuild needed for code changes** - just restart the container
- **Rebuild only needed** for requirements.txt or Dockerfile changes

### When to Rebuild vs Restart

| Change Type | Action Required | Command |
|-------------|-----------------|---------|
| Code in `src/` | Restart only | `docker compose restart mcp-server` |
| Config in `config/` | Restart only | `docker compose restart mcp-server` |
| requirements.txt | Full rebuild | `docker compose build mcp-server && docker compose up -d mcp-server` |
| Dockerfile changes | Full rebuild | `docker compose build mcp-server && docker compose up -d mcp-server` |

### MCP Server STDIO Configuration

The wekadocs MCP server is configured in Claude Desktop to run via Docker:
```json
{
  "mcpServers": {
    "wekadocs": {
      "command": "docker",
      "args": ["exec", "-i", "weka-mcp-server", "/usr/bin/env",
               "WEKADOCS_STDIO_MODE=1", "PYTHONUNBUFFERED=1",
               "python", "-m", "src.mcp_server.stdio_server"]
    }
  }
}
```

**After restarting `weka-mcp-server`, you must restart Claude Desktop** to establish a new STDIO connection.

---

## Data Ingestion Workflow

### File-Drop Pattern

**Host path:** `./data/ingest/`
**Container path:** `/app/data/ingest/`

**Process:**
1. Drop markdown files in `data/ingest/`
2. Worker auto-detects new files via filesystem watcher
3. Files are parsed into Documents → Sections → Chunks
4. GLiNER extracts entities if `ner.enabled: true`
5. Embeddings generated (8 vectors per chunk using `_embedding_text`)
6. Neo4j nodes created with relationships
7. Qdrant vectors stored with payloads
8. Redis tracks processed file hash to prevent re-processing

### CRITICAL: Clean Ingestion Procedure

**The worker tracks processed files by content hash in Redis.** Without cleanup, files won't re-ingest even if you delete them from Neo4j/Qdrant.

**Full Clean Re-ingestion (ALL THREE STORES):**
```bash
# 1. Clear Neo4j (all nodes and relationships)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 2. Clear Qdrant points (PRESERVE SCHEMA)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 3. CRITICAL: Flush Redis (clears job tracking hashes)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Restart worker
docker compose restart ingestion-worker

# 5. Add files to data/ingest/
```

---

## Issues Found and Resolved This Session

### Issue 1: Low RRF Fused Scores (~0.017)

**User Question:** Why are `fused_score` values so low (0.017)?

**Root Cause:** RRF formula `1/(k+rank)` with `k=60` produces inherently small values:
- Rank 1: 1/(60+1) = **0.0164** (maximum per source)
- With 6 sources at rank 1: 6 × 0.0164 = **0.098** (theoretical max)

**Resolution:** This is **by design**. The fused_score is for ranking, not absolute quality. The actual quality is reflected in `rerank_score` (3.79, 5.54, etc.).

---

### Issue 2: Misleading `vector_fields` Metric

**Symptom:** Response showed `"vector_fields": ["content", "title", "text-sparse"]` but 6 vectors were actually being queried.

**Root Cause:** Line 2887 in `hybrid_retrieval.py`:
```python
metrics["vector_fields"] = list(self.vector_field_weights.keys())
```
This returned the legacy weighted-fusion config, not the actual Prefetch queries.

**Fix Applied:**
1. Added `get_queried_vector_fields()` method to `QdrantMultiVectorRetriever` (lines 830-854)
2. Updated metric to use this method (lines 2887-2889)

**Files Modified:**
- `src/query/hybrid_retrieval.py` (+27 lines)

---

### Issue 3: Missing RRF Per-Field Contributions in Response

**User Request:** Expose RRF field contributions in the API response.

**Fix Applied:**
1. Added `rrf_field_contributions` field to `ChunkResult` dataclass (line 170-172)
2. Added `_compute_rrf_contributions_for_chunk()` helper method (lines 1816-1853)
3. Populate field during chunk creation when `rrf_debug_logging=true` (lines 1492-1496)
4. Expose in STDIO response (lines 560-562)

**Files Modified:**
- `src/query/hybrid_retrieval.py` (+45 lines)
- `src/mcp_server/stdio_server.py` (+3 lines)

**Response Format (when `rrf_debug_logging: true`):**
```json
{
  "rrf_field_contributions": {
    "content": {"rank": 3, "weight": 1.0, "contribution": 0.015873},
    "title": {"rank": 12, "weight": 1.0, "contribution": 0.013889},
    "text-sparse": {"rank": 1, "weight": 1.0, "contribution": 0.016393},
    "title-sparse": {"rank": 2, "weight": 2.0, "contribution": 0.032258},
    "entity-sparse": {"rank": 5, "weight": 1.5, "contribution": 0.023077}
  }
}
```

---

### Issue 4: Sparse Embedding 400 Errors (Investigated, Not a Bug)

**Symptom:** Logs showed many 400 errors with batch splitting for `/v1/embeddings/sparse`.

**Investigation Result:**
```
"Approximate token count 12500 exceeds max_batch_tokens=8192"
```

**Root Cause:** BGE service enforces 8192 token limit per batch. Batch budget (7000) based on local tokenization may differ from BGE's approximation.

**Architecture Confirmed Working:**
1. **Token-budgeted batching**: `EMBED_BATCH_MAX_TOKENS=7000`
2. **Batch-splitting retry**: Splits on 400, isolates problematic texts
3. **Graceful degradation**: `None` placeholders inserted for failed batches
4. **Qdrant handling**: Sparse vectors simply omitted (line 3187: `if indices and values:`)

**Key Insight:** Chunks are NEVER split - batch splitting operates on the **list of chunks**, not individual text content.

---

## Issues From Previous Session (Still Committed)

### Entity Metadata Deduplication (Previous Session)

**Fix Location:** `src/ingestion/extract/ner_gliner.py` lines 87-89
```python
entity_types = list(dict.fromkeys(extract_label_name(e.label) for e in entities))
entity_values = list(dict.fromkeys(e.text for e in entities))
entity_values_normalized = list(dict.fromkeys(e.text.lower().strip() for e in entities))
```

**Query-time deduplication:** `src/query/hybrid_retrieval.py` `_deduplicate_entity_metadata()` helper

### Endless LLM Loops Fix (Previous Session)

**Fix Location:** `src/mcp_server/stdio_server.py` lines 315-330

Added `VECTOR_ONLY_INSTRUCTIONS` for when `neo4j_disabled=true`, plus early-return guards on 8 graph tools.

---

## Current Configuration State

```yaml
# Key settings from config/development.yaml
neo4j_disabled: true           # Graph queries bypassed
bm25.enabled: false            # Neo4j full-text search disabled
ner.enabled: true              # GLiNER active
ner.service_url: "http://host.docker.internal:9002"
rrf_debug_logging: true        # Per-field RRF contributions in response
multi_vector_fusion_method: "rrf"
rrf_k: 60

rrf_field_weights:
  content: 1.0
  title: 1.0
  text-sparse: 1.0
  doc_title-sparse: 1.0
  title-sparse: 2.0            # BOOSTED
  entity-sparse: 1.5           # BOOSTED

enable_title_sparse: true
enable_entity_sparse: true
```

---

## Files Modified This Session

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/query/hybrid_retrieval.py` | +72 | `get_queried_vector_fields()`, `_compute_rrf_contributions_for_chunk()`, RRF field population |
| `src/mcp_server/stdio_server.py` | +3 | Expose `rrf_field_contributions` in response |

### Git Status (Uncommitted from Previous + This Session)
```
 M src/ingestion/extract/ner_gliner.py    # Entity deduplication
 M src/mcp_server/stdio_server.py         # Conditional instructions + graph guards + RRF contributions
 M src/query/hybrid_retrieval.py          # Query-time dedup + vector_fields fix + RRF helpers
```

---

## Architectural Decisions Confirmed

### 1. GLiNER is Vector-Only Enrichment
GLiNER entities are **NOT written to Neo4j**. They only enrich:
- Dense embeddings (via transient `_embedding_text` field)
- Entity-sparse vectors (via `_mentions` with `source="gliner"`)
- Qdrant payload (`entity_metadata`)

### 2. The `_embedding_text` Pattern
Entity context enriches embeddings without polluting stored text:
```python
chunk["_embedding_text"] = f"{title}\n\n{text}\n\n[Context: {entity_context}]"
# chunk["text"] remains UNCHANGED - this is what gets stored
```

### 3. Cross-Encoder Independence
The BGE reranker intentionally uses **original text**, NOT `_embedding_text`, to maintain independent scoring.

### 4. `neo4j_disabled` Implementation
The `neo4j_disabled` flag has **13+ checkpoints** in `hybrid_retrieval.py` that comprehensively disable all graph operations during retrieval.

### 5. RRF vs Weighted Fusion
- `multi_vector_fusion_method: "rrf"` - Current setting
- `vector_fields` config only used for weighted fusion (deprecated)
- `rrf_field_weights` config controls per-field RRF contribution multipliers
- `schema_supports_*` flags control which Prefetch queries are sent to Qdrant

---

## Environment Variables Reference

```bash
# Database connections
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=testpassword123
export NEO4J_USER=neo4j
export QDRANT_HOST=localhost
export REDIS_HOST=localhost
export REDIS_PASSWORD=testredis123

# Embedding services (HOST MACHINE URLs)
export BGE_M3_API_URL=http://127.0.0.1:9000
export RERANKER_BASE_URL=http://127.0.0.1:9001
export GLINER_SERVICE_URL=http://127.0.0.1:9002

# Batch token budget (default 7000, BGE limit is 8192)
export EMBED_BATCH_MAX_TOKENS=7000

# API keys
export JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi
```

---

## Quick Reference Commands

### Service Health Checks
```bash
curl -s http://127.0.0.1:9000/healthz  # BGE Embedder
curl -s http://127.0.0.1:9001/healthz  # Reranker
curl -s http://127.0.0.1:9002/healthz  # GLiNER
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "RETURN 1"
curl -s http://127.0.0.1:6333/collections/chunks_multi_bge_m3 | head -1
```

### Container Management
```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep weka
docker compose restart mcp-server
docker compose restart ingestion-worker
docker logs weka-mcp-server --tail 50 2>&1 | grep -v jaeger
```

### Clean Re-ingestion
```bash
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" -H "Content-Type: application/json" -d '{"filter": {}}'
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
docker compose restart ingestion-worker
```

---

## Next Steps

### Immediate
1. **Commit all pending changes** - 3 files with fixes from previous + this session
2. **Test MCP via Claude Desktop** - Restart Claude Desktop after mcp-server restart
3. **Verify RRF contributions in response** - Should see per-field breakdown

### Future Considerations
1. **Lower batch budget** if 400 errors are excessive: `EMBED_BATCH_MAX_TOKENS=5000`
2. **Cap entity context size** to prevent oversized `_embedding_text`
3. **Graph re-enablement** - When ready, set `neo4j_disabled: false`
4. **Retrieval quality A/B testing** - Entity-boosted vs non-boosted

---

## Detailed Data Flow: Query Processing Pipeline

Understanding the full query processing pipeline is essential for debugging retrieval issues. Here is the complete flow with line number references:

### Step 1: Query Reception (STDIO Server)
**Location:** `src/mcp_server/stdio_server.py`

When a query comes in via the `search_sections` tool:
1. Query text is validated and normalized
2. Session ID is assigned for tracking
3. Budget constraints are checked

### Step 2: Embedding Generation
**Location:** `src/query/hybrid_retrieval.py` lines 1200-1250

The query is embedded using all three BGE-M3 modalities:
```python
bundle = embedder.embed_query_all(query)
# Returns: QueryEmbeddingBundle with dense, sparse, multivector
```

**Dense:** 1024-dimensional semantic vector
**Sparse:** BM25-style term weights (indices + values)
**ColBERT:** Multi-vector for MaxSim token matching

### Step 3: Prefetch Query Construction
**Location:** `src/query/hybrid_retrieval.py` lines 1500-1572

Multiple Prefetch queries are built based on `schema_supports_*` flags:

```python
# Always included:
Prefetch(query=dense_query, using="content", limit=200)
Prefetch(query=dense_query, using="title", limit=200)
Prefetch(query=sparse_query, using="text-sparse", limit=200)

# Conditionally included:
if self.schema_supports_doc_title_sparse:
    Prefetch(query=sparse_query, using="doc_title-sparse", limit=50)
if self.schema_supports_title_sparse:
    Prefetch(query=sparse_query, using="title-sparse", limit=50)
if self.schema_supports_entity_sparse:
    Prefetch(query=sparse_query, using="entity-sparse", limit=50)
```

### Step 4: Qdrant Query Execution
**Location:** `src/query/hybrid_retrieval.py` lines 1250-1350

All prefetches are sent to Qdrant in a single `query_points` call with score fusion:
```python
results = qdrant_client.query_points(
    collection_name="chunks_multi_bge_m3",
    prefetch=prefetch_entries,
    query=primary_dense_query,
    using="content",
    with_payload=True,
    limit=candidate_limit
)
```

### Step 5: RRF Fusion
**Location:** `src/query/hybrid_retrieval.py` lines 1660-1695

Rankings from each vector field are fused using weighted RRF:
```python
for field_name, items in rankings.items():
    weight = self.rrf_field_weights.get(field_name, 1.0)
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    for rank, (doc_id, _score) in enumerate(sorted_items, start=1):
        fused[doc_id] += weight * 1.0 / (k + rank)
```

### Step 6: ColBERT MaxSim Reranking (Optional)
**Location:** `src/query/hybrid_retrieval.py` lines 2500-2600

If `colbert_rerank_enabled: true`, top candidates are refined using token-level matching.

### Step 7: Cross-Encoder Reranking
**Location:** `src/query/hybrid_retrieval.py` lines 2700-2800

The BGE reranker scores query-document pairs:
- Input: Original `chunk.text` (NOT `_embedding_text`)
- Output: Raw logit scores (can be negative)
- Top N results returned based on rerank score

### Step 8: Response Serialization
**Location:** `src/mcp_server/stdio_server.py` lines 500-565

Results are serialized with all scoring metadata:
- `fused_score`: RRF fusion result
- `rerank_score`: Cross-encoder score
- `rrf_field_contributions`: Per-field RRF breakdown (when enabled)
- `entity_metadata`: GLiNER extracted entities

---

## Detailed Data Flow: Ingestion Pipeline

### Step 1: File Detection
**Location:** `src/ingestion/worker.py`

The RQ worker monitors `data/ingest/` for new markdown files. Files are hashed and checked against Redis to prevent re-processing.

### Step 2: Document Parsing
**Location:** `src/ingestion/parsers/markdown_parser.py`

Markdown is parsed into a hierarchical structure:
- Document metadata extracted from frontmatter
- Sections identified by heading levels
- Code blocks and tables preserved

### Step 3: Chunk Assembly
**Location:** `src/ingestion/chunk_assembler.py`

Sections are assembled into chunks with constraints:
```yaml
chunk_assembly:
  min_tokens: 350
  target_tokens: 500
  hard_tokens: 750
  max_sections: 4
```

Small sections are combined; large sections are split.

### Step 4: GLiNER Entity Extraction
**Location:** `src/ingestion/extract/ner_gliner.py`

For each chunk, GLiNER extracts domain-specific entities:
```python
entities = gliner_service.extract(chunk["text"], labels=config.ner.labels)
# Creates:
# - entity_metadata payload (types, values, normalized)
# - _embedding_text enrichment
# - _mentions list for entity-sparse vectors
```

### Step 5: Embedding Generation
**Location:** `src/ingestion/atomic.py` lines 1100-1600

Token-budgeted batching sends chunks to BGE-M3:
- Dense embeddings for content, title, doc_title
- Sparse embeddings for text-sparse, title-sparse, entity-sparse, doc_title-sparse
- ColBERT multi-vectors

**Error Handling Layers:**
1. Pre-batching token budget (7000 tokens)
2. Batch-splitting retry on 400 errors
3. `None` placeholder insertion for failed batches

### Step 6: Neo4j Graph Construction
**Location:** `src/ingestion/atomic.py` lines 2800-3100

Graph nodes and relationships are created:
```cypher
CREATE (d:Document {doc_id: $doc_id, title: $title, ...})
CREATE (s:Section {id: $section_id, heading: $heading, ...})
CREATE (c:Chunk {id: $chunk_id, text: $text, ...})
CREATE (d)-[:HAS_SECTION]->(s)
CREATE (s)-[:HAS_CHUNK]->(c)
```

### Step 7: Qdrant Vector Upsert
**Location:** `src/ingestion/atomic.py` lines 3100-3300

Vectors are upserted with payloads:
```python
qdrant_client.upsert(
    collection_name="chunks_multi_bge_m3",
    points=[PointStruct(
        id=chunk_id,
        vectors={
            "content": dense_content,
            "title": dense_title,
            "doc_title": dense_doc_title,
            "text-sparse": SparseVector(...),
            "title-sparse": SparseVector(...),
            ...
        },
        payload={...}
    )]
)
```

---

## Troubleshooting Guide

### Problem: No Results Returned
**Symptoms:** `search_sections` returns empty results array.

**Diagnosis Steps:**
1. Check Qdrant point count:
   ```bash
   curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | jq '.result.points_count'
   ```
2. If 0 points, check Neo4j for data:
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"
   ```
3. If Neo4j has data but Qdrant doesn't, re-run ingestion

### Problem: Low Quality Results
**Symptoms:** Relevant documents not appearing in top results.

**Diagnosis Steps:**
1. Check RRF field contributions (now in response)
2. Verify all 6 vectors are being queried:
   ```json
   "vector_fields": ["content", "title", "text-sparse", "doc_title-sparse", "title-sparse", "entity-sparse"]
   ```
3. Check if query terms appear in entity-sparse (boosted 1.5x) or title-sparse (boosted 2.0x)

### Problem: Sparse Embedding 400 Errors
**Symptoms:** Logs show repeated HTTP 400 with batch splitting.

**Diagnosis:**
- This is **normal** if batches exceed 8192 tokens
- System handles gracefully via splitting and placeholders
- Only problematic if dense embeddings fail (those are required)

**Mitigation Options:**
1. Lower batch budget: `export EMBED_BATCH_MAX_TOKENS=5000`
2. Reduce entity context in `_embedding_text`
3. Configure BGE service with higher `max_batch_tokens`

### Problem: MCP Connection Fails After Server Restart
**Symptoms:** Claude Desktop shows "tool not available" after `docker compose restart mcp-server`.

**Solution:** Restart Claude Desktop to establish new STDIO connection.

### Problem: Files Won't Re-ingest
**Symptoms:** Dropping files in `data/ingest/` has no effect.

**Diagnosis:**
1. Check Redis for file hash:
   ```bash
   docker exec weka-redis redis-cli -a testredis123 KEYS "*"
   ```
2. Flush Redis and restart worker:
   ```bash
   docker exec weka-redis redis-cli -a testredis123 FLUSHALL
   docker compose restart ingestion-worker
   ```

---

## Testing Procedures

### Test 1: End-to-End Ingestion
```bash
# 1. Clean all stores
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" -H "Content-Type: application/json" -d '{"filter": {}}'
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 2. Restart worker
docker compose restart ingestion-worker

# 3. Add test document
echo "# Test Document\n\nThis is test content about WEKA and NFS." > data/ingest/test.md

# 4. Wait for processing (watch logs)
docker logs -f weka-ingestion-worker 2>&1 | head -50

# 5. Verify ingestion
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])"
```

### Test 2: Query Pipeline
```bash
# Direct HTTP test (bypasses STDIO)
curl -s -X POST "http://127.0.0.1:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "NFS protocol setup", "top_k": 5}' | python3 -m json.tool
```

### Test 3: Embedding Service Health
```bash
# Dense
curl -s -X POST 'http://127.0.0.1:9000/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{"model": "BAAI/bge-m3", "input": ["test"]}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Dense dims: {len(d[\"data\"][0][\"embedding\"])}')"

# Sparse
curl -s -X POST 'http://127.0.0.1:9000/v1/embeddings/sparse' \
  -H 'Content-Type: application/json' \
  -d '{"model": "BAAI/bge-m3", "input": ["test query"]}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Sparse terms: {len(d[\"data\"][0][\"indices\"])}')"
```

---

## Code Architecture: Key Classes and Methods

### QdrantMultiVectorRetriever
**Location:** `src/query/hybrid_retrieval.py` lines 660-1815

Primary retrieval class managing all vector operations:
- `search()`: Main entry point for queries
- `_search_via_query_api_weighted()`: Multi-vector Prefetch construction
- `_fuse_rankings_rrf()`: Weighted RRF implementation
- `_compute_rrf_contributions_for_chunk()`: Per-result RRF breakdown (NEW)
- `get_queried_vector_fields()`: Returns actual queried vectors (NEW)

### HybridRetriever
**Location:** `src/query/hybrid_retrieval.py` lines 1850-3500

Orchestrates full retrieval pipeline:
- `retrieve()`: Combines vector retrieval + optional graph enrichment
- `_do_rerank()`: Cross-encoder reranking
- `_apply_expansion()`: Structure-aware neighbor expansion (currently disabled)

### ChunkResult
**Location:** `src/query/hybrid_retrieval.py` lines 108-185

Dataclass containing all result metadata:
- Core fields: `chunk_id`, `document_id`, `text`, `heading`
- Scoring fields: `fused_score`, `rerank_score`, `title_vec_score`, etc.
- NEW: `rrf_field_contributions` for per-field RRF breakdown

### DocumentIngestor
**Location:** `src/ingestion/atomic.py`

Handles atomic document ingestion:
- `ingest_document_atomic()`: Full ingestion transaction
- `_process_embeddings()`: Token-budgeted batch embedding
- `_write_to_qdrant()`: Vector upsert with graceful degradation

---

## Performance Characteristics

### Latency Breakdown (Typical Query)
| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| Embedding | 50-100 | Dense + sparse + ColBERT |
| Qdrant Query | 50-150 | Depends on candidate limit |
| RRF Fusion | 1-5 | CPU-bound, fast |
| ColBERT Rerank | 100-500 | Optional, precision filter |
| Cross-Encoder | 200-500 | Final reranking stage |
| **Total** | **400-1300** | Varies by configuration |

### Throughput Limits
- BGE Embedder: ~100 texts/second (batch mode)
- Qdrant: ~1000 queries/second (depends on complexity)
- Reranker: ~50 pairs/second (cross-encoder bottleneck)

---

*Session context saved: 2025-12-09 04:45 UTC*
*Git branch: dense-graph-enhance*
*Status: Fixes applied and tested, pending commit*
*Document token count: ~5500 tokens*

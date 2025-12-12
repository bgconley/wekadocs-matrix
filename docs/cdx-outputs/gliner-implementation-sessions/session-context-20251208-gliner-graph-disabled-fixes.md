# Session Context: GLiNER Integration & Graph-Disabled Mode Fixes

**Session Date:** 2025-12-08 (evening) / 2025-12-09 (early morning UTC)
**Branch:** `dense-graph-enhance`
**Last Commit:** `d0c0e75` - "feat: complete GLiNER NER integration (Phases 1-5)"
**Document Version:** 10.0

---

## Executive Summary

This session addressed two critical issues:
1. **Entity metadata duplication** - GLiNER was storing duplicate entity values (e.g., "NFS" appearing 8x)
2. **Endless LLM loops** - MCP instructions told LLM to use graph tools that return empty results when `neo4j_disabled=true`

Both issues have been fixed with code changes pending commit.

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, and their relationships (HAS_SECTION, HAS_CHUNK, MENTIONS, NEXT_CHUNK, etc.). With `neo4j_disabled: true`, Neo4j is bypassed during retrieval but **still used during ingestion**.

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
# Direct cypher-shell access (CRITICAL: remove -it flag for non-TTY contexts)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count chunks
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"

# Example: Count all node types
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"

# Clear ALL data (for clean re-ingestion)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"
```

**Python Access from Host:**
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    print(result.single()["count"])
```

**CRITICAL FINDING THIS SESSION:** Neo4j CPU spikes (~60-190%) every 10-12 seconds are **NOT query-related** - they are internal JVM garbage collection / checkpoint maintenance. This was initially suspected as evidence of graph queries despite `neo4j_disabled=true`, but investigation confirmed it's background maintenance unrelated to queries.

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection with 8 distinct vector types per chunk.

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

**Payload Indexes (28+ total):** Includes 4 entity metadata indexes:
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

Redis serves two purposes: L2 query result caching and RQ job queue for async ingestion. **Understanding Redis's role is critical for debugging ingestion issues.**

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

# Flush all data (REQUIRED for clean re-ingestion)
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

Provides dense (1024-D), sparse (BM25-style), and ColBERT multi-vector embeddings.

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
curl -s http://127.0.0.1:9000/healthz
# Returns: {"status":"ok"}

# Generate dense embeddings
curl -s -X POST "http://127.0.0.1:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": ["test query"]}'
```

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

### Issue 1: Entity Metadata Duplication

**Symptom:** `entity_metadata.entity_values` contained duplicates like "NFS" appearing 8 times.

**Root Cause:** In `src/ingestion/extract/ner_gliner.py`, `entity_types` was deduplicated with `set()` but `entity_values` and `entity_values_normalized` were raw lists from GLiNER output.

**Fix Applied:**
1. **Ingestion-time deduplication** (`ner_gliner.py` lines 87-89):
   ```python
   entity_types = list(dict.fromkeys(extract_label_name(e.label) for e in entities))
   entity_values = list(dict.fromkeys(e.text for e in entities))
   entity_values_normalized = list(dict.fromkeys(e.text.lower().strip() for e in entities))
   ```

2. **Query-time deduplication** (`hybrid_retrieval.py`):
   - Added `_deduplicate_entity_metadata()` helper function (lines 186-213)
   - Applied at both retrieval points (lines 1137, 5525)

**Result:** Entity values are now unique, existing data cleaned at query time.

---

### Issue 2: Endless LLM Loops with Graph Disabled

**Symptom:** LLM kept making tool calls endlessly when using MCP.

**Root Cause:** `GRAPH_FIRST_INSTRUCTIONS` told the LLM to:
> "explore the neighborhood with expand_neighbors, get_paths_between, describe_nodes, list_children, list_parents..."

But with `neo4j_disabled: true`, these tools returned empty results, causing the LLM to loop endlessly trying to "map the graph."

**Fix Applied:**
1. **Conditional instructions** (`stdio_server.py` lines 315-330):
   - Added `VECTOR_ONLY_INSTRUCTIONS` for when graph is disabled
   - Auto-selects based on `neo4j_disabled` config

2. **Graph tool guards** (`stdio_server.py`):
   - Added early-return guards to 8 graph tools
   - Returns clear error message: "Graph traversal is disabled (neo4j_disabled=true)"

**Tools Protected:**
- `traverse_relationships` (line 631)
- `describe_nodes` (line 714)
- `expand_neighbors` (line 760)
- `get_paths_between` (line 811)
- `list_children` (line 854)
- `list_parents` (line 892)
- `get_entities_for_sections` (line 927)
- `get_sections_for_entities` (line 966)

**New Instructions (when neo4j_disabled=true):**
```
"You are connected to the Weka docs via MCP vector search tools.
Graph traversal is DISABLED - do NOT use expand_neighbors, get_paths_between,
list_children, list_parents, or traverse_relationships.
Use search_sections to find relevant documentation, then get_section_text to retrieve content.
Complete your response in 2-3 tool calls maximum."
```

---

### Issue 3: entity_metadata: null Bug (Previous Session)

**Symptom:** Retrieval results showed `entity_metadata: null` despite data existing in Qdrant.

**Root Cause:** `entity_metadata` was not in the `payload_keys` allowlist in `QdrantMultiVectorRetriever`.

**Fix Applied:** Added `"entity_metadata"` to `payload_keys` list at line 778-779.

---

## Architectural Decisions

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

### 4. neo4j_disabled Implementation

The `neo4j_disabled` flag has **13+ checkpoints** in `hybrid_retrieval.py` that comprehensively disable all graph operations during retrieval.

---

## Current System State

### What Works
- GLiNER entity extraction with MPS acceleration (65x faster)
- Entity-enriched embeddings
- Entity-sparse vectors with 1.5x RRF weight
- Entity metadata deduplication (both ingestion and query time)
- `neo4j_disabled` properly bypassing graph operations
- MCP instructions conditional on graph availability
- Graph tools return early with clear error when disabled

### Current Configuration
```yaml
neo4j_disabled: true    # Graph queries bypassed
bm25.enabled: false     # Neo4j full-text search disabled
ner.enabled: true       # GLiNER active
ner.service_url: "http://host.docker.internal:9002"
```

### Data State
- 54 Documents ingested
- 288 Qdrant points with 8 vectors each
- ~1,100+ entities extracted
- Entity metadata now deduplicated

---

## Files Modified This Session

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/ingestion/extract/ner_gliner.py` | +4/-3 | Ingestion-time entity deduplication |
| `src/query/hybrid_retrieval.py` | +38/-2 | Query-time entity deduplication helper + calls |
| `src/mcp_server/stdio_server.py` | +94/-1 | Conditional instructions + graph tool guards |

### Git Status (Uncommitted)
```
 src/ingestion/extract/ner_gliner.py |  7 +--
 src/mcp_server/stdio_server.py      | 94 +++++++++++++++++++++++++++++
 src/query/hybrid_retrieval.py       | 38 +++++++++++-
 3 files changed, 132 insertions(+), 7 deletions(-)
```

---

## Next Steps

### Immediate
1. **Commit the fixes** - All three files need to be committed
2. **Test MCP with Claude Desktop** - Verify instructions mode and graph tool guards
3. **Test entity deduplication** - Verify search results show unique entity values

### Future Considerations
1. **Re-ingestion for clean entity data** - Optional, query-time dedup handles existing data
2. **Turn limiting** - Consider adding session-level turn counter if LLM still loops
3. **Retrieval quality testing** - A/B test entity-boosted vs non-boosted retrieval
4. **Graph re-enablement** - When ready to enable graph, update config and test

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

*Session context saved: 2025-12-09 03:55 UTC*
*Git branch: dense-graph-enhance*
*Status: Fixes applied, pending commit*

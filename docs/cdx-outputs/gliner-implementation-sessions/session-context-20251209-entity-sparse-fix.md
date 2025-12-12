# Session Context: Entity-Sparse Vector Fix & RRF Configuration

**Session Date:** 2025-12-09
**Branch:** `dense-graph-enhance`
**Last Commit:** `d0c0e75` - "feat: complete GLiNER NER integration (Phases 1-5)"
**Document Version:** 1.0
**Previous Context:** `session-context-20251207-vector-architecture-reform.md`

---

## Executive Summary

This session diagnosed and fixed three critical bugs that were causing poor retrieval quality:

1. **Bug 1 (atomic.py:550-572):** GLiNER-extracted `_mentions` were being overwritten by structural mentions during ingestion
2. **Bug 2 (atomic.py:1520-1525):** Entity name lookup for sparse embeddings ignored GLiNER's direct `name` field
3. **Bug 3 (config.py:297):** `multi_vector_fusion_method` field was missing from Pydantic schema, causing RRF to silently fall back to weighted fusion

All three fixes have been implemented, verified in production containers, and databases have been cleared for re-ingestion.

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, and their relationships (HAS_SECTION, HAS_CHUNK, MENTIONS, NEXT_CHUNK, etc.). With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval but **still used during ingestion** for storing document structure.

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

# Clear ALL data (preserves schema - indexes and constraints remain)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# Check indexes (verify schema preserved after data clear)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "SHOW INDEXES"

# Check constraints
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "SHOW CONSTRAINTS"
```

**Python Access from Host:**
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    print(result.single()["count"])
```

**CRITICAL FINDING:** Neo4j CPU spikes (~60-190%) every 10-12 seconds are **NOT query-related** - they are internal JVM garbage collection / checkpoint maintenance. This was investigated in a previous session and confirmed as background maintenance unrelated to queries.

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection with **8 distinct vector types per chunk** (6 actively queried for RRF).

**Connection Details:**
```
REST API: http://127.0.0.1:6333
gRPC endpoint: localhost:6334
Authentication: None (development mode)
Collection: chunks_multi_bge_m3
Container name: weka-qdrant
Docker service name: qdrant
```

**Current Vector Schema (8 vectors per chunk, 6 in RRF):**

| Vector Name | Type | Dimensions | RRF Weight | Purpose |
|-------------|------|------------|------------|---------|
| `content` | Dense | 1024 | 1.0 | Main semantic content embedding |
| `title` | Dense | 1024 | 1.0 | Section heading semantic embedding |
| `doc_title` | Dense | 1024 | N/A | Document title semantic embedding |
| `late-interaction` | Dense (multi) | 1024 × N | N/A | ColBERT MaxSim token-level matching |
| `text-sparse` | Sparse | Variable | 1.0 | BM25-style lexical content matching |
| `title-sparse` | Sparse | Variable | **2.0** | Section heading lexical matching (BOOSTED) |
| `doc_title-sparse` | Sparse | Variable | 1.0 | Document title lexical matching |
| `entity-sparse` | Sparse | Variable | **1.5** | Entity name lexical matching (BOOSTED) |

**Access Patterns:**
```bash
# Check collection status and point count
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}\")"

# Delete all points (PRESERVE SCHEMA - critical for re-ingestion)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# Check entity-sparse vector status for a specific chunk
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
results, _ = client.scroll("chunks_multi_bge_m3", limit=5, with_payload=["heading", "entity_metadata"], with_vectors=["entity-sparse"])
for p in results:
    sparse = p.vector.get("entity-sparse", {})
    terms = len(sparse.get("indices", [])) if isinstance(sparse, dict) else 0
    print(f"{p.payload.get('heading', '')[:40]}: {terms} sparse terms")
EOF
```

---

### Redis Cache and Queue

Redis serves two purposes: **L2 query result caching** and **RQ job queue** for async ingestion. **CRITICAL:** Redis tracks processed files by content hash - must flush before re-ingesting same documents.

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
- **ALWAYS flush Redis** before re-ingesting the same documents (worker tracks by content hash)
- Debugging stuck ingestion jobs
- Cache invalidation during development

**When NOT to Access Redis:**
- Vector retrieval testing (Redis not in retrieval path)
- Schema verification (Qdrant only)
- Graph queries (Neo4j only)

---

### BGE-M3 Embedding Service

Provides **dense (1024-D)**, **sparse (BM25-style)**, and **ColBERT multi-vector** embeddings. Native macOS service running outside Docker.

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

**CRITICAL - Token Limits:**
```
max_batch_tokens = 8192 (enforced by BGE service)
EMBED_BATCH_MAX_TOKENS = 7000 (configured in ingestion pipeline)
```

When a batch exceeds the BGE limit, it returns HTTP 400. The system handles this gracefully via batch-splitting retry.

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

---

### GLiNER NER Service (Native MPS-Accelerated)

Native macOS service with Metal Performance Shaders acceleration for entity extraction.

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

The MCP server and ingestion containers mount source code as **live volumes**:
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:rw
```

**This means:**
- **Code changes are immediately visible** inside containers (same filesystem)
- **NO rebuild needed for code changes** - just restart the container
- **Rebuild only needed** for requirements.txt or Dockerfile changes

### When to Rebuild vs Restart

| Change Type | Action Required | Command |
|-------------|-----------------|---------|
| Code in `src/` | Restart only | `docker compose restart mcp-server ingestion-worker` |
| Config in `config/` | Restart only | `docker compose restart mcp-server ingestion-worker` |
| requirements.txt | Full rebuild | `docker compose build mcp-server && docker compose up -d` |
| Dockerfile changes | Full rebuild | `docker compose build && docker compose up -d` |

### Verifying Code is Active in Containers

```bash
# Compare checksums between host and container
echo "Host: $(md5 -q src/ingestion/atomic.py)"
echo "Container: $(docker exec weka-ingestion-worker md5sum /app/src/ingestion/atomic.py | awk '{print $1}')"

# Check for specific fix markers
docker exec weka-ingestion-worker grep -c "Merge structural mentions" /app/src/ingestion/atomic.py
```

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
# 1. Clear Neo4j (all nodes and relationships - schema preserved)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 2. Clear Qdrant points (PRESERVE SCHEMA)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 3. CRITICAL: Flush Redis (clears job tracking hashes)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Restart worker to pick up clean state
docker compose restart ingestion-worker

# 5. Add files to data/ingest/ (or they're already there)
```

---

## Bugs Discovered and Fixed This Session

### Bug 1: GLiNER Mentions Overwritten (CRITICAL)

**Location:** `src/ingestion/atomic.py` line 558 (now 550-572)

**Symptom:** Entity-sparse vectors were always empty despite GLiNER extracting entities correctly.

**Root Cause:** GLiNER enrichment runs in `_prepare_ingestion()` and adds `_mentions` to each section. Then in `ingest_document_atomic()`, structural mentions were attached via direct assignment:
```python
section["_mentions"] = unique_mentions  # OVERWROTE GLiNER mentions!
```

**Fix Applied:**
```python
# Merge structural mentions with any existing GLiNER mentions
existing_gliner_mentions = section.get("_mentions", [])

# Deduplicate by entity_id across both sources
seen_entity_ids = set()
merged_mentions = []

# Add GLiNER mentions first (higher quality - model-extracted)
for m in existing_gliner_mentions:
    eid = m.get("entity_id")
    if eid and eid not in seen_entity_ids:
        seen_entity_ids.add(eid)
        merged_mentions.append(m)

# Then add structural mentions (regex-extracted)
for m in section_mentions:
    eid = m.get("entity_id")
    if eid and eid not in seen_entity_ids:
        seen_entity_ids.add(eid)
        merged_mentions.append(m)

section["_mentions"] = merged_mentions
```

---

### Bug 2: Entity Name Lookup Ignored GLiNER's Direct Name Field

**Location:** `src/ingestion/atomic.py` line 1520-1525

**Symptom:** Even if Bug 1 were fixed, entity-sparse embeddings would still be empty because the name lookup failed.

**Root Cause:** The code looked up entity names via `entity_id_to_name` dict:
```python
if entity_id and entity_id in entity_id_to_name:
    entity_names.append(entity_id_to_name[entity_id])
```

But GLiNER mentions have entity IDs like `gliner:cli_command:abc123` which aren't in the lookup. GLiNER stores the name **directly on the mention object** in the `name` field.

**Fix Applied:**
```python
for m in section_mentions:
    # First: check for direct 'name' field (GLiNER mentions)
    if m.get("name"):
        entity_names.append(m["name"])
    # Fallback: lookup by entity_id (structural entities)
    elif m.get("entity_id") in entity_id_to_name:
        entity_names.append(entity_id_to_name[m["entity_id"]])
```

---

### Bug 3: Pydantic Schema Missing `multi_vector_fusion_method`

**Location:** `src/shared/config.py` line 297

**Symptom:** API response showed `rrf_field_contributions: null` for all results, and RRF wasn't actually being used.

**Root Cause:** The YAML config had:
```yaml
multi_vector_fusion_method: "rrf"
```

But `HybridSearchConfig` Pydantic model didn't have this field defined. Pydantic silently ignores unknown fields, so the code fell back to the default `"weighted"` fusion method.

**Fix Applied:**
```python
class HybridSearchConfig(BaseModel):
    # ... existing fields ...
    method: str = "rrf"
    # Multi-vector fusion method: how to combine scores from multiple vector fields
    # "rrf" = Reciprocal Rank Fusion (rank-based, robust to score scale differences)
    # "weighted" = Weighted sum (uses vector_fields weights)
    multi_vector_fusion_method: str = "rrf"
    # ... rest of fields ...
```

---

## Current Configuration State

```yaml
# Key settings from config/development.yaml
neo4j_disabled: true           # Graph queries bypassed in retrieval
bm25.enabled: false            # Neo4j full-text search disabled
ner.enabled: true              # GLiNER active
ner.service_url: "http://host.docker.internal:9002"
rrf_debug_logging: true        # Per-field RRF contributions in response
multi_vector_fusion_method: "rrf"
rrf_k: 60

# All 6 vector fields active in RRF
rrf_field_weights:
  content: 1.0
  title: 1.0
  text-sparse: 1.0
  doc_title-sparse: 1.0
  title-sparse: 2.0            # BOOSTED for heading matches
  entity-sparse: 1.5           # BOOSTED for entity matches

enable_sparse: true
enable_title_sparse: true
enable_entity_sparse: true
enable_doc_title_sparse: true
```

---

## Files Modified This Session

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/ingestion/atomic.py` | 550-572, 1520-1525 | Bug 1 + Bug 2 fixes |
| `src/shared/config.py` | 294-297 | Bug 3 fix (added `multi_vector_fusion_method` field) |

### Git Status (Uncommitted Changes)
```
M src/ingestion/atomic.py
M src/shared/config.py
M src/ingestion/extract/ner_gliner.py    # From previous session
M src/mcp_server/stdio_server.py         # From previous session
M src/query/hybrid_retrieval.py          # From previous session
```

---

## What Works vs What's Currently Broken

### ✅ Working

1. **All 3 bug fixes implemented and verified in production containers**
2. **All 6 vector fields configured and active in RRF fusion**
3. **GLiNER service running with MPS acceleration**
4. **BGE embedder and reranker services healthy**
5. **Neo4j, Qdrant, Redis all operational**
6. **Database schema preserved after data clear**
7. **Volume mounts working - code changes immediately visible in containers**

### ⚠️ Requires Action

1. **Corpus needs re-ingestion** - Existing chunks have empty entity-sparse vectors due to Bug 1 + Bug 2
2. **MCP STDIO connection needs restart** - After mcp-server restart, Claude Desktop must be restarted
3. **Uncommitted changes** - Need to commit the fixes before they could be lost

### 🔄 Current State

- **Databases cleared:** Neo4j (0 nodes), Qdrant (0 points), Redis (0 keys)
- **Schema preserved:** All indexes and constraints intact
- **Ready for re-ingestion:** Worker restarted with fixed code

---

## Next Steps

1. **Start re-ingestion** - Drop files in `data/ingest/` or trigger via API
2. **Monitor ingestion logs** - `docker logs -f weka-ingestion-worker`
3. **Verify entity-sparse populated** - Check chunks have non-empty entity-sparse vectors
4. **Test retrieval quality** - Query "inodes metadata architecture" should now find relevant docs
5. **Commit changes** - All three fixes need to be committed
6. **Restart Claude Desktop** - To establish new STDIO connection with MCP server

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

### 4. Dual Config for Legacy Compatibility
- `vector_fields` - Legacy weighted fusion config (used for `dense_vector_names`)
- `rrf_field_weights` - RRF-specific weights for all 6 vectors

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
docker compose restart mcp-server ingestion-worker
docker logs weka-ingestion-worker --tail 50 2>&1 | grep -v jaeger
```

### Verify Fixes in Containers
```bash
# Check checksums match
md5 -q src/ingestion/atomic.py
docker exec weka-ingestion-worker md5sum /app/src/ingestion/atomic.py

# Check fix markers present
docker exec weka-ingestion-worker grep -c "Merge structural mentions" /app/src/ingestion/atomic.py
docker exec weka-mcp-server grep -c "multi_vector_fusion_method: str" /app/src/shared/config.py
```

---

*Session context saved: 2025-12-09 02:30 EST*
*Git branch: dense-graph-enhance*
*Status: Fixes applied, databases cleared, ready for re-ingestion*
*Document token count: ~5200 tokens*

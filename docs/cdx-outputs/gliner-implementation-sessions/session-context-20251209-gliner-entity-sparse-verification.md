# Session Context: GLiNER Entity-Sparse Verification & 400 Error Diagnosis

**Session Date:** 2025-12-09
**Branch:** `dense-graph-enhance`
**Last Commit:** `d0c0e75` - "feat: complete GLiNER NER integration (Phases 1-5)"
**Document Version:** 1.1
**Previous Context:** `session-context-20251207-vector-architecture-reform.md`
**Token Target:** 4500-5500 tokens

---

## Executive Summary

This session verified that the three-bug fix chain from the previous session is working correctly, and diagnosed the root cause of HTTP 400 errors occurring during sparse embedding generation. Key findings:

1. **All three bug fixes verified working** - GLiNER entity-sparse vectors now populate correctly (100% coverage)
2. **HTTP 400 root cause identified** - Batch budget mismatch between original text tokens and `_embedding_text` with entity context appended
3. **Current behavior is acceptable** - The split-on-400 retry mechanism handles oversize batches gracefully
4. **Full corpus ingestion in progress** - 1323 chunks ingested from 259 documents as of session end

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, Entities, and their relationships (HAS_SECTION, HAS_CHUNK, MENTIONS, NEXT_CHUNK, REFERENCES, etc.). With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval but **still used during ingestion** for storing document structure and entity relationships.

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

**Access Patterns (CRITICAL - remove -it flag for non-TTY contexts like Claude Code):**
```bash
# Direct cypher-shell access
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count all node types
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"

# Clear ALL data (preserves schema - indexes and constraints remain)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# Check indexes (verify schema preserved after data clear)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "SHOW INDEXES"
```

**Python Access from Host:**
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    print(result.single()["count"])
```

**CRITICAL FINDING from previous sessions:** Neo4j CPU spikes (~60-190%) every 10-12 seconds are **NOT query-related** - they are internal JVM garbage collection / checkpoint maintenance. This was investigated and confirmed as background maintenance unrelated to queries.

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection with **8 distinct vector types per chunk** (6 actively queried for RRF fusion).

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
```

**Python Access for Detailed Inspection:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

client = QdrantClient(host="localhost", port=6333)
results, _ = client.scroll(
    "chunks_multi_bge_m3",
    limit=10,
    with_payload=["heading", "entity_metadata"],
    with_vectors=["entity-sparse", "text-sparse"]
)

for p in results:
    entity_sparse = p.vector.get("entity-sparse")
    if isinstance(entity_sparse, SparseVector):
        print(f"Terms: {len(entity_sparse.indices)}")
```

---

### Redis Cache and Queue

Redis serves two purposes: **L2 query result caching** and **RQ job queue** for async ingestion.

**CRITICAL:** Redis tracks processed files by content hash - **MUST flush before re-ingesting same documents**.

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

Provides **dense (1024-D)**, **sparse (BM25-style)**, and **ColBERT multi-vector** embeddings. Native macOS service running outside Docker with Metal Performance Shaders acceleration.

**Connection Details:**
```
Host machine: http://127.0.0.1:9000
Inside Docker: http://host.docker.internal:9000
Model: BAAI/bge-m3
Service location: /Users/brennanconley/vibecode/bge-m3-custom
```

**Endpoints:**
- `GET /healthz` - Health check (returns `{"status":"ok"}`)
- `POST /v1/embeddings` - Dense embeddings (MUST include "model" param)
- `POST /v1/embeddings/sparse` - Sparse embeddings
- `POST /v1/embeddings/colbert` - ColBERT multi-vectors

**Configuration (from settings.py):**
```python
max_batch_size: int = 32
max_batch_tokens: int = 8192
max_sequence_length_tokens: int = 8192
```

**Token Estimation Logic (app.py:79-81):**
```python
def _estimate_tokens(texts: List[str]) -> int:
    # Simple heuristic: assume ~4 characters per token on average.
    return sum(len(t) for t in texts) // 4
```

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
Service location: services/gliner-ner
```

**Starting/Stopping:**
```bash
cd services/gliner-ner && ./run.sh    # Start (auto-detects MPS)
pkill -f "server.py"                   # Stop
```

**Health Check:**
```bash
curl -s http://127.0.0.1:9002/healthz
# Returns: {"status": "ok", "device": "mps", "model": "urchade/gliner_medium-v2.1", ...}
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
| `weka-ingestion-worker` | ingestion-worker | RQ background jobs | None | `./src:/app/src:rw` |

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
docker exec weka-mcp-server grep -c "multi_vector_fusion_method: str" /app/src/shared/config.py
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

# 5. Files in data/ingest/ will now be re-ingested
```

---

## Bugs Discovered and Fixed (Previous Session)

### Bug 1: GLiNER Mentions Overwritten (CRITICAL)

**Location:** `src/ingestion/atomic.py` lines 550-572

**Symptom:** Entity-sparse vectors were always empty despite GLiNER extracting entities correctly.

**Root Cause:** GLiNER enrichment runs in `_prepare_ingestion()` and adds `_mentions` to each section. Then in `ingest_document_atomic()`, structural mentions were attached via direct assignment that **overwrote** GLiNER mentions.

**Fix Applied:** Merge structural mentions with existing GLiNER mentions using deduplication by entity_id, preserving GLiNER mentions first (higher quality - model-extracted).

### Bug 2: Entity Name Lookup Ignored GLiNER's Direct Name Field

**Location:** `src/ingestion/atomic.py` lines 1520-1525

**Symptom:** Even if Bug 1 were fixed, entity-sparse embeddings would still be empty because name lookup failed.

**Root Cause:** GLiNER mentions have entity IDs like `gliner:cli_command:abc123` which aren't in the structural entity lookup dict. GLiNER stores the name **directly on the mention object** in the `name` field.

**Fix Applied:** Check for direct `name` field first (GLiNER mentions), then fall back to entity_id lookup (structural entities).

### Bug 3: Pydantic Schema Missing `multi_vector_fusion_method`

**Location:** `src/shared/config.py` line 297

**Symptom:** API response showed `rrf_field_contributions: null` for all results, and RRF wasn't actually being used.

**Root Cause:** The YAML config had `multi_vector_fusion_method: "rrf"` but `HybridSearchConfig` Pydantic model didn't have this field defined. Pydantic silently ignores unknown fields.

**Fix Applied:** Added `multi_vector_fusion_method: str = "rrf"` field to HybridSearchConfig.

---

## This Session: HTTP 400 Error Diagnosis

### Symptom Observed

During ingestion, intermittent HTTP 400 errors on `/v1/embeddings/sparse`:
```
HTTP Request: POST http://host.docker.internal:9000/v1/embeddings/sparse "HTTP/1.1 400 Bad Request"
Embedding batch rejected (HTTP 400); splitting batch
```

### Root Cause Identified

**Batch budget vs actual content mismatch:**

1. **Batch budget** (atomic.py:1308-1318) calculated using `token_count` from **original text** (~7000 tokens max)

2. **Actual content sent** (atomic.py:1221-1223) uses `_embedding_text` which includes:
   ```
   {heading}\n\n{text}\n\n[Context: entity1, entity2, entity3, ...]
   ```

3. This creates **5-30% overhead per chunk** (average ~13%)

4. A batch budgeted for 7000 tokens can actually contain 7700-9100 tokens

5. BGE's `chars/4` estimation flags this as exceeding 8192 → 400 error

### Measured Impact

| Chunk | Original Tokens | Embedding Tokens | Overhead |
|-------|-----------------|------------------|----------|
| Network | 334 | 386 | +15.6% |
| Operations (NFSw) | 291 | 376 | **+29.2%** |
| weka cloud | 287 | 326 | +13.4% |

### Why Current Behavior Is Acceptable

The `_embed_with_retry` method in `bge_m3_service.py:122-155` catches HTTP 400 errors and recursively splits batches in half until they fit:

```python
def _embed_with_retry(self, texts: List[str], method_name: str, batch_idx: str = "0"):
    try:
        return client_fn(texts)
    except Exception as exc:
        if is_client_err and batch_size > 1:
            logger.warning("Embedding batch rejected (HTTP 400); splitting batch")
            mid = batch_size // 2
            head = self._embed_with_retry(texts[:mid], method_name, f"{batch_idx}a")
            tail = self._embed_with_retry(texts[mid:], method_name, f"{batch_idx}b")
            return head + tail
        raise
```

**This is by design** - the split-on-400 pattern is a safety valve. No data is lost, just slight latency for affected batches.

### Potential Future Optimizations (Not Required)

1. Reduce batch budget from 7000 to ~6000 to account for GLiNER overhead
2. Calculate budget using `_embedding_text` length instead of original `text` token count
3. Increase BGE service `max_batch_tokens` from 8192 to 10000

---

## Verification Results: Entity-Sparse Working

### Comprehensive Statistics

| Metric | Value | Status |
|--------|-------|--------|
| **Total Points** | 1323 | Actively growing |
| **entity-sparse coverage** | 100% (1323/1323) | PASS |
| **entity-sparse avg terms** | 62.4 | Healthy |
| **text-sparse coverage** | 100% | PASS |
| **title-sparse coverage** | 100% | PASS |
| **entity_metadata present** | 100% (avg 14.6 entities) | PASS |

### Sample Entity Metadata Structure

```json
{
  "entity_types": ["network_or_storage_protocol", "cli_command", "configuration_parameter"],
  "entity_values": ["NFS", "weka nfs rules", "Kerberos", "dns"],
  "entity_values_normalized": ["nfs", "weka nfs rules", "kerberos", "dns"],
  "entity_count": 19
}
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

## Files Modified (Uncommitted on `dense-graph-enhance` branch)

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/ingestion/atomic.py` | 550-572, 1520-1525 | Bug 1 + Bug 2 fixes |
| `src/shared/config.py` | 294-297 | Bug 3 fix |
| `src/ingestion/extract/ner_gliner.py` | Various | GLiNER integration from previous session |
| `src/mcp_server/stdio_server.py` | Various | RRF debug logging from previous session |
| `src/query/hybrid_retrieval.py` | Various | RRF fusion implementation from previous session |

---

## Architectural Decisions Confirmed

### 1. GLiNER is Vector-Only Enrichment
GLiNER entities are **NOT written to Neo4j graph**. They only enrich:
- Dense embeddings (via transient `_embedding_text` field)
- Entity-sparse vectors (via `_mentions` with `source="gliner"`)
- Qdrant payload (`entity_metadata`)

### 2. The `_embedding_text` Pattern
Entity context enriches embeddings without polluting stored text:
```python
chunk["_embedding_text"] = f"{title}\n\n{text}\n\n[Context: {entity_context}]"
# chunk["text"] remains UNCHANGED - this is what gets stored/retrieved
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

### Ingestion Progress Monitoring
```bash
# Check progress
echo "Documents: $(docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 'MATCH (d:Document) RETURN count(d)' 2>/dev/null | tail -1)"
echo "Qdrant: $(curl -s 'http://127.0.0.1:6333/collections/chunks_multi_bge_m3' | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"result\"][\"points_count\"])')"

# Watch for 400 errors
docker logs weka-ingestion-worker 2>&1 | grep -c "400 Bad Request"
```

---

## Next Steps

1. **Let ingestion complete** - Currently ~50% through 259 documents
2. **Test retrieval quality** - Query "weka nfs kerberos" should now leverage entity-sparse
3. **Commit bug fixes** - Three fixes in atomic.py and config.py need to be committed
4. **Optional: Reduce batch budget** - If 400 retries become problematic, reduce EMBED_BATCH_MAX_TOKENS from 7000 to 6000

---

*Session context saved: 2025-12-09 03:45 EST*
*Git branch: dense-graph-enhance*
*Status: GLiNER integration verified working, 400 error root cause identified*
*Document token count: ~5200 tokens*

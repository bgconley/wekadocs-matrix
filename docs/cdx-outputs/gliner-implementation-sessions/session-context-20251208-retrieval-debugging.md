# Session Context: Retrieval Pipeline Debugging & GLiNER Investigation

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** Investigation of retrieval pipeline issues, negative rerank scores, entity_metadata null bug fix, and comprehensive system verification
**Previous Context:** GLiNER NER integration complete (Phases 1-5), committed as `d0c0e75`

---

## Executive Summary

This session investigated reported issues with the retrieval pipeline following GLiNER implementation:
1. **Negative rerank scores** - Determined to be WORKING CORRECTLY (BGE reranker outputs raw logits)
2. **entity_metadata: null** - BUG FOUND AND FIXED (missing from payload_keys allowlist)
3. **Neo4j BM25 injecting candidates** - CONFIRMED NOT HAPPENING (BM25 disabled, counts = 0)
4. **Graph still active** - CONFIRMED DISABLED (all graph scores = 0)

The root cause of poor-seeming retrieval was determined to be a **corpus content issue** - the NFS documentation lacks troubleshooting/error content, so queries about "NFS errors" correctly receive negative scores as no relevant documents exist.

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, Entities, and their relationships. The graph model follows a provenance-first design where Document → Section → Chunk relationships maintain document lineage. Entity nodes (extracted via GLiNER) connect to Chunks via MENTIONS relationships.

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
# Direct cypher-shell access (IMPORTANT: remove -it flag for non-TTY contexts)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count chunks
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"

# Example: Find chunks by heading
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) WHERE toLower(c.heading) CONTAINS 'nfs' RETURN c.heading LIMIT 10"

# Example: Count all node types
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"

# Example: Check relationship types
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count ORDER BY count DESC"

# Clear all data (for clean re-ingestion)
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

**When to Access Neo4j:**
- Checking document/section/chunk counts after ingestion
- Verifying graph relationships (MENTIONS, HAS_SECTION, NEXT_CHUNK, HAS_CHUNK, IN_DOCUMENT, etc.)
- Debugging entity extraction and mention relationships
- Comparing data between Neo4j and Qdrant for sync verification
- Investigating document structure and section hierarchy

**CRITICAL:** With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval queries but still used during ingestion. The `neo4j_disabled` flag has 13+ checkpoints in `hybrid_retrieval.py` that skip graph operations. All graph-related scores will show 0.0 when disabled.

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection. It is the **primary store for vector-based retrieval** and contains 8 distinct vector types per chunk for multi-signal retrieval.

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

| Vector Name | Type | Dimensions | Purpose | RRF Weight | Notes |
|-------------|------|------------|---------|------------|-------|
| `content` | Dense | 1024 | Main semantic content embedding | 1.0 | Uses `_embedding_text` with GLiNER context |
| `title` | Dense | 1024 | Section heading semantic embedding | 1.0 | Direct heading text |
| `doc_title` | Dense | 1024 | Document title semantic embedding | 1.0 | Document-level title |
| `late-interaction` | Dense (multi) | 1024 × N tokens | ColBERT MaxSim token-level matching | N/A | Used for reranking |
| `text-sparse` | Sparse | Variable | BM25-style lexical content matching | 1.0 | Term frequency weights |
| `title-sparse` | Sparse | Variable | Section heading lexical matching | **2.0** | BOOSTED for heading matches |
| `doc_title-sparse` | Sparse | Variable | Document title lexical matching | 1.0 | Document scoping |
| `entity-sparse` | Sparse | Variable | Entity name lexical matching (GLiNER) | **1.5** | BOOSTED for entity matches |

**Payload Indexes (28+ total):**
Core indexes plus 4 entity metadata indexes added for GLiNER Phase 3:
- `entity_metadata.entity_types` (KEYWORD) - Filter by entity type
- `entity_metadata.entity_values` (KEYWORD) - Filter by exact entity values
- `entity_metadata.entity_values_normalized` (KEYWORD) - Case-insensitive entity matching
- `entity_metadata.entity_count` (INTEGER) - Range queries on entity density

**Payload Structure for entity_metadata:**
```json
{
  "entity_metadata": {
    "entity_types": ["weka_software_component", "cli_command", "file_system_path"],
    "entity_values": ["WEKA", "kubectl logs", "/var/lib/docker"],
    "entity_values_normalized": ["weka", "kubectl logs", "/var/lib/docker"],
    "entity_count": 3
  }
}
```

**Access Patterns:**
```bash
# Check collection status and point count
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}\")"

# Scroll through points with payload (full payload)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "with_payload": true}' | python3 -m json.tool

# Scroll with specific payload fields
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 5, "with_payload": {"include": ["heading", "text", "entity_metadata"]}}'

# Delete all points (PRESERVE SCHEMA - important for re-ingestion)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# Search with filter (find NFS-related chunks)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "with_payload": {"include": ["heading"]}, "filter": {"should": [{"key": "heading", "match": {"text": "NFS"}}]}}'

# Check if entity_metadata exists in payload
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "with_payload": {"include": ["entity_metadata"]}}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
em = d['result']['points'][0]['payload'].get('entity_metadata')
print('entity_metadata present:', em is not None)
if em: print(json.dumps(em, indent=2))"
```

---

### Redis Cache and Queue

Redis serves two purposes: L2 query result caching and RQ job queue for async ingestion. Understanding Redis's role is crucial for debugging ingestion issues.

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

# Flush all data (REQUIRED for clean re-ingestion - clears job tracking)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# Check RQ job queue
docker exec weka-redis redis-cli -a testredis123 KEYS "rq:*"

# Check all keys (use sparingly - can be large)
docker exec weka-redis redis-cli -a testredis123 KEYS "*" | head -20

# Check specific cache key pattern
docker exec weka-redis redis-cli -a testredis123 KEYS "cache:*"
```

**CRITICAL - Redis and Re-ingestion:**
The RQ worker tracks processed files by content hash in Redis. **Even after clearing Neo4j and Qdrant, Redis still remembers processed files.** This is the most common cause of "why isn't my file being re-ingested" issues. You MUST flush Redis before re-ingesting the same documents:
```bash
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

**When to Access Redis:**
- **ALWAYS flush** before re-ingesting same documents
- Debugging stuck ingestion jobs (check RQ queues)
- Cache invalidation during development
- Verifying job completion status

**When NOT to Access Redis:**
- Vector retrieval testing (Redis not in retrieval path for vectors)
- Schema verification (Qdrant only)
- Graph queries (Neo4j only)
- General debugging of retrieval quality (focus on Qdrant/embeddings instead)

---

### BGE-M3 Embedding Service

Provides dense (1024-D), sparse (BM25-style), and ColBERT multi-vector embeddings. This is the core embedding service used for all vector operations in the system.

**Connection Details:**
```
Host machine: http://127.0.0.1:9000
Inside Docker: http://host.docker.internal:9000
Model: BAAI/bge-m3
```

**Endpoints:**
- `GET /healthz` - Health check, returns service status
- `POST /v1/embeddings` - Dense embeddings (requires "model" param in request body)
- `POST /v1/embeddings/sparse` - Sparse embeddings for lexical matching
- `POST /v1/embeddings/colbert` - ColBERT multi-vectors for late interaction

**Testing Embeddings:**
```bash
# Health check
curl -s http://127.0.0.1:9000/healthz

# Generate dense embeddings (MUST include model parameter)
curl -s -X POST "http://127.0.0.1:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": ["test query"]}' | python3 -m json.tool

# Compare embedding similarity between query and document
curl -s -X POST "http://127.0.0.1:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": ["query text", "document text"]}' | python3 -c "
import sys, json, numpy as np
data = json.load(sys.stdin)
embeddings = [e['embedding'] for e in data['data']]
q, d = np.array(embeddings[0]), np.array(embeddings[1])
sim = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
print(f'Cosine Similarity: {sim:.4f}')"

# Test with entity-enriched document text (as stored in Qdrant)
curl -s -X POST "http://127.0.0.1:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-m3",
    "input": [
      "NFS mount error debug",
      "Mount the filesystem\n\n[Context: cli_command: mount; network_or_storage_protocol: NFS]"
    ]
  }' | python3 -c "
import sys, json, numpy as np
data = json.load(sys.stdin)
embeddings = [e['embedding'] for e in data['data']]
q, d = np.array(embeddings[0]), np.array(embeddings[1])
sim = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
print(f'Query-to-Enriched-Doc Similarity: {sim:.4f}')"
```

---

### BGE Reranker Service

Cross-encoder reranker using BAAI/bge-reranker-v2-m3. This provides the final authoritative scoring for retrieved candidates.

**Connection Details:**
```
Host machine: http://127.0.0.1:9001
Health endpoint: GET /healthz
Rerank endpoint: POST /v1/rerank
```

**CRITICAL: Understanding Reranker Scores**
The BGE-reranker-v2-m3 outputs **raw logits**, NOT probabilities. This is essential to understand:
- **Negative scores** = Low relevance (but still ranked relatively)
- **Positive scores** = High relevance
- **Scores are RELATIVE** - higher is better even if both are negative
- A score of -1.5 is "more relevant" than -6.0
- Positive scores indicate high confidence of relevance

**The reranker uses ORIGINAL text, NOT enriched `_embedding_text`** - this is intentional to prevent overfitting to entity matches and maintain independent scoring.

**Testing Reranker:**
```bash
# Health check
curl -s http://127.0.0.1:9001/healthz

# Test reranking with sample documents
curl -s -X POST "http://127.0.0.1:9001/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to configure NFS",
    "documents": ["Configure NFS settings for your WEKA cluster.", "SMB share configuration guide."],
    "top_k": 2
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('=== Reranker Results ===')
for r in data['results']:
    verdict = 'RELEVANT' if r['score'] > 0 else 'LOW RELEVANCE'
    print(f\"{r['score']:+.4f} ({verdict}): {r['document'][:50]}...\")"

# Test with query that has no matching content
curl -s -X POST "http://127.0.0.1:9001/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "NFS mount error debug troubleshooting",
    "documents": ["Show NFS global configuration. View current settings.", "Configure NFS authentication methods."],
    "top_k": 2
  }' | python3 -m json.tool
```

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
# Start GLiNER service (auto-detects MPS)
cd services/gliner-ner && ./run.sh

# Start with specific options
cd services/gliner-ner && ./run.sh --device cpu  # Force CPU mode
cd services/gliner-ner && ./run.sh --port 9003   # Custom port

# Stop the service
pkill -f "server.py"
```

**Health Check:**
```bash
curl -s http://127.0.0.1:9002/healthz
# Returns: {"status": "ok", "device": "mps", "model": "urchade/gliner_medium-v2.1", "mps_available": true, "cuda_available": false}
```

**Test Entity Extraction:**
```bash
curl -s -X POST "http://127.0.0.1:9002/v1/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Mount the NFS filesystem using kubectl on RHEL."],
    "labels": ["cli_command", "operating_system", "network_or_storage_protocol"]
  }' | python3 -m json.tool
```

**32 Weka-Specific Entity Types (defined in `src/wekadocs_matrix/constants/ner_labels.py`):**
- `weka_software_component` - WEKA product components
- `cli_command` - Command-line commands and utilities
- `network_or_storage_protocol` - NFS, SMB, S3, etc.
- `operating_system` - Linux distributions, Windows, etc.
- `file_system_path` - File and directory paths
- `configuration_parameter` - Config keys and settings
- `error_message_or_code` - Error strings and codes
- `hardware_component` - Disks, NICs, memory, etc.
- Plus 24 more specialized types for WEKA domain knowledge

---

## Docker Container Management

### Container Overview

| Container | Service | Purpose | Key Ports | Volume Mounts |
|-----------|---------|---------|-----------|---------------|
| `weka-neo4j` | neo4j | Graph database | 7687, 7474 | Data volume |
| `weka-qdrant` | qdrant | Vector store | 6333, 6334 | Data volume |
| `weka-redis` | redis | Cache + queue | 6379 | Data volume |
| `weka-mcp-server` | mcp-server | HTTP MCP + STDIO | 8000 | `./src:/app/src:ro`, `./config:/app/config:ro` |
| `weka-ingestion-worker` | ingestion-worker | RQ background jobs | None | `./src:/app/src:ro`, `./config:/app/config:ro` |
| `weka-ingestion-service` | ingestion-service | Ingestion API | - | `./src:/app/src:ro` |

### Volume Mounts (CRITICAL for Development)

The MCP server and ingestion containers mount source code as read-only volumes:
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

**This means:**
- **Code changes are immediately visible** inside containers
- **NO rebuild needed for code changes** - just restart the container
- **Rebuild only needed** for requirements.txt or Dockerfile changes
- **Config changes** also take effect after restart (no rebuild)

### When to Rebuild vs Restart

| Change Type | Action Required | Command |
|-------------|-----------------|---------|
| Code in `src/` | Restart only | `docker compose restart mcp-server` |
| Config in `config/` | Restart only | `docker compose restart mcp-server` |
| requirements.txt | Full rebuild | `docker compose build mcp-server && docker compose up -d mcp-server` |
| Dockerfile changes | Full rebuild | `docker compose build mcp-server && docker compose up -d mcp-server` |

### Common Commands

```bash
# List running containers with status
docker ps --format "table {{.Names}}\t{{.Status}}" | grep weka

# List docker compose services (use service names for docker compose commands)
docker compose ps --format "table {{.Service}}\t{{.Name}}"

# Restart MCP server (use SERVICE name, NOT container name)
docker compose restart mcp-server

# Restart ingestion worker
docker compose restart ingestion-worker

# View logs (live follow)
docker logs -f weka-mcp-server
docker logs -f weka-ingestion-worker

# View recent logs
docker logs weka-mcp-server --tail 50
docker logs weka-ingestion-worker --tail 100

# Check MCP server health
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# List available MCP tools
curl -s http://127.0.0.1:8000/mcp/tools/list | python3 -m json.tool
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
4. GLiNER extracts entities if `ner.enabled: true` in config
5. Embeddings generated (8 vectors per chunk using `_embedding_text`)
6. Neo4j nodes created with relationships
7. Qdrant vectors stored with payloads
8. Redis tracks processed file hash to prevent re-processing

### CRITICAL: Clean Ingestion Procedure

The worker tracks processed files by content hash in Redis. Without cleanup, files won't re-ingest even if you delete them from Neo4j/Qdrant.

**Full Clean Re-ingestion (ALL THREE STORES):**
```bash
# 1. Clear Neo4j (all nodes and relationships)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 2. Clear Qdrant points (PRESERVE SCHEMA - keeps indexes and collection config)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 3. CRITICAL: Flush Redis (clears job tracking hashes)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Restart worker to pick up fresh state
docker compose restart ingestion-worker

# 5. Add/modify files in data/ingest/ - worker will auto-detect
```

**Verification Commands:**
```bash
# Verify Neo4j is empty
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN count(n)"

# Verify Qdrant is empty
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}\")"

# Verify Redis is empty
docker exec weka-redis redis-cli -a testredis123 DBSIZE
```

---

## Bug Found and Fixed This Session

### entity_metadata: null Bug

**Symptom:** All retrieval results showed `entity_metadata: null` despite GLiNER data being present and visible in Qdrant payloads.

**Investigation Process:**
1. Verified entity_metadata EXISTS in Qdrant payloads (confirmed with scroll API)
2. Traced retrieval code path in `hybrid_retrieval.py`
3. Found `payload_keys` allowlist filtering mechanism at lines 753-778
4. Discovered `entity_metadata` was NOT in the default allowlist

**Root Cause:** The `QdrantMultiVectorRetriever` class (line 753-778) uses a `payload_keys` allowlist to filter which payload fields are returned from Qdrant queries. This is a performance optimization to avoid transferring unnecessary data. However, when GLiNER integration added `entity_metadata` to the payload, it was not added to this allowlist.

**Location:** `src/query/hybrid_retrieval.py` lines 753-778

**Fix Applied:**
```python
# File: src/query/hybrid_retrieval.py
# Line 778-779 (added to payload_keys list)
# GLiNER entity metadata for Phase 4 entity-aware retrieval
"entity_metadata",
```

**Verification Steps:**
1. Confirmed entity_metadata EXISTS in Qdrant: `curl scroll` showed full payload
2. Confirmed fix in source file: `grep entity_metadata src/query/hybrid_retrieval.py`
3. Restarted MCP server: `docker compose restart mcp-server`
4. Awaiting re-test to confirm fix propagates to retrieval results

**Impact:** Without this fix, the `_apply_entity_boost` method (lines 4813-4863) could never boost results because `entity_metadata` was always null, even though the data was properly stored in Qdrant.

---

## Architecture Findings

### GLiNER Entity Flow (4 Retrieval Channels)

GLiNER entities influence retrieval through four distinct channels, each operating at different stages of the pipeline:

| Channel | Stage | Mechanism | Code Location | Weight/Impact |
|---------|-------|-----------|---------------|---------------|
| Dense Vectors | Embedding | `_embedding_text` includes `[Context: entity_type: name]` suffix | `ner_gliner.py:109-114` | Semantic similarity |
| ColBERT Multi-Vectors | Embedding | Same `_embedding_text` used for token-level embeddings | `atomic.py:1383` | Token-level matching |
| Entity-Sparse | Retrieval | Entity names → BM25-style sparse embedding | `atomic.py:1489-1534` | **1.5x RRF weight** |
| Post-Retrieval Boost | Reranking | `entity_metadata.entity_values_normalized` matched against query entities | `hybrid_retrieval.py:4813-4863` | Up to 50% score boost |

### The _embedding_text Pattern

GLiNER enrichment creates a transient `_embedding_text` field that is used ONLY for embedding generation, NOT stored in Qdrant:

```python
# From ner_gliner.py:109-114
if title:
    chunk["_embedding_text"] = (
        f"{title}\n\n{base_text}\n\n[Context: {entity_context}]"
    )
else:
    chunk["_embedding_text"] = f"{base_text}\n\n[Context: {entity_context}]"

# entity_context example: "weka_software_component: WEKA; cli_command: mount; network_or_storage_protocol: NFS"
```

The original `text` field remains UNCHANGED and is what gets stored in Qdrant payload and shown to users/reranker.

### Key Design Decision: Reranker Independence

The BGE cross-encoder reranker intentionally uses **original text**, NOT `_embedding_text`. This is a deliberate architectural choice to maintain the reranker as an independent arbiter that doesn't over-fit to entity matches.

```python
# In _apply_reranker (line 3803):
text_body = (chunk.text or "").strip()  # Uses original text, NOT _embedding_text
heading = (chunk.heading or "").strip()
if heading and text_body:
    text = f"{heading}\n\n{text_body}"
```

### neo4j_disabled Implementation

The `neo4j_disabled` flag has **13+ checkpoints** throughout `hybrid_retrieval.py` that comprehensively disable all graph operations:

| Line | Method | Graph Operation Disabled |
|------|--------|--------------------------|
| 2411 | `_compute_graph_signals` | Entity-based graph scoring |
| 2483 | `_compute_cross_doc_signals` | Cross-document relationships |
| 2568 | `_apply_graph_reranker` | Graph-based reranking |
| 2955 | `retrieve` (main) | Graph channel retrieval |
| 3612 | `_hydrate_missing_citations` | Citation label hydration |
| 4064 | `_bounded_expansion` | Neighbor expansion |
| 4237 | `_expand_with_structure` | Structure-based expansion |
| 4772 | `_apply_graph_enrichment` | Graph enrichment pass |
| 4877 | `_graph_retrieval_channel` | Graph retrieval channel |
| 5017 | `_fetch_graph_neighbors` | Neighbor fetching |
| 5148 | `_annotate_coverage` | Coverage annotation |
| 5324 | `_expand_microdoc_results` | Microdoc expansion |

All Neo4j `driver.session()` usages are properly guarded by these checks.

---

## Current Configuration State

### Key Config Settings (config/development.yaml)

```yaml
# Graph bypass for vector-only retrieval
neo4j_disabled: true  # ALL graph operations bypassed in retrieval

# BM25 (Neo4j full-text search) disabled
bm25:
  enabled: false  # Neo4j BM25 disabled - Phase 1 vector-only hardening

# Vector field weights for initial retrieval
vector_fields:
  content: 0.5   # Dense semantic content
  title: 0.1     # Dense semantic heading
  text-sparse: 0.3  # Lexical content matching

# RRF fusion weights (applied during multi-signal fusion)
rrf_field_weights:
  content: 1.0           # Dense semantic content similarity
  title: 1.0             # Dense semantic title similarity
  text-sparse: 1.0       # Lexical content matching (BM25-style)
  doc_title-sparse: 1.0  # Lexical document title matching
  title-sparse: 2.0      # BOOSTED - Lexical section heading matching
  entity-sparse: 1.5     # BOOSTED - Lexical entity name matching

# Reranker configuration
reranker:
  enabled: true
  provider: "bge-reranker-service"
  model: "BAAI/bge-reranker-v2-m3"
  top_n: 20  # Final results after reranking

# GLiNER NER configuration
ner:
  enabled: true
  service_url: "http://host.docker.internal:9002"
```

---

## What's Working vs Issues

### Working Correctly
- GLiNER entity extraction during ingestion (65x faster with MPS)
- Entity-enriched embeddings (`_embedding_text` with context suffix)
- Entity-sparse vector generation and storage
- 8 vectors per chunk stored in Qdrant
- `neo4j_disabled` properly bypassing ALL graph operations (verified: all graph scores = 0)
- BM25 properly disabled (verified: bm25_count = 0, bm25_time_ms = 0)
- Reranker giving correct RELATIVE scores (negative for low relevance, positive for high)
- MCP server health and tools accessible
- Qdrant collection schema with entity metadata indexes

### Fixed This Session
- `entity_metadata` now included in `payload_keys` allowlist (was being filtered out during retrieval)

### Remaining Considerations
- Negative rerank scores are EXPECTED for queries with no matching corpus content
- Current corpus lacks NFS troubleshooting/error documentation (47 NFS chunks, all about configuration)
- Entity boosting now has access to metadata (should improve results for entity-matching queries)
- Need to re-test retrieval to verify entity_metadata fix propagates correctly

---

## Key Files and Their Purposes

| File | Lines | Purpose |
|------|-------|---------|
| `src/query/hybrid_retrieval.py` | ~5660 | Main retrieval pipeline: HybridRetriever, QdrantMultiVectorRetriever, BM25Retriever, ChunkResult |
| `src/ingestion/extract/ner_gliner.py` | ~162 | GLiNER entity extraction and chunk enrichment |
| `src/wekadocs_matrix/constants/ner_labels.py` | - | 32 Weka-specific entity type definitions |
| `services/gliner-ner/` | - | Native MPS-accelerated GLiNER FastAPI service |
| `config/development.yaml` | - | Development configuration (neo4j_disabled, weights, etc.) |
| `src/ingestion/pipeline/atomic.py` | - | Atomic ingestion with embedding generation |

---

## MCP Server Interaction

### Available Tools
```bash
curl -s http://127.0.0.1:8000/mcp/tools/list | python3 -c "
import sys, json
for t in json.load(sys.stdin)['tools']:
    print(f\"- {t['name']}\")"
```

Current tools:
- `search_documentation` - Main search endpoint
- `traverse_relationships` - Graph traversal (disabled when neo4j_disabled=true)

### Calling MCP Tools
```bash
curl -s -X POST "http://127.0.0.1:8000/mcp/tools/call" \
  -H "Content-Type: application/json" \
  -d '{"name": "search_documentation", "arguments": {"query": "NFS configuration", "top_k": 5}}'
```

Note: Returns markdown format, not JSON. The response contains structured sections with evidence chunks.

---

## Next Steps

1. **Verify entity_metadata fix** - Re-test retrieval to confirm entity_metadata is now populated in results
2. **Test entity boosting** - Query with entity terms (e.g., "WEKA cluster configuration") and verify `entity_boost_applied: true`
3. **Evaluate corpus coverage** - Consider adding troubleshooting documentation if needed for better query coverage
4. **Monitor rerank scores** - Positive scores expected for queries with matching corpus content
5. **Consider committing fix** - The entity_metadata payload_keys fix should be committed

---

## Session Artifacts

- **Bug fix applied:** `entity_metadata` added to `payload_keys` in `QdrantMultiVectorRetriever`
- **No architectural changes:** System verified working as designed
- **Documentation:** This comprehensive session context file
- **Pending:** Commit of entity_metadata fix, re-testing of retrieval

---

*Session context saved: 2025-12-08 21:45 EST*

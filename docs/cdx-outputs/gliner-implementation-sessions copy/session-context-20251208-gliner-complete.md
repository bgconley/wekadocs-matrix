# Session Context: GLiNER Integration Complete

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Last Commit:** `d0c0e75` - "feat: complete GLiNER NER integration (Phases 1-5)"
**Session Focus:** GLiNER code review completion, STDIO fix, OTEL error resolution
**Document Version:** 8.0

---

## Executive Summary

This session completed the GLiNER NER (Named Entity Recognition) integration for the WekaDocs GraphRAG system. All 5 phases are implemented, code reviewed, and pushed to remote. The implementation adds entity-aware retrieval boosting using zero-shot NER with GLiNER.

**Key Accomplishments:**
1. Fixed OTEL/Jaeger connection errors in ingestion-worker
2. Fixed STDIO server to expose `entity_metadata` and `entity_boost_applied` to Agent
3. Completed 100% code review - no issues found
4. Committed and pushed with comprehensive commit message

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, Entities, and their relationships (HAS_SECTION, HAS_CHUNK, MENTIONS, NEXT_CHUNK, etc.).

**Connection Details:**
```
URI from host machine: bolt://localhost:7687
URI from inside Docker: bolt://neo4j:7687
Neo4j Browser UI: http://localhost:7474
Username: neo4j
Password: testpassword123
Database: neo4j (default)
Container name: weka-neo4j
```

**Access Patterns:**

```bash
# Direct cypher-shell access (CRITICAL: remove -it flag for non-TTY contexts like scripts)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count chunks
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"

# Example: Find chunks with entity mentions
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) WHERE toLower(c.heading) CONTAINS 'nfs' RETURN c.heading LIMIT 10"

# Clear ALL data (DESTRUCTIVE - for clean re-ingestion)
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
- Verifying graph relationships (MENTIONS, HAS_SECTION, NEXT_CHUNK, etc.)
- Debugging entity extraction and mention relationships
- Comparing data between Neo4j and Qdrant for sync verification

**Important Note:** With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval queries but **still used during ingestion**. GLiNER entities are NOT written to Neo4j (vector-only enrichment pattern).

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection. It is the **primary store for vector-based retrieval**.

**Connection Details:**
```
REST API: http://127.0.0.1:6333
gRPC endpoint: localhost:6334
Authentication: None (development mode)
Collection: chunks_multi_bge_m3
Container name: weka-qdrant
```

**Current Vector Schema (8 vectors per chunk):**

| Vector Name | Type | Dimensions | Purpose | RRF Weight |
|-------------|------|------------|---------|------------|
| `content` | Dense | 1024 | Main semantic content embedding | 1.0 |
| `title` | Dense | 1024 | Section heading semantic embedding | 1.0 |
| `doc_title` | Dense | 1024 | Document title semantic embedding | 1.0 |
| `late-interaction` | Dense (multi) | 1024 × N tokens | ColBERT MaxSim token-level matching | N/A |
| `text-sparse` | Sparse | Variable | BM25-style lexical content matching | 1.0 |
| `title-sparse` | Sparse | Variable | Section heading lexical matching | **2.0** |
| `doc_title-sparse` | Sparse | Variable | Document title lexical matching | 1.0 |
| `entity-sparse` | Sparse | Variable | Entity name lexical matching | **1.5** |

**Payload Indexes (28 total):**

The collection has 28 payload indexes including the 4 entity metadata indexes added in Phase 3:
- `entity_metadata.entity_types` (KEYWORD) - Entity type labels for filtering
- `entity_metadata.entity_values` (KEYWORD) - Raw entity text values
- `entity_metadata.entity_values_normalized` (KEYWORD) - Lowercased for case-insensitive matching
- `entity_metadata.entity_count` (INTEGER) - Count per chunk for filtering

**Access Patterns:**

```bash
# Check collection status and point count
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}\")"

# Delete all points (PRESERVE SCHEMA) - use for clean re-ingestion
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# Scroll through points with payloads (verify entity_metadata)
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("chunks_multi_bge_m3", limit=5, with_payload=True, with_vectors=False)
for p in points:
    em = p.payload.get("entity_metadata", {})
    print(f"{p.payload.get('heading', '')[:40]}: {em.get('entity_count', 0)} entities")
EOF
```

**Payload Structure for Entity-Aware Retrieval:**
Each Qdrant point payload includes:
- `entity_metadata.entity_types` - List of entity type labels
- `entity_metadata.entity_values` - List of raw entity text
- `entity_metadata.entity_values_normalized` - Lowercased for matching
- `entity_metadata.entity_count` - Integer count

---

### Redis Cache and Queue

Redis serves two purposes: L2 query result caching and RQ job queue for async ingestion.

**Connection Details:**
```
Host: localhost (from host) / redis (from Docker)
Port: 6379
Password: testredis123
Database: 0
Container name: weka-redis
```

**Access Patterns:**
```bash
# Check database size
docker exec weka-redis redis-cli -a testredis123 DBSIZE

# Flush all data (REQUIRED for clean re-ingestion - clears job tracking)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# Check RQ job queue
docker exec weka-redis redis-cli -a testredis123 KEYS "rq:*"
```

**CRITICAL: When to Access Redis:**
- **ALWAYS flush Redis** before re-ingesting the same documents (worker tracks processed files by content hash)
- Debugging stuck ingestion jobs (RQ queue inspection)
- Cache invalidation during development

**When NOT to Access Redis:**
- Vector retrieval testing (Redis not in retrieval path)
- Schema verification (Qdrant only)
- Graph queries (Neo4j only)
- Most debugging scenarios - Redis is auxiliary

---

### BGE-M3 Embedding Service

Provides dense (1024-D), sparse (BM25-style), and ColBERT multi-vector embeddings. **Critical dependency for all ingestion operations.**

**CRITICAL: URL varies by access context!**

| Context | Base URL |
|---------|----------|
| Host machine (scripts, tests) | `http://127.0.0.1:9000` |
| Inside Docker container | `http://host.docker.internal:9000` |

**Endpoints:**
- `GET /healthz` - Health check, returns `{"status": "ok"}`
- `POST /v1/embeddings` - Dense embeddings (1024-D vectors)
- `POST /v1/embeddings/sparse` - Sparse embeddings (BM25-style term weights)
- `POST /v1/embeddings/colbert` - ColBERT multi-vectors (token-level embeddings)

**Health Check:**
```bash
curl -s http://127.0.0.1:9000/healthz
# Returns: {"status": "ok"}
```

**Example Usage:**
```python
import requests
response = requests.post(
    "http://127.0.0.1:9000/v1/embeddings",
    json={"input": ["text to embed"]}
)
embeddings = [e["embedding"] for e in response.json()["data"]]  # 1024-D vectors
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

**Usage:** Called after initial retrieval to re-score candidates with cross-encoder attention. Sets the authoritative `rerank_score` field in results. The reranker uses original clean text, NOT GLiNER-enriched text.

---

### GLiNER NER Service (Native MPS-Accelerated)

Native macOS service with MPS (Metal Performance Shaders) acceleration for Apple Silicon. Provides **65x faster** entity extraction than CPU-based Docker inference.

**Connection Details:**
```
Host machine: http://127.0.0.1:9002
Inside Docker: http://host.docker.internal:9002
Container name: N/A (runs natively on macOS, NOT in Docker)
```

**Endpoints:**
- `GET /healthz` - Health check with device info (mps/cuda/cpu)
- `POST /v1/extract` - Batch entity extraction
- `GET /v1/config` - Current model configuration

**Starting/Stopping the Service:**
```bash
cd services/gliner-ner && ./run.sh    # Start (auto-detects MPS)
pkill -f "server.py"                   # Stop
```

**Health Check:**
```bash
curl -s http://127.0.0.1:9002/healthz
# Returns: {"status": "ok", "device": "mps", "model": "urchade/gliner_medium-v2.1", ...}
```

**Performance Metrics:**
- Model load (cached): ~6s
- Entity extraction (32 chunks): ~3s on MPS vs ~194s on CPU = **65x faster**
- Entity coverage: 100% of chunks (vs 41% on CPU)

---

## Docker Container Architecture and Management

### Container Overview

| Container | Purpose | Key Ports | Volume Mounts | Rebuild Trigger |
|-----------|---------|-----------|---------------|-----------------|
| `weka-neo4j` | Graph database | 7687, 7474 | Data volume | N/A |
| `weka-qdrant` | Vector store | 6333, 6334 | Data volume | N/A |
| `weka-redis` | Cache + queue | 6379 | Data volume | N/A |
| `weka-mcp-server` | HTTP MCP + STDIO | 8000 | `./src:/app/src:ro` | requirements.txt |
| `weka-ingestion-worker` | RQ background jobs | None | `./src:/app/src:ro` | requirements.txt |
| `weka-alloy` | OTEL collector | 4318 | Config volume | N/A |

### Volume Mounts (CRITICAL for Development)

The MCP server and ingestion worker mount source code as read-only volumes:
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

**This means:**
- **Code changes are immediately visible** inside containers
- **NO rebuild needed for code changes** - just restart the container
- Restart: `docker compose restart mcp-server` or `docker compose restart ingestion-worker`
- **Rebuild only needed** for Dockerfile or requirements.txt changes

### When to Rebuild vs Restart

| Change Type | Action Required |
|-------------|-----------------|
| Code in `src/` | `docker compose restart <service>` |
| Config in `config/` | `docker compose restart <service>` |
| New package in requirements.txt | `docker compose build <service> && docker compose up -d <service>` |
| Dockerfile changes | `docker compose build <service> && docker compose up -d <service>` |
| docker-compose.yml env vars | `docker compose up -d <service>` (recreates container) |

### Environment Variable Precedence Issue (DISCOVERED THIS SESSION)

**Problem Found:** Docker Compose reads shell environment **BEFORE** `.env` file. If your host shell has `OTEL_EXPORTER_OTLP_ENDPOINT` set, it overrides `.env`.

**Fix:** When recreating containers, explicitly set the env var:
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4318 docker compose up -d --force-recreate ingestion-worker
```

### HuggingFace Model Cache

**Important Fix Applied in Phase 1:** Docker containers use `HF_HOME=/opt/hf-cache` (NOT `HF_CACHE`).

```
Host: ./hf-cache/hub/models--urchade--gliner_medium-v2.1 (~1.5GB)
      ↓ volume mount
Container: /opt/hf-cache/hub/models--urchade--gliner_medium-v2.1
```

### Essential Docker Commands

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Restart after code changes (NO rebuild needed)
docker compose restart ingestion-worker
docker compose restart mcp-server

# Rebuild for requirements.txt changes
docker compose build ingestion-worker && docker compose up -d ingestion-worker

# View logs (filter out Jaeger noise which is cosmetic)
docker logs -f weka-ingestion-worker 2>&1 | grep -v jaeger

# Shell into container for debugging
docker exec -it weka-mcp-server bash

# Verify config loaded inside container
docker exec weka-mcp-server python3 -c "
from src.shared.config import get_config
config = get_config()
print('ner.enabled:', config.ner.enabled)
print('ner.service_url:', config.ner.service_url)
"

# Check container env vars
docker exec weka-ingestion-worker printenv | grep -i otel
```

---

## Data Ingestion Workflow

### File-Drop Pattern

The ingestion worker monitors `/app/data/ingest/` for new markdown files.

**Host path:** `./data/ingest/`
**Container path:** `/app/data/ingest/`

**Process:**
1. Drop markdown files in `data/ingest/`
2. Worker auto-detects and processes within seconds
3. Creates: Neo4j nodes (Document, Section/Chunk) + Qdrant vectors (8 per chunk)
4. GLiNER enrichment runs if `ner.enabled: true` in config
5. If `ner.service_url` configured, uses HTTP mode (MPS-accelerated native service)

### CRITICAL: Clean Ingestion Procedure

**The worker tracks processed files by content hash in Redis.** Without cleanup, files won't re-ingest even after code changes or container restart.

**Complete Clean Ingestion Procedure:**
```bash
# 1. Delete Qdrant points (PRESERVE schema/indexes)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 2. Clear Neo4j (all nodes and relationships)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 3. Clear Redis (CRITICAL: clears RQ job tracking so files can re-ingest)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Remove processed files from ingest folder
rm -f data/ingest/*.md

# 5. Re-add test files
cp docs/sample_ingest/*.md data/ingest/
```

**Why Each Step Matters:**
- **Qdrant delete**: Removes old vectors but preserves schema and indexes
- **Neo4j clear**: Removes old graph nodes so new ingestion doesn't create duplicates
- **Redis flush**: Clears RQ job tracking - **without this, worker thinks files are already processed**
- **File removal**: Allows fresh file drop to trigger ingestion

---

## GLiNER Implementation Status

### All Phases Complete ✅

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1.0 | Core Infrastructure - GLiNERService singleton, NERConfig, labels | ✅ Complete |
| Phase 2.0 | Ingestion Pipeline - enrich_chunks_with_entities(), entity_metadata | ✅ Complete |
| Phase 2.5 | MPS Acceleration - Native macOS service at :9002 | ✅ Complete |
| Phase 3.0 | Qdrant Schema - 4 entity metadata payload indexes | ✅ Complete |
| Phase 4.0 | Hybrid Search - QueryDisambiguator, post-retrieval boosting | ✅ Complete |
| Phase 4.4 | STDIO Entity Exposure - entity_metadata + entity_boost_applied | ✅ Complete |
| Phase 5.0 | Performance - 600s ingestion timeout | ✅ Complete |

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/providers/ner/__init__.py` | 12 | Package exports |
| `src/providers/ner/labels.py` | 79 | 11 domain-specific entity labels |
| `src/providers/ner/gliner_service.py` | 608 | Singleton with HTTP + local fallback |
| `src/ingestion/extract/ner_gliner.py` | 163 | Chunk enrichment function |
| `src/query/processing/__init__.py` | 5 | Package exports |
| `src/query/processing/disambiguation.py` | 178 | QueryDisambiguator class |
| `services/gliner-ner/server.py` | 367 | FastAPI MPS service |
| `services/gliner-ner/run.sh` | 64 | Startup script |
| `services/gliner-ner/requirements.txt` | 16 | Dependencies |

### Files Modified

| File | Changes |
|------|---------|
| `src/shared/config.py` | Added NERConfig class with service_url |
| `config/development.yaml` | Added ner: block with 11 labels |
| `docker-compose.yml` | Fixed HF_CACHE → HF_HOME |
| `src/ingestion/atomic.py` | GLiNER enrichment hooks, entity_metadata in payload |
| `src/query/hybrid_retrieval.py` | entity_metadata field, _apply_entity_boost() |
| `src/shared/qdrant_schema.py` | 4 entity metadata indexes |
| `src/mcp_server/stdio_server.py` | Expose entity_metadata + entity_boost_applied |

---

## Architectural Decisions

### 1. GLiNER is Vector-Only Enrichment

GLiNER entities are **NOT written to Neo4j**. They only enrich:
- Dense embeddings (via transient `_embedding_text` field)
- Entity-sparse vectors (via `_mentions` with `source="gliner"`)
- Qdrant payload (`entity_metadata`)

**Benefit:** With `neo4j_disabled: true`, you get 100% of GLiNER benefits through pure vector retrieval.

### 2. HTTP + Local Fallback Pattern

The `GLiNERService` implements a two-tier architecture:
1. **Primary (HTTP):** Calls native MPS-accelerated service at `service_url`
2. **Fallback (Local):** Loads model in-process if HTTP unavailable

### 3. Transient `_embedding_text` Field

Entity context enriches embeddings without polluting stored text:
```python
chunk["_embedding_text"] = f"{title}\n\n{text}\n\n[Context: {entity_context}]"
# chunk["text"] remains untouched - this is what gets stored
```

### 4. Post-Retrieval Boosting (NOT Qdrant Filters)

Qdrant's `should` filters don't boost scores - they only filter. Entity boosting is done in Python after retrieval:
- Over-fetch 2x candidates when query entities detected
- Boost matching chunks up to 50% (configurable)
- Set `entity_boost_applied=True` flag on boosted results

### 5. Cross-Encoder Not Affected by GLiNER

The BGE cross-encoder reranker uses original clean text, NOT `_embedding_text`. This prevents "over-fitting" to entity matches.

---

## Domain-Specific Entity Labels (11 Types)

```yaml
labels:
  - "weka_software_component (e.g. backend, frontend, agent, client)"
  - "operating_system (e.g. RHEL, Ubuntu, Rocky Linux)"
  - "hardware_component (e.g. NVMe, NIC, GPU, switch)"
  - "filesystem_object (e.g. inode, snapshot, file, directory)"
  - "cloud_provider_or_service (e.g. AWS, S3, Azure, EC2)"
  - "cli_command (e.g. weka fs, mount, systemctl)"
  - "configuration_parameter (e.g. --net-apply, stripe-width)"
  - "network_or_storage_protocol (e.g. NFS, SMB, S3, POSIX, TCP)"
  - "error_message_or_code (e.g. 10054, Connection refused)"
  - "performance_metric (e.g. IOPS, latency, throughput)"
  - "file_system_path (e.g. /mnt/weka, /etc/fstab)"
```

---

## Issues Resolved This Session

### 1. OTEL/Jaeger Connection Errors

**Problem:** Ingestion worker logs showed repeated connection errors to `jaeger:4318`

**Root Cause:** Host shell had `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318` set, overriding `.env` file's `http://alloy:4318`

**Fix:** Recreated container with explicit env override:
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4318 docker compose up -d --force-recreate ingestion-worker
```

### 2. STDIO Server Missing Entity Fields

**Problem:** Agent couldn't see entity boosting information in search results

**Fix:** Added to `src/mcp_server/stdio_server.py` line 540-541:
```python
"entity_boost_applied": getattr(chunk, "entity_boost_applied", False),
"entity_metadata": getattr(chunk, "entity_metadata", None),
```

### 3. Commit Message Oversimplification

**Problem:** Git amend replaced comprehensive GLiNER commit message with STDIO-only message

**Fix:** Re-amended with comprehensive message listing all phases

---

## What Works vs What's Disabled

### Working Features
1. **GLiNER entity extraction** - Fully functional with MPS acceleration
2. **Entity-enriched embeddings** - Dense vectors include entity context
3. **Entity-sparse vectors** - Lexical matching on entity names
4. **Post-retrieval entity boosting** - Query-time entity matching
5. **STDIO entity exposure** - Agent can see entity_metadata and boost flags
6. **OTEL tracing** - Connected to Alloy collector

### Intentionally Disabled
1. **neo4j_disabled: true** - Graph queries bypassed during retrieval (ingestion still uses Neo4j)
2. **Graph channel** - Disabled when neo4j_disabled is true
3. **Microdoc stubs** - Disabled in previous session
4. **doc_fallback** - Disabled to prevent cross-topic chunk pollution

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

# Embedding services (HOST MACHINE URLs - different inside Docker!)
export BGE_M3_API_URL=http://127.0.0.1:9000
export RERANKER_BASE_URL=http://127.0.0.1:9001

# GLiNER service (HOST MACHINE URL)
export GLINER_SERVICE_URL=http://127.0.0.1:9002

# HuggingFace cache
export HF_HOME=./hf-cache

# API keys
export JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi
```

---

## Quick Reference Commands

### Start Full Stack
```bash
docker compose up -d
cd services/gliner-ner && ./run.sh  # Separate terminal for MPS acceleration
```

### Verify All Services Health
```bash
curl -s http://127.0.0.1:9000/healthz  # BGE Embedder
curl -s http://127.0.0.1:9001/healthz  # Reranker
curl -s http://127.0.0.1:9002/healthz  # GLiNER
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "RETURN 1"
curl -s http://127.0.0.1:6333/collections/chunks_multi_bge_m3 | head -1
```

### Monitor Ingestion
```bash
docker logs -f weka-ingestion-worker 2>&1 | grep -E "(gliner|enrichment|entity)"
```

---

## Next Steps

### Potential Future Work
1. **Full corpus re-ingestion** with GLiNER enabled
2. **A/B testing** retrieval quality with/without entity enrichment
3. **Tuning** boost parameters based on evaluation metrics
4. **GPU deployment** for production scale

---

## Related Documentation

- **Canonical Plan:** `/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md` (v1.7)
- **Schema Reference:** `/scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json`
- **Configuration:** `/config/development.yaml`

---

## Glossary

| Term | Definition |
|------|------------|
| **GLiNER** | Generalist Lightweight Named Entity Recognition - zero-shot NER model |
| **MPS** | Metal Performance Shaders - Apple Silicon GPU acceleration |
| **`_embedding_text`** | Transient field with entity context for embedding generation (NOT stored) |
| **`entity_metadata`** | Qdrant payload field with entity_types, entity_values, entity_count |
| **`entity_boost_applied`** | ChunkResult flag indicating chunk received entity-based score boost |
| **Post-retrieval boosting** | Score multiplication after fusion based on entity matches (max 50%) |
| **entity-sparse** | Sparse vector from entity names for lexical matching (RRF weight 1.5x) |
| **RRF** | Reciprocal Rank Fusion - method for combining multiple retrieval signals |
| **ColBERT** | Late-interaction retrieval using token-level MaxSim scoring |

---

## Code Review Findings (100% Complete)

The comprehensive code review verified the following components:

### Phase 1: Core Infrastructure
- **GLiNERService singleton pattern** - Properly implemented with `_instance` class variable
- **HTTP + fallback architecture** - Correctly tries HTTP service first, falls back to local model
- **Circuit breaker** - `_model_load_failed` flag prevents repeated failed load attempts
- **LRU cache** - Query entity extraction cached for repeated short queries (<200 chars)

### Phase 2: Ingestion Pipeline
- **enrich_chunks_with_entities()** - Correctly batches chunks for extraction
- **`_embedding_text` transient field** - Set but NOT stored in Qdrant payload
- **`entity_metadata` persistence** - Correctly added to Qdrant payload at line 3153 of atomic.py
- **`source="gliner"` filter** - GLiNER entities filtered from Neo4j MENTIONS creation
- **Deduplication** - Prevents duplicate entities when structural and GLiNER extraction overlap

### Phase 3: Qdrant Schema
- **4 payload indexes** - All correctly defined with proper PayloadSchemaType
- **Canonical schema updated** - `chunks_multi_bge_m3_schema.json` includes new indexes
- **Validation parameter** - `require_entity_metadata_indexes` added for schema validation

### Phase 4: Hybrid Search
- **QueryDisambiguator** - Lazy initialization via `_get_disambiguator()`
- **Over-fetch logic** - 2x candidates when query entities detected (line 2802)
- **Boost calculation** - `boost_factor = 1.0 + min(0.5, matches * 0.1)` caps at 50%
- **Graceful handling of None** - `entity_metadata = res.entity_metadata or {}` at line 4845
- **Microdoc chunks** - Correctly get `entity_metadata` from Qdrant payload (line 5491)
- **Graph channel chunks** - Intentionally don't have entity_metadata (Neo4j doesn't store it)

### STDIO Entity Exposure
- **entity_boost_applied** - Added at line 540 of stdio_server.py
- **entity_metadata** - Added at line 541 of stdio_server.py

---

## GLiNER Entity Embedding Flow (Complete Documentation)

### During Ingestion (Document Time)

GLiNER entities are embedded in **4 different ways**:

1. **Dense Vectors (content)** - Uses `_embedding_text` which includes `[Context: entity_type: entity_value; ...]`
2. **ColBERT Multi-Vectors** - Same `_embedding_text` used, so token-level embeddings include entity context
3. **Entity-Sparse Vector** - Built from `_mentions` field containing entity names → sparse BM25-style embedding
4. **Entity Metadata Payload** - Stored in Qdrant for query-time boosting

### During Retrieval (Query Time)

1. **Step 0: Query Disambiguation** - GLiNER extracts entities from user query, producing `boost_terms`
2. **Step 1: 6-Signal Vector Search** - All vectors queried including entity-sparse (weight 1.5x)
3. **Step 2: RRF Fusion** - Reciprocal Rank Fusion combines all signals
4. **Step 3: Entity Boosting** - Post-retrieval soft filtering: chunks with matching entities get up to 50% score boost
5. **Step 4: ColBERT Rerank** - Optional token-level MaxSim (uses enriched embeddings indirectly)
6. **Step 5: BGE Cross-Encoder** - Final authoritative scoring (uses original text, NOT enriched)

### Key Design Decision

The cross-encoder reranker is intentionally **NOT affected** by GLiNER enrichment - it uses original clean text. This prevents "over-fitting" to entity matches and maintains cross-encoder's role as an independent final arbiter.

---

## Two MCP Server Architecture

There are TWO MCP servers - don't confuse them:

| Server | File | Protocol | Primary Tool | Usage |
|--------|------|----------|--------------|-------|
| HTTP Server | `src/mcp_server/main.py` | HTTP REST on :8000 | `search_documentation` | LEGACY - human-facing |
| STDIO Server | `src/mcp_server/stdio_server.py` | stdin/stdout pipes | `search_sections` | PRODUCTION - Agent/Claude |

**STDIO Server Invocation:**
```bash
docker exec -i weka-mcp-server python -m src.mcp_server.stdio_server
```

The STDIO server is what Claude/Agent uses. The `search_sections` tool returns raw chunk data with all fusion scores, entity metadata, and boost flags. The HTTP server's `search_documentation` synthesizes human-readable answers.

---

## Test File Locations

| Test File | Count | Type | Purpose |
|-----------|-------|------|---------|
| `tests/unit/test_gliner_service.py` | 21 | Unit | GLiNERService singleton, extraction |
| `tests/unit/test_ner_gliner_enrichment.py` | 15 | Unit | Chunk enrichment function |
| `tests/unit/test_query_disambiguation.py` | 15 | Unit | QueryDisambiguator |
| `tests/integration/test_gliner_config.py` | 13 | Integration | Config loading |
| `tests/integration/test_gliner_live.py` | 8 | Live | Real model tests |
| `tests/integration/test_gliner_ingestion_flow.py` | 9 | Integration | Ingestion pipeline |
| `tests/integration/test_phase4_entity_retrieval.py` | 13 | Integration | Entity boosting |
| `tests/shared/test_qdrant_schema.py` | 4 | Unit | Schema indexes |
| `tests/shared/test_qdrant_schema_validation.py` | 4 | Integration | Schema validation |

**Run GLiNER Tests:**
```bash
# All GLiNER tests
python -m pytest tests/unit/test_gliner_service.py tests/unit/test_ner_gliner_enrichment.py tests/unit/test_query_disambiguation.py tests/integration/test_gliner_config.py tests/integration/test_gliner_ingestion_flow.py tests/integration/test_phase4_entity_retrieval.py -v

# Live model tests (requires GLiNER service running)
python -m pytest tests/integration/test_gliner_live.py -v -m live
```

---

## Troubleshooting Guide

### GLiNER Service Won't Start

**Symptom:** `./run.sh` fails or service crashes immediately

**Common Causes:**
1. **Missing dependencies**: Run `pip install -r requirements.txt` in `services/gliner-ner/`
2. **Port conflict**: Check if 9002 is in use: `lsof -i :9002`
3. **PyTorch MPS issue**: Verify MPS: `python3 -c "import torch; print(torch.backends.mps.is_available())"`

**Solution:**
```bash
cd services/gliner-ner
pip install -r requirements.txt
./run.sh --port 9003  # Try different port if 9002 in use
```

### Docker Can't Reach Native GLiNER Service

**Symptom:** Logs show "HTTP service connection failed", falls back to CPU

**Common Causes:**
1. **Service not running**: Start with `cd services/gliner-ner && ./run.sh`
2. **Wrong URL in config**: Should be `http://host.docker.internal:9002` (not localhost)
3. **Firewall blocking**: macOS firewall may block Docker

**Verification:**
```bash
# From host - should work
curl http://127.0.0.1:9002/healthz

# From inside Docker - should also work
docker exec weka-ingestion-worker curl http://host.docker.internal:9002/healthz
```

### No Entities Being Extracted

**Symptom:** `entity_metadata.entity_count = 0` for all chunks

**Common Causes:**
1. **NER disabled**: Check `config.ner.enabled` in `config/development.yaml`
2. **Labels mismatch**: Labels must match document domain
3. **Threshold too high**: Try lowering from 0.45 to 0.35

**Verification:**
```bash
docker exec weka-mcp-server python3 -c "
from src.shared.config import get_config
config = get_config()
print('NER enabled:', config.ner.enabled)
print('Threshold:', config.ner.threshold)
print('Service URL:', config.ner.service_url)
"
```

### Documents Won't Re-Ingest

**Symptom:** Files in `data/ingest/` are ignored, no processing logs

**Root Cause:** RQ worker tracks processed files by content hash in Redis. Even after clearing Neo4j and Qdrant, Redis still remembers the file was processed.

**Solution:** ALWAYS flush Redis for re-ingestion:
```bash
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

### entity_metadata Missing from Qdrant Payloads

**Symptom:** GLiNER logs success but Qdrant payloads don't have entity_metadata

**Root Cause:** This was a bug fixed in Phase 2 - the `entity_metadata` field wasn't included in the Qdrant payload dict in `atomic.py`.

**Fix Applied:** Line 3153 of `src/ingestion/atomic.py` now includes:
```python
"entity_metadata": section.get("entity_metadata"),
```

**Verification:**
```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("chunks_multi_bge_m3", limit=1, with_payload=True, with_vectors=False)
print("entity_metadata present:", "entity_metadata" in points[0].payload)
```

---

## Caveats and Gotchas

### 1. Docker Volume Mount Timing
Code changes in `src/` are immediately visible inside containers due to volume mounts. However, **Python module caching** means the worker may not pick up changes until restart:
```bash
# After code changes, ALWAYS restart:
docker compose restart ingestion-worker
```

### 2. host.docker.internal Only Works on macOS/Windows
The `host.docker.internal` DNS name is a Docker Desktop feature. On Linux:
- Use the host's actual IP address
- Or use `--network host` mode
- Or create a custom bridge network

### 3. MPS Memory Limits
Apple Silicon MPS shares memory with the system. Large batches may cause memory pressure:
```yaml
# If OOM errors, reduce batch size in config/development.yaml:
ner:
  batch_size: 16  # Instead of 32
```

### 4. GLiNER Token Limit (384)
GLiNER has a 384 token limit. Longer chunks get truncated with a warning:
```
UserWarning: Sentence of length 385 has been truncated to 384
```
This is normal and doesn't cause errors - entities are still extracted from the truncated text.

### 5. Entity Type Overlap
GLiNER may assign the same text span to multiple entity types. For example, "LDAP" might be tagged as both `network_or_storage_protocol` and `cloud_provider_or_service`. This is by design for zero-shot models and improves recall at the cost of precision.

---

*Session context saved: 2025-12-08 20:30 EST*
*Git commit: d0c0e75 on dense-graph-enhance branch*
*Status: GLiNER integration complete and production-ready*

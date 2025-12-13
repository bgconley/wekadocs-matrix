# Session Context: GLiNER Integration - Phase 3 Complete

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** GLiNER Phase 3 - Qdrant Collection Schema Implementation
**Previous Context:** Phase 1 (Core Infrastructure) + Phase 2 (Ingestion Pipeline) + Phase 2.5 (MPS Acceleration)
**Document Version:** 5.0

---

## Executive Summary

This session completed **Phase 3 of the GLiNER NER integration** - the Qdrant Collection Schema update. Four new payload indexes were added for entity metadata fields, enabling efficient filtering and post-retrieval boosting for entity-aware hybrid search.

**Key Accomplishments:**
1. Added 4 entity metadata payload indexes to `src/shared/qdrant_schema.py`
2. Applied indexes to live Qdrant collection (24 → 28 total indexes)
3. Updated canonical schema file with new indexes
4. Added `require_entity_metadata_indexes` validation parameter
5. Wrote 4 new tests (2 unit, 2 integration) and fixed 3 outdated tests
6. Verified live schema matches canonical schema

**Test Results:** 8 tests passing (4 unit + 4 integration)

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

# Example: Find chunks with entity metadata
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) WHERE c.heading CONTAINS 'NFS' RETURN c.heading LIMIT 10"

# Clear ALL data (for clean re-ingestion - DESTRUCTIVE)
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

**Important Note:** With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval queries but **still used during ingestion**. GLiNER entities are NOT written to Neo4j (vector-only enrichment).

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

**Payload Indexes (28 total - Phase 3 Complete):**

The collection now has 28 payload indexes including the 4 new entity metadata indexes:
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

# Scroll through points with payloads
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("chunks_multi_bge_m3", limit=5, with_payload=True, with_vectors=False)
for p in points:
    em = p.payload.get("entity_metadata", {})
    print(f"{p.payload.get('heading', '')[:40]}: {em.get('entity_count', 0)} entities")
EOF
```

**Schema Verification Script:**
```bash
python3 << 'EOF'
import json
from qdrant_client import QdrantClient

# Load canonical schema
with open("scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json") as f:
    canonical = json.load(f)

client = QdrantClient(host="localhost", port=6333)
info = client.get_collection("chunks_multi_bge_m3")

live_indexes = set(info.payload_schema.keys())
expected_indexes = set(canonical["payload_indexes"])

if live_indexes >= expected_indexes:
    print("✅ Schema verification passed")
else:
    print(f"❌ Missing indexes: {expected_indexes - live_indexes}")
EOF
```

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
- **ALWAYS flush Redis** before re-ingesting the same documents (worker tracks processed files by hash)
- Debugging stuck ingestion jobs (RQ queue inspection)
- Cache invalidation during development

**When NOT to Access Redis:**
- Vector retrieval testing (Redis not in retrieval path)
- Schema verification (Qdrant only)
- Graph queries (Neo4j only)

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

**Usage:** Called after initial retrieval to re-score candidates with cross-encoder attention. Sets the authoritative `rerank_score` field in results.

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

**Starting the Service:**
```bash
cd services/gliner-ner && ./run.sh                    # Auto-detect device (MPS preferred)
cd services/gliner-ner && ./run.sh --device cpu       # Force CPU mode
cd services/gliner-ner && ./run.sh --port 9003        # Custom port
```

**Health Check:**
```bash
curl -s http://127.0.0.1:9002/healthz
# Returns: {"status": "ok", "device": "mps", "model": "urchade/gliner_medium-v2.1", "mps_available": true, "cuda_available": false}
```

**Test Entity Extraction:**
```bash
curl -s -X POST http://127.0.0.1:9002/v1/extract -H "Content-Type: application/json" \
  -d '{"texts": ["Mount NFS on RHEL 8"], "labels": ["operating_system", "network_or_storage_protocol"], "threshold": 0.4}'
```

**Stopping the Service:**
```bash
pkill -f "server.py"
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

**The worker tracks processed files by content hash.** Without cleanup, files won't re-ingest even after code changes or container restart.

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
# Or: cp path/to/your/test/docs/*.md data/ingest/
```

---

## GLiNER Implementation Status

### Phase 1: Core Infrastructure ✅ COMPLETE

**Files Created:**
- `src/providers/ner/__init__.py` - Package exports
- `src/providers/ner/labels.py` - Label configuration & helpers (11 domain-specific labels)
- `src/providers/ner/gliner_service.py` - Singleton service with HTTP + local fallback

**Files Modified:**
- `src/shared/config.py` - Added `NERConfig` class with `service_url` field
- `config/development.yaml` - Added `ner:` block with 11 labels and service_url
- `docker-compose.yml` - Fixed `HF_CACHE` → `HF_HOME` (3 occurrences)

### Phase 2: Document Ingestion Pipeline ✅ COMPLETE

**Files Created:**
- `src/ingestion/extract/ner_gliner.py` (148 lines) - `enrich_chunks_with_entities()` function
- `tests/unit/test_ner_gliner_enrichment.py` (290 lines) - 15 unit tests
- `tests/integration/test_gliner_ingestion_flow.py` (424 lines) - 9 integration tests

**Files Modified:**
- `src/ingestion/atomic.py` - 4 integration hooks:
  - Lines 991-1006: GLiNER enrichment call after chunk assembly
  - Lines 1204-1207: Prefer `_embedding_text` for embedding generation
  - Line 3153: **BUG FIX** - Added `entity_metadata` to Qdrant payload
  - Lines 2384-2386: Filter `source="gliner"` from Neo4j MENTIONS

### Phase 2.5: MPS Acceleration ✅ COMPLETE

**Files Created:**
- `services/gliner-ner/server.py` (280 lines) - FastAPI MPS service
- `services/gliner-ner/requirements.txt` - Dependencies
- `services/gliner-ner/run.sh` - Startup script

**Files Modified:**
- `src/providers/ner/gliner_service.py` - Complete rewrite with HTTP client + local fallback
- `src/shared/config.py` - Added `service_url: Optional[str]` to NERConfig
- `config/development.yaml` - Added `service_url: "http://host.docker.internal:9002"`

### Phase 3: Qdrant Collection Schema ✅ COMPLETE (This Session)

**Files Modified:**
- `src/shared/qdrant_schema.py`:
  - Added 4 entity metadata indexes to `build_qdrant_schema()` (lines 159-165)
  - Added `require_entity_metadata_indexes` parameter to `validate_qdrant_schema()` (line 187)
  - Added validation logic for entity indexes (lines 303-320)

- `scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json`:
  - Updated canonical schema with 4 new indexes (24 → 28 total)
  - Added `_updated_at` and `_update_notes` metadata fields

- `tests/shared/test_qdrant_schema.py`:
  - Fixed `test_schema_plan_dense_only_profile` to expect `doc_title` vector
  - Fixed `test_schema_plan_bge_profile_includes_sparse_and_colbert` to expect `entity-sparse`
  - Added `test_schema_plan_includes_entity_metadata_indexes` - verifies all 4 indexes present
  - Added `test_schema_plan_entity_metadata_index_types` - verifies correct PayloadSchemaType

- `tests/shared/test_qdrant_schema_validation.py`:
  - Fixed `test_validate_qdrant_schema_strict_passes` to include all required vectors
  - Added `test_validate_entity_metadata_indexes_present` - validates indexes exist
  - Added `test_validate_entity_metadata_indexes_missing_fails` - validates error handling

**Live Qdrant Changes Applied:**
```python
# Entity metadata indexes now live in production Qdrant collection
("entity_metadata.entity_types", PayloadSchemaType.KEYWORD)
("entity_metadata.entity_values", PayloadSchemaType.KEYWORD)
("entity_metadata.entity_values_normalized", PayloadSchemaType.KEYWORD)
("entity_metadata.entity_count", PayloadSchemaType.INTEGER)
```

### Phase 4: Hybrid Search Enhancement - NOT STARTED

**Planned Work (per canonical plan):**
1. Create `src/query/processing/disambiguation.py` with `QueryDisambiguator` class
2. Modify `src/query/hybrid_retrieval.py` to:
   - Extract entities from user queries using GLiNER
   - Perform post-retrieval boosting based on entity matches
   - Over-fetch candidates (2x top_k) when entity boosting is active
3. Add integration tests for entity-aware retrieval

**Key Implementation Detail from Plan:**
```python
# Post-retrieval boosting (NOT Qdrant filters - they don't boost scores)
for res in results:
    doc_entities = res.payload.get("entity_metadata", {}).get("entity_values_normalized", [])
    matches = sum(1 for term in boost_terms if term in doc_entities)
    if matches > 0:
        boost_factor = 1.0 + min(0.5, matches * 0.1)  # Max 50% boost
        res.score *= boost_factor
```

### Phase 5: Performance Optimization - NOT STARTED

**Planned Work:**
- Production tuning for batch sizes
- Thread safety for high concurrency
- Increased ingestion timeouts for GLiNER-enabled processing

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

## Architectural Decisions and Discoveries

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

### 4. Dense Entity Vector Deprecated

The original plan included a dense "entity" vector. This was **removed** because:
- It duplicated the content embedding (broken implementation)
- Replaced by `entity-sparse` for lexical entity name matching
- The `include_entity` parameter is now a no-op for backward compatibility

### 5. Post-Retrieval Boosting (NOT Qdrant Filters)

Qdrant `should` filters don't boost scores - they only filter. Entity boosting must be done in Python after retrieval (Phase 4 work).

---

## Known Issues and Warnings

### 1. Jaeger Tracing Errors (Cosmetic)
```
socket.gaierror: [Errno -2] Name or service not known
```
**Status:** Jaeger container not running. Does not affect functionality.

### 2. GLiNER API Deprecation Warning
```
FutureWarning: GLiNER.batch_predict_entities is deprecated.
Please use GLiNER.inference instead.
```
**Status:** Noted for future migration before GLiNER 1.0.

### 3. Truncation Warnings
```
UserWarning: Sentence of length 385 has been truncated to 384
```
**Status:** Normal for long chunks. GLiNER has 384 token limit.

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

## Test Results Summary

```
Phase 1 (Core Infrastructure):
  - Unit tests:           21 passed ✓
  - Config integration:   13 passed ✓
  - Live model tests:      8 passed ✓

Phase 2 (Ingestion Pipeline):
  - Unit tests:           15 passed ✓
  - Integration tests:     9 passed ✓

Phase 3 (Qdrant Schema) - This Session:
  - Unit tests:            4 passed ✓
  - Integration tests:     4 passed ✓
─────────────────────────────────────────
Total Verified:           74 passed ✓
```

---

## Quick Reference Commands

### Start Full Stack
```bash
# Start databases and app services
docker compose up -d

# Start native GLiNER service (separate terminal, NOT in Docker)
cd services/gliner-ner && ./run.sh
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
docker logs -f weka-ingestion-worker 2>&1 | grep -E "(gliner|enrichment|ingestion_complete)"
```

### Verify Entity Extraction
```bash
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("chunks_multi_bge_m3", limit=5, with_payload=["heading", "entity_metadata"], with_vectors=False)
for p in points:
    em = p.payload.get("entity_metadata", {})
    print(f"{p.payload['heading'][:40]}: {em.get('entity_count', 0)} entities")
EOF
```

---

## Related Documentation

- **GLiNER Implementation Plan:** `/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md` (v1.7 - CANONICAL)
- **Canonical Qdrant Schema:** `/scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json`
- **Configuration Reference:** `/config/development.yaml`
- **Previous Session Context:** `/docs/cdx-outputs/session-context-20251207-vector-architecture-reform.md`

---

## Next Steps

### Immediate (Phase 4 Implementation)
1. Create `src/query/processing/disambiguation.py` with `QueryDisambiguator`
2. Add query-time entity extraction using GLiNER service
3. Modify `src/query/hybrid_retrieval.py` for post-retrieval entity boosting
4. Write integration tests for entity-aware retrieval

### Post-Phase 4
1. Full corpus re-ingestion with GLiNER enabled
2. A/B testing retrieval quality with/without entity enrichment
3. Performance tuning (Phase 5)

---

## Troubleshooting Guide

### Issue: GLiNER Service Won't Start

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

### Issue: Docker Can't Reach Native GLiNER Service

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

### Issue: No Entities Being Extracted

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

### Issue: entity_metadata Missing from Qdrant Payloads

**Symptom:** GLiNER logs success but Qdrant payloads don't have entity_metadata

**Root Cause:** This was a bug fixed in Phase 2 - the `entity_metadata` field wasn't included in the Qdrant payload dict in `atomic.py`.

**Fix Applied:** Line 3153 of `src/ingestion/atomic.py` now includes:
```python
"entity_metadata": section.get("entity_metadata"),
```

**Verification:**
```bash
python3 << 'EOF'
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("chunks_multi_bge_m3", limit=1, with_payload=True, with_vectors=False)
print("entity_metadata present:", "entity_metadata" in points[0].payload)
EOF
```

### Issue: Documents Won't Re-Ingest

**Symptom:** Files in `data/ingest/` are ignored, no processing logs

**Root Cause:** RQ worker tracks processed files by content hash in Redis. Even after clearing Neo4j and Qdrant, Redis still remembers the file was processed.

**Solution:** ALWAYS flush Redis for re-ingestion:
```bash
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

### Issue: Schema Validation Fails on Entity Indexes

**Symptom:** `RuntimeError: Qdrant collection missing required entity metadata payload indexes`

**Root Cause:** Entity indexes haven't been applied to the live collection.

**Solution:** Apply the indexes manually:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

client = QdrantClient(host="localhost", port=6333)
for field, schema in [
    ("entity_metadata.entity_types", PayloadSchemaType.KEYWORD),
    ("entity_metadata.entity_values", PayloadSchemaType.KEYWORD),
    ("entity_metadata.entity_values_normalized", PayloadSchemaType.KEYWORD),
    ("entity_metadata.entity_count", PayloadSchemaType.INTEGER),
]:
    client.create_payload_index("chunks_multi_bge_m3", field, schema, wait=True)
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

### 5. Two MCP Server Architecture

There are TWO MCP servers - don't confuse them:

| Server | File | Protocol | Primary Tool | Usage |
|--------|------|----------|--------------|-------|
| HTTP Server | `src/mcp_server/main.py` | HTTP REST on :8000 | `search_documentation` | LEGACY - human-facing |
| STDIO Server | `src/mcp_server/stdio_server.py` | stdin/stdout pipes | `search_sections` | PRODUCTION - Agent/Claude |

**STDIO Server Invocation:**
```bash
docker exec -i weka-mcp-server python -m src.mcp_server.stdio_server
```

### 6. Qdrant Payload Index Timing

When you create a payload index, Qdrant indexes existing data asynchronously. For large collections, this may take seconds to minutes. Use `wait=True` in API calls to block until complete.

---

## File Locations Reference

### Core Implementation Files

| File | Purpose | Phase |
|------|---------|-------|
| `src/providers/ner/gliner_service.py` | GLiNER singleton with HTTP + fallback | Phase 1 |
| `src/providers/ner/labels.py` | Domain-specific entity labels | Phase 1 |
| `src/ingestion/extract/ner_gliner.py` | Chunk enrichment function | Phase 2 |
| `src/ingestion/atomic.py` | Main ingestion pipeline (modified) | Phase 2 |
| `src/shared/qdrant_schema.py` | Schema definition + validation | Phase 3 |
| `services/gliner-ner/server.py` | Native MPS service | Phase 2.5 |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/development.yaml` | Main config with NER settings |
| `docker-compose.yml` | Container definitions |
| `scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json` | Canonical schema |

### Test Files

| File | Tests | Type |
|------|-------|------|
| `tests/unit/test_gliner_service.py` | 21 | Unit |
| `tests/unit/test_ner_gliner_enrichment.py` | 15 | Unit |
| `tests/integration/test_gliner_config.py` | 13 | Integration |
| `tests/integration/test_gliner_live.py` | 8 | Live model |
| `tests/integration/test_gliner_ingestion_flow.py` | 9 | Integration |
| `tests/shared/test_qdrant_schema.py` | 4 | Unit |
| `tests/shared/test_qdrant_schema_validation.py` | 4 | Integration |

---

## Current System State

### What Works
1. **GLiNER service** - Fully functional with MPS acceleration (65x faster)
2. **Entity extraction** - 11 domain-specific labels, ~15 entities/chunk average
3. **Ingestion pipeline** - GLiNER enrichment integrated, entity_metadata persisted
4. **Qdrant schema** - 28 payload indexes including 4 entity metadata indexes
5. **Validation** - Schema validation with `require_entity_metadata_indexes` parameter
6. **All existing functionality** - No regressions introduced

### What's Disabled (Intentionally)
1. **ner.enabled: true** - GLiNER is enabled for ingestion
2. **neo4j_disabled: true** - Graph queries bypassed during retrieval (but ingestion still uses Neo4j)
3. **microdoc_stub creation** - Disabled in previous session
4. **doc_fallback** - Disabled to prevent cross-topic chunk pollution

### What's Not Yet Implemented
1. **Query-time entity extraction** (Phase 4) - `QueryDisambiguator` not created
2. **Post-retrieval entity boosting** (Phase 4) - `HybridRetriever` not modified
3. **Performance tuning** (Phase 5) - Production optimization not done

### Database State
All three databases contain production data:
- **Neo4j:** Documents, sections, chunks, structural entities
- **Qdrant:** 8-vector embeddings per chunk with entity_metadata in payloads, 28 payload indexes
- **Redis:** RQ job history and query cache

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **GLiNER** | Generalist Lightweight Named Entity Recognition - zero-shot NER model |
| **MPS** | Metal Performance Shaders - Apple Silicon GPU acceleration |
| **Zero-shot** | Model can extract entity types it wasn't explicitly trained on |
| **Transient field** | Field used during processing but not persisted to storage (e.g., `_embedding_text`) |
| **Circuit breaker** | Pattern that fails gracefully when dependencies unavailable |
| **entity-sparse** | Sparse vector generated from entity names for lexical matching |
| **`_mentions`** | Internal field containing entity references for a chunk |
| **Payload index** | Qdrant index on metadata fields for efficient filtering |
| **RRF** | Reciprocal Rank Fusion - method for combining multiple retrieval signals |
| **ColBERT** | Late-interaction retrieval using token-level MaxSim scoring |

---

*Session context saved: 2025-12-08 17:50 EST*
*Document word count: ~4800 words (~6400 tokens)*
*Next session: Begin Phase 4 (Hybrid Search Enhancement)*

# Session Context: GLiNER Integration Complete + OTEL Suppression

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** GLiNER code review completion, STDIO server fix, OTEL error investigation
**Previous Context:** GLiNER Phases 1-5 implementation complete
**Document Version:** 7.0
**Last Commit:** `44a1280` - "feat: complete GLiNER NER integration (Phases 1-5)"

---

## Executive Summary

This session completed the comprehensive code review of the GLiNER NER integration and fixed a critical issue where entity metadata fields were not exposed to the Agent via the STDIO server. Additionally, we investigated the Jaeger/OpenTelemetry errors in the ingestion worker logs and identified the fix (work in progress).

**Key Accomplishments:**
1. Fixed STDIO server to expose `entity_boost_applied` and `entity_metadata` fields
2. Verified microdoc chunks correctly populate entity_metadata from Qdrant payloads
3. Confirmed graph channel chunks intentionally lack entity_metadata (by design - vector-only enrichment)
4. Verified entity boost logic handles missing entity_metadata gracefully
5. Documented complete GLiNER embedding and retrieval flow
6. Identified OTEL/Jaeger error source and fix approach (not yet applied)

**Test Results:**
- Phase 4 Tests: 28 passed
- Qdrant Schema Tests: 8 passed
- All formatting/linting: Passed

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
```

---

## GLiNER Implementation Status

### Phase 1: Core Infrastructure ✅ COMPLETE

**Files Created:**
- `src/providers/ner/__init__.py` - Package exports
- `src/providers/ner/labels.py` - 11 domain-specific labels
- `src/providers/ner/gliner_service.py` - Singleton with HTTP + local fallback

**Files Modified:**
- `src/shared/config.py` - Added `NERConfig` class with `service_url` field
- `config/development.yaml` - Added `ner:` block with labels and service_url
- `docker-compose.yml` - Fixed `HF_CACHE` → `HF_HOME` (3 occurrences)

### Phase 2: Document Ingestion Pipeline ✅ COMPLETE

**Files Created:**
- `src/ingestion/extract/ner_gliner.py` (148 lines) - `enrich_chunks_with_entities()` function

**Files Modified:**
- `src/ingestion/atomic.py`:
  - Lines 991-1006: GLiNER enrichment call after chunk assembly
  - Lines 1204-1209: Prefer `_embedding_text` for dense and ColBERT embedding generation
  - Line 3153: `entity_metadata` added to Qdrant payload
  - Lines 2384-2388: Filter `source="gliner"` from Neo4j MENTIONS

### Phase 2.5: MPS Acceleration ✅ COMPLETE

**Files Created:**
- `services/gliner-ner/server.py` (280 lines) - FastAPI MPS service
- `services/gliner-ner/requirements.txt`
- `services/gliner-ner/run.sh`

### Phase 3: Qdrant Collection Schema ✅ COMPLETE

**Files Modified:**
- `src/shared/qdrant_schema.py` - Added 4 entity metadata indexes
- `scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json` - Updated canonical schema

### Phase 4: Hybrid Search Enhancement ✅ COMPLETE

**Files Created:**
- `src/query/processing/__init__.py`
- `src/query/processing/disambiguation.py` - `QueryDisambiguator` class

**Files Modified:**
- `src/query/hybrid_retrieval.py`:
  - Lines 165-168: Added `entity_metadata`, `entity_boost_applied` to ChunkResult
  - Line 1104: Populate `entity_metadata` from Qdrant payload
  - Lines 2775-2815: Query disambiguation and over-fetch logic
  - Lines 4814-4864: `_apply_entity_boost()` method

### Phase 5: Performance Optimization ✅ COMPLETE

**Files Modified:**
- `config/development.yaml` - `timeout_seconds: 300 → 600`
- `services/gliner-ner/server.py` - Thread safety documentation

---

## Fix Applied This Session

### STDIO Server Entity Fields

**File:** `src/mcp_server/stdio_server.py` (lines 539-541)

**Problem:** The STDIO `search_sections` response did not include `entity_boost_applied` or `entity_metadata`, meaning the Agent could not see entity boosting information.

**Fix Applied:**
```python
# GLiNER entity boosting metadata (Phase 4) - enables Agent to see entity-aware ranking
"entity_boost_applied": getattr(chunk, "entity_boost_applied", False),
"entity_metadata": getattr(chunk, "entity_metadata", None),
```

**Tests:** All 28 Phase 4 tests pass, all 8 Qdrant schema tests pass.

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

## OTEL/Jaeger Error Investigation (In Progress)

### Problem

The ingestion worker logs show repeated errors:
```
Exception while exporting Span batch.
socket.gaierror: [Errno -2] Name or service not known
requests.exceptions.ConnectionError: HTTPConnectionPool(host='jaeger', port=4318): Max retries exceeded
```

### Root Cause

The `.env` file sets `OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4318` but neither Alloy nor Jaeger containers are running. The docker-compose.yml for ingestion-worker has:
```yaml
- OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://alloy:4318}
- OTEL_TRACES_EXPORTER=otlp
```

### Fix Options (Not Yet Applied)

1. **Option A:** Comment out or remove `OTEL_EXPORTER_OTLP_ENDPOINT` from `.env` and set a sentinel value in docker-compose that the code will recognize as "disabled"

2. **Option B:** Modify `src/shared/observability/tracing.py` to check for a disable sentinel or add explicit disable flag

3. **Option C:** Simply start the Jaeger container if tracing is desired

**Relevant Code Location:** `src/shared/observability/tracing.py` lines 152-168 - if `endpoint` is falsy, no exporter is configured and no errors occur.

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

### Run GLiNER Tests
```bash
python -m pytest tests/unit/test_query_disambiguation.py tests/integration/test_phase4_entity_retrieval.py -v
```

### Monitor Ingestion (Filter OTEL Noise)
```bash
docker logs -f weka-ingestion-worker 2>&1 | grep -v "Exception while exporting" | grep -v "jaeger"
```

---

## Next Steps

### Immediate
1. **Suppress OTEL errors** - Apply fix to disable tracing when Jaeger/Alloy unavailable
2. **Commit STDIO fix** - The entity metadata exposure fix should be committed

### Future Work
1. Full corpus re-ingestion with GLiNER enabled
2. A/B testing retrieval quality with/without entity enrichment
3. Tuning boost parameters based on evaluation metrics

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

*Session context saved: 2025-12-08 ~19:30 EST*
*Document word count: ~4800 words (~6200 tokens)*
*Next session: Apply OTEL suppression fix, commit changes*

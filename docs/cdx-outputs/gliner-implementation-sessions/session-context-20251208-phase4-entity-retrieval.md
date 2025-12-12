# Session Context: GLiNER Phase 4 - Hybrid Search Entity Boosting

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** GLiNER NER Integration - Phase 4 Hybrid Search Implementation
**Previous Context:** Phases 1-3 Complete (Core Infrastructure, Ingestion Pipeline, Qdrant Schema)
**Document Version:** 5.0

---

## Executive Summary

This session completed **Phase 4 of the GLiNER NER integration** - the Hybrid Search Enhancement layer. The implementation adds query-time entity extraction using GLiNER and post-retrieval boosting to improve retrieval quality for entity-centric queries.

**Key Accomplishments:**
1. Created `QueryDisambiguator` class in `src/query/processing/disambiguation.py`
2. Added `entity_metadata` and `entity_boost_applied` fields to `ChunkResult` dataclass
3. Implemented `_apply_entity_boost()` method for post-retrieval score boosting
4. Added over-fetch logic (2x candidates) when query entities are detected
5. Integrated entity extraction into `HybridRetriever.retrieve()` method
6. Wrote 28 tests (15 unit + 13 integration) - all passing

**Test Results:**
- Phase 4 Unit Tests: 15 passed ✓
- Phase 4 Integration Tests: 13 passed ✓
- Qdrant Schema Tests: 8 passed ✓

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

**Payload Indexes (28 total - Phase 3 Complete):**

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

**Payload Structure for Entity-Aware Retrieval (Phase 4):**
Each Qdrant point payload now includes:
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

### Phase 3: Qdrant Collection Schema ✅ COMPLETE

**Files Modified:**
- `src/shared/qdrant_schema.py`:
  - Added 4 entity metadata indexes to `build_qdrant_schema()` (lines 159-165)
  - Added `require_entity_metadata_indexes` parameter to `validate_qdrant_schema()` (line 187)
  - Added validation logic for entity indexes (lines 303-320)

- `scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json`:
  - Updated canonical schema with 4 new indexes (24 → 28 total)

### Phase 4: Hybrid Search Enhancement ✅ COMPLETE (This Session)

**Files Created:**
- `src/query/processing/__init__.py` (5 lines) - Package exports
- `src/query/processing/disambiguation.py` (165 lines) - `QueryDisambiguator` class
- `tests/unit/test_query_disambiguation.py` (250 lines) - 15 unit tests
- `tests/integration/test_phase4_entity_retrieval.py` (280 lines) - 13 integration tests

**Files Modified:**
- `src/query/hybrid_retrieval.py`:
  - Line 50: Added import for `QueryAnalysis, QueryDisambiguator`
  - Lines 165-167: Added `entity_metadata` and `entity_boost_applied` to `ChunkResult` dataclass
  - Line 1102-1103: Populate `entity_metadata` from Qdrant payload
  - Line 2103: Added `_disambiguator` attribute initialization
  - Lines 4762-4818: Added `_get_disambiguator()` and `_apply_entity_boost()` methods
  - Lines 2775-2815: Added query disambiguation and over-fetch logic in `retrieve()`
  - Lines 2927-2946: Added entity boost application after fusion
  - Line 5383-5384: Added `entity_metadata` to microdoc ChunkResult

---

## Architectural Decisions (Phase 4)

### 1. Post-Retrieval Boosting (NOT Qdrant Filters)

Qdrant's `should` filters don't boost scores - they only filter results. Entity boosting must be done in Python after retrieval. This is a **soft filtering** approach that preserves recall while improving precision for entity-centric queries.

### 2. Over-Fetch Pattern for Entity Queries

When entities are found in the query, we fetch **2x more candidates** (6x top_k instead of 3x) to ensure good matches aren't cut before boosting is applied:
```python
entity_overfetch_multiplier = 2 if boost_terms else 1
candidate_k = min(top_k * 3 * entity_overfetch_multiplier, 200)
```

### 3. Boost Factor Calculation

Boost is capped at 50% maximum, with 10% per matching entity:
```python
boost_factor = 1.0 + min(0.5, matches * 0.1)  # Max 50% boost
```

### 4. Lazy Initialization

`QueryDisambiguator` is lazily initialized on first use via `_get_disambiguator()` to avoid unnecessary service connections when NER is disabled.

### 5. Graceful Degradation

If GLiNER service is unavailable or extraction fails, retrieval continues normally without boosting - no exceptions are raised to the caller.

---

## Known Issues and Technical Debt

### 1. Pre-existing GLiNER Service Test Failures

Some unit tests from Phase 1 fail due to the HTTP mode refactor in Phase 2.5:
- `test_explicit_device_config` - Tests `_device` attribute which is only set in local mode
- `test_auto_device_cpu_fallback` - Same issue
- `test_extract_entities_success` - Mock setup doesn't match HTTP-first architecture

**Status:** These are pre-existing failures, not caused by Phase 4 changes.

### 2. Config Test Failure

`test_config_ner_default_disabled` expects `ner.enabled: false` but config has it set to `true` from Phase 2 testing.

**Status:** Config intentionally set to `true` for active testing.

### 3. GLiNER API Deprecation Warning
```
FutureWarning: GLiNER.batch_predict_entities is deprecated.
Please use GLiNER.inference instead.
```
**Status:** Noted for future migration before GLiNER 1.0.

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

## Test Results Summary

```
Phase 1 (Core Infrastructure):
  - Unit tests:           21 passed ✓ (some pre-existing failures)
  - Config integration:   13 passed ✓ (1 pre-existing failure)
  - Live model tests:      8 passed ✓

Phase 2 (Ingestion Pipeline):
  - Unit tests:           15 passed ✓
  - Integration tests:     9 passed ✓

Phase 3 (Qdrant Schema):
  - Unit tests:            4 passed ✓
  - Integration tests:     4 passed ✓

Phase 4 (Hybrid Search) - This Session:
  - Unit tests:           15 passed ✓
  - Integration tests:    13 passed ✓
─────────────────────────────────────────────────
Total Phase 4:            28 passed ✓
```

---

## Next Steps

### Phase 5: Performance Optimization (NOT STARTED)
1. Production tuning for batch sizes
2. Thread safety for high concurrency
3. Increased ingestion timeouts for GLiNER-enabled processing
4. Apple Silicon (MPS) optimization verification

### Production Readiness
1. Full corpus re-ingestion with GLiNER enabled
2. A/B testing retrieval quality with/without entity enrichment
3. Tuning boost parameters (`max_boost`, `per_entity_boost`) based on evaluation
4. Consider GPU deployment for production scale
5. Add Prometheus metrics dashboard for GLiNER service

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

### Run Phase 4 Tests
```bash
python -m pytest tests/unit/test_query_disambiguation.py tests/integration/test_phase4_entity_retrieval.py -v
```

### Monitor Ingestion with GLiNER
```bash
docker logs -f weka-ingestion-worker 2>&1 | grep -E "(gliner|enrichment|entity_boost)"
```

---

## Related Documentation

- **GLiNER Implementation Plan:** `/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md` (v1.7 - CANONICAL)
- **Canonical Qdrant Schema:** `/scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json`
- **Configuration Reference:** `/config/development.yaml`
- **Previous Session Context:** `/docs/cdx-outputs/session-context-20251208-phase3-schema.md`

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **GLiNER** | Generalist Lightweight Named Entity Recognition - zero-shot NER model |
| **MPS** | Metal Performance Shaders - Apple Silicon GPU acceleration |
| **QueryDisambiguator** | Phase 4 class that extracts entities from user queries using GLiNER |
| **Post-retrieval boosting** | Applying score multipliers after fusion based on entity matches |
| **Over-fetch** | Retrieving more candidates than needed to ensure good results survive boosting |
| **boost_terms** | Normalized entity text values from query for matching against chunk entity_metadata |
| **entity_metadata** | Qdrant payload field containing entity_types, entity_values, entity_values_normalized, entity_count |
| **entity_boost_applied** | ChunkResult flag indicating this chunk received entity-based score boost |
| **Soft filtering** | Boosting matching results rather than excluding non-matching ones (preserves recall) |

---

*Session context saved: 2025-12-08 18:10 EST*
*Document word count: ~4800 words (~6400 tokens)*
*Next session: Phase 5 Performance Optimization or production deployment preparation*

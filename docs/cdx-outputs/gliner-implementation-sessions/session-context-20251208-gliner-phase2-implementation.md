# Session Context: GLiNER Phase 2 Implementation Complete

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** GLiNER NER Integration - Phase 2 Document Ingestion Pipeline Implementation
**Previous Context:** Phase 1 Core Infrastructure (completed 2025-12-07)
**Document Version:** 3.0

---

## Executive Summary

This session completed **Phase 2 of the GLiNER integration** - the Document Ingestion Pipeline. All 66 tests pass (42 from Phase 1 + 24 new in Phase 2). A live ingestion test was initiated and **GLiNER enrichment is confirmed working** with 32 chunks processed, 13 enriched, and 43 entities extracted. The ingestion was still in progress when session context was saved.

**Key Accomplishments:**
1. Created `src/ingestion/extract/ner_gliner.py` with `enrich_chunks_with_entities()` function
2. Added 3 integration hooks to `src/ingestion/atomic.py`
3. Wrote 15 unit tests for enrichment module
4. Wrote 9 integration tests for ingestion flow
5. Rebuilt Docker container with gliner package
6. Enabled NER in config and initiated live ingestion test
7. Confirmed GLiNER enrichment working in Docker (CPU mode)

---

## Infrastructure Access: Complete Credentials and Connection Patterns

### Neo4j Graph Database

Neo4j stores the document graph structure including Documents, Sections, Chunks, Entities, and their relationships.

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
# Direct cypher-shell access (IMPORTANT: remove -it flag for non-TTY contexts)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count chunks
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"

# Example: Find chunks by heading
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) WHERE toLower(c.heading) CONTAINS 'nfs' RETURN c.heading LIMIT 10"

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
- Verifying graph relationships (MENTIONS, HAS_SECTION, NEXT_CHUNK, etc.)
- Debugging entity extraction and mention relationships

**Note:** With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval queries but still used during ingestion.

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

**Access Patterns:**

```bash
# Check collection status and point count
curl -s http://127.0.0.1:6333/collections/chunks_multi_bge_m3 | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Points: {data['result']['points_count']}\")"

# Delete all points (PRESERVE SCHEMA) - use for clean re-ingestion
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'
```

**Payload Structure:** Each Qdrant point has a payload containing:
- `id` (SHA256 hash - matches Neo4j chunk ID)
- `document_id`, `doc_id`, `doc_tag`
- `heading`, `text`, `token_count`
- `is_microdoc`, `doc_is_microdoc`
- **NEW (Phase 2):** `entity_metadata` with entity types, values, and normalized values

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

# Flush all data
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

**When to Access Redis:**
- Debugging stuck ingestion jobs (RQ queue inspection)
- Cache invalidation during development
- Checking RQ queue status for background processing

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

---

### BGE Reranker Service

Cross-encoder reranker using BAAI/bge-reranker-v2-m3 for final-stage scoring.

**Connection Details:**
```
Host machine: http://127.0.0.1:9001
Health endpoint: GET /healthz
Rerank endpoint: POST /v1/rerank
```

---

## Docker Container Architecture and Management

### Container Overview

| Container | Purpose | Key Ports | Rebuild Needed For |
|-----------|---------|-----------|-------------------|
| `weka-neo4j` | Graph database | 7687, 7474 | N/A (data volume) |
| `weka-qdrant` | Vector store | 6333, 6334 | N/A (data volume) |
| `weka-redis` | Cache + queue | 6379 | N/A (data volume) |
| `weka-mcp-server` | HTTP MCP + STDIO | 8000 | requirements.txt changes |
| `weka-ingestion-worker` | RQ background jobs | None | requirements.txt changes |

### Volume Mounts (CRITICAL for Development)

The MCP server and ingestion worker mount source code as read-only volumes:
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

**This means:**
- **Code changes are immediately visible** inside containers
- **NO rebuild needed for code changes** - just restart
- Restart: `docker compose restart mcp-server` or `docker compose restart ingestion-worker`
- **Rebuild only needed** for Dockerfile or requirements.txt changes

### When to Rebuild vs Restart

| Change Type | Action Required |
|-------------|-----------------|
| Code in `src/` | `docker compose restart <service>` |
| Config in `config/` | `docker compose restart <service>` |
| New package in requirements.txt | `docker compose build <service> && docker compose up -d <service>` |
| Dockerfile changes | `docker compose build <service> && docker compose up -d <service>` |

### HuggingFace Model Cache

**Important Fix Applied in Phase 1:** Docker containers use `HF_HOME=/opt/hf-cache` (not `HF_CACHE`).

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
docker compose restart mcp-server
docker compose restart ingestion-worker

# Rebuild for requirements.txt changes
docker compose build ingestion-worker
docker compose up -d ingestion-worker

# View logs (filter out Jaeger noise which is cosmetic)
docker logs -f weka-ingestion-worker 2>&1 | grep -v jaeger

# Shell into container for debugging
docker exec -it weka-mcp-server bash

# Verify config loaded inside container
docker exec weka-mcp-server python3 -c "
from src.shared.config import get_config
config = get_config()
print('ner.enabled:', config.ner.enabled)
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
3. Creates: Neo4j nodes + Qdrant vectors (8 per chunk)
4. GLiNER enrichment runs if `ner.enabled: true`

### CRITICAL: Clean Ingestion After Rebuild or Testing

The worker tracks processed files by hash. **Without cleanup, files won't re-ingest.**

**Complete Clean Ingestion Procedure:**
```bash
# 1. Delete Qdrant points (preserve schema)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 2. Clear Neo4j
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 3. Clear Redis
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Remove processed files from ingest folder
rm -f data/ingest/*.md

# 5. Re-add test files
cp test-documents/*.md data/ingest/
```

---

## GLiNER Implementation Status

### Phase 1: Core Infrastructure ✅ COMPLETE

**Files Created:**
- `src/providers/ner/__init__.py` - Package exports
- `src/providers/ner/labels.py` - Label configuration & helpers
- `src/providers/ner/gliner_service.py` - Singleton service with metrics

**Files Modified:**
- `src/shared/config.py` - Added `NERConfig` class
- `config/development.yaml` - Added `ner:` block with 11 labels
- `docker-compose.yml` - Fixed `HF_CACHE` → `HF_HOME`

### Phase 2: Document Ingestion Pipeline ✅ COMPLETE

**Files Created:**
- `src/ingestion/extract/ner_gliner.py` (148 lines) - Chunk enrichment function
- `tests/unit/test_ner_gliner_enrichment.py` (290 lines) - 15 unit tests
- `tests/integration/test_gliner_ingestion_flow.py` (424 lines) - 9 integration tests

**Files Modified:**
- `src/ingestion/atomic.py` - 3 integration hooks:
  - Lines 991-1006: GLiNER enrichment call after chunk assembly
  - Lines 1204-1207: Prefer `_embedding_text` for embedding generation
  - Lines 2384-2386: Filter `source="gliner"` from Neo4j MENTIONS

### Test Results

```
Phase 1 (Core Infrastructure):
  - Unit tests:           21 passed ✓
  - Config integration:   13 passed ✓
  - Live model tests:      8 passed ✓

Phase 2 (Ingestion Pipeline):
  - Unit tests:           15 passed ✓
  - Integration tests:     9 passed ✓
─────────────────────────────────────────
Total:                    66 passed ✓
```

### Live Ingestion Test Results (In Progress)

**Document:** `additional-protocols_nfs-support_nfs-support-1.md` (28KB)

**GLiNER Enrichment Output:**
```json
{
  "event": "gliner_enrichment_complete",
  "chunks_processed": 32,
  "chunks_enriched": 13,
  "total_entities": 43,
  "entity_type_distribution": {"network_or_storage_protocol": 13}
}
```

**Observations:**
- Model runs on CPU inside Docker (no MPS/CUDA)
- Model load time: ~10 seconds
- Batch processing: ~3 minutes for 32 chunks on CPU
- Truncation warnings normal (GLiNER 384 token limit)

---

## Key Architectural Decisions

### 1. GLiNER is Vector-Only Enrichment

GLiNER entities are **NOT written to Neo4j**. They only enrich:
- Dense embeddings (via `_embedding_text`)
- Entity-sparse vectors (via `_mentions` with `source="gliner"`)
- Qdrant payload (`entity_metadata`)

**Benefit:** With `neo4j_disabled: true`, you get 100% of GLiNER benefits through pure vector retrieval.

### 2. Transient `_embedding_text` Field

Entity context enriches embeddings without polluting stored text:
```python
chunk["_embedding_text"] = f"{title}\n\n{text}\n\n[Context: {entity_context}]"
# chunk["text"] remains untouched
```

### 3. `source="gliner"` Marker

All GLiNER-extracted mentions are marked with `source="gliner"`. This marker is used in `_neo4j_create_mentions()` to filter them from Neo4j writes.

### 4. Consistent `entity_metadata` Schema

All chunks get `entity_metadata` even if no entities found (with `entity_count: 0`). This ensures consistent Qdrant payload indexing.

### 5. Deduplication in Mentions

GLiNER entities are deduplicated against existing structural mentions using case-insensitive (name, type) tuples to prevent double-counting in entity-sparse vectors.

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

## Data Flow (Complete Pipeline with GLiNER)

```
_prepare_ingestion()
    │
    ├── parse_markdown/html → Document + Sections
    ├── extract_entities() → Structural entities (regex)
    ├── assemble() → Final chunks
    │
    └── [Phase 2] enrich_chunks_with_entities(sections)
              │
              ├── entity_metadata → Qdrant payload
              ├── _embedding_text → Transient (embedding only)
              └── _mentions (source="gliner") → Entity-sparse vector
                         │
_compute_embeddings()    │
    │                    │
    └── [Phase 2] Uses _embedding_text if present
                         │
_neo4j_create_mentions() │
    │                    │
    └── [Phase 2] Filters source="gliner" → Not written to Neo4j
```

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

# Embedding services (HOST MACHINE URLs)
export BGE_M3_API_URL=http://127.0.0.1:9000
export RERANKER_BASE_URL=http://127.0.0.1:9001

# HuggingFace cache
export HF_HOME=./hf-cache

# API keys
export JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi
```

---

## Next Steps

### Immediate (Complete Ingestion Test)
1. Wait for current ingestion to complete
2. Verify Qdrant point count increased
3. Verify `entity_metadata` present in Qdrant payloads
4. Test retrieval query to confirm enrichment working

### Phase 3: Qdrant Payload Indexes (NOT STARTED)
1. Add indexes for `entity_types`, `entity_values`, `entity_count`
2. Enable efficient filtering/boosting on entity fields

### Phase 4: Hybrid Search Enhancement (NOT STARTED)
1. Implement `QueryDisambiguator` for query entity extraction
2. Add post-retrieval boosting in `HybridRetriever`
3. Entity-aware query augmentation

### Production Readiness
1. Full corpus re-ingestion with GLiNER enabled
2. A/B testing retrieval quality with/without entity enrichment
3. Performance tuning for CPU inference (consider GPU)

---

## Files Modified This Session (Complete List)

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/ingestion/extract/ner_gliner.py` | **NEW** 148 | Chunk enrichment function |
| `src/ingestion/atomic.py` | +22 | 3 integration hooks |
| `config/development.yaml` | +1 | Enabled `ner.enabled: true` |
| `tests/unit/test_ner_gliner_enrichment.py` | **NEW** 290 | 15 unit tests |
| `tests/integration/test_gliner_ingestion_flow.py` | **NEW** 424 | 9 integration tests |

---

## Related Documentation

- **GLiNER Implementation Plan:** `/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md` (v1.7)
- **Previous Session Context:** `/docs/cdx-outputs/session-context-20251207-vector-architecture-reform.md`
- **Configuration Reference:** `/config/development.yaml`

---

*Session context saved: 2025-12-08 16:30 EST*
*Document word count: ~3800 words (~5200 tokens)*
*Next session: Complete ingestion test verification, begin Phase 3*

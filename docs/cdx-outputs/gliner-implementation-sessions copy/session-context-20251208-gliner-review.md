# Session Context: GLiNER End-to-End Implementation Review

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** End-to-end code review of GLiNER NER integration (Phases 1-5)
**Previous Context:** GLiNER Phase 5 completion and commit
**Document Version:** 6.0
**Commit SHA:** `44a1280`

---

## Executive Summary

This session conducted an end-to-end code review of the complete GLiNER NER integration after all 5 phases were implemented and committed. The review was approximately 60% complete when context limits were reached.

**Review Status:**
- Phase 1 (Core Infrastructure): ✅ Reviewed - No issues found
- Phase 2 (Ingestion Pipeline): ✅ Reviewed - No issues found
- Phase 3 (Qdrant Schema): ✅ Reviewed - No issues found
- Phase 4 (Hybrid Search): ✅ Reviewed - No issues found
- Phase 5 (Performance): ✅ Already complete
- Data Flow Consistency: ⚠️ **IN PROGRESS - ISSUE DISCOVERED**

**Critical Finding:** The STDIO MCP server (`src/mcp_server/stdio_server.py`) does NOT expose `entity_metadata` or `entity_boost_applied` fields in search results, meaning the Agent cannot see entity boosting information.

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
# Direct cypher-shell access (CRITICAL: remove -it flag for non-TTY contexts)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "YOUR_QUERY"

# Example: Count chunks
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN count(c)"

# Clear ALL data (DESTRUCTIVE - for clean re-ingestion)
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"
```

**Important Note:** With `neo4j_disabled: true` in config, Neo4j is bypassed during retrieval queries but **still used during ingestion**. GLiNER entities are NOT written to Neo4j (vector-only enrichment pattern).

---

### Qdrant Vector Store

Qdrant stores all vector embeddings in the `chunks_multi_bge_m3` collection.

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
| `content` | Dense | 1024 | Main semantic content | 1.0 |
| `title` | Dense | 1024 | Section heading | 1.0 |
| `doc_title` | Dense | 1024 | Document title | 1.0 |
| `late-interaction` | Multi | 1024 × N | ColBERT MaxSim | N/A |
| `text-sparse` | Sparse | Variable | BM25-style content | 1.0 |
| `title-sparse` | Sparse | Variable | Heading terms | **2.0** |
| `doc_title-sparse` | Sparse | Variable | Document title terms | 1.0 |
| `entity-sparse` | Sparse | Variable | Entity names | **1.5** |

**Payload Indexes (28 total):**
Includes 4 entity metadata indexes added in Phase 3:
- `entity_metadata.entity_types` (KEYWORD)
- `entity_metadata.entity_values` (KEYWORD)
- `entity_metadata.entity_values_normalized` (KEYWORD)
- `entity_metadata.entity_count` (INTEGER)

**Access Patterns:**
```bash
# Check point count
curl -s "http://127.0.0.1:6333/collections/chunks_multi_bge_m3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}\")"

# Delete all points (PRESERVE SCHEMA)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'
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

**CRITICAL: Redis and Re-ingestion:**
The RQ worker tracks processed files by content hash in Redis. **Even after clearing Neo4j and Qdrant, Redis still remembers processed files.** You MUST flush Redis for re-ingestion:

```bash
# Check database size
docker exec weka-redis redis-cli -a testredis123 DBSIZE

# Flush all data (REQUIRED for clean re-ingestion)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL
```

**When to Access Redis:**
- **ALWAYS flush** before re-ingesting same documents
- Debugging stuck ingestion jobs
- Cache invalidation during development

**When NOT to Access Redis:**
- Vector retrieval testing (not in retrieval path)
- Schema verification (Qdrant only)
- Graph queries (Neo4j only)

---

### BGE-M3 Embedding Service

Provides dense (1024-D), sparse (BM25-style), and ColBERT multi-vector embeddings.

**CRITICAL: URL varies by context!**

| Context | Base URL |
|---------|----------|
| Host machine | `http://127.0.0.1:9000` |
| Inside Docker | `http://host.docker.internal:9000` |

**Endpoints:**
- `GET /healthz` - Health check
- `POST /v1/embeddings` - Dense embeddings
- `POST /v1/embeddings/sparse` - Sparse embeddings
- `POST /v1/embeddings/colbert` - ColBERT multi-vectors

**Health Check:**
```bash
curl -s http://127.0.0.1:9000/healthz
# Returns: {"status": "ok"}
```

---

### BGE Reranker Service

Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

**Connection Details:**
```
Host machine: http://127.0.0.1:9001
Health endpoint: GET /healthz
Rerank endpoint: POST /v1/rerank
```

---

### GLiNER NER Service (Native MPS-Accelerated)

Native macOS service with Metal Performance Shaders acceleration. **65x faster** than CPU.

**Connection Details:**
```
Host machine: http://127.0.0.1:9002
Inside Docker: http://host.docker.internal:9002
```

**Starting/Stopping:**
```bash
cd services/gliner-ner && ./run.sh    # Start
pkill -f "server.py"                   # Stop
```

**Health Check:**
```bash
curl -s http://127.0.0.1:9002/healthz
# Returns: {"status": "ok", "device": "mps", ...}
```

---

## Docker Container Architecture

### Container Overview

| Container | Purpose | Key Ports | Volume Mounts |
|-----------|---------|-----------|---------------|
| `weka-neo4j` | Graph database | 7687, 7474 | Data volume |
| `weka-qdrant` | Vector store | 6333, 6334 | Data volume |
| `weka-redis` | Cache + queue | 6379 | Data volume |
| `weka-mcp-server` | HTTP MCP + STDIO | 8000 | `./src:/app/src:ro` |
| `weka-ingestion-worker` | RQ background jobs | None | `./src:/app/src:ro` |

### Volume Mounts (CRITICAL for Development)

```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

**This means:**
- **Code changes are immediately visible** inside containers
- **NO rebuild needed for code changes** - just restart
- **Rebuild only needed** for requirements.txt or Dockerfile changes

### When to Rebuild vs Restart

| Change Type | Action Required |
|-------------|-----------------|
| Code in `src/` | `docker compose restart <service>` |
| Config in `config/` | `docker compose restart <service>` |
| requirements.txt | `docker compose build <service> && docker compose up -d <service>` |

### HuggingFace Model Cache

**Important Fix Applied:** Docker containers use `HF_HOME=/opt/hf-cache` (NOT `HF_CACHE`).

---

## Data Ingestion Workflow

### File-Drop Pattern

**Host path:** `./data/ingest/`
**Container path:** `/app/data/ingest/`

**Process:**
1. Drop markdown files in `data/ingest/`
2. Worker auto-detects and processes
3. Creates Neo4j nodes + Qdrant vectors (8 per chunk)
4. GLiNER enrichment runs if `ner.enabled: true`

### CRITICAL: Clean Ingestion Procedure

**The worker tracks processed files by content hash.** Without cleanup, files won't re-ingest.

```bash
# 1. Delete Qdrant points (PRESERVE schema)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 2. Clear Neo4j
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 3. Clear Redis (CRITICAL!)
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Remove and re-add test files
rm -f data/ingest/*.md
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
- `src/shared/config.py` - Added `NERConfig` class
- `config/development.yaml` - Added `ner:` block
- `docker-compose.yml` - Fixed `HF_CACHE` → `HF_HOME`

**Code Review Findings:** No issues found. Proper singleton pattern, circuit breaker, LRU caching.

### Phase 2: Document Ingestion Pipeline ✅ COMPLETE

**Files Created:**
- `src/ingestion/extract/ner_gliner.py` - `enrich_chunks_with_entities()` function

**Files Modified:**
- `src/ingestion/atomic.py`:
  - Lines 991-1006: GLiNER enrichment call after chunk assembly
  - Lines 1204-1207: Prefer `_embedding_text` for embedding generation
  - Line 3153: `entity_metadata` added to Qdrant payload
  - Lines 2384-2388: Filter `source="gliner"` from Neo4j MENTIONS

**Code Review Findings:** No issues found. Proper transient field pattern, deduplication, Neo4j filtering.

### Phase 2.5: MPS Acceleration ✅ COMPLETE

**Files Created:**
- `services/gliner-ner/server.py` - FastAPI MPS service
- `services/gliner-ner/requirements.txt`
- `services/gliner-ner/run.sh`

**Code Review Findings:** No issues found. Proper thread safety documentation added.

### Phase 3: Qdrant Collection Schema ✅ COMPLETE

**Files Modified:**
- `src/shared/qdrant_schema.py` - Added 4 entity metadata indexes
- `scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json` - Updated

**Code Review Findings:** No issues found. Indexes correctly defined.

### Phase 4: Hybrid Search Enhancement ✅ COMPLETE

**Files Created:**
- `src/query/processing/__init__.py`
- `src/query/processing/disambiguation.py` - `QueryDisambiguator` class

**Files Modified:**
- `src/query/hybrid_retrieval.py`:
  - Lines 165-168: Added `entity_metadata`, `entity_boost_applied` to ChunkResult
  - Line 1104: Populate `entity_metadata` from Qdrant payload
  - Lines 2775-2815: Query disambiguation and over-fetch logic
  - Lines 4808-4812: `_get_disambiguator()` method
  - Lines 4814-4864: `_apply_entity_boost()` method

**Code Review Findings:** No issues found in core logic. See ISSUE DISCOVERED below.

### Phase 5: Performance Optimization ✅ COMPLETE

**Files Modified:**
- `config/development.yaml` - `timeout_seconds: 300 → 600`
- `services/gliner-ner/server.py` - Thread safety documentation

---

## CRITICAL ISSUE DISCOVERED

### STDIO Server Missing Entity Fields

**Location:** `src/mcp_server/stdio_server.py` lines 494-539

**Problem:** The `search_sections` tool response builds results but does NOT include:
- `entity_boost_applied` - Whether chunk was boosted by entity matching
- `entity_metadata` - The entity metadata from the chunk payload

**Impact:** The Agent (Claude) cannot see:
1. Which results were entity-boosted
2. What entities are in each chunk
3. Whether entity boosting affected ranking

**Current Response Fields (lines 494-539):**
```python
{
    "section_id": chunk.chunk_id,
    "title": chunk.heading,
    "tokens": chunk.token_count,
    "score": float(score),
    "source": source,
    "fusion_method": chunk.fusion_method,
    "fused_score": ...,
    "title_vec_score": ...,
    "entity_vec_score": ...,  # ← This exists (sparse vector score)
    # MISSING: entity_boost_applied
    # MISSING: entity_metadata
}
```

**Recommended Fix:**
Add to the result dict at line ~538:
```python
"entity_boost_applied": getattr(chunk, "entity_boost_applied", False),
"entity_metadata": getattr(chunk, "entity_metadata", None),
```

---

## Code Review: Remaining Items to Check

The review was interrupted at approximately 60% completion. The following items need verification:

### 1. Microdoc/Stub Chunks Entity Handling
**Files to check:**
- `src/query/hybrid_retrieval.py` lines 5380-5528 (microdoc ChunkResult creation)
- Verify `entity_metadata` is populated for microdoc chunks

**Concern:** Microdoc chunks are created dynamically during retrieval. Need to verify they either:
- Pull `entity_metadata` from Qdrant payload, OR
- Have `entity_metadata=None` which is handled gracefully in boosting

### 2. Graph Channel ChunkResult
**Files to check:**
- `src/query/hybrid_retrieval.py` lines 4866-4889 (`_graph_retrieval_channel`)
- Verify graph-retrieved chunks have `entity_metadata`

### 3. Edge Case: Empty entity_metadata in Boosting
**Location:** `src/query/hybrid_retrieval.py` lines 4843-4850

**Current Code:**
```python
entity_metadata = res.entity_metadata or {}
doc_entities = entity_metadata.get("entity_values_normalized", [])
if not doc_entities:
    continue
```

**Status:** This looks correct - handles None gracefully.

### 4. Edge Case: Service URL Not Set
**Location:** `src/providers/ner/gliner_service.py`

**Status:** Reviewed - handled correctly. Falls back to local model.

### 5. Singleton Reset Between Tests
**Location:** `src/providers/ner/gliner_service.py` line 584-598

**Status:** `reset()` method exists but need to verify tests are using it.

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

# Embedding services
export BGE_M3_API_URL=http://127.0.0.1:9000
export RERANKER_BASE_URL=http://127.0.0.1:9001
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

## Test Results (All Phases)

```
Phase 1 (Core Infrastructure): 42 passed
Phase 2 (Ingestion Pipeline):  24 passed
Phase 3 (Qdrant Schema):        8 passed
Phase 4 (Hybrid Search):       28 passed
─────────────────────────────────────────
Total:                        102+ passed
```

---

## Quick Reference Commands

### Start Full Stack
```bash
docker compose up -d
cd services/gliner-ner && ./run.sh  # Separate terminal
```

### Verify Services
```bash
curl -s http://127.0.0.1:9000/healthz  # BGE
curl -s http://127.0.0.1:9001/healthz  # Reranker
curl -s http://127.0.0.1:9002/healthz  # GLiNER
```

### Monitor Ingestion
```bash
docker logs -f weka-ingestion-worker 2>&1 | grep -E "(gliner|enrichment|entity)"
```

---

## Next Steps

### Immediate (Code Review Completion)
1. **Fix STDIO server** - Add `entity_boost_applied` and `entity_metadata` to response
2. Verify microdoc chunks handle entity_metadata correctly
3. Verify graph channel chunks handle entity_metadata correctly
4. Complete remaining edge case verification

### Post-Review
1. Full corpus re-ingestion with GLiNER enabled
2. A/B testing retrieval quality with/without entity enrichment
3. Tune boost parameters based on evaluation

---

## Related Documentation

- **Canonical Plan:** `/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md` (v1.7)
- **Schema Reference:** `/scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json`
- **Configuration:** `/config/development.yaml`

---

## Glossary

| Term | Definition |
|------|------------|
| **GLiNER** | Generalist Lightweight Named Entity Recognition |
| **MPS** | Metal Performance Shaders (Apple Silicon GPU) |
| **entity-sparse** | Sparse vector from entity names |
| **`_embedding_text`** | Transient field with entity context for embeddings |
| **`entity_metadata`** | Qdrant payload field with entity types/values |
| **`entity_boost_applied`** | ChunkResult flag for boosted results |
| **Post-retrieval boosting** | Score multiplication after fusion |

---

*Session context saved: 2025-12-08 ~19:00 EST*
*Review status: ~60% complete*
*Critical issue: STDIO server missing entity fields*
*Next session: Fix STDIO server, complete review*

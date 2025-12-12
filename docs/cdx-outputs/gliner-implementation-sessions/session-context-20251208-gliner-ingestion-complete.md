# Session Context: GLiNER Integration Complete - Full Corpus Ingestion

**Session Date:** 2025-12-08 (evening) / 2025-12-09 (early morning UTC)
**Branch:** `dense-graph-enhance`
**Last Commit:** `d0c0e75` - "feat: complete GLiNER NER integration (Phases 1-5)"
**Document Version:** 9.0

---

## Executive Summary

This session completed the first full corpus ingestion with GLiNER NER (Named Entity Recognition) enabled. All 54 test documents were successfully ingested, producing 288 Qdrant vector points with entity-enriched embeddings. GLiNER extracted over 1,100 domain-specific entities across 11 configured entity types.

**Key Metrics:**
- Documents ingested: 54
- Qdrant points: 288
- GLiNER batches processed: 54
- Total entities extracted: ~1,100+
- Entity types active: 11

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

# Example: Count documents
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (d:Document) RETURN count(d)"

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

# View logs (filter out OTEL/Jaeger noise which is cosmetic)
docker logs -f weka-ingestion-worker 2>&1 | grep -v "Exception while exporting" | grep -v jaeger

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

## GLiNER Implementation Status: ALL PHASES COMPLETE

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1.0 | Core Infrastructure (GLiNERService singleton, NERConfig) | ✅ Complete |
| Phase 2.0 | Ingestion Pipeline (enrich_chunks_with_entities) | ✅ Complete |
| Phase 2.5 | MPS Acceleration (native service at :9002) | ✅ Complete |
| Phase 3.0 | Qdrant Schema (4 entity metadata indexes) | ✅ Complete |
| Phase 4.0 | Hybrid Search (QueryDisambiguator, post-retrieval boosting) | ✅ Complete |
| Phase 4.4 | STDIO Entity Exposure (entity_metadata + entity_boost_applied) | ✅ Complete |
| Phase 5.0 | Performance (600s ingestion timeout) | ✅ Complete |

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

## GLiNER Entity Extraction Results (This Session)

### Summary Statistics

| Metric | Value |
|--------|-------|
| Documents processed | 54 |
| GLiNER batches | 54 |
| Total entities extracted | ~1,111+ |
| Average entities/chunk | ~4-5 |

### Entity Type Distribution (Sample of 20 Batches)

| Entity Type | Count | Examples |
|-------------|-------|----------|
| `cloud_provider_or_service` | 57 | S3, AWS, IAM, GPUDirect Storage |
| `network_or_storage_protocol` | 54 | NFS, NFSv4, SMB, LDAP, Kerberos, POSIX |
| `configuration_parameter` | 41 | container-ids, policy, KRB5, KRB5i, KRB5p |
| `filesystem_object` | 32 | bucket, snapshot, inode, ACLs, objects |
| `cli_command` | 31 | weka s3, weka nfs, smb, bash |
| `file_system_path` | 14 | /mnt/weka, some/dir, existing-path |
| `operating_system` | 13 | RHEL, Ubuntu, Windows |
| `weka_software_component` | 7 | backend, frontend, WEKA |
| `error_message_or_code` | 2 | - |
| `performance_metric` | 2 | - |

### Rich Entity Examples

**"Manage the S3 protocol" (39 entities):**
- Types: cloud_provider_or_service, network_or_storage_protocol, filesystem_object
- Values: S3, WEKA, NFS, SMB, POSIX, GPUDirect Storage, objects, files

**"Configure the NFS global settings" (35 entities):**
- Types: configuration_parameter, network_or_storage_protocol
- Values: NFS, NFSv4, Kerberos, KRB5, KRB5i, KRB5p, POSIX, ACLs

**"Guidelines for adding an SMB share" (28 entities):**
- Types: cli_command, network_or_storage_protocol, file_system_path
- Values: SMB, POSIX permissions, smb, bash, some/dir

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

## Known Issues and Warnings

### 1. OTEL/Jaeger Connection Errors (Cosmetic)

The ingestion worker logs show repeated connection errors to `jaeger:4318`. These are cosmetic and do not affect ingestion functionality. Filter them when monitoring:

```bash
docker logs -f weka-ingestion-worker 2>&1 | grep -v "Exception while exporting" | grep -v jaeger
```

### 2. GLiNER API Deprecation Warning

```
FutureWarning: GLiNER.batch_predict_entities is deprecated.
Please use GLiNER.inference instead.
```

**Status:** Noted for future migration before GLiNER 1.0 release.

### 3. Truncation Warnings

```
UserWarning: Sentence of length 385 has been truncated to 384
```

**Status:** Normal for long chunks. GLiNER has 384 token limit. Entities are still extracted from truncated text.

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

## Current System State

### What Works

1. **GLiNER entity extraction** - Fully functional with MPS acceleration (65x faster)
2. **Entity-enriched embeddings** - Dense vectors include entity context via `_embedding_text`
3. **Entity-sparse vectors** - Lexical matching on entity names (1.5x RRF weight)
4. **Post-retrieval entity boosting** - Query-time entity matching up to 50% boost
5. **STDIO entity exposure** - Agent can see `entity_metadata` and `entity_boost_applied`
6. **Full corpus ingestion** - 54 documents, 288 points, ~1,100+ entities

### What's Disabled (Intentionally)

1. **neo4j_disabled: true** - Graph queries bypassed during retrieval (ingestion still uses Neo4j)
2. **Graph channel** - Disabled when neo4j_disabled is true
3. **Microdoc stubs** - Disabled to prevent empty chunks
4. **doc_fallback** - Disabled to prevent cross-topic chunk pollution

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
docker logs -f weka-ingestion-worker 2>&1 | grep -E "(gliner|enrichment|ingestion_complete|Job done)"
```

### Check Entity Extraction Stats
```bash
docker logs weka-ingestion-worker 2>&1 | grep "gliner_enrichment_complete" | tail -10
```

---

## Next Steps

### Potential Future Work

1. **Retrieval quality testing** - A/B test entity-boosted vs non-boosted retrieval
2. **Tuning boost parameters** - Adjust `max_boost` and `per_entity_boost` values
3. **Label refinement** - Evaluate which entity types provide most retrieval value
4. **GPU deployment** - Consider GPU for production scale if CPU fallback is too slow
5. **Full production corpus** - Ingest complete WEKA documentation set

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

*Session context saved: 2025-12-09 02:10 UTC*
*Git commit: d0c0e75 on dense-graph-enhance branch*
*Status: GLiNER integration complete, full corpus ingested successfully*

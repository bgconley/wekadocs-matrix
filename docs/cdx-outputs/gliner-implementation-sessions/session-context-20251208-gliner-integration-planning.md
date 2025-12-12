# Session Context: GLiNER Integration Planning & Architecture Review

**Session Date:** 2025-12-08
**Branch:** `dense-graph-enhance`
**Session Focus:** GLiNER NER integration plan review, iterative refinement, and preparation for implementation
**Previous Context:** Vector pipeline pre-implementation fixes, configuration verification, database cleanup
**Context Document Version:** 1.0

---

## Executive Summary

This session focused on comprehensive review and iterative refinement of the GLiNER (Generalist Lightweight Named Entity Recognition) integration plan for the WekaDocs GraphRAG system. The plan evolved through 6 versions (v1.2 → v1.7), addressing critical architectural issues including text mutation, graph consistency, boosting mechanisms, and observability. The final v1.7 plan is approved for implementation.

**Key Accomplishments:**
1. Reviewed and refined GLiNER integration plan through 6 iterations
2. Identified and resolved 7 critical architectural issues
3. Added domain-specific WEKA entity labels (11 types)
4. Simplified document structure for implementation clarity
5. Documented complete infrastructure access patterns and credentials

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
- Comparing data between Neo4j and Qdrant for sync verification

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

**CRITICAL Shell Quoting Issue:** When using curl with JSON in bash, quotes can get stripped. For complex queries, write to a Python script file instead of inline bash.

**Payload Structure:** Each Qdrant point has a payload containing:
- `id` (SHA256 hash - matches Neo4j chunk ID)
- `node_id`, `kg_id` (same as id)
- `document_id`, `doc_id`, `doc_tag`
- `heading`, `text`, `token_count`
- `is_microdoc`, `doc_is_microdoc`, `is_microdoc_stub`
- `parent_section_id`, `original_section_ids`
- `boundaries_json`, `semantic_metadata`
- Embedding metadata (version, provider, timestamp)

---

### Redis Cache and Queue

Redis serves two purposes: L2 query result caching and RQ job queue for async ingestion.

**Connection Details:**
```
Host: localhost (from host) / redis (from Docker)
Port: 6379
Password: testredis123
Database: 0 (changed from 1)
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

**Example: Get dense embeddings:**
```python
import requests
response = requests.post(
    "http://127.0.0.1:9000/v1/embeddings",
    json={"input": ["text to embed"]}
)
embeddings = [e["embedding"] for e in response.json()["data"]]
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

## Docker Container Architecture and Management

### Container Architecture Overview

| Container | Purpose | Key Ports | Persistent Data | Mount Pattern |
|-----------|---------|-----------|-----------------|---------------|
| `weka-neo4j` | Graph database | 7687, 7474 | Yes (volume) | N/A |
| `weka-qdrant` | Vector store | 6333, 6334 | Yes (volume) | N/A |
| `weka-redis` | Cache + queue | 6379 | Yes (volume) | N/A |
| `weka-mcp-server` | HTTP MCP + STDIO | 8000 | No | `./src:/app/src:ro` |
| `weka-ingestion-worker` | RQ background jobs | None | No | `./src:/app/src:ro` |
| `weka-ingestion-service` | Ingestion HTTP API | 8081 | No | N/A |

### Two MCP Server Architecture (CRITICAL DISTINCTION)

| Server | File | Protocol | Primary Tool | Usage |
|--------|------|----------|--------------|-------|
| HTTP Server | `src/mcp_server/main.py` | HTTP REST on :8000 | `search_documentation` | LEGACY - human-facing answer synthesis |
| STDIO Server | `src/mcp_server/stdio_server.py` | stdin/stdout pipes | `search_sections` | PRODUCTION - Agent/Claude raw data access |

**STDIO Server Invocation:**
```bash
docker exec -i weka-mcp-server python -m src.mcp_server.stdio_server
```

### Volume Mounts (CRITICAL for Development)

The MCP server and ingestion worker mount source code as read-only volumes:
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

**This means:**
- Code changes are immediately visible inside containers
- **NO rebuild needed for code changes** - just restart
- Restart: `docker compose restart mcp-server` or `docker compose restart ingestion-worker`
- Rebuild only needed for Dockerfile or requirements.txt changes

### Essential Docker Commands

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Restart after code changes (NO rebuild needed)
docker compose restart mcp-server
docker compose restart ingestion-worker

# Full recreate (needed for docker-compose.yml env var changes)
docker compose up -d ingestion-worker ingestion-service

# View logs (filter out Jaeger noise which is cosmetic)
docker logs -f weka-ingestion-worker 2>&1 | grep -v jaeger

# Shell into container for debugging
docker exec -it weka-mcp-server bash

# Verify config loaded inside container
docker exec weka-mcp-server python3 -c "
from src.shared.config import get_config
config = get_config()
print('neo4j_disabled:', config.search.hybrid.neo4j_disabled)
"

# Verify assembler configuration
docker exec weka-ingestion-worker python3 -c "
from src.shared.config import get_config
from src.ingestion.chunk_assembler import get_chunk_assembler
config = get_config()
assembler = get_chunk_assembler(config.ingestion.chunk_assembly)
print(f'microdoc_enabled: {assembler.microdoc_enabled}')
print(f'doc_fallback_enabled: {assembler.doc_fallback_enabled}')
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
3. Creates: Neo4j nodes (Document, Section/Chunk, Entity) + Qdrant vectors (8 per chunk)
4. All 8 vector types generated per chunk via BGE-M3

### CRITICAL: Clean Ingestion After Rebuild or Testing

The worker tracks processed files by hash. Without cleanup, files won't re-ingest.

**Complete Clean Ingestion Procedure:**
```bash
# 1. Delete Qdrant points (preserve schema)
curl -s -X POST "http://127.0.0.1:6333/collections/chunks_multi_bge_m3/points/delete" \
  -H "Content-Type: application/json" -d '{"filter": {}}'

# 2. Clear Neo4j
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) DETACH DELETE n"

# 3. Clear Redis
docker exec weka-redis redis-cli -a testredis123 FLUSHALL

# 4. Remove processed files from ingest folder (if you want same files to re-ingest)
rm -f data/ingest/*.md

# 5. Re-add test files
cp test-documents/*.md data/ingest/
```

**Why this matters:** The ingestion worker maintains an internal registry of processed file hashes. If you want to re-ingest the same content, you must either:
- Delete the files and re-add them
- Clear all three stores completely
- Use the cleanup procedure above

---

## Configuration Architecture Discovery (From Previous Session)

### Critical Finding: Two Configuration Paths

The chunk assembler has two modes controlled by `CHUNK_ASSEMBLER` env var:

| Mode | Class | Config Source | When Used |
|------|-------|---------------|-----------|
| `structured` (default) | `StructuredChunker` | YAML config file | Production |
| `greedy` | `GreedyCombinerV2` | Environment variables | Fallback/testing |

**The Issue Previously Discovered:**
- Docker-compose sets `CHUNK_ASSEMBLER=structured`
- `StructuredChunker` passes `assembly_config` to parent constructor
- This sets `structured_mode = True`
- In structured_mode, env vars like `COMBINE_MICRODOC_ENABLED` are **completely ignored**
- Settings are read from `config/development.yaml` instead

**The Fix Applied:**
- Added explicit `microdoc.enabled: false` to `config/development.yaml`
- The env vars remain as backup for `greedy` mode

---

## GLiNER Integration Plan: Evolution and Final State

### Plan Version History

| Version | Focus | Status |
|---------|-------|--------|
| v1.2 | Initial architecture | 7 critical issues identified |
| v1.3 | Core issue fixes | Text mutation, graph consistency, boosting fixed |
| v1.4 | Observability + Migration | Metrics, tracing, migration strategy added |
| v1.5 | Code completeness | All code snippets complete |
| v1.6 | Labels + Performance | Domain-specific labels, Apple Silicon optimization |
| v1.7 | Structure simplification | **FINAL - Approved for implementation** |

### Critical Issues Resolved Through Review

| Issue | Problem | Resolution |
|-------|---------|------------|
| **Text Mutation** | `chunk["text"]` was being modified with entity context, polluting stored content | Use transient `_embedding_text` field for embedding generation only |
| **Graph Inconsistency** | GLiNER entities created `_mentions` entries but no Neo4j Entity nodes | Filter GLiNER entities from Neo4j creation via `source="gliner"` marker |
| **Schema Mismatch** | Plan used `label` field but existing schema uses `type` | Changed to `type` to match existing `_mentions` structure |
| **Boosting Ineffective** | Plan tried to use Qdrant `should` filters for boosting | Implemented post-retrieval Python rescoring instead |
| **Hardcoded Device** | Plan hardcoded `mps` device | Added auto-detection: MPS → CUDA → CPU fallback |
| **No Error Handling** | Model failures would crash ingestion | Added circuit breaker pattern with graceful degradation |
| **No Observability** | No metrics or tracing | Added Prometheus histograms/counters + OpenTelemetry spans |

### Final Architecture (v1.7)

**GLiNER Integration Points:**
1. **Phase 1: Core Infrastructure** - `src/providers/ner/gliner_service.py` (singleton with caching)
2. **Phase 2: Document Ingestion** - `src/ingestion/extract/ner_gliner.py` + hooks in `atomic.py`
3. **Phase 3: Qdrant Schema** - New payload indexes for entity metadata
4. **Phase 4: Hybrid Search** - Post-retrieval boosting in `hybrid_retrieval.py`
5. **Phase 5: Performance** - Apple Silicon optimization (MPS, batch size 32)

**Domain-Specific Entity Labels (11 types):**
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

## Files Modified (Previous Session - Still Relevant)

### `docker-compose.yml`
- **Lines 277-279:** Added `COMBINE_DOC_FALLBACK_ENABLED=false` and `COMBINE_MICRODOC_ENABLED=false` to ingestion-worker
- **Lines 379-381:** Added same env vars to ingestion-service
- **Purpose:** Disable problematic chunking features in greedy mode

### `config/development.yaml`
- **Lines 229-234:** Added microdoc configuration block with `enabled: false`
- **Purpose:** Disable microdoc stubs in structured mode (production)

### `src/ingestion/chunk_assembler.py`
- **Lines 173-175:** Changed from hardcoded `True` to env var read for `microdoc_enabled`
- **Purpose:** Allow runtime control via env var in greedy mode

### `src/ingestion/build_graph.py`
- **Lines 444-475:** Updated mentions attachment to check `original_section_ids` for combined chunks
- **Purpose:** Fix entity-sparse vectors for combined chunks

---

## Files To Be Created/Modified (GLiNER Implementation)

### New Files
- `src/providers/ner/__init__.py` - Package init
- `src/providers/ner/gliner_service.py` - GLiNER singleton service
- `src/providers/ner/labels.py` - Label configuration helper
- `src/ingestion/extract/ner_gliner.py` - Chunk enrichment function
- `src/query/processing/disambiguation.py` - Query entity extraction
- `tests/integration/test_gliner_flow.py` - Integration tests
- `tests/unit/test_atomic_ingestion.py` - Unit tests

### Files To Modify
- `src/shared/config.py` - Add `NERConfig` class
- `config/development.yaml` - Add `ner:` configuration block
- `src/ingestion/atomic.py` - Hook GLiNER enrichment + embedding consumption
- `src/shared/qdrant_schema.py` - Add entity payload indexes
- `src/query/hybrid_retrieval.py` - Add post-retrieval boosting
- `requirements.txt` - Add `gliner>=0.2.24`

---

## Current System State

### What Works
1. **Neo4j completely bypassed during retrieval** - `neo4j_disabled: true` in config
2. **6-Signal Weighted RRF Fusion** - All signals computed with configurable weights
3. **Entity-Sparse Score Visibility** - Fixed key mismatch ("entity-sparse" not "entity")
4. **ColBERT Vectors** - Stored in Qdrant (44-316 tokens × 1024 dims per chunk)
5. **BGE Cross-Encoder Reranking** - Service running at :9001
6. **STDIO Endpoint Signal Exposure** - All fusion fields visible to Agent
7. **Entity extraction** - Works correctly via atomic.py path
8. **Mentions attachment for combined chunks** - Fixed in build_graph.py

### What's Disabled (Intentionally)
1. **doc_fallback** - Prevents cross-topic chunk pollution
2. **microdoc_stub creation** - No more empty chunks
3. **Graph channel** - `graph_channel_enabled: false`
4. **Graph enrichment** - `graph_enrichment_enabled: false`
5. **BM25 retrieval** - `bm25.enabled: false`
6. **Expansion** - `expansion.enabled: false`

### Current Database State
All three databases are **empty** and ready for fresh ingestion after GLiNER implementation:
- Neo4j: 0 nodes
- Qdrant: 0 points (schema preserved)
- Redis: 0 keys

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

# API keys
export JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi
```

---

## Next Steps for Implementation

### Immediate (GLiNER Integration)
1. Add `gliner>=0.2.24` to requirements.txt
2. Create `src/providers/ner/` package with `gliner_service.py`
3. Add `NERConfig` to `src/shared/config.py`
4. Add `ner:` block to `config/development.yaml` with labels
5. Create `src/ingestion/extract/ner_gliner.py` with `enrich_chunks_with_entities()`
6. Modify `atomic.py` to call enrichment and consume `_embedding_text`
7. Add Neo4j MENTIONS filter for `source="gliner"`
8. Update Qdrant schema with entity payload indexes
9. Add post-retrieval boosting to `hybrid_retrieval.py`
10. Write tests

### Post-Implementation
1. Full corpus re-ingestion with GLiNER enabled
2. A/B testing of entity labels for extraction quality
3. Tune boosting factor based on retrieval metrics

---

## Key Learnings and Patterns

### Configuration Hierarchy
In structured_mode (production), YAML config takes precedence over environment variables. Changes to env vars in docker-compose.yml are ignored unless also in YAML.

### Defensive Coding Patterns
All consumers of `original_section_ids` and `_mentions` use defensive patterns:
- `section.get("original_section_ids", [])`
- `payload.get("_mentions") or []`

### GLiNER Zero-Shot Label Format
Including examples in parentheses `(e.g. ...)` is a valid technique for GLiNER. The model uses these hints to understand what kind of entities to extract.

### Apple Silicon Considerations
- Use `mps` device for Metal Performance Shaders
- Batch size 32 is optimal for M1/M2/M3 Max
- Avoid `bitsandbytes` - it's CUDA-optimized and fails on MPS

---

## Plan Document Location

**Final approved plan:** `/Users/brennanconley/vibecode/wekadocs-matrix/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md`

**Version:** 1.7 (Simplified Structure)

**Status:** ✅ APPROVED FOR IMPLEMENTATION

---

## Detailed GLiNER Architecture Decisions and Rationale

### Why Transient `_embedding_text` Instead of Modifying `chunk["text"]`

The original plan (v1.2) proposed directly modifying chunk text:
```python
# BAD - v1.2 approach
chunk["text"] = f"{chunk['text']}\n\n[Context: {entity_str}]"
```

This was rejected because:
1. **Storage pollution**: The modified text would be stored in both Neo4j (`Chunk.text` property) and Qdrant payload (`text` field)
2. **User-facing contamination**: Search results returned to users would include the synthetic `[Context: ...]` annotations
3. **MCP server output corruption**: The STDIO endpoint would return polluted text to Claude/agents
4. **Irreversibility**: Once stored, the original clean text would be lost

The approved approach uses a transient field:
```python
# GOOD - v1.7 approach
chunk["_embedding_text"] = f"{chunk['text']}\n\n[Context: {entity_str}]"
# chunk["text"] remains untouched
```

The embedding generation code then checks:
```python
content_text = section.get("_embedding_text") or builder._build_section_text_for_embedding(section)
```

This ensures:
- Dense vectors are enriched with entity context for better semantic matching
- Stored text remains clean and user-friendly
- The pattern is backwards-compatible (chunks without `_embedding_text` still work)

### Why Post-Retrieval Boosting Instead of Qdrant `should` Filters

The original plan attempted to use Qdrant's filter mechanism for entity boosting:
```python
# BAD - v1.2 approach
query_filter=Filter(should=[
    FieldCondition(key="entity_values", match=MatchValue(value="WEKA"))
])
```

This fails because Qdrant's `should` in filters works as a disjunction (OR) with `must` clauses, not as a score multiplier. Documents either match or don't—there's no "soft" boost.

The approved post-retrieval approach:
```python
# GOOD - v1.7 approach
for res in results:
    doc_entities = res.payload.get("entity_metadata", {}).get("entity_values_normalized", [])
    matches = sum(1 for term in boost_terms if term in doc_entities)
    if matches > 0:
        boost_factor = 1.0 + min(0.5, matches * 0.1)  # Max 50% boost
        res.score *= boost_factor
```

Benefits:
- Actual score modification (multiplicative boost)
- Configurable boost factors per entity match
- No zero-recall risk (all candidates retrieved first, then boosted)
- Debug visibility via `res.boosted = True` marker

### Why GLiNER Entities Are Filtered from Neo4j MENTIONS

The existing pipeline creates Entity nodes in Neo4j with MENTIONS relationships:
```
Document → Section → Chunk ←MENTIONS→ Entity (Neo4j node)
```

GLiNER entities are added to `_mentions` with a synthetic ID:
```python
"entity_id": f"gliner:{e.label}:{eid_hash}"
```

Without filtering, this would create MENTIONS edges pointing to non-existent Entity nodes (since we don't create the Entity nodes for GLiNER extractions). This was resolved by adding a source marker:
```python
new_mentions.append({
    "name": e.text,
    "type": e.label,
    "entity_id": f"gliner:{e.label}:{eid_hash}",
    "source": "gliner",  # <-- Marker for filtering
    "confidence": e.score
})
```

And filtering during Neo4j creation:
```python
def _neo4j_create_mentions(self, tx, mentions):
    mentions_to_create = [m for m in mentions if m.get("source") != "gliner"]
```

This keeps GLiNER entities in the sparse vector path (via `_mentions` → `entity-sparse`) without corrupting the graph.

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. GLiNER Model Fails to Load

**Symptom:** Logs show `Failed to load GLiNER model: ...`

**Causes and Fixes:**
- **HuggingFace rate limiting**: Wait and retry, or set `HF_TOKEN` env var
- **Network issues**: Check internet connectivity from container
- **MPS initialization failure**: Verify macOS version supports MPS, try `device: cpu` as fallback
- **Insufficient memory**: Reduce batch_size or use smaller model

**Debug command:**
```bash
docker exec weka-ingestion-worker python3 -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

#### 2. Entities Not Appearing in Search Results

**Symptom:** `entity_metadata` is null/empty in Qdrant payloads

**Checklist:**
1. Verify `ner.enabled: true` in `config/development.yaml`
2. Check ingestion logs for `GLiNER enrichment failed` warnings
3. Verify model loaded: look for `GLiNER model loaded successfully` in logs
4. Confirm `_embedding_text` is being generated (add debug logging if needed)

#### 3. Ingestion Times Out

**Symptom:** Workers timeout after 300 seconds

**Fix:** Increase timeout in config:
```yaml
ingestion:
  queue_recovery:
    job_timeout_seconds: 600  # Was 300
```

#### 4. Boosting Has No Effect

**Symptom:** Results order unchanged despite entity matches

**Debug steps:**
1. Verify query entities are being extracted (log `boost_terms`)
2. Check payload `entity_values_normalized` contains expected values
3. Confirm `ner.enabled: true` in retrieval config
4. Add debug logging to boost loop:
```python
if matches > 0:
    logger.debug(f"Boosting {res.id}: {matches} matches, factor={boost_factor}")
```

---

## Performance Benchmarks and Expectations

### Expected Latency Impact

| Operation | Without GLiNER | With GLiNER | Delta |
|-----------|---------------|-------------|-------|
| Single chunk ingestion | ~200ms | ~250ms | +50ms |
| Batch (32 chunks) | ~2s | ~3s | +1s |
| Query entity extraction | N/A | ~20ms | +20ms |
| Post-retrieval boosting | N/A | ~5ms | +5ms |

### Memory Requirements

| Component | Memory (M1 Max) |
|-----------|-----------------|
| GLiNER Medium v2.1 | ~400MB GPU/unified |
| GLiNER Large v2.5 | ~700MB GPU/unified |
| Per-batch inference | ~200MB temp |

**Total additional memory:** ~600MB for medium model, ~1GB for large model

### Recommended Hardware Configurations

| Configuration | Model | Batch Size | Expected Throughput |
|--------------|-------|------------|---------------------|
| M1 Max 32GB | Medium v2.1 | 32 | ~50 chunks/sec |
| M1 Pro 16GB | Medium v2.1 | 16 | ~25 chunks/sec |
| Intel Mac (CPU) | Medium v2.1 | 8 | ~10 chunks/sec |

---

## Testing Strategy for GLiNER Integration

### Unit Tests Required

1. **GLiNERService Singleton**
   - Verify only one instance created across multiple imports
   - Verify auto-device detection works correctly
   - Verify circuit breaker returns empty list on failure

2. **Entity Extraction**
   - Test extraction with known text produces expected entities
   - Test empty text returns empty list
   - Test batch extraction maintains correct ordering

3. **Chunk Enrichment**
   - Verify `_embedding_text` is set when entities found
   - Verify `text` field is NOT modified
   - Verify `_mentions` contains correct schema (`type` not `label`)
   - Verify `source: "gliner"` marker is present
   - Verify deduplication works (no duplicate entities)

4. **Neo4j Filtering**
   - Verify `source="gliner"` mentions are filtered out
   - Verify non-gliner mentions are NOT filtered

### Integration Tests Required

1. **End-to-End Ingestion**
   - Ingest test document with GLiNER enabled
   - Verify Qdrant payload contains `entity_metadata`
   - Verify Neo4j has no GLiNER Entity nodes
   - Verify `entity-sparse` vectors are populated

2. **Query Boosting**
   - Query with entity that appears in some documents
   - Verify boosted documents rank higher
   - Verify `boosted: true` marker is set

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **GLiNER** | Generalist Lightweight Named Entity Recognition - zero-shot NER model |
| **MPS** | Metal Performance Shaders - Apple Silicon GPU acceleration |
| **Transient field** | Field used during processing but not persisted to storage |
| **Circuit breaker** | Pattern that fails gracefully when dependencies unavailable |
| **Post-retrieval boosting** | Score modification after initial retrieval, before final ranking |
| **Entity-sparse** | Sparse vector generated from entity names for lexical matching |
| **`_mentions`** | Internal field containing entity references for a chunk |
| **`_embedding_text`** | Transient field with entity-enriched text for embedding generation |

---

## Related Documentation

- **GLiNER Implementation Plan:** `/docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md`
- **Previous Session Context:** `/docs/cdx-outputs/session-context-20251207-vector-architecture-reform.md` (if exists)
- **Configuration Reference:** `/config/development.yaml`
- **Ingestion Pipeline:** `/src/ingestion/atomic.py`, `/src/ingestion/build_graph.py`
- **Retrieval Pipeline:** `/src/query/hybrid_retrieval.py`

---

*Session context saved: 2025-12-08*
*Document token count: ~5000 tokens*
*Next session: Begin GLiNER implementation starting with Phase 1 (Core Infrastructure)*

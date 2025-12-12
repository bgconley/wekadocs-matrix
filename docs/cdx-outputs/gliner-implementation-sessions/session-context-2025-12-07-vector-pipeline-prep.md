# Session Context: Vector Pipeline Pre-Implementation Fixes

**Session Date:** 2025-12-07
**Branch:** `dense-graph-enhance`
**Session Focus:** Pre-implementation fixes for vector pipeline optimization, configuration verification, database cleanup
**Previous Context:** Neo4j bypass implementation, RRF fusion signal exposure, semantic chunking research

---

## Executive Summary

This session completed critical pre-implementation work for the vector pipeline optimization project:

1. **Disabled doc_fallback** - Prevents cross-topic chunk pollution
2. **Disabled microdoc stub creation** - Eliminates empty chunks invisible to vector search
3. **Fixed entity-sparse mentions bug** - Combined chunks now get proper entity embeddings
4. **Discovered and fixed config hierarchy issue** - Env vars were ignored in production mode
5. **Cleaned all three databases** - Ready for fresh ingestion

**Key Discovery:** The `CHUNK_ASSEMBLER=structured` setting in production causes `StructuredChunker` to read from YAML config, completely ignoring Docker environment variables. This required adding explicit config to `development.yaml`.

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
- `GET /health` - Health check, returns `{"status": "healthy"}`
- `POST /v1/embeddings` - Dense embeddings (1024-D vectors)
- `POST /v1/embeddings/sparse` - Sparse embeddings (BM25-style term weights)
- `POST /v1/embeddings/colbert` - ColBERT multi-vectors (token-level embeddings)

**Health Check:**
```bash
curl -s http://127.0.0.1:9000/health
# Returns: {"status": "healthy"}
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

## Configuration Architecture Discovery

### Critical Finding: Two Configuration Paths

The chunk assembler has two modes controlled by `CHUNK_ASSEMBLER` env var:

| Mode | Class | Config Source | When Used |
|------|-------|---------------|-----------|
| `structured` (default) | `StructuredChunker` | YAML config file | Production |
| `greedy` | `GreedyCombinerV2` | Environment variables | Fallback/testing |

**The Issue We Discovered:**
- Docker-compose sets `CHUNK_ASSEMBLER=structured`
- `StructuredChunker` passes `assembly_config` to parent constructor
- This sets `structured_mode = True`
- In structured_mode, env vars like `COMBINE_MICRODOC_ENABLED` are **completely ignored**
- Settings are read from `config/development.yaml` instead

**The Fix:**
- Added explicit `microdoc.enabled: false` to `config/development.yaml`
- The env vars remain as backup for `greedy` mode

---

## Architectural Issues Resolved This Session

### Issue #1: doc_fallback Cross-Topic Pollution

**Problem:** doc_fallback was merging unrelated H1 sections within the same source file when total tokens < threshold.

**Resolution:**
- In structured_mode: Already hardcoded to `False` in `GreedyCombinerV2.__init__` line 127
- In greedy_mode: Added `COMBINE_DOC_FALLBACK_ENABLED=false` to docker-compose.yml

**Status:** RESOLVED

---

### Issue #2: Microdoc Stubs Creating Empty Chunks

**Problem:** `_ensure_microdoc_stub()` was creating chunks with `text=""` for graph traversal. These are invisible to vector search (can't embed empty text).

**Resolution:**
- Added `microdoc.enabled: false` to `config/development.yaml` (structured_mode)
- Added `COMBINE_MICRODOC_ENABLED=false` to docker-compose.yml (greedy_mode)
- Added env var handler in `chunk_assembler.py:173-175`

**Status:** RESOLVED

---

### Issue #3: Entity-Sparse Vectors Empty for Combined Chunks

**Problem:** In `build_graph.py:448-454`, mentions were attached using only `section["id"]` (the NEW combined chunk ID). But mentions are keyed by ORIGINAL section IDs. Combined chunks have `original_section_ids` list but this was ignored.

**Resolution:** Updated `build_graph.py:444-475` to:
1. Check current section ID
2. Also iterate through `original_section_ids`
3. Deduplicate by entity_id
4. This mirrors the correct pattern from `atomic.py:539-558`

**Status:** RESOLVED

---

## Files Modified This Session

### `docker-compose.yml`
- **Lines 277-279:** Added `COMBINE_DOC_FALLBACK_ENABLED=false` and `COMBINE_MICRODOC_ENABLED=false` to ingestion-worker
- **Lines 379-381:** Added same env vars to ingestion-service
- **Purpose:** Disable problematic chunking features in greedy mode

### `config/development.yaml`
- **Lines 229-234:** Added microdoc configuration block:
```yaml
microdoc:
  enabled: false
  doc_token_threshold: 2000
  min_split_tokens: 400
```
- **Purpose:** Disable microdoc stubs in structured mode (production)

### `src/ingestion/chunk_assembler.py`
- **Lines 173-175:** Changed from hardcoded `True` to env var read:
```python
self.microdoc_enabled = (
    os.getenv("COMBINE_MICRODOC_ENABLED", "true").lower() == "true"
)
```
- **Purpose:** Allow runtime control via env var in greedy mode

### `src/ingestion/build_graph.py`
- **Lines 444-475:** Replaced simple mentions attachment with comprehensive version that checks `original_section_ids`:
```python
for section in sections:
    section_mentions = []
    section_id = section.get("id")
    if section_id and section_id in mentions_by_section:
        section_mentions.extend(mentions_by_section[section_id])
    # Check original section IDs (from chunk assembly - combined chunks)
    original_ids = section.get("original_section_ids", [])
    for orig_id in original_ids:
        if orig_id in mentions_by_section:
            section_mentions.extend(mentions_by_section[orig_id])
    # Deduplicate by entity_id
    seen_entity_ids = set()
    unique_mentions = []
    for m in section_mentions:
        eid = m.get("entity_id")
        if eid and eid not in seen_entity_ids:
            seen_entity_ids.add(eid)
            unique_mentions.append(m)
    section["_mentions"] = unique_mentions
```
- **Purpose:** Fix entity-sparse vectors for combined chunks

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
8. **Mentions attachment for combined chunks** - Now fixed in build_graph.py

### What's Disabled (Intentionally)

1. **doc_fallback** - Prevents cross-topic chunk pollution
2. **microdoc_stub creation** - No more empty chunks
3. **Graph channel** - `graph_channel_enabled: false`
4. **Graph enrichment** - `graph_enrichment_enabled: false`
5. **BM25 retrieval** - `bm25.enabled: false`
6. **Expansion** - `expansion.enabled: false`

### Current Database State

| Database | Status | Count |
|----------|--------|-------|
| Neo4j | **Empty** | 0 nodes |
| Qdrant | **Empty** | 0 points (schema preserved) |
| Redis | **Empty** | 0 keys |

---

## Implementation Plan: Vector Pipeline Optimization

### Immediate Priority (NOT YET STARTED)

| Task | Tool | Status | Time Est. |
|------|------|--------|-----------|
| Add semantic chunking module | BGE-M3 | NOT STARTED | 2 hours |
| Add TextRank keyphrase extraction | pytextrank/yake | NOT STARTED | 1 hour |
| Add keyphrase-sparse vector to schema | BGE-M3 sparse | NOT STARTED | 30 min |
| Update RRF weights for keyphrase-sparse | Config | NOT STARTED | 10 min |
| Add GLiNER NER integration | gliner | NOT STARTED | 1 hour |
| Context-enriched chunk text (prepend doc_title) | String ops | NOT STARTED | 30 min |

### Key Dependencies for New Features

```bash
# For Semantic Chunking
pip install nltk  # For sent_tokenize
# BGE-M3 service already running at http://127.0.0.1:9000

# For TextRank Keyphrases
pip install yake  # Recommended - no spaCy dependency

# For GLiNER NER
pip install gliner
# Model: urchade/gliner_medium-v2.5 (~400MB, downloads on first use)
```

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

## Key Codebase Discoveries

### Two Independent `microdoc_enabled` Variables

1. **Ingestion-time** (`chunk_assembler.py`): Controls whether empty stub chunks are created
   - Controlled by: YAML config `ingestion.chunk_assembly.microdoc.enabled` OR env var `COMBINE_MICRODOC_ENABLED`

2. **Retrieval-time** (`hybrid_retrieval.py:2198`): Controls neighbor stitching during queries
   - Controlled by: `MICRODOC_MAX_NEIGHBORS` env var (defaults to 2)
   - Completely independent - NOT affected by our changes

### Configuration Priority

In structured_mode (production):
- YAML config takes precedence over environment variables
- Changes to env vars in docker-compose.yml are ignored unless also in YAML

### Defensive Coding Patterns

All consumers of `original_section_ids` and `_mentions` use defensive patterns:
- `section.get("original_section_ids", [])`
- `payload.get("_mentions") or []`

This makes our changes backwards-compatible.

---

## Research Findings: Advanced Vector Pipeline Architecture (From Previous Sessions)

### Semantic Chunking with BGE-M3 (RECOMMENDED)

Traditional chunking splits text at fixed token counts or character boundaries. Semantic chunking uses embedding similarity to find natural topic boundaries.

**Algorithm:**
1. Split text into sentences using `nltk.sent_tokenize()`
2. Add context window (±1 sentence around each) for better embedding context
3. Generate dense embeddings via BGE-M3 `/v1/embeddings` endpoint
4. Calculate cosine distances between consecutive sentence embeddings
5. Find breakpoints where similarity drops below percentile threshold (80th recommended)
6. Create chunks between breakpoints

**Benchmark Results (Superlinked Evaluation on HotpotQA + SQUAD):**

| Method | Context Precision | Context Recall | Faithfulness | Relevancy | Latency |
|--------|------------------|----------------|--------------|-----------|---------|
| Embedding-similarity (BGE-M3) | 0.85 | 0.88 | 0.91 | **0.95** | 5.24s |
| Hierarchical-clustering | 0.82 | 0.84 | 0.63 | 0.89 | 5.21s |
| LLM-based | **0.89** | **0.92** | **0.94** | 0.93 | 6.88s |

**Key Insight:** Embedding-similarity achieves highest relevancy scores with reasonable latency. LLM-based excels at complex semantic relationships but adds 30% latency.

**Implementation Pattern:**
```python
def semantic_chunk_with_bge_m3(text: str, percentile: int = 80) -> list[str]:
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return [text]  # Too short for semantic analysis

    # Add context window
    combined = [" ".join(sentences[max(0,i-1):min(len(sentences),i+2)])
                for i in range(len(sentences))]

    # Get embeddings from BGE-M3 service
    embeddings = get_embeddings_bge_m3(combined)

    # Calculate cosine distances
    distances = [1 - cosine_similarity(embeddings[i], embeddings[i+1])
                 for i in range(len(embeddings)-1)]

    # Find breakpoints
    threshold = np.percentile(distances, percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    return create_chunks_from_breakpoints(sentences, breakpoints)
```

---

### GLiNER for Zero-Shot NER (RECOMMENDED over spaCy)

**spaCy Problems:**
- Fixed entity types (PERSON, ORG, LOC) - can't detect domain-specific entities
- Trained on news/web data - struggles with technical terminology like CLI commands, config parameters
- No zero-shot capability - needs fine-tuning for every new domain

**GLiNER (NAACL 2024) Advantages:**
- Zero-shot NER - define entity types at inference time
- Outperforms ChatGPT and fine-tuned LLMs on zero-shot benchmarks
- 400MB model (`urchade/gliner_medium-v2.5`), ~50ms per chunk, CPU-only
- Perfect for domain-specific entities

**Domain-Specific Entity Types for WekaDocs:**
```python
WEKA_ENTITY_LABELS = [
    "CLI command",           # weka cluster create, weka nfs add
    "configuration parameter",  # stripe_width, num_replicas
    "file path",                # /etc/weka/config.yaml
    "WEKA feature",          # tiering, snapshots, SMB
    "storage protocol",      # NFS, SMB, S3, POSIX
    "error code",            # WEKA-1234, ERR_TIMEOUT
    "version number",        # v4.2.1, 4.2.1.23
]

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.5")
entities = model.predict_entities(chunk_text, WEKA_ENTITY_LABELS, threshold=0.5)
```

---

### TextRank/YAKE for Keyphrase Extraction (RECOMMENDED)

Graph-based algorithm, no LLM needed, fast and domain-agnostic.

**YAKE is Recommended Because:**
1. No dependency on spaCy (avoiding spaCy NER limitations)
2. Unsupervised - works on any domain without training
3. Fast (~5ms per chunk)
4. Language-independent
5. No GPU required

```python
import yake

def extract_keyphrases(text: str, limit: int = 10) -> list[str]:
    kw_extractor = yake.KeywordExtractor(top=limit, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords]
```

---

## Proposed Vector-Only Architecture (Post-Enhancement)

```
Document
    │
    ▼
┌─────────────────────────────────────────────────┐
│  SEMANTIC CHUNKING (BGE-M3)                     │
│  ├── Sentence tokenization                      │
│  ├── Context window (±1 sentence)               │
│  ├── Dense embedding per sentence               │
│  ├── Cosine distance calculation                │
│  └── Percentile-based breakpoint detection      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  METADATA ENRICHMENT                            │
│  ├── TextRank/YAKE keyphrases ──► keyphrase-sparse │
│  ├── GLiNER entities ────────────► entity-sparse    │
│  └── Doc title prepend ──────────► Already have     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  MULTI-VECTOR EMBEDDING (BGE-M3)                │
│  ├── content (dense 1024-D)                     │
│  ├── title (dense 1024-D)                       │
│  ├── doc_title (dense 1024-D)                   │
│  ├── text-sparse                                │
│  ├── title-sparse (2.0× weight)                 │
│  ├── doc_title-sparse                           │
│  ├── entity-sparse (1.5× weight)                │
│  ├── keyphrase-sparse (NEW - 1.5× weight)       │
│  └── late-interaction (ColBERT)                 │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  QDRANT STORAGE                                 │
│  └── 9 vectors per point (adding keyphrase)     │
└─────────────────────────────────────────────────┘
```

---

## Retrieval Pipeline Flow (Current)

```
Query
  │
  ├── Qdrant 6-signal prefetch
  │   ├── content (dense)
  │   ├── title (dense)
  │   ├── text-sparse
  │   ├── title-sparse (2.0×)
  │   ├── doc_title-sparse
  │   └── entity-sparse (1.5×)
  │
  ├── Weighted RRF Fusion (k=60)
  │
  ├── ColBERT MaxSim Rerank
  │
  └── BGE Cross-Encoder Final Rerank
```

---

## Testing and Validation Approach

### Unit Tests for New Components

```python
# tests/test_semantic_chunking.py
def test_semantic_chunker_respects_topic_boundaries():
    text = """
    NFS is a network file system protocol. It allows remote access to files.

    SMB is a different protocol used by Windows. It provides file sharing.
    """
    chunks = semantic_chunk(text, percentile=80)
    assert len(chunks) >= 2  # Should split at topic boundary

def test_gliner_extracts_weka_entities():
    text = "Run 'weka cluster status' to check /etc/weka/config.yaml"
    entities = ner.extract(text)
    labels = [e['label'] for e in entities]
    assert 'CLI command' in labels
    assert 'file path' in labels

def test_keyphrase_extraction():
    text = "Configure NFS client mounts for optimal performance"
    keyphrases = extract_keyphrases(text)
    assert 'NFS' in keyphrases or 'client mounts' in keyphrases
```

### Integration Validation Script

```python
# Verify counts after clean ingestion
def verify_ingestion():
    # Neo4j
    neo4j_chunks = cypher("MATCH (c:Chunk) RETURN count(c)")

    # Qdrant
    qdrant_points = requests.get(
        "http://127.0.0.1:6333/collections/chunks_multi_bge_m3"
    ).json()["result"]["points_count"]

    # Must match
    assert neo4j_chunks == qdrant_points, f"Drift: {neo4j_chunks} vs {qdrant_points}"

    # No empty chunks
    empty = cypher("MATCH (c:Chunk) WHERE c.text = '' RETURN count(c)")
    assert empty == 0, f"Found {empty} empty chunks"
```

---

## Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GLiNER model download fails | Low | High | Pre-download model, cache in Docker volume |
| Semantic chunking creates too many tiny chunks | Medium | Medium | Implement min_chunk_tokens with merging |
| Keyphrase extraction slows ingestion | Low | Low | YAKE is fast (~5ms), parallelize if needed |
| New vector schema breaks existing queries | Medium | High | Feature flag, gradual rollout, backward compatibility |
| BGE-M3 service overloaded | Low | High | Already handles current load, monitor latency |

---

## Metadata Enrichment Weights (From Elastic RAG Research)

The Elastic Advanced RAG team uses this composite embedding approach:

| Field | Embedding Weight | Purpose |
|-------|-----------------|---------|
| Original text | 70% | Main semantic content |
| Keyphrases | 15% | Exact match search, filtering |
| Potential questions | 10% | Match user query phrasing directly |
| Entities | 5% | Cross-document linking, exact match |

**Recommendation:** Start with keyphrase-sparse at 1.5× RRF weight (matching entity-sparse). Tune based on A/B testing with real queries.

---

## Deferred Work (Graph-Related)

These items are intentionally deferred per user decision to focus on vector-only optimization:

| Item | Reason | Future Phase |
|------|--------|--------------|
| Neo4j graph enrichment | User wants vector-first | Phase 2 |
| Structure-aware expansion | Graph-dependent | Phase 2 |
| Cross-document linking via graph | Graph-dependent | Phase 2 |
| MENTIONS relationship optimization | Graph-dependent | Phase 2 |

---

## Session Artifacts

- **Branch:** `dense-graph-enhance`
- **Research sources consulted:** Microsoft Azure RAG docs, Firecrawl 2025 benchmarks, Elastic Advanced RAG, Superlinked semantic chunking guide, RAKG paper (arXiv 2504.09823)
- **Models evaluated:** BGE-M3, GLiNER, NuNER, Flair TARS, spaCy, TextRank
- **Architecture patterns:** Composite embeddings, semantic chunking, zero-shot NER

---

*Document created: 2025-12-07*
*Session duration: Approximately 2 hours*
*Token count target: 4500-5500 tokens*
*Primary accomplishments: Fixed configuration hierarchy issue, disabled problematic chunking features, fixed entity-sparse mentions bug, cleaned databases for fresh ingestion*
*Next session: Begin implementation of semantic chunking module with BGE-M3*

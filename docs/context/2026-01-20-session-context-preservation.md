# Session Context Preservation Document

**Date:** 2026-01-20
**Session Duration:** Extended debugging and design session
**Branch:** `multi-embedder-reranker`
**Primary Focus:** Fixing retrieval pipeline bugs, multi-embedder query-time support, diagtool design

---

## Executive Summary

This session addressed critical bugs in the WekaDocs MCP retrieval pipeline that were causing:
1. Claude to hallucinate about CORS support (claiming S3 CORS APIs exist when they don't)
2. Graph traversal tools failing with type errors
3. Sparse embeddings not being used at query time despite being configured

We discovered that the multi-embedder infrastructure **already existed** but had bugs preventing it from working. We fixed these bugs and then designed a comprehensive diagnostic tool (`diagtool`) for future troubleshooting.

---

## Bugs Found and Fixed

### Bug #1: Circuit Breaker Logging - Reserved LogRecord Field

**File:** `src/shared/resilience/circuit_breaker.py`

**Problem:** The circuit breaker was using `"name"` as a key in the `extra` dict passed to structlog. `name` is a reserved attribute of Python's `LogRecord` class, causing:
```
KeyError("Attempt to overwrite 'name' in LogRecord")
```

**Impact:** This caused the reranker to fail silently, returning unranked results. Dense-only results meant generic S3 docs ranked above specific CORS docs (which don't exist anyway).

**Fix:** Changed all 6 instances of `"name"` to `"circuit_name"`:
- Line 171: `"name": name,` → `"circuit_name": name,`
- Line 217: `"name": self.name,` → `"circuit_name": self.name,`
- Line 245: `"name": self.name,` → `"circuit_name": self.name,`
- Line 273: `"name": self.name,` → `"circuit_name": self.name,`
- Line 284: `"name": self.name,` → `"circuit_name": self.name,`
- Line 305: `"name": self.name,` → `"circuit_name": self.name,`

**Status:** ✅ Fixed and deployed

---

### Bug #2: Anti-Hallucination Instructions Missing

**File:** `src/mcp_server/mcp_app.py`

**Problem:** Claude was inferring S3 CORS API support based on finding `cors-trusted-sites` documentation (a WEKA-specific security feature), then fabricating API names like `GetBucketCors`, `PutBucketCors`, `DeleteBucketCors`.

**Root Cause:** The MCP server instructions didn't tell Claude to only claim support for explicitly documented features.

**Fix:** Added to both `GRAPH_FIRST_INSTRUCTIONS` and `VECTOR_ONLY_INSTRUCTIONS` (lines 954-956 and 964-966):
```python
"CRITICAL: Only state that a feature, API, or capability is supported if it is EXPLICITLY listed in the documentation. "
"Do NOT infer or assume support based on related concepts or similar terminology. "
"If something is not explicitly documented as supported, clearly state that you could not find documentation for it rather than guessing or fabricating details."
```

**Status:** ✅ Fixed and deployed

---

### Bug #3: MCP Tool Type Coercion - Graph Tools Failing

**File:** `src/mcp_server/mcp_app.py`

**Problem:** Claude Desktop passes all MCP tool arguments as **JSON strings**, not native Python types:
- `node_ids` sent as `"[\"abc123\"]"` instead of `["abc123"]`
- `max_hops` sent as `"1"` instead of `1`

When Python tried to compare `min("1", 3)`, it failed with:
```
'<' not supported between instances of 'int' and 'str'
```

**Affected Functions (all fixed):**
1. `expand_neighbors` (line 2362-2371) - Added coercion for `node_ids`, `max_hops`, `page_size`, `include_snippet`
2. `get_paths_between` (line 2419-2432) - Added coercion for `a_ids`, `b_ids`, `rel_types`, `max_hops`, `max_paths`
3. `describe_nodes` (line 2321-2327) - Added coercion for `node_ids`, `fields`
4. `list_children` (line 2457-2459) - Added coercion for `page_size`
5. `list_parents` (line 2490-2493) - Added coercion for `section_ids`
6. `get_entities_for_sections` (line 2552-2560) - Added coercion for `section_ids`, `labels`, `max_per_section`
7. `get_sections_for_entities` (line 2598-2603) - Added coercion for `entity_ids`, `max_per`

**Fix Pattern:**
```python
# Coerce types - MCP clients may send strings instead of proper types
if isinstance(node_ids, str):
    import json as _json
    node_ids = _json.loads(node_ids)
if isinstance(max_hops, str):
    max_hops = int(max_hops)
```

**Status:** ✅ Fixed and deployed

---

### Bug #4: Multi-Embedder Not Used at Query Time (THE BIG ONE)

**File:** `src/query/hybrid_retrieval.py`

**Problem:** The embedding plan was correctly configured for multi-embedder:
```yaml
plan:
  dense: "snowflake_arctic_v2l"    # Snowflake Arctic (dense only)
  sparse: "bge_m3"                  # BGE-M3 (supports sparse)
  colbert: "bge_m3"                 # BGE-M3 (supports ColBERT)
  enable_sparse: true
  enable_colbert: true
```

And the `MultiVectorRetriever.__init__` correctly created separate embedders:
- `self.embedder` = SnowflakeArcticProvider (for dense)
- `self.sparse_embedder` = BGEM3ServiceProvider (for sparse)
- `self.colbert_embedder` = BGEM3ServiceProvider (for ColBERT)

**But** the `_build_sparse_query` method was using the **wrong embedder**:

```python
# BEFORE (Bug at lines 1058 and 1061):
def _build_sparse_query(self, query: str):
    if not self.supports_sparse or not hasattr(self.embedder, "embed_sparse"):  # WRONG!
        return None
    sparse_vectors = self.embedder.embed_sparse([query])  # WRONG!

# AFTER (Fixed):
def _build_sparse_query(self, query: str):
    if not self.supports_sparse or not hasattr(self.sparse_embedder, "embed_sparse"):  # CORRECT
        return None
    sparse_vectors = self.sparse_embedder.embed_sparse([query])  # CORRECT
```

**Also fixed the capability checks at lines 869-874:**
```python
# BEFORE:
self.supports_sparse = (
    hasattr(self.embedder, "embed_sparse") and self.schema_supports_sparse
)
self.supports_colbert = (
    hasattr(self.embedder, "embed_colbert") and self.schema_supports_colbert
)

# AFTER:
self.supports_sparse = (
    hasattr(self.sparse_embedder, "embed_sparse") and self.schema_supports_sparse
)
self.supports_colbert = (
    hasattr(self.colbert_embedder, "embed_colbert") and self.schema_supports_colbert
)
```

**Status:** ✅ Fixed and deployed

**Verification:**
```bash
docker exec weka-mcp-server python3 -c "
from src.mcp_server.query_service import QueryService
qs = QueryService()
retriever = qs._get_7e_retriever()
sparse_query = retriever.vector_retriever._build_sparse_query('CORS configuration')
print(f'Sparse query has {len(sparse_query.get(\"indices\", []))} indices!')
"
# Output: ✅ Sparse query has 3 indices!
```

---

## Architecture Understanding Gained

### Multi-Embedder Query Pipeline

```
Query → HybridRetriever
         │
         ├── embedding_plan (from get_embedding_plan())
         │   ├── dense: EmbeddingRolePlan(profile_name="snowflake_arctic_v2l")
         │   ├── sparse: EmbeddingRolePlan(profile_name="bge_m3")
         │   └── colbert: EmbeddingRolePlan(profile_name="bge_m3")
         │
         └── vector_retriever (QdrantMultiVectorRetriever)
             ├── embedder: SnowflakeArcticProvider (dense queries)
             ├── sparse_embedder: BGEM3ServiceProvider (sparse queries)
             └── colbert_embedder: BGEM3ServiceProvider (ColBERT queries)
```

### Key Files and Their Roles

| File | Purpose |
|------|---------|
| `src/shared/config.py` | `EmbeddingPlan`, `EmbeddingRolePlan`, `get_embedding_plan()` |
| `src/providers/factory.py` | `ProviderFactory.create_embedding_provider_for_role()` |
| `src/providers/embeddings/snowflake_arctic.py` | Dense-only embedder (raises NotImplementedError for sparse) |
| `src/providers/embeddings/bge_m3_service.py` | Multi-head embedder (dense + sparse + ColBERT) |
| `src/query/hybrid_retrieval.py` | `QdrantMultiVectorRetriever`, `HybridRetriever` |
| `src/mcp_server/query_service.py` | `QueryService._get_7e_retriever()` |
| `src/mcp_server/mcp_app.py` | MCP tool definitions, instructions |
| `src/shared/resilience/circuit_breaker.py` | Circuit breaker for external services |

### Qdrant Collection Schema

Collection: `chunks_multi_snowflake_arctic_v2l`

**Dense vectors:**
- `content`: 1024 dims (main chunk embedding)
- `title`: 1024 dims (section title)
- `doc_title`: 1024 dims (document title)
- `late-interaction`: Variable dims (ColBERT multi-vector)

**Sparse vectors:**
- `text-sparse`: Lexical matching for chunk text
- `title-sparse`: Lexical matching for section titles
- `doc_title-sparse`: Lexical matching for document titles
- `entity-sparse`: Lexical matching for entity names

### Service Endpoints

| Service | Port | Purpose |
|---------|------|---------|
| MCP Server | 8000 | Main MCP endpoint (inside Docker) |
| BGE-M3 | 9000 | Sparse + ColBERT embeddings |
| Reranker | 9005 | Qwen3-Reranker-0.6B (currently shut down for RAM) |
| Snowflake Arctic | 9010 | Dense embeddings via Chonkie |
| Neo4j | 7687 | Graph database |
| Qdrant | 6333 | Vector database |
| Redis | 6379 | Cache |

---

## Diagnostic Infrastructure

### Retrieval Diagnostics

**Location:** `reports/retrieval_diagnostics/{date}/`

**Files:**
- `retrieval_diagnostics.jsonl` - Complete query data (JSON Lines format)
- `{diagnostic-id}.md` - Human-readable summary per query

**Key fields in JSONL:**
```json
{
  "diagnostic_id": "uuid",
  "timestamp": "ISO8601",
  "session_id": "server-xxx",
  "query": {
    "raw": "original query",
    "rewritten": {"applied": true, "result": "rewritten query", "reason": "keyword_stuffed"}
  },
  "timing_ms": {"bm25": 0, "vector_search": 105, "fusion": 0.02, "rerank": 5, "graph_expansion": 55},
  "results": [
    {
      "rank": 1,
      "chunk_id": "xxx",
      "source": "dense|sparse|colbert|graph_expanded|bm25",
      "scores": {"dense": 0.72, "sparse": 0.31, "rerank": 0.92, "fused": 0.015}
    }
  ]
}
```

### MCP Log

**Location:** `~/Library/Logs/Claude/mcp-server-wekadocs-matrix.log`

**Format:** JSON-RPC messages with `Message from client` and `Message from server` prefixes

### Existing Scripts

- `scripts/retrieval_diagnostics/show.py` - Lookup single diagnostic by ID
- `scripts/diagnose_missing_chunks.py` - Debug missing chunks
- `scripts/generate_preflight_report.py` - Pre-flight checks

---

## diagtool Design (Complete)

We designed a comprehensive CLI diagnostic tool. Full design document at:
`docs/plans/2026-01-20-diagtool-design.md`

### Commands Summary

| Command | Purpose |
|---------|---------|
| `diagtool replay` | Session replay - all tool calls chronologically |
| `diagtool query` | Single query deep-dive with full provenance |
| `diagtool stats` | Aggregate statistics and pattern analysis |
| `diagtool watch` | Live monitoring of MCP traffic |

### Key Design Decisions

1. **CLI with Typer + Rich** - Beautiful terminal output, scriptable
2. **Session numbering** - Human-readable `#1`, `#2` instead of UUIDs
3. **Provenance tracking** - Show source (dense/sparse/colbert/graph), pre/post rerank positions
4. **Expandable detail** - Default compact, `--detail` for provenance, `--full` for JSON
5. **Text excerpts in watch** - See actual returned content, not just metadata
6. **Semantic colors** - Green=success, Red=error, Blue=info, Yellow=waiting

### Implementation Notes

- Single file: `scripts/diagtool.py`
- Data sources: JSONL diagnostics + MCP log
- Session discovery: Scan JSONL for unique session_ids, group and number them

---

## Outstanding Work

### Immediate (Not Committed)

The following changes were made but not yet committed:

1. **Circuit breaker fix** - `src/shared/resilience/circuit_breaker.py`
2. **Anti-hallucination instructions** - `src/mcp_server/mcp_app.py`
3. **Type coercion fixes** - `src/mcp_server/mcp_app.py`
4. **Multi-embedder fix** - `src/query/hybrid_retrieval.py`
5. **diagtool design doc** - `docs/plans/2026-01-20-diagtool-design.md`

### To Implement

1. **diagtool CLI** - Full implementation per design doc
2. **Test CORS query** - Verify anti-hallucination instructions work
3. **Reranker** - Currently shut down for RAM, needs to be restarted for full pipeline

---

## Environment Notes

### Docker Containers

```
weka-mcp-server   - MCP server (healthy)
weka-neo4j        - Neo4j graph DB
weka-qdrant       - Qdrant vector DB
weka-redis        - Redis cache
```

### Key Environment Variables (in container)

```
BGE_M3_API_URL=http://host.docker.internal:9000
CHONKIE_EMBEDDINGS_BASE_URL=http://host.docker.internal:9010/v1
EMBEDDING_NAMESPACE_MODE=profile
EMBEDDINGS_PROFILE=bge_m3 (ignored due to embedding plan)
```

### Config Files

- `config/development.yaml` - Main config
- `config/embedding_profiles.yaml` - Embedding profiles and plan

---

## Verification Commands

### Check Multi-Embedder Status

```bash
docker exec weka-mcp-server python3 -c "
from src.shared.config import get_embedding_plan
plan = get_embedding_plan()
print(f'Dense: {plan.dense.profile_name} ({plan.dense.profile.provider})')
print(f'Sparse: {plan.sparse.profile_name if plan.sparse else None}')
print(f'ColBERT: {plan.colbert.profile_name if plan.colbert else None}')
"
```

### Check Sparse Embedding Works

```bash
docker exec weka-mcp-server python3 -c "
from src.mcp_server.query_service import QueryService
qs = QueryService()
retriever = qs._get_7e_retriever()
result = retriever.vector_retriever._build_sparse_query('test query')
print(f'Sparse indices: {len(result.get(\"indices\", [])) if result else 0}')
"
```

### Check Services Health

```bash
curl -s http://localhost:9000/healthz   # BGE-M3
curl -s http://localhost:9005/healthz   # Reranker (if running)
curl -s http://localhost:9010/health    # Snowflake Arctic
```

---

## Key Insights for Next Session

1. **The reranker is shut down** - User mentioned RAM issues. Pipeline works without it but quality is reduced.

2. **CORS hallucination root cause** - Not just retrieval, but Claude inferring from related content. The anti-hallucination instructions should help but need testing.

3. **Multi-embedder was always supposed to work** - The infrastructure existed, just had bugs. Now sparse and ColBERT are active at query time.

4. **diagtool is designed but not implemented** - Full design approved, ready for implementation.

5. **Git status is dirty** - Multiple uncommitted changes across several files. Should commit bug fixes before starting diagtool implementation.

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/shared/resilience/circuit_breaker.py` | "name" → "circuit_name" (6 instances) |
| `src/mcp_server/mcp_app.py` | Anti-hallucination instructions, type coercion for 7 functions |
| `src/query/hybrid_retrieval.py` | Fixed `_build_sparse_query` and capability checks to use correct embedders |
| `docs/plans/2026-01-20-diagtool-design.md` | NEW - Complete diagtool design |

---

## Restart Instructions

1. Read this document to restore context
2. Check git status for uncommitted changes
3. Decide whether to commit bug fixes first or continue with diagtool
4. If implementing diagtool, use the design doc as specification
5. If testing CORS fix, restart Claude Desktop and ask about S3 CORS support

---

*End of context preservation document*

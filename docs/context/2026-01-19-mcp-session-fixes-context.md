# MCP Server Session Fix - Complete Context Preservation Document

**Date**: 2026-01-19
**Session Duration**: Extended debugging and implementation session
**Primary Objective**: Fix `kb_read_excerpt` "Unknown passage_id" errors in WekaDocs Matrix MCP Server

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [System Architecture Understanding](#system-architecture-understanding)
4. [Root Cause Analysis](#root-cause-analysis)
5. [All Code Changes Made](#all-code-changes-made)
6. [ID Format Documentation](#id-format-documentation)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Testing Results](#testing-results)
9. [Remaining Issues](#remaining-issues)
10. [Next Steps](#next-steps)
11. [Key File Locations](#key-file-locations)
12. [Codebase Research Findings](#codebase-research-findings)

---

## Executive Summary

This session addressed critical issues with the WekaDocs Matrix MCP server where `kb_read_excerpt` and `get_section_text` calls were failing with "Unknown passage_id" errors. The investigation revealed **multiple layers of issues**:

1. **Session ID instability**: The MCP SDK's `request_context.session` object changes between tool calls, causing session mismatch
2. **Missing fallback paths**: When scratch storage lookup failed, there was no fallback to fetch from Neo4j directly
3. **AI ID fabrication**: The AI model (Claude in Claude Desktop) was fabricating UUIDs instead of using the actual IDs returned by search tools
4. **Uninformative response messages**: The text response from `kb_search` only said "returned N results" without listing the actual passage_ids

All four issues were addressed with code changes that have been deployed and verified in the Docker container.

---

## Problem Statement

### Symptoms Observed

1. User calls `kb_search` - returns results with passage_ids
2. User calls `kb_read_excerpt` with a passage_id - **FAILS** with "Unknown passage_id"
3. User calls `get_section_text` with section_ids - returns **0 results**

### Example Error Messages

```
kb_read_excerpt error: Unknown passage_id '5444519942e14c2db12eac800f3dc5d6' for this session.
get_section_text returned 0 results.
```

### Key Observation

The passage_id used by the AI (`5444519942e14c2db12eac800f3dc5d6`) was **completely different** from what `kb_search` actually returned (`3bd8ce2e4d59447885627dbe5cc3377a`). The AI was fabricating IDs.

---

## System Architecture Understanding

### Data Storage Architecture

| Component | What It Stores | ID Format | Purpose |
|-----------|---------------|-----------|---------|
| **Neo4j** | Chunk nodes with text, graph relationships | Long hash: `44a70b90961c632133ee1c30f5bbd55a5922cbf8884702bef1f5aac29db7aeb4_chunk_3_b1b751c610940693` | Source of truth for text and graph |
| **Qdrant** | Vector embeddings + text payload | Same chunk ID as Neo4j (in `payload.id`) | Semantic search |
| **Scratch Storage** | kb_search results (temporary, 30min TTL) | Random UUID: `3bd8ce2e4d59447885627dbe5cc3377a` | Session-scoped cache for excerpt reading |

### Key Insight: Both Neo4j and Qdrant Have Same Text

Qdrant is not just vectors - it stores full text in `payload.text`:

```json
{
  "id": "0010b72e-894a-5604-874f-e5085adcdf7c",
  "payload": {
    "id": "44a70b90..._chunk_3_b1b751c610940693",
    "text": "To download the TLS certificate, use the CLI command...",
    "heading": "Download the TLS certificate",
    "doc_tag": "security",
    "token_count": 28
    // ... 50+ other fields
  }
}
```

### MCP Server Configuration

The MCP server runs via Docker, configured in Claude Desktop:

```json
{
  "command": "docker",
  "args": ["docker", "exec", "-i", "weka-mcp-server", "python", "-X", "utf8", "-u", "-m", "src.mcp_server.stdio_server"]
}
```

**Important**: Local source files ARE mounted into the container:
- `/Users/brennanconley/vibecode/wekadocs-matrix/src` -> `/app/src`

This means code changes to local files are immediately visible in the container (no rebuild needed for Python changes).

### Tool Relationships

```
┌─────────────────┐     passage_id      ┌──────────────────┐
│   kb_search     │ ──────────────────► │  kb_read_excerpt │
│ (stores in      │                     │  (reads from     │
│  scratch)       │                     │   scratch)       │
└─────────────────┘                     └──────────────────┘

┌─────────────────┐     section_id      ┌──────────────────┐
│ search_sections │ ──────────────────► │ get_section_text │
│ (NO scratch)    │                     │  (reads from     │
│                 │                     │   Neo4j directly)│
└─────────────────┘                     └──────────────────┘
```

**Critical Difference**:
- `kb_search` stores full text in scratch, returns `passage_id` (UUID)
- `search_sections` does NOT store in scratch, returns `section_id` (chunk ID)

---

## Root Cause Analysis

### Layer 1: Session ID Instability

**Location**: `src/mcp_server/mcp_app.py:141-160`

**Original Problem**: The `_resolve_session_id` function used a `WeakKeyDictionary` keyed by `request_context.session` object. However, the MCP SDK creates different session objects for each tool call, causing mismatches.

**Original Code**:
```python
_SESSION_IDS: "WeakKeyDictionary[Any, str]" = WeakKeyDictionary()

def _resolve_session_id(ctx: Any | None, provided: Optional[str]) -> str:
    if provided:
        return provided
    request_context = _get_request_context(ctx)
    session = request_context.session
    existing = _SESSION_IDS.get(session)  # FAILS - different object each time!
    if not existing:
        existing = f"server-{uuid4()}"
        _SESSION_IDS[session] = existing
    return existing
```

**Fix**: Use a single global session ID:
```python
_GLOBAL_SESSION_ID: Optional[str] = None

def _resolve_session_id(ctx: Any | None, provided: Optional[str]) -> str:
    global _GLOBAL_SESSION_ID
    if provided:
        return provided
    if _GLOBAL_SESSION_ID is None:
        _GLOBAL_SESSION_ID = f"server-{uuid4()}"
    return _GLOBAL_SESSION_ID
```

### Layer 2: Missing Scratch Fallback

**Location**: `src/mcp_server/scratch_store.py:87-115`

**Problem**: If passage_id lookup failed, there was no way to find the entry by section_id.

**Fix**: Added `find_by_section_id` method:
```python
async def find_by_section_id(
    self, session_id: str, section_id: str
) -> Optional[Dict[str, Any]]:
    """Fallback lookup: find entry by section_id within a session."""
    now = time.time()
    async with self._lock:
        self._evict_expired_locked(now)
        for key, entry in self._entries.items():
            if key[0] != session_id:
                continue
            entry_section_id = entry.payload.get("section_id", "")
            if entry_section_id == section_id:
                return dict(entry.payload)
            # Also check partial match
            if section_id and len(section_id) >= 8:
                if entry_section_id.startswith(section_id) or entry_section_id.endswith(section_id):
                    return dict(entry.payload)
        return None
```

### Layer 3: Missing Neo4j Direct Fetch Fallback

**Location**: `src/mcp_server/mcp_app.py:1703-1727` and `1820-1844`

**Problem**: When AI uses `search_sections` (which doesn't populate scratch) and then calls `kb_read_excerpt`, there's no content in scratch.

**Fix**: Added Neo4j direct fetch as fallback:
```python
# Fallback 2: if still not found, try fetching directly from Neo4j via TextService
fetched_from_neo4j = False
if not entry and deps.text:
    try:
        budget = _new_budget()
        text_result = deps.text.get_section_text(
            section_ids=[passage_id],
            max_bytes_per=MAX_TEXT_BYTES_PER_CALL,
            budget=budget,
        )
        if text_result.results and len(text_result.results) > 0:
            neo4j_entry = text_result.results[0]
            if neo4j_entry.get("text"):
                entry = {
                    "text": neo4j_entry.get("text", ""),
                    "section_id": passage_id,
                    "doc_tag": neo4j_entry.get("doc_tag", ""),
                    "heading": neo4j_entry.get("heading", ""),
                }
                fetched_from_neo4j = True
    except Exception as e:
        logger.warning(f"kb_read_excerpt: Neo4j fallback failed: {e}")
```

### Layer 4: AI Fabricating IDs (Prompt Engineering Issue)

**Location**: `src/mcp_server/mcp_app.py:2940-2965` (`_summary_for_tool` function)

**Problem**: The text response only said `"kb_search returned 2 results."` without listing the actual passage_ids. The AI couldn't "see" which IDs to use and fabricated its own.

**Fix**: Enhanced `_summary_for_tool` to explicitly list IDs:
```python
def _summary_for_tool(name: str, result: dict) -> str:
    if isinstance(result, dict) and "error" in result:
        message = result.get("error", {}).get("message", "error")
        return f"{name} error: {message}"
    if "results" in result and isinstance(result["results"], list):
        results = result["results"]
        count = len(results)
        # For kb_search, explicitly list passage_ids
        if name == "kb_search" and count > 0:
            passage_ids = [r.get("passage_id") for r in results if r.get("passage_id")]
            if passage_ids:
                ids_str = ", ".join(passage_ids[:5])
                return (
                    f"{name} returned {count} results. "
                    f"Use these passage_ids with kb_read_excerpt: [{ids_str}]"
                )
        # For search_sections, explicitly list section_ids
        if name == "search_sections" and count > 0:
            section_ids = [r.get("section_id") for r in results if r.get("section_id")]
            if section_ids:
                ids_str = ", ".join(section_ids[:3])
                return (
                    f"{name} returned {count} results. "
                    f"Use these section_ids with get_section_text: [{ids_str[:150]}...]"
                )
        return f"{name} returned {count} results."
    # ... other cases
```

---

## All Code Changes Made

### File 1: `src/mcp_server/mcp_app.py`

| Line Range | Change Type | Description |
|------------|-------------|-------------|
| 71 | Comment | Changed `_SESSION_IDS` to comment explaining removal |
| 141-160 | Replacement | Replaced per-session tracking with global `_GLOBAL_SESSION_ID` |
| 1699-1727 | Addition | Added Neo4j fallback in `kb_read_excerpt` |
| 1703-1708 | Modification | Updated error message to mention both passage_id and section_id |
| 1816-1844 | Addition | Added Neo4j fallback in `kb_expand_excerpt` |
| 2940-2965 | Modification | Enhanced `_summary_for_tool` to list actual IDs |

### File 2: `src/mcp_server/scratch_store.py`

| Line Range | Change Type | Description |
|------------|-------------|-------------|
| 87-115 | Addition | New `find_by_section_id` method for fallback lookup |

---

## ID Format Documentation

### Passage ID (Generated by kb_search)

- **Format**: 32-character hex UUID
- **Example**: `3bd8ce2e4d59447885627dbe5cc3377a`
- **Generated by**: `uuid4().hex` in kb_search
- **Stored in**: Scratch storage with 30-minute TTL
- **Used with**: `kb_read_excerpt`, `kb_expand_excerpt`

### Section ID / Chunk ID (From Neo4j/Qdrant)

- **Format**: `{document_hash}_chunk_{index}_{content_hash}`
- **Example**: `44a70b90961c632133ee1c30f5bbd55a5922cbf8884702bef1f5aac29db7aeb4_chunk_3_b1b751c610940693`
- **Components**:
  - Document hash: 64 chars
  - Literal `_chunk_`
  - Chunk index: variable
  - Content hash: 16 chars
- **Used with**: `get_section_text`, Neo4j direct queries

### FABRICATED IDs (What AI Was Creating)

- **Short fabricated**: `sec_55d7f1f7` (NOT VALID)
- **UUID fabricated**: `5444519942e14c2db12eac800f3dc5d6` (NOT from kb_search)

---

## Data Flow Diagrams

### Correct Flow: kb_search -> kb_read_excerpt

```
1. User calls kb_search(query="S3 primitives")
   │
   ▼
2. Server generates passage_id = uuid4().hex = "abc123..."
   │
   ▼
3. Server fetches full text from QueryService (Neo4j + Qdrant hybrid)
   │
   ▼
4. Server stores in scratch: scratch.put(session_id, passage_id, {text, section_id, ...})
   │
   ▼
5. Server returns: {passage_id: "abc123...", section_id: "44a70b90..._chunk_3_...", preview: "..."}
   │
   ▼
6. NEW: Text message says: "kb_search returned 2 results. Use these passage_ids: [abc123...]"
   │
   ▼
7. User calls kb_read_excerpt(passage_id="abc123...")
   │
   ▼
8. Server looks up: scratch.get(session_id, "abc123...") -> Returns full text
```

### Fallback Flow: search_sections -> kb_read_excerpt (NEW)

```
1. User calls search_sections(query="S3 primitives")
   │
   ▼
2. Server queries Neo4j/Qdrant, returns section_ids (NO scratch storage)
   │
   ▼
3. User calls kb_read_excerpt(passage_id="44a70b90..._chunk_3_...")
   │
   ▼
4. Scratch lookup: FAILS (not stored)
   │
   ▼
5. Section ID fallback: FAILS (not in scratch)
   │
   ▼
6. NEW: Neo4j direct fetch: TextService.get_section_text([section_id])
   │
   ▼
7. Returns text from Neo4j directly
```

---

## Testing Results

### Test 1: Initial State (Before Fixes)

**Query**: "What S3 operations does WEKA support?"
**Result**:
- `kb_search` returned results
- `kb_read_excerpt` failed with "Unknown passage_id"
- Session IDs were different between calls

### Test 2: After Global Session ID Fix

**Result**: Session IDs now consistent, but still failing because AI was using wrong IDs

### Test 3: After All Fixes

**Query**: "Please use wekadocs matrix to help me understand what primitives are supported by the WEKA s3 protocol"

**Result**:
- Good answer returned (user confirmed quality was good)
- Still seeing some errors:
  - `kb_read_excerpt` with fabricated UUID
  - `get_section_text` with fabricated `sec_xxxx` IDs

### Current Status

The backend fixes are complete. The remaining issue is the AI model still sometimes fabricating IDs instead of using the ones in the response. The enhanced text messages should help, but needs testing.

---

## Remaining Issues

1. **AI ID Fabrication**: Even with explicit ID listing in responses, need to verify AI uses them correctly
2. **Qdrant Fallback**: Not implemented - could add as additional redundancy layer
3. **Tool Description Enhancement**: Could further improve tool descriptions to emphasize ID usage

---

## Next Steps

1. **Test Enhanced Messages**: Restart Claude Desktop and verify AI uses correct IDs from new explicit messages
2. **Monitor Logs**: Check if passage_ids in requests match those returned by kb_search
3. **Consider Qdrant Fallback**: If Neo4j fallback insufficient, add Qdrant as additional layer
4. **Tool Description Updates**: If AI still fabricates IDs, enhance tool input_schema descriptions

---

## Key File Locations

| File | Purpose |
|------|---------|
| `src/mcp_server/mcp_app.py` | Main MCP server with all tool implementations |
| `src/mcp_server/scratch_store.py` | Scratch storage for temporary session data |
| `src/mcp_server/query_service.py` | QueryService for Neo4j/Qdrant hybrid search |
| `src/services/text_service.py` | TextService for direct Neo4j text fetch |
| `docker-compose.yml` | Docker configuration for all services |
| `~/Library/Logs/Claude/mcp-server-wekadocs-matrix.log` | MCP server logs |
| `~/Library/Application Support/Claude/claude_desktop_config.json` | Claude Desktop MCP config |

---

## Codebase Research Findings

### Deps Class Structure (`mcp_app.py:861-920`)

```python
class Deps:
    query: Optional[QueryService] = None      # Hybrid search
    graph: Optional[GraphService] = None      # Graph traversal
    text: Optional[TextService] = None        # Direct text fetch
    summarizer: Optional[SummarizationService] = None
    assembler: Optional[ContextAssemblerService] = None
    scratch: Optional[ScratchStore] = None    # Session-scoped cache
```

### TextService Implementation (`text_service.py:38-65`)

Fetches from Neo4j `Chunk` nodes:
```python
query = """
MATCH (s:Chunk)
WHERE s.id IN $section_ids
RETURN s.id AS id,
       'Chunk' AS label,
       coalesce(s.title, s.name, s.heading, '') AS title,
       coalesce(s.text, s.content, '') AS text,
       s.doc_tag AS doc_tag
"""
```

### Qdrant Collections

Multiple collections exist:
- `chunks_multi_snowflake_arctic_v2l` (current primary)
- `chunks_multi_bge_m3`
- `chunks_multi`
- `chunks_multi_voyage_context_3`

### kb_search Result Structure (`mcp_app.py:473-506`)

```python
result = {
    "passage_id": passage_id,           # UUID for scratch lookup
    "section_id": chunk.chunk_id,       # Neo4j chunk ID
    "doc_tag": chunk.doc_tag,
    "title": chunk.heading,
    "rank": offset + idx + 1,
    "score": float(score),
    "source": source,                   # "reranked", "rrf_fusion", etc.
    "preview": preview,
    "scratch_uri": ScratchStore.build_uri(effective_session, passage_id),
    "size_bytes": size_bytes,
    "anchor": getattr(chunk, "anchor", None),
}
```

### search_sections Result Structure (`mcp_app.py:2065-2118`)

```python
results.append({
    "section_id": chunk.chunk_id,       # Only section_id, NO passage_id
    "title": chunk.heading,
    "tokens": chunk.token_count,
    "doc_tag": chunk.doc_tag,
    "score": float(score),
    "rank": offset + idx + 1,
    "source": source,
    "fusion_method": chunk.fusion_method,
    # ... many more score fields
})
```

Note: `search_sections` is "lightweight" - it does NOT store in scratch and does NOT generate passage_ids.

---

## Commands for Verification

```bash
# Check container status
docker ps --filter "name=weka-mcp-server"

# Check logs
tail -100 ~/Library/Logs/Claude/mcp-server-wekadocs-matrix.log

# Verify code in container
docker exec weka-mcp-server cat /app/src/mcp_server/mcp_app.py | grep "_GLOBAL_SESSION_ID"

# Restart container
docker compose restart mcp-server

# Check Neo4j chunk IDs
docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (c:Chunk) RETURN c.id LIMIT 3"

# Check Qdrant payload
curl -s "http://localhost:6333/collections/chunks_multi_snowflake_arctic_v2l/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "with_payload": true, "with_vector": false}'
```

---

## Session Metadata

- **Branch**: `multi-embedder-reranker`
- **Container**: `weka-mcp-server` (healthy)
- **Neo4j**: `weka-neo4j` (healthy)
- **Qdrant**: `weka-qdrant` (healthy)
- **Redis**: `weka-redis` (healthy)

---

*End of Context Preservation Document*

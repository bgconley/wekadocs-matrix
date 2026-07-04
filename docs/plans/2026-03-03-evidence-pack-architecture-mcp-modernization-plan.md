# Evidence Pack Architecture + MCP Modernization Plan

## Context

The retrieval pipeline (signal-diverse rerank pool, 8K cross-encoder, parent_path_norm enrichment) produces rich `ChunkResult` objects with ~25 score fields — then the MCP layer throws them away. `kb_retrieve_evidence` retrieves only 5 passages, stores 6 fields in ScratchStore (dropping all scores), and re-scores spans with keyword overlap. The `confidence` field in evidence quotes has no relationship to `rerank_score` or `fused_score`.

Meanwhile, the MCP tool surface exposes 24 tools (7 exact duplicates), graph-first instructions push multi-hop tool planning, legacy REST is default-on while streamable is default-off, and tool names in code (underscore) don't match docs/contract tests (dot notation).

**Goal:** Make `kb.retrieve_evidence` the first-class, server-orchestrated evidence path that fully uses the modern retrieval stack. Clean up the tool surface, add tool profiles, fix naming, and replace the instructions.

**Scope boundary — active path only:**
- BUILD ON: Streamable MCP + STDIO via `build_mcp_server()`, `kb_*` tools, `graph_*` prefixed tools, `search_sections_light()` → `HybridRetriever.retrieve()`
- DO NOT TOUCH: Legacy REST (`/mcp/*`), `search_documentation`, `traverse_relationships`, bare-name graph duplicates (remove only), `qwen3_triton.py`/`qwen3_embedding_client.py` (legacy providers)

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|---|---|---|
| Tool naming | Dot notation (`kb.search`, `graph.expand`) | Matches contract tests + api-contracts.md; temporary underscore aliases for backward compat |
| Escalation model | Server-side, always-on graph enrichment | Graph context is always valuable, not just a low-coverage fallback |
| Tool profiles | Config-driven via `MCP_TOOL_PROFILE` env var | Simple, deployment-level control |
| Production tools | 3: `kb.retrieve_evidence`, `kb.read_excerpt`, `graph.expand` | Answer, drill-down, follow-up navigation |
| Query handling | Always-on LLM reformulation with dual-query strategy | Qwen2.5-1.5B-Instruct for reformulation; original for BM25/sparse, reformulated for dense/reranker |
| Instruction prefixes | Domain-tuned for embedding + reranker | Qwen3-Embedding + Qwen3-Reranker both support and benefit from task-specific instructions |

---

## Implementation Phases

### Phase 1: Evidence Pack Data Flow Fix

The core problem: ChunkResult scores are lost at two points in `mcp_app.py`.

**Loss point 1 — ScratchStore write (`_kb_search_candidates`, line 464-474):**
Currently stores only `section_id`, `doc_tag`, `title`, `text`, `source_uri`, `created_at`. All score fields dropped.

**Loss point 2 — Evidence extraction (`_extract_evidence_from_passages`, line 578-601):**
Reads from scratch, re-scores with keyword overlap. `confidence` = token hit rate.

#### 1a. Enrich ScratchStore payload

**File: `src/mcp_server/mcp_app.py`** — `_kb_search_candidates()`, lines 464-474

Add retrieval scores to the scratch payload:

```python
scratch_payload = {
    # Existing fields
    "section_id": chunk.chunk_id,
    "doc_tag": chunk.doc_tag,
    "title": chunk.heading,
    "text": chunk.text,
    "source_uri": getattr(chunk, "source_path", None),
    "created_at": datetime.utcnow().isoformat() + "Z",
    # NEW: retrieval scores
    "rerank_score": chunk.rerank_score,
    "fused_score": chunk.fused_score,
    "vector_score": chunk.vector_score,
    "bm25_score": chunk.bm25_score,
    "graph_score": getattr(chunk, "graph_score", None),
    "parent_path_norm": chunk.parent_path_norm,
    "rerank_rank": chunk.rerank_rank,
    "fusion_method": chunk.fusion_method,
    "is_expanded": chunk.is_expanded,
    "expansion_source": chunk.expansion_source,
}
```

ScratchStore byte budget is 256MB with LRU eviction — adding ~200 bytes per entry for score fields is negligible.

#### 1b. Decouple retrieval depth from answer size

**File: `src/mcp_server/mcp_app.py`** — `kb_retrieve_evidence()`, lines 1979-2059

Currently `top_k` controls both retrieval depth AND passage count. Separate them:

- Add `internal_fetch_k` parameter (default 60, max 150) — how deep to search
- Keep `max_quotes` (default 6, max 12) — how many evidence quotes to return
- `top_k` becomes the external-facing parameter that maps to `max_quotes` (backward compat)

The call to `_kb_search_candidates()` uses `fetch_k=internal_fetch_k` (not `top_k`). Evidence extraction selects the best `max_quotes` from the larger candidate set.

**Constants:**
```python
KB_EVIDENCE_INTERNAL_FETCH_K = 60     # How deep to search internally
KB_EVIDENCE_MAX_FETCH_K = 150         # Hard cap on internal retrieval
```

#### 1c. Replace keyword overlap with retrieval-score-based evidence ranking

**File: `src/mcp_server/mcp_app.py`** — `_extract_evidence_from_passages()`, lines 563-637

Currently: reads text from scratch → splits into spans → scores by keyword overlap → sorts by overlap score.

Replace with a two-stage scoring approach:

**Stage 1 — Passage-level ranking by retrieval score:**
Read `rerank_score` (or `fused_score` fallback) from the enriched scratch payload. Sort passages by retrieval score descending. This determines which passages contribute quotes.

**Stage 2 — Span-level selection within top passages:**
For the top passages (by retrieval score), extract the most relevant spans. Use a blended score:
```python
span_score = (retrieval_weight * passage_retrieval_score) + (lexical_weight * keyword_overlap)
```

Default weights: `retrieval_weight=0.7`, `lexical_weight=0.3`. This preserves some keyword sensitivity (useful for highlighting the specific terms the user asked about) while primarily trusting the cross-encoder's judgment.

**The `confidence` field in output quotes becomes the retrieval-based score**, not keyword overlap. This is a breaking semantic change — document it in the output schema description.

#### 1d. Enrich evidence pack response

**File: `src/mcp_server/mcp_app.py`** — `_extract_evidence_from_passages()` return schema

Current quote:
```json
{"quote": "...", "passage_id": "...", "section_id": "...", "title": "...", "uri": "...", "confidence": 0.85}
```

Enriched quote:
```json
{
  "quote": "...",
  "passage_id": "...",
  "section_id": "...",
  "doc_tag": "weka_docs/4.3/admin/s3",
  "title": "Bucket Settings",
  "parent_path": "Configuration > S3 Backend > Bucket Settings",
  "uri": "wekadocs://scratch/<session>/<pid>",
  "confidence": 0.92,
  "source": "reranked",
  "rank": 1
}
```

New fields: `doc_tag`, `parent_path`, `source` (signal provenance), `rank` (retrieval rank).

Enriched outer payload:
```json
{
  "quotes": [...],
  "coverage": {
    "documents_searched": 45,
    "documents_with_evidence": 3,
    "retrieval_depth": 60,
    "reranker_applied": true,
    "signal_pool_active": false,
    "graph_expansion_applied": true
  },
  "partial": false,
  "limit_reason": "none",
  "session_id": "...",
  "meta": { "usage": {...} },
  "diagnostic_id": "..."
}
```

The `coverage` block gives the LLM (and operators) visibility into what the retrieval pipeline actually did. No gating decisions — purely informational.

---

### Phase 2: Always-On Graph Enrichment

Graph expansion is always valuable — it provides structural context that vector search cannot.

#### 2a. Always hydrate parent_path_norm for evidence candidates

**Already done.** `_hydrate_parent_paths()` runs unconditionally in `hybrid_retrieval.py:3366` for all reranker calls. The evidence pack just needs to read it from the enriched scratch payload (Phase 1a).

#### 2b. Structural expansion in the evidence pack

**File: `src/mcp_server/mcp_app.py`** — `kb_retrieve_evidence()`

After `_kb_search_candidates()` returns, before evidence extraction:

1. Take the top 10 passage `section_id` values from the search results
2. Run a lightweight graph query to fetch structural neighbors:
   - NEXT_CHUNK (±1 hop) for sequential context
   - Sibling chunks (same `parent_section_id`) for section completeness
3. Add these structural neighbors to the scratch store
4. Include them in the candidate pool for evidence extraction

This reuses the same structural expansion the retriever does, but ensures the evidence pack layer has access to it even for candidates that entered via post-rerank expansion (which currently gets synthetic scores).

**Cypher query (batch, single round-trip):**
```cypher
UNWIND $ids AS sid
MATCH (c:Chunk {id: sid})
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
OPTIONAL MATCH (sib:Chunk {parent_section_id: c.parent_section_id})
  WHERE sib.id <> c.id
RETURN c.id AS source_id,
       collect(DISTINCT next.id) AS next_ids,
       collect(DISTINCT prev.id) AS prev_ids,
       collect(DISTINCT sib.id)[..3] AS sibling_ids
```

Skip if `neo4j_disabled`. Cost: ~5-15ms for 10 seeds.

#### 2c. Evidence extraction incorporates structural context

When extracting evidence spans, structural neighbors are included in the candidate pool with their retrieval scores. The blended scoring (Phase 1c) naturally ranks them — a NEXT_CHUNK neighbor with a good rerank_score will surface; one with a poor score won't. No special-casing needed.

The `coverage` metadata includes `graph_expansion_applied: true` to indicate structural context was included.

---

### Phase 3: Query Reformulation + Instruction Prefixes

#### 3a. Deploy Qwen2.5-1.5B-Instruct for query reformulation

**New model on the unified gateway at `10.25.0.50:8080`:**

| Model | Params | VRAM (FP16) | Endpoint | Purpose |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | `POST /v1/chat/completions` | Query reformulation |

Current GPU usage: 12,875 MiB / 24,576 MiB (52%). Adding 3 GB → ~65% utilization, 8 GB headroom remaining.

**Why 1.5B, not 0.5B:** The 0.5B model is inconsistent at following the "rewrite only, don't answer" constraint — it tends to hallucinate extra content. 1.5B follows instructions reliably for this simple task.

**Why same-family Qwen:** Stack consistency. All models (embedding, reranker, reformulator) are Qwen family, served through one gateway.

**System prompt:**
```
You are a query reformulator for a technical documentation search system about WEKA (a distributed file system).
Rewrite the user's input as a clear, natural language question. Output ONLY the rewritten question, nothing else.
If the input is already a well-formed question, return it unchanged.
```

**Latency:** ~50-80ms for 30-token output on 3090. Runs **in parallel with BM25/sparse search** so it doesn't add to the critical path — by the time the reformulated query is ready, BM25 results are already back.

#### 3b. LLM-based reformulation in the retrieval pipeline

**File: `src/mcp_server/query_service.py`** — `_rewrite_keyword_query()`, line 486

Replace the heuristic template (`f"Explain {query}. How does this work and what is the technical architecture?"`) with an LLM call to Qwen2.5-1.5B-Instruct:

```python
async def _rewrite_keyword_query(self, query: str) -> Tuple[str, bool]:
    """Reformulate keyword-stuffed queries using LLM."""
    if self._is_well_formed_query(query):
        return query, False

    reformulated = await self._llm_reformulate(query)
    return reformulated, True
```

Keep the existing well-formed detection logic (question words, function-word ratio, etc.) as a fast-path bypass — well-formed queries skip the LLM call entirely.

**Fallback:** If the LLM call fails (timeout, gateway down), fall back to the heuristic templates as a degraded path. Log the fallback with `query_rewrite_method: "heuristic_fallback"`.

**New heuristic fallback templates** (replacing the single generic one):
```python
REWRITE_TEMPLATES = {
    "config":    "How do I configure {query} in WEKA?",
    "error":     "How do I troubleshoot {query} in WEKA?",
    "procedure": "What are the steps to {query} in WEKA?",
    "default":   "Explain {query} in the context of WEKA documentation.",
}
```

Intent detection via keyword signals for template selection.

**Diagnostics:** Add to metrics: `query_rewrite_method` ("llm" | "heuristic_fallback" | "passthrough"), `query_rewrite_latency_ms`, `query_rewrite_model`.

#### 3c. Dual-query strategy

**File: `src/mcp_server/query_service.py`** — `search_sections_light()`, line 302

Currently: one query string (rewritten) → all signals.

Change to: return both forms to the retriever.

```python
original_query = query
query, was_rewritten = await self._rewrite_keyword_query(query)
# Pass both to retriever
chunks, metrics = self._get_7e_retriever().retrieve(
    query=query,                    # reformulated → dense, colbert, reranker
    query_original=original_query,  # raw → bm25, sparse
    top_k=fetch_k,
    filters=filters or {},
    expand=expand,
)
```

**File: `src/query/hybrid_retrieval.py`** — `retrieve()` method

Add `query_original: Optional[str] = None` parameter. When provided:
- BM25 search uses `query_original` (keywords work better for lexical)
- SPLADE/sparse search uses `query_original` (term-level matching)
- Dense embedding uses `query` (reformulated natural language)
- ColBERT uses `query` (token-level semantic matching)
- Reranker uses `query` (cross-attention works best with questions)

When `query_original is None`, all signals use `query` (backward compat).

#### 3d. Qwen3-Embedding instruction prefix

**File: `src/providers/embeddings/embedding_service.py`** — `embed_query()`, lines 192-197

Currently adds BGE-M3 instruction prefix: `"Represent this sentence for searching relevant passages: "`. This should be profile-aware.

Qwen3-Embedding-0.6B supports and benefits from task-specific instruction prefixes. The format:
```
Instruct: Given a technical query about WEKA distributed file system, retrieve relevant documentation passages covering configuration, administration, troubleshooting, and CLI commands
Query: {query}
```

This is **asymmetric** — only queries get the instruction, documents are embedded as-is. The existing `embed_query()` / `embed_documents()` split already supports this pattern.

**Implementation:** Add `query_instruction` field to the embedding profile YAML schema:

```yaml
# config/embedding_profiles.yaml
qwen3_0_6b:
  provider: "embedding-service"
  model_id: "Qwen/Qwen3-Embedding-0.6B"
  query_instruction: "Instruct: Given a technical query about WEKA distributed file system, retrieve relevant documentation passages covering configuration, administration, troubleshooting, and CLI commands\nQuery: "
  dims: 1024
  ...
```

`EmbeddingServiceProvider.embed_query()` reads the instruction from the profile and prepends it:
```python
instruction = self._profile.get("query_instruction", "")
return self.embed_documents([instruction + text])[0]
```

For backward compat, BGE-M3 profiles keep their existing instruction. Profiles without `query_instruction` use empty string (no prefix).

#### 3e. Qwen3-Reranker domain-tuned instruction

**File: `src/providers/rerank/local_reranker_service.py`**

Qwen3-Reranker-4B supports custom instructions via its chat template. The default is generic ("Judge whether the document is relevant to the search query"). A domain-tuned instruction directly addresses the intro-vs-depth problem:

```
Judge whether the document is relevant to the search query about WEKA distributed file system.
Consider configuration procedures, CLI commands, parameter references, and step-by-step instructions as highly relevant.
Introductory overviews and general summaries are less relevant than specific technical procedures and settings.
Answer only "yes" or "no".
```

**Implementation depends on gateway support:**

1. **If the gateway `/v1/rerank` endpoint supports an `instruction` field:** Add it to the reranker request payload. This is the cleanest path.

2. **If the gateway doesn't support `instruction`:** Prepend the instruction to the query string: `f"{instruction}\n\nQuery: {query}"`. The Qwen3-Reranker model will interpret the combined text correctly since it uses the same tokenizer.

3. **If gateway requires server-side config:** Configure the instruction as a model parameter in the gateway deployment config (outside this codebase).

**Verification needed:** Test the unified gateway's `/v1/rerank` endpoint to determine which approach it supports. Add a `reranker_instruction` field to `RerankerConfig` in `config.py` so it's configurable per deployment.

**Impact:** This is the cheapest improvement in the plan — a string change that tells the 4B-parameter cross-encoder what "relevant" means for WEKA docs. Zero latency, zero VRAM, directly attacks the intro-dominance problem.

---

### Phase 4: Tool Surface Modernization

#### 4a. Dot notation migration

**File: `src/mcp_server/mcp_app.py`** — `_tool_specs()`, lines 2800-3014

Rename all tool registrations from underscore to dot:

| Current | New | Backward Alias |
|---|---|---|
| `kb_search` | `kb.search` | `kb_search` (temporary) |
| `kb_read_excerpt` | `kb.read_excerpt` | `kb_read_excerpt` (temporary) |
| `kb_expand_excerpt` | `kb.expand_excerpt` | `kb_expand_excerpt` (temporary) |
| `kb_extract_evidence` | `kb.extract_evidence` | `kb_extract_evidence` (temporary) |
| `kb_retrieve_evidence` | `kb.retrieve_evidence` | `kb_retrieve_evidence` (temporary) |
| `graph_describe` | `graph.describe` | `graph_describe` (temporary) |
| `graph_expand` | `graph.expand` | `graph_expand` (temporary) |
| `graph_paths` | `graph.paths` | `graph_paths` (temporary) |
| `graph_parents` | `graph.parents` | `graph_parents` (temporary) |
| `graph_children` | `graph.children` | `graph_children` (temporary) |
| `graph_entities_for_sections` | `graph.entities_for_sections` | `graph_entities_for_sections` (temporary) |
| `graph_sections_for_entities` | `graph.sections_for_entities` | `graph_sections_for_entities` (temporary) |
| `search_sections` | `kb.search_sections` | `search_sections` (temporary) |
| `get_section_text` | `kb.get_section_text` | `get_section_text` (temporary) |
| `summarize_neighborhood` | `graph.summarize` | `summarize_neighborhood` (temporary) |
| `compute_context_bundle` | `graph.context_bundle` | `compute_context_bundle` (temporary) |

Backward aliases: register both dot and underscore names in `_tool_specs()`, with the underscore versions flagged as deprecated in their descriptions. Both point to the same handler. Remove aliases after one release cycle.

#### 4b. Remove 7 duplicate bare-name graph tools

**File: `src/mcp_server/mcp_app.py`** — `_tool_specs()`, lines 2902-2967

Remove these exact-duplicate registrations (they're redundant with `graph.*` prefixed versions):

- `describe_nodes` (duplicate of `graph.describe`)
- `expand_neighbors` (duplicate of `graph.expand`)
- `get_paths_between` (duplicate of `graph.paths`)
- `list_parents` (duplicate of `graph.parents`)
- `list_children` (duplicate of `graph.children`)
- `get_entities_for_sections` (duplicate of `graph.entities_for_sections`)
- `get_sections_for_entities` (duplicate of `graph.sections_for_entities`)

This reduces the active tool count from 24 to 17 unique tools (before profile filtering).

#### 4c. Tool profiles

**File: `src/mcp_server/mcp_app.py`** — new `_filter_tools_by_profile()` function

Add env var: `MCP_TOOL_PROFILE` with values `"production"`, `"analyst"`, `"full"` (default: `"production"`).

Profile definitions:

```python
TOOL_PROFILES = {
    "production": {
        "kb.retrieve_evidence",
        "kb.read_excerpt",
        "graph.expand",
    },
    "analyst": {
        # All kb.* tools
        "kb.search", "kb.read_excerpt", "kb.expand_excerpt",
        "kb.extract_evidence", "kb.retrieve_evidence",
        "kb.search_sections", "kb.get_section_text",
        # All graph.* tools
        "graph.describe", "graph.expand", "graph.paths",
        "graph.parents", "graph.children",
        "graph.entities_for_sections", "graph.sections_for_entities",
        "graph.summarize", "graph.context_bundle",
    },
    "full": None,  # No filtering — all registered tools (includes backward aliases)
}
```

Filter application point — in `build_mcp_server()`, after `_tool_specs()`:

```python
profile = os.getenv("MCP_TOOL_PROFILE", "production")
allowed = TOOL_PROFILES.get(profile)
if allowed is not None:
    tool_specs = [s for s in tool_specs if s["name"] in allowed]
```

Log the active profile and tool count at startup.

#### 4d. Instruction replacement

**File: `src/mcp_server/mcp_app.py`** — lines 950-977

Replace `GRAPH_FIRST_INSTRUCTIONS` and `VECTOR_ONLY_INSTRUCTIONS` with profile-aware instructions:

```python
PRODUCTION_INSTRUCTIONS = (
    "You are connected to the WEKA documentation knowledge base. "
    "Start with kb.retrieve_evidence to get an evidence pack for any question. "
    "The evidence pack includes quotes, document context, and confidence metadata. "
    "Use kb.read_excerpt to read the full text of a passage if you need more context. "
    "Use graph.expand only for follow-up navigation (e.g., 'what else is in that section?'). "
    "Do NOT use graph.expand for initial research — kb.retrieve_evidence handles that. "
    "CRITICAL: Only state that a feature, API, or capability is supported if it is "
    "EXPLICITLY listed in the documentation. If something is not explicitly documented, "
    "clearly state that you could not find documentation for it."
)

ANALYST_INSTRUCTIONS = (
    "You are connected to the WEKA documentation knowledge base with full tool access. "
    "For most queries, start with kb.retrieve_evidence for an evidence pack. "
    "Use kb.search for browsing, graph.* tools for structural exploration, "
    "and kb.get_section_text for full text retrieval. "
    "CRITICAL: Only state that a feature, API, or capability is supported if it is "
    "EXPLICITLY listed in the documentation."
)
```

Select based on `MCP_TOOL_PROFILE`:
```python
_profile = os.getenv("MCP_TOOL_PROFILE", "production")
_instructions = PRODUCTION_INSTRUCTIONS if _profile == "production" else ANALYST_INSTRUCTIONS
```

---

### Phase 5: Transport Policy

#### 5a. Flip transport defaults

**File: `src/mcp_server/main.py`** — lines 66-75

```python
MCP_HTTP_STREAMABLE_ENABLED = os.getenv("MCP_HTTP_STREAMABLE_ENABLED", "true")   # was "false"
MCP_HTTP_LEGACY_REST_ENABLED = os.getenv("MCP_HTTP_LEGACY_REST_ENABLED", "false") # was "true"
```

This makes streamable MCP the default and legacy REST opt-in. Existing deployments that explicitly set env vars are unaffected.

#### 5b. Log legacy REST deprecation warning at startup

**File: `src/mcp_server/main.py`** — startup event, around line 355

If `MCP_HTTP_LEGACY_REST_ENABLED` is true, log a warning:
```
"Legacy REST MCP endpoints (/mcp/*) are deprecated and will be removed. "
"Set MCP_HTTP_STREAMABLE_ENABLED=true and use /_mcp instead."
```

No code removal — legacy REST stays functional but is no longer the default path.

---

### Phase 6: Tests and Contract Alignment

#### 6a. Fix contract tests

**File: `tests/contracts/test_mcp_streamable_contracts.py`** — line 122

Contract tests already expect dot notation (`kb.search`, `kb.retrieve_evidence`). After Phase 4a renames tools to dots, these tests will pass against the real server for the first time.

Add new assertions for the enriched evidence pack schema:
- `kb.retrieve_evidence` response must include `coverage` block
- Quote objects must include `doc_tag`, `parent_path`, `source`, `rank`

#### 6b. Evidence pack integration tests

**New file: `tests/mcp/test_evidence_pack.py`**

Test cases:
- Evidence pack returns enriched quotes with retrieval scores (not keyword overlap)
- `internal_fetch_k` retrieves more candidates than `max_quotes` returns
- Scratch payload includes score fields
- Coverage metadata is populated
- Graph expansion neighbors appear in evidence when available
- Graceful degradation when Neo4j is disabled (no graph enrichment, still returns evidence)

#### 6c. Query reformulation tests

**Extend: `tests/mcp/test_query_service.py`** (or create if not exists)

Test cases:
- Keyword soup is detected and reformulated
- Well-formed questions pass through unchanged
- Intent-aware template selection (config vs error vs procedure)
- Dual-query: `query_original` populated when rewrite occurs
- Diagnostics include `rewrite_template_used`

#### 6d. Tool profile tests

**New file: `tests/mcp/test_tool_profiles.py`**

Test cases:
- Production profile exposes exactly 3 tools
- Analyst profile exposes all `kb.*` and `graph.*` tools
- Full profile exposes all tools including backward aliases
- Unknown profile falls back to production
- `_call_tool` with a tool not in active profile returns error

---

### Phase 7: Retrieval Trace System

Full-pipeline observability for the server-orchestrated evidence path. Every `kb.retrieve_evidence` call produces a human-readable trace file showing exactly what happened at each decision point.

**Design:** Always-on, session-correlated, file + MCP resource access.

#### 7a. RetrievalTraceBuilder

**New file: `src/mcp_server/retrieval_trace.py`**

A trace builder that accumulates data at each pipeline stage:

```python
class RetrievalTraceBuilder:
    def __init__(self, trace_id: str, session_id: str): ...
    def record_query(self, client_query, reformulated, method, latency_ms, dual_query): ...
    def record_candidates(self, signal_name, candidates: list[dict]): ...
    def record_signal_pool(self, enabled, pool_size, slot_fills, degraded): ...
    def record_reranker(self, model, instruction, input_count, output_count, latency_ms, top_results): ...
    def record_graph_enrichment(self, seeds, neighbors_added, neighbor_details): ...
    def record_evidence_pack(self, quotes, coverage): ...
    def record_followup_call(self, tool_name, arguments, result_summary): ...
    def format(self) -> str: ...  # Human-readable text with section headers
    def to_dict(self) -> dict: ...  # Structured JSON for programmatic access
```

Each `record_*` method stores its data. `format()` produces the modular text output with section headers and tables. `to_dict()` produces structured JSON.

Trace sections:
1. **QUERY** — client query, reformulated query, method, latency, dual-query status
2. **CANDIDATES BY SIGNAL** — top 5 per signal type with scores and chunk IDs
3. **SIGNAL POOL** — enabled, slot fill table, degraded flag
4. **RERANKER** — model, instruction, input/output count, latency, top 10 with rank movement
5. **GRAPH ENRICHMENT** — seeds, neighbors added, relationship types
6. **EVIDENCE PACK** — quotes with scores, coverage metadata
7. **FOLLOW-UP CALLS** — appended as graph.expand / kb.read_excerpt are called

#### 7b. Trace file writer

**File: `src/mcp_server/retrieval_trace.py`**

Always-on: every `kb.retrieve_evidence` call writes a trace file.

```python
TRACE_DIR = os.getenv("MCP_RETRIEVAL_TRACE_DIR", "logs/retrieval_traces")

def write_trace(trace: RetrievalTraceBuilder) -> str:
    """Write trace to file. Returns the file path."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    filename = f"{trace.timestamp:%Y%m%d_%H%M%S}_{trace.trace_id[:8]}.txt"
    path = os.path.join(TRACE_DIR, filename)
    with open(path, "w") as f:
        f.write(trace.format())
    # Also write JSON sidecar for programmatic access
    json_path = path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)
    return path
```

Retention: configurable via `MCP_RETRIEVAL_TRACE_RETENTION_HOURS` (default 72). Cleanup of old traces on write.

#### 7c. Integration into kb.retrieve_evidence

**File: `src/mcp_server/mcp_app.py`**

Wire the trace builder into the evidence pack flow:

```python
async def kb_retrieve_evidence(...):
    trace = RetrievalTraceBuilder(trace_id=uuid4().hex, session_id=effective_session)
    # After reformulation:
    trace.record_query(...)
    # After _kb_search_candidates:
    trace.record_candidates(...)  # per-signal breakdown from metrics
    trace.record_signal_pool(...)  # from metrics
    # After reranker:
    trace.record_reranker(...)
    # After graph enrichment:
    trace.record_graph_enrichment(...)
    # After evidence extraction:
    trace.record_evidence_pack(...)
    # Write trace
    trace_path = write_trace(trace)
    # Add trace_id to response for correlation
    finalized["trace_id"] = trace.trace_id
```

#### 7d. Follow-up call correlation

**File: `src/mcp_server/mcp_app.py`**

Session-based correlation: maintain a `_ACTIVE_TRACES` dict keyed by session_id.

When `graph.expand` or `kb.read_excerpt` is called, check if there's an active trace for the session and append the follow-up call:

```python
_ACTIVE_TRACES: dict[str, RetrievalTraceBuilder] = {}

# In kb_retrieve_evidence, after writing:
_ACTIVE_TRACES[effective_session] = trace

# In graph.expand / kb.read_excerpt handlers:
active_trace = _ACTIVE_TRACES.get(effective_session)
if active_trace:
    active_trace.record_followup_call(tool_name, args, result_summary)
    write_trace(active_trace)  # Update the file
```

#### 7e. MCP resource for trace access

**File: `src/mcp_server/mcp_app.py`**

Register trace resources in `build_mcp_server()`:

- `wekadocs://traces/latest` — most recent trace for the current session
- `wekadocs://traces/{trace_id}` — specific trace by ID

Resource handler reads the formatted text from the trace file. This lets MCP clients (including LLM analysts) access traces without filesystem access.

#### 7f. Trace content depth strategy

Traces show **preview snippets** (first 150 chars) for candidates, with deeper inspection available:

| Trace Section | Content Level |
|---|---|
| Candidates by signal (sec 2) | 150-char preview + chunk_id + parent_path |
| Signal pool (sec 3) | Slot fill counts only |
| Reranker output (sec 4) | 150-char preview + chunk_id + rank movement |
| Evidence pack (sec 6) | Full quote text (max 500 chars) |
| **Full Text Appendix** | **Full text of top 20 reranked candidates** |

The appendix at the bottom of each trace includes the complete text, heading, doc_tag, and parent_path for the top 20 reranked chunks. This makes the trace self-contained for the most important candidates.

#### 7g. CLI chunk inspector

**New file: `src/tools/inspect_chunk.py`** (~50 lines)

Ad-hoc Qdrant point lookup for any chunk by ID:

```bash
python -m src.tools.inspect_chunk chunk_abc123
```

Output: full text, heading, doc_tag, parent_path, token_count, all score fields from the Qdrant payload. Uses the Qdrant client directly (no Neo4j dependency). For operators who need to inspect chunks outside the top 20 appendix.

#### 7h. Candidate-level data collection

The trace needs per-signal candidate data. This data is partially available in `retrieve()` metrics but needs to be surfaced more explicitly.

**File: `src/query/hybrid_retrieval.py`**

Add to `retrieve()` metrics dict:
```python
metrics["candidates_by_signal"] = {
    "bm25": [{"id": c.chunk_id, "score": c.bm25_score} for c in bm25_results[:5]],
    "content_dense": [{"id": c.chunk_id, "score": c.vector_score} for c in vec_results[:5]],
    ...
}
metrics["reranker_input_count"] = len(rerank_candidates)
metrics["reranker_output_top10"] = [
    {"id": c.chunk_id, "score": c.rerank_score, "original_rank": c.rerank_original_rank}
    for c in reranked[:10]
]
```

This data flows through `_kb_search_candidates()` → `diagnostic_context["metrics"]` → trace builder.

---

## Execution Order (Dependency Graph)

```
Phase 1 (evidence data flow)        ──┐
Phase 3a (deploy Qwen2.5-1.5B)      ──┤── independent, can parallel
Phase 3d-e (instruction prefixes)    ──┤   (3d-e are config/provider changes, no gateway dep)
Phase 4a-b (naming + cleanup)        ──┘
                                       ↓
Phase 2 (graph enrichment)          ── depends on Phase 1 (enriched scratch)
Phase 3b-c (LLM rewrite + dual-q)   ── depends on Phase 3a (gateway running)
Phase 4c (tool profiles)             ── depends on Phase 4a (dot names)
                                       ↓
Phase 4d (instructions)              ── depends on Phase 4c (profile names)
Phase 5 (transport)                  ── independent, can parallel with Phase 4d
                                       ↓
Phase 6 (tests)                      ── after all implementation phases
```

**Recommended execution:**
1. Deploy Qwen2.5-1.5B-Instruct to gateway + verify `/v1/rerank` instruction support (infra, pre-code)
2. Phases 1 + 3d-e + 4a-b in parallel (3 independent code workstreams) — **DONE**
3. Phases 2 + 3b-c + 4c (dependent on step 2) — **DONE**
4. Phases 4d + 5 (dependent on step 3)
5. Phase 6 (tests, after all implementation)
6. Phase 7 (retrieval traces — after all pipeline changes are stable)

**Pre-implementation verification (Step 1):**
- Confirm Qwen2.5-1.5B-Instruct serves at `/v1/chat/completions` through the gateway
- Test `/v1/rerank` with an `instruction` field to determine reranker instruction passthrough support
- Measure reformulation latency (target: <100ms on 3090)

---

## Files Summary

| File | Action | Phase |
|---|---|---|
| `src/mcp_server/mcp_app.py` | Major: evidence extraction rewrite, scratch enrichment, tool naming, profiles, instructions, evidence response schema | 1, 2, 4 |
| `src/mcp_server/query_service.py` | Modify: LLM-based `_rewrite_keyword_query`, dual-query return from `search_sections_light` | 3 |
| `src/query/hybrid_retrieval.py` | Modify: add `query_original` parameter to `retrieve()`, route to BM25/sparse | 3 |
| `src/providers/embeddings/embedding_service.py` | Modify: profile-aware instruction prefix in `embed_query()` | 3 |
| `src/providers/rerank/local_reranker_service.py` | Modify: domain-tuned reranker instruction | 3 |
| `config/embedding_profiles.yaml` | Modify: add `query_instruction` field to profiles | 3 |
| `src/shared/config.py` | Modify: add `reranker_instruction` to `RerankerConfig` | 3 |
| `src/mcp_server/main.py` | Modify: flip transport defaults, add legacy deprecation warning | 5 |
| `tests/contracts/test_mcp_streamable_contracts.py` | Modify: add enriched evidence pack assertions | 6 |
| `tests/mcp/test_evidence_pack.py` | **NEW**: evidence pack integration tests | 6 |
| `tests/mcp/test_tool_profiles.py` | **NEW**: tool profile tests | 6 |
| `src/mcp_server/retrieval_trace.py` | **NEW**: trace builder + file writer + formatting | 7 |
| `src/query/hybrid_retrieval.py` | Modify: add candidates_by_signal + reranker_output to metrics | 7 |
| `src/tools/inspect_chunk.py` | **NEW**: CLI chunk inspector (Qdrant point lookup) | 7 |
| `tests/mcp/test_retrieval_trace.py` | **NEW**: trace builder + formatting tests | 7 |

---

## Verification

```bash
# Unit/integration tests
pytest tests/mcp/test_evidence_pack.py -v
pytest tests/mcp/test_tool_profiles.py -v
pytest tests/contracts/test_mcp_streamable_contracts.py -v
pytest tests/query/test_signal_pool.py -v        # no regressions
pytest tests/query/test_reranker_integration.py -v  # no regressions

# Entry point imports
python -c "from src.mcp_server.main import app"
python -c "from src.mcp_server.mcp_app import build_mcp_server"

# Tool profile verification (STDIO)
MCP_TOOL_PROFILE=production python -c "
from src.mcp_server.mcp_app import _tool_specs, TOOL_PROFILES
specs = _tool_specs()
profile = TOOL_PROFILES['production']
filtered = [s for s in specs if s['name'] in profile]
print(f'Production tools: {[s[\"name\"] for s in filtered]}')
assert len(filtered) == 3
"

# Evidence pack smoke test (requires running services)
# Start server with MCP_HTTP_STREAMABLE_ENABLED=true MCP_TOOL_PROFILE=production
# Call kb.retrieve_evidence with a test query
# Verify: quotes have doc_tag, parent_path, confidence > 0
# Verify: coverage block present with documents_searched > 0
# Verify: graph_expansion_applied is true (when Neo4j available)

# CI guard (no new ACTIVE→DEAD violations)
python scripts/ci/check_dead_imports.py
```

---

## Model Stack Reference

| Model | Role | Context | VRAM | Endpoint | Instruction Support |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | Dense embedding | 32K tokens | ~1.5 GB | `/v1/embeddings` | Yes — `Instruct:` prefix on queries |
| SPLADEv3 | Learned sparse | 512 tokens | ~0.5 GB | `/v1/embeddings/sparse` | No |
| ColBERTv2 | Late interaction | 512 tokens | ~0.5 GB | `/v1/embeddings/colbert` | No |
| Qwen3-Reranker-4B | Cross-encoder | 8K tokens | ~8 GB | `/v1/rerank` | Yes — custom system instruction |
| GLiNER Medium v2.1 | NER | — | ~1 GB | `/v1/extract` | No |
| **Qwen2.5-1.5B-Instruct** | **Query reformulation** | **32K tokens** | **~3 GB** | **`/v1/chat/completions`** | **N/A — IS the instruction follower** |

**Total projected VRAM:** ~14.5 GB / 24 GB (60%) — 9.5 GB headroom.

All models served through unified gateway at `10.25.0.50:8080`.

The 512-token ColBERT/SPLADE limit + 8K reranker asymmetry means the evidence pack's deep retrieval (60-150 candidates) captures chunks that were truncated at retrieval time but can be fully evaluated by the cross-encoder.

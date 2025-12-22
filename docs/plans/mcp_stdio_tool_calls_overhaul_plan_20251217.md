# MCP STDIO Tool-Call Overhaul Plan (Hybrid Graph+Vector RAG + Scratch Resources)
**Project:** `wekadocs-matrix`
**Date:** 2025-12-17
**Primary target:** `src/mcp_server/stdio_server.py` (STDIO transport)
**Secondary targets:** `src/mcp_server/query_service.py`, `src/services/graph_service.py`, `src/services/text_service.py`, `src/services/context_assembler.py`, `docs/mcp/retrieval_playbook.md`, `docs/api-contracts.md`

---

## 0) Executive Summary (What will change)

We will overhaul the STDIO MCP tool surface so the LLM can:

1. **Choose the right retrieval mode** (vector-only vs graph-first hybrid) without guessing.
2. **Make correct, scoped queries** (explicit, validated filters + project/doc scope).
3. **Consume retrieval results without context blowups** via:
   - progressive disclosure (snippets → bounded excerpts → optional expansion),
   - **server-side evidence extraction** (high-signal spans, not full sections),
   - **server-side scratch storage** (large blobs never returned directly by default).
4. **Avoid selection pathologies** (“LLM ignores top-5”) by making the *preview* match the reranker signal (best evidence span, deduped, top-K small).
5. **Stay hardened and regression-resistant** (strict schemas, budgets, pagination, error taxonomy, feature flags, contract tests).

This plan is prescriptive: exact tool names, defaults, caps, and roll-out steps are specified.

---

## 1) Grounding (Authoritative guidance we will implement)

### MCP protocol (tools/resources/schema)
- Tools are **model-controlled**; good schemas + descriptions materially affect tool selection and correctness.
- Tool results support **resource links**, **embedded resources**, **structuredContent**, and **outputSchema**.
- Resources are **application-driven**; hosts decide whether/how to include them in model context. Resource metadata like `size` and annotations (e.g., audience/priority/lastModified) are specifically intended to help hosts manage inclusion.
- Servers **MUST validate** resource URIs; access controls **SHOULD** exist for sensitive resources.

### Agent/tool ergonomics and token efficiency (Anthropic)
- Tool definitions and intermediate tool results can overload context windows.
- Good tools enforce bounded outputs, pagination, truncation, and “progressive disclosure”.
- Descriptions should explicitly encode usage heuristics (when to use, when not to use, defaults/caps, follow-up tool).

### GraphRAG patterns (Microsoft GraphRAG)
- **Local search**: seed entities (vector) → traverse graph neighborhood → pull linked text units → prioritize+filter to context budget.
- **Global search**: map-reduce over pre-summarized communities; prune irrelevant reports early (dynamic selection) to control cost.

We will adopt GraphRAG’s **local search** philosophy directly (it matches our hybrid graph+vector pipeline), and we will only implement “global search” if/when the graph has community/report artifacts.

### MCP SDK prerequisites (pinned capabilities)
This plan intentionally migrates STDIO tool handling to the **low-level MCP server** (see Phase 3). For client interoperability, the runtime MUST support:
- `tools/list` returning `inputSchema` and optional `outputSchema`
- `call_tool` returning `structuredContent` (validated against `outputSchema` when present)
- `resources/templates/list` + `resources/read` for scratch URIs
- STDIO transport (`mcp.server.stdio`) for Claude Desktop

Practical requirement: pin the MCP Python SDK (the `mcp` package) in `requirements`/lockfiles to a version that provides:
- `mcp.server.lowlevel.server.Server`
- `mcp.server.stdio.stdio_server`
- JSON schema validation for tool inputs/outputs

### Tool naming constraints (MCP SEP-986)
To maximize compatibility across MCP clients (Claude Desktop, VS Code, LibreChat, OpenWebUI, and custom hosts), tool names MUST adhere to the accepted MCP naming guidance:
- Length: 1–64 characters
- Case-sensitive
- Allowed characters: `A–Z a–z 0–9 _ - . /`
- Disallowed: spaces, commas, and other special characters

This plan’s namespaced tool set (`kb.search`, `graph.paths`, etc.) is intentionally compliant (dots are allowed and recommended for namespacing clarity).

### Client compatibility strategy (STDIO)
Different MCP hosts may differ in how they surface tool metadata, `structuredContent`, and resources. To remain robust:
- Every tool call MUST return a small, human-readable `content` summary (even when `structuredContent` is present).
- Machine-parseable data MUST be in `structuredContent`, validated against `outputSchema` when provided.
- Scratch resources MUST be optional from the client’s perspective:
  - do not assume a client will auto-fetch `resource_link` URIs
  - provide follow-up bounded tools (`kb.read_excerpt`, `kb.expand_excerpt`) so hosts that do not use resources can still retrieve needed text safely
- Avoid relying on host-specific tool-name qualifiers (e.g., “ServerName:tool_name”) in the tool contract; MCP `Tool.name` is server-local and should remain stable. Hosts may display a qualified name in UI, but the tool registry should not depend on it.

---

## 2) Current State Audit (STDIO server gaps that cause bad tool use)

### 2.1 Tool surface and naming
- STDIO currently exposes low-level tools (e.g., `search_sections`, `expand_neighbors`, `get_section_text`) without a namespace.
- `FastMCP` server instructions refer to `search_sections(verbosity="snippet")` but `search_sections` has **no `verbosity` parameter**.
- `docs/api-contracts.md` describes `search_documentation` and traversal tools as the primary interface, but STDIO has `search_documentation` **disabled by default** (env gated). This forces the LLM to orchestrate low-level steps and increases mis-tooling risk.

### 2.2 Output bloat and LLM mis-selection
- `search_sections` returns a large per-result metadata payload by default (scores, debug fields, entity metadata, RRF contributions, etc.) and defaults `top_k=20`.
- The tool returns **no “preview span”** (a short evidence snippet). This causes the LLM to choose based on headings/titles rather than the same signal the reranker scored.
- `get_section_text` returns truncated excerpts (good) but still pushes raw text into the conversation (no scratch indirection).

### 2.3 Schema and error semantics
- Tool outputs are mostly untyped `dict` returns. In practice, this causes:
  - weak/implicit schemas (harder for LLMs and clients),
  - inconsistent error signaling (`{"error": ...}` in successful tool result instead of `isError` semantics / consistent error envelopes).

### 2.4 “Scratch space” is missing
- There is no server-side scratch artifact store with resource URIs for large blobs, so “retrieve big text” and “paste into context” remains the default pattern.

---

## 3) Non-negotiable invariants (to prevent regressions)

### 3.1 Backwards compatibility
- Existing tool names remain callable for at least one full release cycle.
- Output fields currently depended on by tests remain available **when requested** (via explicit `verbosity="debug"` / `include_debug=true`), but will not be returned by default.
 - Compatibility wrappers MUST preserve:
   - the **exact input signature** of legacy tools (including parameter defaults)
   - legacy defaults (e.g., `search_sections.top_k=20`) unless a feature flag explicitly opts into new defaults
   - legacy output shape unless `include_debug=false` is explicitly requested

### 3.2 Strict budgets (server-enforced)
- All retrieval tools enforce **hard caps** on:
  - `top_k`, `page_size`,
  - snippet length,
  - excerpt token/byte length,
  - total response bytes (`MAX_RESPONSE_BYTES`).
- No tool returns unbounded text (server-side truncation always on).

### 3.3 Scope isolation (no cross-project mixing)
Even though this repo is one project, we design as if multi-tenant:
- Every query tool MUST accept a `scope` object (see §4.2) and enforce it server-side.
- Default scope is derived from server config, not from the model’s guess.

---

## 4) Target Tool Surface (Namespaced, progressive disclosure, hybrid-aware)

### 4.1 Namespacing strategy
We will introduce namespaced tools and preserve old names as wrappers:

**New (preferred):**
- `kb.search`
- `kb.read_excerpt`
- `kb.expand_excerpt`
- `kb.extract_evidence`
- `kb.retrieve_evidence` (high-level “do the right thing”)
- `graph.describe`
- `graph.expand`
- `graph.paths`
- `graph.parents`
- `graph.children`
- `graph.entities_for_sections`
- `graph.sections_for_entities`

**Existing (compat wrappers):**
- `search_sections` → `kb.search` (with a compatibility mapping)
- `get_section_text` → `kb.read_excerpt` (compat mode)
- existing graph tools map 1:1 to `graph.*`

Rationale: namespacing reduces tool confusion and makes the LLM’s routing decisions easier.

### 4.2 Canonical scope model (required for all retrieval tools)
All retrieval tools MUST accept:

```json
{
  "scope": {
    "project_id": "wekadocs-matrix",
    "environment": "dev|prod|scratch",
    "doc_tags": ["weka", "release-notes"],
    "repositories": ["..."]
  }
}
```

Server behavior:
- If `scope` is omitted, server injects defaults from config.
- If `scope.project_id` is provided and does not match server config, tool returns an error result (no cross-project mixing).
- If `doc_tags` is present, **enforce** it in Qdrant and Neo4j queries.

### 4.3 Tool: `kb.search` (discovery-only; no section text)

**Purpose:** Return a shortlist of candidates with *answer-shaped previews* and stable IDs; never dump full text.

**Default behavior (non-negotiable):**
- `top_k` default: **5**
- `max_per_doc` default: **1** (dedupe)
- preview: **best evidence span**, not section prefix
- return only minimal fields by default; debug fields only when requested

**Input (exact):**
```json
{
  "query": "string",
  "top_k": 5,
  "cursor": "string|null",
  "page_size": 5,
  "scope": { "...": "..." },
  "filters": {
    "doc_tag": ["..."],
    "path_prefix": "string|null",
    "updated_after": "ISO-8601|null"
  },
  "options": {
    "max_snippet_chars": 280,
    "max_per_doc": 1,
    "include_scores": false,
    "include_debug": false,
    "mode": "auto|vector_only|hybrid_local"
  }
}
```

**Output (exact):**
```json
{
  "results": [
    {
      "passage_id": "opaque string (server generated)",
      "section_id": "chunk id",
      "doc_tag": "string",
      "title": "string",
      "rank": 1,
      "score": 0.82,
      "source": "reranked|rrf_fusion|graph_expanded|vector|bm25|hybrid",
      "preview": "string <= max_snippet_chars",
      "scratch_uri": "wekadocs://scratch/<session>/<passage_id>",
      "size_bytes": 12345,
      "anchor": "string|null"
    }
  ],
  "next_cursor": "string|null",
  "partial": true,
  "limit_reason": "page_size|token_cap|byte_cap|none",
  "session_id": "string"
}
```

**Server implementation requirements:**
- Compute `preview` via server-side span selection:
  - sentence split the chunk text,
  - score sentences by lexical overlap with query (deterministic; see §4.6.1),
  - return top 1–3 sentences, clipped to `max_snippet_chars`.
- Store full chunk text in scratch (see §5) keyed by `passage_id`.

**Compatibility note (non-negotiable):**
- `kb.search` uses the new defaults (`top_k=5`, `max_per_doc=1`) for agent ergonomics.
- The legacy `search_sections` wrapper MUST retain `top_k=20` as its default and should map `top_k` through unchanged when callers provide it.

### 4.4 Tool: `kb.read_excerpt` (bounded excerpt, from scratch)

**Purpose:** Read a small excerpt of a known candidate without retrieving the full section.

**Input (exact):**
```json
{
  "passage_id": "string",
  "max_tokens": 300,
  "start_char": 0,
  "scope": { "...": "..." },
  "options": {
    "format": "text|bullets",
    "include_citation": true
  }
}
```

**Output (exact):**
```json
{
  "passage_id": "string",
  "excerpt": "string (hard-capped by max_tokens AND max_bytes)",
  "truncated": true,
  "next_start_char": 1200,
  "citation": {
    "section_id": "string",
    "doc_tag": "string",
    "title": "string",
    "uri": "wekadocs://scratch/<session>/<passage_id>"
  }
}
```

**Hard caps:**
- `max_tokens` default 300, max 800
- `max_bytes` per call: 32KB (reuse `MAX_TEXT_BYTES_PER_CALL`)

### 4.5 Tool: `kb.expand_excerpt` (small incremental expansion)
Same as `kb.read_excerpt`, but input is deltas:
```json
{
  "passage_id": "string",
  "before_tokens": 150,
  "after_tokens": 150,
  "scope": { "...": "..." }
}
```
Server returns a bounded expanded excerpt with `truncated` + `next_start_char`.

### 4.6 Tool: `kb.extract_evidence` (preferred “anti-bloat” evidence tool)

**Purpose:** Return only the minimal set of quotes/sentences that answer the question.

**Input:**
```json
{
  "question": "string",
  "passage_ids": ["string"],
  "max_quotes": 6,
  "max_quote_tokens": 80,
  "include_context_tokens": 20,
  "scope": { "...": "..." }
}
```

**Output:**
```json
{
  "quotes": [
    {
      "quote": "string (<= max_quote_tokens)",
      "passage_id": "string",
      "section_id": "string",
      "title": "string",
      "uri": "wekadocs://scratch/<session>/<passage_id>",
      "confidence": 0.0
    }
  ]
}
```

**Server implementation requirements (deterministic, no extra LLM):**
#### 4.6.1 Evidence extraction algorithm (exact, deterministic)
To remove ambiguity and keep behavior stable under regression tests:
1. **Normalize** `question`:
   - lowercase
   - split on non-alphanumerics to tokens
   - drop tokens shorter than 3 chars
2. **Split text into spans** for each `passage_id`:
   - Treat fenced code blocks (```…```) as single spans (cap span length by truncation rules below).
   - Otherwise split on:
     - blank lines, bullet boundaries, and sentence-ending punctuation (`.?!`) followed by whitespace/newline.
3. **Score each span** with a simple weighted overlap:
   - `score(span) = sum_{t in Q} (1 if t in span else 0) / max(1, |Q|)`
   - Tie-breakers (in order): shorter span length, earlier position in passage.
4. **Select quotes**:
   - take top spans across all passages, respecting `max_quotes`
   - hard-cap each quote to `max_quote_tokens` (token estimate) and also to a fixed char budget (e.g., 500 chars) to avoid pathological long lines.
5. **Optional rerank (span-level)**:
   - if the cross-encoder reranker is enabled, it MAY rerank only the top M spans (e.g., M=24) before final selection.
6. **Return** the selected quotes with citations, never full text.

This algorithm is intentionally simple to:
- avoid adding extra model calls
- keep results stable and testable
- keep MCP payloads bounded

### 4.7 Tool: `kb.retrieve_evidence` (high-level default; “do the right thing”)

**Purpose:** Stop making the LLM do retrieval orchestration. This tool:
- runs `kb.search` (top 5, deduped),
- then runs `kb.extract_evidence` on those candidates,
- returns only quotes + citations and minimal diagnostics.

**Default recommendation:** The LLM should call `kb.retrieve_evidence` unless the user explicitly requests exploration.

---

## 5) Scratch Space (Server-side artifacts + MCP resources)

### 5.1 Scratch store requirements
Implement `ScratchStore` (new module, e.g. `src/mcp_server/scratch_store.py`) with:
- in-memory LRU + TTL (default TTL 30 minutes),
- bounded memory usage (configurable; default 256MB),
- per-session namespace (`session_id` from STDIO context),
- opaque IDs (`passage_id`) not derived from raw content hashes.
 - async-safety:
   - concurrent tool calls MUST NOT corrupt LRU state or allow cross-session reads
   - use an async lock strategy (global or per-session) and keep critical sections small
 - operational visibility:
   - emit counters for evictions and a warning log when memory pressure triggers frequent evictions

Stored payload per `passage_id`:
```json
{
  "section_id": "...",
  "doc_tag": "...",
  "title": "...",
  "text": "... full chunk text ...",
  "source_uri": "...",
  "created_at": "ISO-8601",
  "size_bytes": 12345
}
```

### 5.2 MCP resources (UI pointers; not auto-included)
Register a resource template:
- `wekadocs://scratch/{session_id}/{passage_id}`

Resource metadata MUST include:
- `size` (bytes),
- `mimeType: text/plain`,
- `annotations.audience: ["assistant"]`,
- `annotations.priority: 0.1` for large blobs,
- `annotations.lastModified` (cache write time or source doc time).

**Security (non-negotiable):**
- Validate all resource URIs.
- Enforce that `session_id` in URI matches the active session (no cross-session access).
- If running multi-tenant in the future, include `project_id` in the URI and enforce it.

### 5.3 “No context pollution” rule
Tools MUST NOT return full scratch text by default.
- `kb.search` returns only previews + URIs.
- `kb.retrieve_evidence` returns only quotes (bounded).
- `kb.read_excerpt` returns bounded excerpt only.

---

## 6) Graph+Vector integration (Hybrid GraphRAG local search)

### 6.1 Retrieval modes (server-selected)
`options.mode`:
- `vector_only`: no graph expansion, no graph tools suggested.
- `hybrid_local`: use existing `HybridRetriever.retrieve(expand=True)` as the default.
- `auto`: server chooses:
  - if Neo4j disabled → `vector_only`
  - else → `hybrid_local`

### 6.2 Graph exploration tools (tight, projection-only)
We keep the existing projection-only graph tools, but:
- rename into `graph.*`,
- default `page_size` small (25), `max_hops` small (1–2),
- ensure all graph tools return **projections only** unless `include_snippet=true`.

### 6.3 When to use graph tools (tool descriptions MUST encode this)
Every `graph.*` tool description MUST include:
- “Use after `kb.search` to explore relationships between the top 1–3 candidates.”
- “Do not call if `neo4j_disabled=true` (use `kb.search`/`kb.retrieve_evidence` only).”
- “Do not fetch text until you have narrowed to ≤3 passages.”

---

## 7) Tool descriptions and annotations (LLM routing correctness)

### 7.1 Description format (mandatory)
Each tool description MUST include:
1. Trigger condition (“Use when…”)
2. Anti-trigger (“Do not use when…”)
3. Output budget (“Returns at most…”)
4. Follow-up path (“If you need more, call…”)
5. Defaults and caps (“Defaults: … Max: …”)

### 7.2 Tool annotations (mandatory)
All retrieval tools:
- `readOnlyHint: true`
- `openWorldHint: false`
- `idempotentHint: true`
- `destructiveHint: false`

Graph traversal tools:
- same (read-only, closed world).

---

## 8) Output format strategy (token efficiency + correctness)

### 8.1 Immediate fix (no library changes)
For FastMCP, returning dicts produces pretty-printed JSON tool output, inflating tokens.

**Policy:** Retrieval tools MUST return a `str` containing **minified JSON** (no indentation) unless returning `ContentBlock` objects (resource links, etc.).

This reduces context cost immediately without waiting on SDK changes.

### 8.2 Structured outputs (Phase 3 hardening)
Introduce explicit `outputSchema` + `structuredContent` for all tools, but do it without duplicating large JSON in `content`.

**Implementation choice (prescriptive):**
- Migrate STDIO server off `FastMCP` tool handling to the low-level MCP server handler so tools can return:
  - **unstructured**: a short text summary (or empty),
  - **structuredContent**: the full JSON object validated against `outputSchema`.
- Tool functions MUST return `(content_blocks, structured_dict)` tuples.
  - Returning a raw `dict` causes the low-level handler to serialize JSON into `content` (indented), which is token-expensive.
  - Use `content_blocks` for a short human-readable summary and (optionally) a small, minified “compat summary” JSON object if a target MCP client is known to ignore `structuredContent`.

Deliverable: tool results are schema-validated and context-light.

---

## 9) Error handling (hardened, unambiguous)

### 9.1 Error taxonomy (exact)
All tool failures must map to one of:
- `INVALID_ARGUMENT`
- `SCOPE_VIOLATION`
- `BACKEND_UNAVAILABLE`
- `TIMEOUT`
- `BUDGET_EXCEEDED`
- `INTERNAL_ERROR`

### 9.2 Error envelope (exact)
All tools return either:
- normal result object, or
- `{ "error": { "code": "...", "message": "...", "details": {...} } }`

Additionally:
- runtime errors should use MCP tool error signaling (`isError: true`) once Phase 3 is complete.

---

## 10) Implementation Plan (phased, hardened, no regressions)

### Phase 0 — Baseline + golden set (1 day)
1. Add a “MCP STDIO golden query set” (10–20 queries) and expected behaviors:
   - max tool calls,
   - max bytes per tool,
   - top-1 doc should match (where deterministic).
2. Add lightweight instrumentation to record:
   - response sizes,
   - cursor usage,
   - dedupe count,
   - excerpt truncation rate.
3. Reuse the existing repo golden sets (do not invent a new one):
   - `docs/golden-set-queries.md` (frozen baseline)
   - `tests/fixtures/golden_query_set.yaml` and `tests/e2e/test_golden_set.py`

Rollback (Phase 0): no behavior changes yet.

**DoD:** reproducible baseline metrics committed.

### Phase 1 — New namespaced tools + preview spans (2–3 days)
1. Implement `ScratchStore` + passage ID mapping.
2. Implement `kb.search` using `QueryService.search_sections_light`, but:
   - default `top_k=5`,
   - `max_per_doc=1` dedupe,
   - compute `preview` span,
   - store full text in scratch and return `passage_id`.
3. Implement wrappers:
   - `search_sections` delegates to `kb.search` with `include_debug=true` to preserve current payload fields when needed.
4. Fix server instructions string to match actual parameters (remove nonexistent `verbosity` mention).

Rollback (Phase 1):
- Set `MCP_KB_V2_ENABLED=false` (or keep disabled by default) so only legacy tools are used.

**DoD:** `kb.search` works end-to-end; old tools still work; no tests broken.

### Phase 2 — Excerpt + evidence tools (2–4 days)
1. Implement `kb.read_excerpt` and `kb.expand_excerpt` reading from scratch.
2. Implement `kb.extract_evidence` (deterministic span extraction).
3. Implement `kb.retrieve_evidence` as the recommended default retrieval tool.
4. Update `docs/mcp/retrieval_playbook.md` to recommend `kb.retrieve_evidence` by default, with graph exploration as an optional deep-dive.

Rollback (Phase 2):
- Keep `MCP_KB_V2_ENABLED=false` and/or hide `kb.*` tools from `tools/list` while retaining legacy tool surface.

**DoD:** Typical Q&A completes with ≤2 tool calls and bounded context growth.

### Phase 3 — Structured outputs + outputSchema (3–5 days)
1. Migrate STDIO server tool handling to low-level MCP server so tools can return `(content, structuredContent)` without duplicating JSON in text.
2. Define Pydantic models (inputs and outputs) for every tool; generate `inputSchema` and `outputSchema`.
3. Validate outputs against `outputSchema` in CI.

Rollback (Phase 3):
- Set `MCP_STRUCTURED_OUTPUT_ENABLED=false` to return legacy-compatible unstructured outputs (minified and capped) while keeping the low-level server in place.

**DoD:** `tools/list` includes outputSchema; `call_tool` returns structuredContent; content is short.

### Phase 4 — Hardening + security (2–3 days)
1. Enforce scope validation in every tool and in scratch resource reads.
2. Add rate limits / timeouts for Neo4j and Qdrant operations.
3. Ensure all logs stay on stderr in STDIO mode; add redaction for secrets and large text.

Rollback (Phase 4):
- Disable new operator/debug toggles and revert to strict “no raw text in logs” mode.

**DoD:** Threat-model checklist satisfied; no sensitive leakage in logs.

### Phase 5 — Rollout (1–2 days + monitoring)
1. Feature flags:
   - `MCP_KB_V2_ENABLED`
   - `MCP_STRUCTURED_OUTPUT_ENABLED`
2. Canary enable in dev; compare golden-set metrics.
3. Publish a migration note for clients (old tool names remain; new tools recommended).

Rollback (Phase 5):
- Flip feature flags off (`MCP_KB_V2_ENABLED=false`, `MCP_STRUCTURED_OUTPUT_ENABLED=false`) and re-run the golden set.

**DoD:** Stable performance, reduced context bloat, no regressions.

---

## 11) Acceptance Criteria (pass/fail, not subjective)

1. **Default retrieval does not dump text**:
   - `kb.search` never returns full section text.
2. **Bounded excerpt reads**:
   - `kb.read_excerpt` returns ≤800 tokens and ≤32KB bytes, always.
3. **Correct previews**:
   - at least 80% of golden queries: the answer is present in the top-5 candidates’ extracted evidence spans.
4. **Reduced “LLM ignores top-5” incidents**:
   - with `kb.retrieve_evidence`, the LLM never selects outside top-5 because it is no longer asked to select.
5. **Scope enforcement**:
   - attempting a mismatched `project_id` fails with `SCOPE_VIOLATION`.
6. **Schema stability**:
   - output schemas are versioned; changes are additive unless explicitly approved.
7. **Performance contracts preserved**:
   - meet or improve the repo’s published latency targets in `docs/api-contracts.md` for equivalent modes:
     - treat `kb.search` as `snippet` mode
     - treat excerpt reads as `full` mode
     - treat graph projections as `graph` mode
8. **Multi-client compatibility verified**:
   - Claude Desktop (STDIO): tools are listed, callable, and return bounded outputs; resources work or degrade gracefully via `kb.read_excerpt`
   - VS Code (MCP): tools list includes namespaced names; `structuredContent` is visible/consumable; resource links do not auto-bloat context
   - LibreChat/OpenWebUI/custom hosts (HTTP or bridged): can parse tool schemas; tolerate presence/absence of `structuredContent`; bounded text tools provide a safe fallback path

---

## 12) Concrete file-level work list (where edits will happen)

- `src/mcp_server/stdio_server.py`
  - add `kb.*` and `graph.*` tools
  - fix instructions string
  - keep compatibility wrappers
- `src/mcp_server/query_service.py`
  - expose a “raw chunk text access” path only for scratch population
  - add optional rewrite toggles scoped to tool options (avoid harming exact-match queries)
- `src/mcp_server/scratch_store.py` (new)
  - LRU+TTL scratch artifact storage
- `src/services/text_service.py`
  - refactor to optionally source text from scratch when available
- `docs/mcp/retrieval_playbook.md`
  - update playbook to the new flow
- `docs/api-contracts.md`
  - add the new `kb.*` and `graph.*` contracts as additive extensions
- `tests/…`
  - add contract tests for the new schemas and hard caps

---

## 13) Open questions (must be resolved before Phase 3)

1. Which MCP clients must we support (Claude Desktop only, VS Code, custom host)? This determines how much we can rely on resources being “out-of-band”.
2. Do we want `kb.retrieve_evidence` to be the default surfaced tool (recommended), or keep only low-level tools and rely on prompting?
3. Should query rewriting be controllable per tool call (recommended: yes, default on for `kb.search`, off for excerpt reads)?

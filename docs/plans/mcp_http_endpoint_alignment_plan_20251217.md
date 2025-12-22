# MCP HTTP Endpoint Alignment Plan (Streamable HTTP + Tool Parity with STDIO)
**Project:** `wekadocs-matrix`
**Date:** 2025-12-17
**Goal:** Overhaul the HTTP-facing MCP surface so it is fully aligned with `docs/plans/mcp_stdio_tool_calls_overhaul_plan_20251217.md` (same tools, same schemas, same scratch/resources strategy, same budgets, same GraphRAG+vector behavior).

---

## 0) Executive Summary

The current FastAPI “MCP” endpoints in `src/mcp_server/main.py` are not protocol-faithful MCP:
- They are REST-style endpoints (`/mcp/initialize`, `/mcp/tools/list`, `/mcp/tools/call`) rather than JSON-RPC over a supported MCP transport.
- Tool schemas mismatch actual arguments (e.g., `traverse_relationships` schema vs implementation).
- Tool outputs use a non-standard `"type": "json"` content block instead of MCP’s `structuredContent` + `outputSchema`.
- `resources` are disabled (no scratch resources / resource templates / `resources/read`), so HTTP clients cannot use the “resource-link pointer” pattern that prevents context bloat.

**We will replace the HTTP MCP surface with the MCP SDK’s Streamable HTTP transport (low-level server + session manager)** and ensure:
1. Tool parity with STDIO (`kb.*`, `graph.*`, scratch resources, progressive disclosure).
2. Schema parity (identical `inputSchema`/`outputSchema`, `structuredContent`, token-efficient content blocks).
3. Capability parity (`tools`, `resources`, `prompts` if we keep curated prompts).
4. Backwards compatibility: keep the legacy REST endpoints temporarily behind a feature flag while we migrate clients.

**Critical constraint:** `search_documentation` must remain purposefully disabled for general LLM tool use. The aligned HTTP MCP surface must not accidentally re-enable it or replicate its side effects (details in §2.5 and §4.2).

---

## 1) Grounding (Authoritative constraints we must satisfy)

### MCP transport reality
MCP is JSON-RPC. For HTTP, use an MCP-defined HTTP transport (e.g., Streamable HTTP or SSE) rather than ad-hoc REST endpoints that “look like MCP”.

### Tool naming constraints (MCP SEP-986)
To maintain cross-client compatibility (Claude Desktop, VS Code, LibreChat, OpenWebUI, and custom hosts), tool names MUST follow the accepted MCP naming guidance:
- Length: 1–64 characters
- Allowed characters: `A–Z a–z 0–9 _ - . /`
- Avoid spaces and other special characters

This plan’s tool names (`kb.search`, `graph.*`) are intentionally compliant.

### Tool output best practice
To avoid context blowups and parsing errors:
- Define `outputSchema`.
- Return data in `structuredContent` (not huge JSON blobs inside text).
- Use **resource links** and **resources/read** for big content, not embedded text.

### Token efficiency and agent ergonomics
HTTP must match the STDIO overhaul principles:
- Progressive disclosure (search → evidence → bounded excerpts only when needed).
- Server-side scratch store (big text stays out-of-band).
- Budgeted, deduped top-K defaults.
- Tool descriptions that encode routing heuristics (“use X first; don’t call Y unless…”).

### Client compatibility strategy (HTTP)
Different hosts integrate MCP over HTTP differently (SSE streaming vs JSON responses, resource support, `structuredContent` handling). This plan must support:
- **VS Code** MCP client (Streamable HTTP / SSE expected)
- **LibreChat/OpenWebUI** (may rely on custom MCP bridges; HTTP+JSON response mode may be required)
- **Custom hosts** (unknown capabilities; must be standards-compliant and tolerant)

Therefore:
- Always include a small `content` summary alongside `structuredContent`.
- Do not assume automatic resource fetching; provide bounded excerpt tools as the universal fallback.
- Prefer Streamable HTTP with SSE as the default, but keep an explicit compatibility path:
  - support JSON response mode when a host cannot use SSE (configurable in the HTTP transport/session manager).

---

## 2) Current HTTP MCP Implementation Audit (What’s broken)

### 2.1 Endpoints are not MCP protocol handlers
`src/mcp_server/main.py` exposes:
- `POST /mcp/initialize`
- `GET /mcp/tools/list`
- `POST /mcp/tools/call`

These endpoints are not MCP JSON-RPC endpoints and cannot support:
- MCP paging cursors for tool lists (as a JSON-RPC method),
- MCP `structuredContent` + output validation,
- MCP `resources/list`, `resources/read`, `resources/templates/list`,
- subscriptions / list-changed notifications.

### 2.2 Tool list mismatch
The HTTP tool list is hard-coded and incomplete:
- Only `search_documentation` and `traverse_relationships` are listed.
- The `traverse_relationships` input schema uses `node_id`, but implementation reads `start_ids` and `rel_types`.
- The HTTP `search_documentation` schema and “verbosity enum” do not match the STDIO direction (STDIO uses `search_sections` etc, and `search_documentation` is gated off).

### 2.3 Non-standard tool result encoding
`MCPToolCallResponse` returns `content: [{type:"text",...},{type:"json", json:{...}}]`.
- MCP `content` blocks do not define a `"json"` content type.
- The correct approach is: `content` (small) + `structuredContent` (validated) per MCP spec.

### 2.4 Capabilities lie
HTTP initialize response advertises:
- `tools: True`
- `resources: False`

But the STDIO overhaul plan requires resources (scratch URIs + templates).

### 2.5 Why `search_documentation` must remain disabled (critical)
`search_documentation` (as implemented via `QueryService.search` and `ResponseBuilder`) is not a safe “read-only retrieval primitive”:

- **Side effects:** when session tracking is enabled, it creates and updates Neo4j nodes/relationships (session/query/answer provenance). LLM tool calls must not mutate the KB unless explicitly intended and authenticated.
- **Context bloat:** `ResponseBuilder` fetches “full text” (byte-capped but still large) for multiple evidence items, and can append stitched context to the Markdown response. This is exactly the “retrieve big blobs and stuff them into the prompt” failure mode.
- **Hidden defaults:** it bundles rewriting, retrieval, formatting, provenance, and large outputs into one call, making it hard for the host/LLM to keep outputs bounded and scoped.

Therefore:
- HTTP must **not** list `search_documentation` in `tools/list` by default.
- If we ever expose it, it must be behind a secure flag and must be annotated as non-read-only, with hard output caps and explicit warnings.

---

## 3) Target Architecture (One tool registry, multiple transports)

### 3.1 Single source of truth: “MCP app factory”
Create a new module that constructs and returns the **low-level MCP server** instance used by *both* transports:
- `src/mcp_server/mcp_app.py` (new)
  - builds `mcp.server.lowlevel.server.Server(...)`
  - implements handlers for:
    - `tools/list` (publishes `kb.*` + `graph.*` tools with schemas + annotations)
    - `tools/call` (routes to the same tool implementations as STDIO)
    - `resources/templates/list` + `resources/read` for scratch URIs
    - prompts (optional)
  - provides dependency injection / lifespan shared with STDIO (`Deps` construction, single init)

Then:
- `src/mcp_server/stdio_server.py` runs the low-level server over STDIO transport (`mcp.server.stdio`)
- `src/mcp_server/main.py` mounts Streamable HTTP via `StreamableHTTPSessionManager` at a stable path

This guarantees tool parity and eliminates drift between STDIO and HTTP.

### 3.2 HTTP transport: Streamable HTTP (preferred)
Use the MCP SDK’s Streamable HTTP server transport with the low-level server:
- `mcp.server.streamable_http_manager.StreamableHTTPSessionManager`
- mount a handler that forwards requests to `session_manager.handle_request(...)`

Prescriptive mount strategy:
- Mount MCP Streamable HTTP at `/_mcp` (new) or `/mcp` (replacement).
- Keep the existing REST endpoints at `/mcp/*` behind a feature flag for one release cycle.

Rationale:
- Prevents breaking existing internal clients immediately.
- Allows MCP-compliant clients (VS Code, other hosts) to connect via a standard transport.

### 3.3 Capability parity
The HTTP MCP server MUST advertise:
- `tools` capability with listChanged if supported
- `resources` capability (subscribe/listChanged optional)
- `prompts` capability if curated prompts are registered

### 3.4 Explicit HTTP endpoint map (prescriptive)
Once implemented, the HTTP service will expose:

- **MCP Streamable HTTP (new, canonical):**
  - Base path: `/_mcp` (mount)
  - All MCP JSON-RPC methods (tools/resources/prompts) are served by the MCP SDK’s Streamable HTTP session manager + low-level server behind this mount.
- **Legacy REST shim (temporary, deprecated):**
  - `/mcp/initialize`
  - `/mcp/tools/list`
  - `/mcp/tools/call`

Important: once Streamable HTTP is live, the `/mcp/*` REST shim must stop evolving. It is not MCP and will be removed after migration.

---

## 4) Tool Parity Requirements (HTTP = STDIO)

The HTTP transport MUST expose the exact same tools as STDIO after the overhaul:
- `kb.search`, `kb.read_excerpt`, `kb.expand_excerpt`, `kb.extract_evidence`, `kb.retrieve_evidence`
- `graph.describe`, `graph.expand`, `graph.paths`, `graph.parents`, `graph.children`, `graph.entities_for_sections`, `graph.sections_for_entities`
- Compatibility wrappers for legacy tool names (if any are still published)

All tool schemas MUST be identical between transports:
- same defaults (`top_k=5`, `max_per_doc=1`, excerpt caps, etc.)
- same `scope` enforcement (project/environment/doc_tag)
- same budgets, same pagination semantics
- same error taxonomy and envelope

### 4.1 Canonical tool set (repeat for completeness)
HTTP must publish the same “agent-facing” retrieval tools:
- `kb.search`
- `kb.read_excerpt`
- `kb.expand_excerpt`
- `kb.extract_evidence`
- `kb.retrieve_evidence`

and the same projection-only graph tools:
- `graph.describe`
- `graph.expand`
- `graph.paths`
- `graph.parents`
- `graph.children`
- `graph.entities_for_sections`
- `graph.sections_for_entities`

Legacy tool names (if still exposed) must be wrappers that delegate to these tools and only return extra debug fields when explicitly requested.

### 4.2 `search_documentation` policy (HTTP)
- `search_documentation` MUST NOT appear in HTTP `tools/list` by default.
- Any attempt to call it must return a deterministic “disabled tool” error unless an explicit secure enable flag is set.

---

## 5) Scratch + Resources Over HTTP (Non-negotiable)

### 5.1 Resource template availability
HTTP MCP MUST support:
- `resources/templates/list` (shows `wekadocs://scratch/{session_id}/{passage_id}`)
- `resources/read` (reads scratch resources; server validates URIs)
- `resources/list` (optional; template-only is acceptable initially)

### 5.2 “No context pollution” policy
The tool result must not embed large scratch text.
- `kb.search` returns `resource_link` pointers + short previews.
- `kb.read_excerpt` returns bounded excerpts only.
- `kb.retrieve_evidence` returns quotes only (bounded).

This makes HTTP behavior consistent with STDIO and prevents huge intermediate results from being passed through the model.

### 5.3 Resource access controls (HTTP)
Because HTTP is more exposed than STDIO, `resources/read` must validate:
- URI scheme and structure (`wekadocs://scratch/<session>/<passage_id>`)
- session ownership (no cross-session reads)
- scope correctness (`project_id`, `environment`, and doc scoping if present)
- TTL expiry (no stale reads)

Resources must include correct `size` and low `priority` annotations for large blobs to discourage host auto-inclusion.

---

## 6) Output Schema + Structured Content (Fix the current HTTP “json block” approach)

### 6.1 Replace `"type":"json"` with `structuredContent`
All tools must define `outputSchema` and return:
- minimal `content` (usually one short text summary, or empty)
- full result in `structuredContent`

### 6.2 Avoid duplicated JSON in text
The low-level MCP handler will serialize a `dict` into indented JSON text in `content` if tools return structured output only.

Prescriptive requirement:
- Tools MUST return `(content_blocks, structured_dict)` so that `structuredContent` is present and `content` remains small (never a large indented JSON blob).

### 6.3 Eliminate pretty-printed JSON in any fallback text (token-cost regression guard)
If any tool must include JSON in `content.text` for backwards compatibility, it MUST be:
- minified (no indentation),
- byte-capped,
- and clearly marked as “diagnostic only”.

---

## 7) Security + Hardening (HTTP-specific)

### 7.1 Auth and access controls
Decide one of:
- **Option A:** keep HTTP MCP private (network-level access control) + strict URI validation
- **Option B (recommended for future):** add explicit authentication (bearer/OAuth) at the edge (reverse proxy) or via server middleware, and require scopes for tool calls and resource reads

### 7.2 Rate limits and timeouts
For HTTP tool calls:
- enforce per-request timeouts (Neo4j/Qdrant)
- enforce max concurrency for expensive operations
- reject pathological inputs early (huge queries, huge top_k/page_size)

### 7.3 Log hygiene
Never log:
- full `arguments` (contains user content)
- full retrieved text

Log only:
- tool name, request id, budget usage, counts, durations, truncation flags.

### 7.4 CORS and network exposure (prescriptive)
- If the HTTP MCP endpoint is exposed beyond localhost/cluster-internal access:
  - do not enable permissive CORS
  - enforce request size limits at the reverse proxy
  - require authentication (recommended) and scope checks for tool calls and resource reads

---

## 8) Migration Strategy (No regressions)

### 8.1 Feature flags (exact)
- `MCP_HTTP_STREAMABLE_ENABLED` (default false initially)
- `MCP_HTTP_LEGACY_REST_ENABLED` (default true initially; flips later)

### 8.2 Endpoint compatibility timeline
Phase 1:
- Add Streamable HTTP mount at `/_mcp` (new path).
- Keep existing `/mcp/*` REST endpoints unchanged.

Phase 2:
- Update internal clients to use the Streamable HTTP MCP transport.

Phase 3:
- Deprecate `/mcp/*` REST endpoints (warnings + docs).
- Remove them after a full release cycle.

---

## 9) Prescriptive Implementation Phases

### Phase 0 — Contract inventory (0.5–1 day)
1. Identify all clients calling `/mcp/*` today (repo search + logs).
   - repo search: grep/rg for hard-coded `/mcp/` URLs and `tools/call` payload shapes
   - logs: sample access logs for `/mcp/tools/call` usage, tool names, and payload sizes (do not log arguments)
2. Record expected behaviors for:
   - `search_documentation` output shape (if any consumers depend on it),
   - traversal result format.
3. Verify MCP SDK transport APIs in the pinned dependency set:
   - low-level server exists (`mcp.server.lowlevel.server.Server`)
   - Streamable HTTP session manager exists (`mcp.server.streamable_http_manager.StreamableHTTPSessionManager`)
4. Validate host transport compatibility for the in-scope clients:
   - VS Code: Streamable HTTP endpoint reachable and tool listing works
   - LibreChat/OpenWebUI: confirm whether they can consume SSE; if not, plan to enable JSON response mode for their bridge
   - Custom hosts: document required behaviors (JSON-RPC compliance, `structuredContent`, resource reads)

**DoD:** a migration checklist with owners per client.

### Phase 1 — Build shared MCP app factory (1–2 days)
1. Create `src/mcp_server/mcp_app.py`:
   - exports `build_mcp_server() -> mcp.server.lowlevel.server.Server`
   - contains lifespan/deps identical to STDIO
   - registers new `kb.*` + `graph.*` tools and scratch resources
2. Update `src/mcp_server/stdio_server.py` to consume the factory (no behavioral change yet).

**DoD:** STDIO server still runs; tool list identical.

### Phase 2 — Mount Streamable HTTP in FastAPI (1–2 days)
1. In `src/mcp_server/main.py`, mount:
   - `server = build_mcp_server()`
   - `session_manager = StreamableHTTPSessionManager(server, ...)`
   - wire `session_manager.handle_request(...)` at `/_mcp` and run `session_manager.run()` in app lifespan
2. Add a health endpoint that verifies MCP mount is alive (HTTP GET probe).

**DoD:** a compliant MCP client can connect via HTTP to `/_mcp`.

### Phase 3 — Capability parity + resources (1–2 days)
1. Ensure `resources/templates/list` returns the scratch template.
2. Ensure `resources/read` validates session + passage scope.
3. Ensure tool results use resource links rather than embedding big text.

**DoD:** `kb.search` returns resource links; `kb.read_excerpt` works; no big blob outputs.

### Phase 4 — Deprecate legacy REST endpoints (1–2 days + rollout)
1. Add deprecation warnings to `/mcp/*` REST responses.
2. Provide a client migration doc snippet showing the Streamable HTTP base URL.
3. Flip default flags after client migration.

**DoD:** no REST usage in prod; endpoints removable.

---

## 10) Tests and Validation (must be automated)

### 10.1 Protocol-level smoke tests
Add tests that:
- call `tools/list` via Streamable HTTP and assert:
  - tool names include `kb.search` and `kb.retrieve_evidence`
  - `inputSchema` includes `scope`
  - `outputSchema` present for core tools
- call `kb.search` and assert output caps:
  - ≤5 results default
  - preview length ≤ max_snippet_chars

### 10.2 Regression guards
Add contract tests that fail if:
- tool schema drifts between STDIO and HTTP (the factory prevents this, but tests enforce it)
- scratch resources can be read cross-session
- any tool returns embedded full section text by default

### 10.3 Negative tests for `search_documentation` (must-have)
Add tests that assert:
- HTTP `tools/list` does not include `search_documentation` by default.
- Any attempt to call `search_documentation` returns a deterministic “disabled tool” error unless explicitly enabled via a secure test-only flag.

---

## 11) Concrete file list (planned edits)

- `src/mcp_server/main.py`
  - mount Streamable HTTP MCP app
  - keep legacy REST endpoints behind flag
- `src/mcp_server/models.py`
  - mark legacy REST models as deprecated OR remove once no clients depend on them
- `src/mcp_server/mcp_app.py` (new)
  - shared low-level MCP server construction
- `src/mcp_server/stdio_server.py`
  - consume shared factory (and ultimately remove duplicated definitions)
- `src/mcp_server/scratch_store.py` (new; shared with STDIO plan)
- `docs/api-contracts.md`
  - add Streamable HTTP connection contract + tool schemas (additive)
- `docs/mcp/retrieval_playbook.md`
  - confirm HTTP and STDIO parity and recommended `kb.retrieve_evidence` flow

---

## 12) Acceptance Criteria (pass/fail)

1. A compliant MCP client can connect to HTTP MCP via Streamable HTTP at the configured base path.
2. HTTP tool list and schemas exactly match STDIO (same factory).
3. HTTP supports scratch resources (`resources/templates/list`, `resources/read`) and tools return resource links, not embedded large text.
4. No `"type":"json"` content blocks remain in the MCP tool result path.
5. Legacy REST endpoints remain functional until all known clients migrate, then can be removed cleanly.
6. `search_documentation` remains disabled by default and cannot be accidentally exposed over HTTP without an explicit secure flag and non-read-only annotations.

7. Performance contracts preserved:
   - meet or improve the repo’s published latency targets in `docs/api-contracts.md` for equivalent modes over HTTP:
     - `kb.search` (snippet-like), bounded by defaults/caps
     - graph projection tools (graph-like)

8. Multi-client compatibility verified (HTTP):
   - VS Code: Streamable HTTP connection works; namespaced tool names are visible; `structuredContent` is consumable.
   - LibreChat/OpenWebUI: a bridge/client can call tools and parse schemas; if SSE is unsupported, JSON response mode works without breaking MCP semantics.
   - Custom hosts: documented minimum requirements (JSON-RPC, tools/resources support, schema validation) and a smoke-test script or checklist exists.

---

## 13) Explicit migration hazards (must be called out to clients)
1. `traverse_relationships` currently has a schema/implementation mismatch.
   - Treat any schema correction as a **breaking change** for HTTP REST shim consumers.
   - During migration, provide either:
     - a compat layer that accepts both `node_id` and `start_ids` shapes, or
     - a clear client migration guide + versioned endpoint behavior.

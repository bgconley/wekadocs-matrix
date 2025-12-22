# MCP Retrieval Diagnostics Persistence + OTel Plan (Do Not Bloat MCP Payloads)
**Project:** `wekadocs-matrix`
**Date:** 2025-12-17
**Purpose:** Persist and operationalize full-fidelity retrieval/ranking diagnostics (vector/BM25/fusion/rerank/graph) without polluting MCP tool outputs, while enabling human-friendly debugging and OTel export.

---

## 0) Scope and relationship to the other plans (must be implemented together)

This plan is **additive** and must be implemented alongside:

1. STDIO overhaul plan: `docs/plans/mcp_stdio_tool_calls_overhaul_plan_20251217.md`
2. HTTP alignment plan: `docs/plans/mcp_http_endpoint_alignment_plan_20251217.md`

Integration points:
- **STDIO plan** introduces `kb.search` / `kb.retrieve_evidence` and explicitly reduces default tool payload size. This plan provides the *replacement home* for the removed debug metadata.
- **HTTP plan** mandates Streamable HTTP and structured outputs; this plan ensures diagnostics are available via persistent artifacts + OTel correlation instead of inline payloads.

**Non-goal:** changing tool behavior or re-enabling `search_documentation`. Diagnostics are read-only and should not mutate the KB.

---

## 1) Problem statement

`search_sections` currently returns large per-result metadata (scores, RRF contributions, entity metadata, graph signals). This:
- increases token cost (tool outputs are injected into the LLM context),
- confuses tool selection (model sees noisy fields instead of answer-shaped previews),
- and makes it hard to debug (debug data is ephemeral and scattered across transcripts).

We need:
1. **Persistent**, query-addressable debug artifacts (for humans).
2. **Safe**, scope-respecting storage (no sensitive leakage).
3. **Low-cardinality** OTel export (for monitoring) with a clean link from trace/log → artifact.
4. A way to retrieve and inspect diagnostics **without** embedding them into the LLM conversation.

---

## 2) Design principles (non-negotiable)

1. **MCP payloads stay small by default.**
   - MCP tools return a `diagnostic_id`/URI pointer, not a giant debug blob.
2. **Artifacts are durable and human-readable.**
   - JSONL for ingestion/searchability + Markdown summary for quick inspection.
3. **OTel is for correlation and aggregates, not for full per-candidate dumps.**
   - full fidelity goes to artifacts/log storage; OTel spans/logs only carry summaries + pointers.
4. **Hardened by default:**
   - explicit redaction policy,
   - explicit sampling policy,
   - explicit retention policy,
   - explicit access control when exposed via resources.

---

## 3) What to log (diagnostic artifact schema)

### 3.1 Artifact types
We will emit two artifact files per retrieval call:

1. **Primary artifact (JSONL record)**: machine-ingestable, full fidelity.
2. **Human summary (Markdown)**: condensed, visually scannable.

### 3.2 Canonical identifiers (must be consistent everywhere)
- `diagnostic_id`: globally unique ID (use the same value as `request_id`).
- `session_id`: MCP session id (from STDIO/HTTP session context).
- `trace_id`/`span_id`: from OTel context if present.

### 3.3 JSONL record: `RetrievalDiagnosticRecord`
Each JSONL line is one complete record (no multi-line JSON). Minimum fields:

```json
{
  "diagnostic_id": "uuid",
  "timestamp": "ISO-8601",
  "transport": "stdio|http",
  "tool": "kb.search|kb.retrieve_evidence|search_sections",
  "session_id": "string",
  "correlation_id": "string|null",
  "otel": {"trace_id": "hex|null", "span_id": "hex|null"},

  "scope": {
    "project_id": "wekadocs-matrix",
    "environment": "development|prod|...",
    "doc_tag": "REGPACK-08|null",
    "snapshot_scope": "string|null",
    "embedding_version": "string|null"
  },

  "query": {
    "raw": "string (optional redacted)",
    "normalized": "string (optional redacted)",
    "rewritten": {"applied": true, "result": "string", "reason": "keyword_stuffed|..."}
  },

  "config": {
    "neo4j_disabled": false,
    "hybrid_enabled": true,
    "reranker": {"enabled": true, "provider": "bge-reranker-service", "model": "Qwen/Qwen3-Reranker-4B"},
    "colbert": {"enabled": true, "scoring": "maxsim_sum|..."},
    "fusion": {"method": "rrf", "params": {"k": 60}}
  },

  "timing_ms": {
    "bm25": 0.0,
    "vector_search": 0.0,
    "fusion": 0.0,
    "rerank": 0.0,
    "graph_expansion": 0.0,
    "context_assembly": 0.0,
    "total": 0.0
  },

  "counts": {
    "candidates_initial": 0,
    "candidates_post_filter": 0,
    "candidates_post_dedupe": 0,
    "returned": 0,
    "dropped": {"dedupe": 0, "veto": 0, "scope_mismatch": 0}
  },

  "results": [
    {
      "rank": 1,
      "chunk_id": "string",
      "doc_tag": "string|null",
      "token_count": 0,
      "source": "reranked|rrf_fusion|graph_expanded|vector|bm25|hybrid",
      "scores": {
        "bm25": 0.0,
        "vector": 0.0,
        "fused": 0.0,
        "rerank": 0.0,
        "graph": 0.0
      },
      "explain": {
        "rrf_field_contributions": {"field": 0.0},
        "entity_boost_applied": false,
        "entity_metadata": {"...": "..."},
        "graph_distance": 0,
        "graph_path": ["..."]
      }
    }
  ],

  "budgets": {
    "response_bytes": 0,
    "tokens_estimate": 0,
    "partial": false,
    "limit_reason": "none|token_cap|byte_cap|page_size"
  }
}
```

### 3.4 Markdown summary format
Write a one-file-per-diagnostic summary:
- header: ids + timestamps + scope + config snapshot + timings
- a compact table for the top N (default 10):
  - rank, title/chunk_id, doc_tag, token_count, final_score, rerank_score, source
- a short “drops” section (dedupe/veto/scope mismatch)

**Template (prescriptive, stable):**
```md
# Retrieval Diagnostics — {diagnostic_id}

- timestamp: {iso}
- transport: {stdio|http}
- tool: {tool_name}
- session_id: {session_id}
- trace: {trace_id}/{span_id}
- scope: project_id={...} env={...} doc_tag={...}

## Timings (ms)
- bm25: {..}
- vector_search: {..}
- fusion: {..}
- rerank: {..}
- graph_expansion: {..}
- total: {..}

## Counts
- candidates_initial: {..}
- candidates_post_filter: {..}
- candidates_post_dedupe: {..}
- returned: {..}
- dropped: dedupe={..} veto={..} scope_mismatch={..}

## Top Results
| rank | chunk_id | doc_tag | source | fused | rerank | graph | token_count |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... | ... | ... | ... |

## Budgets
- response_bytes: {..}
- tokens_estimate: {..}
- partial: {true|false}
- limit_reason: {none|token_cap|byte_cap|page_size}
```

---

## 4) Where to persist it (durable storage)

### 4.1 Default storage: local filesystem (volume-mounted)
Primary location (inside container/app):
- `/app/reports/retrieval_diagnostics/YYYY-MM-DD/retrieval_diagnostics.jsonl`
- `/app/reports/retrieval_diagnostics/YYYY-MM-DD/<diagnostic_id>.md`

Repo-local development location (host):
- `reports/retrieval_diagnostics/YYYY-MM-DD/...`

**Docker requirement:** add a read-write volume mount:
- `./reports:/app/reports`

Verification step (must-do):
- Confirm `docker-compose.yml` (or deployment manifests) actually mount `./reports:/app/reports` for the MCP server container; otherwise diagnostics will be lost on restart.

### 4.2 Retention policy (prescriptive)
Default:
- JSONL retained 14 days
- Markdown retained 14 days

Make retention configurable:
- `RETRIEVAL_DIAGNOSTICS_RETENTION_DAYS=14`

Retention enforcement (prescriptive):
- Implement an **on-start cleanup** routine that deletes diagnostic artifacts older than `RETRIEVAL_DIAGNOSTICS_RETENTION_DAYS`.
- Cleanup must be best-effort and must never block tool calls (time-bound the cleanup).

### 4.3 Sampling policy (prescriptive)
Default:
- log artifacts at `1%` sample rate
- always log when:
  - `RETRIEVAL_DIAGNOSTICS_FORCE=1`, or
  - `diagnostic_debug=true` argument (explicitly set by an operator), or
  - tool call errors (100% of errors).

Config:
- `RETRIEVAL_DIAGNOSTICS_SAMPLE_RATE=0.01`

### 4.4 Redaction policy (prescriptive)
Default:
- Do not store raw user query text unless explicitly enabled.
- Store:
  - `query_hash` (sha256 of query),
  - `query_length`,
  - optionally the rewritten query if it contains no secrets.

Config:
- `RETRIEVAL_DIAGNOSTICS_STORE_QUERY_TEXT=false` by default.

---

## 5) How MCP tools reference diagnostics (without payload bloat)

### 5.1 Minimal fields returned by retrieval tools
Per `kb.search` / `kb.retrieve_evidence` result, return:
- `diagnostic_id`
- `diagnostic_hint` (short string): “See retrieval diagnostics <id>”
- optionally a resource link to a Markdown summary **only** (not the JSONL).

Do not return:
- RRF field contributions
- entity metadata
- graph debug paths
- detailed stage counters

### 5.2 Diagnostic access tools (operator-only)
Expose a **separate** operator-facing mechanism (not intended for general LLM selection):

Option A (preferred): MCP resources (read-only) behind a flag
- resource template: `wekadocs://diagnostics/{date}/{diagnostic_id}`
- returns Markdown summary only
- JSONL is intentionally not exposed directly to avoid huge reads and leakage

Option B: CLI helper (works even without MCP clients)
- `scripts/retrieval_diagnostics/show.py --id <diagnostic_id>`
- prints the Markdown summary and/or extracts the JSONL record.

Both options can coexist.

Client compatibility note (non-negotiable):
- Do not assume every host will support reading MCP resources or surfacing them in a UI.
  - Claude Desktop and VS Code generally support MCP resources, but third-party hosts (LibreChat/OpenWebUI) may not without a bridge.
- Therefore Option B (CLI helper) is required for operator workflows and must remain supported even if resources are disabled.

---

## 6) OTel export strategy (safe + low-cardinality)

### 6.1 Traces (span attributes + events)
For each retrieval tool call, create/augment a span:
- attributes (low-cardinality):
  - `retrieval.diagnostic_id`
  - `retrieval.transport`
  - `retrieval.mode` (vector_only|hybrid_local)
  - `retrieval.top_k`
  - `retrieval.returned_count`
  - `retrieval.dedupe_dropped`
  - `retrieval.partial`
  - `retrieval.limit_reason`
  - stage timings (bm25_ms, vector_ms, fusion_ms, rerank_ms, expansion_ms, total_ms)

Add span events for **top 3** candidates only:
- event name: `retrieval.candidate`
- fields: rank, chunk_id (or short hash), source, final_score, rerank_score, doc_tag (if low cardinality)

Hard cap: never emit more than 3 candidate events per call.

### 6.2 Logs (OTLP logs exporter + correlation)
You already have OTLP logs export scaffolding (`src/shared/observability/logging.py`). Extend usage so:
- a single log entry per `diagnostic_id` is emitted at INFO level:
  - includes `diagnostic_id`, file paths, trace_id/span_id, and summary counters.
- do not log the full per-candidate list into OTel logs; that stays in JSONL artifacts.

### 6.3 Filelog tailing via OTel Collector (optional but recommended)
If you run an OTel Collector:
- use a `filelog` receiver to tail:
  - `/app/reports/retrieval_diagnostics/**/*.jsonl`
- parse JSON and export to your log backend (Loki/Elastic/etc.)

This provides searchability (“find all queries where rerank_score < 0”) without dumping into LLM context or span attributes.

---

## 7) Implementation plan (phased, hardened)

### Phase 0 — Baseline + schema lock (0.5–1 day)
1. Define the JSONL schema and Markdown format (sections above).
2. Add a version field to the record: `schema_version: 1`.
3. Add a small “golden diagnostics” corpus in `reports/fixtures/` for tests.

**DoD:** schema documented + fixtures added.

### Phase 1 — Diagnostic emitter library (1–2 days)
Create a new module (suggested):
- `src/shared/observability/retrieval_diagnostics.py`
  - `RetrievalDiagnosticEmitter`
  - handles:
    - sampling
    - redaction
    - JSONL append (async-safe)
    - Markdown summary generation
    - returns `(diagnostic_id, md_path, jsonl_path)`

Async-safety requirement (prescriptive):
- Concurrent tool calls must not interleave JSONL lines or corrupt Markdown writes.
- Use one of:
  - a single async write lock around JSONL append + Markdown write, OR
  - a write queue/worker that serializes filesystem writes.

**DoD:** can write artifacts locally with unit tests.

### Phase 2 — Wire into retrieval pipeline (2–4 days)
Wire the emitter at the correct seam:
- emit after `HybridRetriever.retrieve(...)` produces the ranked list and metrics (so we can log all component scores, including reranker outputs and RRF contributions).

Prescriptive requirements:
- log both “returned results” and (optionally) “pre-dedupe top N” if debug flag enabled.
- always attach `diagnostic_id` to the MCP response **as a pointer only**.

**DoD:** `kb.search` includes `diagnostic_id` without adding payload bloat.

### Phase 3 — OTel correlation (1–2 days)
1. Add/extend spans for retrieval calls (reuse existing exemplars/tracing patterns):
   - add attributes (summary only)
   - add max-3 candidate events
2. Emit one structured log line referencing artifact paths + ids.

**DoD:** traces and logs can be joined to artifacts via `diagnostic_id`.

### Phase 4 — Operator access (1–2 days)
Implement one of:
- MCP resources for Markdown summaries (behind a feature flag), OR
- CLI helper script to fetch by id, OR both.

Security requirements:
- resource URIs must be validated
- do not expose JSONL wholesale via MCP (too big + too sensitive)

**DoD:** humans can easily retrieve the Markdown summary for a given `diagnostic_id`.

### Phase 5 — Container persistence + retention (0.5–1 day)
1. Add `./reports:/app/reports` volume mount (rw) to the MCP container.
2. Add a scheduled retention cleanup (cron-like) or on-start cleanup routine.

**DoD:** artifacts survive container restart; retention enforced.

---

## 8) Test plan (must prevent regressions)

1. **Payload-size regression test:** `kb.search` must not include debug fields by default; enforce max response bytes.
2. **Artifact creation test:** when diagnostics enabled, artifacts exist and are well-formed JSONL + Markdown.
3. **Redaction test:** by default, raw query text must not be persisted.
4. **OTel correlation test:** spans include `retrieval.diagnostic_id` and stage timing attributes.
5. **Cardinality guard test:** no metrics labels include `chunk_id`, `doc_tag`, or `diagnostic_id`.

---

## 9) Acceptance criteria (pass/fail)

1. Full per-candidate debug metadata is persisted (JSONL) and readable (Markdown) for sampled calls.
2. MCP tool responses stay small by default; diagnostics are referenced by pointer only.
3. Operators can locate diagnostics via `diagnostic_id` in logs/traces.
4. OTel contains low-cardinality summaries and a stable pointer to artifacts.
5. No sensitive query content is stored by default; enabling full-text logging is explicit and auditable.

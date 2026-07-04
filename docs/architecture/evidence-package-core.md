# Evidence Package Core

`EvidencePackage` is the canonical retrieval product contract. It carries the
evidence quotes, coverage, gaps, an optional citation-preserving answer draft,
and an internal `retrieval_metrics` telemetry superset that is never sent to
clients.

## Layering

- `src/evidence/` is pure. It must never import `src/mcp_server/`.
- `src/mcp_server/` depends on `src/evidence/`, never the reverse.

## Ownership

- MCP (`mcp_tools.py`) is a transport adapter: it normalizes the request
  (backward compatibility, fetch-depth clamp to `KB_EVIDENCE_MAX_FETCH_K`, and
  graph-enrichment default), calls `EvidenceService`, applies output budgets,
  reconciles the partial state into `coverage` plus a `budget_exceeded` gap,
  emits diagnostics with post-budget values, persists the trace, and serializes.
  It does not own retrieval orchestration, quote scoring, coverage/gap logic,
  trace semantics, or LLM enhancement.
- Trace semantics live in `retrieval_trace.py::record_evidence_package`, which
  rebuilds all nine facets from `package.retrieval_metrics`.

## Invariants

- Default response mode is `evidence_only`.
- `evidence_plus_draft` may add a draft only when every claim cites `quote_id`
  values present in the same package. The `EvidencePackage` validator enforces
  this at construction and on assignment; the service drops any draft that
  violates it and records an `uncited_draft_rejected` gap. LLM enhancement
  cannot launder an uncited final answer.
- Partial/budget state is reported once: top-level `partial`/`limit_reason` and
  `coverage.partial`/`coverage.limit_reason` always agree.

## Out Of Scope

Embedding, sparse (SPLADE), ColBERT, reranker, Qdrant, and Neo4j quality checks
are validated live in a separate pass from this architecture-only work.

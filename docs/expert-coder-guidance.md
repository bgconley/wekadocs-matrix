# 3) `/docs/expert-coder-guidance.md` — **Expert Coder Guidance (v2, canonical)**

**Status:** Canonical v2. Phase/task IDs align 1.1→5.4.

### Guidance pattern (applies to each task)

* **Do this:** concrete steps and priority order.
* **Pitfalls:** frequent errors and how to avoid them.
* **Definition of Done:** objective checks.
* **Acceptance checklist:** what the reviewer will verify.
* **Notes for performance & safety:** limits, timeouts, guardrails.

---

## Phase 1

**1.1 Docker env**

* **Do this:** parameterize service URLs; healthchecks; secrets via Docker/K8s; set Neo4j heap/pagecache once; volumes.
* **Pitfalls:** conflicting memory env vars; missing healthcheck timeouts.
* **DoD:** all services healthy; restart preserves data.
* **Checklist:** `docker ps`, health, persistence, logs rotation.

**1.2 MCP server**

* **Do this:** FastAPI, MCP endpoints, OTel, structured logs, connection pools, graceful shutdown.
* **Pitfalls:** not closing Bolt sessions; missing correlation IDs.
* **DoD:** tools list/call works; traces & metrics present.
* **Checklist:** curl endpoints; View traces in collector.

**1.3 Schema**

* **Do this:** add `Document/Section`; vector indexes from config; `schema_version` node.
* **Pitfalls:** hard‑coding vector dims; label/property drift.
* **DoD:** indexes exist; idempotent re‑runs.
* **Checklist:** `CALL db.indexes()`; re-run script success.

**1.4 Security**

* **Do this:** JWT auth; Redis token bucket; parameterized Cypher; audit everything.
* **Pitfalls:** trusting literals; forgetting timeouts.
* **DoD:** blocked invalid JWT; 429 under burst; audit entries exist.
* **Checklist:** negative tests pass; audit log lines visible.

---

## Phase 2

**2.1 NL→Cypher**

* **Do this:** intent classifier; entity linker; **templates-first**; fallback LLM proposal; normalize & parameterize; inject limits early.
* **Pitfalls:** executing raw LLM output; missing LIMIT at right place.
* **DoD:** 90% via templates; fallback queries pass validator.
* **Checklist:** corpus run; parameter enforcement.

**2.2 Validator**

* **Do this:** correct `*min..max`; parameter enforcement; run `EXPLAIN`; cap scans/expansions/depth; enforce timeout; early LIMITs.
* **Pitfalls:** regex-only validation; appending LIMIT too late.
* **DoD:** malicious queries blocked; legitimate pass; FP <5%.
* **Checklist:** negative suite; plan stats checked.

**2.3 Hybrid search**

* **Do this:** choose vector SoT; top‑K Sections (optional Entities); 1–2 hop expansion; connecting paths; rank by semantic+graph+recency.
* **Pitfalls:** explosive expansions; missing dedupe.
* **DoD:** P95 < 500ms at K=20; bounded hops.
* **Checklist:** load test; top‑K correctness.

**2.4 Response**

* **Do this:** Markdown+JSON; evidence (Section, path); confidence estimate; “Why these results?”; disambiguation.
* **Pitfalls:** missing evidence IDs; unbounded JSON size.
* **DoD:** schema‑valid JSON; confidence ∈ [0,1].
* **Checklist:** E2E snapshots; schema validator.

---

## Phase 3

**3.1 Parser**

* **Do this:** retain anchors, code, tables; compute deterministic `section_id`; token counts.
* **Pitfalls:** losing anchors; non‑deterministic parsing.
* **DoD:** identical output across runs.
* **Checklist:** checksum comparison.

**3.2 Extraction**

* **Do this:** pattern+light NLP; `MENTIONS` with spans + confidence.
* **Pitfalls:** over‑dedupe of near‑aliases.
* **DoD:** >95% precision on commands/configs.
* **Checklist:** labeled evaluation; metrics report.

**3.3 Graph build**

* **Do this:** MERGE IDs; provenance on edges; batch writes; embeddings; vector upsert; set `embedding_version`.
* **Pitfalls:** non‑idempotent merges; drift between graph and vectors.
* **DoD:** re‑ingestion diffs=0; vector parity.
* **Checklist:** counts stable; reconciliation parity.

**3.4 Incremental**

* **Do this:** staged labels; atomic swap; partial re‑embed; nightly reconciliation + repair.
* **Pitfalls:** mass re‑embeddings; label swap leaving orphans.
* **DoD:** O(changed sections) updates; drift <0.5%.
* **Checklist:** controlled delta; drift metric.

---

## Phase 4

**4.1 Complex templates**

* **Do this:** pre‑approve; input/output schemas; plan guardrails.
* **Pitfalls:** unbounded traversals.
* **DoD:** run within depth/time budgets.
* **Checklist:** plan inspection; result shape validated.

**4.2 Optimization**

* **Do this:** analyze slow queries; recommend indexes; rewrite templates; plan caching.
* **Pitfalls:** adding indexes that fight each other.
* **DoD:** measurable P95 improvement.
* **Checklist:** before/after perf CSV.

**4.3 Caching**

* **Do this:** L1+L2; version‑prefixed keys; warmers; optional materialization.
* **Pitfalls:** stale caches after model/schema change.
* **DoD:** hit >80%; correctness post‑rotation.
* **Checklist:** rotate `embedding_version`; verify.

**4.4 Learning**

* **Do this:** collect feedback; tune ranker; propose templates/indexes.
* **Pitfalls:** training on noisy feedback without guardrails.
* **DoD:** uplift (NDCG) on held‑out set.
* **Checklist:** offline eval; A/B flags.

---

## Phase 5

**5.1 External**

* **Do this:** connectors; webhooks/poll; queue; circuit breakers.
* **Pitfalls:** token scopes; API rate limits.
* **DoD:** steady ingestion; degraded mode OK.
* **Checklist:** throttle tests; backoff observed.

**5.2 Monitoring**

* **Do this:** Prom/Grafana; OTel; alerts; runbooks.
* **Pitfalls:** high‑cardinality labels; missing exemplars.
* **DoD:** on‑call diagnoses in <10m.
* **Checklist:** alert drill passes.

**5.3 Testing framework**

* **Do this:** **no‑mocks** unit/integration/E2E/perf/security/chaos; determinism checks; CI gates.
* **Pitfalls:** accidental mocking via fixtures.
* **DoD:** CI green; reproducible tests on live stack.
* **Checklist:** artifacts in `/reports/phase-5`.

**5.4 Production**

* **Do this:** blue/green, canary; backups; DR drills.
* **Pitfalls:** partial rollbacks leaving schema mismatches.
* **DoD:** canary 5% for 1h; DR RTO 1h, RPO 15m.
* **Checklist:** rollback & DR drill evidence.

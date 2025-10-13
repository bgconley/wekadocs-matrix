# 2) `/docs/implementation-plan.md` — **Implementation Plan (v2, canonical)**

**Status:** Canonical v2. Phase/task IDs match spec and guidance exactly.
**Rule:** Before exiting a phase, produce **no‑mocks** test results in `/reports/phase-N/` and pass the **Phase Gate**.

### Conventions

* **Owner**, **Deps**, **Deliverables**, **DoD** (definition of done), **Tests (NO MOCKS)**, **Artifacts**, **Gate**.
* All commands run from repo root unless noted.
* Configurable via `config/*.yaml`.

---

## Phase 1 – Core Infrastructure

### **Task 1.1 – Docker environment setup**

**Owner:** Platform | **Deps:** —
**Steps:** Compose services (`mcp-server`, `neo4j`, `qdrant`, `redis`, `ingestion-worker`, optional `nginx`); CPU/mem limits; healthchecks with realistic timeouts; volumes; secrets via Docker/K8s secrets; `weka-net` network.
**Deliverables:** `docker-compose.yml`, `.env.example`, `docker/*` Dockerfiles.
**DoD:** `docker compose up -d` → all healthy; restart preserves data; secrets not visible via `docker inspect`.
**Tests (NO MOCKS):**

* Spin stack; hit `/health`, `/ready`; Redis `PING`; Qdrant `/health` (if enabled); Bolt connect.
* Restart containers; validate data persistence.
  **Artifacts:** `/reports/phase-1/junit.xml`, `/reports/phase-1/summary.json` (schema below).
  **Gate:** P1 infra green per tests; artifacts present.

---

### **Task 1.2 – MCP server foundation**

**Owner:** Backend | **Deps:** 1.1
**Steps:** FastAPI app; MCP endpoints (`initialize`, `tools/list`, `tools/call`, `completion`); connection pools; graceful shutdown; OTel middleware; structured logs with correlation IDs; `/metrics`.
**Deliverables:** `src/mcp_server/main.py`, `src/mcp_server/tools/*`, OTel config.
**DoD:** `tools/list` shows tools; `tools/call` executes; traces appear; metrics scrape OK.
**Tests (NO MOCKS):** Call endpoints against running stack; assert JSON shape, tool outputs; verify traces & metrics via scrape.
**Artifacts:** `/reports/phase-1/*`.
**Gate:** Included in P1 gate.

---

### **Task 1.3 – Database schema initialization**

**Owner:** Graph Eng | **Deps:** 1.1
**Steps:** Create **Document/Section**; constraints/indexes; vector indexes with **config dims/similarity**; `schema_version` node.
**Deliverables:** `scripts/neo4j/create_schema.cypher`, `src/shared/schema.py`.
**DoD:** `CALL db.indexes()` shows property + vector indexes; re‑run scripts idempotent.
**Tests (NO MOCKS):** Apply schema to live Neo4j; re‑apply; query `db.indexes`; attempt writes using expected labels and properties.
**Artifacts:** `/reports/phase-1/*`.
**Gate:** Included in P1 gate.

---

### **Task 1.4 – Security layer**

**Owner:** Platform+Backend | **Deps:** 1.2
**Steps:** JWT auth; Redis token bucket; audit log store; parameterized Cypher only; validator interface ready for P2.
**Deliverables:** auth middleware; rate limiter; audit sink.
**DoD:** Auth required on `/mcp`; rate limits trigger under load; audit entries recorded with correlation IDs.
**Tests (NO MOCKS):** Burst requests to trigger 429; invalid JWT denied; audit log entries verified via DB or log scraping.
**Artifacts:** `/reports/phase-1/*`.
**Gate:** Included in P1 gate.

---

## Phase 2 – Query Processing Engine

### **Task 2.1 – NL → Cypher translation**

**Owner:** Retrieval | **Deps:** 1.2, 1.3
**Steps:** Intent classifier; entity linker; **template library** for known intents; optional LLM‑proposal path; normalization & parameterization; hard caps injected early.
**Deliverables:** `src/query/planner.py`, `src/query/templates/*.cypher`.
**DoD:** ≥90% user stories resolved via templates; fallback proposals pass validator.
**Tests (NO MOCKS):** For a corpus of real prompts, ensure produced queries are parameterized, limited, depth‑bounded; run against live Neo4j test data; verify outputs.
**Artifacts:** `/reports/phase-2/*`.
**Gate:** Included in P2 gate.

---

### **Task 2.2 – Cypher validation system**

**Owner:** Graph Eng | **Deps:** 2.1
**Steps:** Regex guardrails; fix variable‑length pattern; parameter enforcement; run **`EXPLAIN`**; reject plans exceeding thresholds (label scans, expansions, depth); enforce server timeouts.
**Deliverables:** `src/mcp_server/validation.py`.
**DoD:** Injection/expensive patterns blocked; false positives < 5%.
**Tests (NO MOCKS):** Execute malicious & deep queries against live DB; assert 4xx/blocked; verify legitimate templates pass and return results.
**Artifacts:** `/reports/phase-2/*`.
**Gate:** Included in P2 gate.

---

### **Task 2.3 – Hybrid search**

**Owner:** Retrieval | **Deps:** 1.3, 2.2
**Steps:** Choose **primary vector store** (`qdrant | neo4j`); implement vector top‑K (Sections + optional Entities) → controlled 1–2 hop expansion → optional connecting paths; ranker blends semantic, graph, recency.
**Deliverables:** `src/query/hybrid_search.py`, `src/query/ranking.py`.
**DoD:** P95 < 500ms at K=20 (warmed); deterministic ranking ties.
**Tests (NO MOCKS):** Seed real vectors; run queries under Locust/k6; capture latency percentiles; verify expansions bounded by config.
**Artifacts:** `/reports/phase-2/*`.
**Gate:** Included in P2 gate.

---

### **Task 2.4 – Response generation**

**Owner:** Backend | **Deps:** 2.3
**Steps:** Build human Markdown + JSON (`answer, evidence[{section_id, path}], confidence, diagnostics`); “Why these results?” reveals ranking features; disambiguation for homonyms.
**Deliverables:** `src/query/response_builder.py`.
**DoD:** Answers include evidence & confidence; JSON schema validated.
**Tests (NO MOCKS):** E2E queries against live stack; assertions on JSON schema, evidence paths, confidence bounds.
**Artifacts:** `/reports/phase-2/*`.
**Gate:** Included in P2 gate.

---

## Phase 3 – Ingestion Pipeline

### **Task 3.1 – Multi‑format parser**

**Owner:** Ingestion | **Deps:** 1.3
**Steps:** Parse Markdown/HTML/Notion → **Document/Section**; preserve anchors, code, tables; compute `section_checksum`, `section_id`; store token counts.
**Deliverables:** `src/ingestion/parsers/{markdown,html,notion}.py`.
**DoD:** Deterministic sections; rerun yields identical IDs.
**Tests (NO MOCKS):** Feed real docs; compare IDs/checksums across runs; ensure anchors preserved.
**Artifacts:** `/reports/phase-3/*`.
**Gate:** Included in P3 gate.

---

### **Task 3.2 – Entity extraction**

**Owner:** Ingestion | **Deps:** 3.1
**Steps:** Pattern + light NLP to extract `Command/Configuration/Procedure/Error/Concept/Example/Step/Parameter`; write `MENTIONS` with spans + confidence (no heavy inference yet).
**Deliverables:** `src/ingestion/extract/*`.
**DoD:** >95% precision on commands/configs against a labeled sample.
**Tests (NO MOCKS):** Run against labeled real docs; compute precision/recall; store metrics file.
**Artifacts:** `/reports/phase-3/*`.
**Gate:** Included in P3 gate.

---

### **Task 3.3 – Graph construction**

**Owner:** Graph Eng | **Deps:** 3.2
**Steps:** MERGE by deterministic IDs; set provenance & timestamps; batch with `apoc.periodic.iterate` and tx timeouts; compute embeddings; upsert to **primary vector store**; set `embedding_version`.
**Deliverables:** `src/ingestion/build_graph.py`.
**DoD:** Idempotent; re‑ingestion of unchanged docs results in no diffs.
**Tests (NO MOCKS):** Ingest same doc twice; assert node/edge counts stable; verify vector count parity with graph.
**Artifacts:** `/reports/phase-3/*`.
**Gate:** Included in P3 gate.

---

### **Task 3.4 – Incremental update**

**Owner:** Ingestion | **Deps:** 3.3
**Steps:** Diff via `section_checksum`; stage `:Section_Staged`; atomic relabel swap; re‑embed changed & adjacent only; nightly reconciliation & repair.
**Deliverables:** `src/ingestion/incremental.py`, `src/ingestion/reconcile.py`.
**DoD:** Small edits update O(changed sections); drift <0.5% overnight.
**Tests (NO MOCKS):** Edit one section; verify only minimal graph/vector delta; run reconciliation and confirm parity.
**Artifacts:** `/reports/phase-3/*`.
**Gate:** Included in P3 gate.

---

## Phase 4 – Advanced Query Features

### **Task 4.1 – Complex query templates**

**Owner:** Retrieval | **Deps:** 2.x
**Steps:** Pre‑approved templates: dependency chain, impact analysis, troubleshooting path, comparison, temporal “as of version Y”; define input/output schemas & plan guardrails.
**Deliverables:** `src/query/templates/advanced/*.cypher`.
**DoD:** Templates execute within depth/time budgets; validated.
**Tests (NO MOCKS):** Execute each template on live graph; assert outputs, plan bounds, runtime.
**Artifacts:** `/reports/phase-4/*`.
**Gate:** Included in P4 gate.

---

### **Task 4.2 – Query optimization**

**Owner:** Graph Eng | **Deps:** 4.1
**Steps:** Slow‑query analysis; index recommendations; query rewriting; compiled plan cache for hot templates.
**Deliverables:** `src/ops/optimizer.py`, dashboards.
**DoD:** Documented P95 improvement on hot paths.
**Tests (NO MOCKS):** Before/after benchmarks with Locust/k6; attach comparison CSV.
**Artifacts:** `/reports/phase-4/*`.
**Gate:** Included in P4 gate.

---

### **Task 4.3 – Caching & performance**

**Owner:** Backend | **Deps:** 2.x
**Steps:** L1 in‑proc + L2 Redis; cache keys prefixed with `{schema_version}:{embedding_version}`; daily warmers; optional materialization of expensive patterns.
**Deliverables:** `src/shared/cache.py`, warmers.
**DoD:** >80% hit rate steady; correctness under model/schema rotation.
**Tests (NO MOCKS):** Rotate embedding version; verify cache invalidation & correctness; measure hit rate over load.
**Artifacts:** `/reports/phase-4/*`.
**Gate:** Included in P4 gate.

---

### **Task 4.4 – Learning & adaptation**

**Owner:** Retrieval | **Deps:** 2–4
**Steps:** Log query→result→feedback; update ranking weights; propose new templates & indexes.
**Deliverables:** `src/learning/*`.
**DoD:** Relevance lift (NDCG) on held‑out set.
**Tests (NO MOCKS):** Offline evaluation from real logs; attach metrics plots/CSV.
**Artifacts:** `/reports/phase-4/*`.
**Gate:** Included in P4 gate.

---

## Phase 5 – Integration & Deployment

### **Task 5.1 – External systems**

**Owner:** Platform | **Deps:** 3.x
**Steps:** Notion/GitHub/Confluence connectors; webhooks or polling; queue ingestion; circuit breakers; Slack notifications.
**Deliverables:** `src/connectors/*`, runbooks.
**DoD:** Steady ingestion under rate limits; degraded mode works.
**Tests (NO MOCKS):** Use demo workspaces/repos; ingest deltas end‑to‑end; throttle to trigger backoff.
**Artifacts:** `/reports/phase-5/*`.
**Gate:** Included in P5 gate.

---

### **Task 5.2 – Monitoring & observability**

**Owner:** SRE | **Deps:** 1.2
**Steps:** Prometheus exporters; Grafana dashboards; OTel traces; alerts (P99, errors, drift, OOM).
**Deliverables:** `deploy/monitoring/*`, runbooks.
**DoD:** On‑call can diagnose slow/failed queries in <10 min.
**Tests (NO MOCKS):** Fire alerts via synthetic load; capture traces and screenshots.
**Artifacts:** `/reports/phase-5/*`.
**Gate:** Included in P5 gate.

---

### **Task 5.3 – Testing framework**

**Owner:** QA | **Deps:** all
**Steps:** **No‑mocks** unit/integration/E2E/perf/security/chaos; golden graph determinism; CI gates.
**Deliverables:** `tests/*`, CI workflow, chaos scenarios.
**DoD:** CI blocks on any determinism or evidence regression.
**Tests (NO MOCKS):** Entire matrix against live stack from compose; chaos: kill vector/slow Neo4j; verify degraded behavior.
**Artifacts:** `/reports/phase-5/*`.
**Gate:** Included in P5 gate.

---

### **Task 5.4 – Production deployment**

**Owner:** Platform/SRE | **Deps:** 5.2, 5.3
**Steps:** Blue/green + canary; feature flags; backups; DR drills (RTO 1h, RPO 15m).
**Deliverables:** K8s manifests/Helm, CI/CD, runbooks.
**DoD:** Canary 5% for 1h, auto‑rollback on SLI breach; DR drill passes.
**Tests (NO MOCKS):** Stage→Prod canary; forced rollback; documented DR restore.
**Artifacts:** `/reports/phase-5/*`.
**Gate:** Launch.

---

### Test artifact schema (applies to all phases)

`/reports/phase-N/summary.json`:

```json
{
  "phase": "N",
  "date_utc": "2025-10-12T00:00:00Z",
  "commit": "<git-sha>",
  "results": [
    {"task": "1.1", "name": "stack_healthy", "status": "pass", "duration_ms": 4123},
    {"task": "1.2", "name": "mcp_tools_list", "status": "pass", "duration_ms": 203}
  ],
  "metrics": {
    "latency_ms_p50": 120,
    "latency_ms_p95": 480,
    "cache_hit_rate": 0.83,
    "reconciliation_drift_pct": 0.2
  },
  "artifacts": [
    "junit.xml",
    "logs.tar.gz",
    "perf.csv"
  ]
}
```

`/reports/phase-N/junit.xml`: standard JUnit XML for CI.

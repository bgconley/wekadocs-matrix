Below are **five copy‑paste‑ready prompts**—one for each phase. Give each prompt to your agentic coder at the start of that phase. They are **self‑contained**, reference only the **v2** materials, and enforce the **phase gates**, **no‑mocks test policy**, and **artifact outputs** you requested. The task numbers (1.1→5.4) match your v2 spec/plan/pseudocode and the earlier expert guidance.

---

## 🔶 Phase 1 Prompt — Core Infrastructure (Tasks **1.1 → 1.4**)

**You are the agentic coder for Phase 1.** Your mission is to stand up a production‑shaped local stack and a minimal MCP server skeleton with security and schema creation, then **prove it works** using **no‑mocks** tests and machine‑readable reports.

### Context & constraints (canonical v2)

* Stack: **FastAPI MCP server**, **Neo4j 5.x**, **Qdrant (optional primary vector store)**, **Redis**, **ingestion worker**; containerized via **Docker Compose** on a single private bridge network.
* Security **now**, not later: **JWT auth**, **Redis token bucket rate‑limits**, **audit log** with correlation IDs.
* Schema: **Document/Section** nodes, domain nodes (Command/Configuration/Procedure/Error/Concept/Example/Step/Parameter/Component), **vector indexes created from config** (no hard‑coded dims), and a `schema_version` singleton.
* **No mocks**. All tests must hit the real services started by Compose.

### Deliverables

1. **Compose & Dockerfiles**
   `docker-compose.yml`, `.env.example`, `docker/*` with healthchecks, CPU/mem limits, volumes, secrets (no plaintext credentials).
2. **MCP server skeleton**
   `src/mcp_server/main.py` with endpoints `/mcp`, `/health`, `/ready`, `/metrics`; tool registry scaffold; connection pools; graceful shutdown; **OpenTelemetry** tracing & structured logs.
3. **Security layer**
   JWT verification middleware, Redis token bucket rate‑limiter, audit logger writing request method, client, correlation ID, and result (success/error).
4. **Schema creation**
   `scripts/neo4j/create_schema.cypher` and `src/shared/schema.py` that create **Document/Section** (with constraints), domain nodes (constraints/indexes), **config‑driven vector indexes**, and write a `schema_version` node.
5. **Config**
   `config/development.yaml` with `embedding.model_name`, `embedding.dims`, `embedding.similarity`, `search.vector.primary ∈ {qdrant, neo4j}`.

### Tests (NO MOCKS) & artifacts

* Put tests under `tests/` with names prefixed `p1_*.py`.
* Start the stack in tests (compose or pre‑started), then assert:

  * **Health**: `/health`, `/ready` return OK; Redis `PING`; Qdrant `/health` (if enabled); **Neo4j Bolt** connect.
  * **Security**: invalid JWT → 401; burst requests → 429; audit entries logged.
  * **Schema**: run schema script twice (idempotent), verify constraints/indexes exist; vector index dims match config.
* Emit:

  * **JUnit XML** → `reports/phase-1/junit.xml`
  * **Summary JSON** → `reports/phase-1/summary.json` (include pass/fail per test and key metrics)
* Provide a tiny helper (`scripts/test/run_phase.sh`) to run only Phase 1 tests and write artifacts.

### Exit gate (must pass before Phase 2)

* All Phase‑1 tests **green**, artifacts present.
* Security enforced (401/429) and audit lines observed.
* Schema idempotent; vector indexes use config dims; OTel traces visible.

### Work plan (do in this order)

1. Compose & healthchecks → 2. MCP skeleton + OTel → 3. JWT + rate‑limit + audit → 4. Schema scripts (config‑driven vectors) → 5. Tests & reports.

---

## 🔷 Phase 2 Prompt — Query Processing Engine (Tasks **2.1 → 2.4**)

**You are the agentic coder for Phase 2.** Your mission is to turn NL queries into **safe, parameterized Cypher**, implement **plan‑gated validation**, build **hybrid retrieval**, and produce **structured responses** with evidence and confidence—then **prove** safety and performance with **no‑mocks** tests and artifacts.

### Context & constraints (canonical v2)

* **Templates‑first** NL→Cypher; LLM proposal allowed **only as proposer**, never executed raw.
* **Validator** combines static guards with **Neo4j `EXPLAIN` plan checks**; blocks excessive scans/expansions/depth; enforces **timeouts**/**LIMITs** early.
* **Hybrid search**: vector top‑K (Sections + optional Entities) → bounded 1–2 hop graph expansion → optional connecting paths → rank and dedupe.
* Responses are dual‑form: **Markdown** + **JSON** `{answer, evidence[{section_id, path}], confidence, diagnostics{ranking_features}}`.

### Deliverables

1. **NL→Cypher Planner** (`src/query/planner.py`, `src/query/templates/*.cypher`)

   * Intent classifier; entity linker; parameterization; inject `LIMIT` and depth caps early.
2. **Cypher Validator** (`src/mcp_server/validation.py`)

   * Regex guards, **correct `*min..max` pattern**, parameter enforcement, **`EXPLAIN` plan gate**, timeout application.
3. **Hybrid Search** (`src/query/hybrid_search.py`, `src/query/ranking.py`)

   * Respect `search.vector.primary`; seed from vectors; expand H=1..2 typed edges; ranking blends semantic score, path proximity, recency; deterministic tie‑break.
4. **Response Builder** (`src/query/response_builder.py`)

   * Render Markdown; emit JSON with evidence & confidence; “**Why these results?**” diagnostics.

### Tests (NO MOCKS) & artifacts

* Name tests `p2_*.py`. **Seed a real mini‑graph** for tests via a deterministic script (no mocks):
  `scripts/dev/seed_minimal_graph.py` → Creates ~10 Sections + a handful of Commands/Procedures/Errors, sets embeddings (use local model), and (if Qdrant is primary) inserts vectors with payload.
* Required test groups:

  * **Validator negative**: injection attempts, deep traversals, Cartesian products → **blocked with helpful errors**.
  * **Templates happy path**: common intents map to parameterized templates; queries pass validator and return results.
  * **Hybrid performance**: with the seeded graph, warmed caches → P95 < **500ms** at K=20 (measure and record).
  * **Response schema**: JSON contains evidence Section IDs, optional paths, confidence ∈ [0,1].
* Emit:

  * `reports/phase-2/junit.xml`
  * `reports/phase-2/summary.json` (include latency percentiles, block/allow counts, schema validations)

### Exit gate (must pass before Phase 3)

* Validator blocks attacks; **false positives < 5%** on provided positive set.
* Hybrid search P95 < **500ms** (warmed).
* Responses carry evidence & confidence; artifacts present.

### Work plan

1. Templates + planner → 2. Validator (regex + `EXPLAIN`) → 3. Hybrid search + ranking → 4. Response builder → 5. Seed script + tests + reports.

---

## 🟩 Phase 3 Prompt — Ingestion Pipeline (Tasks **3.1 → 3.4**)

**You are the agentic coder for Phase 3.** Your mission is to implement a **deterministic, idempotent ingestion pipeline** from Markdown/HTML/Notion into **Document → Section → Entities** with provenance and vectors, plus **incremental updates** and **reconciliation**—then prove determinism and drift control with **no‑mocks** tests.

### Context & constraints (canonical v2)

* **Document** and **Section** are first‑class; `Section.id := hash(source_uri + anchor + normalized_text)`.
* **MENTIONS** edges from Section to Entities capture `confidence`, `start`, `end`, and `source_section_id`.
* **Primary vector store**: configurable (Qdrant **or** Neo4j vectors). **Dual‑write** possible but behind a flag.
* Inference edges (REQUIRES/AFFECTS/RESOLVES/etc.) can happen later; first pass focuses on MENTIONS + provenance.

### Deliverables

1. **Parsers** (`src/ingestion/parsers/{markdown,html,notion}.py`)

   * Preserve headings/anchors, code fences, tables; compute `section_checksum`, token counts; deterministic `section_id`.
2. **Entity extraction** (`src/ingestion/extract/*`)

   * Rules + light NLP for Commands/Configurations/Procedures/Errors/Concepts/Examples/Steps/Parameters; create **MENTIONS** with spans/confidence.
3. **Graph builder** (`src/ingestion/build_graph.py`)

   * MERGE by deterministic IDs; provenance on edges; batch via `apoc.periodic.iterate`; timeouts; embeddings; vector upsert; set `embedding_version`.
4. **Incremental & reconciliation** (`src/ingestion/{incremental.py,reconcile.py}`)

   * Diff on `section_checksum`; stage with `:Section_Staged` then atomic relabel swap; re‑embed changed & adjacent; nightly reconciliation (graph vs vectors) with repair.

### Tests (NO MOCKS) & artifacts

* Name tests `p3_*.py`. Use **real sample docs** in `data/samples/` (at least 3 Markdown + 1 HTML). Notion tests may target a sandbox workspace or be skipped behind a flag if credentials unavailable.
* Required test groups:

  * **Determinism**: re‑ingest same docs twice → **no diffs** in node/edge counts and ID sets.
  * **Provenance**: MENTIONS edges have source Section IDs and spans.
  * **Vectors parity**: count(Sections with embedding_version) == count(vectors in primary store).
  * **Incremental**: edit one Section; only that Section (and immediately dependent items) update; re‑embed minimal set.
  * **Reconciliation**: simulate vector deletion for N sections; nightly job repairs to zero drift.
* Emit:

  * `reports/phase-3/junit.xml`
  * `reports/phase-3/summary.json` (include determinism hashes, drift %, changed counts for incremental)

### Exit gate (must pass before Phase 4)

* Deterministic ingestion (repeat runs stable).
* Incremental updates limited to changed sections.
* Vector/graph **drift < 0.5%**; artifacts present.

### Work plan

1. Parsers → 2. Extractors → 3. Graph builder + embeddings → 4. Incremental + reconciliation → 5. Tests + reports.

---

## 🟦 Phase 4 Prompt — Advanced Query Features (Tasks **4.1 → 4.4**)

**You are the agentic coder for Phase 4.** Your mission is to deliver **complex query templates**, **optimizer**, **caching**, and a simple **learning loop**—and prove performance uplift and cache correctness with **no‑mocks** tests.

### Context & constraints (canonical v2)

* Complex templates: **dependency chains**, **impact analysis**, **troubleshooting paths**, **temporal “as of version Y”**, **comparisons**; **pre‑approved** and **plan‑gated**.
* Optimizer: analyze slow queries; recommend/create indexes; support **compiled plan cache** for hot patterns.
* Caching: L1 in‑proc + L2 Redis, **keys prefixed with `{schema_version}:{embedding_version}`**; optional materialization for expensive patterns.
* Learning: log query→result→feedback, adjust ranking weights; propose new templates/indexes.

### Deliverables

1. **Advanced templates** (`src/query/templates/advanced/*.cypher`) with input/output schemas and plan guardrails.
2. **Optimizer** (`src/ops/optimizer.py`) to analyze plans, suggest indexes/hints, and cache compiled plans.
3. **Caching** (`src/shared/cache.py`, `src/ops/warmers/`) with version‑prefixed keys; daily warmers/materializers.
4. **Learning loop** (`src/learning/*`) to collect feedback and tune ranking weights; emit suggestions.

### Tests (NO MOCKS) & artifacts

* Name tests `p4_*.py`. Populate graph with Phase‑3 data.
* Required test groups:

  * **Templates**: each advanced template runs within configured depth/time budgets; outputs match schema.
  * **Optimization uplift**: run before/after perf suite; demonstrate statistically significant P95 improvement on hot queries; export CSV of timings.
  * **Cache correctness**: rotate `embedding_version` and `schema_version`; assert caches invalidate; measure steady‑state hit rate > **80%** under load.
  * **Learning**: offline evaluation updates ranking weights and improves held‑out NDCG.
* Emit:

  * `reports/phase-4/junit.xml`
  * `reports/phase-4/summary.json` (include perf deltas, cache hit rate, NDCG lift)
  * Attach `perf_before.csv` / `perf_after.csv` if produced.

### Exit gate (must pass before Phase 5)

* Advanced templates pass guardrails and tests.
* Cache hit rate > **80%** steady‑state; **before/after** perf shows uplift.
* Learning loop demonstrates offline improvement; artifacts present.

### Work plan

1. Advanced templates → 2. Optimizer & plan cache → 3. Caching & warmers → 4. Learning loop → 5. Perf/cache tests + reports.

---

## 🟥 Phase 5 Prompt — Integration & Deployment (Tasks **5.1 → 5.4**)

**You are the agentic coder for Phase 5.** Your mission is to ship production‑grade **connectors**, **observability**, **test matrix**, and **deployment**—with disaster‑ready runbooks and verifiable artifacts.

### Context & constraints (canonical v2)

* Connectors: Notion/GitHub/Confluence (at least one end‑to‑end), queue ingestion, **circuit breakers/backoff** under rate limits.
* Monitoring: Prometheus metrics, Grafana dashboards, **OpenTelemetry** traces, alerts (P99, error rate, drift, OOM).
* Tests: complete **no‑mocks** matrix—unit/integration/E2E/perf/security/chaos—executed against the Compose stack.
* Deployment: blue/green + canary; backups; quarterly DR drills (**RTO 1h / RPO 15m**).

### Deliverables

1. **Connectors** (`src/connectors/*`) and runbooks; Slack notifications optional.
2. **Monitoring** (`deploy/monitoring/*`) with dashboards & alert rules; trace exemplars for slow queries.
3. **Testing framework** (`tests/*`, CI workflow, chaos scenarios) running the **entire** suite **without mocks**.
4. **Deployment** (`deploy/k8s/*` or Helm, `.github/workflows/ci.yml`, backup/restore scripts, feature flags, canary controls) + DR runbook.

### Tests (NO MOCKS) & artifacts

* Name tests `p5_*.py`. Required groups:

  * **Connector E2E**: ingest deltas from a real sandbox (or local webhook generator), queue processing, backoff on throttling.
  * **Monitoring drills**: synthetic alerts fire; dashboards render key panels; traces include query exemplars.
  * **Chaos**: kill vector service → degraded (graph‑only) operation; simulate Neo4j backpressure → ingestion backs off.
  * **Canary**: script deploys new build to 5%, monitors SLIs for 60 minutes (shortened in CI), rolls forward/backs off.
  * **Backup/restore**: snapshot, restore into clean environment, point app, verify parity.
* Emit:

  * `reports/phase-5/junit.xml`
  * `reports/phase-5/summary.json` (include alert outcomes, chaos results, DR timings)
  * Attach screenshots/links to dashboards (paths or exported PNG/JSON).

### Exit gate (Launch)

* Full **test matrix green**; monitoring & alerts live; canary + rollback proven; **DR drill passes** within targets; artifacts present.

### Work plan

1. One connector end‑to‑end → 2. Monitoring + alerts + traces → 3. Full test matrix + chaos → 4. Deployment & DR runbooks → 5. Canary rehearsal + reports.

---

### Hand‑off tips for the coder (applies to all phases)

* **Do not** reference any earlier documents—v2 files are canonical.
* **Never** use mocks. If a dependency is needed, stand up the real service in Compose.
* Every phase must produce: `reports/phase-*/junit.xml` and `reports/phase-*/summary.json`.
* Do not start Phase N+1 until Phase N **exit gate** is met and artifacts are committed.

If you want, I can also generate **phase‑specific kickstart checklists** as issue templates (one per phase) or deliver these prompts as ready‑to‑paste GitHub issues to assign to the agentic coder.

Below are **two separate documents** that supersede your v1 spec and plan, and **incorporate every improvement** I proposed. They keep the **same phases and task numbering** used in the “Expert Coder Guidance” I already delivered so that all three documents align one‑to‑one.

> These v2 documents build directly on your v1 Application Spec and v1 Implementation Plan.

---

# Document 1 — **WekaDocs GraphRAG MCP**

## **Application Specification — v2**

### 0) Executive summary

WekaDocs GraphRAG MCP is a secure, explainable documentation intelligence layer that lets LLMs query a **Neo4j** knowledge graph enriched with **vector search**. It ingests Markdown/HTML/Notion docs into a **provenance‑first** graph (`Document → Section → Entities`) and answers complex technical questions via **hybrid retrieval** (semantic + graph), **safe Cypher templates + validation**, and **structured responses** (Markdown + machine‑readable JSON).

This v2 spec formalizes: explicit **Document/Section** nodes, **idempotent IDs**, **provenance on every edge**, configurable **embedding models/dimensions**, a **primary vector store selection** (Qdrant *or* Neo4j vectors; dual‑write optional), plan‑based query **validation with EXPLAIN**, **auth + rate‑limits**, **OpenTelemetry**, strict **cache versioning**, and stronger **testing/chaos** requirements.

---

## 1) Goals, scope, and non‑goals

**Goals**

* Accurate, explainable answers with **evidence paths** down to document sections.
* **Deterministic ingestion** and **incremental updates** with rollback safety.
* **Sub‑500ms P95** for common queries with warmed caches.
* **Defense‑in‑depth** against Cypher injection and runaway traversals.

**Non‑goals**

* General web search or arbitrary document editing.
* Multi‑tenant data isolation beyond single‑org scope (can be added later).

---

## 2) High‑level architecture

```
LLM Client (MCP) ─► FastAPI MCP Server
                     ├─ Tools Registry (Search / Traverse / Compare / Troubleshoot / Explain)
                     ├─ Query Planner (Intent→Template, LLM fallback)
                     ├─ Cypher Validator (regex guard + AST/EXPLAIN limits)
                     ├─ Hybrid Retrieval (Vector + Graph, Ranker)
                     ├─ Response Builder (Markdown + JSON + Evidence)
                     ├─ Auth + Rate Limits + Audit
                     ├─ Cache (L1 in‑proc, L2 Redis)
                     └─ OpenTelemetry Tracing + Metrics

Ingestion Worker ─► Parsers (Markdown/HTML/Notion)
                    ├─ Normalizer (Document→Section; code/tables)
                    ├─ Entity Extractors (Command, Config, Procedure, Error, Concept, Example, Step, Parameter)
                    ├─ Graph Builder (MERGE by deterministic IDs, provenance)
                    ├─ Embeddings (configurable model/dims)
                    ├─ Vector Upsert (Primary store: Qdrant OR Neo4j)
                    └─ Reconciliation & Drift Repair

Storage ─► Neo4j 5.x (+ APOC/GDS)  ◄──►  Qdrant (optional primary)  ◄──► Redis
```

---

## 3) Data model (v2)

### 3.1 Node labels

* **Document** `{id, source_uri, source_type, title, version, checksum, last_edited, embedding_version?}`
* **Section** `{id, document_id, level, title, anchor, order, text, tokens, checksum}`
* **Command / Configuration / Procedure / Error / Concept / Example / Step / Parameter / Component**

  * All entities include `{id, name|title|code|term..., description?, category?, introduced_in?, deprecated_in?, updated_at, vector_embedding?}`

### 3.2 Relationships (all include provenance)

* `(:Document)-[:HAS_SECTION {order}]->(:Section)`
* `(:Section)-[:MENTIONS {confidence, start, end, source_section_id}]->(:Entity)`
* Inferred, provenance‑bearing edges (each has `derived_from_section_id`, `confidence`):

  * `REQUIRES`, `AFFECTS`, `RESOLVES`, `CONTAINS_STEP{order}`, `EXECUTES`, `RELATED_TO`, `HAS_PARAMETER`, `EQUIVALENT_TO`, `DEPENDS_ON`, `CAUSES`
* Optional temporal semantics on nodes/edges: `introduced_in`, `deprecated_in`.

### 3.3 IDs, versions, and consistency

* **Deterministic IDs** for all nodes and relationships (e.g., SHA‑256 of normalized `{source_uri|term|anchor|text}` tuples).
* `schema_version` singleton node; all cache keys and vector payloads are **prefixed** with `{schema_version}:{embedding_version}` to prevent stale results.
* **Embedding config is dynamic**: `{model_name, dims, similarity, multilingual}`. No hard‑coded 384D.

---

## 4) Retrieval & reasoning

### 4.1 Hybrid retrieval

1. **Vector seed** search over **Section** (and optionally Entity) embeddings.
2. **Controlled graph expansion** (1–2 hops; typed edges only).
3. **Connecting paths** (bounded `shortestPath`).
4. **Ranking** by semantic score, path proximity, entity priority, recency (`Document.last_edited`), and coverage.

### 4.2 Query planning & safety

* **Templates first** (intent ↦ pre‑approved Cypher).
* LLM‑generated Cypher only as fallback → pass through **validator**:

  * Regex guards + **parameterization enforcement**
  * **EXPLAIN** plan inspection: reject NodeByLabelScan and Expand(All) beyond thresholds; cap estimated rows/hops; enforce timeouts.
* Hard caps: traversal depth, result size, server‑side timeouts.

---

## 5) Responses & explainability

* Dual output:

  * **Markdown** (human friendly), and
  * **JSON** `{answer, evidence[{node_id, section_id, path}], confidence, diagnostics{ranking_features}}`.
* **Why these results?** toggle enumerates ranking features.
* **Disambiguation** card when multiple homonyms exist.

---

## 6) Security, privacy, compliance

* **JWT auth**, **per‑client rate limiting** (Redis token bucket), **audit logging** (query, params hash, client, plan stats).
* **Parameter‑only** user inputs; no raw literals in `WHERE`.
* Secrets via Docker/K8s secrets or Vault; TLS/mTLS for service links.
* GDPR‑friendly: delete/forget actions map to source filters + re‑ingestion.

---

## 7) Observability & SLOs

* **OpenTelemetry** tracing on FastAPI, Neo4j driver, and vector calls.
* Metrics: P50/P95/P99 latency, cache hit/miss, slow Cypher, vector latency, queue lag, reconciliation drift.
* **Targets**: P50<200ms, P95<500ms, P99<2s; availability 99.9%.

---

## 8) Ingestion & reconciliation (golden path)

1. **Parse to Document/Section**, preserve code fences/tables; compute `checksum`.
2. **Chunk**: Section is primary chunk; split only when very long; always prepend *title trail*.
3. **Extract entities** (pattern + light NLP) → create `MENTIONS` with spans + confidence.
4. **MERGE graph** (deterministic IDs) with `apoc.periodic.iterate` batches.
5. **Embeddings** (configurable) → **primary vector store** (choose **Qdrant** *or* **Neo4j vectors**). Dual‑write optional and behind feature flag.
6. **Reconcile nightly**: compare `{Section}` set vs vector collection by `embedding_version`; repair drift.

---

## 9) Interfaces

### 9.1 MCP tools (minimum set)

* `search_documentation`, `traverse_relationships`, `compare_systems`, `troubleshoot_error`, `explain_architecture`.
* Utility tools: `disambiguate(term)`, `explain_ranking(result_id)`, `show_path(a,b)`, `list_configs_affecting(component)`.

### 9.2 HTTP surface

* `/mcp` (MCP JSON‑RPC), `/health`, `/metrics`, `/ready`.
* Auth required on `/mcp`; rate‑limited; correlation IDs logged.

---

## 10) Phases & tasks (exact alignment with plan & guidance)

* **Phase 1 – Core Infrastructure**
  1.1 Docker environment setup
  1.2 MCP server foundation
  1.3 Database schema initialization
  1.4 Security layer

* **Phase 2 – Query Processing Engine**
  2.1 NL→Cypher translation
  2.2 Cypher validation system
  2.3 Hybrid search
  2.4 Response generation

* **Phase 3 – Ingestion Pipeline**
  3.1 Multi‑format parser
  3.2 Entity extraction
  3.3 Graph construction
  3.4 Incremental update

* **Phase 4 – Advanced Query Features**
  4.1 Complex query patterns
  4.2 Query optimization
  4.3 Caching & performance
  4.4 Learning & adaptation

* **Phase 5 – Integration & Deployment**
  5.1 External systems
  5.2 Monitoring & observability
  5.3 Testing framework
  5.4 Production deployment

*(This list mirrors the earlier Expert Coder Guidance for one‑to‑one traceability.)*

---

## 11) Success criteria & risks (delta from v1)

* **Evidence** attached to every answer; **confidence** score in [0,1].
* **Dual‑store clarity** (declare SoT; dual‑write behind a flag).
* **Validator** rejects unsafe plans (EXPLAIN‑based).
* **Chaos tests** for vector outage, Neo4j backpressure, cache poisoning.
* **Drift** alerts if graph/vector mismatch > 0.5% of sections.
  (Extends the v1 targets while preserving the spirit of your original spec. )

---

# Document 2 — **WekaDocs GraphRAG MCP**

## **Implementation Plan — v2**

> The following tasks (by phase) map **exactly** to the Spec (§10) and to the previously delivered **Expert Coder Guidance**. Where helpful, I point to areas in the v1 plan that are being upgraded in this v2 implementation.

### Conventions

* **Owner:** primary implementer; **Deps:** dependencies; **Deliverables:** tangible outputs; **DoD:** definition of done; **Tests:** acceptance tests.
* **Config keys** are illustrative and should live in `config/*.yaml`.

---

## Phase 1 – Core Infrastructure

### **Task 1.1 – Docker environment setup**

**Owner:** Platform
**Deps:** —
**Steps:**

1. Compose stack for `mcp-server`, `neo4j`, `qdrant`, `redis`, `ingestion-worker`, optional `nginx`.
2. Resource limits per service; healthchecks with realistic timeouts.
3. Volumes for data/logs; central log rotation.
4. Secrets via Docker/K8s secrets (no plaintext in env).
   **Deliverables:** `docker-compose.yml`, `docker/*.Dockerfile`, `.env.example`.
   **DoD:** `compose up` → all healthy; restart preserves data; secrets not visible via `docker inspect`.
   **Tests:** cold start, restart, and data persistence.
   **Notes:** Tighten Neo4j heap/pagecache; avoid conflicting envs. (Upgrade of v1 compose.)

---

### **Task 1.2 – MCP server foundation**

**Owner:** Backend
**Deps:** 1.1
**Steps:**

1. FastAPI app with `/mcp`, `/health`, `/ready`, `/metrics`.
2. Tools registry; initialization handshake; graceful shutdown.
3. **OpenTelemetry** tracing; structured logs with correlation IDs.
4. Connection pools for Neo4j/Qdrant/Redis; backpressure handling.
   **Deliverables:** `src/mcp_server/main.py`, `src/mcp_server/tools/*`, OTel config.
   **DoD:** tools list callable; health green; traces visible.
   **Tests:** tool call smoke tests; shutdown closes pools.

---

### **Task 1.3 – Database schema initialization**

**Owner:** Graph Eng
**Deps:** 1.1
**Steps:**

1. Create **Document** and **Section** labels; constraints & composite indexes (`Section {document_id, anchor}`).
2. Entity schemas (Command, Configuration, Procedure, Error, Concept, Example, Step, Component, Parameter).
3. **Vector indexes** created from **config** `{embedding.dims, similarity}` (no hard‑codes).
4. `schema_version` node write.
   **Deliverables:** `scripts/neo4j/create_schema.cypher`, `src/shared/schema.py`.
   **DoD:** `CALL db.indexes()` shows property + vector indexes; idempotent re‑runs OK.
   **Tests:** schema creation on empty and populated DBs.
   *(Supersedes v1’s fixed 384D indexes.)*

---

### **Task 1.4 – Security layer**

**Owner:** Platform + Backend
**Deps:** 1.2
**Steps:**

1. **JWT auth**, **Redis token bucket** rate‑limits; 429s logged.
2. Audit: store `{client_id, query_hash, params_hash, plan_stats}`.
3. Configurable query **maxDepth**, **maxRows**, **timeoutMs**.
   **Deliverables:** FastAPI middlewares, audit sink, policies.
   **DoD:** injection/over‑depth tests blocked with clear errors.
   **Tests:** auth bypass attempts, burst tests.

---

## Phase 2 – Query Processing Engine

### **Task 2.1 – NL→Cypher translation**

**Owner:** Retrieval
**Deps:** 1.2, 1.3
**Steps:**

1. Intent classifier; entity linker.
2. **Template library** for known intents; LLM fallback only when needed.
3. Map inputs → **parameterized** templates; never inline literals.
   **Deliverables:** `src/query/templates/`, `src/query/planner.py`.
   **DoD:** ≥90% of stories hit templates; fallback output passes validator.
   **Tests:** gold prompts → gold Cypher.

---

### **Task 2.2 – Cypher validation system**

**Owner:** Graph Eng
**Deps:** 2.1
**Steps:**

1. Regex guards (correct `*d..d` pattern fix) + parameter enforcement.
2. **EXPLAIN** plan inspection: reject excessive scans/expands; require index usage for large label scans; enforce server timeouts.
3. Depth/result caps inserted early in query.
   **Deliverables:** `src/mcp_server/validation.py`.
   **DoD:** malicious/expensive queries rejected; FP < 5%.
   **Tests:** negative suite: injections, deep traversals, Cartesian products.
   *(Replaces v1’s regex‑only checker.)*

---

### **Task 2.3 – Hybrid search**

**Owner:** Retrieval
**Deps:** 1.3, 2.2
**Steps:**

1. **Primary vector store** decision: `search.vector.primary ∈ {qdrant, neo4j}`; `search.vector.dual_write` flag.
2. Vector top‑K (Sections + optional Entities) → controlled 1–2 hop expansion.
3. **Ranker** blends semantic score, path proximity, entity priors, recency.
   **Deliverables:** `src/query/hybrid_search.py`, `src/query/ranking.py`.
   **DoD:** P95 < 500ms at K=20 (warmed); deterministic ties.
   **Tests:** relevance gold set; latency under load.

---

### **Task 2.4 – Response generation**

**Owner:** Backend
**Deps:** 2.3
**Steps:**

1. Markdown + **JSON** response with `evidence[{section_id, path}]` and `confidence`.
2. **Explain ranking** feature with feature weights.
3. Disambiguation card for homonyms.
   **Deliverables:** `src/query/response_builder.py`.
   **DoD:** E2E returns evidence & confidence; JSON schema validated.
   **Tests:** snapshot tests for answer formatting; schema validation.

---

## Phase 3 – Ingestion Pipeline

### **Task 3.1 – Multi‑format parser**

**Owner:** Ingestion
**Deps:** 1.3
**Steps:**

1. Parse Markdown/HTML/Notion → **Document/Section**; preserve `title trail`, code fences (language), tables.
2. Compute `section_checksum`, `tokens`; **deterministic `section_id`** from `{source_uri, anchor, normalized_text}`.
   **Deliverables:** `src/ingestion/parsers/*`.
   **DoD:** re‑parse → identical IDs; anchors preserved.
   **Tests:** round‑trip & determinism suite.

---

### **Task 3.2 – Entity extraction**

**Owner:** Ingestion
**Deps:** 3.1
**Steps:**

1. Pattern/NLP extract: Command, Config, Procedure, Error, Concept, Example, Step, Parameter.
2. Create **MENTIONS** edges from Section with offsets & confidence (no heavy inference yet).
   **Deliverables:** `src/ingestion/extract/*`.
   **DoD:** >95% precision for Commands/Configs on validation set.
   **Tests:** precision/recall metrics; span correctness.

---

### **Task 3.3 – Graph construction**

**Owner:** Graph Eng
**Deps:** 3.2
**Steps:**

1. **MERGE** nodes/edges by deterministic IDs; set provenance properties; `updated_at`.
2. Batch with `apoc.periodic.iterate` (1k–5k); tx timeout 30s.
3. **Embeddings** for Sections (+ optional Entities) then upsert to **primary vector store**; attach `embedding_version`.
   **Deliverables:** `src/ingestion/build_graph.py`.
   **DoD:** idempotent; re‑ingest unchanged content → no diffs.
   **Tests:** golden snapshot compare; partial failure retry.

---

### **Task 3.4 – Incremental update**

**Owner:** Ingestion
**Deps:** 3.3
**Steps:**

1. Diff by `section_checksum`; stage changes under `:Staged` labels; atomic swap.
2. Re‑embed changed + adjacent only; enqueue reconciliation.
3. Nightly **reconciliation** (graph vs vectors) with repair.
   **Deliverables:** `src/ingestion/incremental.py`, `src/ingestion/reconcile.py`.
   **DoD:** small doc edit updates O(changed sections); drift < 0.5% by morning.
   **Tests:** edit/rollback drills; simulated vector outage.

---

## Phase 4 – Advanced Query Features

### **Task 4.1 – Complex query patterns**

**Owner:** Retrieval
**Deps:** 2.x
**Steps:**

1. Pre‑approved templates: dependency chain, impact analysis, troubleshooting path, system comparison, temporal (“as of version Y”).
2. Input/Output schemas + plan guardrails for each.
   **Deliverables:** `src/query/templates/advanced/*.cypher`.
   **DoD:** Patterns run within depth/latency budgets; covered by tests.
   **Tests:** E2E for each template; plan assertions.

---

### **Task 4.2 – Query optimization**

**Owner:** Graph Eng
**Deps:** 4.1
**Steps:**

1. Plan analysis → **index recommendations**; hints where needed.
2. Compile & cache parameterized forms for hot templates.
3. Periodic slow‑query analysis with auto PRs for new indexes.
   **Deliverables:** `src/ops/optimizer.py`, dashboards.
   **DoD:** measurable P95 improvement post‑tuning.
   **Tests:** before/after benchmarks.

---

### **Task 4.3 – Caching & performance**

**Owner:** Backend
**Deps:** 2.x
**Steps:**

1. L1 in‑proc + L2 Redis; **keys prefixed with `{schema_version}:{embedding_version}`**.
2. Daily warm‑up for top intents; materialize expensive patterns.
   **Deliverables:** `src/shared/cache.py`, warmers, materializers.
   **DoD:** >80% hit rate steady‑state; correctness under version rotations.
   **Tests:** cache coherence under schema/model change.

---

### **Task 4.4 – Learning & adaptation**

**Owner:** Retrieval
**Deps:** 2–4
**Steps:**

1. Log query→result→user rating; learn ranking weights & entity patterns.
2. Suggest new templates and indexes from usage.
   **Deliverables:** `src/learning/*`.
   **DoD:** measurable relevance lift on held‑out set.
   **Tests:** offline evaluation; A/B flags.

---

## Phase 5 – Integration & Deployment

### **Task 5.1 – External systems**

**Owner:** Platform
**Deps:** 3.x
**Steps:**

1. Notion/GitHub/Confluence sync via webhooks or polling; queue to ingestion; circuit breakers.
2. Slack notifications for large updates.
   **Deliverables:** connectors + runbooks.
   **DoD:** steady ingestion under rate‑limits; degraded mode works.
   **Tests:** token expiry; rate‑limit backoff.

---

### **Task 5.2 – Monitoring & observability**

**Owner:** SRE
**Deps:** 1.2
**Steps:**

1. Prometheus exporters, Grafana dashboards, OTel traces.
2. Alerts: P99, error rate, reconciliation drift, OOM kills.
   **Deliverables:** `deploy/monitoring/*`, runbooks.
   **DoD:** on‑call can diagnose slow/failed queries in <10 min.
   **Tests:** alert fire drills; trace sampling sanity.

---

### **Task 5.3 – Testing framework**

**Owner:** QA
**Deps:** all
**Steps:**

1. Unit, integration, E2E, perf, **security**, **chaos**.
2. Golden doc set → golden graph snapshot; determinism enforced in CI.
   **Deliverables:** `tests/*`, CI pipelines, chaos scenarios.
   **DoD:** CI blocks non‑determinism or evidence regressions.
   **Tests:** injection attempts, runaway traversals, vector outage, cache poisoning.

---

### **Task 5.4 – Production deployment**

**Owner:** Platform/SRE
**Deps:** 5.2, 5.3
**Steps:**

1. Blue/green + canary; feature flags.
2. Automated backups; **restore drills** quarterly; DR: RTO 1h, RPO 15m.
   **Deliverables:** K8s manifests/Helm, GitHub Actions/GitLab CI, runbooks.
   **DoD:** 5% canary for 1h; auto rollback on SLI breach; DR test pass.
   **Tests:** failover and rollback simulations.

---

## Configuration (key excerpts)

```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2   # example
  dims: 384                                            # not hard-coded in code
  similarity: cosine
  multilingual: false
  version: "e1"

search:
  vector:
    primary: "qdrant"     # or "neo4j"
    dual_write: false
    collection: "weka_sections_v1"

limits:
  max_depth: 3
  max_results: 200
  cypher_timeout_ms: 30000

security:
  auth: jwt
  rate_limit:
    burst: 30
    per_minute: 600
```

---

## Acceptance gates (per phase)

* **P1 Gate:** stack healthy; schema created; auth & rate‑limit enforced; traces visible.
* **P2 Gate:** validator blocks attacks; hybrid search <500ms P95; responses include evidence+confidence.
* **P3 Gate:** ingestion deterministic; incremental update touches only changed sections; drift <0.5%.
* **P4 Gate:** advanced templates pass plan guardrails; cache hit >80%.
* **P5 Gate:** full test matrix green; monitoring/alerts live; DR drill passed.

---

## Timeline (indicative)

* **Weeks 1–2:** Phase 1
* **Weeks 3–4:** Phase 2
* **Weeks 5–6:** Phase 3
* **Weeks 7–8:** Phase 4
* **Weeks 9–10:** Phase 5

---

### Notes on v1→v2 changes

* Adds **Document/Section** & provenance; embedding config; SoT decision for vectors; **EXPLAIN‑based** validator; **OpenTelemetry**; **versioned caches**; **chaos tests**—all layered on top of your v1 foundation.

---

## What’s next

Your existing “Expert Coder Guidance” now perfectly aligns with these v2 docs (same phases/tasks). If you want, I can generate **PR stubs and file scaffolds** for each task number to kick off implementation.

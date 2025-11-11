# Implementation Plan: Deterministic Document-Scoped Retrieval & Reranking

**Repo:** wekadocs-matrix | **Author:** opencode | **Date:** 2025-11-02

> NOTE: This plan **does not modify code yet**. All paths reference existing modules to guide future PRs.

---

## 0. Preparation
1. **Create feature branch:** `feat/doc-router-reranker`.
2. **Enable pre-commit hooks & run static analysis** to baseline (~400 existing mypy errors—scope only new code).
3. **Activate shadow logging** (no prod effect) to capture router metrics before rollout.

---

### Reference Directory Map (quick links)
```
config/                # YAML & feature_flags.json  ➜ shared.config
data/                  # sample docs + golden queries
docker/                # container definitions
ingest/                # legacy wrapper (now auto/)
migration/             # baseline counts, analysis, test scripts
monitoring/            # dashboards + alert rules
scripts/               # admin + migration scripts
src/                   # application code (see mapping below)
  ingestion/           # build_graph, auto orchestrator, extract commands
  providers/           # embeddings, rerank providers
  query/               # retrieval, assembly, planner
  mcp_server/          # FastAPI entry + QueryService
  shared/              # config, cache, connections, logging
  monitoring/          # runtime metrics, SLOs
  ops/                 # warmers + cleanup jobs
  tools/               # misc maintenance utilities
```

---

## 1. Phase 1 — Doc-Tag Scoping (Spec 1)
### 1.1 Code Touch Points
| Module | Action |
|--------|--------|
| `src/ingestion/build_graph.py` | Add `extract_doc_tag()` helper; include `doc_tag` field in `create_*_node()` builders + Qdrant payload |
| `src/ingestion/__init__.py` | Expose `DOC_TAG_PATTERN` constant for reuse |
| `src/mcp_server/query_service.py` | Detect tag in query → `filters['doc_tag']` |
| `src/query/hybrid_retrieval.py` | Accept `filters` dict; push payload_filter to Qdrant; BM25 post-filter; same-doc neighbor gating |
| `src/query/context_assembly.py` | Sort citation labels, add fallback guard |
| `src/ingestion/chunk_assembler.py` | Preserve `title` in `boundaries_json` |

### 1.2 Migration Script & Validation
Use existing patterns:
* Template: `scripts/apply_complete_schema_v2_1.py` demonstrates how to run idempotent Cypher migrations inside app context.
* Baseline verification helpers: `migration/collect_baseline.py`, `migration/debug_baseline.py` – use them to ensure counts before/after match.
* Add new script `scripts/migrations/backfill_doc_tag.py` modelled on `scripts/backfill_document_tokens.py`.

After migration, run `scripts/run_baseline_queries.py` to confirm no query regressions.
File: `scripts/migrations/backfill_doc_tag.py`
* For existing `tests://regression_pack/` docs, derive tag from `source_uri` → set Neo4j property & Qdrant payload.

### 1.3 Tests
* `tests/regression/test_doc_tag_gating.py` — Load sample query, assert all returned chunks share same `doc_tag`.
* `tests/unit/test_citation_sort.py` — Verify deterministic ordering.

### 1.4 Rollout
1. **Canary environment** with regression pack corpus only.
2. **Metrics spike**: `cross_doc_mix_rate` expected → 0.

---

## 2. Phase 2 — Document Router (Spec 2)
### 2.1 New Components
* `src/query/document_router.py`
  * `DocumentIndex` — BM25 (Neo4j full-text) + Vector (Qdrant points aggregated per doc).
  * `DocumentRouter` — exposes `.route(query, k, delta, margin)`.

* `src/ingestion/document_summary.py`
  * Generate doc summary: `title + top H1/H2 + first 2 sentences`.

### 2.2 Modifications
* `ingestion/build_graph.py` — compute and store `doc_id` (UUID) + `doc_key` (slug | hash).
* `query/hybrid_retrieval.py` — add `allowed_doc_ids` param used for payload filter.
* `mcp_server/query_service.py` — orchestrate 2-stage retrieval: router → chunk retriever.

### 2.3 Config
Add to `config/development.yaml`:
```yaml
router:
  delta: 0.30
  margin: 0.12
  k_docs: 1
```
Dynamic via ENV overlay: `ROUTER_STRICT_MODE`.

### 2.4 Telemetry & Monitoring Integration
* **Metrics Module** – `src/monitoring/metrics.py` exposes `Metrics` singleton wrapping `prometheus_client`. Add:
  * `Counter('router_ambiguous_total', 'Queries where router abstained')`
  * `Histogram('router_latency_ms', buckets=[5,10,25,50,100,250])`
  * `Histogram('rerank_latency_ms', buckets=[5,10,25,50,100,250])`
* **Dashboards** – Update Grafana JSON under `monitoring/dashboards/phase7e_retrieval.json` via `scripts/monitoring/update_dashboards.py` (pattern exists). Verify alert thresholds in `monitoring/alerts/phase7e_slo_alerts.yaml`.
* **Exemplars** – `shared/observability/exemplars.py` supports linking traces to metrics; record exemplar on `router_latency_ms` for slowest 0.1% samples.

* Counter `ambiguous_flag` increments when router abstains.
* Histogram `router_latency_ms`.

### 2.5 Tests
* `tests/router/test_single_doc_property.py` — ensure stitched context ≤1 `document_id`.
* `tests/router/test_ambiguous_flow.py` — ambiguous query returns structured candidates.

### 2.6 Rollout
1. **Shadow mode** for one week capturing router scores but not gating.
2. Enable strict mode behind flag.

---

## 3. Phase 3 — Jina Reranker v2 (Spec 3)
### 3.1 External Dependency
* Add `jina-reranker==2.*` to `requirements.txt` (or use hosted API via `jina-hub-sdk`).
* Create `src/shared/jina_client.py` with retry + circuit breaker (10 s timeout).

### 3.2 Pipeline Changes (code touchpoints)
* `providers/rerank/jina.py` – Implement `JinaReranker` subclass; register in `providers.factory.RerankProviderFactory`.
* `shared/connections.py` – Add pooled `httpx.AsyncClient` for Jina; follow pattern used for `QdrantClient`.
* `monitoring/metrics.py` – Add `Histogram('rerank_latency_ms', ...)` and `Counter('router_ambiguous_total', ...)`.
* `monitoring/slos.py` – Insert new SLO definitions (latency ≤25 ms, ambiguous ≤5%).
* `tools/redis_epoch_bump.py` – Extend CLI to bump `doc_epoch` & `chunk_epoch` after reranker-enabled ingestion.

* **Document stage** – After `DocumentRouter.hybrid_search()`, rerank top K docs (K≤32) with Jina.
* **Chunk stage** – After scoped chunk retrieval (N≤60), rerank with Jina, keep top M chunks (M≤12).

### 3.3 Caching Layer
* Redis TTL cache keyed by SHA256(query+candidate_id) → score.
* Config flag `RERANK_CACHE_TTL_SEC`.

### 3.4 Threshold Calibration Utility
* Notebook `notebooks/calibrate_reranker_thresholds.ipynb` (not executed in CI) outputs YAML mapping corpus→(δ, μ).

### 3.5 Tests
* Mock Jina client to return deterministic scores; unit test sorting & fallback path.

### 3.6 Rollout
1. Enable in canary with `sample_rate=0.2`.
2. Monitor `rerank_latency_ms`, `cache_hit_rate`.
3. Gradually ramp to 1.0.

---

## 4. Ops & Deployment
| Item | Details |
|------|---------|
| **Env Vars** | `JINA_API_KEY`, `ROUTER_STRICT_MODE`, `RERANK_CACHE_TTL_SEC` |
| **Secrets** | Add `JINA_API_KEY` placeholder to `.env.example`; instruct ops to inject in vault |
| **CI** | Add `make validate-router` stage that runs new unit tests + type checks |
| **Helm Chart** | New config map `router-config` with thresholds; secret for API key |

---

## 4.5 Deployment Integration Notes
The application is shipped via **Docker + Kubernetes** (kustomize base in `deploy/k8s`).
Key integration points your PR **must** respect:

1. **Config Map Mount** – All runtime YAML in `config/` is mounted at `/app/config` (see `deploy/k8s/base/mcp-server-deployment.yaml`, volume `config`). Do **not** hard-code paths; use `shared.config.load()` which already resolves `WEKADOCS_CONFIG_DIR` → default `/app/config`.
2. **Environment Variables** – The server pod defines:
   * `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (secret)
   * `REDIS_URL`
   * `QDRANT_URL`
   * `OTEL_EXPORTER_JAEGER_ENDPOINT`
   * New in this feature:
     * `ROUTER_STRICT_MODE` – default `false`
     * `RERANKER_ENABLED` – default `false`
     * `JINA_API_KEY` (secret) – only required when reranker enabled
   Ensure your code reads **only** from `config/*` or `os.getenv` for these keys; do not reference Helm values directly. Add placeholders to `.env.example` (handled in separate chores).
3. **Liveness / Readiness** – Keep `/health` and `/ready` endpoints unchanged; router & reranker initialization must finish **before** readiness probe passes.
4. **Resource Limits** – Current pod limits (1Gi / 1 vCPU). Reranker client buffering shall keep memory <250 Mi. Use streaming API or chunked batches if needed.
5. **Jaeger Tracing** – The OpenTelemetry tracer auto-instruments `httpx` calls; tag reranker spans with `component="jina-reranker"` for observability.
6. **Blue-Green Deploy** – Two deployments `mcp-server-blue`/`green`; newly introduced flags must default to *off* so that old pods can co-exist during rollout.

---

## 5. Backward Compatibility
* If `router.strict_mode=false`, pipeline behaves exactly as today (legacy fusion).
* `doc_tag` property ignored when absent → safe for old corpora.
* On Jina outage, fallback path bypasses rerank (latency ≤ legacy).

---

## 6. Milestones & Timeline
| Week | Deliverable |
|------|-------------|
| 45 | Phase 1 code + tests merged; canary enabled |
| 46 | Phase 2 router behind shadow flag; ingestion backfill script done |
| 47 | Phase 2 strict mode enabled; ambiguous UX wired |
| 48 | Phase 3 reranker dependency landed; canary @20% |
| 49 | Full rollout; KPI review & project retrospective |

---

## 7. Appendix
### 7.1 Qdrant Purge Script by `doc_tag`
```python
from qdrant_client import QdrantClient
client = QdrantClient("localhost", port=6333)
client.delete(collection_name="chunks",
              key_condition={"doc_tag": {"match": "REGPACK-01"}})
```

### 7.2 Cypher Assertion (single-doc stitched answer)
```cypher
MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
WHERE c.id IN $chunk_ids
WITH collect(DISTINCT d.id) AS docs
RETURN size(docs) = 1 AS is_single_doc;
```

---

**End of Implementation Plan**

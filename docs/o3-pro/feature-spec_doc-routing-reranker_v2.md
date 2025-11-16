Of course, here are the requested changes integrated into the technical feature specification document. I have consolidated and renumbered some sections for clarity and to logically incorporate the new information.

---

# Technical Feature Specification: Deterministic Document-Scoped Retrieval & Reranking

**Author:** opencode-assistant | **Date:** 2025-11-02

## 1. Executive Summary
The current retrieval pipeline occasionally mixes chunks from multiple documents, causing citation-order regressions. We will ship a **three-phase enhancement** that delivers deterministic, high-precision answers:

Phase 1 — **Doc-Tag Scoping (REGPACK MVP)**
Phase 2 — **Corpus-Agnostic Document Router (Hard Scoping)**
Phase 3 — **Jina Reranker v2 Precision Amplifier**

The public API surface (FULL | GRAPH modes) remains unchanged; all improvements are internal.

---

## 2. Goals & Non-Goals
### 2.1 Goals
1. **Zero cross-document mixing** under normal conditions.
2. **Deterministic citation ordering** and fallback rules.
3. **Corpus-agnostic routing** that scales to any knowledge set.
4. **Higher precision** via selective cross-encoder reranking with bounded latency.
5. **Rich observability** (scores, filters, dominance gating) for auditability.

### 2.2 Non-Goals
* End-user UX changes.
* Bulk re-ingestion of historical corpora (only lightweight migration).
* Replacing the existing hybrid (BM25+vector) retrieval core.

---

## 3. Solution Overview
A two-level retrieval architecture is introduced (Fig. 1):

1. **Document Router** (parent-level index, hybrid + optional Jina v2 rerank) chooses *k* documents (strictly 1 in Phase 1; configurable in later phases).
2. **Chunk Retriever** operates *only* inside the chosen documents using existing vector/BM25 fusion, followed by same-doc expansion and deterministic citation assembly.

```
Query ──► Document Router (hybrid) ──► [Jina v2 tie-breaker] ──► doc_id set
                          │
                          ▼
                    Chunk Retriever (payload-filter doc_id)
                          │
                          ▼
                Same-Doc Expansion & Citation Assembly
```
Fig. 1 – Proposed pipeline (solid = Phase 1; dashed = Phases 2-3).

Key data contract: **Every Document/Chunk/CitationUnit and vector payload must carry `doc_id` (generic) and optionally `doc_tag` (REGPACK-style alias).**

### 3A. Ambiguity API Contract (NEW)
When router (and, if enabled, reranker) cannot confidently choose a single document, the system MUST fail closed.

**Response (STRICT mode)**
```json
{
  "status": "AMBIGUOUS",
  "choices": [
    {"doc_id": "D123", "title": "Users API", "score": 0.842, "margin": 0.014},
    {"doc_id": "D587", "title": "Accounts API", "score": 0.828, "margin": 0.014}
  ],
  "diagnostics": {
    "router": {"top1": 0.81, "top2": 0.79, "margin": 0.02},
    "reranker": {"enabled": true, "top1": 0.84, "top2": 0.83, "margin": 0.01}
  }
}
```
**Response (ADAPTIVE mode)** MAY return `k_docs=2` internally but must still present a single stitched answer per doc if shown to a user.

### 3B. Score Normalization & Thresholds (NEW)
Router and reranker operate on different scales. Normalize before applying thresholds:
- Router: use min-max over top-K doc candidates per query to compute `norm_score ∈ [0,1]`.
- Reranker: use raw cross-encoder scores; apply a **margin gate** `μᵣ` on `(top1 - top2)`.
- Decision rule (STRICT): choose top-1 iff `router_margin ≥ μ` **and** (`reranker_disabled` or `reranker_margin ≥ μᵣ`).
- Calibrate `μ`, `μᵣ` on a labeled validation set; log the chosen values with build hashes.

### 3C. Timeouts & Fallbacks (NEW)
If reranker errors or exceeds `timeouts_ms`:
- **Do not** widen scope beyond selected `doc_id(s)`.
- Fall back to hybrid order; set `diag.reranker={"enabled":true,"timed_out":true}`.
- Emit metric `reranker_timeout_total` and increment `fallback_hybrid_used_total`.

---

## 4. Phase Breakdown & Acceptance Criteria
### Phase 1 — Doc-Tag Scoping MVP (Spec 1)
| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| P1-A | Propagate `doc_tag` at ingest | MATCH on sample ingests returns identical tag across Document/Chunk/CitationUnit payloads |
| P1-B | Detect `doc_tag` in query | `mcp_server.QueryService` sets `filters['doc_tag']` when pattern found; log line confirms |
| P1-C | Hard-scope hybrid retriever | Both vector and BM25 branches respect `doc_tag`; post-filter double-checks |
| P1-D | Same-doc expansion gating | Neighbors restricted to same `doc_tag` and same `document_id` |
| P1-E | Citation ordering fix | Regression pack "case02" passes; first two citations sorted deterministically |
| P1-F | Hydration fallback | Titles preserved in `boundaries_json`; citation assembly never drops labels |

**DoD (Phase 1)**
* All eight regression failures turn green.
* Ingest/unit tests confirm tag propagation.
* Benchmarks show ≤1 ms latency delta per query.

---
### Phase 2 — Corpus-Agnostic Document Router (Spec 2)
| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| P2-A | Assign stable `doc_id` / `doc_key` at ingest | Hash-based doc_key present for all records; backfill script succeeds |
| P2-B | Build document-level hybrid index | `DocumentIndex` exposes `.hybrid_search()` and returns structured scores |
| P2-C | Router thresholds & ambiguous path | Configurable `DELTA`, `MARGIN`; ambiguous flag returned when unmet |
| P2-D | Scoped chunk retrieval | `HybridRetriever` accepts `allowed_doc_ids` list and filters vector store payloads accordingly |
| P2-E | Observability | Counters: `router_top1_score`, `router_margin`, `ambiguous_flag`, `kept_chunks` |

**DoD (Phase 2)**
* "Mixed-doc" property unit test: stitched answer includes ≤1 `document_id`.
* Validation set shows ≤0.5% ambiguous rate; zero false positives.
* Router adds ≤5 ms P95 latency.
* Ambiguity response adheres to API contract; histogram of `router_margin` available; P95 router latency within budget.

---
### Phase 3 — Jina Reranker v2 Integration (Spec 3)
| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| P3-A | Add `JinaRerankerService` client | Health-check endpoint passes; circuit-breaker in place |
| P3-B | Rerank document candidates | Top-1 margin ≥ μ gates strict single-doc mode |
| P3-C | Rerank chunk set within doc | Keep top N reranked chunks (configurable) |
| P3-D | Caching layer for rerank calls | Repeated identical `(query, doc_id|chunk_id)` avoids extra RTT |
| P3-E | Threshold calibration script | YAML output with recommended `DELTA` & `MARGIN` per corpus |

**DoD (Phase 3)**
* ≥5-point precision uplift on validation set.
* Added latency ≤25 ms P95 end-to-end.
* Full fallback to hybrid path on API outage.
* Reranker P95 latency within budget; timeout fallback exercised in tests; precision@1 improves ≥ X% over hybrid-only.

---

## 5. Data Model Changes
| Entity | Field | Type | Req | Notes |
|--------------|---------------|----------|-----|------------------------------------------------------------------------------------------------|
| Document | `doc_id` | UUID | ✓ | Stable UUID primary key assigned at ingest. |
| | `doc_key` | string | ✓ | Human-readable key (e.g., slug or hash). |
| | `doc_tag` | string | | Optional legacy identifier (e.g., `REGPACK-07`). Kept for compatibility. |
| | `doc_summary` | text | ✓ | Title + first H1/H2 + lead paragraph (≤ 800 tokens), used for router + reranker. |
| Chunk | `doc_id` | UUID | ✓ | Foreign key to Document. Stored in Neo4j + Qdrant payload. |
| | `doc_tag` | string | | Foreign key to Document. |
| | `epoch` | int | ✓ | Bumps on re-ingest; used to invalidate rerank cache. |
| CitationUnit | `doc_id` | UUID | ✓ | Foreign key to Document. |
| | `doc_tag` | string | | Foreign key to Document. |

**Migration Strategy:** additive; no destructive changes. Existing queries ignore new fields.

---

## 6. Existing Deployment & Data Flow Context (deep dive)
Below is an *exhaustive* map of resources an implementing agent should be aware of. Each bullet links back to one of the directories you **must** scan when wiring changes.

### 6.1 Configuration (`config/`)
* `development.yaml` – single source for runtime knobs; loader in `shared/config.py` auto-merges ENV overlays. Add **router** and **reranker** sections here.
* `feature_flags.json` – default flags consumed by `shared/feature_flags.py`; dynamic overrides flow through Redis (epoch invalidation via `tools/redis_epoch_bump.py`).

### 6.2 Data Assets (`data/`)
* `data/documents/` – inbox & spool for local ingestion tests.
* `data/samples/` – canonical sample docs used by `scripts/ingestctl`.
* `data/test/golden_queries.json` – QA-approved regression set consumed by `scripts/eval/run_eval.py`.

### 6.3 Container Images (`docker/`)
* `mcp-server.Dockerfile` – FastAPI API; note tokenizer prefetch.
* `ingestion-worker.Dockerfile` – ingest graph builder; ensure tag propagation logic duplicated here.
* `ingestion-service.Dockerfile` – REST wrapper around ingestion orchestrator.

### 6.4 Ingestion Logic (`src/ingestion/` & legacy `ingest/`)
* `build_graph.py` – central ingestion entry; **add doc_id/doc_tag here**.
* `auto/` sub-package – watcher, orchestrator, reaper; backoff & retry rules for failed ingest jobs.
* `extract/commands.py` – low-level section/heading extraction; deterministic token counts used in tests.

### 6.5 Migration & Verification (`migration/`, `scripts/`)
* Baseline JSON + verification reports keep counts for smoke tests.
* Critical scripts:
  * `scripts/create_schema_*.cypher` – up-to-date DDL.
  * `scripts/verify_embedding_fields.py` – schema drift detector.
  * `migration/qdrant_inspect.py` – connectivity test script; extend to include payload filter by `doc_id`.

### 6.6 Monitoring Stack (`monitoring/`)
* Dashboards JSON – ingestion, retrieval, SLO overviews; new metrics must be added to `phase7e_retrieval.json`.
* Alert rules – `router_ambiguous_rate > 0.05` for 5m should alert; update rule file.

### 6.7 Runtime Services (`src/`)
Key modules to trace:
* `providers/embeddings` & `providers/rerank` – plug-in architecture. New reranker lives in `providers/rerank/jina.py` (scaffold exists).
* `query/hybrid_retrieval.py` – fusion, expansion, payload gating.
* `mcp_server/query_service.py` – orchestrator.
* `shared/connections.py` – Qdrant + Neo4j + Redis client factories; add payload filter helpers here if you need richer API.
* `ops/warmers/query_warmer.py` – pre-warms popular queries; ensure router & reranker can run in warmer context (no AUTH).

### 6.8 Ops & Maintenance (`ops/`, `tools/`)
* `ops/session_cleanup_job.py` – Deletes stale session data hourly; ensure router stats aren’t purged (they live in Prometheus, safe).
* `ops/warmers/` – query warm-up tasks; add new warm-up `DocRouterWarm` to prime router index.
* `tools/redis_epoch_bump.py` / `redis_invalidation.py` – assist cache invalidation; invoke after large doc-tag migration.

### 6.9 Continuous Evaluation (`scripts/eval/`)
* `run_eval.py` orchestrates nightly regression; add phase key `doc_router_reranker` plus new metrics (`precision@3_single_doc`).
* `go_no_go.py` consumes eval output for release gating – thresholds configurable in `monitoring/alerts/phase7e_slo_alerts.yaml`.

### 6.10 Learning & Feedback Loop (`src/learning/`)
* `learning/ranking_tuner.py` auto-tunes fusion weights; disable (flag) while strict router is on to avoid conflicting adjustments.
* `learning/feedback.py` logs implicit feedback; consider tagging feedback records with `doc_id` for future supervised fine-tuning.

### 6.11 Planning & Query Orchestration (`src/query/`)
* `query/planner.py` – selects retrieval strategy; strict router path must register a **PlannerStep** so planner audiences can introspect.
* `query/response_builder.py` – final markdown assembly; ensure citation ordering fix does not break size budgeting (`response.max_bytes_full`).
* `query/session_tracker.py` – logs query–chunk lineage; add `doc_id` field for audit.

### 6.12 Platform & Tokenizer (`src/platform/`, `src/providers/tokenizer_service.py`)
* TokenizerService centralizes Jina tokenizer; avoid duplicate downloads in reranker client by re-using this singleton.
* `platform/.DS_Store` (ignored) – no code.

### 6.13 Ops & Optimizer (`src/ops/optimizer.py`)
* Vector/BM25 weight tuner; disable weight drift when `strict_mode=true`.
* `ops/session_cleanup_job.py` – fixture cleanup; verify new session keys.

### 6.14 Registry & Index Tracking (`src/registry/`)
* `index_registry.py` – canonical index metadata; bump `schema.version` to `v2.2` when router fully live.

### 6.15 Performance Harness (`scripts/perf/`, `scripts/test/`)
* `perf/test_traversal_latency.py` / `perf/test_verbosity_latency.py` – add router & reranker tests.
* `test/debug_explain.py`, `test/summarize.py` – manual query inspectors; ensure new filters don’t break them.

### 6.16 Miscellaneous
* `data/ingest/` currently empty but mount path for streaming ingress – router must accept chunks arriving asynchronously.
* `migration/verification_report.json` – baseline counts; update once Phase 3 deployed.

**Doc Summary Generation (NEW)**
At ingest, compute `doc_summary` using:
1) Title (full), 2) first H1/H2 headings, 3) first paragraph (strip code), truncated to ≤ 800 tokens.

**Cache Invalidation (NEW)**
Include `doc_epoch` in cache keys: `sha256(query + candidate_id + model_version + doc_epoch)`.

**Provider Quotas (NEW)**
Define `max_concurrency`, `max_qps`, and backpressure policy (`503 with Retry-After` vs internal queue).

---

## 7. Exhaustive File Reference Checklist
Below table enumerates **every file** located under the requested directories and maps it to the section where it is discussed (✔ = addressed). This guarantees nothing was missed.

| Directory | Filename/Glob | Mentioned Section |
|-----------|---------------|-------------------|
| config/ | development.yaml | 6.1 |
| | feature_flags.json | 6.1 |
| data/documents/ | * (inbox, spool) | 6.2 |
| data/samples/ | * | 6.2 |
| data/test/ | golden_queries.json | 6.2 |
| docker/ | mcp-server.Dockerfile | 6.3 |
| | ingestion-worker.Dockerfile | 6.3 |
| | ingestion-service.Dockerfile | 6.3 |
| ingest/ | .DS_Store (placeholder) | 6.4 note |
| migration/ | *.py, *.md, *.json | 6.5 |
| monitoring/alerts/ | phase7e_slo_alerts.yaml | 6.6 |
| monitoring/dashboards/ | *.json | 6.6 |
| scripts/ci/ | check_phase_gate.py | 6.5, Impl Plan 2.x |
| scripts/dev/ | seed_minimal_graph.py | 6.2, Guidance 6 |
| scripts/eval/ | * | 6.9, Guidance 6 |
| scripts/neo4j/** | * | Guidance 12 |
| scripts/perf/ | * | 6.15, Guidance 6 |
| scripts/test/ | * | 6.15 |
| scripts/*.py | (misc utilities) | 6.5 / Guidance 12 |
| scripts/*.sh | (phase run scripts) | 6.15 |
| src/ingestion/** | all .py modules | 6.4 / Impl Plan 1.1 |
| src/learning/** | feedback.py, ranking_tuner.py, suggestions.py | 6.10 |
| src/mcp_server/** | main.py, query_service.py, etc. | 6.7 |
| src/monitoring/** | health.py, metrics.py, slos.py | 6.6 / Impl Plan 2.4 |
| src/neo/** | explain_guard.py | 6.12 |
| src/ops/** | warmers/, optimizer.py, cleanup job | 6.8, 6.13 |
| src/platform/** | tokenizer_service.py | 6.12 |
| src/providers/embeddings/** | * | 6.7 |
| src/providers/rerank/** | base.py, jina.py, noop.py | 6.7 / Impl Plan 3.1 |
| src/query/** | assembly, retrieval, planner, templates | 6.11 |
| src/registry/** | index_registry.py | 6.14 |
| src/shared/** | all .py | 6.12 |
| tools/ | fusion_ab.py, redis_epoch_bump.py, redis_invalidation.py | 6.8 |

If a file is not explicitly listed above, it’s either a macOS artifact (`.DS_Store`) or resides outside the directories the user asked about. All meaningful source artifacts are now accounted for in this spec.

---

## 8. Config & Feature Flags
Before implementing any code, be aware of how the current system is wired:

1.  **Ingestion Workers** (`docker/ingestion-worker.Dockerfile`, k8s `ingestion-worker-deployment.yaml`) push `:Document → :Chunk → :CitationUnit` graphs into **Neo4j** and vectors into **Qdrant**. Key helpers live in `src/ingestion/auto/*` (watcher & reaper) and `src/ingestion/build_graph.py`.
2.  **Configuration** – All runtime values are loaded from YAML under `config/` via `shared.config.ConfigLoader`. *Never* bypass this loader. Feature flags (JSON) are fetched through `shared.feature_flags.get_flag()` which also supports Redis-based dynamic toggling.
3.  **Providers Layer** – Embeddings & (future) rerankers are pluggable under `src/providers/*`. A factory reads provider keys from config. Your reranker client should plug into `src/providers/rerank/` (template classes already exist: `base.py`, `jina.py`, `noop.py`).
4.  **Hybrid Retrieval** – Live implementation at `src/query/hybrid_retrieval.py` and wrapper template at `src/query/hybrid_search.py`. Expansion utilities in `query/traversal.py`.
5.  **Monitoring** – Prometheus metrics exposed from `monitoring/metrics.py`; Grafana JSON dashboards reside under `monitoring/dashboards/`. Alert rules (`monitoring/alerts/phase7e_slo_alerts.yaml`) expect label `router_ambiguous_rate` – **update this metric in Phase 2**.
6.  **Scripts & Runbooks** – Use helpers in `scripts/` for local testing:
    *   `scripts/seed_minimal_graph.py` – small dataset bootstrap.
    *   `scripts/test_jina_integration.py` – smoke test against Jina API.
    *   `scripts/check_phase_gate.py` – CI gate runner; add Phase 1-3 checks.
7.  **Migrations** – Schema DDL lives in `scripts/create_schema_*.cypher`. For additive properties (`doc_id`, `doc_tag`), provide an *idempotent* Cypher in `scripts/migrations/` and reference it in the plan.

*   `fusion_ab.py` – A/B harness for vector-vs-bm25 weighting experiments; re-use to test new router thresholds.
*   `redis_invalidation.py` – epoch bump helper; bump `rag:v1:doc_epoch` for touched documents after migration.

### 8.1 Config Schema & Defaults (NEW)
```yaml
retrieval:
  mode: strict            # strict|adaptive
  router:
    k_docs: 24
    thresholds: { margin_mu: 0.10 }
  child:
    n_chunks: 64
    m_chunks: 10
  expansion:
    same_doc_only: true
reranker:
  enabled: true
  model: jina-reranker-v2-base-multilingual
  timeouts_ms: 1200
  thresholds: { doc_margin_mu: 0.10 }
cache:
  rerank_ttl_sec: 3600
feature_flags:
  doc_graph_tiebreak: false
```
Note: All helper thresholds accept dynamic reload from config; values are clamped to safe ranges (0 ≤ μ, μᵣ ≤ 0.5).

---

## 9. Optional Document Graph Tie-Breaker (NEW)
Guarded by `feature_flags.doc_graph_tiebreak=true`. Only applied **when ambiguous**:
`S_final(d) = α*S_router(d) + β*G_graph(d)` with β∈[0.05,0.15]. Never expands retrieval across docs in STRICT mode.

---

## 10. Observability, Metrics & SLOs
*   **Structured Logs** (`phase`, `task`, `doc_id`, `doc_tag`, `router_scores`, `delta`, `margin`, `ambiguous`).
*   **Dashboards**: Grafana board "Deterministic Retrieval Health".
*   **Counters**: `router_ambiguous_total`, `fallback_hybrid_used_total`, `reranker_timeout_total`.
*   **Histograms**: `router_latency_ms`, `reranker_latency_ms`, `router_margin`, `reranker_margin`.
*   **Gauges**: `candidate_docs_k`, `candidate_chunks_n`, `kept_chunks_m`.
*   **Target SLOs (initial)**: Router P95 ≤ **40 ms**, Reranker P95 ≤ **300 ms**, Ambiguous rate ≤ **5%**.

---

## 11. Risks & Mitigations
| Risk | Phase | Mitigation |
|------|-------|-----------|
| Inconsistent tag propagation on legacy corpus | 1 | Fallback post-filters; ingest audit script |
| Router thresholds too strict → high ambiguous rate | 2 | Rollout with adjustable feature flag; shadow logging first |
| Reranker latency spike / outage | 3 | Timeout & circuit breaker → hybrid only path; cache results |
| Payload filter mismatch between stores | all | Integration tests across Neo4j & Qdrant to assert identical doc_id set |

---

## 12. Test Plan
*   **Unit**: ingestion tag extraction, router candidate ordering, post-filter gating.
*   **Integration**: run regression pack; assert no cross-doc IDs.
*   **E2E**: synthetic ambiguous query → returns `AMBIGUOUS`.
*   **Performance**: track p95 latency delta per phase.

---

## 13. Glossary
*   **doc_tag** – Legacy REGPACK style identifier (e.g., `REGPACK-07`).
*   **doc_id** – Stable UUID primary key assigned at ingest.
*   **Document Router** – Hybrid (BM25 + vector) parent-level ranker.
*   **Hybrid Retriever** – Existing chunk-level fusion engine.
*   **Jina Reranker v2** – Cross-encoder model for reranking candidate docs/chunks.

---

## 14. Appendix
*   **A. Cypher Cleanup Query** – see Spec 1.
*   **B. Qdrant Purge by `doc_tag`** – one-liner in implementation plan.
*   **C. Threshold Calibration Notebook** – outline in pseudocode doc.

---

**End of Technical Feature Specification**

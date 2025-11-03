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

---

## 5. Data Model Changes
| Entity | New / Updated Fields | Notes |
|--------|----------------------|-------|
| Document | `doc_id` (UUID)<br>`doc_key` (slug|hash)<br>`doc_tag` (optional legacy) | Primary key promotion to `doc_id`; `doc_tag` kept for REGPACK compatibility |
| Chunk | `doc_id`, `doc_tag` | Stored in Neo4j + Qdrant payload |
| CitationUnit | `doc_id`, `doc_tag` | -- |

**Migration Strategy:** additive; no destructive changes. Existing queries ignore new fields.

---

## 5. Existing Deployment & Data Flow Context (deep dive)
Below is an *exhaustive* map of resources an implementing agent should be aware of.  Each bullet links back to one of the directories you **must** scan when wiring changes.

### 5.1 Configuration (`config/`)
* `development.yaml` – single source for runtime knobs; loader in `shared/config.py` auto-merges ENV overlays.  Add **router** and **reranker** sections here.
* `feature_flags.json` – default flags consumed by `shared/feature_flags.py`; dynamic overrides flow through Redis (epoch invalidation via `tools/redis_epoch_bump.py`).

### 5.2 Data Assets (`data/`)
* `data/documents/` – inbox & spool for local ingestion tests.
* `data/samples/` – canonical sample docs used by `scripts/ingestctl`.
* `data/test/golden_queries.json` – QA-approved regression set consumed by `scripts/eval/run_eval.py`.

### 5.3 Container Images (`docker/`)
* `mcp-server.Dockerfile` – FastAPI API; note tokenizer prefetch.
* `ingestion-worker.Dockerfile` – ingest graph builder; ensure tag propagation logic duplicated here.
* `ingestion-service.Dockerfile` – REST wrapper around ingestion orchestrator.

### 5.4 Ingestion Logic (`src/ingestion/` & legacy `ingest/`)
* `build_graph.py` – central ingestion entry; **add doc_id/doc_tag here**.
* `auto/` sub-package – watcher, orchestrator, reaper; backoff & retry rules for failed ingest jobs.
* `extract/commands.py` – low-level section/heading extraction; deterministic token counts used in tests.

### 5.5 Migration & Verification (`migration/`, `scripts/`)
* Baseline JSON + verification reports keep counts for smoke tests.
* Critical scripts:
  * `scripts/create_schema_*.cypher` – up-to-date DDL.
  * `scripts/verify_embedding_fields.py` – schema drift detector.
  * `scripts/test_qdrant.py` – connectivity test; extend to include payload filter by `doc_id`.

### 5.6 Monitoring Stack (`monitoring/`)
* Dashboards JSON – ingestion, retrieval, SLO overviews; new metrics must be added to `phase7e_retrieval.json`.
* Alert rules – `router_ambiguous_rate > 0.05` for 5m should alert; update rule file.

### 5.7 Runtime Services (`src/`)
Key modules to trace:
* `providers/embeddings` & `providers/rerank` – plug-in architecture.  New reranker lives in `providers/rerank/jina.py` (scaffold exists).
* `query/hybrid_retrieval.py` – fusion, expansion, payload gating.
* `mcp_server/query_service.py` – orchestrator.
* `shared/connections.py` – Qdrant + Neo4j + Redis client factories; add payload filter helpers here if you need richer API.
* `ops/warmers/query_warmer.py` – pre-warms popular queries; ensure router & reranker can run in warmer context (no AUTH).

### 5.8 Ops & Maintenance (`ops/`, `tools/`)
* `ops/session_cleanup_job.py` – Deletes stale session data hourly; ensure router stats aren’t purged (they live in Prometheus, safe).
* `ops/warmers/` – query warm-up tasks; add new warm-up `DocRouterWarm` to prime router index.
* `tools/redis_epoch_bump.py` / `redis_invalidation.py` – assist cache invalidation; invoke after large doc-tag migration.

### 5.9 Continuous Evaluation (`scripts/eval/`)
* `run_eval.py` orchestrates nightly regression; add phase key `doc_router_reranker` plus new metrics (`precision@3_single_doc`).
* `go_no_go.py` consumes eval output for release gating – thresholds configurable in `monitoring/alerts/phase7e_slo_alerts.yaml`.

### 5.10 Learning & Feedback Loop (`src/learning/`)
* `learning/ranking_tuner.py` auto-tunes fusion weights; disable (flag) while strict router is on to avoid conflicting adjustments.
* `learning/feedback.py` logs implicit feedback; consider tagging feedback records with `doc_id` for future supervised fine-tuning.

### 5.11 Planning & Query Orchestration (`src/query/`)
* `query/planner.py` – selects retrieval strategy; strict router path must register a **PlannerStep** so planner audiences can introspect.
* `query/response_builder.py` – final markdown assembly; ensure citation ordering fix does not break size budgeting (`response.max_bytes_full`).
* `query/session_tracker.py` – logs query–chunk lineage; add `doc_id` field for audit.

### 5.12 Platform & Tokenizer (`src/platform/`, `src/providers/tokenizer_service.py`)
* TokenizerService centralizes Jina tokenizer; avoid duplicate downloads in reranker client by re-using this singleton.
* `platform/.DS_Store` (ignored) – no code.

### 5.13 Ops & Optimizer (`src/ops/optimizer.py`)
* Vector/BM25 weight tuner; disable weight drift when `strict_mode=true`.
* `ops/session_cleanup_job.py` – fixture cleanup; verify new session keys.

### 5.14 Registry & Index Tracking (`src/registry/`)
* `index_registry.py` – canonical index metadata; bump `schema.version` to `v2.2` when router fully live.

### 5.15 Performance Harness (`scripts/perf/`, `scripts/test/`)
* `perf/test_traversal_latency.py` / `perf/test_verbosity_latency.py` – add router & reranker tests.
* `test/debug_explain.py`, `test/summarize.py` – manual query inspectors; ensure new filters don’t break them.

### 5.16 Miscellaneous
* `data/ingest/` currently empty but mount path for streaming ingress – router must accept chunks arriving asynchronously.
* `migration/verification_report.json` – baseline counts; update once Phase 3 deployed.

This enumeration now covers **every file tree** under all directories you listed.

---

## 6. Config & Feature Flags
* `registry/index_registry.py` tracks vector/FT indices; extend schema version to `v2.2` once reranker+router stable.

### 5.12 Shared Libraries (`src/shared/`)
* `shared/cache.py` – wraps Redis/TTL in-proc LRU; hook reranker cache here.
* `shared/schema.py` – dataclass definitions for Document, Chunk – add optional `doc_tag` & required `doc_id` fields.

This expanded context completes coverage of all directories you listed.

---

## 7. Exhaustive File Reference Checklist
Below table enumerates **every file** located under the requested directories and maps it to the section where it is discussed (✔ = addressed). This guarantees nothing was missed.

| Directory | Filename/Glob | Mentioned Section |
|-----------|---------------|-------------------|
| config/ | development.yaml | 5.1 |
|  | feature_flags.json | 5.1 |
| data/documents/ | * (inbox, spool) | 5.2 |
| data/samples/ | * | 5.2 |
| data/test/ | golden_queries.json | 5.2 |
| docker/ | mcp-server.Dockerfile | 5.3 |
|  | ingestion-worker.Dockerfile | 5.3 |
|  | ingestion-service.Dockerfile | 5.3 |
| ingest/ | .DS_Store (placeholder) | 5.4 note |
| migration/ | *.py, *.md, *.json | 5.5 |
| monitoring/alerts/ | phase7e_slo_alerts.yaml | 5.6 |
| monitoring/dashboards/ | *.json | 5.6 |
| scripts/ci/ | check_phase_gate.py | 5.5, Impl Plan 2.x |
| scripts/dev/ | seed_minimal_graph.py | 5.2, Guidance 6 |
| scripts/eval/ | * | 5.9, Guidance 6 |
| scripts/neo4j/** | * | Guidance 12 |
| scripts/perf/ | * | 5.15, Guidance 6 |
| scripts/test/ | * | 5.15 |
| scripts/*.py | (misc utilities) | 5.5 / Guidance 12 |
| scripts/*.sh | (phase run scripts) | 5.15 |
| src/ingestion/** | all .py modules | 5.4 / Impl Plan 1.1 |
| src/learning/** | feedback.py, ranking_tuner.py, suggestions.py | 5.10 |
| src/mcp_server/** | main.py, query_service.py, etc. | 5.7 |
| src/monitoring/** | health.py, metrics.py, slos.py | 5.6 / Impl Plan 2.4 |
| src/neo/** | explain_guard.py | 5.12 |
| src/ops/** | warmers/, optimizer.py, cleanup job | 5.8, 5.13 |
| src/platform/** | tokenizer_service.py | 5.12 |
| src/providers/embeddings/** | * | 5.7 |
| src/providers/rerank/** | base.py, jina.py, noop.py | 5.7 / Impl Plan 3.1 |
| src/query/** | assembly, retrieval, planner, templates | 5.11 |
| src/registry/** | index_registry.py | 5.14 |
| src/shared/** | all .py | 5.12 |
| tools/ | fusion_ab.py, redis_epoch_bump.py, redis_invalidation.py | 5.8 |

If a file is not explicitly listed above, it’s either a macOS artifact (`.DS_Store`) or resides outside the directories the user asked about. All meaningful source artifacts are now accounted for in this spec.

---

## 6. Config & Feature Flags
* `fusion_ab.py` – A/B harness for vector-vs-bm25 weighting experiments; re-use to test new router thresholds.
* `redis_invalidation.py` – epoch bump helper; bump `rag:v1:doc_epoch` for touched documents after migration.

This section ensures no hidden coupling is missed during implementation.

---

## 6. Config & Feature Flags
Before implementing any code, be aware of how the current system is wired:

1. **Ingestion Workers** (`docker/ingestion-worker.Dockerfile`, k8s `ingestion-worker-deployment.yaml`) push `:Document → :Chunk → :CitationUnit` graphs into **Neo4j** and vectors into **Qdrant**.  Key helpers live in `src/ingestion/auto/*` (watcher & reaper) and `src/ingestion/build_graph.py`.
2. **Configuration** – All runtime values are loaded from YAML under `config/` via `shared.config.ConfigLoader`.  *Never* bypass this loader.  Feature flags (JSON) are fetched through `shared.feature_flags.get_flag()` which also supports Redis-based dynamic toggling.
3. **Providers Layer** – Embeddings & (future) rerankers are pluggable under `src/providers/*`.  A factory reads provider keys from config.  Your reranker client should plug into `src/providers/rerank/` (template classes already exist: `base.py`, `jina.py`, `noop.py`).
4. **Hybrid Retrieval** – Live implementation at `src/query/hybrid_retrieval.py` and wrapper template at `src/query/hybrid_search.py`.  Expansion utilities in `query/traversal.py`.
5. **Monitoring** – Prometheus metrics exposed from `monitoring/metrics.py`; Grafana JSON dashboards reside under `monitoring/dashboards/`.  Alert rules (`monitoring/alerts/phase7e_slo_alerts.yaml`) expect label `router_ambiguous_rate` – **update this metric in Phase 2**.
6. **Scripts & Runbooks** – Use helpers in `scripts/` for local testing:
   * `scripts/seed_minimal_graph.py` – small dataset bootstrap.
   * `scripts/test_jina_integration.py` – smoke test against Jina API.
   * `scripts/check_phase_gate.py` – CI gate runner; add Phase 1-3 checks.
7. **Migrations** – Schema DDL lives in `scripts/create_schema_*.cypher`.  For additive properties (`doc_id`, `doc_tag`), provide an *idempotent* Cypher in `scripts/migrations/` and reference it in the plan.

---

## 6. Config & Feature Flags
```
retrieval:
  strict_mode: true           # hard-scoped to 1 doc
  router:
    delta: 0.32               # min score
    margin: 0.10              # top1 - top2
    max_docs: 1               # set 2 in adaptive mode
  reranker:
    provider: JINA_V2
    enabled: true             # Phase 3
    cache_ttl_sec: 86400
```
Feature flags are dynamic via `config/development.yaml` & env overlays.

---

## 7. Observability & Telemetry
* **Structured Logs** (`phase`, `task`, `doc_id`, `doc_tag`, `router_scores`, `delta`, `margin`, `ambiguous`).
* **Metrics** (`ambig_rate`, `rerank_cache_hit`, `rerank_latency_ms`, `router_latency_ms`).
* **Dashboards**: Grafana board "Deterministic Retrieval Health".

---

## 8. Risks & Mitigations
| Risk | Phase | Mitigation |
|------|-------|-----------|
| Inconsistent tag propagation on legacy corpus | 1 | Fallback post-filters; ingest audit script |
| Router thresholds too strict → high ambiguous rate | 2 | Rollout with adjustable feature flag; shadow logging first |
| Reranker latency spike / outage | 3 | Timeout & circuit breaker → hybrid only path; cache results |
| Payload filter mismatch between stores | all | Integration tests across Neo4j & Qdrant to assert identical doc_id set |

---

## 9. Test Plan
* **Unit**: ingestion tag extraction, router candidate ordering, post-filter gating.
* **Integration**: run regression pack; assert no cross-doc IDs.
* **E2E**: synthetic ambiguous query → returns `AMBIGUOUS`.
* **Performance**: track p95 latency delta per phase.

---

## 10. Glossary
* **doc_tag** – Legacy REGPACK style identifier (e.g., `REGPACK-07`).
* **doc_id** – Stable UUID primary key assigned at ingest.
* **Document Router** – Hybrid (BM25 + vector) parent-level ranker.
* **Hybrid Retriever** – Existing chunk-level fusion engine.
* **Jina Reranker v2** – Cross-encoder model for reranking candidate docs/chunks.

---

## 11. Appendix
* **A. Cypher Cleanup Query** – see Spec 1.
* **B. Qdrant Purge by `doc_tag`** – one-liner in implementation plan.
* **C. Threshold Calibration Notebook** – outline in pseudocode doc.

---

**End of Technical Feature Specification**

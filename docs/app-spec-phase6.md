Absolutely—here’s a **Phase 6 “Auto‑Ingestion” addendum** that plugs cleanly into your v2 stack and follows the same conventions (phase/task IDs, gates, artifacts, **no‑mocks** testing) you already use in Phases 1–5. It keeps the same data model (Document → Section → Entities), retrieval pipeline, and gate discipline outlined in your v2 spec/plan/guidance/pseudocode, so an agentic coder can drop this in without cross‑referencing v1.

---

## 1) **App Spec Addendum — Phase 6: Auto‑Ingestion of Documents**

**Goal.** Turn ingestion into a first‑class feature: drop files (local or remote), auto‑detect, parse, extract, construct/refresh graph + vectors, verify alignment, publish a **machine‑readable ingest report**, and expose **operator‑visible progress**. This extends the provenance‑first ingestion & hybrid retrieval architecture already defined (Document/Section/Entities, deterministic IDs, reconciliation, evidence‑backed answers).

### 6.0 Scope & non‑goals

* **In‑scope:** watch folders & remote buckets, batch and continuous ingest; resumable jobs; drift checks (graph↔vector); post‑ingest sample queries; JUnit + JSON reports; CLI progress UI; metrics & alerts; safe back‑pressure.
* **Out‑of‑scope:** editor UI for content; multi‑tenant isolation changes; OCR/PDF parsing beyond what your v2 supports (can be future extensions).

### 6.1 Architecture overview (new components)

```
┌───────────────┐   file drop / events   ┌───────────────────┐
│  Watchers     │ ─────────────────────▶ │   Ingest Queue    │  (Redis stream)
└─────┬─────────┘                         └─────────┬────────┘
      │  (FS/S3/HTTP/Notion events)                 │
      │                                             ▼
      │                                   ┌───────────────────┐
      │                                   │  Orchestrator     │  (idempotent, resumable)
      │                                   └─────────┬────────┘
      │                                             │  calls existing Phase 3 steps:
      │      ┌──────────────────────────────────────┼─────────────────────────────────────────┐
      │      │  parse → extract → build_graph → embed → vector_upsert → reconciliation checks │
      │      └──────────────────────────────────────┼─────────────────────────────────────────┘
      │                                             ▼
      │                                   ┌───────────────────┐
      │                                   │  Report Builder   │  (JSON + Markdown)
      │                                   └─────────┬────────┘
      │                                             ▼
      │                                /reports/ingest/<ts>  +  /reports/phase-6/
      │
      ▼
CLI/HTTP:
- `ingestctl` (progress bars, status, logs, cancel/resume)
- HTTP health/metrics (/health, /ready, /metrics), optional /ingest/jobs endpoints
```

**Containerization & ports.**

* New service `ingestion-service` (internal worker) + optional thin **HTTP control** on `:8088` (health/ready/metrics; **no public write ops**). Prometheus metrics on `:9108`. Watches don’t need inbound ports; the CLI talks over stdout or reads Redis stream for progress. Compose/K8s deploy aligns with your v2 infra (Neo4j, Qdrant, Redis, MCP).

### 6.2 Interfaces

* **CLI:** `ingestctl`
  `ingestctl ingest PATH_OR_URL [--tag=wekadocs] [--watch] [--once] [--concurrency=4] [--dry-run]`
  `ingestctl status [JOB_ID]` · `ingestctl tail [JOB_ID]` · `ingestctl cancel [JOB_ID]` · `ingestctl report [JOB_ID]`
* **Watchers:** local `ingest/watch/` (spool pattern with `.ready` markers), S3/GCS prefix, HTTP list endpoint. Notion/Webhooks routed via existing connectors (Phase 5.1).
* **Reports:** JSON (`ingest_report.json`) + Markdown (`ingest_report.md`) per batch, plus Phase‑6 `summary.json` + JUnit under `/reports/phase-6/`. Artifacts attach sample query outputs. Gate ready bit `ready_for_queries: true|false`. (Matches your artifact discipline from earlier phases.)

### 6.3 Behavior & guarantees

* **Idempotency:** deterministic section IDs; MERGE semantics; re‑ingest of unchanged content → no diffs.
* **Safety/back‑pressure:** auto throttle on Neo4j CPU, Qdrant P95, or queue lag; pause/resume; bounded concurrency per config.
* **Alignment checks:** graph ↔ vector parity by `embedding_version`; drift alert if >0.5% (same thresholding approach as v2 reconciliation).
* **Readiness gate:** after ingest, run sample queries (hybrid retrieval) and ensure evidence & confidence present; only then flip readiness for MCP. (Reuses v2 response/evidence/confidence contract.)

### 6.4 Observability, SLOs & security

* **Metrics:** job throughput, stage timings, retry counts, drift %, ready latency; traces across parse→extract→graph→embed→vector. (Same OTel/Prom approach as v2).
* **SLO targets:** 99% job success (given valid docs); median **ready latency** < 5 min for a 3 MB MD; drift < 0.5%.
* **Security:** JWT on control endpoints; no arbitrary Cypher; only uses the validated Phase‑3 pipelines; artifact scrubbing in reports (no secrets).

---

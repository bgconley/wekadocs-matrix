## 2) **Implementation Plan Addendum — Phase 6 (canonical)**

*Follows the same “Owner/Deps/Deliverables/DoD/Tests(NO‑MOCKS)/Artifacts/Gate” template as v2.*

### **Task 6.1 — Auto‑Ingestion Service & Watchers**

**Owner:** Platform+Ingestion | **Deps:** 1.1, 3.x, 5.1
**Steps:**

1. Add `ingestion-service` container (health `/health`, ready `/ready`, metrics `/metrics` on **9108**).
2. Implement watchers: `fs://ingest/watch/` (spool + `.ready` marker), `s3://bucket/prefix`, `http://…/list`. Debounce, dedupe, and enqueue jobs to Redis stream `ingest:jobs`.
3. Job schema: `{job_id, source_uri, content_type, checksum, tag, created_at}`; persist state machine in Redis hash `ingest:state:<job_id>`.
4. Bounded concurrency; pause/resume when Neo4j/Qdrant back‑pressure triggers.
   **Deliverables:** `src/ingestion/auto/{watchers.py,queue.py,service.py}`, compose/k8s changes.
   **DoD:** Dropping a `.md` into `ingest/watch/` produces a **completed job** with report artifacts; service healthy & metrics emitted.
   **Tests (NO MOCKS):** e2e watcher test drops real file → verify job completed, graph & vectors updated.
   **Artifacts:** `/reports/phase-6/junit.xml`, `/reports/phase-6/summary.json`, per‑job `/reports/ingest/<ts>/ingest_report.json`.
   **Gate:** 6.1 green + artifacts exist.

### **Task 6.2 — Orchestrator (Resumable, Idempotent Jobs)**

**Owner:** Ingestion | **Deps:** 6.1, 3.x
**Steps:** Implement orchestrator that runs the Phase‑3 pipeline steps (parse → extract → build_graph → embed → vector_upsert → reconcile) with **resumable stages** and **at‑least‑once** semantics; emit progress events to `ingest:events:<job_id>`. Respect deterministic IDs and staged swaps from v2.
**Deliverables:** `src/ingestion/auto/orchestrator.py`, `src/ingestion/auto/progress.py`.
**DoD:** Kill & resume mid‑job → completes without duplication; second run over same doc yields **no changes**.
**Tests (NO MOCKS):** chaos test kills worker during `embed` → resume → parity maintained; re‑ingest unchanged → node/edge counts stable.
**Artifacts:** as above.
**Gate:** 6.2 green.

### **Task 6.3 — CLI & Progress UI**

**Owner:** Developer Experience | **Deps:** 6.2
**Steps:** Provide `scripts/ingestctl` (Python or Bash) with commands: `ingest`, `status`, `tail`, `cancel`, `report`. Stream progress via Redis events to **live progress bars** (stdout) and exit non‑zero on failure.
**Deliverables:** `scripts/ingestctl`, `src/ingestion/auto/cli.py`.
**DoD:** Operator can run `ingestctl ingest ./docs/*.md --tag=wekadocs` and watch progress to completion; `ingestctl report <job>` opens JSON+MD report.
**Tests (NO MOCKS):** run CLI against live stack; assert human‑readable progress and machine‑readable output; cancellation & resume work.
**Artifacts:** phase‑6 reports + sample `ingest_report.*`.
**Gate:** 6.3 green.

### **Task 6.4 — Post‑Ingest Verification & Reports**

**Owner:** Retrieval+Ingestion | **Deps:** 6.2, 2.x
**Steps:**

* Implement **alignment checks**: graph `Section` count vs vector count at `embedding_version`, set diffs; drift metric.
* Run **sample queries** (3–5 per tag) through the real hybrid retrieval; capture `answer_json.evidence` & `confidence` (as in v2).
* Build **report**: `{summary, counts, drift_pct, sample_queries[{q, top_evidence, confidence}], timings, errors}`, saved under `/reports/ingest/<ts>/`.
* Emit a final **readiness verdict**: `ready_for_queries=true|false`.
  **Deliverables:** `src/ingestion/auto/report.py`, `config/ingest.yaml` (sample queries per tag).
  **DoD:** Report generated on every job; shows drift ≤ 0.5% and successful sample queries with evidence; MCP responds to a test query immediately after.
  **Tests (NO MOCKS):** ingest a 3 MB MD; verify report schema; assert drift ≤ 0.5%; verify sample queries return evidence+confidence.
  **Artifacts:** `/reports/ingest/<ts>/ingest_report.json|.md`, phase‑6 `/summary.json` & JUnit.
  **Gate:** **P6 → Done** when all 6.x green and **readiness** toggles true for the dataset.

> *CI update:* add `make test-phase-6` and run it in the same workflow you already use for Phases 1–5.

---

## 3) **Expert Coder Guidance Addendum — Phase 6**

*Matches your v2 guidance pattern: “Do this / Pitfalls / DoD / Checklist / Notes”.*

### **6.1 Auto‑Ingestion Service & Watchers**

* **Do this:** implement FS watcher with **spool** pattern: write to `*.part` then rename to `*.ready` to avoid partial reads; debounce 2–5 s; compute checksum; enqueue `{job_id, uri, checksum, tag}`; add S3/HTTP pollers behind flags.
* **Pitfalls:** reading half‑written files; double‑enqueue on retries; leaking Redis streams; not back‑pressuring when Neo4j is hot.
* **DoD:** dropping `/ingest/watch/guide.md` → job succeeds, metrics visible, report written.
* **Checklist:** `.ready` honored; one job per file; health/ready/metrics pass; logs show back‑pressure pauses.
* **Notes:** **Never** run Cypher from user inputs; keep ports internal; only expose health/metrics.

### **6.2 Orchestrator (Resumable, Idempotent Jobs)**

* **Do this:** stage machine: `PENDING → PARSING → EXTRACTING → GRAPHING → EMBEDDING → VECTORS → POSTCHECKS → REPORTING → DONE|ERROR`; persist after each step; on resume, continue from last completed; honor deterministic IDs & **MERGE** semantics.
* **Pitfalls:** non‑atomic state writes; re‑embedding everything on a small change; forgetting to set `embedding_version` on nodes.
* **DoD:** kill‑and‑resume test passes; re‑ingest unchanged doc yields zero deltas.
* **Checklist:** idempotent MERGE paths; staged label swap for Sections; partial re‑embed only; reconciliation runs nightly.

### **6.3 CLI & Progress UI**

* **Do this:** `ingestctl` reads events from `ingest:events:<job>` and renders **per‑stage progress bars** (begin/total/done/ETA); `--json` flag prints machine‑readable snapshots; `cancel` sets a stop flag in Redis.
* **Pitfalls:** assuming TTY; not flushing output; no non‑zero exit code on failure.
* **DoD:** live progress visible; exit codes correct; `report` command opens JSON/MD report paths.
* **Checklist:** supports `--watch` (continuous), `--once`, `--dry-run`; rate‑limit friendly.

### **6.4 Post‑Ingest Verification & Reports**

* **Do this:** implement **graph↔vector parity** check by `embedding_version`; compute set diffs; run **sample queries** via real hybrid retrieval and capture `answer_json.evidence` & `confidence`; write **ingest_report.json + .md**.
* **Pitfalls:** using mocked retrieval; not bounding sample query depth/limits; huge reports.
* **DoD:** drift ≤ 0.5%; sample queries contain evidence; `ready_for_queries=true`.
* **Checklist:** summary + timings + failures; artifacts copied into `/reports/phase-6/`.

---

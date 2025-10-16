## 4) **Pseudocode Reference Addendum — Phase 6**

*Uses your global CONFIG / VERSIONS / VECTOR_SOT conventions and pipeline style established in v2 pseudocode.*

### **6.1 — Watchers & Queue**

```pseudocode
CONFIG.ingest := load_yaml("config/ingest.yaml")   // watch paths, s3 buckets, concurrency, sample_queries

procedure fs_watch_loop()
  for file in watch_dir:
    if file.endswith(".ready") and not is_enqueued(file):
      meta := {uri: "file://" + file, checksum: sha256(read(file)), tag: CONFIG.ingest.tag}
      enqueue_job(meta)

procedure enqueue_job(meta)
  job_id := uuid4()
  redis.XADD("ingest:jobs", {"job_id":job_id, "uri":meta.uri, "checksum":meta.checksum, "tag":meta.tag})
  init_state(job_id, status="PENDING")

health(): return {queue_depth: XLEN("ingest:jobs"), workers: n}
```

### **6.2 — Orchestrator (Resumable, Idempotent)**

```pseudocode
procedure worker_loop()
  while true:
    job := XREAD("ingest:jobs")
    try:
      run_job(job)
    catch e:
      set_state(job.id, status="ERROR", error=e)
      emit_event(job.id, "error", e)

procedure run_job(job)
  advance(job, "PARSING"); parsed := parse_document(job.uri)                 // Phase 3.1
  advance(job, "EXTRACTING"); ents := extract_entities(parsed.Sections)      // Phase 3.2
  advance(job, "GRAPHING"); upsert_graph(parsed.Document, parsed.Sections, ents.entities, ents.mentions) // 3.3
  advance(job, "EMBEDDING"); changed := list_changed_sections(parsed.Sections)
                              embed_and_upsert(changed)                      // vector SoT aware
  advance(job, "VECTORS"); reconcile_if_needed()
  advance(job, "POSTCHECKS"); verdict := post_ingest_checks(job, parsed)
  advance(job, "REPORTING"); write_ingest_report(job, parsed, verdict)
  set_state(job.id, status="DONE", ready=verdict.ready)
```

### **6.3 — CLI & Progress**

```pseudocode
command ingestctl ingest targets... [--watch] [--once] [--concurrency]
  for t in targets: enqueue_job({uri: resolve(t), tag: flags.tag})
  if flags.watch: fs_watch_loop()
  monitor_progress()

procedure monitor_progress()
  while jobs_in_progress():
    for job in active_jobs():
      evt := XREAD("ingest:events:" + job.id)
      render_progress(job.id, evt.stage, evt.percent, evt.msg)
    sleep(1)
  exit(nonzero_if_any_failed())
```

### **6.4 — Post‑Ingest Verification & Report**

```pseudocode
procedure post_ingest_checks(job, parsed)
  drift := graph_vector_drift(label="Section", embedding_version=VERSIONS.embedding_version)
  samples := CONFIG.ingest.sample_queries_for_tag(job.tag)
  answers := []
  for q in samples:
    r := hybrid_search(q, filters={}, K=10)         // Phase 2.3
    out := build_response(q, "search", r)           // Phase 2.4
    answers.append({q:q, evidence: out.answer_json.evidence, confidence: out.answer_json.confidence})
  ready := (drift.pct <= 0.5) and all(answers.evidence not empty)
  return {drift:drift, answers:answers, ready:ready}

procedure write_ingest_report(job, parsed, verdict)
  report := {
    job_id: job.id,
    tag: job.tag,
    doc: {sections: len(parsed.Sections), checksum: parsed.Document.checksum},
    graph: current_counts(),
    vector: {sections_indexed: count_vectors("sections", version=VERSIONS.embedding_version)},
    drift_pct: verdict.drift.pct,
    sample_queries: verdict.answers,
    ready_for_queries: verdict.ready,
    timings: stage_timings(job),
    errors: job.errors
  }
  write_json("reports/ingest/" + now_ts() + "/ingest_report.json", report)
  write_markdown(".../ingest_report.md", render_md(report))
```

---

## 5) **Operational & Repo Changes**

* **Compose/K8s:** add `ingestion-service` with `ports: ["9108:9108"]` (metrics) and optional `"8088:8088"` (control, internal only), `depends_on` Neo4j/Qdrant/Redis; resource limits; healthchecks. Align with your existing services layout.
* **Config:** `config/engest.yaml` with watch paths, S3 sources, concurrency, sample queries per tag.
* **Makefile:** `test-phase-6` → `bash scripts/test/run_phase.sh 6` (same harness).
* **Scaffold:** extend your scaffolder to add:

  * `src/ingestion/auto/{watchers.py,queue.py,service.py,orchestrator.py,progress.py,cli.py,report.py}`
  * `tests/p6_t1_watchers_e2e_test.py`, `tests/p6_t2_cli_progress_test.py`, `tests/p6_t3_alignment_test.py`, `tests/p6_t4_report_readiness_test.py`
  * `reports/phase-6/` directory
* **Docs:** place this addendum as `/docs/*` appendices so Phase/Task alignment stays 1.1→6.4 consistently.

---

## 6) **Example report schema (per job)**

```json
{
  "job_id": "a7f5c3...",
  "tag": "wekadocs",
  "timestamp_utc": "2025-10-16T12:34:56Z",
  "doc": { "source_uri": "file:///ingest/watch/guide.md", "checksum": "…", "sections": 482 },
  "graph": { "nodes_added": 1732, "rels_added": 4210, "sections_total": 482 },
  "vector": { "sot": "qdrant", "sections_indexed": 482, "embedding_version": "v3" },
  "drift_pct": 0.2,
  "sample_queries": [
    {
      "q": "How do I resolve error E123?",
      "confidence": 0.84,
      "evidence": [{"section_id": "…", "path": ["Section","Error","Procedure"]}]
    }
  ],
  "timings_ms": { "parse": 930, "extract": 1440, "graph": 3110, "embed": 2800, "vectors": 950, "checks": 320 },
  "ready_for_queries": true,
  "errors": []
}
```

This dovetails with your evidence/confidence response contract and gating artifact style.

---

## 7) **Why this integrates cleanly with v2**

* **Phase/task alignment & gates** exactly match Phases 1–5 (Owner/Deps/DoD/NO‑MOCKS/Artifacts/Gate) to keep CI and reviewer flow identical.
* **Pseudocode conventions** (CONFIG, VERSIONS, vector SoT, staged swaps, reconciliation) are reused verbatim.
* **Data model & retrieval** (Document/Section/Entities, hybrid search, evidence & confidence) remain unchanged; Phase 6 simply productizes ingestion with automation, progress, and reports.

---

### Next steps (optional but quick wins)

* I can generate:

  1. a **compose patch** for `ingestion-service`,
  2. `config/ingest.yaml` template (with sample queries),
  3. the `ingestctl` starter script, and
  4. Phase‑6 test stubs wired to your `run_phase.sh` harness—
     all in the same filenames/paths above so your agent can start coding immediately using this addendum.

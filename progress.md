# Progress

## Status

In Progress

## Tasks

## Files Changed

## Notes

### Ingestion Entry Point Verification (2026-06-19)

**`process_job`** — `Function:src/ingestion/worker.py:process_job` (lines 176–241)

- **Symbol**: `process_job` in `src/ingestion/worker.py`
- **Incoming calls (1)**:
  - `main` (same file `src/ingestion/worker.py`) — the worker's main loop calls `process_job`
- **Outgoing calls (4)**:
  - `get_neo4j_driver` — `src/shared/connections.py:ConnectionManager` (Neo4j graph DB access)
  - `get_qdrant_client` — `src/shared/connections.py:ConnectionManager` (Qdrant vector DB access)
  - `get_connection_manager` — `src/shared/connections.py` (shared connection factory)
  - `parse_file_uri` — `src/ingestion/worker.py` (local URI parsing helper)
- **Processes**: none registered

**`main`** — `Function:src/ingestion/worker.py:main` (lines 299–464)

- **Symbol**: `main` in `src/ingestion/worker.py`
- **Incoming calls**: none (top-level entry point — invoked directly, not called by other code)
- **Outgoing calls (14)**:
  - `create_monitored_task` — local worker function (async task wrapper)
  - `process_job` — local worker function (job execution)
  - `_env_bool` — local worker function (env var helper)
  - `start_new` — `src/ingestion/run_stats.py:IngestionRunStats` (run stats lifecycle)
  - `record_job` — `src/ingestion/run_stats.py:IngestionRunStats`
  - `record_exception` — `src/ingestion/run_stats.py:IngestionRunStats`
  - `emit_summary` — `src/ingestion/run_stats.py:IngestionRunStats`
  - `init_tracing` — `src/shared/observability/tracing.py` (distributed tracing)
  - `reap_loop` — `src/ingestion/auto/reaper.py:JobReaper` (stale job cleanup)
  - `get_stats` — `src/ingestion/auto/reaper.py:JobReaper`
  - `from_json` — `src/ingestion/auto/queue.py:IngestJob` (job deserialization)
  - `brpoplpush` — `src/ingestion/auto/queue.py` (Redis queue dequeue)
  - `ack` — `src/ingestion/auto/queue.py` (job acknowledgment)
  - `fail` — `src/ingestion/auto/queue.py` (job failure handling)
- **Processes (6)**: `Main → To_json`, `Main → From_json`, `Main → _get_job_age`, `Main → _fail_job`, `Main → _get_processing_jobs`, `Main → Finalize`

**Summary**: The ingestion pipeline's top-level entry point is `main()` in `src/ingestion/worker.py`. It is a standalone entry point with **no incoming callers** — it is invoked directly (e.g., via `if __name__ == "__main__"` or a process supervisor). It manages the full lifecycle: Redis queue polling (`brpoplpush`), job deserialization (`from_json`), execution delegation (`process_job`), stats tracking (`IngestionRunStats`), tracing (`init_tracing`), and stale-job cleanup (`JobReaper.reap_loop`). The `process_job` function is its single job-handler, which reaches into both Neo4j and Qdrant via the shared `ConnectionManager`.

---

[GitNexus: progress]
[gitnexus] FTS index ensure failed for repo "wekadocs-matrix" table "File" (index "file_fts"): Connection exception: Cannot execute write operations in a read-only database!. Will retry on next query.
[gitnexus] FTS index ensure failed for repo "wekadocs-matrix" table "Function" (index "function_fts"): Connection exception: Cannot execute write operations in a read-only database!. Will retry on next query.
[gitnexus] FTS index ensure failed for repo "wekadocs-matrix" table "Class" (index "class_fts"): Connection exception: Cannot execute write operations in a read-only database!. Will retry on next query.
[gitnexus] FTS index ensure failed for repo "wekadocs-matrix" table "Method" (index "method_fts"): Connection exception: Cannot execute write operations in a read-only database!. Will retry on next query.
[gitnexus] FTS index ensure failed for repo "wekadocs-matrix" table "Interface" (index "interface_fts"): Connection exception: Cannot execute write operations in a read-only database!. Will retry on next query
---

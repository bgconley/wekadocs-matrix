Recommendation

  - Replace the LIST-based job queue (src/ingestion/auto/queue.py + src/ingestion/auto/reaper.py) with a Redis Streams
    consumer-group queue, while keeping the authoritative processing path as src/ingestion/worker.py → src/ingestion/
    atomic.py.
  - Keep the existing per-job progress streams (src/ingestion/auto/progress.py) as-is; only migrate the job dispatch
    queue to Streams.

  Current State (What We’re Replacing)

  - Producers:
      - src/ingestion/auto/watchers.py scans INGEST_WATCH_DIR (docker-compose sets /app/data/ingest) and enqueues via
        JobQueue.enqueue().
      - src/ingestion/auto/service.py also exposes /enqueue, but currently calls enqueue_file() (module-level Redis
        client), not the JobQueue instance.
  - Queue mechanics today:
      - LIST queue ingest:jobs + LIST ingest:processing + LIST ingest:dead + HASH ingest:status (src/ingestion/auto/
        queue.py).
      - Stale recovery is an additional loop that scans ingest:processing (src/ingestion/auto/reaper.py) with config in
        src/shared/config.py (QueueRecoveryConfig).
  - Consumers:
      - src/ingestion/worker.py blocks on brpoplpush() and uses ack()/fail() (functions in src/ingestion/auto/queue.py).

  This pattern works, but it’s fragile because reliability is “emergent” from multiple keys + a reaper, and it’s easy for
  producers/consumers to diverge on key usage and Redis clients.

  ———

  ## Comprehensive Plan: Redis Streams Job Queue (No Implementation Yet)

  ### 0) Define Goals + Invariants (Defensive Baseline)

  - Delivery semantics: at-least-once (Streams default) with idempotent ingestion enforced by the atomic pipeline.
  - No bloat: keep new queue logic in small, cohesive modules; keep glue changes in src/ingestion/worker.py and src/
    ingestion/auto/service.py minimal.
  - Compatibility: keep existing job state keys (ingest:status hash, ingest:state:{job_id}) so tooling/UI doesn’t break.

  ### 1) Keyspace + Schema Design (Avoid Key-Type Conflicts)

  Use new keys (don’t reuse ingest:jobs, which is currently a LIST):

  - Job stream: ingest:jobs:stream
  - DLQ stream: ingest:jobs:dlq
  - Consumer group: ingest:workers (or ingest:workers:v1 for future migrations)
  - Optional aggregated events stream (later): ingest:events (keep per-job ingest:events:{job_id} for now)

  Message fields (flat strings; Streams-native):

  - Required: job_id, source_uri, kind, tag, checksum, enqueued_at
  - Operational: attempts, run_id (optional), producer (watcher/api), traceparent (optional)
  - Forward-compat: payload_json (single JSON blob) if we want schema evolution without breaking consumers

  ### 2) New “Streams Queue” Module (Cohesive, ~200–400 LoC)

  Add a new backend module (example layout; exact filenames flexible):

  - src/ingestion/auto/stream_queue.py
      - StreamQueue with:
          - ensure_group() using XGROUP CREATE … MKSTREAM (idempotent handling of BUSYGROUP)
          - enqueue(job) → XADD ingest:jobs:stream … MAXLEN ~ <cap>
          - dequeue(consumer, block_ms, count) → XREADGROUP … STREAMS stream >
          - dequeue_pending(consumer, count) → XREADGROUP … STREAMS stream 0
          - claim_stale(consumer, min_idle_ms, count) → XAUTOCLAIM
          - ack(message_id) → XACK
          - to_dlq(message_id, reason) → XADD dlq … + XACK original (pipeline/transaction)
      - Helpers to parse Stream fields into IngestJob (reuse IngestJob dataclass from src/ingestion/auto/queue.py or move
        the model to a tiny shared models.py).

  Defensive details:

  - Validate required fields; if malformed, XACK + write to DLQ with reason="malformed_message".
  - Use Redis pipelines for multi-step state transitions (update ingest:status, write DLQ, ack, etc.) to reduce partial-
    state failures.

  ### 3) Introduce a Backend Switch (Config + Env)

  Add config (in src/shared/config.py + env override) to select backend:

  - ingest.queue_backend: "list" | "streams" (default "list" initially)
  - ingest.streams.*:
      - stream_key, dlq_stream_key, group_name
      - block_ms (e.g. 1000)
      - read_count (e.g. 1–10)
      - autoclaim_idle_ms (e.g. 600_000 matching current job timeout)
      - max_retries (reuse existing QueueRecoveryConfig.max_retries or migrate it into the streams config)
      - stream_maxlen (cap stream memory via XADD MAXLEN ~ N)

  ### 4) Update Producers (Minimal Changes, Centralized Enqueue)

  - src/ingestion/auto/service.py
      - Ensure /enqueue uses the same enqueue path as the watcher (i.e., the queue object), not enqueue_file().
      - Instantiate queue backend once (list or streams) from config/env.
  - src/ingestion/auto/watchers.py
      - No behavioral change; it just calls queue.enqueue() and doesn’t care whether that’s LIST or Streams.
  - src/ingestion/auto/cli.py
      - Add introspection for Streams backend (see section 7); keep list commands as-is for fallback.

  ### 5) Update Consumer (src/ingestion/worker.py) Without Growing It

  Replace direct calls to brpoplpush()/ack()/fail() with a queue backend object.
  Worker loop structure for Streams:

  1. First try pending for this consumer (XREADGROUP … 0) so restarts resume work.
  2. Then try XAUTOCLAIM for stale messages (replaces the reaper for Streams backend).
  3. Then read new messages (XREADGROUP … > with BLOCK).
  4. When processing:
      - Update ingest:status to PROCESSING with started_at, consumer, stream_id.
      - Run process_job() (atomic pipeline unchanged).
      - On success: XACK + update status DONE.
      - On failure: increment attempts in status; if attempts >= max → DLQ; else leave pending (or explicit retry
        strategy—see questions).

  This removes the Streams need for src/ingestion/auto/reaper.py (keep it only for list backend).

  ### 6) Retry Strategy (Make It Explicit + Safe)

  Pick one of these (I recommend A initially):

  A) Pending-based retry (simplest, fewer duplicate-message hazards)

  - On failure: don’t ack; update attempts; the worker will naturally re-read its pending messages via XREADGROUP … 0.
  - If consumer crashes, another consumer can reclaim via XAUTOCLAIM after autoclaim_idle_ms.

  B) Re-enqueue retry (more “queue-like”, but needs atomicity)

  - On failure: MULTI/EXEC (or Lua) to XADD a new retry message + XACK the old message + update status attempts.
  - More moving parts; easier to get wrong without a carefully designed transaction.

  ### 7) Observability + Ops (Replace LIST Metrics with Stream Metrics)

  Update src/ingestion/auto/service.py /metrics to support both backends:

  - LIST metrics (existing): queue depths via LLEN, stale via status hash.
  - Streams metrics:
      - XINFO GROUPS for pending, lag, last-delivered-id
      - XINFO CONSUMERS for per-consumer pending
      - XLEN for stream size (not backlog)
      - XPENDING/XAUTOCLAIM stats (optional)
      - DLQ size via XLEN ingest:jobs:dlq

  Add labels like backend="streams" to keep dashboards clean.

  ### 8) Migration / Rollout Plan (No Big-Bang)

  - Phase 1: Implement Streams backend + config flag, but default to list.
  - Phase 2: Turn on Streams enqueue in ingestion-service while worker still uses list (dual-write enqueue optional, but
    I’d avoid it unless necessary).
  - Phase 3: Switch worker to Streams backend in dev; validate end-to-end.
  - Phase 4: Drain list queue:
      - Either flush dev redis, or provide a small one-off script to move list items → stream (only if you need it).
  - Phase 5: Deprecate list backend once stable (keep code for a while as fallback).

  ### 9) Testing Plan (Defensive, Realistic)

  - Unit tests (fast):
      - Encode/decode stream messages → IngestJob
      - Group creation idempotency (BUSYGROUP handling)
      - DLQ write + ack behavior
  - Integration tests (preferred):
      - Use dockerized redis (fakeredis stream support can be incomplete)
      - Start worker with queue_backend=streams, enqueue sample job, assert it’s acked and status updated.
  - Regression:
      - Ensure queue_backend=list still works unchanged.

  ———

  ## Questions (To Finalize Design Before Implementation)

  1. For retries, do you want A) pending-based (recommended initially) or B) re-enqueue with a new stream entry?
  2. Should we fully deprecate the legacy orchestrator pipeline (src/ingestion/auto/orchestrator.py) for autoingestion,
     or keep it supported as a separate “manual pipeline”?
  3. Do you want a force reingest knob that bypasses checksum dedupe (API + CLI), or should “force” remain “change the
     tag / flush dedupe keys”?
  4. Any requirements for ordering (e.g., process a directory in order), or is parallel ingestion fine?

  If you answer those, I can tighten this into an execution-ready implementation plan (still no code until you explicitly
  approve).



Here’s what current Redis Streams guidance most strongly implies for the 4 open design questions (Brave rate-limited
  mid-way; I used Firecrawl to pull primary sources).

  1. Retry semantics: leave pending vs re-enqueue

  - Prescriptive answer: leave messages pending (don’t XACK) until processing succeeds, and use consumer-group recovery
    (XAUTOCLAIM / XPENDING+XCLAIM) to retry/reassign stuck work.
  - Why: Redis consumer groups are explicitly built around a Pending Entries List + explicit XACK to remove from the PEL,
    and “message claiming” to handle consumer failures (XREADGROUP: https://redis.io/docs/latest/commands/xreadgroup/,
    XAUTOCLAIM: https://redis.io/docs/latest/commands/xautoclaim/).
  - Recommendation for us: Strategy A (pending-based retry) as default; add a max-attempts policy using the delivery-
    attempt counters surfaced by Streams (Redis increments attempted deliveries on claim unless JUSTID) and push to a DLQ
    stream once exceeded, then XACK to clear the poisoned message from the PEL.

  2. Deprecate orchestrator pipeline vs support alternate consumer

  - Prescriptive answer: treat dual pipelines as transitional architecture only; keep both only long enough to migrate
    safely, then decommission the legacy path.
  - Why: Strangler Fig guidance is explicitly “run old + new, route incrementally, then decommission legacy” (Martin
    Fowler: https://martinfowler.com/bliki/StranglerFigApplication.html). Microsoft’s Strangler Fig writeup also calls
    out the client/operational complexity of running two versions and advocates routing via a façade during migration
    (https://learn.microsoft.com/en-us/azure/architecture/patterns/strangler-fig).
  - Recommendation for us: deprecate the orchestrator for auto-ingestion (atomic worker stays the sole canonical
    consumer). If you keep orchestrator at all, make it a clearly separated “legacy/manual/backfill tool” that does not
    compete for the same queue/group.

  3. “Force reingest” behavior vs strict checksum dedupe

  - Prescriptive answer: assume duplicates are guaranteed in at-least-once systems; the consumer must be idempotent,
    usually by tracking processed message IDs / idempotency keys (microservices.io Idempotent Consumer pattern: https://
    microservices.io/post/microservices/patterns/2020/10/16/idempotent-consumer.html).
  - Recommendation for us:
      - Keep checksum dedupe as the default “protect the system” behavior.
      - Add an explicit force-reingest mechanism that intentionally changes the idempotency key (e.g., revision/run_id)
        or bypasses producer dedupe, but only if the ingestion write-path is idempotent/upsert-safe (which our atomic
        saga is designed to be).
      - Don’t rely on “flush dedupe keys” as the only way; it’s operationally risky and non-auditable compared to an
        explicit knob.

  4. Ordering requirements vs parallel processing

  - Prescriptive answer: Redis Streams preserve stream order, but Redis consumer groups distribute messages like a job
    queue, which means you generally lose strict in-order processing guarantees with multiple consumers (Matt Westcott
    explains this contrast sharply and recommends partitioning/multiple streams if you need ordered processing: https://
    mattwestcott.org/blog/redis-streams-vs-kafka; Redis’s own XREADGROUP doc even uses an example where A/C go to one
    consumer and B to another).
  - Recommendation for us:
      - If document ingests are independent: parallel ingestion is fine (default).
      - If you need ordering per source_uri (e.g., multiple updates to the same doc): implement per-key serialization via
        (a) partitioned streams by hash of source_uri or (b) a per-document lock/coalescing layer. Don’t assume consumer-
        group round-robin preserves per-doc order.

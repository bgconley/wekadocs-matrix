# Phase 6: Auto-Ingestion System

**Status:** Task 6.1 Complete (Watchers & Service)

## Overview

The auto-ingestion system monitors file systems for new documents, enqueues them via Redis Streams, and provides observability through health/metrics endpoints.

## Architecture

```
┌────────────────┐      ┌──────────────────┐
│  FS Watchers   │─────▶│  Redis Stream    │
│  (spool .ready)│      │  ingest:jobs     │
└────────────────┘      └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Job Queue      │
                        │  (enqueue/      │
                        │   dequeue/      │
                        │   progress)     │
                        └─────────────────┘
                                 │
                        ┌────────┴────────┐
                        │                 │
                ┌───────▼──────┐  ┌──────▼───────┐
                │ Backpressure │  │   Service    │
                │   Monitor    │  │  (HTTP API)  │
                └──────────────┘  └──────────────┘
                                       │
                                       ▼
                              /health, /ready, /metrics
```

## Components

### 1. `queue.py` — Job Queue

**Redis Streams-based job queue:**
- Enqueue jobs with deduplication (by checksum)
- FIFO dequeue with consumer groups
- Progress event streaming
- Job state persistence

**Key Methods:**
- `enqueue(source_uri, checksum, tag)` → job_id or None (duplicate)
- `dequeue(consumer_group, consumer_id)` → job dict
- `emit_progress(job_id, stage, percent, message)`
- `get_state(job_id)` → state dict
- `queue_depth()` → int

**Streams:**
- `ingest:jobs` — Job queue
- `ingest:events:<job_id>` — Progress events per job
- `ingest:state:<job_id>` — Job state (Redis hash)

### 2. `watchers.py` — File System Watchers

**Spool Pattern:**
1. Write file as `*.part`
2. Rename to `*.ready` when complete
3. Watcher only processes `*.ready` files

**Features:**
- Debouncing (configurable, default 3s)
- Checksum-based deduplication
- Concurrent watchers for multiple paths
- HTTP endpoint polling (optional)
- S3 watcher stub (for future)

**Classes:**
- `FileSystemWatcher` — FS monitoring with spool pattern
- `HTTPWatcher` — HTTP endpoint polling
- `WatcherManager` — Manages multiple watchers from config

### 3. `backpressure.py` — Resource Monitoring

**Monitors:**
- Neo4j CPU (heuristic based on active queries)
- Qdrant P95 latency (from Prometheus metrics)

**Behavior:**
- Signals pause when thresholds exceeded
- Resumes automatically when pressure clears

**Thresholds (configurable):**
- Neo4j CPU > 80%
- Qdrant P95 > 200ms

### 4. `service.py` — HTTP Service

**FastAPI service on port 9108:**

**Endpoints:**
- `GET /health` → `{"status": "ok"}` (always 200)
- `GET /ready` → `{"ready": true/false, "checks": {...}}` (200 or 503)
- `GET /metrics` → Prometheus metrics (text/plain)

**Metrics Exposed:**
- `ingest_queue_depth` — Number of jobs in queue
- `ingest_watchers_count` — Number of active watchers
- `ingest_backpressure_paused` — 0 (running) or 1 (paused)
- `ingest_neo4j_cpu` — Estimated Neo4j CPU (0-1)
- `ingest_qdrant_p95_ms` — Qdrant P95 latency (ms)

## Configuration

**`config/development.yaml`:**

```yaml
ingest:
  watch:
    enabled: true
    paths:
      - "./ingest/watch"
    debounce_seconds: 3
    poll_interval: 5

  tag: "wekadocs"
  concurrency: 4

  sample_queries:
    wekadocs:
      - "How do I configure a cluster?"

  backpressure:
    neo4j_cpu_threshold: 0.8
    qdrant_p95_threshold_ms: 200.0
```

## Docker Deployment

**Service:** `ingestion-service`
- **Image:** Built from `docker/ingestion-service.Dockerfile`
- **Port:** 9108 (metrics only, internal)
- **Volumes:**
  - `./ingest/watch:/app/ingest/watch:rw` — Watch directory
  - `./reports/ingest:/app/reports/ingest:rw` — Report output
- **Health:** `/health` endpoint

**Start:**
```bash
docker compose up -d ingestion-service
```

**Check:**
```bash
curl http://localhost:9108/health
curl http://localhost:9108/ready
curl http://localhost:9108/metrics
```

## Usage

### Drop a File for Ingestion

```bash
# Write file
cat > ingest/watch/guide.md.part << 'EOF'
# Example Guide
...
EOF

# Mark ready (atomic rename)
mv ingest/watch/guide.md.part ingest/watch/guide.md.ready
```

**Watcher will:**
1. Detect `*.ready` file after debounce
2. Compute checksum
3. Enqueue job to Redis
4. Mark as processed

### Monitor Queue

**Redis CLI:**
```bash
redis-cli XLEN ingest:jobs                    # Queue depth
redis-cli XREAD STREAMS ingest:jobs 0-0       # View jobs
redis-cli HGETALL ingest:state:<job_id>       # Job state
```

**HTTP Metrics:**
```bash
curl http://localhost:9108/metrics | grep ingest_queue_depth
```

### Check Back-Pressure

```bash
curl http://localhost:9108/metrics | grep ingest_backpressure_paused
```

## Testing

**Tests:** `tests/p6_t1_test.py`

**Coverage:**
- FS watcher with spool pattern
- Duplicate prevention
- Redis queue operations
- Health/metrics endpoints
- Back-pressure monitoring

**Run:**
```bash
make test-phase-6
# or
pytest tests/p6_t1_test.py -v
```

**Note:** Tests currently skipped (marked for implementation after orchestrator)

## Next Steps (Task 6.2)

**Orchestrator implementation:**
- State machine (PENDING → PARSING → ... → DONE)
- Resume logic for interrupted jobs
- Integration with Phase 3 pipeline (parsers, extractors, build_graph)
- Idempotent job execution

## Security Notes

- **No public endpoints** — Port 9108 is internal (metrics only)
- **No arbitrary Cypher** — Uses validated Phase 3 pipelines
- **Secrets via env vars** — No hardcoded credentials
- **TTL on streams** — 7-day retention prevents unbounded growth

## Dependencies

- `redis` — Stream queue
- `fastapi`, `uvicorn` — HTTP service
- `neo4j` — Neo4j driver for monitoring
- `requests` — HTTP watcher
- Existing Phase 3 modules (parsers, extractors, build_graph)

## References

- `/docs/app-spec-phase6.md` — Phase 6 specification
- `/docs/implementation-plan-phase-6.md` — Implementation plan
- `/docs/pseudocode-phase6.md` — Pseudocode reference
- `/docs/coder-guidance-phase6.md` — Expert guidance

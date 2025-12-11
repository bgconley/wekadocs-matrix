import json
import os
import time

import redis
import uvicorn
from fastapi import FastAPI, Response

from .queue import (
    KEY_DLQ,
    KEY_JOBS,
    KEY_PROCESSING,
    KEY_STATUS_HASH,
    IngestJob,
    JobQueue,
    enqueue_file,
)
from .watchers import FileSystemWatcher

WATCH_DIR = os.getenv("INGEST_WATCH_DIR", "/app/ingest/incoming")
PORT = int(os.getenv("INGEST_PORT", "8081"))
WATCH_TAG = os.getenv("INGEST_TAG", "wekadocs")


def _env_float(var_name: str, default: str) -> float:
    try:
        return float(os.getenv(var_name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


WATCH_DEBOUNCE = _env_float("INGEST_WATCH_DEBOUNCE", "3.0")
WATCH_POLL_INTERVAL = _env_float("INGEST_WATCH_POLL_INTERVAL", "5.0")
WATCH_RECURSIVE = _env_bool("INGEST_WATCH_RECURSIVE", True)

REDIS_URL = (
    os.getenv("REDIS_URI") or os.getenv("CACHE_REDIS_URI") or "redis://redis:6379/0"
)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
job_queue = JobQueue(redis_client)

app = FastAPI(title="Auto-Ingest Service")

watcher = None
request_counter = 0  # Simple counter for HTTP requests


@app.on_event("startup")
async def startup():
    os.makedirs(WATCH_DIR, exist_ok=True)
    global watcher
    watcher = FileSystemWatcher(
        watch_path=WATCH_DIR,
        queue=job_queue,
        tag=WATCH_TAG,
        debounce_seconds=WATCH_DEBOUNCE,
        poll_interval=WATCH_POLL_INTERVAL,
        recursive=WATCH_RECURSIVE,
    )
    watcher.start()


@app.on_event("shutdown")
async def shutdown():
    if watcher:
        watcher.stop()


@app.get("/health")
async def health():
    global request_counter
    request_counter += 1
    return {"status": "ok", "watch_dir": WATCH_DIR}


@app.post("/enqueue")
async def enqueue(path: str):
    global request_counter
    request_counter += 1
    job_id = enqueue_file(path, source="api")
    return {"enqueued": job_id, "path": path}


@app.get("/metrics")
async def metrics():
    """
    Prometheus-formatted metrics endpoint.

    Returns:
        - ingest_queue_depth: Number of jobs in queue (pending + processing + dlq)
        - ingest_http_requests_total: Total HTTP requests served
        - ingest_stale_jobs_current: Number of stale jobs in processing queue
        - ingest_processing_age_seconds: Age histogram of jobs in processing
        - ingest_jobs_reaped_total: Total jobs reaped (from reaper stats if available)
    """
    global request_counter
    request_counter += 1

    # Connect to Redis to get queue depth
    redis_url = (
        os.getenv("REDIS_URI") or os.getenv("CACHE_REDIS_URI") or "redis://redis:6379/0"
    )
    r = redis.Redis.from_url(redis_url, decode_responses=True)

    try:
        # Get queue depths
        queued_count = r.llen(KEY_JOBS)
        processing_count = r.llen(KEY_PROCESSING)
        dlq_count = r.llen(KEY_DLQ)
        total_depth = queued_count + processing_count

        # Analyze processing queue for stale jobs and age distribution
        processing_jobs = r.lrange(KEY_PROCESSING, 0, -1)
        stale_count = 0
        job_ages = []
        now = time.time()

        # Default timeout (can be overridden by config)
        job_timeout = int(os.getenv("INGEST_JOB_TIMEOUT", "600"))

        for raw_json in processing_jobs:
            try:
                job = IngestJob.from_json(raw_json)
                status_json = r.hget(KEY_STATUS_HASH, job.job_id)

                if status_json:
                    status = json.loads(status_json)
                    started_at = status.get("started_at")

                    if started_at:
                        age = now - started_at
                        job_ages.append(age)

                        if age >= job_timeout:
                            stale_count += 1
            except Exception:
                continue

        # Calculate age histogram buckets (30s, 60s, 120s, 300s, 600s, +Inf)
        age_buckets = {
            "30": sum(1 for age in job_ages if age <= 30),
            "60": sum(1 for age in job_ages if age <= 60),
            "120": sum(1 for age in job_ages if age <= 120),
            "300": sum(1 for age in job_ages if age <= 300),
            "600": sum(1 for age in job_ages if age <= 600),
            "+Inf": len(job_ages),
        }
        age_sum = sum(job_ages)
        age_count = len(job_ages)

        # Format as Prometheus metrics
        metrics_text = f"""# HELP ingest_queue_depth Number of jobs in ingestion queue
# TYPE ingest_queue_depth gauge
ingest_queue_depth{{state="queued"}} {queued_count}
ingest_queue_depth{{state="processing"}} {processing_count}
ingest_queue_depth{{state="dlq"}} {dlq_count}
ingest_queue_depth{{state="total"}} {total_depth}

# HELP ingest_stale_jobs_current Number of stale jobs in processing queue
# TYPE ingest_stale_jobs_current gauge
ingest_stale_jobs_current {stale_count}

# HELP ingest_processing_age_seconds Age distribution of jobs in processing queue
# TYPE ingest_processing_age_seconds histogram
ingest_processing_age_seconds_bucket{{le="30"}} {age_buckets["30"]}
ingest_processing_age_seconds_bucket{{le="60"}} {age_buckets["60"]}
ingest_processing_age_seconds_bucket{{le="120"}} {age_buckets["120"]}
ingest_processing_age_seconds_bucket{{le="300"}} {age_buckets["300"]}
ingest_processing_age_seconds_bucket{{le="600"}} {age_buckets["600"]}
ingest_processing_age_seconds_bucket{{le="+Inf"}} {age_buckets["+Inf"]}
ingest_processing_age_seconds_sum {age_sum}
ingest_processing_age_seconds_count {age_count}

# HELP ingest_http_requests_total Total HTTP requests to ingestion service
# TYPE ingest_http_requests_total counter
ingest_http_requests_total {request_counter}
"""
        return Response(content=metrics_text, media_type="text/plain")
    except Exception as e:
        # Return error metrics if Redis unavailable
        error_text = f"""# HELP ingest_queue_depth Number of jobs in ingestion queue
# TYPE ingest_queue_depth gauge
ingest_queue_depth{{state="error"}} -1

# HELP ingest_http_requests_total Total HTTP requests to ingestion service
# TYPE ingest_http_requests_total counter
ingest_http_requests_total {request_counter}

# ERROR: {str(e)}
"""
        return Response(content=error_text, media_type="text/plain", status_code=503)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

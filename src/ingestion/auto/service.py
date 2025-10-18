import os

import redis
import uvicorn
from fastapi import FastAPI, Response

from .queue import KEY_JOBS, KEY_PROCESSING, enqueue_file
from .watcher import start_watcher

WATCH_DIR = os.getenv("INGEST_WATCH_DIR", "/app/ingest/incoming")
PORT = int(os.getenv("INGEST_PORT", "8081"))

app = FastAPI(title="Auto-Ingest Service")

observer = None
request_counter = 0  # Simple counter for HTTP requests


@app.on_event("startup")
async def startup():
    os.makedirs(WATCH_DIR, exist_ok=True)
    global observer
    observer = start_watcher(WATCH_DIR)


@app.on_event("shutdown")
async def shutdown():
    if observer:
        observer.stop()
        observer.join()


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
        - ingest_queue_depth: Number of jobs in queue (pending + processing)
        - ingest_http_requests_total: Total HTTP requests served
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
        total_depth = queued_count + processing_count

        # Format as Prometheus metrics
        metrics_text = f"""# HELP ingest_queue_depth Number of jobs in ingestion queue
# TYPE ingest_queue_depth gauge
ingest_queue_depth{{state="queued"}} {queued_count}
ingest_queue_depth{{state="processing"}} {processing_count}
ingest_queue_depth{{state="total"}} {total_depth}

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

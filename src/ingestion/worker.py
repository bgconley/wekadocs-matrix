import asyncio
import os
import signal
import traceback
from urllib.parse import unquote, urlparse

import redis
import structlog

from src.ingestion.auto.queue import IngestJob, JobStatus, ack, brpoplpush, fail
from src.ingestion.auto.reaper import JobReaper

# Optional: wire up your existing ingestion pipeline here
from src.ingestion.build_graph import ingest_document
from src.shared.config import load_config

log = structlog.get_logger()

# Global flag for graceful shutdown
shutdown_requested = False


def parse_file_uri(uri: str) -> str:
    """
    Parse file:// URI and convert host path to container path.

    Args:
        uri: File URI (e.g., file:///host/path/to/file or just /path/to/file)

    Returns:
        Container-local path

    Examples:
        file:///Users/.../wekadocs-matrix/data/ingest/file.md -> /app/data/ingest/file.md
        /app/data/ingest/file.md -> /app/data/ingest/file.md
    """
    # Handle file:// URIs
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        path = unquote(parsed.path)
    else:
        path = uri

    # If already a container path (/app/...), return as-is
    if path.startswith("/app/"):
        return path

    # Convert host path to container path
    # Host paths contain the project root, which maps to /app in container
    # Example: /Users/.../wekadocs-matrix/data/... -> /app/data/...
    if "/data/" in path:
        # Extract everything from /data/ onwards
        data_idx = path.index("/data/")
        container_path = "/app" + path[data_idx:]
        return container_path

    # If no /data/ in path, assume it's already relative to /app
    return path


async def process_job(job: IngestJob):
    """Process an ingestion job using Phase 3 pipeline"""
    if job.kind != "file" or not job.path:
        raise ValueError(f"Unsupported job kind={job.kind} path={job.path}")

    # Parse file URI and get container path
    container_path = parse_file_uri(job.path)

    log.info(
        "Processing job",
        job_id=job.job_id,
        original_path=job.path,
        container_path=container_path,
    )

    # Minimal existence check
    if not os.path.exists(container_path):
        raise FileNotFoundError(f"File not found: {container_path} (from {job.path})")

    # Read file content
    with open(container_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Detect format from extension
    ext = os.path.splitext(container_path)[1].lower()
    if ext in (".md", ".markdown"):
        format = "markdown"
    elif ext in (".html", ".htm"):
        format = "html"
    else:
        format = "markdown"  # default

    # Call Phase 3 ingestion pipeline (use original job.path as source_uri for tracking)
    source_uri = job.path if job.path.startswith("file://") else f"file://{job.path}"
    stats = ingest_document(source_uri, content, format=format)

    log.info("Ingestion completed", job_id=job.job_id, stats=stats)
    return stats


def handle_shutdown(signum, frame):
    """Signal handler for graceful shutdown."""
    global shutdown_requested
    shutdown_requested = True
    log.info("Shutdown signal received, finishing current job...", signal=signum)


async def main():
    global shutdown_requested

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Load config for reaper settings
    try:
        config = load_config()
        reaper_config = config.ingest.queue_recovery
        reaper_enabled = reaper_config.enabled
        job_timeout = reaper_config.job_timeout_seconds
        reaper_interval = reaper_config.reaper_interval_seconds
        max_retries = reaper_config.max_retries
        stale_action = reaper_config.stale_job_action
    except Exception as e:
        log.warning(
            "Failed to load reaper config, using defaults",
            error=str(e),
        )
        reaper_enabled = True
        job_timeout = 600
        reaper_interval = 30
        max_retries = 3
        stale_action = "requeue"

    # Initialize Redis client for reaper
    redis_url = (
        os.getenv("REDIS_URI") or os.getenv("CACHE_REDIS_URI") or "redis://redis:6379/0"
    )
    redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

    # Initialize and start reaper
    reaper = JobReaper(
        redis_client=redis_client,
        timeout_seconds=job_timeout,
        interval_seconds=reaper_interval,
        max_retries=max_retries,
        action=stale_action,
        enabled=reaper_enabled,
    )

    # Start reaper as background task
    reaper_task = asyncio.create_task(reaper.reap_loop())

    log.info(
        "Ingestion worker starting",
        reaper_enabled=reaper_enabled,
        job_timeout=job_timeout,
        reaper_interval=reaper_interval,
    )

    # Main processing loop
    while not shutdown_requested:
        try:
            item = brpoplpush(timeout=1)
            if not item:
                await asyncio.sleep(0.05)
                continue
            raw, job_id = item
            job = IngestJob.from_json(raw)
            try:
                await process_job(job)
                ack(raw, job_id)
                log.info(
                    "Job done", job_id=job_id, path=job.path, status=JobStatus.DONE
                )
            except Exception as e:
                log.error("Job failed", job_id=job_id, error=str(e))
                fail(raw, job_id, reason=str(e), requeue=True)
        except Exception as loop_err:
            log.error(
                "Worker loop error", error=str(loop_err), exc=traceback.format_exc()
            )
            await asyncio.sleep(0.25)

    # Graceful shutdown
    log.info("Shutdown complete, cancelling reaper")
    reaper_task.cancel()
    try:
        await reaper_task
    except asyncio.CancelledError:
        pass

    log.info("Worker exited gracefully", reaper_stats=reaper.get_stats())


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os
import signal
import time
import traceback
from typing import Any, Coroutine
from urllib.parse import unquote, urlparse

import redis

# Atomic ingestion pipeline with saga-coordinated Neo4j + Qdrant writes
# Guarantees: Neo4j commits only if Qdrant succeeds; compensates on failure
from src.ingestion.atomic import AtomicIngestionCoordinator
from src.ingestion.auto.queue import IngestJob, JobStatus, ack, brpoplpush, fail
from src.ingestion.auto.reaper import JobReaper
from src.shared.config import load_config

# LGTM Phase 4: Import observability components for trace-correlated logging
from src.shared.observability import get_logger, setup_logging
from src.shared.observability.tracing import init_tracing

log = get_logger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


# =============================================================================
# Asyncio Exception Handling - Prevents silent failures in background tasks
# Reference: https://superfastpython.com/asyncio-event-loop-exception-handler/
# =============================================================================


def global_exception_handler(
    loop: asyncio.AbstractEventLoop, context: dict[str, Any]
) -> None:
    """
    Global handler for asyncio never-retrieved exceptions.

    Called by the event loop for unhandled exceptions in callbacks and
    never-retrieved task exceptions during shutdown.

    Context dict may contain:
    - 'message': Error message
    - 'exception': Exception object (optional)
    - 'future': asyncio.Future instance (optional)
    - 'task': asyncio.Task instance (optional)
    - 'handle': asyncio.Handle instance (optional)
    """
    # Extract exception details
    exception = context.get("exception")
    message = context.get("message", "Unhandled exception in asyncio")
    task = context.get("task")
    future = context.get("future")

    # Build structured log context
    log_context: dict[str, Any] = {
        "error_message": message,
        "exception_type": type(exception).__name__ if exception else "Unknown",
        "exception_str": str(exception) if exception else None,
    }

    # Add task/future info if available
    if task:
        log_context["task_name"] = task.get_name()
        log_context["task_coro"] = str(task.get_coro())
    elif future:
        log_context["future"] = str(future)

    # Log with full traceback
    if exception:
        log_context["traceback"] = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )

    log.error("asyncio_unhandled_exception", **log_context)


def log_task_exception_callback(task: asyncio.Task) -> None:
    """
    Done callback that logs exceptions immediately when a task completes.

    This provides real-time exception logging, unlike the global handler
    which only fires at shutdown.

    Reference: https://superfastpython.com/asyncio-log-exceptions-done-callback/
    """
    try:
        # Safely retrieve exception (may raise CancelledError)
        exception = task.exception()
        if exception is not None:
            log.error(
                "asyncio_task_failed",
                task_name=task.get_name(),
                exception_type=type(exception).__name__,
                exception_str=str(exception),
                traceback="".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                ),
            )
    except asyncio.CancelledError:
        # Task was cancelled, not an error
        log.debug("asyncio_task_cancelled", task_name=task.get_name())
    except asyncio.InvalidStateError:
        # Task is not done yet (should not happen in done callback)
        pass


def create_monitored_task(coro: Coroutine, name: str | None = None) -> asyncio.Task:
    """
    Create an asyncio task with automatic exception logging.

    Use this instead of asyncio.create_task() for background tasks
    that might fail silently.

    Args:
        coro: Coroutine to wrap in a task
        name: Optional task name for logging

    Returns:
        Task with done callback attached
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(log_task_exception_callback)
    return task


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

    # Robust host->container mapping: anchor to project root if present
    # Default project root on host is the repo root; in container it's /app
    repo_marker = "/wekadocs-matrix/"
    if repo_marker in path:
        idx = path.index(repo_marker)
        return "/app" + path[idx + len(repo_marker) - 1 :]

    # Fallback: if /data/ segment exists, strip to last occurrence
    if "/data/" in path:
        data_idx = path.rfind("/data/")
        return "/app" + path[data_idx:]

    # If no markers, assume already container-relative
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

    # Call atomic ingestion pipeline (saga-coordinated Neo4j + Qdrant writes)
    # Uses original job.path as source_uri for provenance tracking.
    #
    # Return the full AtomicIngestionResult so run_stats can correctly
    # attribute success, document_id, saga_id, and aggregate counts.
    source_uri = job.path if job.path.startswith("file://") else f"file://{job.path}"
    from src.shared.config import get_config
    from src.shared.connections import get_connection_manager

    config = get_config()
    manager = get_connection_manager()
    neo4j_driver = manager.get_neo4j_driver()

    qdrant_client = None
    if config.search.vector.primary == "qdrant" or config.search.vector.dual_write:
        qdrant_client = manager.get_qdrant_client()

    coordinator = AtomicIngestionCoordinator(
        neo4j_driver,
        qdrant_client,
        config,
    )
    result = coordinator.ingest_document_atomic(source_uri, content, format=format)
    if not result.success:
        raise RuntimeError(result.error or "Atomic ingestion failed")

    log.info(
        "Ingestion completed",
        job_id=job.job_id,
        document_id=result.document_id,
        saga_id=result.saga_id,
        stats=result.stats,
    )
    return result


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_structural_health_check_sync() -> dict[str, Any]:
    """
    Check graph structural integrity - emit ERROR if documents need repair.

    With atomic structural edge building in the saga, documents needing repair
    indicates a bug or migration gap, not normal operation. This function
    alerts rather than silently repairs.
    """
    from src.neo.contract_checks import GraphContractChecker
    from src.shared.connections import get_connection_manager

    cm = get_connection_manager()
    driver = cm.get_neo4j_driver()

    with driver.session() as session:
        checker = GraphContractChecker(session)
        doc_ids = checker.find_documents_needing_repair()

        if doc_ids:
            # This is an ERROR condition - atomic ingestion should prevent this
            log.error(
                "structural_integrity_violation",
                documents_needing_repair=len(doc_ids),
                sample_doc_ids=doc_ids[:10],
                message=(
                    "Documents found with missing structural edges after atomic ingestion. "
                    "This indicates a bug in ingestion or a migration gap."
                ),
            )
            return {
                "status": "unhealthy",
                "documents_needing_repair": len(doc_ids),
                "sample_doc_ids": doc_ids[:10],
            }

        return {
            "status": "healthy",
            "documents_needing_repair": 0,
        }


def handle_shutdown(signum, frame):
    """Signal handler for graceful shutdown."""
    global shutdown_requested
    shutdown_requested = True
    log.info("Shutdown signal received, finishing current job...", signal=signum)


async def main():
    global shutdown_requested

    # LGTM Phase 4: Configure structlog with trace context processor
    # Must be called before init_tracing() to enable log-trace correlation
    setup_logging("INFO")

    # LGTM Phase 4: Initialize OTEL tracing for ingestion worker
    # This enables trace context propagation to Alloy → Tempo
    init_tracing(
        service_name="weka-ingestion-worker",
        service_version="1.0.0",
        instrument_redis=True,
    )
    log.info("otel_tracing_initialized", service="weka-ingestion-worker")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Register global asyncio exception handler for never-retrieved exceptions
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(global_exception_handler)
    log.info("asyncio_exception_handler_registered")

    # Load config for reaper settings
    try:
        config, settings = load_config()  # load_config returns (config, settings)
        # correct field name is 'ingestion', not 'ingest'
        reaper_root = getattr(config, "ingestion", None)
        reaper_config = (
            getattr(reaper_root, "queue_recovery", None) if reaper_root else None
        )
        if reaper_config:
            reaper_enabled = reaper_config.enabled
            job_timeout = reaper_config.job_timeout_seconds
            reaper_interval = reaper_config.reaper_interval_seconds
            max_retries = reaper_config.max_retries
            stale_action = reaper_config.stale_job_action
        else:
            raise AttributeError("queue_recovery config not found")
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

    # Start reaper as background task with exception monitoring
    reaper_task = create_monitored_task(reaper.reap_loop(), name="job_reaper")

    log.info(
        "Ingestion worker starting",
        reaper_enabled=reaper_enabled,
        job_timeout=job_timeout,
        reaper_interval=reaper_interval,
    )

    # Run stats and summary configuration
    from src.ingestion.run_stats import IngestionRunStats

    summary_idle_seconds = int(os.getenv("INGESTION_SUMMARY_IDLE_SECONDS", "180"))
    health_check_enabled = _env_bool("NEO4J_STRUCTURAL_HEALTH_CHECK_ENABLED", True)
    run_stats: IngestionRunStats | None = None
    last_job_completed_at: float | None = None

    # Main processing loop
    while not shutdown_requested:
        try:
            item = brpoplpush(timeout=1)
            if not item:
                # Queue is empty - check if we should emit summary
                if (
                    run_stats is not None
                    and run_stats.has_data
                    and last_job_completed_at is not None
                    and (time.monotonic() - last_job_completed_at)
                    >= summary_idle_seconds
                ):
                    # Emit consolidated run summary
                    run_stats.emit_summary()

                    # Run structural health check (alerts if issues found)
                    if health_check_enabled:
                        try:
                            health = await asyncio.to_thread(
                                _run_structural_health_check_sync
                            )
                            log.info(
                                "structural_health_check_complete",
                                status=health.get("status"),
                                documents_needing_repair=health.get(
                                    "documents_needing_repair", 0
                                ),
                            )
                        except Exception as e:
                            log.warning(
                                "structural_health_check_failed",
                                error=str(e),
                                traceback=traceback.format_exc(),
                            )

                    # Reset for next run
                    run_stats = None
                    last_job_completed_at = None

                await asyncio.sleep(0.05)
                continue

            # We have a job to process
            raw, job_id = item
            job = IngestJob.from_json(raw)

            # Start new run if needed
            if run_stats is None:
                run_stats = IngestionRunStats.start_new()
                log.info("ingestion_run_started", run_id=run_stats.run_id)

            try:
                result = await process_job(job)
                run_stats.record_job(job.path, result)
                ack(raw, job_id)
                log.info(
                    "Job done", job_id=job_id, path=job.path, status=JobStatus.DONE
                )
                last_job_completed_at = time.monotonic()
            except Exception as e:
                log.error("Job failed", job_id=job_id, error=str(e))
                run_stats.record_exception(job.path, str(e))
                fail(raw, job_id, reason=str(e), requeue=True)
                last_job_completed_at = time.monotonic()
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

import asyncio
import os
import traceback
from urllib.parse import unquote, urlparse

import structlog

from src.ingestion.auto.queue import IngestJob, JobStatus, ack, brpoplpush, fail

# Optional: wire up your existing ingestion pipeline here
from src.ingestion.build_graph import ingest_document

log = structlog.get_logger()


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


async def main():
    log.info("Ingestion worker starting")
    while True:
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


if __name__ == "__main__":
    asyncio.run(main())

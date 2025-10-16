import asyncio
import os
import traceback

import structlog

from src.ingestion.auto.queue import IngestJob, JobStatus, ack, brpoplpush, fail

# Optional: wire up your existing ingestion pipeline here
from src.ingestion.build_graph import ingest_document

log = structlog.get_logger()


async def process_job(job: IngestJob):
    """Process an ingestion job using Phase 3 pipeline"""
    if job.kind != "file" or not job.path:
        raise ValueError(f"Unsupported job kind={job.kind} path={job.path}")

    # Minimal existence check
    if not os.path.exists(job.path):
        raise FileNotFoundError(job.path)

    # Read file content
    with open(job.path, "r", encoding="utf-8") as f:
        content = f.read()

    # Detect format from extension
    ext = os.path.splitext(job.path)[1].lower()
    if ext in (".md", ".markdown"):
        format = "markdown"
    elif ext in (".html", ".htm"):
        format = "html"
    else:
        format = "markdown"  # default

    # Call Phase 3 ingestion pipeline
    source_uri = f"file://{job.path}"
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

"""
Phase 6, Task 6.3: CLI & Progress UI

Command-line interface for auto-ingestion system with live progress tracking.

See: /docs/implementation-plan-phase-6.md → Task 6.3
See: /docs/pseudocode-phase6.md → Task 6.3
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import List

import redis

# Configure logging early - send structured logging to stderr by default
# This prevents log messages from interfering with JSON output
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)

from src.ingestion.auto.progress import JobStage, ProgressReader  # noqa: E402
from src.ingestion.auto.queue import JobQueue  # noqa: E402
from src.shared.config import get_config, reload_config  # noqa: E402
from src.shared.observability import get_logger  # noqa: E402

logger = get_logger(__name__)

# Ensure all structured logging goes to stderr (after imports)
for handler in logging.root.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        handler.setStream(sys.stderr)


class ProgressUI:
    """Simple progress bar renderer for terminal output."""

    def __init__(self, json_mode: bool = False):
        """
        Initialize progress UI.

        Args:
            json_mode: If True, output JSON lines instead of progress bars
        """
        self.json_mode = json_mode
        self.last_update = {}

    def render(
        self, job_id: str, stage: str, percent: float, message: str, elapsed: float
    ):
        """
        Render progress for a job.

        Args:
            job_id: Job identifier
            stage: Current stage
            percent: Progress percentage (0-100)
            message: Status message
            elapsed: Elapsed seconds
        """
        if self.json_mode:
            # JSON output for machine consumption
            output = {
                "job_id": job_id,
                "stage": stage,
                "percent": round(percent, 1),
                "message": message,
                "elapsed_seconds": round(elapsed, 1),
            }
            print(json.dumps(output), flush=True)
        else:
            # Human-readable progress bar
            bar_width = 30
            filled = int((percent / 100) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Format elapsed time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}m{secs}s" if mins > 0 else f"{secs}s"

            # Print progress (overwrite previous line)
            sys.stdout.write(
                f"\r{stage:12} [{bar}] {percent:5.1f}% | {time_str:6} | {message}"
            )
            sys.stdout.flush()

            self.last_update[job_id] = time.time()

    def finish(self, job_id: str, success: bool, message: str):
        """
        Finish progress display.

        Args:
            job_id: Job identifier
            success: Whether job succeeded
            message: Final message
        """
        if self.json_mode:
            output = {
                "job_id": job_id,
                "status": "completed" if success else "failed",
                "message": message,
            }
            print(json.dumps(output), flush=True)
        else:
            status = "✓" if success else "✗"
            print(f"\n{status} {message}\n")


def compute_file_checksum(file_path: str) -> str:
    """
    Compute SHA-256 checksum of file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal checksum string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def resolve_targets(targets: List[str]) -> List[Path]:
    """
    Resolve file paths from targets (supports globs).

    Args:
        targets: List of file paths, directories, or glob patterns

    Returns:
        List of resolved file paths
    """
    resolved = []

    for target in targets:
        # Expand globs
        matches = glob.glob(target, recursive=True)

        if matches:
            for match in matches:
                path = Path(match)
                if path.is_file():
                    resolved.append(path)
                elif path.is_dir():
                    # Recursively find markdown/html files
                    resolved.extend(path.rglob("*.md"))
                    resolved.extend(path.rglob("*.html"))
        else:
            # Check if it's a single file or URL
            if target.startswith(("http://", "https://", "s3://")):
                # URL - will be handled differently
                resolved.append(Path(target))
            else:
                path = Path(target)
                if path.exists():
                    if path.is_file():
                        resolved.append(path)
                    elif path.is_dir():
                        resolved.extend(path.rglob("*.md"))
                        resolved.extend(path.rglob("*.html"))

    return list(set(resolved))  # Deduplicate


def cmd_ingest(args):
    """
    Implement 'ingestctl ingest' command.

    Enqueues files for ingestion and monitors progress.
    """
    # Redirect ALL logging to stderr in JSON mode
    if args.json:
        import logging
        import sys

        # Configure root logger to use stderr
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.WARNING)

    # Connect to Redis
    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    try:
        if redis_password:
            redis_client = redis.Redis.from_url(
                redis_uri, password=redis_password, decode_responses=True
            )
        else:
            redis_client = redis.Redis.from_url(redis_uri, decode_responses=True)

        # Test connection
        redis_client.ping()
    except Exception as e:
        print(f"Error: Failed to connect to Redis: {e}", file=sys.stderr)
        return 1

    config = get_config()
    queue = JobQueue(redis_client)

    # Resolve targets
    targets = resolve_targets(args.targets)

    if not targets:
        print(f"Error: No files found for targets: {args.targets}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"targets_found": len(targets)}))

    # Dry run mode
    if args.dry_run:
        print(f"Dry run: Would ingest {len(targets)} files:")
        for target in targets:
            print(f"  - {target}")
        return 0

    # Enqueue jobs
    job_ids = []
    for target in targets:
        try:
            # Compute checksum
            if str(target).startswith(("http://", "https://", "s3://")):
                # URL - use URL as checksum for now
                checksum = hashlib.sha256(str(target).encode()).hexdigest()
                source_uri = str(target)
            else:
                checksum = compute_file_checksum(str(target))
                source_uri = f"file://{target.absolute()}"

            # Enqueue
            default_tag = getattr(config, "ingest", None)
            if default_tag and hasattr(default_tag, "tag"):
                default_tag = default_tag.tag
            else:
                default_tag = "wekadocs"

            job_id = queue.enqueue(
                source_uri=source_uri,
                checksum=checksum,
                tag=args.tag or default_tag,
            )

            if job_id:
                job_ids.append(job_id)
                if not args.json:
                    print(f"Enqueued: {target} → {job_id}")
            else:
                if not args.json:
                    print(f"Skipped (duplicate): {target}")

        except Exception as e:
            logger.error(f"Failed to enqueue {target}: {e}")
            if not args.json:
                print(f"Error enqueuing {target}: {e}", file=sys.stderr)

    if not job_ids:
        print("No new jobs enqueued (all duplicates or errors)", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"jobs_enqueued": len(job_ids), "job_ids": job_ids}))

    # Watch mode
    if args.watch:
        print("Watch mode not yet implemented", file=sys.stderr)
        return 1

    # Return early if --no-wait
    if args.no_wait:
        if not args.json:
            print(
                f"\nEnqueued {len(job_ids)} job(s). Use 'ingestctl status' to monitor progress.\n"
            )
        return 0

    # Monitor progress
    if not args.json:
        print(f"\nMonitoring progress for {len(job_ids)} job(s)...\n")

    ui = ProgressUI(json_mode=args.json)
    start_time = time.time()

    # Monitor all jobs
    completed = set()
    errors = []

    while len(completed) < len(job_ids):
        for job_id in job_ids:
            if job_id in completed:
                continue

            reader = ProgressReader(redis_client, job_id)
            latest = reader.get_latest()

            if latest:
                elapsed = time.time() - start_time
                ui.render(
                    job_id=job_id,
                    stage=latest.stage.value,
                    percent=latest.percent,
                    message=latest.message,
                    elapsed=elapsed,
                )

                # Check if complete
                if latest.stage == JobStage.DONE:
                    completed.add(job_id)
                    ui.finish(
                        job_id,
                        success=True,
                        message=f"Job {job_id} completed successfully",
                    )
                elif latest.stage == JobStage.ERROR:
                    completed.add(job_id)
                    errors.append(job_id)
                    error_msg = latest.error or "Unknown error"
                    ui.finish(
                        job_id,
                        success=False,
                        message=f"Job {job_id} failed: {error_msg}",
                    )

        time.sleep(1)

    # Final summary
    if not args.json:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(
            f"Completed {len(completed)} of {len(job_ids)} jobs in {int(total_time)}s"
        )
        if errors:
            print(f"Errors: {len(errors)}")

    return 1 if errors else 0


def cmd_status(args):
    """
    Implement 'ingestctl status' command.

    Shows job status (all jobs or specific job).
    """
    # Redirect ALL logging to stderr in JSON mode
    if args.json:
        import logging
        import sys

        # Configure root logger to use stderr
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.WARNING)

    # Connect to Redis
    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    try:
        if redis_password:
            redis_client = redis.Redis.from_url(
                redis_uri, password=redis_password, decode_responses=True
            )
        else:
            redis_client = redis.Redis.from_url(redis_uri, decode_responses=True)

        redis_client.ping()
    except Exception as e:
        print(f"Error: Failed to connect to Redis: {e}", file=sys.stderr)
        return 1

    _ = get_config()  # noqa: F841
    queue = JobQueue(redis_client)

    if args.job_id:
        # Show specific job
        state = queue.get_state(args.job_id)

        if not state:
            print(f"Error: Job {args.job_id} not found", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(state, indent=2))
        else:
            print(f"Job ID: {args.job_id}")
            print(f"Status: {state.get('status', 'UNKNOWN')}")
            print(f"Source: {state.get('source_uri', 'N/A')}")
            print(f"Tag: {state.get('tag', 'N/A')}")
            print(f"Created: {state.get('created_at', 'N/A')}")
            print(f"Updated: {state.get('updated_at', 'N/A')}")

        return 0
    else:
        # List all jobs
        # Scan for all state keys
        job_states = []
        for key in redis_client.scan_iter(f"{queue.STATE_PREFIX}*", count=100):
            job_id = key.split(":")[-1]
            state = queue.get_state(job_id)
            if state:
                job_states.append(state)

        if args.json:
            print(json.dumps({"jobs": job_states, "count": len(job_states)}, indent=2))
        else:
            print(f"Found {len(job_states)} jobs:")
            print(f"\n{'Job ID':<36} | {'Status':<12} | {'Tag':<12} | {'Created':<20}")
            print("-" * 85)

            for state in sorted(
                job_states, key=lambda x: x.get("created_at", ""), reverse=True
            )[:20]:
                job_id = state.get("job_id", "N/A")[:35]
                status = state.get("status", "UNKNOWN")[:11]
                tag = state.get("tag", "N/A")[:11]
                created = state.get("created_at", "N/A")[:19]

                print(f"{job_id} | {status} | {tag} | {created}")

        return 0


def cmd_tail(args):
    """
    Implement 'ingestctl tail' command.

    Streams real-time progress for a job.
    """
    # Redirect ALL logging to stderr in JSON mode
    if args.json:
        import logging
        import sys

        # Configure root logger to use stderr
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.WARNING)

    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    try:
        if redis_password:
            redis_client = redis.Redis.from_url(
                redis_uri, password=redis_password, decode_responses=True
            )
        else:
            redis_client = redis.Redis.from_url(redis_uri, decode_responses=True)

        redis_client.ping()
    except Exception as e:
        print(f"Error: Failed to connect to Redis: {e}", file=sys.stderr)
        return 1

    reader = ProgressReader(redis_client, args.job_id)
    ui = ProgressUI(json_mode=args.json)

    if not args.json:
        print(f"Tailing job {args.job_id}... (Ctrl+C to exit)\n")

    start_time = time.time()

    # Set up signal handler for clean exit
    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Stream events
    while True:
        events = reader.read_events(count=10, block_ms=1000)

        for event in events:
            elapsed = time.time() - start_time
            ui.render(
                job_id=args.job_id,
                stage=event.stage.value,
                percent=event.percent,
                message=event.message,
                elapsed=elapsed,
            )

            # Exit if complete
            if event.stage == JobStage.DONE:
                ui.finish(
                    args.job_id, success=True, message="Job completed successfully"
                )
                return 0
            elif event.stage == JobStage.ERROR:
                error_msg = event.error or "Unknown error"
                ui.finish(
                    args.job_id, success=False, message=f"Job failed: {error_msg}"
                )
                return 1


def cmd_cancel(args):
    """
    Implement 'ingestctl cancel' command.

    Cancels a running job.
    """
    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    try:
        if redis_password:
            redis_client = redis.Redis.from_url(
                redis_uri, password=redis_password, decode_responses=True
            )
        else:
            redis_client = redis.Redis.from_url(redis_uri, decode_responses=True)

        redis_client.ping()
    except Exception as e:
        print(f"Error: Failed to connect to Redis: {e}", file=sys.stderr)
        return 1

    _ = get_config()  # noqa: F841
    queue = JobQueue(redis_client)

    # Check if job exists
    state = queue.get_state(args.job_id)

    if not state:
        print(f"Error: Job {args.job_id} not found", file=sys.stderr)
        return 1

    # Set cancellation flag
    queue.update_state(args.job_id, status="CANCELLED")

    print(f"Job {args.job_id} marked for cancellation")
    print("Note: Worker will stop at next checkpoint")

    return 0


def cmd_clean(args):
    """
    Implement 'ingestctl clean' command.

    Cleans stale jobs from processing queue.
    """
    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    try:
        if redis_password:
            redis_client = redis.Redis.from_url(
                redis_uri, password=redis_password, decode_responses=True
            )
        else:
            redis_client = redis.Redis.from_url(redis_uri, decode_responses=True)

        redis_client.ping()
    except Exception as e:
        print(f"Error: Failed to connect to Redis: {e}", file=sys.stderr)
        return 1

    from src.ingestion.auto.queue import (
        KEY_DLQ,
        KEY_JOBS,
        KEY_PROCESSING,
        KEY_STATUS_HASH,
        IngestJob,
    )

    # Get all jobs in processing queue
    processing_jobs = redis_client.lrange(KEY_PROCESSING, 0, -1)

    if not processing_jobs:
        print("No jobs in processing queue")
        return 0

    # Analyze each job
    stale_jobs = []
    now = time.time()

    for raw_json in processing_jobs:
        try:
            job = IngestJob.from_json(raw_json)
        except Exception:
            continue

        # Get job status
        status_json = redis_client.hget(KEY_STATUS_HASH, job.job_id)
        if not status_json:
            # No status - definitely stale
            age = None
            stale_jobs.append((raw_json, job, age, "No status timestamp"))
            continue

        status = json.loads(status_json)
        started_at = status.get("started_at")

        if not started_at:
            # No timestamp - stale
            stale_jobs.append((raw_json, job, None, "No started_at timestamp"))
            continue

        age = now - started_at

        # Check if older than threshold
        if args.older_than and age >= args.older_than:
            stale_jobs.append((raw_json, job, age, f"Age: {int(age)}s"))
        elif not args.older_than:
            # No filter - show all
            stale_jobs.append((raw_json, job, age, f"Age: {int(age)}s"))

    if not stale_jobs:
        print(f"No stale jobs found (scanned {len(processing_jobs)} jobs)")
        return 0

    # Display stale jobs
    if args.json:
        output = {
            "total_processing": len(processing_jobs),
            "stale_count": len(stale_jobs),
            "stale_jobs": [
                {
                    "job_id": job.job_id,
                    "path": job.path,
                    "attempts": job.attempts,
                    "age_seconds": int(age) if age else None,
                    "reason": reason,
                }
                for _, job, age, reason in stale_jobs
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nFound {len(stale_jobs)} stale job(s):\n")
        print(f"{'Job ID':<36} | {'Age':<10} | {'Attempts':<8} | {'Path':<40}")
        print("-" * 100)

        for _, job, age, reason in stale_jobs:
            age_str = f"{int(age)}s" if age else "Unknown"
            path_str = (job.path or "N/A")[:39]
            print(f"{job.job_id[:35]} | {age_str:<10} | {job.attempts:<8} | {path_str}")

    # Dry run mode
    if args.dry_run:
        print(f"\nDry run: Would clean {len(stale_jobs)} stale job(s)")
        return 0

    # Confirm before cleaning
    if not args.yes:
        response = input(f"\nClean {len(stale_jobs)} stale job(s)? (y/N): ")
        if response.lower() != "y":
            print("Cancelled")
            return 0

    # Clean stale jobs
    config = get_config()
    max_retries = 3
    if hasattr(config, "ingest") and hasattr(config.ingest, "queue_recovery"):
        max_retries = config.ingest.queue_recovery.max_retries

    requeued = 0
    failed = 0

    for raw_json, job, age, reason in stale_jobs:
        job.attempts += 1

        # Remove from processing
        redis_client.lrem(KEY_PROCESSING, 1, raw_json)

        if job.attempts < max_retries:
            # Requeue
            redis_client.lpush(KEY_JOBS, job.to_json())
            redis_client.hset(
                KEY_STATUS_HASH,
                job.job_id,
                json.dumps(
                    {
                        "status": "queued",
                        "attempts": job.attempts,
                        "last_error": f"Cleaned: {reason}",
                        "requeued_at": time.time(),
                        "cleaned": True,
                    }
                ),
            )
            requeued += 1
        else:
            # Move to DLQ
            redis_client.lpush(KEY_DLQ, raw_json)
            redis_client.hset(
                KEY_STATUS_HASH,
                job.job_id,
                json.dumps(
                    {
                        "status": "failed",
                        "attempts": job.attempts,
                        "error": f"Max retries exceeded. {reason}",
                        "failed_at": time.time(),
                        "cleaned": True,
                    }
                ),
            )
            failed += 1

    print(f"\nCleaned {len(stale_jobs)} stale job(s):")
    print(f"  Requeued: {requeued}")
    print(f"  Failed (to DLQ): {failed}")

    return 0


def cmd_report(args):
    """
    Implement 'ingestctl report' command.

    Displays job report (from Phase 6.4).
    """
    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    # Try to get report path from job state first
    report_file = None
    try:
        if redis_password:
            redis_client = redis.Redis.from_url(
                redis_uri, password=redis_password, decode_responses=True
            )
        else:
            redis_client = redis.Redis.from_url(redis_uri, decode_responses=True)

        redis_client.ping()

        # Load job state to get report path
        queue = JobQueue(redis_client)
        state_dict = queue.get_state(args.job_id)

        if state_dict:
            # Try to get report path from stats
            report_path_str = (
                state_dict.get("stats", {}).get("reporting", {}).get("report_json_path")
            )
            if report_path_str:
                report_file = Path(report_path_str)
                if not report_file.exists():
                    report_file = None
    except Exception as e:
        # Redis connection failed or state not found; fall back to directory scan
        logger.debug(f"Failed to load report path from job state: {e}")

    # Fall back to directory scanning if state lookup failed
    if not report_file:
        reports_dir = Path("reports/ingest")

        if not reports_dir.exists():
            print("Error: No reports directory found", file=sys.stderr)
            return 1

        # Search for report by job_id
        for report_dir in reports_dir.iterdir():
            if report_dir.is_dir():
                candidate = report_dir / "ingest_report.json"
                if candidate.exists():
                    # Check if this report matches job_id
                    try:
                        with open(candidate) as f:
                            data = json.load(f)
                            if data.get("job_id") == args.job_id:
                                report_file = candidate
                                break
                    except Exception:
                        continue

    if not report_file:
        print(f"Error: Report for job {args.job_id} not found", file=sys.stderr)
        print("Ensure the job has completed and verification has run.", file=sys.stderr)
        return 1

    # Load and display report
    try:
        with open(report_file) as f:
            report = json.load(f)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            # Human-readable report
            print(f"\n{'='*60}")
            print(f"Ingestion Report: {args.job_id}")
            print(f"{'='*60}\n")

            print(f"Tag: {report.get('tag', 'N/A')}")
            print(f"Timestamp: {report.get('timestamp_utc', 'N/A')}")
            print(f"Source: {report.get('doc', {}).get('source_uri', 'N/A')}")
            print(f"Sections: {report.get('doc', {}).get('sections', 0)}")

            print("\nGraph:")
            graph = report.get("graph", {})
            print(f"  Nodes added: {graph.get('nodes_added', 0)}")
            print(f"  Relationships added: {graph.get('rels_added', 0)}")

            print("\nVectors:")
            vectors = report.get("vector", {})
            print(f"  Sections indexed: {vectors.get('sections_indexed', 0)}")
            print(f"  Embedding version: {vectors.get('embedding_version', 'N/A')}")

            print(f"\nDrift: {report.get('drift_pct', 0):.2f}%")

            print("\nSample Queries:")
            for query in report.get("sample_queries", []):
                q = query.get("q", "N/A")
                conf = query.get("confidence", 0)
                print(f"  - {q} (confidence: {conf:.2f})")

            print(f"\nReady for queries: {report.get('ready_for_queries', False)}")

            print("\nTimings:")
            timings = report.get("timings_ms", {})
            for stage, ms in timings.items():
                print(f"  {stage}: {ms}ms")

            if report.get("errors"):
                print("\nErrors:")
                for err in report["errors"]:
                    print(f"  - {err}")

            print(f"\n{'='*60}\n")

        return 0

    except Exception as e:
        print(f"Error reading report: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    # Refresh config on startup to honor current env for long-lived processes
    reload_config()
    parser = argparse.ArgumentParser(
        prog="ingestctl",
        description="CLI for auto-ingestion system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest files/URLs")
    ingest_parser.add_argument(
        "targets", nargs="+", help="File paths, directories, or URLs"
    )
    ingest_parser.add_argument("--tag", help="Classification tag for sample queries")
    ingest_parser.add_argument(
        "--watch", action="store_true", help="Watch mode (continuous monitoring)"
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without doing it",
    )
    ingest_parser.add_argument(
        "--json", action="store_true", help="JSON output for machine consumption"
    )
    ingest_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job completion (enqueue only)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument(
        "job_id", nargs="?", help="Optional job ID (shows all if omitted)"
    )
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # Tail command
    tail_parser = subparsers.add_parser("tail", help="Stream real-time progress")
    tail_parser.add_argument("job_id", help="Job ID to monitor")
    tail_parser.add_argument("--json", action="store_true", help="JSON output")

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel running job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")

    # Report command
    report_parser = subparsers.add_parser("report", help="Display job report")
    report_parser.add_argument("job_id", help="Job ID")
    report_parser.add_argument("--json", action="store_true", help="JSON output")

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean stale jobs from processing queue"
    )
    clean_parser.add_argument(
        "--older-than",
        type=int,
        metavar="SECONDS",
        help="Only clean jobs older than N seconds",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without doing it",
    )
    clean_parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    clean_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "tail":
        return cmd_tail(args)
    elif args.command == "cancel":
        return cmd_cancel(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "clean":
        return cmd_clean(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

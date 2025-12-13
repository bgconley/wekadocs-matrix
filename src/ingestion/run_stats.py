"""
Ingestion run statistics accumulator.

This module provides a structured way to accumulate statistics across multiple
ingestion jobs within a single "run" (batch of jobs until idle). It emits
a consolidated summary log when the run completes.

Key design principles:
- Accumulates stats across jobs for consolidated reporting
- Deduplicates and counts warnings
- Tracks failed jobs with file paths and errors
- Emits structured summary log after idle period
- ~200 lines, single responsibility

Usage in worker.py:
    from src.ingestion.run_stats import IngestionRunStats

    run_stats = IngestionRunStats.start_new()

    # After each job:
    run_stats.record_job(job.path, result)

    # After idle period:
    run_stats.emit_summary()
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FailedJob:
    """Details of a failed ingestion job."""

    path: str
    document_id: Optional[str]
    error: str
    saga_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class IngestionRunStats:
    """
    Accumulates statistics across a batch of ingestion jobs.

    A "run" starts when the first job is processed after idle and ends
    when the worker has been idle for a configured period.
    """

    run_id: str
    start_time: float
    end_time: Optional[float] = None

    # Job counts
    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0

    # Document/content counts
    chunks_created: int = 0
    entities_created: int = 0
    vectors_written: int = 0

    # Edge counts by type
    edges: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Failure details
    failed_jobs: List[FailedJob] = field(default_factory=list)

    # Warnings accumulated during run (deduplicated)
    _warnings: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @classmethod
    def start_new(cls) -> "IngestionRunStats":
        """Create a new run stats tracker with generated run_id."""
        return cls(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=time.monotonic(),
        )

    def record_job(
        self,
        job_path: str,
        result: Any,  # AtomicIngestionResult - avoiding circular import
    ) -> None:
        """
        Record the outcome of a single ingestion job.

        Args:
            job_path: File path of the ingested document
            result: AtomicIngestionResult from the ingestion
        """
        self.jobs_processed += 1

        if getattr(result, "success", False):
            self.jobs_succeeded += 1
            self._aggregate_success_stats(result)
        else:
            self.jobs_failed += 1
            self.failed_jobs.append(
                FailedJob(
                    path=job_path,
                    document_id=getattr(result, "document_id", None),
                    error=getattr(result, "error", "Unknown error"),
                    saga_id=getattr(result, "saga_id", None),
                )
            )

    def _aggregate_success_stats(self, result: Any) -> None:
        """Aggregate stats from a successful ingestion."""
        stats = getattr(result, "stats", {}) or {}

        # Core counts
        self.chunks_created += stats.get("sections_upserted", 0)
        self.entities_created += stats.get("entities_upserted", 0)
        self.vectors_written += stats.get("vectors_upserted", 0)

        # Structural edge counts
        structural = stats.get("structural_edges", {})
        if isinstance(structural, dict):
            edge_stats = structural.get("stats", structural)
            for edge_type in [
                "NEXT_CHUNK",
                "PARENT_HEADING",
                "CHILD_OF",
                "PARENT_OF",
                "NEXT",
            ]:
                self.edges[edge_type] += edge_stats.get(edge_type, 0)

            # Record structural warnings
            for warning in structural.get("warnings", []):
                self.record_warning(warning)

    def record_warning(self, message: str) -> None:
        """
        Record a warning, deduplicating by message.

        Args:
            message: Warning message to record
        """
        self._warnings[message] += 1

    def record_exception(self, job_path: str, error: str) -> None:
        """
        Record an exception that occurred outside normal result handling.

        Args:
            job_path: File path of the job that failed
            error: Error message
        """
        self.jobs_processed += 1
        self.jobs_failed += 1
        self.failed_jobs.append(
            FailedJob(
                path=job_path,
                document_id=None,
                error=error,
            )
        )

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the run and generate summary dict.

        Returns:
            Complete summary dictionary for logging
        """
        self.end_time = time.monotonic()
        duration = round(self.end_time - self.start_time, 2)

        return {
            "run_id": self.run_id,
            "duration_seconds": duration,
            "jobs": {
                "processed": self.jobs_processed,
                "succeeded": self.jobs_succeeded,
                "failed": self.jobs_failed,
            },
            "documents": {
                "ingested": self.jobs_succeeded,
                "chunks_created": self.chunks_created,
                "entities_created": self.entities_created,
                "vectors_written": self.vectors_written,
            },
            "edges": dict(self.edges),
            "failures": [
                {
                    "path": f.path,
                    "document_id": f.document_id,
                    "error": f.error,
                    "saga_id": f.saga_id,
                }
                for f in self.failed_jobs
            ],
            "warnings": [
                {"message": msg, "count": count}
                for msg, count in self._warnings.items()
            ],
        }

    def emit_summary(self) -> Dict[str, Any]:
        """
        Emit the run summary as a structured log event.

        Returns:
            The summary dict that was logged
        """
        summary = self.finalize()

        # Main summary log
        logger.info(
            "ingestion_run_summary",
            run_id=summary["run_id"],
            duration_seconds=summary["duration_seconds"],
            jobs_processed=summary["jobs"]["processed"],
            jobs_succeeded=summary["jobs"]["succeeded"],
            jobs_failed=summary["jobs"]["failed"],
            chunks_created=summary["documents"]["chunks_created"],
            entities_created=summary["documents"]["entities_created"],
            vectors_written=summary["documents"]["vectors_written"],
            edges=summary["edges"],
        )

        # Separate failure log if any failures
        if self.jobs_failed > 0:
            logger.warning(
                "ingestion_run_had_failures",
                run_id=self.run_id,
                failed_count=self.jobs_failed,
                failures=summary["failures"],
            )

        # Separate warnings log if any warnings
        if self._warnings:
            logger.info(
                "ingestion_run_warnings",
                run_id=self.run_id,
                warning_count=sum(self._warnings.values()),
                warnings=summary["warnings"],
            )

        return summary

    @property
    def has_data(self) -> bool:
        """Check if any jobs have been recorded."""
        return self.jobs_processed > 0

"""
Job Reaper: Recovers stale jobs stuck in processing queue.

Scans the processing queue periodically and requeues or fails jobs that
exceed the configured timeout. Prevents jobs from being stuck indefinitely
when workers crash.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional

import redis
import structlog

from .queue import (
    KEY_DLQ,
    KEY_JOBS,
    KEY_PROCESSING,
    KEY_STATUS_HASH,
    IngestJob,
    JobStatus,
)

log = structlog.get_logger()


class JobReaper:
    """
    Background task to recover stale jobs from processing queue.

    Periodically scans jobs in processing queue and checks their age.
    Jobs exceeding timeout are either requeued (if retries remaining)
    or moved to dead-letter queue.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        timeout_seconds: int = 600,
        interval_seconds: int = 30,
        max_retries: int = 3,
        action: str = "requeue",
        enabled: bool = True,
    ):
        """
        Initialize job reaper.

        Args:
            redis_client: Redis connection
            timeout_seconds: Max age before job is considered stale
            interval_seconds: How often to scan for stale jobs
            max_retries: Max requeue attempts before DLQ
            action: What to do with stale jobs ("requeue" | "fail" | "dlq")
            enabled: Whether reaper is enabled
        """
        self.redis = redis_client
        self.timeout = timeout_seconds
        self.interval = interval_seconds
        self.max_retries = max_retries
        self.action = action
        self.enabled = enabled
        self.stats = {
            "total_reaped": 0,
            "requeued": 0,
            "failed": 0,
            "dlq": 0,
        }

    def _get_processing_jobs(self) -> List[str]:
        """Get all jobs currently in processing queue."""
        try:
            return self.redis.lrange(KEY_PROCESSING, 0, -1)
        except Exception as e:
            log.error("Failed to fetch processing queue", error=str(e))
            return []

    def _get_job_age(self, job_id: str) -> Optional[float]:
        """
        Get age of job in seconds from status hash.

        Returns None if job has no started_at timestamp.
        """
        try:
            status_json = self.redis.hget(KEY_STATUS_HASH, job_id)
            if not status_json:
                return None

            status = json.loads(status_json)
            started_at = status.get("started_at")
            if not started_at:
                return None

            return time.time() - started_at
        except Exception as e:
            log.warning("Failed to get job age", job_id=job_id, error=str(e))
            return None

    def _requeue_job(self, raw_json: str, job: IngestJob) -> bool:
        """
        Requeue a stale job back to pending queue.

        Returns True if successful.
        """
        try:
            job.attempts += 1

            # Remove from processing
            self.redis.lrem(KEY_PROCESSING, 1, raw_json)

            # Add back to pending
            self.redis.lpush(KEY_JOBS, job.to_json())

            # Update status
            self.redis.hset(
                KEY_STATUS_HASH,
                job.job_id,
                json.dumps(
                    {
                        "status": JobStatus.QUEUED.value,
                        "attempts": job.attempts,
                        "last_error": f"Reaped: exceeded {self.timeout}s timeout",
                        "requeued_at": time.time(),
                        "reaped": True,
                    }
                ),
            )

            self.stats["requeued"] += 1
            return True
        except Exception as e:
            log.error(
                "Failed to requeue job",
                job_id=job.job_id,
                error=str(e),
            )
            return False

    def _fail_job(self, raw_json: str, job: IngestJob, reason: str) -> bool:
        """
        Move job to dead-letter queue and mark as failed.

        Returns True if successful.
        """
        try:
            job.attempts += 1

            # Remove from processing
            self.redis.lrem(KEY_PROCESSING, 1, raw_json)

            # Add to DLQ
            self.redis.lpush(KEY_DLQ, raw_json)

            # Update status
            self.redis.hset(
                KEY_STATUS_HASH,
                job.job_id,
                json.dumps(
                    {
                        "status": JobStatus.FAILED.value,
                        "attempts": job.attempts,
                        "error": reason,
                        "failed_at": time.time(),
                        "reaped": True,
                    }
                ),
            )

            self.stats["dlq"] += 1
            return True
        except Exception as e:
            log.error(
                "Failed to move job to DLQ",
                job_id=job.job_id,
                error=str(e),
            )
            return False

    def _reap_job(self, raw_json: str) -> bool:
        """
        Process a single stale job.

        Returns True if job was reaped.
        """
        try:
            job = IngestJob.from_json(raw_json)
        except Exception as e:
            log.error("Failed to parse job JSON", error=str(e))
            return False

        age = self._get_job_age(job.job_id)
        if age is None:
            log.warning(
                "Job has no started_at timestamp, skipping",
                job_id=job.job_id,
            )
            return False

        # Not stale yet
        if age < self.timeout:
            return False

        log.info(
            "Reaping stale job",
            job_id=job.job_id,
            age_seconds=int(age),
            attempts=job.attempts,
            action=self.action,
        )

        # Decide action based on attempts and config
        if self.action == "requeue" and job.attempts < self.max_retries:
            success = self._requeue_job(raw_json, job)
            if success:
                log.info(
                    "Stale job requeued",
                    job_id=job.job_id,
                    attempts=job.attempts,
                )
        else:
            reason = (
                f"Reaped: exceeded {self.timeout}s timeout "
                f"after {job.attempts} attempts"
            )
            success = self._fail_job(raw_json, job, reason)
            if success:
                log.warning(
                    "Stale job moved to DLQ",
                    job_id=job.job_id,
                    attempts=job.attempts,
                    reason=reason,
                )

        if success:
            self.stats["total_reaped"] += 1
            return True

        return False

    async def reap_once(self) -> Dict[str, int]:
        """
        Scan processing queue once and reap stale jobs.

        Returns stats dict with counts of actions taken.
        """
        if not self.enabled:
            return {"skipped": 1, "reason": "reaper_disabled"}

        processing_jobs = self._get_processing_jobs()
        if not processing_jobs:
            return {"scanned": 0, "reaped": 0}

        reaped_count = 0
        for raw_json in processing_jobs:
            if self._reap_job(raw_json):
                reaped_count += 1

        if reaped_count > 0:
            log.info(
                "Reaper cycle complete",
                scanned=len(processing_jobs),
                reaped=reaped_count,
                total_reaped=self.stats["total_reaped"],
            )

        return {
            "scanned": len(processing_jobs),
            "reaped": reaped_count,
        }

    async def reap_loop(self):
        """
        Run reaper continuously in background.

        Call this from asyncio.create_task() in worker startup.
        """
        log.info(
            "Job reaper starting",
            timeout_seconds=self.timeout,
            interval_seconds=self.interval,
            max_retries=self.max_retries,
            enabled=self.enabled,
        )

        while True:
            try:
                await self.reap_once()
            except Exception as e:
                log.error(
                    "Reaper error",
                    error=str(e),
                    stats=self.stats,
                )

            await asyncio.sleep(self.interval)

    def get_stats(self) -> Dict[str, int]:
        """Get reaper statistics."""
        return dict(self.stats)

# Implements Phase 6, Task 6.2 (Progress event emission)
# See: /docs/app-spec-phase6.md
# See: /docs/implementation-plan-phase-6.md → Task 6.2
# See: /docs/pseudocode-phase6.md → Task 6.2

"""
Progress tracking and event emission for auto-ingestion jobs.

Emits structured progress events to Redis streams for CLI/UI consumption.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import redis

from src.shared.observability import get_logger

logger = get_logger(__name__)


class JobStage(str, Enum):
    """Job processing stages with deterministic ordering."""

    PENDING = "PENDING"
    PARSING = "PARSING"
    EXTRACTING = "EXTRACTING"
    GRAPHING = "GRAPHING"
    EMBEDDING = "EMBEDDING"
    VECTORS = "VECTORS"
    POSTCHECKS = "POSTCHECKS"
    REPORTING = "REPORTING"
    DONE = "DONE"
    ERROR = "ERROR"


# Stage weights for progress calculation (total = 100%)
STAGE_WEIGHTS = {
    JobStage.PENDING: 0,
    JobStage.PARSING: 10,
    JobStage.EXTRACTING: 15,
    JobStage.GRAPHING: 25,
    JobStage.EMBEDDING: 20,
    JobStage.VECTORS: 15,
    JobStage.POSTCHECKS: 5,
    JobStage.REPORTING: 5,
    JobStage.DONE: 5,
    JobStage.ERROR: 0,
}


@dataclass
class ProgressEvent:
    """Structured progress event."""

    job_id: str
    stage: JobStage
    percent: float
    message: str
    timestamp: float
    error: Optional[str] = None
    details: Optional[Dict] = None


class ProgressTracker:
    """
    Tracks job progress and emits events to Redis streams.

    Events are emitted to: ingest:events:<job_id>
    Each event includes: stage, percent, message, timestamp, error (optional)
    """

    def __init__(self, redis_client: redis.Redis, job_id: str):
        """
        Initialize progress tracker.

        Args:
            redis_client: Redis client instance
            job_id: Job ID to track
        """
        self.redis = redis_client
        self.job_id = job_id
        self.stream_key = f"ingest:events:{job_id}"
        self.current_stage = JobStage.PENDING
        self.start_time = time.time()

        logger.debug("Progress tracker initialized", job_id=job_id)

    def emit(
        self,
        stage: JobStage,
        message: str,
        percent: Optional[float] = None,
        error: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """
        Emit progress event to Redis stream.

        Args:
            stage: Current processing stage
            message: Human-readable progress message
            percent: Optional progress percentage (0-100)
            error: Optional error message
            details: Optional additional details dict
        """
        # Auto-calculate percent if not provided
        if percent is None:
            percent = self._calculate_percent(stage)

        event = ProgressEvent(
            job_id=self.job_id,
            stage=stage,
            percent=percent,
            message=message,
            timestamp=time.time(),
            error=error,
            details=details,
        )

        # Build event payload for Redis stream
        payload = {
            "job_id": self.job_id,
            "stage": stage.value,
            "percent": str(percent),
            "message": message,
            "timestamp": str(event.timestamp),
        }

        if error:
            payload["error"] = error

        if details:
            # Serialize details as JSON string
            import json

            payload["details"] = json.dumps(details)

        try:
            # Emit to Redis stream
            self.redis.xadd(self.stream_key, payload)
            self.current_stage = stage

            logger.debug(
                "Progress event emitted",
                job_id=self.job_id,
                stage=stage.value,
                percent=percent,
                message=message,
            )
        except Exception as exc:
            logger.error(
                "Failed to emit progress event",
                job_id=self.job_id,
                error=str(exc),
            )

    def advance(
        self, stage: JobStage, message: str, details: Optional[Dict] = None
    ) -> None:
        """
        Advance to next stage and emit progress event.

        Args:
            stage: Stage to advance to
            message: Progress message
            details: Optional details dict
        """
        self.emit(stage, message, details=details)

    def error(self, error_msg: str, details: Optional[Dict] = None) -> None:
        """
        Emit error event.

        Args:
            error_msg: Error message
            details: Optional error details
        """
        self.emit(
            stage=JobStage.ERROR,
            message=f"Error: {error_msg}",
            percent=self._calculate_percent(self.current_stage),
            error=error_msg,
            details=details,
        )

    def complete(self, message: str = "Job completed successfully") -> None:
        """
        Mark job as complete.

        Args:
            message: Completion message
        """
        self.emit(
            stage=JobStage.DONE,
            message=message,
            percent=100.0,
        )

        # Set TTL on event stream (7 days)
        try:
            self.redis.expire(self.stream_key, 7 * 24 * 60 * 60)
        except Exception as exc:
            logger.warning(
                "Failed to set TTL on event stream", job_id=self.job_id, error=str(exc)
            )

    def _calculate_percent(self, stage: JobStage) -> float:
        """
        Calculate cumulative progress percentage up to and including current stage.

        Args:
            stage: Current stage

        Returns:
            Progress percentage (0-100)
        """
        # Sum weights up to current stage
        cumulative = 0.0
        for s in JobStage:
            if s == JobStage.ERROR:
                continue
            cumulative += STAGE_WEIGHTS.get(s, 0)
            if s == stage:
                break

        return min(100.0, cumulative)

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time since tracker started."""
        return time.time() - self.start_time


class ProgressReader:
    """
    Read progress events from Redis stream.

    Used by CLI/UI to monitor job progress in real-time.
    """

    def __init__(self, redis_client: redis.Redis, job_id: str):
        """
        Initialize progress reader.

        Args:
            redis_client: Redis client instance
            job_id: Job ID to monitor
        """
        self.redis = redis_client
        self.job_id = job_id
        self.stream_key = f"ingest:events:{job_id}"
        self.last_id = "0-0"  # Start from beginning

    def read_events(
        self, count: int = 10, block_ms: Optional[int] = None
    ) -> list[ProgressEvent]:
        """
        Read progress events from stream.

        Args:
            count: Maximum number of events to read
            block_ms: Optional blocking timeout in milliseconds

        Returns:
            List of ProgressEvent objects
        """
        try:
            # Read from stream
            if block_ms:
                # Blocking read for real-time monitoring
                result = self.redis.xread(
                    {self.stream_key: self.last_id}, count=count, block=block_ms
                )
            else:
                # Non-blocking read for batch retrieval
                result = self.redis.xread({self.stream_key: self.last_id}, count=count)

            events = []
            if result:
                for stream_name, messages in result:
                    for msg_id, fields in messages:
                        # Parse event
                        event = self._parse_event(msg_id, fields)
                        if event:
                            events.append(event)
                            self.last_id = msg_id

            return events

        except Exception as exc:
            logger.error(
                "Failed to read progress events", job_id=self.job_id, error=str(exc)
            )
            return []

    def get_latest(self) -> Optional[ProgressEvent]:
        """
        Get latest progress event.

        Returns:
            Latest ProgressEvent or None
        """
        try:
            # Read last event from stream (reverse order)
            result = self.redis.xrevrange(self.stream_key, count=1)
            if result:
                msg_id, fields = result[0]
                return self._parse_event(msg_id, fields)
            return None
        except Exception as exc:
            logger.error(
                "Failed to get latest progress event",
                job_id=self.job_id,
                error=str(exc),
            )
            return None

    def _parse_event(self, msg_id: str, fields: Dict) -> Optional[ProgressEvent]:
        """
        Parse Redis stream message into ProgressEvent.

        Args:
            msg_id: Redis message ID
            fields: Message fields dict

        Returns:
            ProgressEvent or None if parsing fails
        """
        try:
            import json

            job_id = fields.get(b"job_id") or fields.get("job_id")
            stage_str = fields.get(b"stage") or fields.get("stage")
            percent_str = fields.get(b"percent") or fields.get("percent")
            message = fields.get(b"message") or fields.get("message")
            timestamp_str = fields.get(b"timestamp") or fields.get("timestamp")
            error = fields.get(b"error") or fields.get("error")
            details_str = fields.get(b"details") or fields.get("details")

            # Decode bytes if necessary
            if isinstance(job_id, bytes):
                job_id = job_id.decode("utf-8")
            if isinstance(stage_str, bytes):
                stage_str = stage_str.decode("utf-8")
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            if isinstance(error, bytes):
                error = error.decode("utf-8")
            if isinstance(details_str, bytes):
                details_str = details_str.decode("utf-8")

            stage = JobStage(stage_str)
            percent = float(percent_str)
            timestamp = float(timestamp_str)

            details = None
            if details_str:
                details = json.loads(details_str)

            return ProgressEvent(
                job_id=job_id,
                stage=stage,
                percent=percent,
                message=message,
                timestamp=timestamp,
                error=error,
                details=details,
            )

        except Exception as exc:
            logger.warning("Failed to parse progress event", error=str(exc))
            return None

    def wait_for_completion(self, timeout_seconds: Optional[int] = None) -> bool:
        """
        Wait for job to complete or error.

        Args:
            timeout_seconds: Optional timeout in seconds

        Returns:
            True if completed successfully, False if error or timeout
        """
        start_time = time.time()

        while True:
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                logger.warning("Timeout waiting for job completion", job_id=self.job_id)
                return False

            # Get latest event
            latest = self.get_latest()
            if latest:
                if latest.stage == JobStage.DONE:
                    return True
                elif latest.stage == JobStage.ERROR:
                    return False

            # Sleep before next check
            time.sleep(1)

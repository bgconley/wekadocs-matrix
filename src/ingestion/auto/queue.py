import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Tuple

import redis


# -------- Redis wiring -------------------------------------------------------
def _redis_from_env() -> redis.Redis:
    url = (
        os.getenv("REDIS_URI") or os.getenv("CACHE_REDIS_URI") or "redis://redis:6379/0"
    )
    return redis.Redis.from_url(url, decode_responses=True)


r = _redis_from_env()

# Namespacing & keys (list-queue pattern with processing + dead-letter)
NS = os.getenv("INGEST_NS", "ingest")
KEY_JOBS = f"{NS}:jobs"  # LIST (required)
KEY_PROCESSING = f"{NS}:processing"  # LIST (required)
KEY_STATUS_HASH = f"{NS}:status"  # HASH (job_id -> JSON)
KEY_DLQ = f"{NS}:dead"  # LIST (optional)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class IngestJob:
    job_id: str
    kind: str  # "file" | "url" | ...
    path: Optional[str] = None
    source: Optional[str] = None
    enqueued_at: float = 0.0
    attempts: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "IngestJob":
        return IngestJob(**json.loads(s))


def _ensure_list(key: str):
    t = r.type(key)
    if t and t not in ("list", b"list"):
        # quarantine the wrong-typed key; keep for forensics
        r.rename(key, f"{key}:bad:{int(time.time())}")


def ensure_key_types():
    for key in (KEY_JOBS, KEY_PROCESSING, KEY_DLQ):
        if r.exists(key):
            _ensure_list(key)
    # status hash is optional; create lazily on first set


def enqueue_file(path: str, source: Optional[str] = None) -> str:
    ensure_key_types()
    job = IngestJob(
        job_id=str(uuid.uuid4()),
        kind="file",
        path=os.path.abspath(path),
        source=source,
        enqueued_at=time.time(),
        attempts=0,
    )
    r.lpush(KEY_JOBS, job.to_json())
    r.hset(KEY_STATUS_HASH, job.job_id, json.dumps({"status": JobStatus.QUEUED.value}))
    return job.job_id


def brpoplpush(timeout: int = 1) -> Optional[Tuple[str, str]]:
    """Blocking pop from JOBS to PROCESSING. Returns (raw_json, job_id)."""
    ensure_key_types()
    raw = r.brpoplpush(KEY_JOBS, KEY_PROCESSING, timeout=timeout)
    if not raw:
        return None
    job = IngestJob.from_json(raw)
    r.hset(
        KEY_STATUS_HASH, job.job_id, json.dumps({"status": JobStatus.PROCESSING.value})
    )
    return raw, job.job_id


def ack(raw_json: str, job_id: str):
    r.lrem(KEY_PROCESSING, 1, raw_json)
    r.hset(
        KEY_STATUS_HASH,
        job_id,
        json.dumps({"status": JobStatus.DONE.value, "ts": time.time()}),
    )


def fail(
    raw_json: str,
    job_id: str,
    reason: str,
    requeue: bool = False,
    max_attempts: int = 5,
):
    try:
        job = IngestJob.from_json(raw_json)
    except Exception:
        job = IngestJob(job_id=job_id, kind="unknown")
    job.attempts += 1
    r.lrem(KEY_PROCESSING, 1, raw_json)
    if requeue and job.attempts < max_attempts:
        r.lpush(KEY_JOBS, job.to_json())
        r.hset(
            KEY_STATUS_HASH,
            job_id,
            json.dumps(
                {
                    "status": JobStatus.QUEUED.value,
                    "attempts": job.attempts,
                    "last_error": reason,
                }
            ),
        )
    else:
        r.lpush(KEY_DLQ, raw_json)
        r.hset(
            KEY_STATUS_HASH,
            job_id,
            json.dumps(
                {
                    "status": JobStatus.FAILED.value,
                    "attempts": job.attempts,
                    "error": reason,
                }
            ),
        )


def compute_checksum(file_path: str) -> str:
    """Compute SHA-256 checksum for duplicate detection."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_job_state(job_id: str) -> Optional[dict]:
    """Get job state from Redis status hash."""
    raw = r.hget(KEY_STATUS_HASH, job_id)
    if not raw:
        return None
    return json.loads(raw)


def update_job_state(job_id: str, **kwargs):
    """Update job state in Redis status hash."""
    state = get_job_state(job_id) or {}
    state.update(kwargs)
    state["updated_at"] = time.time()
    r.hset(KEY_STATUS_HASH, job_id, json.dumps(state))


# -------- JobQueue class for CLI compatibility -------------------------------
class JobQueue:
    """
    Wrapper class for CLI compatibility.
    Provides object-oriented interface to queue operations.
    """

    STATE_PREFIX = f"{NS}:state:"
    STREAM_JOBS = KEY_JOBS  # Expose for test compatibility
    CHECKSUM_SET = f"{NS}:checksums"  # Base prefix for checksum sets

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        # Use global redis instance for queue operations

    def enqueue(
        self, source_uri: str, checksum: str, tag: str, timestamp: float = None
    ) -> Optional[str]:
        """
        Enqueue a job for ingestion with duplicate detection.

        Args:
            source_uri: File path or URL
            checksum: SHA-256 checksum for duplicate detection
            tag: Ingestion tag (e.g., 'wekadocs')
            timestamp: Enqueue timestamp (defaults to current time)

        Returns:
            job_id: UUID of enqueued job, or None if duplicate
        """
        # Note: Skip ensure_key_types() since we use self.redis_client which handles key creation
        # The global ensure_key_types() uses module-level 'r' which may have different auth

        if timestamp is None:
            timestamp = time.time()

        # Check for duplicate using checksum
        checksum_key = f"{NS}:checksums:{tag}"
        if self.redis_client.sismember(checksum_key, checksum):
            # Duplicate - return None
            return None

        job_id = str(uuid.uuid4())

        # Store checksum to prevent duplicates
        self.redis_client.sadd(checksum_key, checksum)

        # Create job state
        state = {
            "job_id": job_id,
            "source_uri": source_uri,
            "tag": tag,
            "checksum": checksum,
            "status": JobStatus.QUEUED.value,
            "enqueued_at": timestamp,
            "updated_at": timestamp,
            "attempts": 0,
        }

        # Store state in hash
        self.redis_client.hset(KEY_STATUS_HASH, job_id, json.dumps(state))

        # Enqueue job
        job = IngestJob(
            job_id=job_id,
            kind="file",
            path=source_uri if source_uri.startswith("file://") else None,
            source=source_uri,
            enqueued_at=timestamp,
            attempts=0,
        )
        self.redis_client.lpush(KEY_JOBS, job.to_json())

        return job_id

    def get_state(self, job_id: str) -> Optional[dict]:
        """
        Get job state from Redis.

        Args:
            job_id: Job UUID

        Returns:
            Job state dict or None if not found
        """
        return get_job_state(job_id)

    def update_state(self, job_id: str, **kwargs):
        """
        Update job state.

        Args:
            job_id: Job UUID
            **kwargs: Fields to update
        """
        update_job_state(job_id, **kwargs)

    def dequeue(self, timeout: int = 1) -> Optional[Tuple[str, str]]:
        """
        Dequeue a job from the queue (blocking).

        Uses brpoplpush to atomically move from jobs to processing queue.

        Args:
            timeout: Blocking timeout in seconds

        Returns:
            Tuple of (raw_json, job_id) or None if timeout
        """
        raw = self.redis_client.brpoplpush(KEY_JOBS, KEY_PROCESSING, timeout=timeout)
        if not raw:
            return None

        # Decode if bytes
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        job = IngestJob.from_json(raw)
        self.redis_client.hset(
            KEY_STATUS_HASH,
            job.job_id,
            json.dumps({"status": JobStatus.PROCESSING.value}),
        )
        return raw, job.job_id

    def ack(self, raw_json: str, job_id: str):
        """
        Acknowledge successful job processing.

        Args:
            raw_json: The raw JSON string from dequeue
            job_id: The job ID
        """
        self.redis_client.lrem(KEY_PROCESSING, 1, raw_json)
        self.redis_client.hset(
            KEY_STATUS_HASH, job_id, json.dumps({"status": JobStatus.DONE.value})
        )

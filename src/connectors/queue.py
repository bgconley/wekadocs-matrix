"""
Redis-based ingestion queue for connector events.
Supports priority queuing, backpressure monitoring, and graceful degradation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from redis import Redis

from src.connectors.base import IngestionEvent

logger = logging.getLogger(__name__)


class IngestionQueue:
    """
    Redis-based queue for ingestion events from connectors.
    Monitors backpressure and provides degraded mode handling.
    """

    def __init__(
        self,
        redis_client: Redis,
        queue_name: str = "ingestion:queue",
        max_queue_size: int = 10000,
        backpressure_threshold: float = 0.8,
    ):
        """
        Args:
            redis_client: Redis connection
            queue_name: Name of the Redis list for queue
            max_queue_size: Maximum queue size before rejecting
            backpressure_threshold: Queue usage ratio to trigger backpressure (0.0-1.0)
        """
        self.redis = redis_client
        self.queue_name = queue_name
        self.max_queue_size = max_queue_size
        self.backpressure_threshold = backpressure_threshold

        logger.info(
            f"Ingestion queue initialized: {queue_name}, " f"max_size={max_queue_size}"
        )

    async def enqueue(self, event: IngestionEvent, priority: bool = False) -> bool:
        """
        Add event to queue.
        Args:
            event: Event to queue
            priority: If True, add to front of queue (LPUSH vs RPUSH)
        Returns:
            True if queued, False if queue full
        """
        try:
            # Check queue size
            current_size = await asyncio.to_thread(self.redis.llen, self.queue_name)
            if current_size >= self.max_queue_size:
                logger.warning(
                    f"Queue full ({current_size}/{self.max_queue_size}), "
                    f"rejecting event: {event.source_uri}"
                )
                return False

            # Serialize event
            payload = {
                "source_uri": event.source_uri,
                "source_type": event.source_type,
                "event_type": event.event_type,
                "metadata": event.metadata,
                "timestamp": event.timestamp.isoformat(),
            }
            payload_str = json.dumps(payload)

            # Add to queue (priority determines end)
            if priority:
                await asyncio.to_thread(self.redis.lpush, self.queue_name, payload_str)
            else:
                await asyncio.to_thread(self.redis.rpush, self.queue_name, payload_str)

            logger.debug(f"Queued event: {event.source_uri}")
            return True

        except Exception as e:
            logger.error(f"Error enqueueing event: {e}", exc_info=True)
            return False

    async def dequeue(self, timeout_seconds: int = 1) -> Optional[IngestionEvent]:
        """
        Dequeue next event (blocking with timeout).
        Returns:
            Next event or None if timeout
        """
        try:
            # BLPOP blocks until item available or timeout
            result = await asyncio.to_thread(
                self.redis.blpop, self.queue_name, timeout_seconds
            )
            if not result:
                return None

            _, payload_str = result
            payload = json.loads(payload_str)

            # Deserialize event
            event = IngestionEvent(
                source_uri=payload["source_uri"],
                source_type=payload["source_type"],
                event_type=payload["event_type"],
                metadata=payload["metadata"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
            )
            return event

        except Exception as e:
            logger.error(f"Error dequeuing event: {e}", exc_info=True)
            return None

    async def get_size(self) -> int:
        """Get current queue size."""
        return await asyncio.to_thread(self.redis.llen, self.queue_name)

    async def is_backpressure(self) -> bool:
        """Check if queue is experiencing backpressure."""
        size = await self.get_size()
        usage = size / self.max_queue_size
        return usage >= self.backpressure_threshold

    async def get_stats(self) -> dict:
        """Get queue statistics."""
        size = await self.get_size()
        usage = size / self.max_queue_size if self.max_queue_size > 0 else 0.0

        return {
            "queue_name": self.queue_name,
            "size": size,
            "max_size": self.max_queue_size,
            "usage_pct": usage * 100,
            "backpressure": usage >= self.backpressure_threshold,
        }

    async def clear(self) -> int:
        """Clear all events from queue (for testing). Returns count cleared."""
        return await asyncio.to_thread(self.redis.delete, self.queue_name)

"""
Base connector class for external systems integration.
Implements polling, webhooks, queue-based ingestion, and circuit breaker integration.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.connectors.circuit_breaker import CircuitBreaker
    from src.connectors.queue import IngestionQueue

logger = logging.getLogger(__name__)


class ConnectorStatus(str, Enum):
    """Connector operational status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DEGRADED = "degraded"  # Circuit breaker open
    ERROR = "error"


@dataclass
class ConnectorConfig:
    """Configuration for a connector."""

    name: str
    enabled: bool
    poll_interval_seconds: int
    batch_size: int
    max_retries: int
    backoff_base_seconds: float
    circuit_breaker_enabled: bool
    circuit_breaker_failure_threshold: int
    circuit_breaker_timeout_seconds: int
    webhook_secret: Optional[str] = None


@dataclass
class IngestionEvent:
    """Event representing a document to ingest."""

    source_uri: str
    source_type: str  # "github", "notion", "confluence"
    event_type: str  # "created", "updated", "deleted"
    metadata: Dict[str, Any]
    timestamp: datetime


class BaseConnector(ABC):
    """
    Base class for external system connectors.
    Implements common functionality: polling, queue integration, circuit breaker.
    """

    def __init__(
        self,
        config: ConnectorConfig,
        queue: "IngestionQueue",
        circuit_breaker: Optional["CircuitBreaker"] = None,
    ):
        self.config = config
        self.queue = queue
        self.circuit_breaker = circuit_breaker
        self.status = ConnectorStatus.IDLE
        self.last_cursor: Optional[str] = None
        self.stats = {
            "events_received": 0,
            "events_queued": 0,
            "errors": 0,
            "last_sync": None,
        }

    @abstractmethod
    async def fetch_changes(
        self, since: Optional[str] = None
    ) -> tuple[List[IngestionEvent], Optional[str]]:
        """
        Fetch changes from external system since cursor.
        Returns (events, next_cursor).
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify webhook signature.
        Must be implemented by subclasses.
        """
        pass

    async def sync(self) -> Dict[str, Any]:
        """
        Poll for changes and queue them for ingestion.
        Returns sync statistics.
        """
        if not self.config.enabled:
            logger.info(f"Connector {self.config.name} is disabled")
            return {"status": "disabled"}

        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_proceed():
            self.status = ConnectorStatus.DEGRADED
            logger.warning(
                f"Connector {self.config.name} circuit breaker is open, skipping sync"
            )
            return {"status": "degraded", "reason": "circuit_breaker_open"}

        self.status = ConnectorStatus.RUNNING
        start_time = datetime.utcnow()

        try:
            # Fetch changes
            events, next_cursor = await self.fetch_changes(since=self.last_cursor)
            self.stats["events_received"] += len(events)

            # Queue events for ingestion
            queued = 0
            for event in events:
                success = await self.queue.enqueue(event)
                if success:
                    queued += 1

            self.stats["events_queued"] += queued
            self.last_cursor = next_cursor
            self.stats["last_sync"] = datetime.utcnow()

            # Record success with circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            self.status = ConnectorStatus.IDLE
            duration = (datetime.utcnow() - start_time).total_seconds()

            return {
                "status": "success",
                "events_received": len(events),
                "events_queued": queued,
                "duration_seconds": duration,
                "next_cursor": next_cursor,
            }

        except Exception as e:
            self.stats["errors"] += 1
            self.status = ConnectorStatus.ERROR

            # Record failure with circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            logger.error(f"Error syncing {self.config.name}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def process_webhook(
        self, payload: Dict[str, Any], signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming webhook event.
        Returns processing result.
        """
        if not self.config.enabled:
            return {"status": "disabled"}

        # Verify signature if configured
        if self.config.webhook_secret and signature:
            if not await self.verify_webhook_signature(
                str(payload).encode(), signature
            ):
                logger.warning(f"Invalid webhook signature for {self.config.name}")
                return {"status": "error", "error": "invalid_signature"}

        try:
            # Convert webhook payload to ingestion event
            event = await self.webhook_to_event(payload)
            if event:
                success = await self.queue.enqueue(event)
                if success:
                    self.stats["events_queued"] += 1
                    return {"status": "success", "event": event.source_uri}
                else:
                    return {"status": "error", "error": "queue_full"}
            else:
                return {"status": "ignored", "reason": "no_action_needed"}

        except Exception as e:
            logger.error(
                f"Error processing webhook for {self.config.name}: {e}",
                exc_info=True,
            )
            return {"status": "error", "error": str(e)}

    @abstractmethod
    async def webhook_to_event(
        self, payload: Dict[str, Any]
    ) -> Optional[IngestionEvent]:
        """
        Convert webhook payload to IngestionEvent.
        Returns None if event should be ignored.
        Must be implemented by subclasses.
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "stats": self.stats,
            "circuit_breaker": (
                self.circuit_breaker.get_state().value if self.circuit_breaker else None
            ),
        }

"""
Connector manager for coordinating multiple external system connectors.
Handles registration, scheduling, and monitoring of all connectors.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from redis import Redis

from src.connectors.base import BaseConnector, ConnectorConfig
from src.connectors.circuit_breaker import CircuitBreaker
from src.connectors.github import GitHubConnector
from src.connectors.queue import IngestionQueue

logger = logging.getLogger(__name__)


class ConnectorManager:
    """
    Manages lifecycle of external system connectors.
    Coordinates polling, queuing, and circuit breaker state.
    """

    def __init__(self, redis_client: Redis, config: dict):
        """
        Args:
            redis_client: Redis connection for queue
            config: Configuration dict with connector settings
        """
        self.redis = redis_client
        self.config = config
        self.connectors: Dict[str, BaseConnector] = {}
        self.queue = IngestionQueue(
            redis_client,
            queue_name="ingestion:queue",
            max_queue_size=config.get("queue_max_size", 10000),
        )
        self.polling_tasks: List[asyncio.Task] = []
        self.running = False

        logger.info("Connector manager initialized")

    def register_connector(
        self, name: str, connector_type: str, connector_config: dict
    ) -> None:
        """
        Register a new connector.

        Args:
            name: Unique connector name
            connector_type: Type (github, notion, confluence)
            connector_config: Connector-specific configuration
        """
        try:
            # Create connector config
            config = ConnectorConfig(
                name=name,
                enabled=connector_config.get("enabled", True),
                poll_interval_seconds=connector_config.get(
                    "poll_interval_seconds", 300
                ),
                batch_size=connector_config.get("batch_size", 50),
                max_retries=connector_config.get("max_retries", 3),
                backoff_base_seconds=connector_config.get("backoff_base_seconds", 2.0),
                circuit_breaker_enabled=connector_config.get(
                    "circuit_breaker_enabled", True
                ),
                circuit_breaker_failure_threshold=connector_config.get(
                    "circuit_breaker_failure_threshold", 5
                ),
                circuit_breaker_timeout_seconds=connector_config.get(
                    "circuit_breaker_timeout_seconds", 60
                ),
                webhook_secret=connector_config.get("webhook_secret"),
            )
            config.metadata = connector_config.get("metadata", {})

            # Create circuit breaker if enabled
            circuit_breaker = None
            if config.circuit_breaker_enabled:
                circuit_breaker = CircuitBreaker(
                    failure_threshold=config.circuit_breaker_failure_threshold,
                    timeout_seconds=config.circuit_breaker_timeout_seconds,
                )

            # Create connector instance
            if connector_type == "github":
                connector = GitHubConnector(config, self.queue, circuit_breaker)
            else:
                raise ValueError(f"Unknown connector type: {connector_type}")

            self.connectors[name] = connector
            logger.info(f"Registered connector: {name} ({connector_type})")

        except Exception as e:
            logger.error(f"Error registering connector {name}: {e}")
            raise

    def get_connector(self, name: str) -> Optional[BaseConnector]:
        """Get connector by name."""
        return self.connectors.get(name)

    async def start_polling(self) -> None:
        """Start polling loops for all enabled connectors."""
        if self.running:
            logger.warning("Connector polling already running")
            return

        self.running = True
        logger.info("Starting connector polling loops")

        for name, connector in self.connectors.items():
            if connector.config.enabled:
                task = asyncio.create_task(self._poll_loop(name, connector))
                self.polling_tasks.append(task)
                logger.info(
                    f"Started polling for {name} "
                    f"(interval={connector.config.poll_interval_seconds}s)"
                )

    async def stop_polling(self) -> None:
        """Stop all polling loops."""
        if not self.running:
            return

        logger.info("Stopping connector polling loops")
        self.running = False

        # Cancel all tasks
        for task in self.polling_tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self.polling_tasks, return_exceptions=True)
        self.polling_tasks.clear()

        logger.info("All connector polling stopped")

    async def _poll_loop(self, name: str, connector: BaseConnector) -> None:
        """
        Polling loop for a single connector.
        Runs indefinitely until stopped.
        """
        interval = connector.config.poll_interval_seconds

        while self.running:
            try:
                # Check for backpressure
                if await self.queue.is_backpressure():
                    logger.warning(
                        f"Queue backpressure detected, pausing {name} polling"
                    )
                    await asyncio.sleep(interval)
                    continue

                # Sync with connector
                result = await connector.sync()
                logger.debug(f"Connector {name} sync result: {result}")

                # Wait for next poll
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                logger.info(f"Polling cancelled for {name}")
                break
            except Exception as e:
                logger.error(f"Error in polling loop for {name}: {e}")
                # Backoff on error
                await asyncio.sleep(min(interval, 60))

    def get_all_stats(self) -> List[Dict]:
        """Get statistics for all connectors."""
        stats = []
        for connector in self.connectors.values():
            stats.append(connector.get_stats())
        return stats

    async def get_queue_stats(self) -> Dict:
        """Get ingestion queue statistics."""
        return await self.queue.get_stats()

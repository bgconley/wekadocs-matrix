"""
Phase 5, Task 5.1 - External Systems Integration
Connectors for Notion, GitHub, Confluence with queue-based ingestion and circuit breakers.
"""

from src.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from src.connectors.circuit_breaker import CircuitBreaker, CircuitBreakerState
from src.connectors.github import GitHubConnector
from src.connectors.manager import ConnectorManager
from src.connectors.queue import IngestionQueue

__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "ConnectorStatus",
    "CircuitBreaker",
    "CircuitBreakerState",
    "GitHubConnector",
    "IngestionQueue",
    "ConnectorManager",
]

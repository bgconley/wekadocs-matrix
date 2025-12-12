"""Neo4j health checks for runtime monitoring.

This module provides health check functions for Neo4j
to verify connectivity and content availability.

Part of Phase 4 schema alignment hardening.
"""

import time
from dataclasses import dataclass
from typing import Optional

from neo4j import Driver

from src.shared.observability import get_logger

logger = get_logger(__name__)


@dataclass
class Neo4jHealthStatus:
    """Health check result for Neo4j."""

    healthy: bool
    latency_ms: float
    content_node_count: int
    has_content: bool
    message: str
    document_count: Optional[int] = None
    section_count: Optional[int] = None
    chunk_count: Optional[int] = None


def check_neo4j_health(driver: Driver, *, detailed: bool = False) -> Neo4jHealthStatus:
    """Quick health check for Neo4j connection and content.

    Args:
        driver: Neo4j driver instance
        detailed: If True, include per-label counts

    Returns:
        Neo4jHealthStatus with health information
    """
    start = time.time()
    try:
        with driver.session() as session:
            # Fast content check - Chunk nodes
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS content_count")
            record = result.single()
            content_count = record["content_count"] if record else 0
            latency_ms = (time.time() - start) * 1000

            # Optional: detailed counts
            doc_count = None
            chunk_count = None

            if detailed:
                doc_result = session.run("MATCH (d:Document) RETURN count(d) AS cnt")
                doc_count = doc_result.single()["cnt"]

                chunk_result = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt")
                chunk_count = chunk_result.single()["cnt"]

            # Determine health status
            is_healthy = content_count > 0
            if is_healthy:
                message = f"OK - {content_count} content nodes"
            else:
                message = (
                    "No content nodes found - database may be empty or schema mismatch"
                )

            return Neo4jHealthStatus(
                healthy=is_healthy,
                latency_ms=latency_ms,
                content_node_count=content_count,
                has_content=content_count > 0,
                message=message,
                document_count=doc_count,
                chunk_count=chunk_count,
            )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error("Neo4j health check failed", error=str(e))
        return Neo4jHealthStatus(
            healthy=False,
            latency_ms=elapsed_ms,
            content_node_count=0,
            has_content=False,
            message=f"Health check failed: {e}",
        )


def check_neo4j_connectivity(driver: Driver) -> bool:
    """Simple connectivity check for Neo4j.

    Args:
        driver: Neo4j driver instance

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with driver.session() as session:
            session.run("RETURN 1").single()
        return True
    except Exception as e:
        logger.error("Neo4j connectivity check failed", error=str(e))
        return False

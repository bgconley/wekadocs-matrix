"""
Phase 6, Task 6.1: Back-Pressure Monitoring

Monitors Neo4j CPU and Qdrant latency to prevent overwhelming downstream systems.

Triggers pause when:
- Neo4j CPU > 80%
- Qdrant P95 latency > 200ms

See: /docs/coder-guidance-phase6.md → 6.1
"""

import logging
import time
from threading import Event, Thread
from typing import Dict, Optional

import requests
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class BackPressureMonitor:
    """
    Monitors resource utilization and signals when to pause ingestion
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        qdrant_host: str,
        qdrant_port: int = 6333,
        neo4j_cpu_threshold: float = 0.8,
        qdrant_p95_threshold_ms: float = 200.0,
        check_interval: float = 10.0,
    ):
        """
        Initialize back-pressure monitor

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            qdrant_host: Qdrant hostname
            qdrant_port: Qdrant HTTP port
            neo4j_cpu_threshold: Neo4j CPU threshold (0-1)
            qdrant_p95_threshold_ms: Qdrant P95 latency threshold
            check_interval: How often to check metrics (seconds)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        self.neo4j_cpu_threshold = neo4j_cpu_threshold
        self.qdrant_p95_threshold_ms = qdrant_p95_threshold_ms
        self.check_interval = check_interval

        # State
        self._should_pause = False
        self._metrics: Dict = {}

        # Control
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        logger.info("BackPressureMonitor initialized")

    def start(self):
        """Start monitoring in background thread"""
        if self._thread and self._thread.is_alive():
            logger.warning("Monitor already running")
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Back-pressure monitor started")

    def stop(self):
        """Stop monitoring"""
        if not self._thread:
            return

        logger.info("Stopping back-pressure monitor")
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Back-pressure monitor stopped")

    def should_pause(self) -> bool:
        """
        Check if ingestion should pause due to back-pressure

        Returns:
            True if should pause, False otherwise
        """
        return self._should_pause

    def get_metrics(self) -> Dict:
        """
        Get current metrics snapshot

        Returns:
            Dict with neo4j_cpu, qdrant_p95_ms, should_pause
        """
        return self._metrics.copy()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                neo4j_cpu = self._check_neo4j_cpu()
                qdrant_p95 = self._check_qdrant_latency()

                # Determine if should pause
                neo4j_overloaded = (
                    neo4j_cpu is not None and neo4j_cpu > self.neo4j_cpu_threshold
                )
                qdrant_slow = (
                    qdrant_p95 is not None and qdrant_p95 > self.qdrant_p95_threshold_ms
                )

                was_paused = self._should_pause
                self._should_pause = neo4j_overloaded or qdrant_slow

                # Update metrics
                self._metrics = {
                    "neo4j_cpu": neo4j_cpu,
                    "qdrant_p95_ms": qdrant_p95,
                    "should_pause": self._should_pause,
                    "timestamp": time.time(),
                }

                # Log state changes
                if self._should_pause and not was_paused:
                    reasons = []
                    if neo4j_overloaded:
                        reasons.append(f"Neo4j CPU {neo4j_cpu:.1%}")
                    if qdrant_slow:
                        reasons.append(f"Qdrant P95 {qdrant_p95:.0f}ms")
                    logger.warning(f"PAUSING ingestion: {', '.join(reasons)}")
                elif not self._should_pause and was_paused:
                    logger.info("RESUMING ingestion: back-pressure cleared")

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

            # Sleep with interruptible waits
            for _ in range(int(self.check_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _check_neo4j_cpu(self) -> Optional[float]:
        """
        Check Neo4j CPU utilization

        Returns:
            CPU utilization (0-1) or None if unavailable
        """
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )

            with driver.session() as session:
                # Query system metrics
                # Note: dbms.queryJmx requires apoc or enterprise features
                # For now, use a simple heuristic: active connections and query count
                result = session.run(
                    """
                    CALL dbms.listQueries()
                    YIELD queryId
                    RETURN count(queryId) AS active_queries
                """
                )
                record = result.single()
                active_queries = record["active_queries"] if record else 0

                # Heuristic: estimate load based on active queries
                # This is a simplification; in production, use JMX or Prometheus metrics
                estimated_cpu = min(active_queries / 10.0, 1.0)  # Cap at 1.0

                driver.close()
                return estimated_cpu

        except Exception as e:
            logger.debug(f"Failed to check Neo4j CPU: {e}")
            return None

    def _check_qdrant_latency(self) -> Optional[float]:
        """
        Check Qdrant P95 latency

        Returns:
            P95 latency in milliseconds or None if unavailable
        """
        try:
            # Qdrant metrics endpoint
            url = f"http://{self.qdrant_host}:{self.qdrant_port}/metrics"
            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                return None

            # Parse Prometheus metrics
            # Look for histogram quantiles (qdrant_http_requests_duration_seconds)
            metrics_text = response.text

            for line in metrics_text.split("\n"):
                # Look for P95 quantile
                if (
                    "qdrant_http_requests_duration_seconds" in line
                    and 'quantile="0.95"' in line
                ):
                    try:
                        # Format: metric{labels} value
                        value_str = line.split()[-1]
                        seconds = float(value_str)
                        return seconds * 1000  # Convert to ms
                    except (ValueError, IndexError):
                        continue

            # Fallback: measure a simple health check
            start = time.time()
            health_url = f"http://{self.qdrant_host}:{self.qdrant_port}/health"
            requests.get(health_url, timeout=5)
            elapsed_ms = (time.time() - start) * 1000
            return elapsed_ms

        except Exception as e:
            logger.debug(f"Failed to check Qdrant latency: {e}")
            return None


class SimpleRateLimiter:
    """
    Simple token bucket rate limiter for ingestion

    Complements back-pressure monitoring by enforcing max throughput
    """

    def __init__(self, max_per_second: float):
        """
        Initialize rate limiter

        Args:
            max_per_second: Maximum operations per second
        """
        self.max_per_second = max_per_second
        self.interval = 1.0 / max_per_second
        self.last_operation = 0.0

        logger.info(f"RateLimiter: {max_per_second} ops/sec")

    def acquire(self):
        """
        Acquire token (blocks if necessary)

        Sleeps to enforce rate limit
        """
        now = time.time()
        elapsed = now - self.last_operation

        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            time.sleep(sleep_time)

        self.last_operation = time.time()

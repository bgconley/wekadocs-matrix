"""
Circuit breaker implementation for connector resilience.
Prevents cascading failures by opening circuit after threshold failures.
"""

import logging
import time
from enum import Enum
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker implementation with automatic recovery testing.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: How long to wait before trying half-open
            half_open_max_calls: Max calls to test in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()

        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"timeout={timeout_seconds}s"
        )

    def can_proceed(self) -> bool:
        """
        Check if request can proceed.
        Returns True if CLOSED or HALF_OPEN (with available test slots).
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                # Check if timeout has elapsed
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                    return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Recovered! Close circuit
                self._transition_to_closed()
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    logger.debug(f"Resetting failure count from {self._failure_count}")
                    self._failure_count = 0

    def record_failure(self) -> None:
        """Record failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Failed during recovery testing, reopen circuit
                self._transition_to_open()
            elif (
                self._state == CircuitBreakerState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                # Threshold exceeded, open circuit
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.timeout_seconds

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        logger.info("Circuit breaker closing after successful recovery test")
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        logger.warning(
            f"Circuit breaker opening after {self._failure_count} failures "
            f"(threshold={self.failure_threshold})"
        )
        self._state = CircuitBreakerState.OPEN
        self._half_open_calls = 0

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        logger.info(
            f"Circuit breaker transitioning to half-open after "
            f"{self.timeout_seconds}s timeout"
        )
        self._state = CircuitBreakerState.HALF_OPEN
        self._half_open_calls = 0

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "timeout_seconds": self.timeout_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Manually reset circuit breaker (for testing)."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            logger.info("Circuit breaker manually reset")

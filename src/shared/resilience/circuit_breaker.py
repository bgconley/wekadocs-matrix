"""
Thread-safe Circuit Breaker pattern implementation.

This module provides a reusable circuit breaker for resilient external service calls.
Extracted from providers to eliminate code duplication and ensure consistent behavior.

Usage:
    cb = CircuitBreaker(name="reranker", failure_threshold=5, recovery_timeout=30.0)

    if cb.allow_request():
        try:
            result = call_external_service()
            cb.record_success()
            return result
        except Exception as e:
            cb.record_failure()
            raise
    else:
        # Circuit is open, return degraded response
        return degraded_response()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_parse_int(env_var: str, default: int) -> int:
    """Parse integer from environment variable with fallback on invalid input.

    N2 Fix: Prevents application crash on misconfigured environment variables.
    """
    value = os.getenv(env_var)
    if not value:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            logger.warning(
                "circuit_breaker_config_invalid",
                extra={
                    "env_var": env_var,
                    "value": value,
                    "reason": "must_be_positive",
                    "using_default": default,
                },
            )
            return default
        return parsed
    except ValueError:
        logger.warning(
            "circuit_breaker_config_invalid",
            extra={
                "env_var": env_var,
                "value": value,
                "reason": "not_an_integer",
                "using_default": default,
            },
        )
        return default


def _safe_parse_float(env_var: str, default: float) -> float:
    """Parse float from environment variable with fallback on invalid input.

    N2 Fix: Prevents application crash on misconfigured environment variables.
    """
    value = os.getenv(env_var)
    if not value:
        return default
    try:
        parsed = float(value)
        if parsed <= 0:
            logger.warning(
                "circuit_breaker_config_invalid",
                extra={
                    "env_var": env_var,
                    "value": value,
                    "reason": "must_be_positive",
                    "using_default": default,
                },
            )
            return default
        return parsed
    except ValueError:
        logger.warning(
            "circuit_breaker_config_invalid",
            extra={
                "env_var": env_var,
                "value": value,
                "reason": "not_a_float",
                "using_default": default,
            },
        )
        return default


class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing recovery with single request


# Environment variable configuration with sensible defaults
# N2 Fix: Use safe parsers to prevent startup crash on invalid config
DEFAULT_FAILURE_THRESHOLD = _safe_parse_int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5)
DEFAULT_RECOVERY_TIMEOUT = _safe_parse_float("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 30.0)


class CircuitBreaker:
    """
    Thread-safe circuit breaker for external service resilience.

    The circuit breaker prevents cascading failures by:
    1. CLOSED state: Allows requests, tracks consecutive failures
    2. OPEN state: Rejects requests immediately (fail-fast)
    3. HALF_OPEN state: Allows one test request to check recovery

    Thread Safety:
        All state transitions are protected by a threading.Lock to ensure
        correct behavior in multi-threaded environments (e.g., FastAPI with
        thread pool executors).

    Configuration:
        Can be configured via environment variables:
        - CIRCUIT_BREAKER_FAILURE_THRESHOLD: Failures before opening (default: 5)
        - CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Seconds before testing recovery (default: 30)

    Attributes:
        name: Identifier for this circuit breaker (used in logging/metrics)
        failure_threshold: Consecutive failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
    ) -> None:
        """
        Initialize the circuit breaker.

        Args:
            name: Identifier for logging and metrics
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        # State tracking (protected by lock)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

        logger.debug(
            "circuit_breaker_initialized",
            extra={
                "name": name,
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
            },
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state (thread-safe read)."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Current failure count (thread-safe read)."""
        with self._lock:
            return self._failure_count

    def allow_request(self) -> bool:
        """
        Check if the circuit breaker allows a request to pass through.

        This method handles state transitions:
        - CLOSED: Always allows requests
        - OPEN: Checks if recovery timeout has passed, transitions to HALF_OPEN if so
        - HALF_OPEN: Allows one test request

        Returns:
            True if request is allowed, False if circuit is open and should fail-fast

        Thread Safety:
            This method is thread-safe. State transitions are atomic.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        logger.info(
                            "circuit_breaker_half_open",
                            extra={
                                "name": self.name,
                                "elapsed_seconds": elapsed,
                                "reason": "recovery_timeout_reached",
                            },
                        )
                        return True
                return False

            # HALF_OPEN: allow one request to test recovery
            return True

    def record_success(self) -> None:
        """
        Record a successful request.

        In HALF_OPEN state: Transitions to CLOSED (service recovered)
        In CLOSED state: Resets failure count

        Thread Safety:
            This method is thread-safe. State transitions are atomic.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(
                    "circuit_breaker_closed",
                    extra={
                        "name": self.name,
                        "reason": "recovery_success",
                    },
                )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed request.

        In HALF_OPEN state: Transitions back to OPEN (recovery failed)
        In CLOSED state: Increments failure count, opens if threshold reached

        Thread Safety:
            This method is thread-safe. State transitions are atomic.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Recovery test failed, reopen circuit
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker_reopened",
                    extra={
                        "name": self.name,
                        "reason": "recovery_failed",
                    },
                )
            elif self._failure_count >= self.failure_threshold:
                # Threshold reached, open circuit
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker_opened",
                    extra={
                        "name": self.name,
                        "failure_count": self._failure_count,
                        "threshold": self.failure_threshold,
                    },
                )

    def reset(self) -> None:
        """
        Manually reset the circuit breaker to CLOSED state.

        Use with caution - typically used for testing or manual recovery.

        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            previous_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(
                "circuit_breaker_reset",
                extra={
                    "name": self.name,
                    "previous_state": previous_state.value,
                },
            )

    def is_open(self) -> bool:
        """Check if circuit is currently open (fail-fast mode)."""
        with self._lock:
            return self._state == CircuitState.OPEN

    def is_closed(self) -> bool:
        """Check if circuit is currently closed (normal operation)."""
        with self._lock:
            return self._state == CircuitState.CLOSED

    def __repr__(self) -> str:
        """String representation for debugging."""
        with self._lock:
            return (
                f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
                f"failures={self._failure_count}/{self.failure_threshold})"
            )

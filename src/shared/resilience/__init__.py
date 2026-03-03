# =============================================================================
# @status: ACTIVE
# @called-by: (package init)
# =============================================================================
"""Resilience patterns for distributed system robustness."""

from src.shared.resilience.circuit_breaker import CircuitBreaker, CircuitState

__all__ = ["CircuitBreaker", "CircuitState"]

# =============================================================================
# @status: DEAD
# @reason: Entire ops/warmers package is orphaned; 0 imports from any active src/ module
# @safe-to-delete: Yes
# =============================================================================
"""Cache warmers for preloading frequently accessed queries."""

from .query_warmer import QueryWarmer

__all__ = ["QueryWarmer"]

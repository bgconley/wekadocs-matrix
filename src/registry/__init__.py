# =============================================================================
# @status: DEAD
# @reason: Entire registry package is orphaned; 0 imports from any active src/ module
# @safe-to-delete: Yes
# =============================================================================
"""
Registry package for index and provider management.
Phase 7C: Dimension-safe index management and provider tracking.
"""

from src.registry.index_registry import IndexRegistry

__all__ = ["IndexRegistry"]

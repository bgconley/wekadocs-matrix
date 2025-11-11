"""
Neo4j utilities and query safety guards.
Phase 7a: EXPLAIN-plan validation and performance hardening.
"""

from .explain_guard import (
    ExplainGuard,
    PlanRejected,
    PlanTooExpensive,
    validate_query_plan,
)

__all__ = [
    "ExplainGuard",
    "PlanRejected",
    "PlanTooExpensive",
    "validate_query_plan",
]

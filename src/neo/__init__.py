"""
Neo4j utilities and query safety guards.
Phase 7a: EXPLAIN-plan validation and performance hardening.
Phase 3/4: Schema validation and defensive queries.
"""

from .defensive_query import (
    run_defensive_query,
    run_existence_check,
)
from .explain_guard import (
    ExplainGuard,
    PlanRejected,
    PlanTooExpensive,
    validate_query_plan,
)
from .health import (
    Neo4jHealthStatus,
    check_neo4j_connectivity,
    check_neo4j_health,
)
from .schema_validator import (
    SchemaValidationResult,
    validate_neo4j_schema,
)

__all__ = [
    # EXPLAIN guard
    "ExplainGuard",
    "PlanRejected",
    "PlanTooExpensive",
    "validate_query_plan",
    # Schema validation (Phase 3)
    "SchemaValidationResult",
    "validate_neo4j_schema",
    # Defensive queries (Phase 4)
    "run_defensive_query",
    "run_existence_check",
    # Health checks (Phase 4)
    "Neo4jHealthStatus",
    "check_neo4j_health",
    "check_neo4j_connectivity",
]

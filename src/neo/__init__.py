# =============================================================================
# @status: ACTIVE
# @reason: Package init for Neo4j utilities. Eager imports of explain_guard,
#          defensive_query, and health were removed (Phase B cleanup) to
#          eliminate phantom loading. Active code imports directly from
#          submodules (e.g., from src.neo.schema_validator import ...).
# =============================================================================
"""
Neo4j utilities and query safety guards.
Phase 7a: EXPLAIN-plan validation and performance hardening.
Phase 3/4: Schema validation and defensive queries.
"""

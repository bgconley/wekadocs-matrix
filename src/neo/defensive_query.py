"""Defensive query wrapper with logging for empty results.

This module provides a query wrapper that detects and logs
unexpected empty or partial results from Neo4j queries.
Useful for detecting schema drift at runtime.

Part of Phase 4 schema alignment hardening.
"""

from typing import Any, Dict, List, Optional, Sequence

from neo4j import Session

from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    empty_query_results_total,
    partial_query_results_total,
)

logger = get_logger(__name__)


def run_defensive_query(
    session: Session,
    query: str,
    params: Dict[str, Any],
    *,
    query_name: str,
    expected_count: Optional[int] = None,
    input_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Execute query with defensive logging for empty/partial results.

    This wrapper monitors query results and logs warnings when:
    - Query returns 0 results but expected_count > 0
    - Query returns fewer results than expected_count

    Args:
        session: Neo4j session
        query: Cypher query string
        params: Query parameters
        query_name: Human-readable name for logging/metrics
        expected_count: Expected number of results (optional)
        input_ids: Input IDs for correlation in logs (optional)

    Returns:
        List of result dictionaries
    """
    rows = list(session.run(query, **params))
    result = [dict(row) for row in rows]
    actual_count = len(result)

    # Check for unexpected empty results
    if actual_count == 0 and expected_count and expected_count > 0:
        logger.warning(
            "Query returned empty results unexpectedly",
            query_name=query_name,
            expected_count=expected_count,
            input_ids_sample=list(input_ids or [])[:5],
        )
        empty_query_results_total.labels(service="neo4j", operation=query_name).inc()

    # Check for partial results
    elif expected_count and actual_count < expected_count:
        returned_ids = {row.get("id") for row in result if "id" in row}
        missing_ids = set(input_ids or []) - returned_ids
        logger.warning(
            "Query returned partial results",
            query_name=query_name,
            expected=expected_count,
            actual=actual_count,
            missing_sample=list(missing_ids)[:5],
        )
        partial_query_results_total.labels(service="neo4j", operation=query_name).inc()

    return result


def run_existence_check(
    session: Session,
    label: str,
    node_ids: Sequence[str],
) -> Dict[str, bool]:
    """Check which node IDs exist for a given label.

    Useful for pre-validation before complex queries.

    Args:
        session: Neo4j session
        label: Node label to check
        node_ids: IDs to verify

    Returns:
        Dict mapping node_id -> exists (bool)
    """
    if not node_ids:
        return {}

    query = f"""
    UNWIND $ids AS id
    OPTIONAL MATCH (n:`{label}` {{id: id}})
    RETURN id, n IS NOT NULL AS exists
    """
    result = session.run(query, ids=list(node_ids))
    return {record["id"]: record["exists"] for record in result}

"""
EXPLAIN-plan validation guard for Neo4j queries.
Phase 7a: Reject expensive/unbounded queries before execution.

See: /docs/phase-7-integration-plan.md
See: /docs/phase7-target-phase-tasklist.md Day 1
"""

import re
from typing import Any, Dict, Optional

from neo4j import Driver

from src.neo.schema import RELATIONSHIP_TYPES
from src.shared.observability import get_logger

logger = get_logger(__name__)


class PlanRejected(Exception):
    """Raised when a query plan violates safety constraints."""

    pass


class PlanTooExpensive(PlanRejected):
    """Raised when a query plan exceeds cost/row thresholds."""

    pass


class ExplainGuard:
    """
    Validates Neo4j query plans using EXPLAIN before execution.
    Rejects queries that violate safety constraints.
    """

    # Default thresholds (can be overridden)
    MAX_ESTIMATED_ROWS = 10000
    MAX_DB_HITS = 100000
    ALLOWED_LABELS = {"Section", "Entity", "Chunk", "Document", "Topic"}
    ALLOWED_RELATIONSHIPS = RELATIONSHIP_TYPES

    # Dangerous operators that indicate unbounded expansion
    DANGEROUS_OPERATORS = {
        "Expand(All)",
        "VarLengthExpand(All)",
    }

    RELATIONSHIP_PATTERN = re.compile(r"type: +'([A-Z0-9_]+)'")

    def __init__(
        self,
        driver: Driver,
        max_estimated_rows: Optional[int] = None,
        max_db_hits: Optional[int] = None,
        allowed_labels: Optional[set] = None,
        allowed_relationships: Optional[set] = None,
    ):
        """
        Initialize EXPLAIN guard with custom thresholds.

        Args:
            driver: Neo4j driver instance
            max_estimated_rows: Maximum estimated rows (default: 10000)
            max_db_hits: Maximum db hits (default: 100000)
            allowed_labels: Whitelist of node labels (default: Section, Entity, Chunk, Document, Topic)
            allowed_relationships: Whitelist of relationship types (default: MENTIONS, HAS_SECTION)
        """
        self.driver = driver
        self.max_estimated_rows = max_estimated_rows or self.MAX_ESTIMATED_ROWS
        self.max_db_hits = max_db_hits or self.MAX_DB_HITS
        self.allowed_labels = allowed_labels or self.ALLOWED_LABELS
        self.allowed_relationships = allowed_relationships or self.ALLOWED_RELATIONSHIPS

    def validate_plan(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a query plan using EXPLAIN.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            Dict containing plan metadata

        Raises:
            PlanRejected: If plan violates safety constraints
            PlanTooExpensive: If plan exceeds cost thresholds
        """
        params = params or {}

        try:
            with self.driver.session() as session:
                # Get EXPLAIN plan
                result = session.run(f"EXPLAIN {cypher}", params)
                plan = result.consume().plan

                if plan is None:
                    raise PlanRejected("No execution plan returned")

                # Extract plan metadata
                metadata = self._extract_plan_metadata(plan)

                # Run validation checks
                self._check_dangerous_operators(metadata)
                self._check_estimated_rows(metadata)
                self._check_db_hits(metadata)
                self._check_label_scans(metadata)

                logger.debug(
                    "Query plan validated",
                    estimated_rows=metadata.get("estimated_rows"),
                    db_hits=metadata.get("db_hits"),
                    operators=metadata.get("operators", [])[:5],
                )

                return metadata

        except PlanRejected:
            raise
        except Exception as e:
            logger.error(f"EXPLAIN validation failed: {e}", exc_info=True)
            raise PlanRejected(f"Failed to validate query plan: {e}")

    def _extract_plan_metadata(self, plan: Any) -> Dict[str, Any]:
        """Extract relevant metadata from execution plan."""
        metadata = {
            "estimated_rows": 0,
            "db_hits": 0,
            "operators": [],
            "details": [],
        }

        def traverse_plan(node: Any, depth: int = 0):
            """Recursively traverse plan tree."""
            if node is None:
                return

            # Extract operator name
            operator = (
                node.operator_type if hasattr(node, "operator_type") else str(node)
            )
            metadata["operators"].append(operator)

            # Extract estimates
            if hasattr(node, "arguments"):
                args = node.arguments
                if "EstimatedRows" in args:
                    metadata["estimated_rows"] += float(args["EstimatedRows"])
                if "DbHits" in args:
                    metadata["db_hits"] += int(args["DbHits"])

                # Track identifiers/relationship details for validation
                if "Details" in args:
                    details = args["Details"]
                    if isinstance(details, str):
                        metadata["details"].append(details)

            # Traverse children
            if hasattr(node, "children"):
                for child in node.children:
                    traverse_plan(child, depth + 1)

        traverse_plan(plan)
        return metadata

    def _check_dangerous_operators(self, metadata: Dict[str, Any]) -> None:
        """Check for dangerous operators like Expand(All)."""
        operators = metadata.get("operators", [])

        for op in operators:
            for dangerous_op in self.DANGEROUS_OPERATORS:
                if dangerous_op in op:
                    raise PlanRejected(
                        f"Query rejected: contains dangerous operator '{dangerous_op}'. "
                        f"Use bounded traversal with explicit depth limits."
                    )

    def _check_estimated_rows(self, metadata: Dict[str, Any]) -> None:
        """Check if estimated rows exceed threshold."""
        estimated_rows = metadata.get("estimated_rows", 0)

        if estimated_rows > self.max_estimated_rows:
            raise PlanTooExpensive(
                f"Query rejected: estimated rows ({estimated_rows:.0f}) exceeds "
                f"threshold ({self.max_estimated_rows}). Refine query with filters."
            )

    def _check_db_hits(self, metadata: Dict[str, Any]) -> None:
        """Check if estimated db hits exceed threshold."""
        db_hits = metadata.get("db_hits", 0)

        if db_hits > self.max_db_hits:
            raise PlanTooExpensive(
                f"Query rejected: estimated db hits ({db_hits}) exceeds "
                f"threshold ({self.max_db_hits}). Add indexes or constraints."
            )

    def _check_label_scans(self, metadata: Dict[str, Any]) -> None:
        """Check for unlabeled scans or scans outside allowed labels."""
        operators = metadata.get("operators", [])

        for op in operators:
            # Check for all nodes scan (very expensive)
            if "AllNodesScan" in op:
                raise PlanRejected(
                    "Query rejected: contains AllNodesScan (unlabeled). "
                    "Add explicit node labels to query."
                )

            # Check NodeByLabelScan for allowed labels
            if "NodeByLabelScan" in op:
                # Extract label from operator (rough heuristic)
                # More robust parsing would require deeper plan inspection
                for label in self.allowed_labels:
                    if label in op:
                        break
                else:
                    # If no allowed label found in operator string, log warning
                    # (This is a conservative check; may need refinement)
                    logger.warning(
                        f"NodeByLabelScan may use non-whitelisted label: {op}"
                    )

        self._check_relationship_types(metadata)

    def _check_relationship_types(self, metadata: Dict[str, Any]) -> None:
        """Ensure traversed relationship types stay within the allow-list."""
        details = metadata.get("details", [])
        disallowed: set[str] = set()

        for entry in details:
            for rel_type in self.RELATIONSHIP_PATTERN.findall(entry):
                if rel_type not in self.allowed_relationships:
                    disallowed.add(rel_type)

        if disallowed:
            raise PlanRejected(
                "Query rejected: uses non-whitelisted relationships "
                f"{sorted(disallowed)}. Update src/neo/schema.py if these edges "
                "are part of the supported schema."
            )


def validate_query_plan(
    driver: Driver,
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    **guard_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to validate a query plan.

    Args:
        driver: Neo4j driver instance
        cypher: Cypher query string
        params: Query parameters
        **guard_kwargs: Additional arguments for ExplainGuard

    Returns:
        Dict containing plan metadata

    Raises:
        PlanRejected: If plan violates safety constraints
    """
    guard = ExplainGuard(driver, **guard_kwargs)
    return guard.validate_plan(cypher, params)

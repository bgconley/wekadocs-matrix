"""
Graph contract checks for CI gate validation and runtime monitoring.

This module implements the contract checks from the Neo4j Overhaul Plan (Section 5).
These checks ensure the graph contract is intact and can be used as:

1. CI gate: Block merges if graph contract is violated
2. Runtime monitoring: Detect regressions in production
3. End-of-run reconciliation: Identify documents needing repair

The checks are designed to be fast, bounded, and informative.

Usage:
    from src.neo.contract_checks import GraphContractChecker

    checker = GraphContractChecker(neo4j_session)
    result = checker.run_all_checks()

    if not result.passed:
        for check in result.failed_checks:
            logger.error(f"Contract violation: {check.name}: {check.message}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


class CheckSeverity(Enum):
    """Severity levels for contract checks."""

    ERROR = "error"  # Blocks CI, requires immediate fix
    WARNING = "warning"  # Should be fixed, doesn't block
    INFO = "info"  # Informational, for monitoring


@dataclass
class CheckResult:
    """Result of a single contract check."""

    name: str
    passed: bool
    severity: CheckSeverity
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    sample_violations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ContractCheckResult:
    """Aggregate result of all contract checks."""

    passed: bool = True
    checks: List[CheckResult] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    @property
    def failed_checks(self) -> List[CheckResult]:
        """Return only failed checks."""
        return [c for c in self.checks if not c.passed]

    @property
    def error_checks(self) -> List[CheckResult]:
        """Return only ERROR severity checks that failed."""
        return [
            c for c in self.checks if not c.passed and c.severity == CheckSeverity.ERROR
        ]


class GraphContractChecker:
    """
    Validates graph contract invariants.

    Implements checks from Section 5 of the Neo4j Overhaul Plan:
    - Document membership (each chunk belongs to exactly one document)
    - NEXT_CHUNK correctness (no branching, no cycles)
    - Hierarchy coverage (chunks with parent_path have parent_chunk_id)
    - Entity normalization (all entities have normalized_name)

    All checks are bounded and return samples of violations, not exhaustive lists.
    """

    # Threshold for hierarchy coverage (proportion of chunks with resolved parents)
    HIERARCHY_COVERAGE_THRESHOLD = 0.90

    def __init__(self, session, *, sample_limit: int = 50):
        """
        Initialize the contract checker.

        Args:
            session: Neo4j session or driver
            sample_limit: Maximum violations to return per check
        """
        self.session = session
        self.sample_limit = sample_limit

    def run_all_checks(
        self,
        *,
        skip_expensive: bool = False,
    ) -> ContractCheckResult:
        """
        Run all contract checks and return aggregate result.

        Args:
            skip_expensive: Skip expensive checks (cycle detection)

        Returns:
            ContractCheckResult with all check results
        """
        result = ContractCheckResult()

        # Core checks (always run)
        checks = [
            self.check_chunk_document_membership(),
            self.check_next_chunk_no_branching(),
            self.check_hierarchy_coverage(),
            self.check_entity_normalization(),
            self.check_chunk_id_uniqueness(),
        ]

        # Expensive checks (optional)
        if not skip_expensive:
            checks.append(self.check_next_chunk_no_cycles())

        for check in checks:
            result.checks.append(check)
            if not check.passed:
                if check.severity == CheckSeverity.ERROR:
                    result.error_count += 1
                    result.passed = False
                elif check.severity == CheckSeverity.WARNING:
                    result.warning_count += 1

        logger.info(
            "graph_contract_check_complete",
            passed=result.passed,
            error_count=result.error_count,
            warning_count=result.warning_count,
            checks_run=len(result.checks),
        )

        return result

    def check_chunk_document_membership(self) -> CheckResult:
        """
        Check that each chunk belongs to exactly one document via HAS_CHUNK.

        Section 5.1: "Exactly one doc membership per chunk"
        """
        query = """
        MATCH (c:Chunk)
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WITH c, count(d) AS doc_count
        WHERE doc_count <> 1
        RETURN c.chunk_id AS chunk_id, c.document_id AS document_id, doc_count
        LIMIT $limit
        """
        result = self.session.run(query, limit=self.sample_limit)
        violations = [dict(record) for record in result]

        if violations:
            return CheckResult(
                name="chunk_document_membership",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"{len(violations)} chunks have incorrect document membership",
                details={"violation_count": len(violations)},
                sample_violations=violations,
            )

        return CheckResult(
            name="chunk_document_membership",
            passed=True,
            severity=CheckSeverity.ERROR,
            message="All chunks have exactly one document membership",
        )

    def check_next_chunk_no_branching(self) -> CheckResult:
        """
        Check that no chunk has multiple outgoing NEXT_CHUNK edges.

        Section 5.2: "NEXT_CHUNK correctness signals"
        """
        query = """
        MATCH (c:Chunk)-[r:NEXT_CHUNK]->()
        WITH c, count(r) AS out_deg
        WHERE out_deg > 1
        RETURN c.chunk_id AS chunk_id, c.document_id AS document_id, out_deg
        LIMIT $limit
        """
        result = self.session.run(query, limit=self.sample_limit)
        violations = [dict(record) for record in result]

        if violations:
            return CheckResult(
                name="next_chunk_no_branching",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"{len(violations)} chunks have multiple NEXT_CHUNK edges",
                details={"violation_count": len(violations)},
                sample_violations=violations,
            )

        return CheckResult(
            name="next_chunk_no_branching",
            passed=True,
            severity=CheckSeverity.ERROR,
            message="No NEXT_CHUNK branching detected",
        )

    def check_next_chunk_no_cycles(self, max_depth: int = 50) -> CheckResult:
        """
        Check that NEXT_CHUNK relationships have no cycles.

        This is an expensive check - limited depth traversal.
        """
        query = """
        MATCH (c:Chunk)
        WHERE EXISTS { MATCH (c)-[:NEXT_CHUNK]->() }
        WITH c LIMIT 1000
        MATCH p=(c)-[:NEXT_CHUNK*1..$max_depth]->(c)
        RETURN c.chunk_id AS cycle_chunk, c.document_id AS document_id
        LIMIT $limit
        """
        result = self.session.run(query, max_depth=max_depth, limit=self.sample_limit)
        violations = [dict(record) for record in result]

        if violations:
            return CheckResult(
                name="next_chunk_no_cycles",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"{len(violations)} NEXT_CHUNK cycles detected",
                details={"violation_count": len(violations), "max_depth": max_depth},
                sample_violations=violations,
            )

        return CheckResult(
            name="next_chunk_no_cycles",
            passed=True,
            severity=CheckSeverity.ERROR,
            message="No NEXT_CHUNK cycles detected",
        )

    def check_hierarchy_coverage(self) -> CheckResult:
        """
        Check that chunks with parent_path have parent_chunk_id resolved.

        Section 5.3: "Hierarchy coverage threshold"
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.parent_path_norm CONTAINS ' > '
        WITH count(c) AS should_have_parent
        MATCH (c:Chunk)
        WHERE c.parent_path_norm CONTAINS ' > ' AND c.parent_chunk_id IS NOT NULL
        WITH should_have_parent, count(c) AS has_parent
        RETURN
            should_have_parent,
            has_parent,
            CASE
                WHEN should_have_parent = 0 THEN 1.0
                ELSE (has_parent * 1.0 / should_have_parent)
            END AS coverage
        """
        result = self.session.run(query)
        record = result.single()

        if not record:
            return CheckResult(
                name="hierarchy_coverage",
                passed=True,
                severity=CheckSeverity.WARNING,
                message="No chunks with hierarchy detected",
            )

        coverage = record["coverage"]
        should_have = record["should_have_parent"]
        has_parent = record["has_parent"]

        passed = coverage >= self.HIERARCHY_COVERAGE_THRESHOLD

        # Get sample of unresolved chunks if coverage is low
        sample_violations = []
        if not passed:
            sample_query = """
            MATCH (c:Chunk)
            WHERE c.parent_path_norm CONTAINS ' > '
              AND c.parent_chunk_id IS NULL
            RETURN c.chunk_id AS chunk_id, c.document_id AS document_id,
                   c.parent_path_norm AS parent_path
            LIMIT $limit
            """
            sample_result = self.session.run(sample_query, limit=self.sample_limit)
            sample_violations = [dict(r) for r in sample_result]

        return CheckResult(
            name="hierarchy_coverage",
            passed=passed,
            severity=CheckSeverity.WARNING,
            message=f"Hierarchy coverage: {coverage:.1%} ({has_parent}/{should_have})",
            details={
                "coverage": coverage,
                "should_have_parent": should_have,
                "has_parent": has_parent,
                "threshold": self.HIERARCHY_COVERAGE_THRESHOLD,
            },
            sample_violations=sample_violations,
        )

    def check_entity_normalization(self) -> CheckResult:
        """
        Check that all entities have normalized_name set.

        Section 6: "Entity hygiene"
        """
        query = """
        MATCH (e:Entity)
        WHERE e.name IS NOT NULL AND e.normalized_name IS NULL
        RETURN count(e) AS unnormalized_count
        """
        result = self.session.run(query)
        record = result.single()
        unnormalized = record["unnormalized_count"] if record else 0

        if unnormalized > 0:
            # Get samples
            sample_query = """
            MATCH (e:Entity)
            WHERE e.name IS NOT NULL AND e.normalized_name IS NULL
            RETURN e.name AS name, e.entity_type AS entity_type
            LIMIT $limit
            """
            sample_result = self.session.run(sample_query, limit=self.sample_limit)
            sample_violations = [dict(r) for r in sample_result]

            return CheckResult(
                name="entity_normalization",
                passed=False,
                severity=CheckSeverity.WARNING,
                message=f"{unnormalized} entities missing normalized_name",
                details={"unnormalized_count": unnormalized},
                sample_violations=sample_violations,
            )

        return CheckResult(
            name="entity_normalization",
            passed=True,
            severity=CheckSeverity.WARNING,
            message="All entities have normalized_name",
        )

    def check_chunk_id_uniqueness(self) -> CheckResult:
        """
        Check that chunk_id values are unique.

        Precondition for uniqueness constraint.
        """
        query = """
        MATCH (c:Chunk)
        WITH c.chunk_id AS id, count(*) AS n
        WHERE id IS NOT NULL AND n > 1
        RETURN id, n
        ORDER BY n DESC
        LIMIT $limit
        """
        result = self.session.run(query, limit=self.sample_limit)
        violations = [dict(record) for record in result]

        if violations:
            return CheckResult(
                name="chunk_id_uniqueness",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"{len(violations)} duplicate chunk_id values found",
                details={"violation_count": len(violations)},
                sample_violations=violations,
            )

        return CheckResult(
            name="chunk_id_uniqueness",
            passed=True,
            severity=CheckSeverity.ERROR,
            message="All chunk_id values are unique",
        )

    def find_documents_needing_repair(self) -> List[str]:
        """
        Find documents that need structural edge repair.

        Used for end-of-run reconciliation to identify documents with:
        - Missing NEXT_CHUNK edges
        - Missing HAS_CHUNK edges
        - Invalid structural invariants

        Returns:
            List of document IDs needing repair
        """
        # Find docs with chunks but no NEXT_CHUNK edges
        query = """
        MATCH (c:Chunk)
        WHERE c.document_id IS NOT NULL
        WITH c.document_id AS doc_id, collect(c) AS chunks
        WHERE size(chunks) > 1
        OPTIONAL MATCH (c1:Chunk {document_id: doc_id})-[r:NEXT_CHUNK]->()
        WITH doc_id, size(chunks) AS chunk_count, count(r) AS next_chunk_count
        WHERE next_chunk_count < chunk_count - 1
        RETURN doc_id
        ORDER BY chunk_count DESC
        LIMIT 100
        """
        result = self.session.run(query)
        return [record["doc_id"] for record in result]


def run_contract_checks(
    session,
    *,
    skip_expensive: bool = False,
    sample_limit: int = 50,
) -> ContractCheckResult:
    """
    Convenience function to run all contract checks.

    Args:
        session: Neo4j session
        skip_expensive: Skip expensive checks
        sample_limit: Maximum violations per check

    Returns:
        ContractCheckResult with all check results
    """
    checker = GraphContractChecker(session, sample_limit=sample_limit)
    return checker.run_all_checks(skip_expensive=skip_expensive)

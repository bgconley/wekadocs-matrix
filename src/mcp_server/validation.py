"""
Cypher Validation System (Task 2.2)
Implements regex guards + EXPLAIN plan gates for safe query execution.
See: /docs/spec.md §4.2 (Query planning & safety)
See: /docs/pseudocode-reference.md Phase 2, Task 2.2
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.shared.config import get_config


@dataclass
class ValidationError(Exception):
    """Raised when a query fails validation."""

    reason: str
    query: str
    suggestion: Optional[str] = None

    def __str__(self):
        msg = f"Query validation failed: {self.reason}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Result of query validation."""

    valid: bool
    query: str
    params: Dict[str, Any]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class PlanAnalysis:
    """Analysis of Neo4j EXPLAIN plan."""

    estimated_rows: int
    node_by_label_scans: int
    expand_all_ops: int
    max_depth: int
    operator_tree: List[str]
    is_safe: bool
    warnings: List[str]


class CypherValidator:
    """
    Validates Cypher queries using:
    1. Regex guardrails (forbidden keywords, literals)
    2. Parameter enforcement
    3. EXPLAIN plan inspection
    """

    # Forbidden keywords that should never appear in user queries
    FORBIDDEN_KEYWORDS = [
        r"\bDELETE\b",
        r"\bDETACH\s+DELETE\b",
        r"\bSET\b(?!\s+\w+\s*=\s*\$)",  # SET allowed only with params
        r"\bREMOVE\b",
        r"\bDROP\b",
        r"\bCREATE\s+(?:CONSTRAINT|INDEX)\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bMERGE\b",  # MERGE can be expensive and modify graph
        r"\bCALL\s+(?!db\.indexes|db\.constraints)",  # Only allow metadata calls
    ]

    # Dangerous patterns that will raise ValidationError
    BLOCKING_PATTERNS = [
        (
            r"\bMATCH\s+\([^)]*\)\s*,\s*\([^)]*\)",
            "Cartesian product detected (comma-separated nodes)",
        ),
        (r"\bMATCH\s+\(\s*\)\s*-", "Cartesian product detected (match all nodes)"),
    ]

    # Dangerous patterns that generate warnings only
    DANGEROUS_PATTERNS = [
        (r"\bMATCH\s+.*-\[\s*\]-", "Unbounded relationship match"),
        (
            r"\(\s*\w+\s*:\s*\w+\s*\{[^}]*WHERE",
            "WHERE in node pattern (use proper WHERE clause)",
        ),
    ]

    # Correct regex for variable-length patterns: *min..max
    VARIABLE_LENGTH_PATTERN = re.compile(r"\*(\d+)\.\.(\d+)")

    def __init__(self, neo4j_driver=None):
        """Initialize validator with config and optional Neo4j driver for EXPLAIN."""
        self.config = get_config()

        self.max_depth = self.config.validator.max_depth
        self.max_label_scans = self.config.validator.max_label_scans
        self.max_expand_ops = self.config.validator.max_expand_ops
        self.max_estimated_rows = self.config.validator.max_estimated_rows
        self.timeout_seconds = self.config.validator.timeout_seconds
        self.enforce_parameters = self.config.validator.enforce_parameters
        self.enforce_limits = self.config.validator.enforce_limits

        self.neo4j_driver = neo4j_driver

    def validate(self, query: str, params: Dict[str, Any]) -> ValidationResult:
        """
        Validate a Cypher query through multiple checks.
        Returns ValidationResult or raises ValidationError.
        """
        warnings = []

        # Step 1: Strip comments and normalize whitespace
        normalized_query = self._normalize_query(query)

        # Step 2: Check for forbidden keywords
        self._check_forbidden_keywords(normalized_query)

        # Step 3: Check for blocking patterns (Cartesian products, etc.)
        self._check_blocking_patterns(normalized_query)

        # Step 4: Check for dangerous patterns (warnings only)
        pattern_warnings = self._check_dangerous_patterns(normalized_query)
        warnings.extend(pattern_warnings)

        # Step 5: Enforce parameterization
        if self.enforce_parameters:
            self._check_parameterization(normalized_query, params)

        # Step 6: Check variable-length path depth
        self._check_traversal_depth(normalized_query)

        # Step 7: Ensure LIMIT clause exists
        if self.enforce_limits and not self._has_limit(normalized_query):
            # Add LIMIT if missing
            limit = params.get("limit", 100)  # Default to 100
            normalized_query = normalized_query.rstrip(";") + f"\nLIMIT {limit};"
            warnings.append(f"Added LIMIT {limit} clause")

        # Step 8: EXPLAIN plan inspection (if driver available)
        if self.neo4j_driver:
            plan_analysis = self._analyze_plan(normalized_query, params)
            if not plan_analysis.is_safe:
                raise ValidationError(
                    reason="Query plan exceeds safety thresholds",
                    query=normalized_query,
                    suggestion=f"Warnings: {', '.join(plan_analysis.warnings)}",
                )
            warnings.extend(plan_analysis.warnings)

        return ValidationResult(
            valid=True, query=normalized_query, params=params, warnings=warnings
        )

    def _normalize_query(self, query: str) -> str:
        """Strip comments and normalize whitespace."""
        # Remove single-line comments
        query = re.sub(r"//.*$", "", query, flags=re.MULTILINE)
        # Remove multi-line comments
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
        # Normalize whitespace
        query = " ".join(query.split())
        return query.strip()

    def _check_forbidden_keywords(self, query: str):
        """Check for forbidden keywords that indicate dangerous operations."""
        query_upper = query.upper()

        for pattern in self.FORBIDDEN_KEYWORDS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                keyword = re.search(pattern, query_upper, re.IGNORECASE).group(0)
                raise ValidationError(
                    reason=f"Forbidden keyword detected: {keyword}",
                    query=query,
                    suggestion="This operation is not allowed for safety reasons",
                )

    def _check_blocking_patterns(self, query: str):
        """Check for blocking patterns that will raise ValidationError."""
        for pattern, message in self.BLOCKING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValidationError(
                    reason=f"Unsafe/expensive pattern detected: {message}",
                    query=query,
                    suggestion="Rewrite query to avoid Cartesian products; use explicit relationships",
                )

    def _check_dangerous_patterns(self, query: str) -> List[str]:
        """Check for dangerous query patterns and return warnings."""
        warnings = []

        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                warnings.append(f"Potentially expensive pattern: {message}")

        return warnings

    def _check_parameterization(self, query: str, params: Dict[str, Any]):
        """Ensure all user inputs are parameterized (no string/number literals in WHERE)."""
        # Find WHERE clauses
        where_clauses = re.finditer(
            r"\bWHERE\b(.*?)(?:\bRETURN\b|\bWITH\b|\bMATCH\b|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )

        for match in where_clauses:
            where_content = match.group(1)

            # Check for unparameterized string literals
            # Allow 'property' = $param, but not 'property' = 'literal'
            literal_matches = re.finditer(r"=\s*['\"]([^'\"]+)['\"]", where_content)
            for literal_match in literal_matches:
                literal_value = literal_match.group(1)
                # Check if this literal is actually a parameter value
                if literal_value not in params.values():
                    raise ValidationError(
                        reason=f"Unparameterized string literal in WHERE clause: '{literal_value}'",
                        query=query,
                        suggestion="Use $param syntax for all dynamic values",
                    )

            # Check for unparameterized numeric literals (except in patterns)
            # Allow threshold constants (0.0-1.0 for confidence/scores) and small integers
            numeric_matches = re.finditer(r"[=><!]+\s*(\d+(?:\.\d+)?)\b", where_content)
            for numeric_match in numeric_matches:
                numeric_value = numeric_match.group(1)
                num_val = float(numeric_value)

                # Allow common threshold constants (0.0 to 1.0 range) - confidence scores, etc.
                if 0.0 <= num_val <= 1.0:
                    continue

                # Allow small integers (0-99) - order, limit-like values
                if numeric_value.isdigit() and int(numeric_value) < 100:
                    continue

                # Block large numbers that look like user input (IDs, etc.)
                raise ValidationError(
                    reason=f"Unparameterized numeric literal in WHERE clause: {numeric_value}",
                    query=query,
                    suggestion="Use $param syntax for all dynamic values",
                )

    def _check_traversal_depth(self, query: str):
        """Check variable-length path depth constraints."""
        # Find all variable-length patterns: *min..max
        for match in self.VARIABLE_LENGTH_PATTERN.finditer(query):
            min_depth = int(match.group(1))
            max_depth = int(match.group(2))

            if max_depth > self.max_depth:
                raise ValidationError(
                    reason=f"Variable-length path depth {max_depth} exceeds maximum {self.max_depth}",
                    query=query,
                    suggestion=f"Reduce path depth to *{min_depth}..{self.max_depth} or lower",
                )

            if min_depth > max_depth:
                raise ValidationError(
                    reason=f"Invalid path range: min ({min_depth}) > max ({max_depth})",
                    query=query,
                    suggestion="Ensure min <= max in *min..max patterns",
                )

    def _has_limit(self, query: str) -> bool:
        """Check if query has a LIMIT clause."""
        return bool(re.search(r"\bLIMIT\b", query, re.IGNORECASE))

    def _analyze_plan(self, query: str, params: Dict[str, Any]) -> PlanAnalysis:
        """
        Run EXPLAIN on query and analyze the execution plan.
        Checks for expensive operations like NodeByLabelScan, Expand(All), etc.
        """
        if not self.neo4j_driver:
            # No driver available - return safe analysis
            return PlanAnalysis(
                estimated_rows=0,
                node_by_label_scans=0,
                expand_all_ops=0,
                max_depth=0,
                operator_tree=[],
                is_safe=True,
                warnings=[],
            )

        warnings = []

        try:
            with self.neo4j_driver.session() as session:
                # Run EXPLAIN (doesn't execute, just plans)
                explain_query = f"EXPLAIN {query}"
                result = session.run(explain_query, params)
                plan = result.consume().plan

                # Analyze plan recursively
                stats = self._analyze_plan_tree(plan)

                # Check thresholds
                is_safe = True

                if stats["node_by_label_scans"] > self.max_label_scans:
                    is_safe = False
                    warnings.append(
                        f"Too many NodeByLabelScan operations: {stats['node_by_label_scans']} > {self.max_label_scans}"
                    )

                if stats["expand_all_ops"] > self.max_expand_ops:
                    is_safe = False
                    warnings.append(
                        f"Too many Expand operations: {stats['expand_all_ops']} > {self.max_expand_ops}"
                    )

                if stats["estimated_rows"] > self.max_estimated_rows:
                    warnings.append(
                        f"High estimated rows: {stats['estimated_rows']} (may be slow)"
                    )

                return PlanAnalysis(
                    estimated_rows=stats["estimated_rows"],
                    node_by_label_scans=stats["node_by_label_scans"],
                    expand_all_ops=stats["expand_all_ops"],
                    max_depth=stats["max_depth"],
                    operator_tree=stats["operators"],
                    is_safe=is_safe,
                    warnings=warnings,
                )

        except Exception as e:
            # EXPLAIN failed - treat as potentially unsafe
            raise ValidationError(
                reason=f"Failed to analyze query plan: {str(e)}",
                query=query,
                suggestion="Query may be malformed or too complex",
            )

    def _analyze_plan_tree(self, plan_node, depth=0) -> Dict[str, Any]:
        """Recursively analyze execution plan tree."""
        stats = {
            "estimated_rows": 0,
            "node_by_label_scans": 0,
            "expand_all_ops": 0,
            "max_depth": depth,
            "operators": [],
        }

        if not plan_node:
            return stats

        # Handle both dict and object representations
        if isinstance(plan_node, dict):
            operator_name = plan_node.get("operatorType", "")
            arguments = plan_node.get("arguments", {})
            children = plan_node.get("children", [])
        else:
            operator_name = getattr(plan_node, "operator_type", "")
            arguments = getattr(plan_node, "arguments", {})
            children = getattr(plan_node, "children", [])

        stats["operators"].append(operator_name)

        # Count expensive operations
        if operator_name == "NodeByLabelScan":
            stats["node_by_label_scans"] += 1

        if operator_name in ["Expand(All)", "VarLengthExpand(All)", "VarLengthExpand"]:
            stats["expand_all_ops"] += 1

        # Get estimated rows
        if "EstimatedRows" in arguments:
            stats["estimated_rows"] = max(
                stats["estimated_rows"], int(arguments["EstimatedRows"])
            )

        # Recurse into children
        for child in children:
            child_stats = self._analyze_plan_tree(child, depth + 1)
            stats["estimated_rows"] = max(
                stats["estimated_rows"], child_stats["estimated_rows"]
            )
            stats["node_by_label_scans"] += child_stats["node_by_label_scans"]
            stats["expand_all_ops"] += child_stats["expand_all_ops"]
            stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
            stats["operators"].extend(child_stats["operators"])

        return stats


def validate_query(
    query: str, params: Dict[str, Any], neo4j_driver=None
) -> ValidationResult:
    """Convenience function to validate a query."""
    validator = CypherValidator(neo4j_driver=neo4j_driver)
    return validator.validate(query, params)

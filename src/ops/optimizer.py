"""
Query Optimizer (Phase 4, Task 4.2)
Analyzes slow queries, recommends indexes, and caches compiled plans.
See: /docs/spec.md (Query optimization)
See: /docs/pseudocode-reference.md Phase 4, Task 4.2
"""

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neo4j import Driver


@dataclass
class SlowQuery:
    """Represents a slow query with execution metadata."""

    query: str
    params: Dict[str, Any]
    duration_ms: float
    timestamp: float
    fingerprint: str


@dataclass
class ExplainPlan:
    """Parsed EXPLAIN plan with key optimization metrics."""

    query: str
    operator_types: List[str]
    label_scans: int
    index_seeks: int
    expand_all_ops: int
    estimated_rows: int
    db_hits: Optional[int] = None
    time_ms: Optional[float] = None


@dataclass
class IndexRecommendation:
    """Recommendation to create or optimize an index."""

    label: str
    properties: List[str]
    reason: str
    priority: int  # 1=high, 2=medium, 3=low
    estimated_improvement: str


@dataclass
class QueryRewrite:
    """Suggestion to rewrite a query for better performance."""

    original: str
    suggested: str
    reason: str
    estimated_improvement: str


@dataclass
class OptimizationReport:
    """Complete optimization analysis report."""

    slow_queries: List[SlowQuery] = field(default_factory=list)
    plans: List[ExplainPlan] = field(default_factory=list)
    index_recommendations: List[IndexRecommendation] = field(default_factory=list)
    query_rewrites: List[QueryRewrite] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class PlanCache:
    """Caches compiled query plans for hot templates."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.max_size = max_size

    def get_fingerprint(self, template_name: str, param_names: List[str]) -> str:
        """Generate fingerprint for template + param signature."""
        param_sig = ",".join(sorted(param_names))
        key = f"{template_name}:{param_sig}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached plan."""
        self.access_counts[fingerprint] += 1
        return self.cache.get(fingerprint)

    def put(
        self, fingerprint: str, plan: Dict[str, Any], metadata: Optional[Dict] = None
    ):
        """Store compiled plan with metadata."""
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[fingerprint] = {
            "plan": plan,
            "metadata": metadata or {},
            "cached_at": time.time(),
        }
        self.access_counts[fingerprint] = 1

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_counts:
            return
        lru_key = min(self.access_counts, key=self.access_counts.get)
        del self.cache[lru_key]
        del self.access_counts[lru_key]

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = sum(self.access_counts.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "hit_rate": (
                total_accesses / len(self.access_counts) if self.access_counts else 0.0
            ),
        }


class QueryOptimizer:
    """
    Main optimizer that analyzes queries and recommends improvements.
    Provides slow-query analysis, index recommendations, and plan caching.
    """

    def __init__(self, driver: Driver, config: Optional[Dict] = None):
        self.driver = driver
        self.config = config or {}
        self.plan_cache = PlanCache(max_size=self.config.get("plan_cache_size", 1000))
        self.slow_query_threshold_ms = self.config.get("slow_query_threshold_ms", 100)
        self.slow_queries: List[SlowQuery] = []

    def record_query(self, query: str, params: Dict[str, Any], duration_ms: float):
        """Record a query execution for analysis."""
        if duration_ms >= self.slow_query_threshold_ms:
            fingerprint = self._compute_query_fingerprint(query)
            slow_query = SlowQuery(
                query=query,
                params=params,
                duration_ms=duration_ms,
                timestamp=time.time(),
                fingerprint=fingerprint,
            )
            self.slow_queries.append(slow_query)

    def _compute_query_fingerprint(self, query: str) -> str:
        """Compute fingerprint for query pattern (normalized)."""
        # Normalize: strip comments, collapse whitespace
        normalized = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def analyze_slow_queries(self, top_n: int = 50) -> OptimizationReport:
        """
        Analyze slow queries and generate optimization report.
        Runs EXPLAIN on each slow query and extracts recommendations.
        """
        report = OptimizationReport()

        # Get top N slowest queries by duration
        sorted_queries = sorted(
            self.slow_queries, key=lambda q: q.duration_ms, reverse=True
        )
        top_queries = sorted_queries[:top_n]

        report.slow_queries = top_queries

        # Analyze each query
        seen_patterns = set()
        for slow_query in top_queries:
            # Skip duplicates (same fingerprint)
            if slow_query.fingerprint in seen_patterns:
                continue
            seen_patterns.add(slow_query.fingerprint)

            try:
                plan = self._explain_query(slow_query.query, slow_query.params)
                report.plans.append(plan)

                # Generate recommendations
                recommendations = self._generate_recommendations(plan)
                report.index_recommendations.extend(recommendations)

                # Check for rewrite opportunities
                rewrites = self._suggest_rewrites(slow_query.query, plan)
                report.query_rewrites.extend(rewrites)

            except Exception as e:
                # Log error but continue analysis
                print(f"Error analyzing query {slow_query.fingerprint}: {e}")
                continue

        # Deduplicate recommendations by (label, properties)
        report.index_recommendations = self._dedupe_recommendations(
            report.index_recommendations
        )

        # Generate summary
        report.summary = self._generate_summary(report)

        return report

    def _explain_query(self, query: str, params: Dict[str, Any]) -> ExplainPlan:
        """Run EXPLAIN on a query and parse the plan."""
        with self.driver.session() as session:
            result = session.run(f"EXPLAIN {query}", params)
            summary = result.consume()
            plan = summary.plan

            if plan is None:
                # No plan available (shouldn't happen but handle gracefully)
                return ExplainPlan(
                    query=query,
                    operator_types=[],
                    label_scans=0,
                    index_seeks=0,
                    expand_all_ops=0,
                    estimated_rows=0,
                )

            # Flatten plan tree to extract operators
            operators = self._flatten_plan(plan)

            # Count specific operator types
            label_scans = sum(
                1 for op in operators if "NodeByLabelScan" in op or "AllNodesScan" in op
            )
            index_seeks = sum(
                1 for op in operators if "NodeIndexSeek" in op or "IndexSeek" in op
            )
            expand_all = sum(1 for op in operators if "Expand(All)" in op)

            # Estimate rows - handle both dict and object representations
            if isinstance(plan, dict):
                args = plan.get("args", {})
                estimated_rows = args.get("EstimatedRows", 0) or 0
            else:
                estimated_rows = (
                    getattr(plan, "rows", None)
                    or getattr(plan, "estimated_rows", 0)
                    or 0
                )

            return ExplainPlan(
                query=query,
                operator_types=operators,
                label_scans=label_scans,
                index_seeks=index_seeks,
                expand_all_ops=expand_all,
                estimated_rows=estimated_rows,
            )

    def _flatten_plan(self, plan) -> List[str]:
        """
        Recursively flatten EXPLAIN plan tree into operator list.
        Neo4j plans can be either dict or object depending on driver version.
        """
        if plan is None:
            return []

        operators = []
        stack = [plan]

        while stack:
            node = stack.pop()
            if node is None:
                continue

            # Handle both dict and object representations
            if isinstance(node, dict):
                op_type = node.get("operatorType", "")
                if op_type:
                    operators.append(op_type)
                children = node.get("children", [])
            else:
                # Object with attributes
                op_type = getattr(node, "operator_type", None) or getattr(
                    node, "operatorType", None
                )
                if op_type:
                    operators.append(op_type)
                children = getattr(node, "children", None) or []

            # Add children to stack
            if children:
                for child in children:
                    if child is not None:
                        stack.append(child)

        return operators

    def _generate_recommendations(self, plan: ExplainPlan) -> List[IndexRecommendation]:
        """Generate index recommendations from EXPLAIN plan."""
        recommendations = []

        # Check for label scans - suggest indexes
        if plan.label_scans > 0:
            # Parse query to find labels and properties
            label_props = self._extract_label_property_patterns(plan.query)

            for label, properties in label_props.items():
                if properties:
                    rec = IndexRecommendation(
                        label=label,
                        properties=properties,
                        reason=f"Query performs {plan.label_scans} label scan(s); index would enable seek",
                        priority=1,
                        estimated_improvement="50-90% reduction in query time",
                    )
                    recommendations.append(rec)

        # Check for high estimated rows without index
        if plan.estimated_rows > 1000 and plan.index_seeks == 0:
            rec = IndexRecommendation(
                label="<inferred from query>",
                properties=["<property from WHERE clause>"],
                reason=f"Query estimates {plan.estimated_rows} rows without index",
                priority=2,
                estimated_improvement="30-70% reduction in query time",
            )
            recommendations.append(rec)

        return recommendations

    def _extract_label_property_patterns(self, query: str) -> Dict[str, List[str]]:
        """
        Extract label:property patterns from query.
        Returns dict of label -> [properties used in WHERE/MATCH].
        """
        patterns = defaultdict(list)

        # Pattern 1: (n:Label {prop: $value})
        inline_matches = re.finditer(
            r"\((\w+):(\w+)\s*\{([^}]+)\}\)", query, re.IGNORECASE
        )
        for match in inline_matches:
            label = match.group(2)
            props_str = match.group(3)
            # Extract property names (before colons)
            props = re.findall(r"(\w+)\s*:", props_str)
            patterns[label].extend(props)

        # Pattern 2: MATCH (n:Label) WHERE n.prop = $value
        where_matches = re.finditer(
            r"MATCH\s+\(\w+:(\w+)\).*?WHERE\s+\w+\.(\w+)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        for match in where_matches:
            label = match.group(1)
            prop = match.group(2)
            patterns[label].append(prop)

        # Deduplicate properties per label
        return {label: list(set(props)) for label, props in patterns.items()}

    def _suggest_rewrites(self, query: str, plan: ExplainPlan) -> List[QueryRewrite]:
        """Suggest query rewrites for performance improvement."""
        rewrites = []

        # Check for unbounded variable-length patterns
        if re.search(r"\*\.\.", query):
            suggested = re.sub(r"\*\.\.(\d+)", r"*1..$max_depth", query)
            rewrites.append(
                QueryRewrite(
                    original=query,
                    suggested=suggested,
                    reason="Unbounded variable-length pattern can cause performance issues",
                    estimated_improvement="Prevents graph explosion; 10-50% faster",
                )
            )

        # Check for Cartesian products (multiple MATCH without connection)
        match_count = len(re.findall(r"\bMATCH\b", query, re.IGNORECASE))
        if match_count > 1 and "Expand(All)" in plan.operator_types:
            rewrites.append(
                QueryRewrite(
                    original=query,
                    suggested="<combine MATCH clauses or add WHERE connection>",
                    reason="Multiple MATCH clauses may create Cartesian product",
                    estimated_improvement="Can reduce rows by orders of magnitude",
                )
            )

        # Check for missing LIMIT
        if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
            suggested = query.rstrip(";") + "\nLIMIT 100;"
            rewrites.append(
                QueryRewrite(
                    original=query,
                    suggested=suggested,
                    reason="No LIMIT clause; may return excessive rows",
                    estimated_improvement="Reduces result set size and memory usage",
                )
            )

        return rewrites

    def _dedupe_recommendations(
        self, recommendations: List[IndexRecommendation]
    ) -> List[IndexRecommendation]:
        """Deduplicate recommendations by (label, properties)."""
        seen = set()
        deduped = []

        for rec in recommendations:
            key = (rec.label, tuple(sorted(rec.properties)))
            if key not in seen:
                seen.add(key)
                deduped.append(rec)

        return sorted(deduped, key=lambda r: r.priority)

    def _generate_summary(self, report: OptimizationReport) -> Dict[str, Any]:
        """Generate summary statistics for optimization report."""
        return {
            "total_slow_queries": len(report.slow_queries),
            "unique_patterns": len(report.plans),
            "index_recommendations": len(report.index_recommendations),
            "query_rewrites": len(report.query_rewrites),
            "avg_duration_ms": (
                sum(q.duration_ms for q in report.slow_queries)
                / len(report.slow_queries)
                if report.slow_queries
                else 0.0
            ),
            "total_label_scans": sum(p.label_scans for p in report.plans),
            "total_index_seeks": sum(p.index_seeks for p in report.plans),
        }

    def get_cached_plan(
        self, template_name: str, param_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached compiled plan for template."""
        fingerprint = self.plan_cache.get_fingerprint(template_name, param_names)
        return self.plan_cache.get(fingerprint)

    def cache_plan(
        self,
        template_name: str,
        param_names: List[str],
        plan: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ):
        """Cache compiled plan for template."""
        fingerprint = self.plan_cache.get_fingerprint(template_name, param_names)
        self.plan_cache.put(fingerprint, plan, metadata)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get plan cache statistics."""
        return self.plan_cache.stats()

    def clear_slow_queries(self):
        """Clear recorded slow queries (for testing or periodic cleanup)."""
        self.slow_queries.clear()

    def export_recommendations_as_cypher(
        self, recommendations: List[IndexRecommendation]
    ) -> str:
        """Export index recommendations as executable Cypher statements."""
        statements = []

        for rec in recommendations:
            label = rec.label
            props = rec.properties

            if len(props) == 1:
                # Single property index
                stmt = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{props[0]});"
            else:
                # Composite index
                props_str = ", ".join(f"n.{p}" for p in props)
                stmt = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON ({props_str});"

            statements.append(f"-- Priority {rec.priority}: {rec.reason}")
            statements.append(stmt)
            statements.append("")

        return "\n".join(statements)


# Utility function for integration with query execution pipeline
def create_optimizer(driver: Driver, config: Optional[Dict] = None) -> QueryOptimizer:
    """Factory function to create optimizer with config."""
    return QueryOptimizer(driver, config)

#!/usr/bin/env python3
"""
GDS Readiness Validation Suite

Validates that the Neo4j graph meets all prerequisites for running
Graph Data Science (GDS) algorithms (community detection, centrality,
similarity projections) on RELATED_TO edges.

Usage:
    python scripts/validate_gds_readiness.py [--neo4j-uri bolt://localhost:7687]
    python scripts/validate_gds_readiness.py --metrics-only

Exit codes:
    0 = All gates pass
    1 = One or more gates failed

Prerequisites:
    - Neo4j running with RELATED_TO v2 edges
    - Guard DDL applied (scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher)
    - Full backfill completed with v2 edge model
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA_VERSION = "v4.1"
REQUIRED_INDEXES = [
    "related_to_score_final_idx",
    "related_to_method_idx",
    "related_to_quality_tier_idx",
    "related_to_is_mutual_idx",
]


@dataclass
class GateResult:
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


def _run_gate(
    session: Any,
    name: str,
    query: str,
    check: Callable[[Any], bool],
    message: str,
    params: Optional[Dict[str, Any]] = None,
) -> GateResult:
    """Execute a gate query and evaluate its result."""
    try:
        result = session.run(query, **(params or {}))
        record = result.single()
        if record is None:
            return GateResult(
                name=name,
                passed=False,
                message=f"FAIL: {message} (no result returned)",
            )
        record_dict = dict(record)
        passed = check(record_dict)
        return GateResult(
            name=name,
            passed=passed,
            message=f"{'PASS' if passed else 'FAIL'}: {message}",
            details=record_dict,
        )
    except Exception as exc:
        return GateResult(
            name=name,
            passed=False,
            message=f"ERROR: {message} ({type(exc).__name__}: {exc})",
        )


def run_all_gates(neo4j_driver: Any) -> List[GateResult]:
    """Run all 8 GDS readiness gate checks."""
    results: List[GateResult] = []

    with neo4j_driver.session() as session:
        # Gate 1: RELATED_TO edges exist
        results.append(
            _run_gate(
                session,
                "related_to_edges_exist",
                "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS edge_count",
                lambda r: r.get("edge_count", 0) > 0,
                "RELATED_TO edges must exist for GDS projection",
            )
        )

        # Gate 2: score_final populated on all edges
        results.append(
            _run_gate(
                session,
                "score_final_populated",
                """
                MATCH ()-[r:RELATED_TO]->()
                WITH count(r) AS total, count(r.score_final) AS with_sf
                RETURN total, with_sf, total - with_sf AS missing
                """,
                lambda r: r.get("missing", 1) == 0,
                "All RELATED_TO edges must have score_final for weighted GDS",
            )
        )

        # Gate 3: method_version is v2.0
        results.append(
            _run_gate(
                session,
                "method_version_v2",
                """
                MATCH ()-[r:RELATED_TO]->()
                WHERE r.method_version IS NULL OR r.method_version <> '2.0'
                RETURN count(r) AS legacy_count
                """,
                lambda r: r.get("legacy_count", 1) == 0,
                "All edges must be v2.0 (run backfill to upgrade legacy edges)",
            )
        )

        # Gate 4: reciprocity computed
        results.append(
            _run_gate(
                session,
                "reciprocity_computed",
                """
                MATCH ()-[r:RELATED_TO]->()
                WHERE r.is_mutual IS NULL
                RETURN count(r) AS unreconciled
                """,
                lambda r: r.get("unreconciled", 1) == 0,
                "All edges must have is_mutual set (run reciprocity reconciliation)",
            )
        )

        # Gate 5: indexes exist
        results.append(
            _run_gate(
                session,
                "indexes_exist",
                "SHOW INDEXES YIELD name RETURN collect(name) AS names",
                lambda r: all(
                    idx in (r.get("names") or []) for idx in REQUIRED_INDEXES
                ),
                f"All {len(REQUIRED_INDEXES)} RELATED_TO relationship indexes must exist",
            )
        )

        # Gate 6: quality tiers populated
        results.append(
            _run_gate(
                session,
                "quality_tiers_populated",
                """
                MATCH ()-[r:RELATED_TO]->()
                WHERE r.quality_tier IS NULL
                RETURN count(r) AS no_tier
                """,
                lambda r: r.get("no_tier", 1) == 0,
                "All edges must have quality_tier for filtered GDS projections",
            )
        )

        # Gate 7: schema version
        results.append(
            _run_gate(
                session,
                "schema_version",
                """
                MATCH (sv:SchemaVersion {id: 'singleton'})
                RETURN sv.version AS version
                """,
                lambda r: r.get("version") == REQUIRED_SCHEMA_VERSION,
                f"SchemaVersion must be {REQUIRED_SCHEMA_VERSION}",
            )
        )

        # Gate 8: marker includes RELATED_TO
        results.append(
            _run_gate(
                session,
                "marker_includes_related_to",
                """
                MATCH (m:RelationshipTypesMarker {id: 'chunk_rel_types_v1'})
                RETURN m.types AS types
                """,
                lambda r: "RELATED_TO" in (r.get("types") or []),
                "RelationshipTypesMarker must include RELATED_TO",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Metrics (informational, not pass/fail)
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    name: str
    value: Any
    description: str


def run_metrics(neo4j_driver: Any) -> List[MetricResult]:
    """Collect GDS-relevant metrics for monitoring and drift detection."""
    metrics: List[MetricResult] = []

    with neo4j_driver.session() as session:
        # Edge count
        try:
            record = session.run(
                "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS total"
            ).single()
            if record:
                metrics.append(
                    MetricResult(
                        "edge_count", record["total"], "Total RELATED_TO edges"
                    )
                )
        except Exception:
            pass

        # Score distribution
        try:
            record = session.run(
                """
                MATCH ()-[r:RELATED_TO]->()
                WHERE r.score_final IS NOT NULL
                RETURN
                    percentileCont(r.score_final, 0.50) AS p50,
                    percentileCont(r.score_final, 0.90) AS p90,
                    percentileCont(r.score_final, 0.99) AS p99,
                    avg(r.score_final) AS mean
                """
            ).single()
            if record:
                metrics.append(
                    MetricResult(
                        "score_distribution",
                        {
                            "p50": round(record["p50"] or 0, 6),
                            "p90": round(record["p90"] or 0, 6),
                            "p99": round(record["p99"] or 0, 6),
                            "mean": round(record["mean"] or 0, 6),
                        },
                        "score_final distribution",
                    )
                )
        except Exception:
            pass

        # Reciprocity ratio
        try:
            record = session.run(
                """
                MATCH ()-[r:RELATED_TO]->()
                WHERE r.is_mutual IS NOT NULL
                RETURN avg(CASE WHEN r.is_mutual THEN 1.0 ELSE 0.0 END) AS ratio
                """
            ).single()
            if record and record["ratio"] is not None:
                metrics.append(
                    MetricResult(
                        "reciprocity_ratio",
                        round(record["ratio"], 4),
                        "Fraction of edges that are mutual",
                    )
                )
        except Exception:
            pass

        # Quality tier distribution
        try:
            result = session.run(
                """
                MATCH ()-[r:RELATED_TO]->()
                RETURN r.quality_tier AS tier, count(*) AS cnt
                ORDER BY cnt DESC
                """
            )
            tiers = {str(record["tier"]): record["cnt"] for record in result}
            if tiers:
                metrics.append(
                    MetricResult(
                        "quality_tier_distribution",
                        tiers,
                        "Edge count by quality tier",
                    )
                )
        except Exception:
            pass

        # Prior coverage
        try:
            record = session.run(
                """
                MATCH ()-[r:RELATED_TO]->()
                RETURN
                    count(r) AS total,
                    count(r.prior_reference) AS with_ref,
                    count(r.prior_entity) AS with_ent,
                    count(r.prior_taxonomy) AS with_tax
                """
            ).single()
            if record:
                total = record["total"] or 1
                metrics.append(
                    MetricResult(
                        "prior_coverage",
                        {
                            "total": total,
                            "prior_reference": record["with_ref"],
                            "prior_entity": record["with_ent"],
                            "prior_taxonomy": record["with_tax"],
                            "ref_pct": round(
                                100 * (record["with_ref"] or 0) / total, 1
                            ),
                            "ent_pct": round(
                                100 * (record["with_ent"] or 0) / total, 1
                            ),
                            "tax_pct": round(
                                100 * (record["with_tax"] or 0) / total, 1
                            ),
                        },
                        "Structural prior field coverage",
                    )
                )
        except Exception:
            pass

        # Document degree distribution
        try:
            record = session.run(
                """
                MATCH (d:Document)-[r:RELATED_TO]->()
                WITH d, count(r) AS out_degree
                RETURN
                    avg(out_degree) AS avg_degree,
                    max(out_degree) AS max_degree,
                    percentileCont(out_degree, 0.90) AS p90_degree,
                    count(d) AS doc_count
                """
            ).single()
            if record:
                metrics.append(
                    MetricResult(
                        "degree_distribution",
                        {
                            "avg": round(record["avg_degree"] or 0, 2),
                            "max": record["max_degree"],
                            "p90": record["p90_degree"],
                            "docs_with_edges": record["doc_count"],
                        },
                        "Outgoing RELATED_TO degree per document",
                    )
                )
        except Exception:
            pass

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_report(gates: List[GateResult], metrics: List[MetricResult]) -> str:
    """Format the validation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("GDS READINESS VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Gates
    passed = sum(1 for g in gates if g.passed)
    total = len(gates)
    lines.append(f"GATES: {passed}/{total} passed")
    lines.append("-" * 40)
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        lines.append(f"  [{status}] {g.name}")
        lines.append(f"         {g.message}")
        if g.details and not g.passed:
            for k, v in g.details.items():
                lines.append(f"         {k}: {v}")
    lines.append("")

    # Metrics
    if metrics:
        lines.append("METRICS (informational)")
        lines.append("-" * 40)
        for m in metrics:
            if isinstance(m.value, dict):
                lines.append(f"  {m.name}: {m.description}")
                for k, v in m.value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {m.name}: {m.value} ({m.description})")
        lines.append("")

    # Summary
    lines.append("=" * 60)
    if passed == total:
        lines.append("RESULT: ALL GATES PASSED — graph is GDS-ready")
    else:
        failed_names = [g.name for g in gates if not g.passed]
        lines.append(f"RESULT: {total - passed} GATE(S) FAILED")
        lines.append(f"  Failed: {', '.join(failed_names)}")
    lines.append("=" * 60)

    return "\n".join(lines)


def main() -> int:
    """Run GDS readiness validation. Returns 0 on pass, 1 on failure."""
    parser = argparse.ArgumentParser(
        description="Validate Neo4j graph for GDS readiness"
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j Bolt URI (default: $NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip gates, only collect metrics",
    )
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            args.neo4j_uri,
            auth=(args.neo4j_user, args.neo4j_password),
        )
        driver.verify_connectivity()
    except ImportError:
        print("ERROR: neo4j Python driver not installed (pip install neo4j)")
        return 1
    except Exception as exc:
        print(f"ERROR: Cannot connect to Neo4j at {args.neo4j_uri}: {exc}")
        return 1

    try:
        gates: List[GateResult] = []
        if not args.metrics_only:
            gates = run_all_gates(driver)

        metrics = run_metrics(driver)

        report = format_report(gates, metrics)
        print(report)

        if args.metrics_only:
            return 0
        return 0 if all(g.passed for g in gates) else 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())

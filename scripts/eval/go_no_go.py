#!/usr/bin/env python3
"""
Phase 7E-4: Go/No-Go Evaluation
Deployment decision based on A/B test results and SLO compliance

Reference: Canonical Spec L3701, L5091
Criteria:
- Hit@3 improvement ≥ +10-15%
- Retrieval p95 ≤ 1.3× baseline
- All ZERO-tolerance SLOs met (oversized chunks, integrity failures)
- All range-based SLOs met (expansion rate 10-40%)

Usage:
    python scripts/eval/go_no_go.py --ab-report fusion_ab_report.json --slo-metrics slo_metrics.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Color:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


class GoNoGoEvaluator:
    """
    Deployment decision maker based on comprehensive criteria.

    Criteria:
    1. Quality: Hit@3 improvement ≥ +10-15%
    2. Performance: Retrieval p95 ≤ 1.3× baseline
    3. Safety: ZERO oversized chunks and integrity failures
    4. Guardrails: Expansion rate 10-40%
    """

    # Thresholds from canonical spec
    MIN_HIT3_IMPROVEMENT = 0.10  # 10% minimum improvement
    MAX_LATENCY_RATIO = 1.3  # Max 1.3× baseline p95
    EXPANSION_MIN = 0.10  # 10% minimum
    EXPANSION_MAX = 0.40  # 40% maximum

    def __init__(self):
        """Initialize evaluator."""
        self.checks_passed: List[str] = []
        self.checks_failed: List[str] = []
        self.warnings: List[str] = []

    def evaluate(
        self, ab_report: Dict[str, any], slo_metrics: Optional[Dict[str, float]] = None
    ) -> tuple[str, str, Dict[str, any]]:
        """
        Evaluate go/no-go decision.

        Args:
            ab_report: A/B test report from fusion_ab.py
            slo_metrics: Optional SLO metrics from production

        Returns:
            Tuple of (decision, rationale, details)
            - decision: "GO", "NO-GO", or "CONDITIONAL-GO"
            - rationale: Human-readable explanation
            - details: Dictionary with detailed check results
        """
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

        # 1. Check quality improvement
        quality_ok = self._check_quality_improvement(ab_report)

        # 2. Check performance
        performance_ok = self._check_performance(ab_report)

        # 3. Check SLOs (if provided)
        slos_ok = True
        if slo_metrics:
            slos_ok = self._check_slos(slo_metrics)

        # Make decision
        if quality_ok and performance_ok and slos_ok:
            decision = "GO"
            rationale = "✅ All criteria met: quality improvement, performance within bounds, SLOs satisfied"
        elif not slos_ok:
            decision = "NO-GO"
            rationale = (
                "❌ CRITICAL: SLO violations detected (ZERO-tolerance checks failed)"
            )
        elif not quality_ok and not performance_ok:
            decision = "NO-GO"
            rationale = "❌ Both quality and performance criteria failed"
        elif not quality_ok:
            decision = "NO-GO"
            rationale = "❌ Quality improvement below 10% threshold"
        elif not performance_ok:
            decision = "NO-GO"
            rationale = "❌ Performance degradation exceeds 1.3× baseline"
        else:
            decision = "CONDITIONAL-GO"
            rationale = "⚠️ Partial criteria met - manual review required"

        details = {
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "quality_ok": quality_ok,
            "performance_ok": performance_ok,
            "slos_ok": slos_ok,
        }

        return decision, rationale, details

    def _check_quality_improvement(self, ab_report: Dict[str, any]) -> bool:
        """Check if quality improvement meets threshold."""
        comparison = ab_report.get("summary", {}).get("comparison", {})

        if "hit_at_3_improvement" not in comparison:
            self.warnings.append(
                "No Hit@3 improvement metric available (no golden set)"
            )
            return True  # Don't fail if metric unavailable

        hit3_improvement = comparison["hit_at_3_improvement"]
        hit3_improvement_pct = hit3_improvement * 100

        if hit3_improvement >= self.MIN_HIT3_IMPROVEMENT:
            self.checks_passed.append(
                f"✅ Hit@3 improvement: {hit3_improvement_pct:+.1f}% (≥{self.MIN_HIT3_IMPROVEMENT*100:.0f}% required)"
            )
            return True
        else:
            self.checks_failed.append(
                f"❌ Hit@3 improvement: {hit3_improvement_pct:+.1f}% (below {self.MIN_HIT3_IMPROVEMENT*100:.0f}% threshold)"
            )
            return False

    def _check_performance(self, ab_report: Dict[str, any]) -> bool:
        """Check if performance degradation is acceptable."""
        comparison = ab_report.get("summary", {}).get("comparison", {})
        latency_ratio = comparison.get("latency_ratio", 0.0)

        if latency_ratio <= self.MAX_LATENCY_RATIO:
            self.checks_passed.append(
                f"✅ p95 latency ratio: {latency_ratio:.2f}× (≤{self.MAX_LATENCY_RATIO}× required)"
            )
            return True
        else:
            self.checks_failed.append(
                f"❌ p95 latency ratio: {latency_ratio:.2f}× (exceeds {self.MAX_LATENCY_RATIO}× limit)"
            )
            return False

    def _check_slos(self, slo_metrics: Dict[str, float]) -> bool:
        """Check SLO compliance."""
        all_ok = True

        # ZERO-tolerance SLOs
        oversized_rate = slo_metrics.get("oversized_chunk_rate", 0.0)
        if oversized_rate == 0.0:
            self.checks_passed.append(
                "✅ Oversized chunk rate: 0.0 (ZERO-tolerance met)"
            )
        else:
            self.checks_failed.append(
                f"❌ CRITICAL: Oversized chunk rate: {oversized_rate:.6f} (ZERO-tolerance violation)"
            )
            all_ok = False

        integrity_failure_rate = slo_metrics.get("integrity_failure_rate", 0.0)
        if integrity_failure_rate == 0.0:
            self.checks_passed.append(
                "✅ Integrity failure rate: 0.0 (ZERO-tolerance met)"
            )
        else:
            self.checks_failed.append(
                f"❌ CRITICAL: Integrity failure rate: {integrity_failure_rate:.6f} (ZERO-tolerance violation)"
            )
            all_ok = False

        # Range-based SLOs
        expansion_rate = slo_metrics.get("expansion_rate")
        if expansion_rate is not None:
            if self.EXPANSION_MIN <= expansion_rate <= self.EXPANSION_MAX:
                self.checks_passed.append(
                    f"✅ Expansion rate: {expansion_rate:.1%} ({self.EXPANSION_MIN:.0%}-{self.EXPANSION_MAX:.0%} required)"
                )
            else:
                self.checks_failed.append(
                    f"❌ Expansion rate: {expansion_rate:.1%} (outside {self.EXPANSION_MIN:.0%}-{self.EXPANSION_MAX:.0%} range)"
                )
                all_ok = False

        # Target-based SLOs
        retrieval_p95 = slo_metrics.get("retrieval_p95_latency")
        if retrieval_p95 is not None:
            if retrieval_p95 <= 500.0:
                self.checks_passed.append(
                    f"✅ Retrieval p95 latency: {retrieval_p95:.1f}ms (≤500ms required)"
                )
            else:
                self.warnings.append(
                    f"⚠️ Retrieval p95 latency: {retrieval_p95:.1f}ms (exceeds 500ms target)"
                )

        return all_ok


def load_json_file(file_path: Path) -> Dict[str, any]:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_report(
    decision: str,
    rationale: str,
    details: Dict[str, any],
    ab_report: Dict[str, any],
    slo_metrics: Optional[Dict[str, float]],
    output_file: Path,
):
    """Save go/no-go report to JSON."""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision": decision,
        "rationale": rationale,
        "details": details,
        "ab_test_summary": ab_report.get("summary", {}),
        "slo_metrics": slo_metrics,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{Color.GREEN}Saved report to {output_file}{Color.END}")


def print_report(
    decision: str, rationale: str, details: Dict[str, any], ab_report: Dict[str, any]
):
    """Print go/no-go report to console."""
    print("\n" + "=" * 80)
    print(f"{Color.BOLD}Phase 7E-4: Go/No-Go Evaluation{Color.END}")
    print("=" * 80)

    # Checks passed
    if details["checks_passed"]:
        print(f"\n{Color.GREEN}{Color.BOLD}Checks Passed:{Color.END}")
        for check in details["checks_passed"]:
            print(f"  {check}")

    # Warnings
    if details["warnings"]:
        print(f"\n{Color.YELLOW}{Color.BOLD}Warnings:{Color.END}")
        for warning in details["warnings"]:
            print(f"  {warning}")

    # Checks failed
    if details["checks_failed"]:
        print(f"\n{Color.RED}{Color.BOLD}Checks Failed:{Color.END}")
        for check in details["checks_failed"]:
            print(f"  {check}")

    # Decision
    print("\n" + "=" * 80)
    if decision == "GO":
        color = Color.GREEN
    elif decision == "NO-GO":
        color = Color.RED
    else:
        color = Color.YELLOW

    print(f"{color}{Color.BOLD}Decision: {decision}{Color.END}")
    print(f"{rationale}")
    print("=" * 80)

    # Summary from A/B test
    print(f"\n{Color.BOLD}A/B Test Summary:{Color.END}")
    summary = ab_report.get("summary", {})

    if "rrf" in summary:
        rrf = summary["rrf"]
        print(
            f"  RRF: p95={rrf['latency']['p95']:.1f}ms, expansion={rrf['expansion_rate']:.1%}"
        )

    if "weighted" in summary:
        weighted = summary["weighted"]
        print(
            f"  Weighted: p95={weighted['latency']['p95']:.1f}ms, expansion={weighted['expansion_rate']:.1%}"
        )

    if "comparison" in summary:
        comp = summary["comparison"]
        print(f"  Latency ratio: {comp['latency_ratio']:.2f}×")
        if "hit_at_3_improvement" in comp:
            print(f"  Hit@3 improvement: {comp['hit_at_3_improvement']*100:+.1f}%")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Go/No-Go evaluation for deployment")
    parser.add_argument(
        "--ab-report",
        type=Path,
        required=True,
        help="Path to A/B test report JSON",
    )
    parser.add_argument(
        "--slo-metrics",
        type=Path,
        help="Path to SLO metrics JSON (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("go_no_go_report.json"),
        help="Output report path",
    )

    args = parser.parse_args()

    # Load A/B test report
    ab_report = load_json_file(args.ab_report)
    print(f"Loaded A/B test report from {args.ab_report}")

    # Load SLO metrics if provided
    slo_metrics = None
    if args.slo_metrics:
        slo_metrics = load_json_file(args.slo_metrics)
        print(f"Loaded SLO metrics from {args.slo_metrics}")

    # Evaluate
    evaluator = GoNoGoEvaluator()
    decision, rationale, details = evaluator.evaluate(ab_report, slo_metrics)

    # Print and save report
    print_report(decision, rationale, details, ab_report)
    save_report(decision, rationale, details, ab_report, slo_metrics, args.output)

    # Exit with status code based on decision
    if decision == "GO":
        sys.exit(0)
    elif decision == "CONDITIONAL-GO":
        sys.exit(2)
    else:  # NO-GO
        sys.exit(1)


if __name__ == "__main__":
    main()

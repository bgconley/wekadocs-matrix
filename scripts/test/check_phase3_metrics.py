#!/usr/bin/env python3
"""Ensure critical Phase 3 metrics (vector parity & idempotence) remain green.

Reads the summary.json emitted by scripts/test/run_phase.sh and verifies that
key gatekeeper tests passed. If any required test is missing or not passed the
script exits with status 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_TESTS = {
    ("tests.p3_t3_integration_test.TestGraphConstruction", "test_vector_parity"),
    (
        "tests.p3_t3_integration_test.TestGraphConstruction",
        "test_idempotent_graph_build",
    ),
    ("tests.p3_t3_test.TestGraphConstruction", "test_vector_parity_with_graph"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Phase 3 parity/idempotence gates"
    )
    parser.add_argument(
        "summary",
        nargs="?",
        default="reports/phase-3/summary.json",
        help="Path to phase-3 summary.json",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}", file=sys.stderr)
        return 2

    data = json.loads(summary_path.read_text())
    cases = data.get("cases") or data.get("test_details") or []

    lookup = {}
    for case in cases:
        key = (case.get("classname"), case.get("name"))
        lookup[key] = case.get("status", "unknown")

    missing = []
    failing = []
    for key in REQUIRED_TESTS:
        status = lookup.get(key)
        if status is None:
            missing.append(key)
        elif status != "passed":
            failing.append((key, status))

    if missing or failing:
        if missing:
            print("Missing required Phase 3 gate tests:")
            for classname, name in missing:
                print(f"  - {classname}.{name}")
        if failing:
            print("Phase 3 gate tests not passing:")
            for (classname, name), status in failing:
                print(f"  - {classname}.{name}: {status}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

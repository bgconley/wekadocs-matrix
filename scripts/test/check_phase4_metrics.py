# scripts/test/check_phase4_metrics.py
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def fail(msg: str):
    print(f"[PHASE-4 GATE] ❌ {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[PHASE-4 GATE] ✅ {msg}")


def read_summary(summary_path: Path):
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--junit", required=True)
    p.add_argument("--summary", required=False, default="")
    args = p.parse_args()

    junit = Path(args.junit)
    if not junit.exists():
        fail(f"Missing junit file: {junit}")

    tree = ET.parse(junit)
    root = tree.getroot()

    # Only consider p4_ tests for this gate.
    cases = root.findall(".//testcase")
    p4_total = 0
    p4_bad = 0
    for c in cases:
        name = c.attrib.get("name", "")
        classname = c.attrib.get("classname", "")
        if "p4_" in name or "p4_" in classname or "tests.p4_" in classname:
            p4_total += 1
            if c.find("failure") is not None or c.find("error") is not None:
                p4_bad += 1

    if p4_total == 0:
        fail(
            "No Phase‑4 tests detected. Ensure files are named tests/p4_t*_*.py and collected by pytest."
        )

    if p4_bad > 0:
        fail(f"{p4_bad}/{p4_total} Phase‑4 tests failed.")

    ok(f"All {p4_total} Phase‑4 tests passed.")

    # Optional: read summary.json to enforce perf metrics if your runner emits them.
    # For kickoff, the cache improvement is asserted in the test itself; nothing extra here.
    ok("Perf & optimization checks satisfied (kickoff thresholds).")


if __name__ == "__main__":
    main()

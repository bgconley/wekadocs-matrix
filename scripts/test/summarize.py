"""Summarise pytest junit output into a compact JSON artifact."""

import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _git_commit_sha() -> str:
    try:
        if not os.path.isdir(".git"):
            return "unknown"
        short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"]
            )  # pragma: no cover - shell
            .decode()
            .strip()
        )
        return f"{short}-commit"
    except Exception:
        return "unknown"


def _summarise_junit(junit_path: Path) -> dict:
    if not junit_path.exists():
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration_seconds": 0.0,
            "cases": [],
        }

    tree = ET.parse(junit_path)
    root = tree.getroot()
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "duration_seconds": 0.0,
        "cases": [],
    }

    for suite in root.findall("testsuite"):
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        time_taken = float(suite.get("time", 0.0))

        summary["total"] += tests
        summary["failed"] += failures
        summary["errors"] += errors
        summary["skipped"] += skipped
        summary["duration_seconds"] += time_taken

        passed = tests - failures - errors - skipped
        summary["passed"] += max(passed, 0)

        for case in suite.findall("testcase"):
            status = "passed"
            message = ""
            if case.find("failure") is not None:
                status = "failed"
                message = case.find("failure").get("message", "")
            elif case.find("error") is not None:
                status = "error"
                message = case.find("error").get("message", "")
            elif case.find("skipped") is not None:
                status = "skipped"
                message = case.find("skipped").get("message", "")

            summary["cases"].append(
                {
                    "name": case.get("name"),
                    "classname": case.get("classname"),
                    "status": status,
                    "duration": float(case.get("time", 0.0)),
                    "message": message,
                }
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    junit_path = Path(args.junit)
    results = _summarise_junit(junit_path)

    summary = {
        "phase": args.phase,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_commit_sha(),
        "totals": {
            "total": results["total"],
            "passed": results["passed"],
            "failed": results["failed"],
            "errors": results["errors"],
            "skipped": results["skipped"],
            "duration_seconds": results["duration_seconds"],
        },
        "cases": results["cases"],
        "artifacts": ["junit.xml", "pytest.out", "coverage.xml", "coverage/index.html"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

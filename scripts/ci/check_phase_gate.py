#!/usr/bin/env python3
import json
import os
import re
import sys


def find_phase_from_pr():
    title = os.getenv("PR_TITLE", "") or ""
    body = os.getenv("PR_BODY", "") or ""
    text = f"{title}\n{body}"
    m = re.search(r"[Pp]hase\s+(\d+)", text)
    if m:
        return int(m.group(1))
    # fallback: check for pN_tM test files in the diff the CI just ran
    # (assumes runner already checked out repo)
    phases = []
    for root, _, files in os.walk("tests"):
        for f in files:
            m = re.match(r"p(\d+)_t(\d+)_", f)
            if m:
                phases.append(int(m.group(1)))
    return max(phases) if phases else None


def main():
    phase = find_phase_from_pr()
    if not phase:
        print("No explicit phase detected; skipping gate check.")
        return 0

    base = f"reports/phase-{phase}"
    junit = os.path.join(base, "junit.xml")
    summ = os.path.join(base, "summary.json")

    errors = []
    if not os.path.exists(junit):
        errors.append(f"Missing {junit}")
    if not os.path.exists(summ):
        errors.append(f"Missing {summ}")
    else:
        try:
            with open(summ) as f:
                data = json.load(f)
            if str(data.get("phase")) not in {str(phase), f"{phase}"}:
                errors.append(
                    f"summary.json phase mismatch: {data.get('phase')} != {phase}"
                )
            if "results" not in data:
                errors.append("summary.json missing 'results'")
        except Exception as e:
            errors.append(f"Failed to parse {summ}: {e}")

    if errors:
        print("Phase gate check FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print(f"Phase gate check OK for phase {phase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

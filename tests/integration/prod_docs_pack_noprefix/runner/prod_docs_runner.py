"""
Convenience runner for the Phase 7E-3 production docs pack (no filename prefixes).
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
test_path = HERE / "tests" / "test_phase7e3_prod_docs_pack_noprefix.py"


def main():
    cmd = [sys.executable, "-m", "pytest", "-q", str(test_path)]
    print("Running:", " ".join(cmd))
    code = subprocess.call(cmd, cwd=str(HERE))
    if code != 0:
        print(
            "Some cases failed. See artifacts/phase7e3_prod_docs_report.json for details."
        )
    else:
        print("All cases passed. See artifacts/phase7e3_prod_docs_report.json.")


if __name__ == "__main__":
    main()

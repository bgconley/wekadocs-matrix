import argparse
import json
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--phase")
p.add_argument("--junit")
p.add_argument("--out")
a = p.parse_args()
summary = {
    "phase": a.phase,
    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "commit": (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        if os.path.isdir(".git")
        else "unknown"
    ),
    "results": [],
    "metrics": {},
    "artifacts": ["junit.xml", "pytest.out"],
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {a.out}")

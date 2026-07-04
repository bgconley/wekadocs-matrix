#!/usr/bin/env python3
"""Print retrieval diagnostics by ID."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _default_base_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "reports" / "retrieval_diagnostics"


def _find_markdown(base_dir: Path, diagnostic_id: str) -> Path | None:
    pattern = f"{diagnostic_id}.md"
    for path in base_dir.rglob(pattern):
        if path.is_file():
            return path
    return None


def _find_jsonl_record(base_dir: Path, diagnostic_id: str) -> dict | None:
    for jsonl_path in base_dir.rglob("retrieval_diagnostics.jsonl"):
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("diagnostic_id") == diagnostic_id:
                        return record
        except OSError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show retrieval diagnostics")
    parser.add_argument("--id", required=True, help="Diagnostic ID to fetch")
    args = parser.parse_args()

    base_dir = Path(os.getenv("RETRIEVAL_DIAGNOSTICS_DIR", ""))
    if not base_dir:
        base_dir = _default_base_dir()

    md_path = _find_markdown(base_dir, args.id)
    if md_path:
        print(md_path.read_text(encoding="utf-8"))
        return 0

    record = _find_jsonl_record(base_dir, args.id)
    if record:
        print(json.dumps(record, indent=2))
        return 0

    print(f"Diagnostic ID not found: {args.id}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

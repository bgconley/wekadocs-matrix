#!/usr/bin/env python3
"""
Hybrid retrieval eval harness.
Loads queries.yaml (id/text/expected_type) and runs HybridRetriever (no gold labels yet).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.eval.run_eval import build_retriever, evaluate_queries  # noqa: E402
from src.shared.config import get_config  # noqa: E402
from src.shared.connections import get_connection_manager  # noqa: E402


def load_queries(path: Path) -> List[Dict]:
    data = yaml.safe_load(path.read_text()) or {}
    entries = []
    for q in data.get("queries", []):
        entries.append(
            {
                "query": q.get("text") or "",
                "expected_section_ids": list(q.get("expected_section_ids") or []),
                "filters": q.get("filters") or {},
                "metadata": {
                    "id": q.get("id"),
                    "expected_type": q.get("expected_type"),
                },
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval eval harness")
    parser.add_argument("queries", type=Path, help="Path to queries.yaml")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/eval_baseline.json")
    )
    args = parser.parse_args()

    gold_entries = load_queries(args.queries)
    config = get_config()
    manager = get_connection_manager()
    retriever, _ = build_retriever(manager, config)

    results, summary = evaluate_queries(retriever, gold_entries, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"results": results, "summary": summary}, indent=2)
    )
    print(f"Wrote eval report to {args.output}")


if __name__ == "__main__":
    main()

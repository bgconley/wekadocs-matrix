#!/usr/bin/env python3
"""Score citation/evidence coverage from retrieval JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _quotes_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(item.get("quotes"), list):
        return item["quotes"]

    structured = item.get("structured_content") or item.get("structuredContent") or {}
    if isinstance(structured, dict) and isinstance(structured.get("quotes"), list):
        return structured["quotes"]

    evidence = item.get("evidence") or {}
    if isinstance(evidence, dict) and isinstance(evidence.get("quotes"), list):
        return evidence["quotes"]

    return []


def _is_cited_quote(quote: dict[str, Any]) -> bool:
    quote_text = str(quote.get("quote") or quote.get("text") or "").strip()
    doc_tag = str(quote.get("doc_tag") or quote.get("docTag") or "").strip()
    return bool(quote_text and doc_tag)


def score_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score whether each result item contains cited evidence quotes."""
    cited = 0
    uncited = 0
    details: list[dict[str, Any]] = []

    for idx, item in enumerate(results, start=1):
        query_id = item.get("id") or item.get("query_id") or f"q{idx}"
        quotes = _quotes_from_item(item)
        ok = any(_is_cited_quote(q) for q in quotes if isinstance(q, dict))

        if ok:
            cited += 1
        else:
            uncited += 1

        details.append({"id": query_id, "cited": ok, "quote_count": len(quotes)})

    total = cited + uncited
    return {
        "cited": cited,
        "uncited": uncited,
        "total": total,
        "pass_rate": cited / total if total else 0.0,
        "details": details,
    }


def _load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "queries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"No result list found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score citation evidence coverage.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args()

    score = score_evidence(_load_results(args.input))
    text = json.dumps(score, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if score["pass_rate"] >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())

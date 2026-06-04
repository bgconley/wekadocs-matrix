#!/usr/bin/env python3
"""Run the frozen canonical retrieval benchmark against kb_retrieve_evidence."""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import yaml

DEFAULT_QUERY_SET = Path("tests/fixtures/canonical_retrieval_benchmark.yaml")
DEFAULT_REPORT_DIR = Path("reports/retrieval_benchmarks")
DEFAULT_TOOL_NAME = "kb_retrieve_evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen 10-query retrieval benchmark against the MCP server."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERY_SET,
        help=f"YAML benchmark definition (default: {DEFAULT_QUERY_SET})",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the MCP server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--tool-name",
        default=DEFAULT_TOOL_NAME,
        help=f"Tool to call (default: {DEFAULT_TOOL_NAME})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to reports/retrieval_benchmarks/<timestamp>.json",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional previous JSON report to compare against",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the YAML and print the frozen query ids without calling the server.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(f"Invalid benchmark file: {path}")
    if not isinstance(data["queries"], list) or not data["queries"]:
        raise ValueError(f"Benchmark file has no queries: {path}")
    return data


def build_report_path(report_arg: Path | None) -> Path:
    if report_arg is not None:
        report_arg.parent.mkdir(parents=True, exist_ok=True)
        return report_arg
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORT_DIR / f"canonical_retrieval_benchmark_{stamp}.json"


def create_connection(base_url: str) -> Tuple[http.client.HTTPConnection, str]:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    mcp_path = f"{base_path}/_mcp/" if base_path else "/_mcp/"
    if scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            host, port, timeout=120
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=120)
    return conn, mcp_path


def mcp_call(
    base_url: str, tool_name: str, question: str, top_k: int, max_quotes: int
) -> Dict[str, Any]:
    conn, mcp_path = create_connection(base_url)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "canonical-benchmark", "version": "1.0"},
        },
    }
    conn.request("POST", mcp_path, json.dumps(initialize), headers)
    init_resp = conn.getresponse()
    session_id = init_resp.getheader("mcp-session-id")
    init_resp.read()
    headers["mcp-session-id"] = session_id
    conn.request(
        "POST",
        mcp_path,
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        headers,
    )
    conn.getresponse().read()
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": {
                "question": question,
                "top_k": top_k,
                "max_quotes": max_quotes,
            },
        },
    }
    start = time.time()
    conn.request("POST", mcp_path, json.dumps(payload), headers)
    body = conn.getresponse().read().decode()
    elapsed_ms = (time.time() - start) * 1000.0
    response = json.loads(body)
    result = response.get("result", {})
    structured = result.get("structuredContent", {})
    return {
        "latency_ms": elapsed_ms,
        "structured_content": structured,
        "raw_response": response,
    }


def normalize_quotes(quotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for index, quote in enumerate(quotes, start=1):
        normalized.append(
            {
                "rank": index,
                "doc_tag": quote.get("doc_tag") or "",
                "heading": quote.get("heading") or quote.get("title") or "",
                "confidence": quote.get("confidence"),
            }
        )
    return normalized


def evaluate_rule(rule: Dict[str, Any], quotes: List[Dict[str, Any]]) -> Dict[str, Any]:
    kind = rule["kind"]
    within = int(rule.get("within", len(quotes)))
    sample = quotes[:within]
    note = rule.get("note", "")

    if kind == "heading_contains":
        needles = [item.lower() for item in rule.get("any_of", [])]
        hits = [
            quote
            for quote in sample
            if any(needle in quote["heading"].lower() for needle in needles)
        ]
        min_matches = int(rule.get("min_matches", 1))
        passed = len(hits) >= min_matches
        return {
            "kind": kind,
            "passed": passed,
            "note": note,
            "within": within,
            "matched_count": len(hits),
            "required_count": min_matches,
            "matched_ranks": [hit["rank"] for hit in hits],
            "matched_values": [hit["heading"] for hit in hits],
        }

    if kind == "doc_tag_equals":
        values = set(rule.get("any_of", []))
        hits = [quote for quote in sample if quote["doc_tag"] in values]
        min_matches = int(rule.get("min_matches", 1))
        passed = len(hits) >= min_matches
        return {
            "kind": kind,
            "passed": passed,
            "note": note,
            "within": within,
            "matched_count": len(hits),
            "required_count": min_matches,
            "matched_ranks": [hit["rank"] for hit in hits],
            "matched_values": [hit["doc_tag"] for hit in hits],
        }

    if kind == "doc_tag_max_count":
        target = rule["doc_tag"]
        hits = [quote for quote in sample if quote["doc_tag"] == target]
        max_count = int(rule.get("max_count", 0))
        passed = len(hits) <= max_count
        return {
            "kind": kind,
            "passed": passed,
            "note": note,
            "within": within,
            "matched_count": len(hits),
            "max_count": max_count,
            "matched_ranks": [hit["rank"] for hit in hits],
            "matched_values": [hit["doc_tag"] for hit in hits],
        }

    raise ValueError(f"Unsupported rule kind: {kind}")


def evaluate_query(
    query: Dict[str, Any], quotes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    rules = query.get("acceptance", {}).get("rules", [])
    evaluations = [evaluate_rule(rule, quotes) for rule in rules]
    passed = all(item["passed"] for item in evaluations)
    return {
        "passed": passed,
        "rule_count": len(evaluations),
        "passed_rule_count": sum(1 for item in evaluations if item["passed"]),
        "rules": evaluations,
    }


def load_previous_report(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = {}
    for query in data.get("queries", []):
        rows[query["id"]] = query
    return rows


def compare_results(
    current: Dict[str, Any], previous: Dict[str, Any] | None
) -> Dict[str, Any] | None:
    if not previous:
        return None
    curr_top5 = [
        (item.get("doc_tag", ""), item.get("heading", ""))
        for item in current.get("top_results", [])[:5]
    ]
    prev_top5 = [
        (item.get("doc_tag", ""), item.get("heading", ""))
        for item in previous.get("top_results", [])[:5]
    ]
    overlap = len(set(curr_top5) & set(prev_top5))
    return {
        "previous_passed": previous.get("evaluation", {}).get("passed"),
        "top1_changed": curr_top5[:1] != prev_top5[:1],
        "previous_top1": prev_top5[0] if prev_top5 else None,
        "current_top1": curr_top5[0] if curr_top5 else None,
        "top5_overlap": overlap,
    }


def print_query_result(query: Dict[str, Any], result: Dict[str, Any]) -> None:
    status = "PASS" if result["evaluation"]["passed"] else "FAIL"
    print(
        f"[{status}] {query['id']} | {query['label']} | {result['latency_ms']:.2f}ms | "
        f"rules {result['evaluation']['passed_rule_count']}/{result['evaluation']['rule_count']}"
    )
    for item in result["top_results"][:5]:
        print(
            f"  #{item['rank']}: {item['doc_tag']} | {item['heading']} | {item['confidence']}"
        )
    if result.get("comparison"):
        comp = result["comparison"]
        print(
            "  Δ overlap@5={} | top1_changed={} | prev_top1={}".format(
                comp["top5_overlap"],
                comp["top1_changed"],
                comp["previous_top1"],
            )
        )
    failed_rules = [
        rule for rule in result["evaluation"]["rules"] if not rule["passed"]
    ]
    for rule in failed_rules:
        print(
            f"  rule-fail: {rule['note']} | matched={rule.get('matched_count')}"
            f" | within={rule.get('within')}"
        )


def main() -> int:
    args = parse_args()
    benchmark = load_benchmark(args.queries)
    queries = benchmark["queries"]

    if args.dry_run:
        print(
            f"benchmark={benchmark['benchmark']['id']} frozen_on={benchmark['benchmark']['frozen_on']}"
        )
        for query in queries:
            print(f"{query['id']}: {query['text']}")
        return 0

    defaults = benchmark.get("benchmark", {}).get("defaults", {})
    top_k = int(defaults.get("top_k", 20))
    max_quotes = int(defaults.get("max_quotes", 10))
    previous_rows = load_previous_report(args.compare)
    report_path = build_report_path(args.report)

    results: List[Dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    total_start = time.time()

    for query in queries:
        response = mcp_call(
            args.base_url, args.tool_name, query["text"], top_k, max_quotes
        )
        quotes = normalize_quotes(response["structured_content"].get("quotes", []))
        evaluation = evaluate_query(query, quotes)
        row = {
            "id": query["id"],
            "label": query["label"],
            "category": query["category"],
            "text": query["text"],
            "latency_ms": response["latency_ms"],
            "top_results": quotes,
            "evaluation": evaluation,
        }
        row["comparison"] = compare_results(row, previous_rows.get(query["id"]))
        results.append(row)
        print_query_result(query, row)
        print()

    summary = {
        "benchmark_id": benchmark["benchmark"]["id"],
        "frozen_on": benchmark["benchmark"]["frozen_on"],
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "tool_name": args.tool_name,
        "query_count": len(results),
        "passed_queries": sum(1 for row in results if row["evaluation"]["passed"]),
        "failed_queries": sum(1 for row in results if not row["evaluation"]["passed"]),
        "avg_latency_ms": sum(row["latency_ms"] for row in results)
        / max(1, len(results)),
        "duration_ms": (time.time() - total_start) * 1000.0,
    }

    report = {"summary": summary, "queries": results}
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"report={report_path}")
    print(
        "summary: passed={}/{} avg_latency_ms={:.2f}".format(
            summary["passed_queries"],
            summary["query_count"],
            summary["avg_latency_ms"],
        )
    )
    return 0 if summary["failed_queries"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

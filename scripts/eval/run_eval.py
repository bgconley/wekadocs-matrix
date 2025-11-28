#!/usr/bin/env python3
"""
Evaluation harness for wekadocs-matrix hybrid retrieval.

Loads a gold set (YAML list of {query, expected_section_ids, filters})
and executes each query through HybridRetriever, emitting per-query metrics
and aggregate recall/latency summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.providers.factory import ProviderFactory  # noqa: E402
from src.providers.tokenizer_service import TokenizerService  # noqa: E402
from src.query.hybrid_retrieval import HybridRetriever  # noqa: E402
from src.shared.config import Config, get_config, get_embedding_settings  # noqa: E402
from src.shared.connections import (  # noqa: E402
    ConnectionManager,
    get_connection_manager,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval evaluation harness")
    parser.add_argument(
        "--gold",
        required=True,
        type=Path,
        help="Path to YAML gold set file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Override EMBEDDINGS_PROFILE for this run",
    )
    parser.add_argument(
        "--use-query-api",
        type=str,
        choices=("true", "false"),
        help="Force search.vector.qdrant.use_query_api to True/False",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Top-K results to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write per-query JSONL metrics",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print per-query results to stdout in addition to summary",
    )
    return parser.parse_args()


def load_gold(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Gold file must be a list of entries")
    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict) or "query" not in entry:
            raise ValueError(f"Entry {idx} missing required 'query' field")
        normalized.append(
            {
                "query": entry["query"],
                "expected_section_ids": list(entry.get("expected_section_ids", [])),
                "filters": entry.get("filters") or {},
                "metadata": entry.get("metadata") or {},
            }
        )
    return normalized


def maybe_override_profile(profile: Optional[str]) -> None:
    if profile:
        os.environ["EMBEDDINGS_PROFILE"] = profile


def maybe_override_query_api(config: Config, value: Optional[str]) -> None:
    if value is None:
        return
    use_query_api = value.lower() == "true"
    config.search.vector.qdrant.use_query_api = use_query_api


def build_retriever(
    manager: ConnectionManager,
    config: Config,
) -> Tuple[HybridRetriever, TokenizerService]:
    neo4j_driver = manager.get_neo4j_driver()
    qdrant_client = manager.get_qdrant_client()
    embedding_settings = get_embedding_settings(config)
    provider = ProviderFactory().create_embedding_provider(settings=embedding_settings)
    tokenizer = TokenizerService()
    retriever = HybridRetriever(
        neo4j_driver,
        qdrant_client,
        provider,
        tokenizer=tokenizer,
        embedding_settings=embedding_settings,
    )
    return retriever, tokenizer


def compute_recall(
    actual_ids: Sequence[str], expected_ids: Sequence[str]
) -> Tuple[Optional[float], float]:
    if not expected_ids:
        return None, 0.0
    expected = set(expected_ids)
    actual = list(actual_ids)
    hits = expected.intersection(actual)
    recall = len(hits) / len(expected) if expected else None
    mrr = 0.0
    for idx, chunk_id in enumerate(actual):
        if chunk_id in expected:
            mrr = 1.0 / float(idx + 1)
            break
    return recall, mrr


def evaluate_queries(
    retriever: HybridRetriever,
    gold_entries: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "total_queries": len(gold_entries),
        "vector_path_counts": {},
        "recalls": [],
        "mrrs": [],
        "latencies_ms": [],
    }
    for entry in gold_entries:
        query = entry["query"]
        filters = entry.get("filters") or {}
        expected_ids = entry.get("expected_section_ids") or []

        start = time.time()
        chunks, metrics = retriever.retrieve(
            query, top_k=top_k, filters=filters, expand=True
        )
        duration_ms = (time.time() - start) * 1000
        actual_ids = [chunk.chunk_id for chunk in chunks]
        recall, mrr = compute_recall(actual_ids, expected_ids)

        vector_path = metrics.get("vector_path", "unknown")
        summary["vector_path_counts"][vector_path] = (
            summary["vector_path_counts"].get(vector_path, 0) + 1
        )
        if recall is not None:
            summary["recalls"].append(recall)
        summary["mrrs"].append(mrr)
        summary["latencies_ms"].append(duration_ms)

        # Preserve scoring metadata per result for analysis
        results_with_scores = []
        for idx, c in enumerate(chunks):
            results_with_scores.append(
                {
                    "chunk_id": c.chunk_id,
                    "rank": idx + 1,
                    "vector_score": c.vector_score,
                    "fused_score": c.fused_score,
                    "rerank_score": c.rerank_score,
                    "bm25_score": c.bm25_score,
                    "vector_score_kind": c.vector_score_kind,
                }
            )

        results.append(
            {
                "query": query,
                "filters": filters,
                "expected_section_ids": expected_ids,
                "actual_section_ids": actual_ids,
                "results": results_with_scores,
                "recall": recall,
                "mrr": mrr,
                "metrics": metrics,
                "duration_ms": duration_ms,
            }
        )

    summary["avg_recall"] = (
        statistics.mean(summary["recalls"]) if summary["recalls"] else None
    )
    summary["avg_mrr"] = statistics.mean(summary["mrrs"]) if summary["mrrs"] else None
    summary["p95_latency_ms"] = (
        statistics.quantiles(summary["latencies_ms"], n=20)[18]
        if summary["latencies_ms"]
        else None
    )
    summary["p50_latency_ms"] = (
        statistics.median(summary["latencies_ms"]) if summary["latencies_ms"] else None
    )
    summary["latencies_ms_all"] = summary["latencies_ms"]
    return results, summary


def write_results(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Average recall: {summary.get('avg_recall')}")
    print(f"Average MRR: {summary.get('avg_mrr')}")
    print(
        "Vector path counts: "
        + ", ".join(
            f"{path}:{count}"
            for path, count in summary.get("vector_path_counts", {}).items()
        )
    )
    print(
        f"Latency p50(ms): {summary.get('p50_latency_ms')}, "
        f"p95(ms): {summary.get('p95_latency_ms')}"
    )


def main() -> None:
    args = parse_args()
    gold_entries = load_gold(args.gold)
    maybe_override_profile(args.profile)
    config = get_config()
    maybe_override_query_api(config, args.use_query_api)
    manager = get_connection_manager()
    retriever, _ = build_retriever(manager, config)
    records, summary = evaluate_queries(retriever, gold_entries, args.top_k)
    if args.print_results:
        for record in records:
            print(json.dumps(record, default=str, indent=2))
    if args.output:
        write_results(args.output, records)
    print_summary(summary)


if __name__ == "__main__":
    main()

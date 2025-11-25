#!/usr/bin/env python3
"""
Compare BM25 (Neo4j full‑text) vs BGE-M3 sparse (Qdrant) lexical retrieval.

Runs both modes against a small gold set and reports recall/MRR/latency,
using the existing HybridRetriever pipeline. Designed to work with the
sample_ingest docs without requiring schema changes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from src.providers.factory import ProviderFactory
from src.providers.tokenizer_service import TokenizerService
from src.query.hybrid_retrieval import HybridRetriever
from src.shared.config import get_config, get_embedding_settings
from src.shared.connections import get_connection_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BM25 vs BGE-sparse lexical retrieval"
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("docs/sample_ingest/gold_sparse_vs_bm25.yaml"),
        help="YAML gold file with queries and expected_doc_ids",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K results to fetch")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL output for per-query results",
    )
    parser.add_argument(
        "--print-results", action="store_true", help="Print per-query records"
    )
    return parser.parse_args()


def load_gold(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Gold file must be a list of entries")
    normalized = []
    for entry in data:
        normalized.append(
            {
                "query": entry["query"],
                "expected_doc_ids": list(entry.get("expected_doc_ids", [])),
                "filters": entry.get("filters") or {},
            }
        )
    return normalized


def configure_mode(mode: str) -> None:
    """Mutate global config to the requested lexical mode."""
    cfg = get_config()
    if mode == "bm25":
        cfg.search.hybrid.bm25.enabled = True
        cfg.search.vector.qdrant.enable_sparse = False
    elif mode == "bge_sparse":
        cfg.search.hybrid.bm25.enabled = False
        cfg.search.vector.qdrant.enable_sparse = True
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_retriever():
    manager = get_connection_manager()
    cfg = get_config()
    embedding_settings = get_embedding_settings(cfg)
    provider = ProviderFactory().create_embedding_provider(settings=embedding_settings)
    tokenizer = TokenizerService()
    retriever = HybridRetriever(
        manager.get_neo4j_driver(),
        manager.get_qdrant_client(),
        provider,
        tokenizer=tokenizer,
        embedding_settings=embedding_settings,
    )
    return retriever


def doc_match(actual: str, expected: str) -> bool:
    return actual == expected or actual.endswith(expected) or expected in actual


def compute_recall(actual_docs: Sequence[str], expected_docs: Sequence[str]):
    if not expected_docs:
        return None, 0.0
    hits = []
    for exp in expected_docs:
        for idx, doc in enumerate(actual_docs):
            if doc_match(doc, exp):
                hits.append((exp, idx))
                break
    recall = len(hits) / len(expected_docs)
    mrr = 0.0
    for _, idx in hits:
        mrr = max(mrr, 1.0 / float(idx + 1))
    return recall, mrr


def run_mode(mode: str, gold: List[Dict[str, Any]], top_k: int):
    configure_mode(mode)
    retriever = build_retriever()
    records = []
    latencies = []
    recalls = []
    mrrs = []
    for entry in gold:
        start = time.time()
        chunks, metrics = retriever.retrieve(
            entry["query"], top_k=top_k, filters=entry.get("filters"), expand=True
        )
        duration_ms = (time.time() - start) * 1000
        latencies.append(duration_ms)
        actual_docs = [c.document_id for c in chunks]
        recall, mrr = compute_recall(actual_docs, entry["expected_doc_ids"])
        if recall is not None:
            recalls.append(recall)
        mrrs.append(mrr)
        records.append(
            {
                "mode": mode,
                "query": entry["query"],
                "expected_doc_ids": entry["expected_doc_ids"],
                "actual_doc_ids": actual_docs,
                "recall": recall,
                "mrr": mrr,
                "duration_ms": duration_ms,
                "metrics": metrics,
            }
        )
    summary = {
        "mode": mode,
        "avg_recall": statistics.mean(recalls) if recalls else None,
        "avg_mrr": statistics.mean(mrrs) if mrrs else None,
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=20)[18] if latencies else None
        ),
    }
    return records, summary


def main():
    args = parse_args()
    gold = load_gold(args.gold)
    all_records = []
    summaries = []
    for mode in ("bm25", "bge_sparse"):
        records, summary = run_mode(mode, gold, args.top_k)
        all_records.extend(records)
        summaries.append(summary)
        if args.print_results:
            print(f"\n=== Results: {mode} ===")
            for rec in records:
                print(json.dumps(rec, default=str, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, default=str) + "\n")
    print("\n=== Summary ===")
    for s in summaries:
        print(
            f"{s['mode']}: avg_recall={s['avg_recall']}, "
            f"avg_mrr={s['avg_mrr']}, p50={s['p50_latency_ms']}, p95={s['p95_latency_ms']}"
        )


if __name__ == "__main__":
    main()

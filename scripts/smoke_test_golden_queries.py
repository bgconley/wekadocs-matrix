#!/usr/bin/env python3
"""
Golden Query Smoke Test

Runs the golden query set against the current system and reports hit@k metrics.
Designed for quick validation after ingestion or system changes.

Usage:
    python scripts/smoke_test_golden_queries.py
    python scripts/smoke_test_golden_queries.py --top-k 5
    python scripts/smoke_test_golden_queries.py --verbose

Output:
    - Per-query pass/fail with matched document
    - Summary statistics (hit rate, by difficulty, by type)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_golden_queries(path: str) -> List[Dict]:
    """Load golden query set from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


def get_neo4j_doc_titles() -> Dict[str, str]:
    """Get mapping of doc_id -> title from Neo4j."""
    from src.shared.connections import get_connection_manager

    cm = get_connection_manager()
    driver = cm.get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.title AS title"
        )
        return {r["doc_id"]: r["title"] for r in result}


def search_qdrant(
    query_text: str, top_k: int = 10, _cache: dict = {}
) -> List[Tuple[str, str, float]]:
    """
    Search Qdrant and return list of (chunk_id, doc_title, score).

    Returns results with document title resolved from Neo4j.
    Uses cached provider and doc_id->title mapping for efficiency.
    """
    from qdrant_client import QdrantClient

    from src.providers.factory import ProviderFactory
    from src.shared.connections import get_connection_manager

    # Cache embedding provider for reuse
    if "provider" not in _cache:
        _cache["provider"] = ProviderFactory.create_embedding_provider()

    # Cache doc_id -> title mapping
    if "doc_titles" not in _cache:
        cm = get_connection_manager()
        with cm.get_neo4j_driver().session() as session:
            result = session.run(
                "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.title AS title"
            )
            _cache["doc_titles"] = {r["doc_id"]: r["title"] for r in result}

    provider = _cache["provider"]
    doc_titles = _cache["doc_titles"]

    query_embedding = provider.embed_query(query_text)

    # Search Qdrant
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    collection = os.getenv("QDRANT_COLLECTION", "chunks_multi_bge_m3")

    results = qdrant.search(
        collection_name=collection,
        query_vector=("content", query_embedding),
        limit=top_k,
        with_payload=True,
    )

    # Extract results with doc titles from Neo4j mapping
    output = []
    for hit in results:
        chunk_id = hit.id
        payload = hit.payload or {}
        doc_id = payload.get("document_id") or payload.get("doc_id")
        doc_title = doc_titles.get(doc_id, payload.get("heading", "Unknown"))
        score = hit.score
        output.append((chunk_id, doc_title, score))

    return output


def evaluate_query(
    query: Dict, results: List[Tuple[str, str, float]], verbose: bool = False
) -> Dict:
    """
    Evaluate a single query against results.

    Returns dict with:
        - hit: bool (any expected doc in results)
        - rank: int or None (rank of first expected doc, 1-indexed)
        - matched_title: str or None
    """
    expected_titles = set(t.lower() for t in query.get("expected_doc_titles", []))

    for rank, (chunk_id, doc_title, score) in enumerate(results, start=1):
        if doc_title.lower() in expected_titles:
            return {
                "hit": True,
                "rank": rank,
                "matched_title": doc_title,
                "score": score,
            }

    return {
        "hit": False,
        "rank": None,
        "matched_title": None,
        "score": None,
    }


def run_smoke_test(
    queries: List[Dict],
    top_k: int = 10,
    verbose: bool = False,
) -> Dict:
    """
    Run smoke test on all queries.

    Returns summary statistics.
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "top_k": top_k,
        "total_queries": len(queries),
        "queries": [],
        "summary": {},
    }

    hits = 0
    mrr_sum = 0.0
    by_type = {}
    by_difficulty = {}

    print(f"\n{'='*70}")
    print(f"Golden Query Smoke Test - top_k={top_k}")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, start=1):
        query_id = query.get("id", f"q{i}")
        query_text = query.get("text", "")
        query_type = query.get("type", "unknown")
        difficulty = query.get("difficulty", "unknown")

        # Search
        try:
            search_results = search_qdrant(query_text, top_k=top_k)
        except Exception as e:
            print(f"[{query_id}] ERROR: {e}")
            continue

        # Evaluate
        eval_result = evaluate_query(query, search_results, verbose)

        # Track stats
        if eval_result["hit"]:
            hits += 1
            mrr_sum += 1.0 / eval_result["rank"]
            status = f"✅ HIT@{eval_result['rank']}"
        else:
            status = "❌ MISS"

        # By type
        by_type.setdefault(query_type, {"hits": 0, "total": 0})
        by_type[query_type]["total"] += 1
        if eval_result["hit"]:
            by_type[query_type]["hits"] += 1

        # By difficulty
        by_difficulty.setdefault(difficulty, {"hits": 0, "total": 0})
        by_difficulty[difficulty]["total"] += 1
        if eval_result["hit"]:
            by_difficulty[difficulty]["hits"] += 1

        # Output
        if verbose or not eval_result["hit"]:
            expected = query.get("expected_doc_titles", [])[:2]
            print(f"[{query_id}] {status}")
            print(f"    Query: {query_text[:60]}...")
            if eval_result["hit"]:
                print(
                    f"    Matched: {eval_result['matched_title']} (score={eval_result['score']:.3f})"
                )
            else:
                print(f"    Expected: {expected}")
                if search_results:
                    print(
                        f"    Got: {search_results[0][1]} (score={search_results[0][2]:.3f})"
                    )
            print()
        elif eval_result["hit"]:
            print(f"[{query_id}] {status} - {eval_result['matched_title'][:40]}")

        # Store result
        results["queries"].append(
            {
                "id": query_id,
                "text": query_text,
                "type": query_type,
                "difficulty": difficulty,
                **eval_result,
            }
        )

    # Summary
    hit_rate = hits / len(queries) if queries else 0
    mrr = mrr_sum / len(queries) if queries else 0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Hit@{top_k}: {hits}/{len(queries)} ({hit_rate:.1%})")
    print(f"MRR@{top_k}: {mrr:.3f}")
    print()
    print("By Type:")
    for t, stats in sorted(by_type.items()):
        rate = stats["hits"] / stats["total"] if stats["total"] else 0
        print(f"  {t}: {stats['hits']}/{stats['total']} ({rate:.0%})")
    print()
    print("By Difficulty:")
    for d, stats in sorted(by_difficulty.items()):
        rate = stats["hits"] / stats["total"] if stats["total"] else 0
        print(f"  {d}: {stats['hits']}/{stats['total']} ({rate:.0%})")

    results["summary"] = {
        "hits": hits,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "by_type": by_type,
        "by_difficulty": by_difficulty,
    }

    # Pass/fail threshold
    threshold = 0.7
    passed = hit_rate >= threshold
    print(f"\n{'='*70}")
    if passed:
        print(f"✅ SMOKE TEST PASSED (hit_rate={hit_rate:.1%} >= {threshold:.0%})")
    else:
        print(f"❌ SMOKE TEST FAILED (hit_rate={hit_rate:.1%} < {threshold:.0%})")
    print(f"{'='*70}\n")

    results["passed"] = passed
    return results


def main():
    parser = argparse.ArgumentParser(description="Golden Query Smoke Test")
    parser.add_argument(
        "--queries",
        default="tests/fixtures/golden_query_set.yaml",
        help="Path to golden query set YAML",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show details for all queries, not just failures",
    )
    parser.add_argument(
        "--report",
        help="Path to save JSON report",
    )
    args = parser.parse_args()

    # Load queries
    query_path = Path(args.queries)
    if not query_path.exists():
        print(f"Error: Query file not found: {query_path}")
        sys.exit(1)

    queries = load_golden_queries(query_path)
    print(f"Loaded {len(queries)} queries from {query_path}")

    # Run smoke test
    results = run_smoke_test(queries, top_k=args.top_k, verbose=args.verbose)

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_path}")

    # Exit code based on pass/fail
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

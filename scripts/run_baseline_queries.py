#!/usr/bin/env python3
"""
Phase 7E.0 - Task 0.4: Baseline Query Execution

Runs baseline query set against current (micro-section) system.
Captures metrics for comparison after Phase 2 chunking implementation.

Usage:
    python scripts/run_baseline_queries.py
    python scripts/run_baseline_queries.py --queries tests/fixtures/baseline_query_set.yaml
    python scripts/run_baseline_queries.py --report reports/phase-7e/baseline-queries.json
    python scripts/run_baseline_queries.py --top-k 10

Features:
- Executes queries using current hybrid search
- Captures latency, result counts, scores
- Exports results for A/B comparison
- Per-query and aggregate metrics
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.factory import ProviderFactory
from src.query.hybrid_search import QdrantVectorStore
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
from src.shared.observability.logging import setup_logging

logger = logging.getLogger(__name__)


class BaselineQueryRunner:
    """Executes baseline queries and captures metrics"""

    def __init__(self, embedding_provider, vector_store, top_k: int = 20):
        """
        Args:
            embedding_provider: Provider for query embeddings
            vector_store: Vector store for search
            top_k: Number of results to retrieve
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k

    def load_queries(self, query_file: Path) -> List[Dict]:
        """Load query set from YAML"""
        with open(query_file, "r") as f:
            queries = yaml.safe_load(f)

        if not queries:
            raise ValueError(f"No queries found in {query_file}")

        logger.info(f"Loaded {len(queries)} queries from {query_file}")
        return queries

    def execute_query(self, query: Dict) -> Dict:
        """
        Execute single query and capture metrics.

        Args:
            query: Query dict with id, text, category, etc.

        Returns:
            Dict with query execution results and metrics
        """
        query_id = query["id"]
        query_text = query["text"]

        logger.debug(f"Executing query: {query_id} - {query_text}")

        start_time = time.time()

        try:
            # Generate query embedding (task=retrieval.query handled internally by embed_query)
            embed_start = time.time()
            query_vector = self.embedding_provider.embed_query(query_text)
            embed_time = time.time() - embed_start

            # Vector search
            search_start = time.time()
            results = self.vector_store.search(
                vector=query_vector, k=self.top_k, filters=None
            )
            search_time = time.time() - search_start

            total_time = time.time() - start_time

            # Extract result metadata
            result_items = []
            for i, hit in enumerate(results):
                result_items.append(
                    {
                        "rank": i + 1,
                        "section_id": hit.get("node_id"),
                        "score": float(hit.get("score", 0.0)),
                        "document_id": hit.get("document_id"),
                        "heading": hit.get("metadata", {}).get("heading", "Unknown"),
                    }
                )

            return {
                "query_id": query_id,
                "query_text": query_text,
                "category": query["category"],
                "token_estimate": query.get("token_estimate", 0),
                "success": True,
                "results": result_items,
                "result_count": len(results),
                "timing": {
                    "embedding_ms": embed_time * 1000,
                    "search_ms": search_time * 1000,
                    "total_ms": total_time * 1000,
                },
                "top_scores": {
                    "top1": results[0]["score"] if results else 0.0,
                    "top3_avg": (
                        sum(r["score"] for r in results[:3]) / min(3, len(results))
                        if results
                        else 0.0
                    ),
                    "top5_avg": (
                        sum(r["score"] for r in results[:5]) / min(5, len(results))
                        if results
                        else 0.0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Query {query_id} failed: {e}", exc_info=True)
            return {
                "query_id": query_id,
                "query_text": query_text,
                "category": query["category"],
                "success": False,
                "error": str(e),
                "timing": {"total_ms": (time.time() - start_time) * 1000},
            }

    def run_all_queries(self, queries: List[Dict]) -> Dict:
        """
        Run all queries and compile results.

        Args:
            queries: List of query dicts

        Returns:
            Dict with all results and aggregate metrics
        """
        start_time = datetime.now()

        logger.info(f"Executing {len(queries)} queries...")

        results = []
        failed_queries = []

        for i, query in enumerate(queries, 1):
            logger.info(f"[{i}/{len(queries)}] {query['id']}: {query['text']}")

            result = self.execute_query(query)
            results.append(result)

            if not result["success"]:
                failed_queries.append(result["query_id"])

            # Brief pause between queries to respect rate limits
            if i < len(queries):
                time.sleep(0.1)

        # Calculate aggregate statistics
        successful_results = [r for r in results if r["success"]]

        if successful_results:
            avg_embedding_ms = sum(
                r["timing"]["embedding_ms"] for r in successful_results
            ) / len(successful_results)

            avg_search_ms = sum(
                r["timing"]["search_ms"] for r in successful_results
            ) / len(successful_results)

            avg_total_ms = sum(
                r["timing"]["total_ms"] for r in successful_results
            ) / len(successful_results)

            avg_results = sum(r["result_count"] for r in successful_results) / len(
                successful_results
            )

            # Latency percentiles
            total_times = sorted([r["timing"]["total_ms"] for r in successful_results])
            n = len(total_times)

            latency_percentiles = {
                "p50": total_times[int(n * 0.50)] if n > 0 else 0.0,
                "p90": total_times[int(n * 0.90)] if n > 0 else 0.0,
                "p95": total_times[int(n * 0.95)] if n > 0 else 0.0,
                "p99": total_times[int(n * 0.99)] if n > 0 else 0.0,
            }

            # Per-category statistics
            by_category = {}
            for result in successful_results:
                cat = result["category"]
                if cat not in by_category:
                    by_category[cat] = {
                        "count": 0,
                        "total_ms_sum": 0.0,
                        "results_sum": 0,
                    }
                by_category[cat]["count"] += 1
                by_category[cat]["total_ms_sum"] += result["timing"]["total_ms"]
                by_category[cat]["results_sum"] += result["result_count"]

            category_stats = {}
            for cat, stats in by_category.items():
                category_stats[cat] = {
                    "query_count": stats["count"],
                    "avg_latency_ms": stats["total_ms_sum"] / stats["count"],
                    "avg_results": stats["results_sum"] / stats["count"],
                }

        else:
            avg_embedding_ms = 0.0
            avg_search_ms = 0.0
            avg_total_ms = 0.0
            avg_results = 0.0
            latency_percentiles = {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
            category_stats = {}

        duration = (datetime.now() - start_time).total_seconds()

        aggregate_stats = {
            "total_queries": len(queries),
            "successful_queries": len(successful_results),
            "failed_queries": len(failed_queries),
            "failed_query_ids": failed_queries,
            "avg_embedding_ms": avg_embedding_ms,
            "avg_search_ms": avg_search_ms,
            "avg_total_ms": avg_total_ms,
            "avg_results_per_query": avg_results,
            "latency_percentiles": latency_percentiles,
            "by_category": category_stats,
            "duration_seconds": duration,
        }

        logger.info(f"Completed in {duration:.2f}s")
        logger.info(f"Success rate: {len(successful_results)}/{len(queries)}")

        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "top_k": self.top_k,
                "embedding_provider": self.embedding_provider.__class__.__name__,
                "collection": self.vector_store.collection_name,
            },
            "queries": results,
            "aggregate_stats": aggregate_stats,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline query set and capture metrics"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="tests/fixtures/baseline_query_set.yaml",
        help="Path to query set YAML file",
    )
    parser.add_argument("--report", type=str, help="Path to save JSON report")
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("=" * 80)
    logger.info("Phase 7E.0 - Task 0.4: Baseline Query Execution")
    logger.info("=" * 80)
    logger.info(f"Query set: {args.queries}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Report: {args.report if args.report else 'None'}")
    logger.info("")

    try:
        # Load configuration
        config = get_config()

        # Initialize connections
        conn_manager = get_connection_manager()
        qdrant_client = conn_manager.get_qdrant_client()

        # Get embedding provider
        factory = ProviderFactory()
        embedding_provider = factory.create_embedding_provider()
        logger.info(
            f"Using embedding provider: {embedding_provider.__class__.__name__}"
        )

        # Get collection name
        collection_name = config.search.vector.qdrant.collection_name
        logger.info(f"Using collection: {collection_name}")

        # Create vector store
        vector_store = QdrantVectorStore(qdrant_client, collection_name)

        # Create runner
        runner = BaselineQueryRunner(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            top_k=args.top_k,
        )

        # Load queries
        query_file = Path(args.queries)
        if not query_file.exists():
            logger.error(f"Query file not found: {query_file}")
            sys.exit(1)

        queries = runner.load_queries(query_file)

        # Run queries
        results = runner.run_all_queries(queries)

        # Print summary
        stats = results["aggregate_stats"]
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Successful: {stats['successful_queries']}")
        logger.info(f"Failed: {stats['failed_queries']}")
        logger.info(f"Average latency: {stats['avg_total_ms']:.1f}ms")
        logger.info(f"Average results: {stats['avg_results_per_query']:.1f}")
        logger.info("")
        logger.info("Latency percentiles:")
        logger.info(f"  p50: {stats['latency_percentiles']['p50']:.1f}ms")
        logger.info(f"  p90: {stats['latency_percentiles']['p90']:.1f}ms")
        logger.info(f"  p95: {stats['latency_percentiles']['p95']:.1f}ms")

        # Per-category breakdown
        if stats["by_category"]:
            logger.info("")
            logger.info("By category:")
            for cat, cat_stats in stats["by_category"].items():
                logger.info(
                    f"  {cat}: {cat_stats['query_count']} queries, "
                    f"avg {cat_stats['avg_latency_ms']:.1f}ms"
                )

        # Save report
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"\nReport saved to: {report_path}")

        sys.exit(0 if stats["failed_queries"] == 0 else 1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

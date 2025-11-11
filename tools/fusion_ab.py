#!/usr/bin/env python3
"""
Phase 7E-4: A/B Testing Framework for Fusion Methods
Compare RRF vs Weighted fusion with comprehensive metrics

Reference: Canonical Spec L3666, L4903, L2797-2798
Usage:
    python tools/fusion_ab.py --queries queries.yaml --report report.json
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.hybrid_retrieval import FusionMethod, HybridRetriever
from src.shared.config import get_config
from src.shared.connections import get_neo4j_driver, get_qdrant_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Single query evaluation result."""

    query_id: str
    query_text: str
    fusion_method: str
    latency_ms: float
    chunks_returned: int
    expanded: bool
    top_k_ids: List[str]
    relevance_scores: Optional[List[float]] = None  # If judgments available


@dataclass
class ABTestResult:
    """Complete A/B test results."""

    test_name: str
    timestamp: datetime
    queries_count: int

    # Per-method results
    rrf_results: List[QueryResult]
    weighted_results: List[QueryResult]

    # Summary metrics
    summary: Dict[str, any]

    # Go/no-go decision
    recommendation: str
    rationale: str


class FusionABTester:
    """
    A/B test framework for comparing fusion methods.

    Metrics computed:
    - Hit@k (Did relevant doc appear in top k?)
    - MRR@k (Mean Reciprocal Rank)
    - nDCG@k (Normalized Discounted Cumulative Gain)
    - Latency p50/p95
    - Expansion rate
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 20,
        golden_set: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize A/B tester.

        Args:
            retriever: HybridRetriever instance
            top_k: Number of top results to retrieve
            golden_set: Optional golden set mapping query_id -> [relevant_chunk_ids]
        """
        self.retriever = retriever
        self.top_k = top_k
        self.golden_set = golden_set or {}

    def run_query(
        self, query_id: str, query_text: str, fusion_method: FusionMethod
    ) -> QueryResult:
        """
        Run a single query with specified fusion method.

        Args:
            query_id: Query identifier
            query_text: Query text
            fusion_method: Fusion method to use

        Returns:
            QueryResult with timing and results
        """
        start_time = time.time()

        try:
            # Execute search
            results = self.retriever.search(
                query_text=query_text,
                top_k=self.top_k,
                fusion_method=fusion_method,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract chunk IDs
            top_k_ids = [r.chunk_id for r in results]

            # Check if expansion occurred
            expanded = any(r.is_expanded for r in results)

            # Compute relevance scores if golden set available
            relevance_scores = None
            if query_id in self.golden_set:
                relevant_ids = set(self.golden_set[query_id])
                relevance_scores = [
                    1.0 if chunk_id in relevant_ids else 0.0 for chunk_id in top_k_ids
                ]

            return QueryResult(
                query_id=query_id,
                query_text=query_text,
                fusion_method=fusion_method.value,
                latency_ms=latency_ms,
                chunks_returned=len(results),
                expanded=expanded,
                top_k_ids=top_k_ids,
                relevance_scores=relevance_scores,
            )

        except Exception as e:
            logger.error(f"Query '{query_id}' failed with {fusion_method.value}: {e}")
            return QueryResult(
                query_id=query_id,
                query_text=query_text,
                fusion_method=fusion_method.value,
                latency_ms=0.0,
                chunks_returned=0,
                expanded=False,
                top_k_ids=[],
                relevance_scores=None,
            )

    def run_ab_test(
        self, queries: List[Dict[str, str]], test_name: str = "Fusion A/B Test"
    ) -> ABTestResult:
        """
        Run complete A/B test on query set.

        Args:
            queries: List of query dicts with 'id' and 'text' keys
            test_name: Name for this test run

        Returns:
            ABTestResult with comprehensive comparison
        """
        logger.info(f"Starting A/B test: {test_name}")
        logger.info(f"Queries: {len(queries)}, Top-K: {self.top_k}")

        rrf_results = []
        weighted_results = []

        for i, query in enumerate(queries, 1):
            query_id = query["id"]
            query_text = query["text"]

            logger.info(f"[{i}/{len(queries)}] Running query: {query_id}")

            # Test RRF
            rrf_result = self.run_query(query_id, query_text, FusionMethod.RRF)
            rrf_results.append(rrf_result)
            logger.info(
                f"  RRF: {rrf_result.latency_ms:.1f}ms, {rrf_result.chunks_returned} chunks"
            )

            # Test Weighted
            weighted_result = self.run_query(
                query_id, query_text, FusionMethod.WEIGHTED
            )
            weighted_results.append(weighted_result)
            logger.info(
                f"  Weighted: {weighted_result.latency_ms:.1f}ms, {weighted_result.chunks_returned} chunks"
            )

        # Compute summary metrics
        summary = self._compute_summary(rrf_results, weighted_results)

        # Make recommendation
        recommendation, rationale = self._make_recommendation(summary)

        return ABTestResult(
            test_name=test_name,
            timestamp=datetime.utcnow(),
            queries_count=len(queries),
            rrf_results=rrf_results,
            weighted_results=weighted_results,
            summary=summary,
            recommendation=recommendation,
            rationale=rationale,
        )

    def _compute_summary(
        self, rrf_results: List[QueryResult], weighted_results: List[QueryResult]
    ) -> Dict[str, any]:
        """Compute summary metrics for both methods."""
        summary = {"rrf": {}, "weighted": {}, "comparison": {}}

        # Compute per-method summaries
        for method_name, results in [
            ("rrf", rrf_results),
            ("weighted", weighted_results),
        ]:
            latencies = [r.latency_ms for r in results if r.latency_ms > 0]
            expanded_count = sum(1 for r in results if r.expanded)

            method_summary = {
                "latency": {
                    "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
                    "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
                    "mean": float(np.mean(latencies)) if latencies else 0.0,
                },
                "expansion_rate": expanded_count / len(results) if results else 0.0,
                "avg_chunks_returned": (
                    float(np.mean([r.chunks_returned for r in results]))
                    if results
                    else 0.0
                ),
            }

            # Quality metrics (if golden set available)
            if any(r.relevance_scores for r in results):
                hit_at_3 = self._compute_hit_at_k(results, k=3)
                mrr_at_10 = self._compute_mrr_at_k(results, k=10)
                ndcg_at_10 = self._compute_ndcg_at_k(results, k=10)

                method_summary["quality"] = {
                    "hit_at_3": hit_at_3,
                    "mrr_at_10": mrr_at_10,
                    "ndcg_at_10": ndcg_at_10,
                }

            summary[method_name] = method_summary

        # Comparison
        rrf_p95 = summary["rrf"]["latency"]["p95"]
        weighted_p95 = summary["weighted"]["latency"]["p95"]

        summary["comparison"] = {
            "latency_ratio": weighted_p95 / rrf_p95 if rrf_p95 > 0 else 0.0,
            "expansion_rate_delta": summary["weighted"]["expansion_rate"]
            - summary["rrf"]["expansion_rate"],
        }

        # Quality comparison (if available)
        if "quality" in summary["rrf"] and "quality" in summary["weighted"]:
            rrf_hit3 = summary["rrf"]["quality"]["hit_at_3"]
            weighted_hit3 = summary["weighted"]["quality"]["hit_at_3"]
            summary["comparison"]["hit_at_3_improvement"] = weighted_hit3 - rrf_hit3

        return summary

    def _compute_hit_at_k(self, results: List[QueryResult], k: int) -> float:
        """Compute Hit@k (fraction of queries with relevant result in top k)."""
        hits = 0
        total = 0

        for result in results:
            if not result.relevance_scores:
                continue

            total += 1
            top_k_relevant = result.relevance_scores[:k]
            if any(score > 0 for score in top_k_relevant):
                hits += 1

        return hits / total if total > 0 else 0.0

    def _compute_mrr_at_k(self, results: List[QueryResult], k: int) -> float:
        """Compute MRR@k (Mean Reciprocal Rank)."""
        reciprocal_ranks = []

        for result in results:
            if not result.relevance_scores:
                continue

            top_k_relevant = result.relevance_scores[:k]
            for i, score in enumerate(top_k_relevant, 1):
                if score > 0:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def _compute_ndcg_at_k(self, results: List[QueryResult], k: int) -> float:
        """Compute nDCG@k (Normalized Discounted Cumulative Gain)."""
        ndcg_scores = []

        for result in results:
            if not result.relevance_scores:
                continue

            top_k_relevant = result.relevance_scores[:k]

            # DCG
            dcg = sum(
                rel / np.log2(i + 2) for i, rel in enumerate(top_k_relevant) if rel > 0
            )

            # IDCG (ideal DCG - sorted by relevance)
            ideal_scores = sorted(top_k_relevant, reverse=True)
            idcg = sum(
                rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores) if rel > 0
            )

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def _make_recommendation(self, summary: Dict[str, any]) -> tuple[str, str]:
        """
        Make go/no-go recommendation based on summary.

        Criteria (from canonical spec L3701):
        - Hit@3 improvement ≥ +10-15%
        - Retrieval p95 ≤ 1.3× baseline
        """
        comparison = summary["comparison"]

        # Check if we have quality metrics
        if "hit_at_3_improvement" in comparison:
            hit3_improvement = comparison["hit_at_3_improvement"]
            hit3_improvement_pct = hit3_improvement * 100

            latency_ratio = comparison["latency_ratio"]

            # Decision logic
            if hit3_improvement_pct >= 10.0 and latency_ratio <= 1.3:
                recommendation = "GO"
                rationale = (
                    f"✅ Hit@3 improvement: +{hit3_improvement_pct:.1f}% (≥10% required) AND "
                    f"p95 latency ratio: {latency_ratio:.2f}× (≤1.3× required)"
                )
            elif hit3_improvement_pct >= 10.0:
                recommendation = "NO-GO"
                rationale = (
                    f"⚠️ Hit@3 improvement: +{hit3_improvement_pct:.1f}% (✅) BUT "
                    f"p95 latency ratio: {latency_ratio:.2f}× (❌ exceeds 1.3× limit)"
                )
            elif latency_ratio <= 1.3:
                recommendation = "NO-GO"
                rationale = (
                    f"⚠️ p95 latency ratio: {latency_ratio:.2f}× (✅) BUT "
                    f"Hit@3 improvement: +{hit3_improvement_pct:.1f}% (❌ below 10% threshold)"
                )
            else:
                recommendation = "NO-GO"
                rationale = (
                    f"❌ Hit@3 improvement: +{hit3_improvement_pct:.1f}% (below 10%) AND "
                    f"p95 latency ratio: {latency_ratio:.2f}× (exceeds 1.3×)"
                )
        else:
            # No golden set - base decision on latency only
            latency_ratio = comparison["latency_ratio"]
            if latency_ratio <= 1.3:
                recommendation = "CONDITIONAL-GO"
                rationale = f"✅ p95 latency ratio: {latency_ratio:.2f}× (≤1.3×). No quality metrics available - manual validation required."
            else:
                recommendation = "NO-GO"
                rationale = (
                    f"❌ p95 latency ratio: {latency_ratio:.2f}× (exceeds 1.3× limit)"
                )

        return recommendation, rationale


def load_queries(queries_file: Path) -> List[Dict[str, str]]:
    """Load queries from YAML file."""
    with open(queries_file, "r") as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


def load_golden_set(golden_file: Path) -> Dict[str, List[str]]:
    """Load golden set from JSON file."""
    with open(golden_file, "r") as f:
        return json.load(f)


def save_report(
    result: ABTestResult, output_file: Path, csv_file: Optional[Path] = None
):
    """Save A/B test results to JSON and optionally CSV."""
    # JSON report
    report = {
        "test_name": result.test_name,
        "timestamp": result.timestamp.isoformat(),
        "queries_count": result.queries_count,
        "summary": result.summary,
        "recommendation": result.recommendation,
        "rationale": result.rationale,
        "results": {
            "rrf": [
                {
                    "query_id": r.query_id,
                    "latency_ms": r.latency_ms,
                    "chunks_returned": r.chunks_returned,
                    "expanded": r.expanded,
                }
                for r in result.rrf_results
            ],
            "weighted": [
                {
                    "query_id": r.query_id,
                    "latency_ms": r.latency_ms,
                    "chunks_returned": r.chunks_returned,
                    "expanded": r.expanded,
                }
                for r in result.weighted_results
            ],
        },
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved JSON report to {output_file}")

    # CSV report (per-query metrics)
    if csv_file:
        import csv

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query_id",
                    "fusion_method",
                    "latency_ms",
                    "chunks_returned",
                    "expanded",
                ]
            )

            for r in result.rrf_results:
                writer.writerow(
                    [r.query_id, "rrf", r.latency_ms, r.chunks_returned, r.expanded]
                )

            for r in result.weighted_results:
                writer.writerow(
                    [
                        r.query_id,
                        "weighted",
                        r.latency_ms,
                        r.chunks_returned,
                        r.expanded,
                    ]
                )

        logger.info(f"Saved CSV report to {csv_file}")


def print_summary(result: ABTestResult):
    """Print A/B test summary to console."""
    print("\n" + "=" * 80)
    print(f"A/B Test Results: {result.test_name}")
    print("=" * 80)
    print(f"Timestamp: {result.timestamp.isoformat()}")
    print(f"Queries: {result.queries_count}")
    print()

    # RRF summary
    print("RRF Fusion:")
    rrf = result.summary["rrf"]
    print(
        f"  Latency: p50={rrf['latency']['p50']:.1f}ms, p95={rrf['latency']['p95']:.1f}ms"
    )
    print(f"  Expansion rate: {rrf['expansion_rate']:.1%}")
    if "quality" in rrf:
        print(f"  Hit@3: {rrf['quality']['hit_at_3']:.1%}")
        print(f"  MRR@10: {rrf['quality']['mrr_at_10']:.3f}")
        print(f"  nDCG@10: {rrf['quality']['ndcg_at_10']:.3f}")
    print()

    # Weighted summary
    print("Weighted Fusion:")
    weighted = result.summary["weighted"]
    print(
        f"  Latency: p50={weighted['latency']['p50']:.1f}ms, p95={weighted['latency']['p95']:.1f}ms"
    )
    print(f"  Expansion rate: {weighted['expansion_rate']:.1%}")
    if "quality" in weighted:
        print(f"  Hit@3: {weighted['quality']['hit_at_3']:.1%}")
        print(f"  MRR@10: {weighted['quality']['mrr_at_10']:.3f}")
        print(f"  nDCG@10: {weighted['quality']['ndcg_at_10']:.3f}")
    print()

    # Comparison
    print("Comparison:")
    comp = result.summary["comparison"]
    print(f"  Latency ratio (weighted/rrf): {comp['latency_ratio']:.2f}×")
    if "hit_at_3_improvement" in comp:
        improvement_pct = comp["hit_at_3_improvement"] * 100
        print(f"  Hit@3 improvement: {improvement_pct:+.1f}%")
    print()

    # Recommendation
    print("=" * 80)
    print(f"Recommendation: {result.recommendation}")
    print(f"Rationale: {result.rationale}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="A/B test fusion methods")
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to queries YAML file",
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        help="Path to golden set JSON file (optional)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("fusion_ab_report.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Output CSV report path (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top results to retrieve",
    )
    parser.add_argument(
        "--test-name",
        type=str,
        default="Fusion A/B Test",
        help="Name for this test run",
    )

    args = parser.parse_args()

    # Load queries
    queries = load_queries(args.queries)
    logger.info(f"Loaded {len(queries)} queries from {args.queries}")

    # Load golden set if provided
    golden_set = None
    if args.golden_set:
        golden_set = load_golden_set(args.golden_set)
        logger.info(f"Loaded golden set with {len(golden_set)} queries")

    # Initialize connections
    config = get_config()
    neo4j_driver = get_neo4j_driver()
    qdrant_client = get_qdrant_client()

    # Create retriever
    retriever = HybridRetriever(
        neo4j_driver=neo4j_driver,
        qdrant_client=qdrant_client,
        qdrant_collection=config.vector.qdrant.collection_name,
    )

    # Run A/B test
    tester = FusionABTester(
        retriever=retriever, top_k=args.top_k, golden_set=golden_set
    )
    result = tester.run_ab_test(queries, test_name=args.test_name)

    # Save and print results
    save_report(result, args.report, args.csv)
    print_summary(result)

    # Exit with non-zero if NO-GO
    if result.recommendation.startswith("NO-GO"):
        sys.exit(1)


if __name__ == "__main__":
    main()

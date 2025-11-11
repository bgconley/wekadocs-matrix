#!/usr/bin/env python3
"""
Phase 7E-2: A/B Comparison Script for Fusion Methods
Compares RRF vs Weighted fusion on a query set with relevance judgments

Usage:
    python scripts/compare_fusion_methods.py --queries tests/fixtures/fusion_test_queries.yaml

Output:
    - Detailed comparison metrics
    - Relevance scores (NDCG, MRR, MAP)
    - Performance timing
    - Recommendation for production settings
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.embeddings.jina import JinaEmbeddingProvider
from src.providers.tokenizer_service import TokenizerService
from src.query.hybrid_retrieval import FusionMethod, HybridRetriever
from src.shared.config import get_config
from src.shared.observability import get_logger

logger = get_logger(__name__)


@dataclass
class QueryTestCase:
    """Test case for fusion comparison."""

    query: str
    relevant_chunks: List[str]  # Ground truth relevant chunk IDs
    query_type: str  # 'keyword', 'semantic', 'mixed'
    complexity: str  # 'simple', 'complex'


@dataclass
class FusionResult:
    """Results from a fusion method on a query."""

    method: str
    top_k_chunks: List[str]
    scores: List[float]
    time_ms: float
    ndcg: float
    mrr: float
    map_score: float
    hits_at_k: Dict[int, float]


class FusionComparator:
    """Compare fusion methods systematically."""

    def __init__(self, neo4j_uri: str, qdrant_host: str):
        """
        Initialize comparator with database connections.

        Args:
            neo4j_uri: Neo4j connection URI
            qdrant_host: Qdrant host address
        """
        # Initialize connections
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=("neo4j", os.getenv("NEO4J_PASSWORD", "testpassword123"))
        )
        self.qdrant_client = QdrantClient(host=qdrant_host, port=6333)

        # Initialize embedder and tokenizer
        self.embedder = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            task="retrieval.passage",  # Default for documents, embed_query() will switch to retrieval.query
        )
        self.tokenizer = TokenizerService()

        # Create retriever
        self.retriever = HybridRetriever(
            neo4j_driver=self.neo4j_driver,
            qdrant_client=self.qdrant_client,
            embedder=self.embedder,
            tokenizer=self.tokenizer,
        )

        logger.info("FusionComparator initialized")

    def load_test_queries(self, query_file: str) -> List[QueryTestCase]:
        """
        Load test queries from YAML file.

        Expected format:
        queries:
          - query: "network configuration setup"
            relevant: ["chunk_id_1", "chunk_id_2"]
            type: "keyword"
            complexity: "simple"
        """
        with open(query_file, "r") as f:
            data = yaml.safe_load(f)

        test_cases = []
        for q in data.get("queries", []):
            test_cases.append(
                QueryTestCase(
                    query=q["query"],
                    relevant_chunks=q.get("relevant", []),
                    query_type=q.get("type", "mixed"),
                    complexity=q.get("complexity", "simple"),
                )
            )

        logger.info(f"Loaded {len(test_cases)} test queries")
        return test_cases

    def run_fusion_method(
        self,
        query: str,
        method: FusionMethod,
        alpha: Optional[float] = None,
        top_k: int = 20,
    ) -> Tuple[List, Dict]:
        """
        Run a specific fusion method on a query.

        Args:
            query: Search query
            method: Fusion method (RRF or WEIGHTED)
            alpha: Alpha parameter for weighted fusion
            top_k: Number of results to retrieve

        Returns:
            Tuple of (results, metrics)
        """
        # Configure retriever
        self.retriever.fusion_method = method
        if alpha is not None:
            self.retriever.fusion_alpha = alpha

        # Run retrieval
        results, metrics = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            expand=False,  # Disable expansion for fair comparison
        )

        return results, metrics

    def calculate_metrics(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        scores: List[float],
    ) -> Dict:
        """
        Calculate retrieval metrics.

        Args:
            retrieved_chunks: Retrieved chunk IDs in order
            relevant_chunks: Ground truth relevant chunk IDs
            scores: Scores for retrieved chunks

        Returns:
            Dict with NDCG, MRR, MAP, and Hits@K
        """
        if not relevant_chunks:
            return {
                "ndcg": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "hits_at_1": 0.0,
                "hits_at_3": 0.0,
                "hits_at_5": 0.0,
                "hits_at_10": 0.0,
            }

        # Convert to relevance scores (1 if relevant, 0 otherwise)
        relevance = [1 if chunk in relevant_chunks else 0 for chunk in retrieved_chunks]

        # NDCG (Normalized Discounted Cumulative Gain)
        def dcg(relevances, k=None):
            if k:
                relevances = relevances[:k]
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

        actual_dcg = dcg(relevance)
        ideal_relevance = sorted(relevance, reverse=True)
        ideal_dcg = dcg(ideal_relevance)
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks):
            if chunk in relevant_chunks:
                mrr = 1.0 / (i + 1)
                break

        # MAP (Mean Average Precision)
        precisions = []
        relevant_found = 0
        for i, chunk in enumerate(retrieved_chunks):
            if chunk in relevant_chunks:
                relevant_found += 1
                precisions.append(relevant_found / (i + 1))
        map_score = sum(precisions) / len(relevant_chunks) if precisions else 0.0

        # Hits@K
        hits_at_k = {}
        for k in [1, 3, 5, 10]:
            hits = sum(1 for chunk in retrieved_chunks[:k] if chunk in relevant_chunks)
            hits_at_k[k] = hits / min(k, len(relevant_chunks))

        return {
            "ndcg": ndcg,
            "mrr": mrr,
            "map": map_score,
            "hits_at_1": hits_at_k[1],
            "hits_at_3": hits_at_k[3],
            "hits_at_5": hits_at_k[5],
            "hits_at_10": hits_at_k[10],
        }

    def compare_on_query(self, test_case: QueryTestCase) -> Dict[str, FusionResult]:
        """
        Compare all fusion methods on a single query.

        Args:
            test_case: Query test case with ground truth

        Returns:
            Dict mapping method name to FusionResult
        """
        results = {}

        # Test RRF
        logger.debug(f"Testing RRF on: {test_case.query[:50]}...")
        rrf_results, rrf_metrics = self.run_fusion_method(
            test_case.query, FusionMethod.RRF
        )
        rrf_chunks = [r.chunk_id for r in rrf_results]
        rrf_scores = [r.fused_score for r in rrf_results]

        metrics = self.calculate_metrics(
            rrf_chunks, test_case.relevant_chunks, rrf_scores
        )

        results["RRF"] = FusionResult(
            method="RRF",
            top_k_chunks=rrf_chunks[:10],
            scores=rrf_scores[:10],
            time_ms=rrf_metrics["fusion_time_ms"],
            ndcg=metrics["ndcg"],
            mrr=metrics["mrr"],
            map_score=metrics["map"],
            hits_at_k={
                1: metrics["hits_at_1"],
                3: metrics["hits_at_3"],
                5: metrics["hits_at_5"],
                10: metrics["hits_at_10"],
            },
        )

        # Test Weighted with different alphas
        for alpha in [0.3, 0.5, 0.7]:
            logger.debug(f"Testing Weighted(α={alpha}) on: {test_case.query[:50]}...")
            w_results, w_metrics = self.run_fusion_method(
                test_case.query, FusionMethod.WEIGHTED, alpha=alpha
            )
            w_chunks = [r.chunk_id for r in w_results]
            w_scores = [r.fused_score for r in w_results]

            metrics = self.calculate_metrics(
                w_chunks, test_case.relevant_chunks, w_scores
            )

            results[f"Weighted_{alpha}"] = FusionResult(
                method=f"Weighted(α={alpha})",
                top_k_chunks=w_chunks[:10],
                scores=w_scores[:10],
                time_ms=w_metrics["fusion_time_ms"],
                ndcg=metrics["ndcg"],
                mrr=metrics["mrr"],
                map_score=metrics["map"],
                hits_at_k={
                    1: metrics["hits_at_1"],
                    3: metrics["hits_at_3"],
                    5: metrics["hits_at_5"],
                    10: metrics["hits_at_10"],
                },
            )

        return results

    def run_comparison(self, test_cases: List[QueryTestCase]) -> Dict:
        """
        Run full comparison on all test cases.

        Args:
            test_cases: List of query test cases

        Returns:
            Aggregated comparison results
        """
        all_results = []
        method_aggregates = {}

        for i, test_case in enumerate(test_cases):
            logger.info(
                f"Processing query {i+1}/{len(test_cases)}: {test_case.query[:50]}..."
            )
            query_results = self.compare_on_query(test_case)

            # Store individual results
            all_results.append(
                {
                    "query": test_case.query,
                    "type": test_case.query_type,
                    "complexity": test_case.complexity,
                    "results": query_results,
                }
            )

            # Aggregate by method
            for method_name, result in query_results.items():
                if method_name not in method_aggregates:
                    method_aggregates[method_name] = {
                        "ndcg_scores": [],
                        "mrr_scores": [],
                        "map_scores": [],
                        "times": [],
                        "hits_at_1": [],
                        "hits_at_3": [],
                        "hits_at_5": [],
                        "hits_at_10": [],
                    }

                agg = method_aggregates[method_name]
                agg["ndcg_scores"].append(result.ndcg)
                agg["mrr_scores"].append(result.mrr)
                agg["map_scores"].append(result.map_score)
                agg["times"].append(result.time_ms)
                agg["hits_at_1"].append(result.hits_at_k[1])
                agg["hits_at_3"].append(result.hits_at_k[3])
                agg["hits_at_5"].append(result.hits_at_k[5])
                agg["hits_at_10"].append(result.hits_at_k[10])

        # Calculate averages
        summary = {}
        for method_name, agg in method_aggregates.items():
            summary[method_name] = {
                "avg_ndcg": np.mean(agg["ndcg_scores"]),
                "avg_mrr": np.mean(agg["mrr_scores"]),
                "avg_map": np.mean(agg["map_scores"]),
                "avg_time_ms": np.mean(agg["times"]),
                "avg_hits_at_1": np.mean(agg["hits_at_1"]),
                "avg_hits_at_3": np.mean(agg["hits_at_3"]),
                "avg_hits_at_5": np.mean(agg["hits_at_5"]),
                "avg_hits_at_10": np.mean(agg["hits_at_10"]),
                "std_ndcg": np.std(agg["ndcg_scores"]),
                "std_time_ms": np.std(agg["times"]),
            }

        return {
            "individual_results": all_results,
            "summary": summary,
            "num_queries": len(test_cases),
        }

    def print_comparison_report(self, comparison_results: Dict):
        """Print formatted comparison report."""
        print("\n" + "=" * 80)
        print("FUSION METHOD COMPARISON REPORT")
        print("=" * 80)

        summary = comparison_results["summary"]
        num_queries = comparison_results["num_queries"]

        print(f"\nEvaluated on {num_queries} queries")

        # Summary table
        table_data = []
        for method, metrics in summary.items():
            table_data.append(
                [
                    method,
                    f"{metrics['avg_ndcg']:.3f}",
                    f"{metrics['avg_mrr']:.3f}",
                    f"{metrics['avg_map']:.3f}",
                    f"{metrics['avg_hits_at_1']:.3f}",
                    f"{metrics['avg_hits_at_3']:.3f}",
                    f"{metrics['avg_hits_at_5']:.3f}",
                    f"{metrics['avg_hits_at_10']:.3f}",
                    f"{metrics['avg_time_ms']:.2f}",
                ]
            )

        headers = [
            "Method",
            "NDCG",
            "MRR",
            "MAP",
            "H@1",
            "H@3",
            "H@5",
            "H@10",
            "Time(ms)",
        ]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

        # Find best method by NDCG
        best_method = max(summary.items(), key=lambda x: x[1]["avg_ndcg"])
        print(f"\n✅ BEST METHOD (by NDCG): {best_method[0]}")
        print(f"   - NDCG: {best_method[1]['avg_ndcg']:.3f}")
        print(f"   - MRR: {best_method[1]['avg_mrr']:.3f}")
        print(f"   - Avg Time: {best_method[1]['avg_time_ms']:.2f}ms")

        # Robustness analysis
        print("\n📊 ROBUSTNESS ANALYSIS:")
        for method, metrics in summary.items():
            stability = metrics["std_ndcg"]
            print(f"   {method}: σ(NDCG) = {stability:.3f}")

        # Performance vs Quality trade-off
        print("\n⚖️  PERFORMANCE vs QUALITY:")
        rrf_ndcg = summary.get("RRF", {}).get("avg_ndcg", 0)
        rrf_time = summary.get("RRF", {}).get("avg_time_ms", 0)

        for method, metrics in summary.items():
            if method != "RRF":
                ndcg_diff = (
                    ((metrics["avg_ndcg"] - rrf_ndcg) / rrf_ndcg * 100)
                    if rrf_ndcg > 0
                    else 0
                )
                time_diff = (
                    ((metrics["avg_time_ms"] - rrf_time) / rrf_time * 100)
                    if rrf_time > 0
                    else 0
                )
                print(f"   {method}: NDCG {ndcg_diff:+.1f}%, Time {time_diff:+.1f}%")

        # Recommendation
        print("\n🎯 RECOMMENDATION:")
        if best_method[0] == "RRF":
            print("   Use RRF fusion - best quality and no tuning required")
        elif "Weighted_0.7" in best_method[0]:
            print("   Use Weighted fusion with α=0.7 - emphasizes semantic search")
        elif "Weighted_0.3" in best_method[0]:
            print("   Use Weighted fusion with α=0.3 - emphasizes keyword search")
        else:
            print(f"   Use {best_method[0]} for optimal results on this dataset")

    def save_results(self, comparison_results: Dict, output_file: str):
        """Save detailed results to JSON file."""

        # Convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        results_native = convert_to_native(comparison_results)

        with open(output_file, "w") as f:
            json.dump(results_native, f, indent=2)

        logger.info(f"Results saved to {output_file}")


def create_sample_queries(output_file: str):
    """Create sample query file for testing."""
    sample_queries = {
        "queries": [
            {
                "query": "network configuration setup",
                "relevant": [],  # Will be populated from actual data
                "type": "keyword",
                "complexity": "simple",
            },
            {
                "query": "how to configure IP addresses and subnet masks",
                "relevant": [],
                "type": "semantic",
                "complexity": "complex",
            },
            {
                "query": "troubleshoot connection timeout",
                "relevant": [],
                "type": "mixed",
                "complexity": "simple",
            },
            {
                "query": "storage volume management and allocation",
                "relevant": [],
                "type": "keyword",
                "complexity": "simple",
            },
            {
                "query": "what are the steps to add a new node to the cluster",
                "relevant": [],
                "type": "semantic",
                "complexity": "complex",
            },
            {
                "query": "MTU settings network optimization",
                "relevant": [],
                "type": "mixed",
                "complexity": "simple",
            },
            {
                "query": "diagnose and fix network connectivity issues between nodes",
                "relevant": [],
                "type": "semantic",
                "complexity": "complex",
            },
            {
                "query": "firewall rules configuration",
                "relevant": [],
                "type": "keyword",
                "complexity": "simple",
            },
            {
                "query": "how do I monitor system performance metrics",
                "relevant": [],
                "type": "semantic",
                "complexity": "simple",
            },
            {
                "query": "backup restore procedures disaster recovery",
                "relevant": [],
                "type": "mixed",
                "complexity": "complex",
            },
        ]
    }

    with open(output_file, "w") as f:
        yaml.dump(sample_queries, f, default_flow_style=False)

    print(f"Sample queries created at: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare fusion methods for hybrid retrieval"
    )
    parser.add_argument("--queries", type=str, help="Path to query test file (YAML)")
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample query file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fusion_comparison_results.json",
        help="Output file for results (default: fusion_comparison_results.json)",
    )

    args = parser.parse_args()

    # Create sample if requested
    if args.create_sample:
        create_sample_queries("tests/fixtures/fusion_test_queries.yaml")
        return

    # Validate queries file
    if not args.queries:
        print("Error: --queries file required (or use --create-sample)")
        sys.exit(1)

    if not Path(args.queries).exists():
        print(f"Error: Query file not found: {args.queries}")
        sys.exit(1)

    # Get configuration
    config = get_config()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")

    # Run comparison
    try:
        comparator = FusionComparator(neo4j_uri, qdrant_host)
        test_cases = comparator.load_test_queries(args.queries)

        if not test_cases:
            print("Error: No test queries found in file")
            sys.exit(1)

        print(f"Running comparison on {len(test_cases)} queries...")
        comparison_results = comparator.run_comparison(test_cases)

        # Print report
        comparator.print_comparison_report(comparison_results)

        # Save results
        comparator.save_results(comparison_results, args.output)

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise
    finally:
        # Cleanup
        if "comparator" in locals():
            comparator.neo4j_driver.close()


if __name__ == "__main__":
    main()

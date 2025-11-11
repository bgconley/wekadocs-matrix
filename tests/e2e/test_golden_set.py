"""
Phase 7a, Day 1: Golden Set Baseline Tests

Tests the 20-query golden set defined in /docs/golden-set-queries.md
Establishes baseline metrics for completeness and latency.

NO MOCKS - All tests run against live Docker stack.

See: /docs/golden-set-queries.md
See: /docs/phase7-target-phase-tasklist.md → Day 1
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Golden set queries from /docs/golden-set-queries.md
GOLDEN_SET_QUERIES = [
    # Installation & Setup (1-4)
    {
        "id": 1,
        "category": "installation",
        "query": "How do I install Weka on Ubuntu?",
        "expected_sections": (3, 5),
    },
    {
        "id": 2,
        "category": "installation",
        "query": "What are the hardware requirements for Weka?",
        "expected_sections": (2, 4),
    },
    {
        "id": 3,
        "category": "installation",
        "query": "How do I set up a Weka cluster?",
        "expected_sections": (4, 6),
    },
    {
        "id": 4,
        "category": "installation",
        "query": "How do I configure Weka licensing?",
        "expected_sections": (2, 3),
    },
    # Configuration & Management (5-8)
    {
        "id": 5,
        "category": "configuration",
        "query": "How do I create a filesystem in Weka?",
        "expected_sections": (3, 5),
    },
    {
        "id": 6,
        "category": "configuration",
        "query": "How do I manage users and permissions in Weka?",
        "expected_sections": (3, 4),
    },
    {
        "id": 7,
        "category": "configuration",
        "query": "How do I configure networking for Weka?",
        "expected_sections": (4, 6),
    },
    {
        "id": 8,
        "category": "configuration",
        "query": "How do I create and manage snapshots?",
        "expected_sections": (3, 5),
    },
    # Operations & Troubleshooting (9-12)
    {
        "id": 9,
        "category": "operations",
        "query": "How do I monitor Weka cluster health?",
        "expected_sections": (4, 6),
    },
    {
        "id": 10,
        "category": "operations",
        "query": "What do I do if a drive fails?",
        "expected_sections": (3, 4),
    },
    {
        "id": 11,
        "category": "operations",
        "query": "How do I diagnose performance issues?",
        "expected_sections": (5, 7),
    },
    {
        "id": 12,
        "category": "operations",
        "query": "How do I collect and analyze Weka logs?",
        "expected_sections": (3, 4),
    },
    # Performance & Optimization (13-16)
    {
        "id": 13,
        "category": "performance",
        "query": "How do I optimize SSD performance in Weka?",
        "expected_sections": (4, 5),
    },
    {
        "id": 14,
        "category": "performance",
        "query": "How do I tune network performance?",
        "expected_sections": (3, 5),
    },
    {
        "id": 15,
        "category": "performance",
        "query": "How do I analyze IO performance?",
        "expected_sections": (4, 6),
    },
    {
        "id": 16,
        "category": "performance",
        "query": "How do I plan capacity for growth?",
        "expected_sections": (3, 4),
    },
    # Advanced Features (17-20)
    {
        "id": 17,
        "category": "advanced",
        "query": "How do I configure tiering to object storage?",
        "expected_sections": (5, 7),
    },
    {
        "id": 18,
        "category": "advanced",
        "query": "How do I set up disaster recovery?",
        "expected_sections": (5, 7),
    },
    {
        "id": 19,
        "category": "advanced",
        "query": "How do I use the Weka REST API?",
        "expected_sections": (4, 6),
    },
    {
        "id": 20,
        "category": "advanced",
        "query": "How do I upgrade Weka to a new version?",
        "expected_sections": (4, 6),
    },
]


@pytest.fixture
def reports_dir():
    """Create reports directory for baseline metrics"""
    report_path = Path("reports/phase-7")
    report_path.mkdir(parents=True, exist_ok=True)
    return report_path


class TestGoldenSetBaseline:
    """Baseline tests for golden set queries"""

    def test_golden_set_queries_baseline(
        self, neo4j_driver, qdrant_client, reports_dir
    ):
        """
        Run all 20 golden set queries and collect baseline metrics.

        DoD:
        - All 20 queries execute successfully
        - Latency measured (P50, P95, P99)
        - Results count tracked
        - Baseline CSV generated

        Expected: No failures, baseline metrics documented
        """
        from sentence_transformers import SentenceTransformer

        from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore
        from src.shared.config import get_config

        config = get_config()

        # Initialize embedder
        embedder = SentenceTransformer(config.embedding.embedding_model)

        # Initialize vector store
        collection_name = config.search.vector.qdrant.collection_name
        vector_store = QdrantVectorStore(qdrant_client, collection_name)

        # Initialize search engine
        search_service = HybridSearchEngine(vector_store, neo4j_driver, embedder)

        results = []
        latencies = []

        for query_def in GOLDEN_SET_QUERIES:
            query_id = query_def["id"]
            query_text = query_def["query"]
            category = query_def["category"]
            expected_min, expected_max = query_def["expected_sections"]

            # Execute query and measure latency
            start_time = time.perf_counter()

            try:
                search_result = search_service.search(
                    query_text=query_text,
                    k=8,
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                # Extract results from HybridSearchResults dataclass
                sections_found = len(search_result.results)
                top_score = (
                    search_result.results[0].score if search_result.results else 0.0
                )

                # Check if within expected range
                in_range = expected_min <= sections_found <= expected_max

                result_entry = {
                    "query_id": query_id,
                    "category": category,
                    "query": query_text,
                    "latency_ms": round(latency_ms, 2),
                    "sections_found": sections_found,
                    "expected_min": expected_min,
                    "expected_max": expected_max,
                    "in_expected_range": in_range,
                    "top_score": round(top_score, 4),
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat(),
                }

                latencies.append(latency_ms)

            except Exception as e:
                result_entry = {
                    "query_id": query_id,
                    "category": category,
                    "query": query_text,
                    "latency_ms": 0,
                    "sections_found": 0,
                    "expected_min": expected_min,
                    "expected_max": expected_max,
                    "in_expected_range": False,
                    "top_score": 0.0,
                    "status": f"error: {str(e)[:100]}",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            results.append(result_entry)

        # Calculate percentiles
        if latencies:
            latencies_sorted = sorted(latencies)
            p50_idx = int(len(latencies_sorted) * 0.5)
            p95_idx = int(len(latencies_sorted) * 0.95)
            p99_idx = int(len(latencies_sorted) * 0.99)

            p50 = latencies_sorted[p50_idx]
            p95 = latencies_sorted[p95_idx]
            p99 = latencies_sorted[p99_idx]
        else:
            p50 = p95 = p99 = 0

        # Write CSV report
        csv_path = reports_dir / "baseline.csv"
        with open(csv_path, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        # Write summary JSON
        summary = {
            "test_run": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_queries": len(GOLDEN_SET_QUERIES),
                "successful_queries": sum(
                    1 for r in results if r["status"] == "success"
                ),
                "failed_queries": sum(1 for r in results if r["status"] != "success"),
            },
            "latency_metrics": {
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "min_ms": round(min(latencies), 2) if latencies else 0,
                "max_ms": round(max(latencies), 2) if latencies else 0,
                "mean_ms": (
                    round(sum(latencies) / len(latencies), 2) if latencies else 0
                ),
            },
            "completeness_metrics": {
                "in_expected_range": sum(1 for r in results if r["in_expected_range"]),
                "below_expected": sum(
                    1 for r in results if r["sections_found"] < r["expected_min"]
                ),
                "above_expected": sum(
                    1 for r in results if r["sections_found"] > r["expected_max"]
                ),
            },
            "by_category": self._calculate_category_stats(results),
        }

        summary_path = reports_dir / "baseline-summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Assertions
        assert (
            summary["test_run"]["successful_queries"] >= 18
        ), f"Too many failures: {summary['test_run']['failed_queries']} failed"

        # Log summary
        print(f"\n{'='*60}")
        print("GOLDEN SET BASELINE RESULTS")
        print(f"{'='*60}")
        print(f"Total queries: {summary['test_run']['total_queries']}")
        print(f"Successful: {summary['test_run']['successful_queries']}")
        print(f"Failed: {summary['test_run']['failed_queries']}")
        print("\nLatency (snippet mode baseline):")
        print(f"  P50: {summary['latency_metrics']['p50_ms']}ms")
        print(f"  P95: {summary['latency_metrics']['p95_ms']}ms")
        print(f"  P99: {summary['latency_metrics']['p99_ms']}ms")
        print("\nCompleteness:")
        print(
            f"  In expected range: {summary['completeness_metrics']['in_expected_range']}/20"
        )
        print("\nReports written to:")
        print(f"  - {csv_path}")
        print(f"  - {summary_path}")
        print(f"{'='*60}\n")

    def _calculate_category_stats(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate per-category statistics"""
        categories = {}

        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {
                    "queries": 0,
                    "successful": 0,
                    "mean_latency_ms": 0,
                    "mean_sections_found": 0,
                }

            categories[cat]["queries"] += 1
            if result["status"] == "success":
                categories[cat]["successful"] += 1
                categories[cat]["mean_latency_ms"] += result["latency_ms"]
                categories[cat]["mean_sections_found"] += result["sections_found"]

        # Calculate means
        for cat in categories:
            if categories[cat]["successful"] > 0:
                categories[cat]["mean_latency_ms"] = round(
                    categories[cat]["mean_latency_ms"] / categories[cat]["successful"],
                    2,
                )
                categories[cat]["mean_sections_found"] = round(
                    categories[cat]["mean_sections_found"]
                    / categories[cat]["successful"],
                    1,
                )

        return categories


class TestGoldenSetSanity:
    """Quick sanity checks for golden set infrastructure"""

    def test_query_service_available(self, neo4j_driver, qdrant_client):
        """
        Verify query service can execute a simple query.

        DoD:
        - Service initializes without errors
        - Single query executes successfully
        """
        from sentence_transformers import SentenceTransformer

        from src.query.hybrid_search import HybridSearchEngine, QdrantVectorStore
        from src.shared.config import get_config

        config = get_config()

        # Initialize embedder
        embedder = SentenceTransformer(config.embedding.embedding_model)

        # Initialize vector store
        collection_name = config.search.vector.qdrant.collection_name
        vector_store = QdrantVectorStore(qdrant_client, collection_name)

        # Initialize search engine
        search_service = HybridSearchEngine(vector_store, neo4j_driver, embedder)

        # Execute simple query
        result = search_service.search(
            query_text="configuration",
            k=3,
        )

        assert hasattr(result, "results")
        assert isinstance(result.results, list)

    def test_feature_flags_configured(self):
        """
        Verify Phase 7a feature flags are configured.

        DoD:
        - verbosity_enabled flag exists and is enabled
        - graph_mode_enabled flag exists and is enabled
        """
        from src.shared.feature_flags import get_feature_flag_manager

        manager = get_feature_flag_manager()

        # Check verbosity flag
        assert manager.is_enabled(
            "verbosity_enabled"
        ), "verbosity_enabled flag not enabled"

        # Check graph mode flag
        assert manager.is_enabled(
            "graph_mode_enabled"
        ), "graph_mode_enabled flag not enabled"

    def test_metrics_collectors_available(self):
        """
        Verify Phase 7a metrics are defined.

        DoD:
        - mcp_search_verbosity_total counter exists
        - mcp_search_response_size_bytes histogram exists
        - mcp_traverse_depth_total counter exists
        - mcp_traverse_nodes_found histogram exists
        """
        from src.shared.observability import metrics

        # Check metrics existence
        assert hasattr(metrics, "mcp_search_verbosity_total")
        assert hasattr(metrics, "mcp_search_response_size_bytes")
        assert hasattr(metrics, "mcp_traverse_depth_total")
        assert hasattr(metrics, "mcp_traverse_nodes_found")

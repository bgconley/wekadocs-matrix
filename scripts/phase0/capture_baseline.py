#!/usr/bin/env python3
"""
Phase 0: Baseline Metrics Capture

Captures baseline metrics before any Phase 1+ changes:
- Reranker latency (P50, P95, P99) with k=50 candidates
- Retrieval quality (MRR, NDCG@10) on baseline query set
- Graph statistics (node/edge counts, cross-doc edges, CONTAINS_STEP)

Usage:
    python scripts/phase0/capture_baseline.py [--output reports/baseline_metrics.json]
    python scripts/phase0/capture_baseline.py --dry-run  # Check connectivity only

Gate Criteria (Phase 0 → Phase 1):
    - Baseline metrics captured in reports/baseline_metrics.json
    - Test harness executes against live Neo4j + Qdrant
    - Graph audit confirms cross-doc edge count (expected: 0 pre-fix)
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import log2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase0.baseline")


@dataclass
class LatencyMetrics:
    """Latency percentiles in milliseconds."""

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    sample_count: int


@dataclass
class QualityMetrics:
    """Retrieval quality metrics."""

    mrr: float  # Mean Reciprocal Rank
    ndcg_at_10: float  # Normalized Discounted Cumulative Gain @ 10
    precision_at_5: float
    recall_at_10: float
    queries_evaluated: int
    queries_with_results: int


@dataclass
class GraphStats:
    """Graph database statistics."""

    # Node counts
    document_count: int
    section_count: int
    chunk_count: int
    entity_count: int
    command_count: int
    config_count: int
    procedure_count: int
    step_count: int

    # Relationship counts
    child_of_count: int
    next_count: int
    prev_count: int
    mentions_count: int
    contains_step_count: int
    references_count: int  # M5: Phase 3 REFERENCES edges (Chunk→Document)

    # Critical checks
    cross_doc_edge_count: int  # Should be 0 pre-Phase 3
    total_nodes: int
    total_relationships: int


@dataclass
class BaselineReport:
    """Complete baseline metrics report."""

    timestamp: str
    git_commit: str
    git_branch: str

    # Component metrics
    reranker_latency: Optional[LatencyMetrics] = None
    retrieval_latency: Optional[LatencyMetrics] = None
    quality: Optional[QualityMetrics] = None
    graph: Optional[GraphStats] = None

    # Metadata
    query_set_path: str = ""
    query_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def get_git_info() -> Tuple[str, str]:
    """Get current git commit and branch."""
    import subprocess

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return commit, branch
    except Exception:
        return "unknown", "unknown"


def load_query_set(path: str) -> List[Dict[str, Any]]:
    """Load baseline query set from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}  # Guard against empty file returning None
    return data.get("queries", [])


def measure_reranker_latency(
    num_candidates: int = 50,
    num_iterations: int = 20,
) -> Optional[LatencyMetrics]:
    """
    Measure reranker latency with synthetic candidates.

    This tests the BGE reranker service directly to establish
    baseline latency before the batching fix (Phase 1.1).
    """
    try:
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider
    except ImportError as e:
        logger.error(f"Cannot import reranker: {e}")
        return None

    try:
        reranker = BGERerankerServiceProvider()
        # Health check
        if not reranker.health_check():
            logger.warning("Reranker health check failed - service may be unavailable")
            return None
    except Exception as e:
        logger.warning(f"Reranker initialization failed: {e}")
        return None

    # Generate synthetic candidates
    query = "How do I create a filesystem snapshot in WEKA?"
    candidates = [
        {
            "text": f"Document {i}: WEKA filesystem operations include snapshots, "
            f"tiering, and data protection features for enterprise storage."
        }
        for i in range(num_candidates)
    ]

    latencies_ms: List[float] = []

    logger.info(
        f"Measuring reranker latency: {num_iterations} iterations, {num_candidates} candidates each"
    )

    for i in range(num_iterations):
        start = time.perf_counter()
        try:
            reranker.rerank(query, candidates, top_k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
            if (i + 1) % 5 == 0:
                logger.info(f"  Iteration {i + 1}/{num_iterations}: {elapsed_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"Rerank iteration {i} failed: {e}")
            continue

    if not latencies_ms:
        return None

    # Use numpy for accurate percentile calculation with linear interpolation
    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    return LatencyMetrics(
        p50=float(p50),
        p95=float(p95),
        p99=float(p99),
        mean=statistics.mean(latencies_ms),
        min=min(latencies_ms),
        max=max(latencies_ms),
        sample_count=len(latencies_ms),
    )


def measure_retrieval_quality(
    queries: List[Dict[str, Any]],
) -> Tuple[Optional[QualityMetrics], Optional[LatencyMetrics]]:
    """
    Measure retrieval quality using topic-based relevance.

    Since we don't have gold-standard relevance labels, we use
    topic matching as a proxy for relevance.
    """
    try:
        from src.query.hybrid_retrieval import HybridRetriever
    except ImportError as e:
        logger.error(f"Cannot import HybridRetriever: {e}")
        return None, None

    try:
        retriever = HybridRetriever()
    except Exception as e:
        logger.warning(f"HybridRetriever initialization failed: {e}")
        return None, None

    reciprocal_ranks: List[float] = []
    ndcg_scores: List[float] = []
    precision_at_5_scores: List[float] = []
    recall_at_10_scores: List[float] = []
    latencies_ms: List[float] = []
    queries_with_results = 0

    logger.info(f"Evaluating {len(queries)} queries for quality metrics")

    for i, q in enumerate(queries):
        query_text = q["text"]
        expected_topics = [t.lower() for t in q.get("expected_topics", [])]

        start = time.perf_counter()
        try:
            results, metrics = retriever.retrieve(query_text, top_k=20)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
        except Exception as e:
            logger.warning(f"Query '{query_text[:30]}...' failed: {e}")
            continue

        if not results:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)
            precision_at_5_scores.append(0.0)
            recall_at_10_scores.append(0.0)
            continue

        queries_with_results += 1

        # Compute topic-based relevance for each result
        def relevance(result) -> float:
            text = (result.text + " " + result.heading).lower()
            matches = sum(1 for topic in expected_topics if topic in text)
            return matches / max(1, len(expected_topics))

        relevances = [relevance(r) for r in results[:20]]

        # MRR: reciprocal rank of first relevant result (relevance > 0.3)
        rr = 0.0
        for rank, rel in enumerate(relevances, 1):
            if rel > 0.3:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # NDCG@10 (using standard log2 denominator for IR-comparable scores)
        dcg = sum(
            rel / log2(rank + 1)
            for rank, rel in enumerate(relevances[:10], 1)
            if rel > 0
        )
        ideal_rels = sorted(relevances[:10], reverse=True)
        idcg = sum(
            rel / log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1) if rel > 0
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

        # Precision@5 (divide by actual count when fewer than 5 results)
        k = min(5, len(relevances))
        relevant_at_5 = sum(1 for rel in relevances[:k] if rel > 0.3)
        precision_at_5_scores.append(relevant_at_5 / k if k > 0 else 0.0)

        # Recall@10 (estimate based on topic coverage)
        covered_topics = set()
        for r in results[:10]:
            text = (r.text + " " + r.heading).lower()
            for topic in expected_topics:
                if topic in text:
                    covered_topics.add(topic)
        recall = len(covered_topics) / max(1, len(expected_topics))
        recall_at_10_scores.append(recall)

        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluated {i + 1}/{len(queries)} queries")

    if not reciprocal_ranks:
        return None, None

    quality = QualityMetrics(
        mrr=statistics.mean(reciprocal_ranks),
        ndcg_at_10=statistics.mean(ndcg_scores),
        precision_at_5=statistics.mean(precision_at_5_scores),
        recall_at_10=statistics.mean(recall_at_10_scores),
        queries_evaluated=len(queries),
        queries_with_results=queries_with_results,
    )

    if latencies_ms:
        # Use numpy for accurate percentile calculation with linear interpolation
        p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
        latency = LatencyMetrics(
            p50=float(p50),
            p95=float(p95),
            p99=float(p99),
            mean=statistics.mean(latencies_ms),
            min=min(latencies_ms),
            max=max(latencies_ms),
            sample_count=len(latencies_ms),
        )
    else:
        latency = None

    return quality, latency


def collect_graph_stats() -> Optional[GraphStats]:
    """
    Collect graph database statistics.

    Key metrics:
    - Node/relationship counts by type
    - cross_doc_edge_count: Should be 0 before Phase 3 (REFERENCES)
    - contains_step_count: Should be 0 before Phase 1.2 fix
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j driver not installed")
        return None

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")

    try:
        # Use context manager to ensure driver is always closed, even on exception
        with GraphDatabase.driver(uri, auth=("neo4j", password)) as driver:
            with driver.session() as session:
                # Node counts
                def count_label(label: str) -> int:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
                    return result.single()["cnt"]

                # Relationship counts
                def count_rel(rel_type: str) -> int:
                    result = session.run(
                        f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as cnt"
                    )
                    return result.single()["cnt"]

                # Cross-document edges (critical check)
                cross_doc = session.run(
                    """
                    MATCH (c1:Chunk)-[r]->(c2:Chunk)
                    WHERE c1.document_id IS NOT NULL
                      AND c2.document_id IS NOT NULL
                      AND c1.document_id <> c2.document_id
                    RETURN count(r) as cnt
                """
                ).single()["cnt"]

                # Total counts
                total_nodes = session.run("MATCH (n) RETURN count(n) as cnt").single()[
                    "cnt"
                ]
                total_rels = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as cnt"
                ).single()["cnt"]

                stats = GraphStats(
                    document_count=count_label("Document"),
                    section_count=count_label("Section"),
                    chunk_count=count_label("Chunk"),
                    entity_count=count_label("Entity"),
                    command_count=count_label("Command"),
                    config_count=count_label("Configuration"),
                    procedure_count=count_label("Procedure"),
                    step_count=count_label("Step"),
                    child_of_count=count_rel("CHILD_OF"),
                    next_count=count_rel("NEXT"),
                    prev_count=count_rel("PREV"),
                    mentions_count=count_rel("MENTIONS"),
                    contains_step_count=count_rel("CONTAINS_STEP"),
                    references_count=count_rel("REFERENCES"),  # M5: Phase 3 gate metric
                    cross_doc_edge_count=cross_doc,
                    total_nodes=total_nodes,
                    total_relationships=total_rels,
                )

                return stats

    except Exception as e:
        logger.error(f"Failed to collect graph stats: {e}")
        return None


def check_connectivity() -> Dict[str, bool]:
    """Check connectivity to all required services."""
    status = {}

    # Neo4j
    try:
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
        driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        status["neo4j"] = True
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        status["neo4j"] = False

    # Qdrant
    try:
        import httpx

        resp = httpx.get("http://localhost:6333/collections", timeout=5)
        status["qdrant"] = resp.status_code == 200
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        status["qdrant"] = False

    # Reranker service - test with minimal rerank request
    try:
        import httpx

        url = os.environ.get("RERANKER_BASE_URL", "http://127.0.0.1:9005")
        resp = httpx.post(
            f"{url}/v1/rerank",
            json={
                "query": "test",
                "documents": ["test doc"],
                "model": "Qwen/Qwen3-Reranker-0.6B",
            },
            timeout=30,  # Qwen3 reranker timeout
        )
        status["reranker"] = resp.status_code == 200 and "results" in resp.json()
    except Exception as e:
        logger.warning(f"Reranker service not available: {e}")
        status["reranker"] = False

    # BGE-M3 embedding service - test with minimal embed request
    try:
        import httpx

        url = os.environ.get("BGE_M3_API_URL", "http://127.0.0.1:9000")
        resp = httpx.post(
            f"{url}/v1/embeddings",
            json={"input": ["test"], "model": "BAAI/bge-m3"},
            timeout=10,
        )
        status["bge_m3"] = resp.status_code == 200 and "data" in resp.json()
    except Exception as e:
        logger.warning(f"BGE-M3 service not available: {e}")
        status["bge_m3"] = False

    return status


def main():
    parser = argparse.ArgumentParser(description="Capture Phase 0 baseline metrics")
    parser.add_argument(
        "--output",
        "-o",
        default="reports/baseline_metrics.json",
        help="Output path for baseline metrics JSON",
    )
    parser.add_argument(
        "--query-set",
        default="tests/fixtures/baseline_query_set.yaml",
        help="Path to baseline query set YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check connectivity, don't run full measurements",
    )
    parser.add_argument(
        "--skip-reranker", action="store_true", help="Skip reranker latency measurement"
    )
    parser.add_argument(
        "--skip-quality", action="store_true", help="Skip quality metrics measurement"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Initialize report
    commit, branch = get_git_info()
    report = BaselineReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        git_commit=commit,
        git_branch=branch,
        query_set_path=args.query_set,
    )

    # Check connectivity
    logger.info("Checking service connectivity...")
    connectivity = check_connectivity()
    for service, ok in connectivity.items():
        status = "OK" if ok else "FAILED"
        logger.info(f"  {service}: {status}")

    if args.dry_run:
        logger.info("Dry run complete - connectivity check only")
        print(json.dumps(connectivity, indent=2))
        return 0 if all(connectivity.values()) else 1

    # Load query set
    if Path(args.query_set).exists():
        queries = load_query_set(args.query_set)
        report.query_count = len(queries)
        logger.info(f"Loaded {len(queries)} queries from {args.query_set}")
    else:
        queries = []
        report.warnings.append(f"Query set not found: {args.query_set}")
        logger.warning(f"Query set not found: {args.query_set}")

    # Collect graph statistics
    logger.info("Collecting graph statistics...")
    report.graph = collect_graph_stats()
    if report.graph:
        logger.info(f"  Total nodes: {report.graph.total_nodes}")
        logger.info(f"  Total relationships: {report.graph.total_relationships}")
        logger.info(f"  Cross-doc edges: {report.graph.cross_doc_edge_count}")
        logger.info(f"  CONTAINS_STEP edges: {report.graph.contains_step_count}")
    else:
        report.errors.append("Failed to collect graph statistics")

    # Measure reranker latency
    if not args.skip_reranker and connectivity.get("reranker"):
        logger.info("Measuring reranker latency...")
        report.reranker_latency = measure_reranker_latency()
        if report.reranker_latency:
            logger.info(f"  P50: {report.reranker_latency.p50:.1f}ms")
            logger.info(f"  P95: {report.reranker_latency.p95:.1f}ms")
            logger.info(f"  P99: {report.reranker_latency.p99:.1f}ms")
        else:
            report.warnings.append("Reranker latency measurement failed")
    elif args.skip_reranker:
        logger.info("Skipping reranker latency (--skip-reranker)")
    else:
        report.warnings.append("Reranker service not available")

    # Measure retrieval quality
    if (
        not args.skip_quality
        and queries
        and report.graph
        and report.graph.chunk_count > 0
    ):
        logger.info("Measuring retrieval quality...")
        report.quality, report.retrieval_latency = measure_retrieval_quality(queries)
        if report.quality:
            logger.info(f"  MRR: {report.quality.mrr:.4f}")
            logger.info(f"  NDCG@10: {report.quality.ndcg_at_10:.4f}")
            logger.info(f"  Precision@5: {report.quality.precision_at_5:.4f}")
        else:
            report.warnings.append("Quality measurement failed")
        if report.retrieval_latency:
            logger.info(f"  Retrieval P95: {report.retrieval_latency.p95:.1f}ms")
    elif args.skip_quality:
        logger.info("Skipping quality metrics (--skip-quality)")
    elif not queries:
        report.warnings.append("No queries available for quality measurement")
    else:
        report.warnings.append("No chunks in database - skipping quality metrics")

    # Convert to dict for JSON serialization
    def to_dict(obj):
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        return obj

    report_dict = to_dict(report)

    # Write report
    with open(args.output, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Baseline metrics written to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 0 BASELINE CAPTURE SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Git: {branch}@{commit}")
    print()

    if report.graph:
        print("Graph Statistics:")
        print(f"  Documents: {report.graph.document_count}")
        print(f"  Chunks: {report.graph.chunk_count}")
        print(f"  Entities: {report.graph.entity_count}")
        print(f"  Cross-doc edges: {report.graph.cross_doc_edge_count} (expected: 0)")
        print(f"  CONTAINS_STEP: {report.graph.contains_step_count} (expected: 0)")
        print(f"  REFERENCES: {report.graph.references_count} (Phase 3 gate: >= 10)")
        print()

    if report.reranker_latency:
        print("Reranker Latency (k=50):")
        print(f"  P50: {report.reranker_latency.p50:.1f}ms")
        print(f"  P95: {report.reranker_latency.p95:.1f}ms (target after fix: <100ms)")
        print(f"  P99: {report.reranker_latency.p99:.1f}ms")
        print()

    if report.quality:
        print("Retrieval Quality:")
        print(f"  MRR: {report.quality.mrr:.4f}")
        print(f"  NDCG@10: {report.quality.ndcg_at_10:.4f}")
        print(f"  Queries evaluated: {report.quality.queries_evaluated}")
        print()

    if report.errors:
        print("ERRORS:")
        for e in report.errors:
            print(f"  - {e}")
        print()

    if report.warnings:
        print("WARNINGS:")
        for w in report.warnings:
            print(f"  - {w}")
        print()

    print("=" * 60)

    # Exit with error if critical failures
    if report.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

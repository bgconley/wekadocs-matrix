"""
Phase 7E-4: Metrics Collector
Comprehensive metrics for ingestion, retrieval, and chunk quality

Reference: Canonical Spec L4849-4892, L1449-1462, L3608
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    """
    Comprehensive chunk-level metrics.

    Reference: Canonical Spec L4862-4892
    """

    # Basic stats
    total_chunks: int
    total_tokens: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    median_tokens: float

    # Percentiles (for SLO monitoring)
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    # Quality metrics
    under_min_count: int = 0  # Chunks < TARGET_MIN (e.g., 200 tokens)
    over_max_count: int = 0  # Chunks > ABSOLUTE_MAX (e.g., 7900 tokens)

    # Decision path metrics (if available)
    decisions: Optional[Dict[str, int]] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        result = {
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "mean_tokens": round(self.mean_tokens, 2),
            "median_tokens": round(self.median_tokens, 2),
            "percentiles": {
                "p50": round(self.p50, 2),
                "p75": round(self.p75, 2),
                "p90": round(self.p90, 2),
                "p95": round(self.p95, 2),
                "p99": round(self.p99, 2),
            },
            "quality": {
                "under_min_count": self.under_min_count,
                "over_max_count": self.over_max_count,
                "oversized_rate": round(
                    self.over_max_count / max(self.total_chunks, 1), 6
                ),
            },
            "timestamp": self.timestamp.isoformat(),
        }

        if self.decisions:
            result["decisions"] = self.decisions

        return result


@dataclass
class RetrievalMetrics:
    """
    Retrieval performance metrics.

    Reference: Canonical Spec L5071-5083
    """

    # Latency metrics (milliseconds)
    p50_latency: float
    p95_latency: float
    p99_latency: float
    mean_latency: float

    # Result metrics
    avg_chunks_returned: float
    expansion_rate: float  # Fraction of queries that triggered expansion

    # Fusion method usage
    fusion_method_counts: Dict[str, int] = field(default_factory=dict)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "latency": {
                "p50_ms": round(self.p50_latency, 2),
                "p95_ms": round(self.p95_latency, 2),
                "p99_ms": round(self.p99_latency, 2),
                "mean_ms": round(self.mean_latency, 2),
            },
            "results": {
                "avg_chunks": round(self.avg_chunks_returned, 2),
                "expansion_rate": round(self.expansion_rate, 4),
            },
            "fusion_methods": self.fusion_method_counts,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IngestionMetrics:
    """
    Ingestion performance metrics.

    Reference: Canonical Spec L1449-1462
    """

    # Per document metrics
    document_id: str
    duration_seconds: float
    chunks_created: int
    total_tokens: int

    # Chunk distribution
    chunk_metrics: ChunkMetrics

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "duration_seconds": round(self.duration_seconds, 2),
            "chunks_created": self.chunks_created,
            "total_tokens": self.total_tokens,
            "chunk_distribution": self.chunk_metrics.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


class MetricsCollector:
    """
    Comprehensive metrics collection for monitoring.

    Usage:
        collector = MetricsCollector()
        chunk_metrics = collector.collect_chunk_metrics(chunks)
        print(f"P95 tokens: {chunk_metrics.p95}")
    """

    def __init__(self, target_min: int = 200, absolute_max: int = 7900):
        """
        Initialize metrics collector.

        Args:
            target_min: Target minimum tokens per chunk
            absolute_max: Absolute maximum tokens per chunk (hard limit)
        """
        self.target_min = target_min
        self.absolute_max = absolute_max

    def collect_chunk_metrics(
        self,
        chunks: List[Dict[str, any]],
        decisions: Optional[Dict[str, int]] = None,
    ) -> ChunkMetrics:
        """
        Collect comprehensive chunk metrics.

        Args:
            chunks: List of chunk dictionaries with 'token_count' field
            decisions: Optional decision path metrics from combiner

        Returns:
            ChunkMetrics with distribution and quality stats
        """
        if not chunks:
            # Return empty metrics
            return ChunkMetrics(
                total_chunks=0,
                total_tokens=0,
                min_tokens=0,
                max_tokens=0,
                mean_tokens=0.0,
                median_tokens=0.0,
                p50=0.0,
                p75=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                under_min_count=0,
                over_max_count=0,
                decisions=decisions,
            )

        # Extract token counts
        token_counts = [chunk.get("token_count", 0) for chunk in chunks]
        token_array = np.array(token_counts)

        # Basic stats
        total_chunks = len(chunks)
        total_tokens = int(token_array.sum())
        min_tokens = int(token_array.min())
        max_tokens = int(token_array.max())
        mean_tokens = float(token_array.mean())
        median_tokens = float(np.median(token_array))

        # Percentiles
        p50 = float(np.percentile(token_array, 50))
        p75 = float(np.percentile(token_array, 75))
        p90 = float(np.percentile(token_array, 90))
        p95 = float(np.percentile(token_array, 95))
        p99 = float(np.percentile(token_array, 99))

        # Quality metrics
        under_min_count = int(np.sum(token_array < self.target_min))
        over_max_count = int(np.sum(token_array > self.absolute_max))

        return ChunkMetrics(
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            mean_tokens=mean_tokens,
            median_tokens=median_tokens,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
            under_min_count=under_min_count,
            over_max_count=over_max_count,
            decisions=decisions,
        )

    def collect_retrieval_metrics(
        self, retrievals: List[Dict[str, any]]
    ) -> RetrievalMetrics:
        """
        Collect retrieval performance metrics.

        Args:
            retrievals: List of retrieval result dictionaries with:
                - latency_ms: retrieval latency in milliseconds
                - chunks_returned: number of chunks returned
                - expanded: boolean indicating if expansion occurred
                - fusion_method: fusion method used ('rrf' or 'weighted')

        Returns:
            RetrievalMetrics with latency and result stats
        """
        if not retrievals:
            return RetrievalMetrics(
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                mean_latency=0.0,
                avg_chunks_returned=0.0,
                expansion_rate=0.0,
                fusion_method_counts={},
            )

        # Extract latencies
        latencies = np.array([r.get("latency_ms", 0.0) for r in retrievals])

        # Calculate percentiles
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))
        mean_latency = float(latencies.mean())

        # Result metrics
        chunks_returned = [r.get("chunks_returned", 0) for r in retrievals]
        avg_chunks_returned = float(np.mean(chunks_returned))

        # Expansion rate
        expanded_count = sum(1 for r in retrievals if r.get("expanded", False))
        expansion_rate = expanded_count / len(retrievals)

        # Fusion method usage
        fusion_method_counts = {}
        for r in retrievals:
            method = r.get("fusion_method", "unknown")
            fusion_method_counts[method] = fusion_method_counts.get(method, 0) + 1

        return RetrievalMetrics(
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            mean_latency=mean_latency,
            avg_chunks_returned=avg_chunks_returned,
            expansion_rate=expansion_rate,
            fusion_method_counts=fusion_method_counts,
        )

    def collect_ingestion_metrics(
        self,
        document_id: str,
        chunks: List[Dict[str, any]],
        duration_seconds: float,
        decisions: Optional[Dict[str, int]] = None,
    ) -> IngestionMetrics:
        """
        Collect ingestion metrics for a document.

        Args:
            document_id: Document identifier
            chunks: List of created chunks
            duration_seconds: Total ingestion duration
            decisions: Optional decision path metrics from combiner

        Returns:
            IngestionMetrics with timing and chunk distribution
        """
        chunk_metrics = self.collect_chunk_metrics(chunks, decisions)

        return IngestionMetrics(
            document_id=document_id,
            duration_seconds=duration_seconds,
            chunks_created=len(chunks),
            total_tokens=chunk_metrics.total_tokens,
            chunk_metrics=chunk_metrics,
        )

    def compute_slo_metrics(
        self,
        chunk_metrics: ChunkMetrics,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        integrity_checks: Optional[Dict[str, any]] = None,
    ) -> Dict[str, float]:
        """
        Compute SLO metrics from collected data.

        Args:
            chunk_metrics: Chunk distribution metrics
            retrieval_metrics: Optional retrieval performance metrics
            integrity_checks: Optional integrity check results

        Returns:
            Dictionary of SLO metric name -> value
        """
        slo_metrics = {}

        # Oversized chunk rate (ZERO TOLERANCE)
        if chunk_metrics.total_chunks > 0:
            oversized_rate = chunk_metrics.over_max_count / chunk_metrics.total_chunks
            slo_metrics["oversized_chunk_rate"] = oversized_rate
        else:
            slo_metrics["oversized_chunk_rate"] = 0.0

        # Retrieval p95 latency
        if retrieval_metrics:
            slo_metrics["retrieval_p95_latency"] = retrieval_metrics.p95_latency
            slo_metrics["expansion_rate"] = retrieval_metrics.expansion_rate

        # Integrity failure rate (ZERO TOLERANCE)
        if integrity_checks:
            total_checks = integrity_checks.get("total", 0)
            failures = integrity_checks.get("failures", 0)
            if total_checks > 0:
                slo_metrics["integrity_failure_rate"] = failures / total_checks
            else:
                slo_metrics["integrity_failure_rate"] = 0.0

        return slo_metrics


class MetricsAggregator:
    """
    Aggregate metrics over time windows for dashboards.

    Usage:
        aggregator = MetricsAggregator()
        aggregator.record_retrieval(latency_ms=450, chunks=5, expanded=True)
        metrics = aggregator.get_window_metrics(window_seconds=300)  # Last 5 minutes
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics aggregator.

        Args:
            max_history: Maximum number of records to keep in memory
        """
        self.max_history = max_history
        self.retrieval_history: List[Dict[str, any]] = []
        self.ingestion_history: List[IngestionMetrics] = []

    def record_retrieval(
        self,
        latency_ms: float,
        chunks_returned: int,
        expanded: bool,
        fusion_method: str,
    ):
        """Record a retrieval event."""
        self.retrieval_history.append(
            {
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "chunks_returned": chunks_returned,
                "expanded": expanded,
                "fusion_method": fusion_method,
            }
        )

        # Trim history if needed
        if len(self.retrieval_history) > self.max_history:
            self.retrieval_history = self.retrieval_history[-self.max_history :]

    def record_ingestion(self, metrics: IngestionMetrics):
        """Record an ingestion event."""
        self.ingestion_history.append(metrics)

        # Trim history if needed
        if len(self.ingestion_history) > self.max_history:
            self.ingestion_history = self.ingestion_history[-self.max_history :]

    def get_window_metrics(
        self, window_seconds: int = 300
    ) -> Dict[str, RetrievalMetrics]:
        """
        Get aggregated metrics for a time window.

        Args:
            window_seconds: Time window in seconds (default 5 minutes)

        Returns:
            Dictionary with retrieval metrics for the window
        """
        cutoff_time = time.time() - window_seconds

        # Filter to window
        window_retrievals = [
            r for r in self.retrieval_history if r["timestamp"] >= cutoff_time
        ]

        if not window_retrievals:
            return {
                "retrieval": RetrievalMetrics(
                    p50_latency=0.0,
                    p95_latency=0.0,
                    p99_latency=0.0,
                    mean_latency=0.0,
                    avg_chunks_returned=0.0,
                    expansion_rate=0.0,
                    fusion_method_counts={},
                )
            }

        collector = MetricsCollector()
        retrieval_metrics = collector.collect_retrieval_metrics(window_retrievals)

        return {"retrieval": retrieval_metrics}


# Singleton instance for application-wide metrics
_global_aggregator: Optional[MetricsAggregator] = None


def get_metrics_aggregator() -> MetricsAggregator:
    """Get or create global metrics aggregator."""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = MetricsAggregator()
    return _global_aggregator

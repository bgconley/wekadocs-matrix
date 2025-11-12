# Implements Phase 5, Task 5.2 (Monitoring & observability)
# See: /docs/spec.md §7 (Observability & SLOs)
# Prometheus metrics for WekaDocs GraphRAG MCP

from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..config import Settings
from .logging import get_logger

logger = get_logger(__name__)

# ===== Request metrics =====
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ===== MCP tool metrics =====
mcp_tool_calls_total = Counter(
    "mcp_tool_calls_total",
    "Total MCP tool calls",
    ["tool_name", "status"],
)

mcp_tool_duration_seconds = Histogram(
    "mcp_tool_duration_seconds",
    "MCP tool execution duration in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# ===== Query processing metrics =====
cypher_query_duration_seconds = Histogram(
    "cypher_query_duration_seconds",
    "Cypher query execution duration in seconds",
    ["template_name", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

cypher_queries_total = Counter(
    "cypher_queries_total",
    "Total Cypher queries executed",
    ["template_name", "status"],
)

cypher_validation_failures_total = Counter(
    "cypher_validation_failures_total",
    "Total Cypher validation failures",
    ["reason"],
)

# ===== Cache metrics =====
cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    ["operation", "layer", "result"],
)

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Cache hit rate (rolling average)",
    ["layer"],
)

cache_size_bytes = Gauge(
    "cache_size_bytes",
    "Current cache size in bytes",
    ["layer"],
)

# ===== Pre-Phase 7 (G1): Embedding provider metrics =====
embedding_request_total = Counter(
    "embedding_request_total",
    "Total embedding requests",
    ["model_id", "operation"],  # operation: documents, query
)

embedding_error_total = Counter(
    "embedding_error_total",
    "Total embedding errors",
    ["model_id", "error_type"],
)

embedding_latency_ms = Histogram(
    "embedding_latency_ms",
    "Embedding generation latency in milliseconds",
    ["model_id", "operation"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

# ===== Pre-Phase 7 (G1): Qdrant metrics =====
qdrant_upsert_total = Counter(
    "qdrant_upsert_total",
    "Total Qdrant upsert operations",
    ["collection_name", "status"],
)

qdrant_search_total = Counter(
    "qdrant_search_total",
    "Total Qdrant search operations",
    ["collection_name", "status"],
)

qdrant_operation_latency_ms = Histogram(
    "qdrant_operation_latency_ms",
    "Qdrant operation latency in milliseconds",
    ["collection_name", "operation"],  # operation: upsert, search
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

qdrant_schema_mismatch_total = Counter(
    "qdrant_schema_mismatch_total",
    "Total Qdrant schema mismatch detections (e.g., unnamed vector query against multi-vector collection).",
    ["collection_name"],
)

# ===== Pre-Phase 7 (G1): Ranking metrics =====
ranking_latency_ms = Histogram(
    "ranking_latency_ms",
    "Ranking operation latency in milliseconds",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500),
)

ranking_candidates_total = Histogram(
    "ranking_candidates_total",
    "Number of candidates ranked",
    buckets=(1, 5, 10, 20, 50, 100, 200),
)

ranking_missing_vector_score_total = Counter(
    "ranking_missing_vector_score_total",
    "Total RRF ranking entries missing vector similarity metadata",
    ["fallback"],
)

ranking_vector_score_distribution = Histogram(
    "ranking_vector_score_distribution",
    "Distribution of raw vector similarity scores seen during ranking",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

ranking_semantic_score_distribution = Histogram(
    "ranking_semantic_score_distribution",
    "Distribution of normalized semantic confidence scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ===== Ingestion: Semantic enrichment metrics =====
semantic_enrichment_total = Counter(
    "semantic_enrichment_total",
    "Total semantic enrichment attempts grouped by provider and status.",
    ["provider", "status"],  # status: success, error, skipped
)

semantic_enrichment_latency_ms = Histogram(
    "semantic_enrichment_latency_ms",
    "Latency of semantic enrichment in milliseconds.",
    ["provider"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

# ===== Phase 7C: Reranking metrics =====
rerank_request_total = Counter(
    "rerank_request_total",
    "Total reranking requests",
    ["model_id", "status"],
)

rerank_error_total = Counter(
    "rerank_error_total",
    "Total reranking errors",
    ["model_id", "error_type"],
)

rerank_latency_ms = Histogram(
    "rerank_latency_ms",
    "Reranking latency in milliseconds",
    ["model_id"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

# ===== Pre-Phase 7 (G1): Response builder metrics =====
response_builder_latency_ms = Histogram(
    "response_builder_latency_ms",
    "Response builder latency in milliseconds",
    ["verbosity"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000),
)

response_builder_evidence_count = Histogram(
    "response_builder_evidence_count",
    "Number of evidence items in response",
    ["verbosity"],
    buckets=(1, 3, 5, 10, 20),
)

# ===== Vector search metrics =====
vector_search_duration_seconds = Histogram(
    "vector_search_duration_seconds",
    "Vector search duration in seconds",
    ["store"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

vector_search_total = Counter(
    "vector_search_total",
    "Total vector searches",
    ["store", "status"],
)

# ===== Hybrid search metrics =====
hybrid_search_duration_seconds = Histogram(
    "hybrid_search_duration_seconds",
    "Hybrid search (vector + graph) duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

graph_expansion_duration_seconds = Histogram(
    "graph_expansion_duration_seconds",
    "Graph expansion duration in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# ===== Enhanced Response Features (E5) =====
mcp_search_verbosity_total = Counter(
    "mcp_search_verbosity_total",
    "Search requests by verbosity level",
    ["verbosity"],
)

mcp_search_response_size_bytes = Histogram(
    "mcp_search_response_size_bytes",
    "Response size distribution by verbosity",
    ["verbosity"],
    buckets=(1024, 5120, 10240, 20480, 40960, 65536),  # 1KB to 64KB
)

mcp_traverse_depth_total = Counter(
    "mcp_traverse_depth_total",
    "Traversal requests by depth",
    ["depth"],
)

mcp_traverse_nodes_found = Histogram(
    "mcp_traverse_nodes_found",
    "Number of nodes found in traversal",
    buckets=(1, 5, 10, 20, 50, 100),
)

# ===== Ingestion metrics =====
ingestion_queue_size = Gauge(
    "ingestion_queue_size",
    "Current ingestion queue size",
)

ingestion_queue_lag_seconds = Gauge(
    "ingestion_queue_lag_seconds",
    "Ingestion queue lag (oldest item age)",
)

ingestion_documents_total = Counter(
    "ingestion_documents_total",
    "Total documents ingested",
    ["status"],
)

ingestion_duration_seconds = Histogram(
    "ingestion_duration_seconds",
    "Document ingestion duration in seconds",
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# ===== Phase 7E-4: Chunk quality metrics =====
chunk_token_distribution = Histogram(
    "chunk_token_distribution",
    "Token count distribution for chunks",
    buckets=(50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7900),
)

chunks_created_total = Counter(
    "chunks_created_total",
    "Total chunks created during ingestion",
    ["document_id"],
)

chunks_oversized_total = Counter(
    "chunks_oversized_total",
    "Total oversized chunks (ZERO tolerance SLO)",
    ["document_id"],
)

chunk_quality_score = Histogram(
    "chunk_quality_score",
    "Chunk quality scores (token count distribution)",
    ["quality_band"],  # under_min, optimal, over_max
    buckets=(0, 200, 500, 1000, 2000, 5000, 7900, 10000),
)

# ===== Phase 7E-4: Integrity check metrics =====
integrity_checks_total = Counter(
    "integrity_checks_total",
    "Total integrity checks performed",
    ["check_type"],  # sha256, dimension, schema
)

integrity_failures_total = Counter(
    "integrity_failures_total",
    "Total integrity check failures (ZERO tolerance SLO)",
    ["check_type", "failure_reason"],
)

# ===== Phase 7E-4: Retrieval expansion metrics =====
retrieval_expansion_total = Counter(
    "retrieval_expansion_total",
    "Total retrieval expansion events",
    ["expansion_reason"],  # query_long, scores_close, forced, disabled
)

retrieval_expansion_chunks_added = Histogram(
    "retrieval_expansion_chunks_added",
    "Number of chunks added via expansion",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)

retrieval_expansion_rate_current = Gauge(
    "retrieval_expansion_rate_current",
    "Current expansion rate (rolling window)",
)

# ===== Reconciliation metrics =====
reconciliation_drift_percentage = Gauge(
    "reconciliation_drift_percentage",
    "Reconciliation drift percentage (graph vs vector)",
)

reconciliation_duration_seconds = Histogram(
    "reconciliation_duration_seconds",
    "Reconciliation run duration in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

reconciliation_repairs_total = Counter(
    "reconciliation_repairs_total",
    "Total drift repairs performed",
)

# ===== Connection pool metrics =====
connection_pool_active = Gauge(
    "connection_pool_active",
    "Active connections in pool",
    ["pool_name"],
)

connection_pool_idle = Gauge(
    "connection_pool_idle",
    "Idle connections in pool",
    ["pool_name"],
)

# ===== Service info =====
service_info = Info(
    "wekadocs_mcp",
    "WekaDocs GraphRAG MCP service information",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP request metrics for Prometheus.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        # Time the request
        with http_request_duration_seconds.labels(
            method=method, endpoint=endpoint
        ).time():
            response = await call_next(request)

        # Record request
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()

        return response


def setup_metrics(settings: Settings) -> None:
    """
    Setup Prometheus metrics collection.

    Args:
        settings: Application settings
    """
    logger.info("Setting up Prometheus metrics")

    # Set service info with available settings
    service_info.info(
        {
            "version": "0.1.0",
            "environment": settings.env,
            "service_name": settings.otel_service_name,
        }
    )

    logger.info("Prometheus metrics enabled")


def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus exposition format.

    Returns:
        Metrics as bytes
    """
    return generate_latest()

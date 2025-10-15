# Implements Phase 5, Task 5.2 (Monitoring & observability)
# OpenTelemetry trace exemplars for Prometheus metrics
# See: /docs/spec.md §7 (Observability & SLOs)

from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from .logging import get_logger
from .metrics import (
    cypher_queries_total,
    cypher_query_duration_seconds,
    graph_expansion_duration_seconds,
    hybrid_search_duration_seconds,
    mcp_tool_calls_total,
    mcp_tool_duration_seconds,
    vector_search_duration_seconds,
)

logger = get_logger(__name__)


def get_trace_context() -> Dict[str, str]:
    """
    Get current trace context for exemplar linking.

    Returns:
        Dictionary with trace_id and span_id
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return {}


@contextmanager
def trace_mcp_tool(tool_name: str, arguments: Dict[str, Any]):
    """
    Context manager to trace MCP tool execution with metrics and exemplars.

    Args:
        tool_name: Name of the MCP tool
        arguments: Tool arguments

    Yields:
        Span object
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        f"mcp.tool.{tool_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            "mcp.tool.name": tool_name,
            "mcp.tool.args": str(arguments),
        },
    ) as span:
        with mcp_tool_duration_seconds.labels(tool_name=tool_name).time():
            try:
                yield span
                mcp_tool_calls_total.labels(tool_name=tool_name, status="success").inc()
                span.set_attribute("mcp.tool.status", "success")
            except Exception as e:
                mcp_tool_calls_total.labels(tool_name=tool_name, status="error").inc()
                span.set_attribute("mcp.tool.status", "error")
                span.set_attribute("mcp.tool.error", str(e))
                span.record_exception(e)
                raise


@contextmanager
def trace_cypher_query(
    template_name: str,
    query: str,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Context manager to trace Cypher query execution with metrics and exemplars.

    Args:
        template_name: Name of the query template
        query: Cypher query string
        params: Query parameters

    Yields:
        Span object
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        f"cypher.query.{template_name}",
        kind=SpanKind.CLIENT,
        attributes={
            "db.system": "neo4j",
            "db.statement": query[:500],  # Truncate for safety
            "db.operation": "query",
            "cypher.template": template_name,
        },
    ) as span:
        # Add exemplar linking
        trace_ctx = get_trace_context()

        with cypher_query_duration_seconds.labels(
            template_name=template_name, status="pending"
        ).time():
            try:
                yield span

                # Record success
                cypher_queries_total.labels(
                    template_name=template_name, status="success"
                ).inc(exemplar=trace_ctx)

                span.set_attribute("cypher.status", "success")

            except Exception as e:
                # Record failure
                cypher_queries_total.labels(
                    template_name=template_name, status="error"
                ).inc(exemplar=trace_ctx)

                span.set_attribute("cypher.status", "error")
                span.set_attribute("cypher.error", str(e))
                span.record_exception(e)
                raise


@contextmanager
def trace_vector_search(
    store: str, query_vector: Optional[list] = None, top_k: int = 20
):
    """
    Context manager to trace vector search with metrics and exemplars.

    Args:
        store: Vector store name (qdrant, neo4j)
        query_vector: Query vector (optional, for span attributes)
        top_k: Number of results to retrieve

    Yields:
        Span object
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        f"vector.search.{store}",
        kind=SpanKind.CLIENT,
        attributes={
            "vector.store": store,
            "vector.top_k": top_k,
            "vector.dims": len(query_vector) if query_vector else 0,
        },
    ) as span:
        trace_ctx = get_trace_context()

        with vector_search_duration_seconds.labels(store=store).time():
            try:
                yield span

                vector_search_total.labels(store=store, status="success").inc(
                    exemplar=trace_ctx
                )
                span.set_attribute("vector.status", "success")

            except Exception as e:
                vector_search_total.labels(store=store, status="error").inc(
                    exemplar=trace_ctx
                )
                span.set_attribute("vector.status", "error")
                span.set_attribute("vector.error", str(e))
                span.record_exception(e)
                raise


@contextmanager
def trace_hybrid_search(query: str, filters: Optional[Dict] = None):
    """
    Context manager to trace hybrid search (vector + graph) with exemplars.

    Args:
        query: Search query string
        filters: Optional filters

    Yields:
        Span object
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        "hybrid.search",
        kind=SpanKind.INTERNAL,
        attributes={
            "search.query": query[:200],  # Truncate for safety
            "search.filters": str(filters) if filters else "",
        },
    ) as span:
        trace_ctx = get_trace_context()

        with hybrid_search_duration_seconds.time():
            try:
                yield span
                span.set_attribute("search.status", "success")

                # Log slow searches (P95 threshold: 500ms)
                # This will be available as exemplar in Grafana
                if span.end_time and span.start_time:
                    duration_ms = (span.end_time - span.start_time) / 1_000_000
                    if duration_ms > 500:
                        logger.warning(
                            "Slow hybrid search detected",
                            duration_ms=duration_ms,
                            query=query[:50],
                            trace_id=trace_ctx.get("trace_id"),
                            span_id=trace_ctx.get("span_id"),
                        )

            except Exception as e:
                span.set_attribute("search.status", "error")
                span.set_attribute("search.error", str(e))
                span.record_exception(e)
                raise


@contextmanager
def trace_graph_expansion(start_node_id: str, max_depth: int, rel_types: list):
    """
    Context manager to trace graph expansion with metrics.

    Args:
        start_node_id: Starting node ID
        max_depth: Maximum traversal depth
        rel_types: Relationship types to follow

    Yields:
        Span object
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        "graph.expansion",
        kind=SpanKind.INTERNAL,
        attributes={
            "graph.start_node": start_node_id,
            "graph.max_depth": max_depth,
            "graph.rel_types": ",".join(rel_types),
        },
    ) as span:
        with graph_expansion_duration_seconds.time():
            try:
                yield span
                span.set_attribute("graph.status", "success")
            except Exception as e:
                span.set_attribute("graph.status", "error")
                span.set_attribute("graph.error", str(e))
                span.record_exception(e)
                raise


# Import vector_search_total (add to metrics.py if not present)
try:
    from .metrics import vector_search_total
except ImportError:
    logger.warning("vector_search_total metric not found in metrics module")

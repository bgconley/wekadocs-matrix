# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §2 (Architecture), §9 (Interfaces)
# See: /docs/implementation-plan.md → Task 1.2 DoD & Tests
# FastAPI MCP server with health, metrics, and MCP protocol endpoints

import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request

from src.shared import close_connections, init_config, initialize_connections
from src.shared.observability import (
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
    setup_tracing,
)
from src.shared.observability.metrics import (
    PrometheusMiddleware,
    get_metrics,
    setup_metrics,
)

from .models import (
    HealthResponse,
    MCPInitializeRequest,
    MCPInitializeResponse,
    MCPTool,
    MCPToolCallRequest,
    MCPToolCallResponse,
    MCPToolsListResponse,
    MetricsResponse,
    ReadinessResponse,
)

# Initialize config and logging
config, settings = init_config()
setup_logging(config.app.log_level)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.app.name,
    version=config.app.version,
    description="WekaDocs GraphRAG MCP Server",
)

# Setup OpenTelemetry tracing
setup_tracing(app, settings)

# Setup Prometheus metrics
setup_metrics(settings)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Metrics storage (in-memory for Phase 1; will be Redis/Prometheus later)
metrics_store = {
    "requests_total": 0,
    "requests_by_endpoint": {},
    "errors_total": 0,
    "latencies": [],
}


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Add correlation ID to request context"""
    # Get or generate correlation ID
    corr_id = request.headers.get("X-Correlation-ID")
    if not corr_id:
        corr_id = get_correlation_id()
    else:
        set_correlation_id(corr_id)

    # Add to response headers
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()

    try:
        response = await call_next(request)
        latency = time.time() - start_time

        # Update metrics
        metrics_store["requests_total"] += 1
        endpoint = request.url.path
        metrics_store["requests_by_endpoint"][endpoint] = (
            metrics_store["requests_by_endpoint"].get(endpoint, 0) + 1
        )
        metrics_store["latencies"].append(latency)

        # Keep last 1000 latencies
        if len(metrics_store["latencies"]) > 1000:
            metrics_store["latencies"] = metrics_store["latencies"][-1000:]

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency * 1000, 2),
        )

        return response
    except Exception as e:
        metrics_store["errors_total"] += 1
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
        )
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("Starting MCP server", version=config.app.version)
    try:
        await initialize_connections()
        logger.info("MCP server started successfully")
    except Exception as e:
        logger.error("Failed to start MCP server", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    logger.info("Shutting down MCP server")
    try:
        await close_connections()
        logger.info("MCP server shut down successfully")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Health endpoints


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=config.app.version,
    )


@app.get("/ready", response_model=ReadinessResponse)
async def readiness():
    """Readiness check endpoint - verifies all dependencies"""
    from src.shared.connections import get_connection_manager

    services = {}
    ready = True

    try:
        manager = get_connection_manager()

        # Check Neo4j
        try:
            driver = manager.get_neo4j_driver()
            driver.verify_connectivity()
            services["neo4j"] = True
        except Exception as e:
            logger.warning("Neo4j not ready", error=str(e))
            services["neo4j"] = False
            ready = False

        # Check Qdrant
        try:
            manager.get_qdrant_client()
            # Simple health check - will throw if not reachable
            services["qdrant"] = True
        except Exception as e:
            logger.warning("Qdrant not ready", error=str(e))
            services["qdrant"] = False
            ready = False

        # Check Redis
        try:
            redis_client = await manager.get_redis_client()
            await redis_client.ping()
            services["redis"] = True
        except Exception as e:
            logger.warning("Redis not ready", error=str(e))
            services["redis"] = False
            ready = False

    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        services["error"] = str(e)
        ready = False

    return ReadinessResponse(
        ready=ready,
        services=services,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from starlette.responses import Response

    return Response(content=get_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json():
    """JSON metrics endpoint (legacy)"""
    latencies = metrics_store["latencies"]

    def percentile(data, p):
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[f]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    return MetricsResponse(
        requests_total=metrics_store["requests_total"],
        requests_by_endpoint=metrics_store["requests_by_endpoint"],
        errors_total=metrics_store["errors_total"],
        latency_p50=round(percentile(latencies, 0.50) * 1000, 2),
        latency_p95=round(percentile(latencies, 0.95) * 1000, 2),
        latency_p99=round(percentile(latencies, 0.99) * 1000, 2),
    )


# MCP Protocol endpoints


@app.post("/mcp/initialize", response_model=MCPInitializeResponse)
async def mcp_initialize(request: MCPInitializeRequest):
    """Initialize MCP connection"""
    logger.info("MCP initialize request", client_info=request.client_info)

    return MCPInitializeResponse(
        protocol_version="1.0",
        server_info={
            "name": config.app.name,
            "version": config.app.version,
        },
        capabilities={
            "tools": True,
            "prompts": False,
            "resources": False,
        },
    )


@app.get("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list():
    """List available MCP tools"""
    # Tool definitions (Phase 2+ will populate these properly)
    tools = [
        MCPTool(
            name="search_documentation",
            description="Search Weka documentation using hybrid retrieval",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        ),
        MCPTool(
            name="traverse_relationships",
            description="Traverse graph relationships from a starting node",
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Starting node ID"},
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relationship types to follow",
                    },
                    "max_depth": {"type": "integer", "default": 2},
                },
                "required": ["node_id"],
            },
        ),
    ]

    logger.info("MCP tools list request", tool_count=len(tools))
    return MCPToolsListResponse(tools=tools)


@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def mcp_tools_call(request: MCPToolCallRequest):
    """Execute an MCP tool"""
    from src.mcp_server.query_service import get_query_service
    from src.shared.observability.exemplars import trace_mcp_tool
    from src.shared.observability.metrics import (
        mcp_tool_calls_total,
        mcp_tool_duration_seconds,
    )

    logger.info("MCP tool call request", tool=request.name, args=request.arguments)

    start_time = time.time()

    try:
        # Trace tool execution
        with trace_mcp_tool(request.name, request.arguments):
            # Execute search_documentation tool
            if request.name == "search_documentation":
                query = request.arguments.get("query", "")
                top_k = request.arguments.get("top_k", 20)

                if not query:
                    result = MCPToolCallResponse(
                        content=[
                            {
                                "type": "text",
                                "text": "Error: query parameter is required",
                            }
                        ],
                        is_error=True,
                    )
                else:
                    try:
                        # Execute search via query service
                        query_service = get_query_service()
                        response = query_service.search(
                            query=query,
                            top_k=top_k,
                            expand_graph=True,
                            find_paths=False,
                        )

                        # Build MCP response with Markdown + JSON
                        result = MCPToolCallResponse(
                            content=[
                                {
                                    "type": "text",
                                    "text": response.answer_markdown,
                                },
                                {
                                    "type": "json",
                                    "json": response.answer_json.to_dict(),
                                },
                            ],
                            is_error=False,
                        )

                    except Exception as e:
                        logger.error(f"Search failed: {e}", exc_info=True)
                        result = MCPToolCallResponse(
                            content=[
                                {
                                    "type": "text",
                                    "text": f"Search failed: {str(e)}",
                                }
                            ],
                            is_error=True,
                        )

            elif request.name == "traverse_relationships":
                result = MCPToolCallResponse(
                    content=[
                        {
                            "type": "text",
                            "text": f"Traverse tool will be implemented in a future phase. Node: {request.arguments.get('node_id', '')}",
                        }
                    ],
                    is_error=False,
                )
            else:
                result = MCPToolCallResponse(
                    content=[
                        {
                            "type": "text",
                            "text": f"Unknown tool: {request.name}",
                        }
                    ],
                    is_error=True,
                )

            # Record metrics
            duration = time.time() - start_time
            status = "error" if result.is_error else "success"
            mcp_tool_calls_total.labels(tool_name=request.name, status=status).inc()
            mcp_tool_duration_seconds.labels(tool_name=request.name).observe(duration)

            return result

    except Exception as e:
        duration = time.time() - start_time
        mcp_tool_calls_total.labels(tool_name=request.name, status="error").inc()
        mcp_tool_duration_seconds.labels(tool_name=request.name).observe(duration)
        logger.error("MCP tool call failed", tool=request.name, error=str(e))
        raise


if __name__ == "__main__":
    uvicorn.run(
        "src.mcp_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.app.log_level.lower(),
    )

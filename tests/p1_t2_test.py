# Phase 1, Task 1.2 Tests - MCP server foundation (NO MOCKS)
# See: /docs/implementation-plan.md → Task 1.2 DoD & Tests

import pytest


def test_mcp_initialize_endpoint(client):
    """Test /mcp/initialize endpoint"""
    response = client.post(
        "/mcp/initialize",
        json={
            "protocol_version": "1.0",
            "client_info": {"name": "test_client", "version": "1.0"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["protocol_version"] == "1.0"
    assert "server_info" in data
    assert "capabilities" in data
    assert data["capabilities"]["tools"] is True


def test_mcp_tools_list_endpoint(client):
    """Test /mcp/tools/list endpoint"""
    response = client.get("/mcp/tools/list")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert len(data["tools"]) > 0

    # Verify tool structure
    tool = data["tools"][0]
    assert "name" in tool
    assert "description" in tool
    assert "input_schema" in tool


def test_mcp_tools_call_endpoint(client):
    """Test /mcp/tools/call endpoint"""
    response = client.post(
        "/mcp/tools/call",
        json={
            "name": "search_documentation",
            "arguments": {"query": "test query"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "is_error" in data


def test_correlation_id_header(client):
    """Test correlation ID is added to response headers"""
    response = client.get("/health")
    assert "X-Correlation-ID" in response.headers
    correlation_id = response.headers["X-Correlation-ID"]
    assert len(correlation_id) > 0


def test_metrics_endpoint(client):
    """Test /metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_total" in data
    assert "requests_by_endpoint" in data
    assert "errors_total" in data
    assert "latency_p50" in data
    assert "latency_p95" in data
    assert "latency_p99" in data
    assert data["requests_total"] >= 0


def test_connection_pools_created(docker_services_running):
    """Test connection pools are created successfully"""
    from src.shared.connections import get_connection_manager

    manager = get_connection_manager()

    # Test Neo4j driver
    driver = manager.get_neo4j_driver()
    assert driver is not None
    driver.verify_connectivity()

    # Test Qdrant client
    qdrant = manager.get_qdrant_client()
    assert qdrant is not None


@pytest.mark.asyncio
async def test_async_connection_pools(docker_services_running):
    """Test async connection pools (Redis)"""
    from src.shared.connections import ConnectionManager

    # Create a fresh manager to avoid event loop conflicts
    manager = ConnectionManager()

    # Test Redis client
    redis_client = await manager.get_redis_client()
    assert redis_client is not None
    pong = await redis_client.ping()
    assert pong is True

    # Cleanup
    await manager.close_redis()


def test_graceful_shutdown_handlers_exist():
    """Test that startup/shutdown handlers are defined"""
    from src.mcp_server.main import app

    # Check that lifecycle handlers exist
    assert len(app.router.on_startup) > 0, "No startup handlers defined"
    assert len(app.router.on_shutdown) > 0, "No shutdown handlers defined"


def test_structured_logging_configured():
    """Test that structured logging is configured"""
    from src.shared.observability import get_logger

    logger = get_logger("test")
    assert logger is not None

    # Log a test message (should not throw)
    logger.info("Test log message", test_field="test_value")


def test_opentelemetry_tracing_setup():
    """Test that OpenTelemetry tracing is set up"""
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
    assert tracer is not None

    # Create a test span
    with tracer.start_as_current_span("test_span"):
        pass  # Should not throw

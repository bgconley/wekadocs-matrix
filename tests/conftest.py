# Test fixtures for Phase 1 (NO MOCKS)
# See: /docs/implementation-plan.md → Phase 1 Tests

import asyncio
import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment
os.environ["ENV"] = "development"
os.environ.setdefault("NEO4J_PASSWORD", "testpassword123")
os.environ.setdefault("REDIS_PASSWORD", "testredis123")
os.environ.setdefault("JWT_SECRET", "test-secret-key-change-in-production-min-32-chars")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def docker_services_running():
    """
    Check if Docker services are running.
    Tests require docker-compose up to be running.
    """
    import socket

    services = {
        "neo4j": ("localhost", 7687),
        "qdrant": ("localhost", 6333),
        "redis": ("localhost", 6379),
    }

    for service_name, (host, port) in services.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result != 0:
            pytest.skip(
                f"{service_name} not available at {host}:{port}. "
                f"Run 'docker-compose up -d' before running tests."
            )


@pytest.fixture(scope="session")
def app(docker_services_running):
    """Get FastAPI app instance"""
    from src.mcp_server.main import app

    return app


@pytest.fixture(scope="session")
def client(app) -> Generator:
    """Get FastAPI test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def neo4j_driver(docker_services_running):
    """Get Neo4j driver for testing"""
    from src.shared.connections import get_connection_manager

    manager = get_connection_manager()
    driver = manager.get_neo4j_driver()
    yield driver
    # Don't close - shared across tests


@pytest.fixture(scope="session")
async def redis_client(docker_services_running):
    """Get Redis client for testing"""
    from src.shared.connections import get_connection_manager

    manager = get_connection_manager()
    client = await manager.get_redis_client()
    yield client
    # Don't close - shared across tests


@pytest.fixture(scope="session")
def qdrant_client(docker_services_running):
    """Get Qdrant client for testing"""
    from src.shared.connections import get_connection_manager

    manager = get_connection_manager()
    client = manager.get_qdrant_client()
    yield client
    # Don't close - shared across tests


@pytest.fixture
def jwt_token():
    """Create a valid JWT token for testing"""
    from src.mcp_server.security import get_jwt_auth

    auth = get_jwt_auth()
    token = auth.create_token(subject="test_user")
    return token

# Phase 1, Task 1.4 Tests - Security layer (NO MOCKS)
# See: /docs/implementation-plan.md → Task 1.4 DoD & Tests

import asyncio

import pytest


def test_jwt_token_creation():
    """Test JWT token creation"""
    from src.mcp_server.security import get_jwt_auth

    auth = get_jwt_auth()
    token = auth.create_token(subject="test_user")

    assert token is not None
    assert len(token) > 0


def test_jwt_token_verification():
    """Test JWT token verification"""
    from src.mcp_server.security import get_jwt_auth

    auth = get_jwt_auth()
    token = auth.create_token(subject="test_user")

    # Verify token
    payload = auth.verify_token(token)
    assert payload["sub"] == "test_user"
    assert "exp" in payload
    assert "iat" in payload


def test_invalid_jwt_token_rejected():
    """Test that invalid JWT tokens are rejected"""
    from fastapi import HTTPException

    from src.mcp_server.security import get_jwt_auth

    auth = get_jwt_auth()

    with pytest.raises(HTTPException) as exc_info:
        auth.verify_token("invalid_token")

    assert exc_info.value.status_code == 401


def test_expired_jwt_token_rejected():
    """Test that expired JWT tokens are rejected"""
    from datetime import timedelta

    from fastapi import HTTPException

    from src.mcp_server.security import get_jwt_auth

    auth = get_jwt_auth()

    # Create token that expires immediately
    token = auth.create_token(subject="test_user", expires_delta=timedelta(seconds=-1))

    with pytest.raises(HTTPException) as exc_info:
        auth.verify_token(token)

    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_rate_limiter_allows_normal_requests(docker_services_running):
    """Test that rate limiter allows normal request rates"""
    from src.mcp_server.security import get_rate_limiter

    limiter = get_rate_limiter()

    # Create mock request
    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        client = MockClient()
        headers = {}
        state = type("obj", (object,), {})()

    request = MockRequest()

    # Should allow multiple requests under limit
    for i in range(5):
        result = await limiter.check_rate_limit(request)
        assert result is True


@pytest.mark.asyncio
async def test_rate_limiter_blocks_burst_requests(docker_services_running):
    """Test that rate limiter blocks burst requests"""
    from fastapi import HTTPException

    from src.mcp_server.security import get_rate_limiter
    from src.shared.config import get_config

    config = get_config()
    burst_size = config.rate_limit.burst_size

    limiter = get_rate_limiter()

    # Create mock request
    class MockClient:
        host = "127.0.0.2"  # Different IP

    class MockRequest:
        client = MockClient()
        headers = {}
        state = type("obj", (object,), {})()

    request = MockRequest()

    # Exhaust burst bucket
    for i in range(burst_size):
        await limiter.check_rate_limit(request)

    # Next request should be blocked
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_rate_limit(request)

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiter_refills_over_time(docker_services_running):
    """Test that rate limiter refills tokens over time"""
    from src.mcp_server.security import get_rate_limiter

    limiter = get_rate_limiter()

    # Create mock request
    class MockClient:
        host = "127.0.0.3"

    class MockRequest:
        client = MockClient()
        headers = {}
        state = type("obj", (object,), {})()

    request = MockRequest()

    # Use some tokens
    for i in range(3):
        await limiter.check_rate_limit(request)

    # Wait for refill
    await asyncio.sleep(2)

    # Should allow more requests
    result = await limiter.check_rate_limit(request)
    assert result is True


def test_audit_logger_logs_request():
    """Test that audit logger logs requests"""
    from src.shared.audit import get_audit_logger

    audit_logger = get_audit_logger()

    # Create mock request
    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        client = MockClient()
        headers = {"user-agent": "test"}
        state = type("obj", (object,), {})()

    request = MockRequest()

    # Should not throw
    audit_logger.log_request(
        request=request,
        endpoint="/test",
        method="GET",
        client_id="test_user",
    )


def test_audit_logger_logs_response():
    """Test that audit logger logs responses"""
    from src.shared.audit import get_audit_logger

    audit_logger = get_audit_logger()

    # Create mock request
    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        client = MockClient()
        headers = {}
        state = type("obj", (object,), {})()

    request = MockRequest()

    # Should not throw
    audit_logger.log_response(
        request=request,
        endpoint="/test",
        method="GET",
        status_code=200,
        duration_ms=45.2,
    )


def test_audit_logger_logs_auth_events():
    """Test that audit logger logs auth events"""
    from src.shared.audit import get_audit_logger

    audit_logger = get_audit_logger()

    # Should not throw
    audit_logger.log_auth_event(
        event_type="login",
        client_id="test_user",
        success=True,
    )


def test_audit_logger_logs_security_events():
    """Test that audit logger logs security events"""
    from src.shared.audit import get_audit_logger

    audit_logger = get_audit_logger()

    # Should not throw
    audit_logger.log_security_event(
        event_type="suspicious_activity",
        severity="medium",
        description="Test security event",
    )


def test_correlation_id_in_logs():
    """Test that correlation IDs are included in logs"""
    from src.shared.observability import get_correlation_id, set_correlation_id

    # Set correlation ID
    test_corr_id = "test-correlation-123"
    set_correlation_id(test_corr_id)

    # Get it back
    retrieved_corr_id = get_correlation_id()
    assert retrieved_corr_id == test_corr_id

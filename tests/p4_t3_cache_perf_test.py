# tests/p4_t3_cache_perf_test.py
import json
import os
import time
import uuid

import pytest
import requests
from redis import Redis

MCP_URL = os.getenv("MCP_BASE_URL", "http://localhost:8000/mcp")


@pytest.fixture(scope="module", name="redis_sync")
def _redis_sync():
    """Return a synchronous Redis client regardless of any async fixture elsewhere."""
    redis_password = os.getenv("REDIS_PASSWORD", "testredis123")
    redis_uri = os.getenv(
        "CACHE_REDIS_URI", f"redis://:{redis_password}@localhost:6379/0"
    )
    return Redis.from_url(redis_uri, decode_responses=True)


def _call_tool(name: str, args: dict):
    payload = {
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
        "id": str(uuid.uuid4()),
    }
    r = requests.post(f"{MCP_URL}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


@pytest.mark.order(6)
def test_cold_to_warm_latency_improves_and_cache_key_exists(redis_sync):
    # Tool name should match your server’s registry (adjust if different)
    tool = os.getenv("P4_CACHE_TEST_TOOL", "search_documentation")
    args = {"query": "fsync configuration", "limit": 5}

    # Compose the cache key the server uses (keep in sync with server logic)
    cache_key = f"tool:{tool}:{json.dumps(args, sort_keys=True)}"

    # Ensure clean slate
    redis_sync.delete(cache_key)

    t0 = time.time()
    _ = _call_tool(tool, args)  # cold
    t1 = time.time()
    cold_ms = (t1 - t0) * 1000.0

    t2 = time.time()
    _ = _call_tool(tool, args)  # warm
    t3 = time.time()
    warm_ms = (t3 - t2) * 1000.0

    assert (
        warm_ms < cold_ms * 0.6
    ), f"Warm call ({warm_ms:.1f}ms) not at least 40% faster than cold ({cold_ms:.1f}ms)."
    # Key must exist in Redis after first call
    assert (
        redis_sync.get(cache_key) is not None
    ), f"Expected Redis key {cache_key} to exist."


@pytest.mark.xfail(reason="Cache invalidation via versioning not yet implemented")
@pytest.mark.order(7)
def test_cache_invalidation_after_graph_update(redis_sync):
    """
    Kickoff guardrail: once you implement cache versioning/bust on writes or epoch keys,
    this should pass automatically.
    """
    tool = os.getenv("P4_CACHE_TEST_TOOL", "search_documentation")
    args = {"query": "fsync configuration", "limit": 5}
    cache_key = f"tool:{tool}:{json.dumps(args, sort_keys=True)}"

    # Simulate an update: bump a cache epoch key the server should include in user cache keys.
    # e.g., redis_sync.incr("cache:epoch:query")
    # For kickoff we assert the server deletes old key or uses a new versioned key.
    old = redis_sync.get(cache_key)
    # Expectation: after an update operation (ingestion or write), the previous key is gone or a new key is used.
    # Implementers: wire this to your write path or reconciliation logic.
    assert (
        old is None
    ), "After update, old cache key should be invalidated or superseded."

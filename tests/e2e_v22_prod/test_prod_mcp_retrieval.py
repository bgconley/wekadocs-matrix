"""E2E v2.2 – MCP server retrieval path (spec-only).

Calls the MCP server's `search_documentation` tool end-to-end using queries derived
from the ingested snapshot. Validates non-error responses and basic payload shape.
"""

from __future__ import annotations

import os
import socket
from typing import List

import pytest
import requests
from qdrant_client import QdrantClient

pytestmark = pytest.mark.integration


def _server_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _extract_titles(
    qdrant: QdrantClient, snapshot_scope: str, limit: int = 8
) -> List[str]:
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    flt = Filter(
        must=[
            FieldCondition(key="snapshot_scope", match=MatchValue(value=snapshot_scope))
        ]
    )
    points, _ = qdrant.scroll(
        collection_name="chunks_multi",
        limit=400,
        with_payload=True,
        with_vectors=False,
        scroll_filter=flt,
    )
    seen: List[str] = []
    for p in points or []:
        payload = p.payload or {}
        h = (payload.get("heading") or "").strip()
        if h and h not in seen:
            seen.append(h)
        if len(seen) >= limit:
            break
    return seen or ["installation", "configuration", "troubleshooting"]


def test_mcp_search_documentation_basic(prod_env, capture_logs):
    host = os.environ.get("MCP_HOST", "localhost")
    port = int(os.environ.get("MCP_PORT", "8000"))
    base_url = os.environ.get("MCP_BASE_URL", f"http://{host}:{port}")

    if not _server_ready(host, port):
        pytest.skip(
            "MCP server not reachable; start src/mcp_server.main app to run this test"
        )

    capture_logs("mcp-before")

    qdrant: QdrantClient = prod_env["qdrant"]
    snapshot_scope: str = prod_env["snapshot_scope"]
    queries = _extract_titles(qdrant, snapshot_scope, limit=5)

    for q in queries:
        resp = requests.post(
            f"{base_url}/mcp/tools/call",
            json={
                "name": "search_documentation",
                "arguments": {"query": q, "top_k": 20, "verbosity": "graph"},
            },
            timeout=10,
        )
        assert resp.status_code == 200, f"HTTP {resp.status_code} from MCP"
        body = resp.json()
        assert body.get("is_error") is False, f"MCP returned error: {body}"
        assert (
            isinstance(body.get("content"), list) and body["content"]
        ), "Empty MCP content"

    capture_logs("mcp-after")

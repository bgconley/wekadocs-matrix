from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable

import anyio
import pytest
import requests
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

MCP_STREAMABLE_URL = os.getenv("MCP_STREAMABLE_URL", "http://localhost:8000/_mcp")


def _streamable_headers() -> dict[str, str] | None:
    if os.getenv("MCP_STREAMABLE_ACCEPT_JSON", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"Accept": "application/json"}
    return None


def _skip_if_streamable_unavailable() -> None:
    health_url = MCP_STREAMABLE_URL.rstrip("/") + "/health"
    try:
        response = requests.get(health_url, timeout=2)
    except Exception as exc:  # pragma: no cover - network skip
        pytest.skip(f"Streamable MCP not reachable: {exc}")
    if response.status_code != 200:
        pytest.skip(f"Streamable MCP not ready (status={response.status_code}).")


def _normalize_schema(schema: Any) -> Any:
    if schema is None:
        return None
    try:
        return json.loads(json.dumps(schema, sort_keys=True))
    except TypeError:
        return schema


def _tool_snapshot(tools: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        snapshot[tool.name] = {
            "inputSchema": _normalize_schema(tool.inputSchema),
            "outputSchema": _normalize_schema(tool.outputSchema),
        }
    return snapshot


async def _list_stdio_tools(
    command: str, args: list[str], cwd: str | None
) -> list[Any]:
    params = StdioServerParameters(
        command=command,
        args=args,
        env=os.environ.copy(),
        cwd=cwd,
        encoding="utf-8",
        encoding_error_handler="replace",
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


async def _list_http_tools(url: str) -> list[Any]:
    async with streamablehttp_client(
        url,
        headers=_streamable_headers(),
        timeout=15.0,
        sse_read_timeout=15.0,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


async def _list_http_resource_templates(url: str) -> list[Any]:
    async with streamablehttp_client(
        url,
        headers=_streamable_headers(),
        timeout=15.0,
        sse_read_timeout=15.0,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_resource_templates()
            return result.resourceTemplates


async def _call_http_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    async with streamablehttp_client(
        MCP_STREAMABLE_URL,
        headers=_streamable_headers(),
        timeout=15.0,
        sse_read_timeout=15.0,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result


def test_streamable_tools_list_contract(docker_services_running) -> None:
    _skip_if_streamable_unavailable()

    async def _run() -> None:
        tools = await _list_http_tools(MCP_STREAMABLE_URL)
        tool_names = {tool.name for tool in tools}

        assert "kb.search" in tool_names
        assert "kb.retrieve_evidence" in tool_names
        assert "search_documentation" not in tool_names

        kb_search = next(tool for tool in tools if tool.name == "kb.search")
        input_schema = kb_search.inputSchema or {}
        properties = input_schema.get("properties", {})
        assert "scope" in properties

    anyio.run(_run)


def test_streamable_search_documentation_disabled(docker_services_running) -> None:
    _skip_if_streamable_unavailable()

    async def _run() -> None:
        result = await _call_http_tool("search_documentation", {"query": "test"})
        payload = result.structuredContent or {}
        assert payload.get("error", {}).get("code") == "INVALID_ARGUMENT"

    anyio.run(_run)


def test_streamable_resource_templates(docker_services_running) -> None:
    _skip_if_streamable_unavailable()

    async def _run() -> None:
        templates = await _list_http_resource_templates(MCP_STREAMABLE_URL)
        names = {template.name for template in templates}
        assert "scratch" in names
        if os.getenv("MCP_DIAGNOSTICS_RESOURCES_ENABLED", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            assert "diagnostics" in names

    anyio.run(_run)


def test_streamable_kb_search_contract(docker_services_running) -> None:
    _skip_if_streamable_unavailable()

    async def _run() -> None:
        result = await _call_http_tool(
            "kb.search",
            {
                "query": "test query",
                "top_k": 3,
                "options": {"max_snippet_chars": 120},
            },
        )
        payload = result.structuredContent or {}
        assert isinstance(payload.get("results"), list)
        assert isinstance(payload.get("session_id"), str)
        assert isinstance(payload.get("partial"), bool)
        assert isinstance(payload.get("limit_reason"), str)
        assert isinstance(payload.get("meta"), dict)

        results = payload.get("results") or []
        if results:
            assert len(results) <= 3
            preview = results[0].get("preview")
            if isinstance(preview, str):
                assert len(preview) <= 120

    anyio.run(_run)


def test_streamable_stdio_schema_parity(docker_services_running) -> None:
    _skip_if_streamable_unavailable()

    async def _run() -> None:
        stdio_tools = await _list_stdio_tools(
            sys.executable,
            ["-m", "src.mcp_server.stdio_server"],
            os.getcwd(),
        )
        http_tools = await _list_http_tools(MCP_STREAMABLE_URL)

        stdio_snapshot = _tool_snapshot(stdio_tools)
        http_snapshot = _tool_snapshot(http_tools)

        assert set(stdio_snapshot.keys()) == set(http_snapshot.keys())
        for name in stdio_snapshot:
            assert stdio_snapshot[name] == http_snapshot[name]

    anyio.run(_run)

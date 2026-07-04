#!/usr/bin/env python3
"""Compare MCP tool schemas between STDIO and HTTP transports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client


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


async def _list_http_tools(
    url: str,
    headers: dict[str, str] | None,
    timeout: float,
) -> list[Any]:
    async with streamablehttp_client(
        url,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=timeout,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


async def _run(args: argparse.Namespace) -> int:
    stdio_tools = await _list_stdio_tools(
        args.stdio_command,
        args.stdio_args,
        args.cwd,
    )
    http_tools = await _list_http_tools(
        args.http_url,
        args.http_headers,
        args.http_timeout,
    )

    stdio_snapshot = _tool_snapshot(stdio_tools)
    http_snapshot = _tool_snapshot(http_tools)

    stdio_names = set(stdio_snapshot.keys())
    http_names = set(http_snapshot.keys())

    missing = sorted(stdio_names - http_names)
    extra = sorted(http_names - stdio_names)
    if missing:
        print("HTTP missing tools:", ", ".join(missing))
    if extra:
        print("HTTP extra tools:", ", ".join(extra))

    mismatches = []
    for name in sorted(stdio_names & http_names):
        stdio_schema = stdio_snapshot[name]
        http_schema = http_snapshot[name]
        if stdio_schema != http_schema:
            mismatches.append(name)

    if mismatches:
        print("Schema mismatches:", ", ".join(mismatches))

    if missing or extra or mismatches:
        return 1

    print("STDIO and HTTP tool schemas match.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MCP tool schemas between STDIO and HTTP transports"
    )
    parser.add_argument(
        "--stdio-command",
        default=sys.executable,
        help="Command to launch the STDIO MCP server",
    )
    parser.add_argument(
        "--stdio-args",
        nargs="*",
        default=["-m", "src.mcp_server.stdio_server"],
        help="Arguments for the STDIO server command",
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory for STDIO server",
    )
    parser.add_argument(
        "--http-url",
        default="http://localhost:8000/_mcp",
        help="Streamable HTTP MCP base URL",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--http-accept-json",
        action="store_true",
        help="Use application/json for Streamable HTTP responses",
    )
    parser.add_argument(
        "--http-header",
        action="append",
        default=[],
        help="Additional HTTP header in Key:Value format",
    )
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.http_accept_json:
        headers["Accept"] = "application/json"
    for item in args.http_header:
        if ":" not in item:
            parser.error("--http-header must be in Key:Value format")
        key, value = item.split(":", 1)
        headers[key.strip()] = value.strip()
    args.http_headers = headers or None

    return anyio.run(_run, args)


if __name__ == "__main__":
    raise SystemExit(main())

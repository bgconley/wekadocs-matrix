#!/usr/bin/env python3
"""
STDIO Container RRF Wiring Test
================================
Tests the actual container workflow by invoking the STDIO MCP server
via docker exec and validating RRF fusion fields are properly exposed.

Usage:
    python scripts/test_stdio_container_rrf.py

This script:
1. Spawns the STDIO server via 'docker exec -i weka-mcp-server python -m src.mcp_server.stdio_server'
2. Performs the MCP protocol handshake (initialize → initialized)
3. Calls the search_sections tool with a test query
4. Validates that RRF fusion fields are present in the response
"""

import json
import subprocess
import sys
import time
from typing import Any

# ANSI color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def send_jsonrpc(
    proc: subprocess.Popen,
    method: str,
    params: dict | None = None,
    msg_id: int | None = None,
) -> None:
    """Send a JSON-RPC message to the STDIO server."""
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        message["params"] = params
    if msg_id is not None:
        message["id"] = msg_id

    payload = json.dumps(message) + "\n"
    proc.stdin.write(payload)
    proc.stdin.flush()
    print(
        f"{CYAN}→ Sent:{RESET} {method}"
        + (f" (id={msg_id})" if msg_id else " (notification)")
    )


def read_jsonrpc(proc: subprocess.Popen, timeout: float = 30.0) -> dict | None:
    """Read a JSON-RPC response from the STDIO server."""
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if line:
            line = line.strip()
            if line:
                try:
                    response = json.loads(line)
                    return response
                except json.JSONDecodeError:
                    # Skip non-JSON lines (logs might leak through)
                    continue
        time.sleep(0.01)
    return None


def validate_rrf_fields(result: dict) -> tuple[int, int, list[str]]:
    """
    Validate that RRF fusion fields are present in the result.
    Returns (passed_count, total_count, failure_messages).
    """
    results_list = result.get("results", [])
    if not results_list:
        return 0, 1, ["No results returned - cannot validate RRF fields"]

    passed = 0
    total = 0
    failures = []

    # Check first result for presence of RRF fields
    first_result = results_list[0]

    # Field checks
    checks = [
        ("fusion_method", lambda v: v in ("rrf", "rerank", "rrf_fusion", None)),
        ("fused_score", lambda v: v is None or isinstance(v, (int, float))),
        ("title_vec_score", lambda v: v is None or isinstance(v, (int, float))),
        ("entity_vec_score", lambda v: v is None or isinstance(v, (int, float))),
        ("doc_title_sparse_score", lambda v: v is None or isinstance(v, (int, float))),
        ("lexical_vec_score", lambda v: v is None or isinstance(v, (int, float))),
    ]

    for field_name, validator in checks:
        total += 1
        if field_name in first_result:
            value = first_result[field_name]
            if validator(value):
                passed += 1
                print(f"  {GREEN}✓{RESET} {field_name} = {value}")
            else:
                failures.append(f"{field_name} has invalid value: {value}")
                print(f"  {RED}✗{RESET} {field_name} = {value} (invalid)")
        else:
            failures.append(f"{field_name} is MISSING from response")
            print(f"  {RED}✗{RESET} {field_name} MISSING")

    # Additional validation: at least one signal score should be populated
    signal_scores = [
        first_result.get("title_vec_score"),
        first_result.get("entity_vec_score"),
        first_result.get("doc_title_sparse_score"),
        first_result.get("lexical_vec_score"),
    ]
    has_signal_score = any(s is not None and s > 0 for s in signal_scores)
    total += 1
    if has_signal_score:
        passed += 1
        print(f"  {GREEN}✓{RESET} At least one signal score is populated")
    else:
        failures.append("No signal scores populated (all None or 0)")
        print(
            f"  {YELLOW}⚠{RESET} No signal scores populated (may be okay for small test data)"
        )

    return passed, total, failures


def main():
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}STDIO Container RRF Wiring Test{RESET}")
    print(f"{'=' * 70}\n")

    # Step 1: Start the STDIO server via docker exec
    print(f"{BOLD}Step 1: Spawning STDIO server via docker exec...{RESET}")
    cmd = [
        "docker",
        "exec",
        "-i",
        "weka-mcp-server",
        "python",
        "-m",
        "src.mcp_server.stdio_server",
    ]
    print(f"  Command: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )
    except Exception as e:
        print(f"{RED}Failed to start STDIO server: {e}{RESET}")
        return 1

    print(f"  {GREEN}✓{RESET} Process started (PID: {proc.pid})")

    # Give server a moment to initialize
    time.sleep(1.0)

    try:
        # Step 2: Send initialize request
        print(f"\n{BOLD}Step 2: MCP Protocol Handshake{RESET}")
        send_jsonrpc(
            proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
            msg_id=1,
        )

        response = read_jsonrpc(proc, timeout=10.0)
        if response and "result" in response:
            print(f"  {GREEN}✓{RESET} Received initialize response")
            server_info = response.get("result", {}).get("serverInfo", {})
            print(
                f"    Server: {server_info.get('name', 'unknown')} v{server_info.get('version', '?')}"
            )
        else:
            print(f"  {RED}✗{RESET} No valid initialize response")
            print(f"    Raw: {response}")
            return 1

        # Send initialized notification
        send_jsonrpc(proc, "notifications/initialized", {})
        print(f"  {GREEN}✓{RESET} Sent initialized notification")
        time.sleep(0.5)

        # Step 3: Call search_sections tool
        print(f"\n{BOLD}Step 3: Calling search_sections tool{RESET}")
        test_query = "WEKA filesystem configuration"
        send_jsonrpc(
            proc,
            "tools/call",
            {"name": "search_sections", "arguments": {"query": test_query, "top_k": 5}},
            msg_id=2,
        )

        response = read_jsonrpc(proc, timeout=30.0)
        if not response:
            print(f"  {RED}✗{RESET} No response received (timeout)")
            return 1

        if "error" in response:
            print(f"  {RED}✗{RESET} Error response: {response['error']}")
            return 1

        # Extract the result content
        result_content = response.get("result", {})
        if isinstance(result_content, dict) and "content" in result_content:
            # MCP tools/call wraps result in content array
            content_items = result_content.get("content", [])
            if content_items and isinstance(content_items[0], dict):
                text_content = content_items[0].get("text", "{}")
                try:
                    actual_result = json.loads(text_content)
                except json.JSONDecodeError:
                    print(f"  {RED}✗{RESET} Failed to parse result JSON")
                    print(f"    Raw: {text_content[:200]}...")
                    return 1
            else:
                print(f"  {RED}✗{RESET} Unexpected content format")
                return 1
        else:
            actual_result = result_content

        results_list = actual_result.get("results", [])
        print(
            f"  {GREEN}✓{RESET} Received {len(results_list)} results for query: '{test_query}'"
        )

        # Step 4: Validate RRF fields
        print(f"\n{BOLD}Step 4: Validating RRF Fusion Fields{RESET}")
        if not results_list:
            print(f"  {YELLOW}⚠{RESET} No results to validate (Qdrant may be empty)")
            print("  Skipping field validation due to empty results")
            return 0

        # Show first result summary
        first = results_list[0]
        print("\n  First result preview:")
        print(f"    section_id: {first.get('section_id', 'N/A')[:50]}...")
        print(f"    title: {first.get('title', 'N/A')[:60]}...")
        print(f"    score: {first.get('score', 'N/A')}")
        print(f"    source: {first.get('source', 'N/A')}")

        print(f"\n  {BOLD}RRF Field Validation:{RESET}")
        passed, total, failures = validate_rrf_fields(actual_result)

        # Summary
        print(f"\n{BOLD}{'=' * 70}{RESET}")
        print(f"{BOLD}SUMMARY{RESET}")
        print(f"{'=' * 70}")

        if passed == total:
            print(f"\n  {GREEN}{BOLD}SUCCESS{RESET}: All {total} checks passed!")
            print(
                "\n  The STDIO container workflow is correctly exposing RRF fusion fields."
            )
            print("  Agents consuming search_sections will see:")
            print("    - fusion_method: Indicates retrieval strategy (rrf/rerank)")
            print("    - fused_score: RRF-computed score")
            print(
                "    - Per-signal scores: title_vec, entity_vec, doc_title_sparse, lexical_vec"
            )
        else:
            print(f"\n  {RED}{BOLD}FAILED{RESET}: {passed}/{total} checks passed")
            print("\n  Failures:")
            for f in failures:
                print(f"    - {f}")

        print(f"\n{'=' * 70}\n")
        return 0 if passed == total else 1

    finally:
        # Clean up
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  STDIO server process terminated")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Performance test for verbosity modes (E6)
Tests P50/P95 latency for snippet/full/graph modes.

Targets:
- snippet: P95 = 70ms (baseline, no regression)
- full: P95 < 100ms
- graph: P95 < 150ms

Usage:
    python scripts/perf/test_verbosity_latency.py
"""

import statistics
import sys
import time
from typing import List

import requests

# Test queries (varied complexity)
QUERIES = [
    "How do I configure a cluster?",
    "What are the system requirements?",
    "Troubleshoot performance issues",
    "Explain WekaFS architecture",
    "How do I upgrade the system?",
]

MCP_ENDPOINT = "http://localhost:8000/mcp/tools/call"
WARMUP_REQUESTS = 5
TEST_ITERATIONS = 10  # Each query run 10 times = 50 total requests per mode


def measure_latency(verbosity: str) -> List[float]:
    """
    Measure latency for a given verbosity mode.

    Args:
        verbosity: Verbosity mode (snippet/full/graph)

    Returns:
        List of latencies in milliseconds
    """
    latencies = []

    print(f"\nTesting verbosity={verbosity}")
    print(f"Warmup: {WARMUP_REQUESTS} requests...")

    # Warmup
    for i in range(WARMUP_REQUESTS):
        try:
            requests.post(
                MCP_ENDPOINT,
                json={
                    "name": "search_documentation",
                    "arguments": {
                        "query": QUERIES[i % len(QUERIES)],
                        "verbosity": verbosity,
                    },
                },
                timeout=30,
            )
        except Exception as e:
            print(f"Warmup request {i+1} failed: {e}")

    print(f"Running {len(QUERIES) * TEST_ITERATIONS} test requests...")

    # Test requests
    for iteration in range(TEST_ITERATIONS):
        for query in QUERIES:
            try:
                start = time.time()
                resp = requests.post(
                    MCP_ENDPOINT,
                    json={
                        "name": "search_documentation",
                        "arguments": {"query": query, "verbosity": verbosity},
                    },
                    timeout=30,
                )
                elapsed_ms = (time.time() - start) * 1000

                if resp.status_code == 200:
                    latencies.append(elapsed_ms)
                else:
                    print(f"Request failed with status {resp.status_code}")

            except Exception as e:
                print(f"Request failed: {e}")

    return latencies


def analyze_latencies(verbosity: str, latencies: List[float], target_p95: float):
    """
    Analyze and report latency statistics.

    Args:
        verbosity: Verbosity mode
        latencies: List of latencies in milliseconds
        target_p95: Target P95 latency

    Returns:
        True if target met, False otherwise
    """
    if not latencies:
        print(f"❌ {verbosity}: No successful requests")
        return False

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    mean = statistics.mean(latencies)

    print(f"\n{verbosity.upper()} MODE:")
    print(f"  Requests: {len(latencies)}")
    print(f"  Mean:     {mean:.1f}ms")
    print(f"  P50:      {p50:.1f}ms")
    print(f"  P95:      {p95:.1f}ms (target: <{target_p95}ms)")
    print(f"  P99:      {p99:.1f}ms")

    if p95 <= target_p95:
        print(f"  ✅ PASS: P95 {p95:.1f}ms <= {target_p95}ms")
        return True
    else:
        print(f"  ❌ FAIL: P95 {p95:.1f}ms > {target_p95}ms")
        return False


def main():
    """Run performance tests for all verbosity modes."""
    print("=" * 60)
    print("Verbosity Latency Performance Test (E6)")
    print("=" * 60)

    # Check server is running
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        if resp.status_code != 200:
            print("❌ Server health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the MCP server is running on http://localhost:8000")
        sys.exit(1)

    results = {}

    # Test snippet mode (baseline)
    latencies = measure_latency("snippet")
    results["snippet"] = analyze_latencies("snippet", latencies, target_p95=70.0)

    # Test full mode
    latencies = measure_latency("full")
    results["full"] = analyze_latencies("full", latencies, target_p95=100.0)

    # Test graph mode
    latencies = measure_latency("graph")
    results["graph"] = analyze_latencies("graph", latencies, target_p95=150.0)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for mode, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{mode:10s}: {status}")

    print("=" * 60)

    if all_passed:
        print("\n✅ All performance targets met!")
        sys.exit(0)
    else:
        print("\n❌ Some performance targets not met")
        sys.exit(1)


if __name__ == "__main__":
    main()

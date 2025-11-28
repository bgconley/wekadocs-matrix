#!/usr/bin/env python3
"""
Performance test for traverse_relationships tool (E6)
Tests P95 latency for graph traversal at different depths.

Targets:
- depth=1: P95 < 100ms
- depth=2: P95 < 200ms
- depth=3: P95 < 300ms

Usage:
    python scripts/perf/test_traversal_latency.py
"""

import statistics
import sys
import time
from typing import List

import requests

MCP_ENDPOINT = "http://localhost:8000/mcp/tools/call"
WARMUP_REQUESTS = 3
TEST_ITERATIONS = 20


# First, get some section IDs to use for traversal
def get_sample_section_ids(count: int = 5) -> List[str]:
    """Get sample section IDs from search results."""
    try:
        resp = requests.post(
            MCP_ENDPOINT,
            json={
                "name": "search_documentation",
                "arguments": {"query": "cluster configuration", "top_k": count},
            },
            timeout=10,
        )

        if resp.status_code == 200:
            data = resp.json()
            evidence = data.get("answer_json", {}).get("evidence", [])
            section_ids = [ev["section_id"] for ev in evidence if ev.get("section_id")]
            return section_ids[:count]
    except Exception as e:
        print(f"Failed to get sample section IDs: {e}")

    return []


def measure_traversal_latency(depth: int, section_ids: List[str]) -> List[float]:
    """
    Measure latency for traversal at a given depth.

    Args:
        depth: Traversal depth
        section_ids: Section IDs to traverse from

    Returns:
        List of latencies in milliseconds
    """
    latencies = []

    print(f"\nTesting depth={depth}")
    print(f"Warmup: {WARMUP_REQUESTS} requests...")

    # Warmup
    for i in range(WARMUP_REQUESTS):
        try:
            requests.post(
                MCP_ENDPOINT,
                json={
                    "name": "traverse_relationships",
                    "arguments": {
                        "start_ids": [section_ids[i % len(section_ids)]],
                        "max_depth": depth,
                    },
                },
                timeout=30,
            )
        except Exception as e:
            print(f"Warmup request {i + 1} failed: {e}")

    print(f"Running {TEST_ITERATIONS} test requests...")

    # Test requests
    for i in range(TEST_ITERATIONS):
        try:
            start_id = section_ids[i % len(section_ids)]
            start = time.time()
            resp = requests.post(
                MCP_ENDPOINT,
                json={
                    "name": "traverse_relationships",
                    "arguments": {
                        "start_ids": [start_id],
                        "max_depth": depth,
                    },
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


def analyze_latencies(depth: int, latencies: List[float], target_p95: float):
    """
    Analyze and report latency statistics.

    Args:
        depth: Traversal depth
        latencies: List of latencies in milliseconds
        target_p95: Target P95 latency

    Returns:
        True if target met, False otherwise
    """
    if not latencies:
        print(f"❌ depth={depth}: No successful requests")
        return False

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    mean = statistics.mean(latencies)

    print(f"\nDEPTH={depth}:")
    print(f"  Requests: {len(latencies)}")
    print(f"  Mean:     {mean:.1f}ms")
    print(f"  P50:      {p50:.1f}ms")
    print(f"  P95:      {p95:.1f}ms (target: <{target_p95}ms)")

    if p95 <= target_p95:
        print(f"  ✅ PASS: P95 {p95:.1f}ms <= {target_p95}ms")
        return True
    else:
        print(f"  ❌ FAIL: P95 {p95:.1f}ms > {target_p95}ms")
        return False


def main():
    """Run performance tests for traversal at different depths."""
    print("=" * 60)
    print("Traversal Latency Performance Test (E6)")
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

    # Get sample section IDs
    print("\nGetting sample section IDs for testing...")
    section_ids = get_sample_section_ids(count=5)

    if not section_ids:
        print("❌ Failed to get sample section IDs")
        sys.exit(1)

    print(f"Using {len(section_ids)} section IDs for testing")

    results = {}

    # Test depth=1
    latencies = measure_traversal_latency(1, section_ids)
    results[1] = analyze_latencies(1, latencies, target_p95=100.0)

    # Test depth=2 (primary use case)
    latencies = measure_traversal_latency(2, section_ids)
    results[2] = analyze_latencies(2, latencies, target_p95=200.0)

    # Test depth=3
    latencies = measure_traversal_latency(3, section_ids)
    results[3] = analyze_latencies(3, latencies, target_p95=300.0)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for depth, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"depth={depth}: {status}")

    print("=" * 60)

    if all_passed:
        print("\n✅ All performance targets met!")
        sys.exit(0)
    else:
        print("\n❌ Some performance targets not met")
        sys.exit(1)


if __name__ == "__main__":
    main()

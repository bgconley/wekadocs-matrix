#!/usr/bin/env python3
"""
Test Jina API payload size limits to validate hypothesis.

This script tests:
1. Small payload (should succeed)
2. Large payload ~160KB (should fail with 400)

Expected behavior:
- Small: HTTP 200
- Large: HTTP 400 Bad Request (payload too large)
"""

import json
import os

import httpx

# Jina API configuration
JINA_API_KEY = os.getenv(
    "JINA_API_KEY",
    "jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi",  # pragma: allowlist secret
)
API_URL = "https://api.jina.ai/v1/embeddings"


def test_small_payload():
    """Test with small payload (should succeed)."""
    print("=" * 80)
    print("TEST 1: Small Payload (~100 bytes)")
    print("=" * 80)

    payload = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage",
        "input": ["Short test text for validation"],
        "truncate": False,
        "normalized": True,
        "embedding_type": "float",
    }

    payload_size = len(json.dumps(payload))
    print(f"Payload size: {payload_size} bytes")

    try:
        client = httpx.Client(timeout=30.0)
        response = client.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response.elapsed.total_seconds():.2f}s")

        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("data", [])
            print(f"✅ SUCCESS: Received {len(embeddings)} embeddings")
            if embeddings:
                emb_dims = len(embeddings[0].get("embedding", []))
                print(f"   Embedding dimensions: {emb_dims}")
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"   Response: {response.text[:200]}")

        client.close()
        return response.status_code == 200

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False


def test_large_payload():
    """Test with large payload ~160KB (should fail with 400)."""
    print("\n" + "=" * 80)
    print("TEST 2: Large Payload (~160KB, 32 texts × 5KB each)")
    print("=" * 80)

    # Create 32 texts, each 5,000 characters
    large_text = "x" * 5000
    texts = [large_text] * 32

    payload = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage",
        "input": texts,
        "truncate": False,
        "normalized": True,
        "embedding_type": "float",
    }

    payload_size = len(json.dumps(payload))
    print(f"Payload size: {payload_size:,} bytes (~{payload_size / 1024:.1f} KB)")
    print(f"Number of texts: {len(texts)}")
    print(f"Characters per text: {len(texts[0]):,}")
    print(f"Total characters: {sum(len(t) for t in texts):,}")

    try:
        client = httpx.Client(timeout=30.0)
        response = client.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 400:
            print("✅ EXPECTED FAILURE: HTTP 400 (payload too large)")
            print(f"   Response: {response.text[:500]}")
        elif response.status_code == 200:
            print("⚠️  UNEXPECTED SUCCESS: API accepted large payload")
            print("   This contradicts our hypothesis - 160KB was accepted")
        else:
            print(f"❓ UNEXPECTED: {response.status_code}")
            print(f"   Response: {response.text[:200]}")

        client.close()
        return response.status_code

    except httpx.ReadTimeout:
        print("⏱️  TIMEOUT: Request timed out after 30s")
        print("   This suggests processing started but is too slow")
        return "timeout"

    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {e}")
        return "error"


def test_medium_payload():
    """Test with medium payload ~80KB (16 texts × 5KB each)."""
    print("\n" + "=" * 80)
    print("TEST 3: Medium Payload (~80KB, 16 texts × 5KB each)")
    print("=" * 80)

    # Create 16 texts, each 5,000 characters
    large_text = "x" * 5000
    texts = [large_text] * 16

    payload = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage",
        "input": texts,
        "truncate": False,
        "normalized": True,
        "embedding_type": "float",
    }

    payload_size = len(json.dumps(payload))
    print(f"Payload size: {payload_size:,} bytes (~{payload_size / 1024:.1f} KB)")
    print(f"Number of texts: {len(texts)}")

    try:
        client = httpx.Client(timeout=30.0)
        response = client.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("data", [])
            print(f"✅ SUCCESS: Received {len(embeddings)} embeddings")
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"   Response: {response.text[:200]}")

        client.close()
        return response.status_code

    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {e}")
        return "error"


if __name__ == "__main__":
    print("Jina API Payload Size Limit Testing")
    print("=" * 80)
    print(f"API URL: {API_URL}")
    print(f"API Key: {JINA_API_KEY[:20]}...")
    print()

    # Test 1: Small payload
    small_success = test_small_payload()

    # Test 2: Large payload
    large_result = test_large_payload()

    # Test 3: Medium payload (find the boundary)
    medium_result = test_medium_payload()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Small payload (~100 bytes):  {'✅ PASS' if small_success else '❌ FAIL'}")
    print(f"Medium payload (~80KB):      {medium_result}")
    print(f"Large payload (~160KB):      {large_result}")
    print()

    if large_result == 400:
        print("✅ HYPOTHESIS CONFIRMED: Jina API rejects payloads ~160KB with HTTP 400")
        print("   Adaptive batching with ~50KB limit will prevent these errors.")
    elif large_result == 200:
        print("❌ HYPOTHESIS REJECTED: Jina API accepted 160KB payload")
        print("   Need to investigate other root causes for 400 errors.")
    else:
        print(f"⚠️  INCONCLUSIVE: Unexpected result ({large_result})")
        print("   May need additional testing with different payload sizes.")

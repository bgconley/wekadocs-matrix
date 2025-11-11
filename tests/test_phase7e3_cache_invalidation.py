"""
Phase 7E-3: Cache Invalidation Tests

Tests epoch-based and pattern-scan cache invalidation to ensure no stale reads
after document re-ingestion.

Acceptance Criteria (from canonical spec):
- After re-ingest of any doc, search results are fresh under both epoch and scan modes
- Automated test demonstrates no stale keys
- Zero stale results served after invalidation

Reference: Canonical Spec L3184-3281 (epoch), L3116-3183 (scan), Task 4.4
"""

import os
from typing import List

import pytest
import redis
from neo4j import GraphDatabase

from src.ingestion.build_graph import GraphBuilder
from src.shared.cache import TieredCache
from src.shared.chunk_utils import generate_chunk_id
from src.shared.config import load_config
from src.shared.connections import CompatQdrantClient
from tools.redis_epoch_bump import (
    bump_chunk_epochs,
    bump_doc_epoch,
    get_chunk_epoch,
    get_doc_epoch,
)
from tools.redis_invalidation import invalidate


@pytest.fixture(scope="module")
def config():
    """Load config with localhost forcing."""
    cfg, _ = load_config()  # Returns (Config, Settings) tuple
    return cfg


def _flush_patterns(r: redis.Redis, patterns: List[str]) -> None:
    """Helper to flush multiple Redis key patterns."""
    for pattern in patterns:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=pattern, count=500)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break


@pytest.fixture(scope="function")  # Changed from "module" to "function"
def redis_client():
    """
    Redis client for cache operations.

    Cleans BOTH namespaces to prevent cross-run contamination:
    - rag:v1:* (epoch counters: doc_epoch, chunk_epoch)
    - weka:cache:v1:* (actual cache data: fusion, vector, bm25, answer)
    """
    redis_url = os.getenv("CACHE_REDIS_URI", "redis://localhost:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()  # Test connection

    # Pre-clean to avoid cross-run contamination
    _flush_patterns(r, ["rag:v1:*", "weka:cache:v1:*"])

    yield r

    # Post-clean (keeps other tests isolated)
    _flush_patterns(r, ["rag:v1:*", "weka:cache:v1:*"])


@pytest.fixture(scope="module")
def neo4j_driver():
    """Neo4j driver for ingestion."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv(
        "NEO4J_PASSWORD", "testpassword123"
    )  # pragma: allowlist secret

    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()


@pytest.fixture(scope="module")
def qdrant_client():
    """Qdrant client for ingestion."""
    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    client = QdrantClient(host=host, port=port, timeout=30)
    yield CompatQdrantClient(client)


def test_epoch_based_invalidation(redis_client, config):
    """
    Test epoch-based cache invalidation (O(1) preferred method).

    Verifies:
    1. Epoch counters start at 0
    2. Bumping increments epoch atomically
    3. Old cache keys with stale epochs are not read

    Reference: Canonical Spec L3184-3281
    """
    namespace = "rag:v1"
    doc_id = "test_doc_epoch_001"
    chunk_ids = ["chunk_epoch_001", "chunk_epoch_002"]

    # Step 1: Verify initial epochs are 0
    assert get_doc_epoch(redis_client, namespace, doc_id) == 0
    for cid in chunk_ids:
        assert get_chunk_epoch(redis_client, namespace, cid) == 0

    # Step 2: Simulate cache write with epoch 0
    from src.shared.cache import TieredCache

    cache = TieredCache(
        config=config,
        redis_client=redis_client,
        schema_version="v2.1",
        embedding_version="jina-embeddings-v3",
    )

    # Write cache entry with current epoch (0)
    fusion_key_v0 = cache.make_fusion_cache_key(doc_id, "test query")
    cache.put_fusion_cached(doc_id, "test query", {"results": "version_0"})

    # Verify cache hit with epoch 0
    cached = cache.get_fusion_cached(doc_id, "test query")
    assert cached is not None
    assert cached["results"] == "version_0"

    # Step 3: Bump epochs (simulating re-ingestion)
    new_doc_epoch = bump_doc_epoch(redis_client, namespace, doc_id)
    assert new_doc_epoch == 1

    new_chunk_epochs = bump_chunk_epochs(redis_client, namespace, chunk_ids)
    assert new_chunk_epochs == 2  # Sum of 2 chunks at epoch 1 each

    # Step 4: CRITICAL - Old cache key should now miss (epoch 0 != 1)
    fusion_key_v1 = cache.make_fusion_cache_key(doc_id, "test query")
    assert fusion_key_v0 != fusion_key_v1, "Epoch bump should change cache key"

    # Try to read with new epoch - should MISS (key changed)
    cached_after_bump = cache.get_fusion_cached(doc_id, "test query")
    assert cached_after_bump is None, "STALE READ DETECTED! Epoch invalidation failed"

    # Step 5: Write new cache with epoch 1
    cache.put_fusion_cached(doc_id, "test query", {"results": "version_1_fresh"})

    # Verify cache hit with epoch 1
    cached_fresh = cache.get_fusion_cached(doc_id, "test query")
    assert cached_fresh is not None
    assert cached_fresh["results"] == "version_1_fresh"

    # Step 6: Bump again and verify second invalidation
    new_doc_epoch_2 = bump_doc_epoch(redis_client, namespace, doc_id)
    assert new_doc_epoch_2 == 2

    cached_after_second_bump = cache.get_fusion_cached(doc_id, "test query")
    assert cached_after_second_bump is None, "Second epoch bump should also invalidate"


def test_pattern_scan_invalidation(redis_client, config):
    """
    Test pattern-scan cache invalidation (fallback method).

    Verifies:
    1. Pattern-scan finds and deletes matching keys
    2. All document-scoped caches are cleared
    3. Chunk-scoped caches are cleared

    Reference: Canonical Spec L3116-3183
    """
    namespace = "rag:v1"
    doc_id = "test_doc_scan_001"
    chunk_ids = ["chunk_scan_001", "chunk_scan_002"]

    # Step 1: Write multiple cache entries for this document
    test_keys = [
        f"{namespace}:fusion:doc:{doc_id}:epoch:0:q:abc123",
        f"{namespace}:answer:doc:{doc_id}:epoch:0:q:abc123",
        f"{namespace}:bm25:doc:{doc_id}:epoch:0:q:abc123",
        f"{namespace}:vector:chunk:{chunk_ids[0]}:epoch:0",
        f"{namespace}:vector:chunk:{chunk_ids[1]}:epoch:0",
    ]

    for key in test_keys:
        redis_client.setex(key, 3600, "test_value")

    # Verify keys exist
    for key in test_keys:
        assert redis_client.get(key) == "test_value"

    # Step 2: Run pattern-scan invalidation
    redis_url = os.getenv("CACHE_REDIS_URI", "redis://localhost:6379/0")
    deleted_count = invalidate(
        redis_url=redis_url,
        namespace=namespace,
        document_id=doc_id,
        chunk_ids=chunk_ids,
    )

    # Step 3: Verify all keys deleted
    assert deleted_count >= len(
        test_keys
    ), f"Expected {len(test_keys)} deletions, got {deleted_count}"

    for key in test_keys:
        assert redis_client.get(key) is None, f"Key {key} should be deleted"


def test_no_stale_reads_after_reingest(
    redis_client, neo4j_driver, qdrant_client, config
):
    """
    CRITICAL TEST: Verify no stale reads after document re-ingestion.

    This is the PRIMARY acceptance criterion for Phase 7E-3.

    Flow:
    1. Ingest document v1 → cache results
    2. Query and cache fusion results
    3. Re-ingest document v2 (triggers cache invalidation)
    4. Query again → MUST return fresh results (no stale cache)

    Reference: Canonical Spec Task 4.4 acceptance criteria
    """
    # Set cache mode to epoch for this test
    os.environ["CACHE_MODE"] = "epoch"

    doc_id = "test_doc_reingest_001"

    # Step 1: Create initial document version 1
    document_v1 = {
        "id": doc_id,
        "source_uri": "test://doc_reingest",
        "source_type": "test",
        "title": "Test Document v1",
        "version": "1.0",
        "checksum": "checksum_v1",
        "last_edited": "2025-01-01T00:00:00Z",
    }

    sections_v1 = [
        {
            "id": generate_chunk_id(doc_id, ["section_1"]),
            "document_id": doc_id,
            "level": 1,
            "order": 0,
            "heading": "Introduction v1",
            "text": "This is version 1 of the document.",
            "is_combined": False,
            "is_split": False,
            "original_section_ids": ["section_1"],
            "boundaries_json": "{}",
            "token_count": 10,
        }
    ]

    # Step 2: Ingest v1
    builder = GraphBuilder(neo4j_driver, config, qdrant_client)
    stats_v1 = builder.upsert_document(document_v1, sections_v1, {}, [])

    assert stats_v1["sections_upserted"] == 1
    assert stats_v1["cache_invalidation"]["success"] is True
    epoch_v1 = stats_v1["cache_invalidation"].get("doc_epoch", 1)

    # Step 3: Simulate caching fusion results for v1
    from src.shared.cache import TieredCache

    cache = TieredCache(
        config=config,
        redis_client=redis_client,
        schema_version="v2.1",
        embedding_version="jina-embeddings-v3",
    )

    query = "introduction"
    cache.put_fusion_cached(doc_id, query, {"version": "v1", "text": "version 1"})

    # Verify cache hit for v1
    cached_v1 = cache.get_fusion_cached(doc_id, query)
    assert cached_v1 is not None
    assert cached_v1["version"] == "v1"

    # Step 4: Re-ingest document version 2 (modified content)
    document_v2 = {
        **document_v1,
        "title": "Test Document v2",
        "version": "2.0",
        "checksum": "checksum_v2",
    }

    sections_v2 = [
        {
            **sections_v1[0],
            "heading": "Introduction v2",
            "text": "This is version 2 of the document with updated content.",
            "token_count": 12,
        }
    ]

    # Re-ingest triggers cache invalidation
    stats_v2 = builder.upsert_document(document_v2, sections_v2, {}, [])

    assert stats_v2["sections_upserted"] == 1
    assert stats_v2["cache_invalidation"]["success"] is True
    epoch_v2 = stats_v2["cache_invalidation"].get("doc_epoch", 2)
    assert epoch_v2 > epoch_v1, "Document epoch should increment on re-ingest"

    # Step 5: CRITICAL - Attempt to read cached results
    # This MUST return None (cache miss) to avoid stale reads
    cached_after_reingest = cache.get_fusion_cached(doc_id, query)

    assert (
        cached_after_reingest is None
    ), "CRITICAL FAILURE: Stale cache returned after re-ingest! Epoch invalidation failed."

    # Step 6: Cache new results for v2
    cache.put_fusion_cached(doc_id, query, {"version": "v2", "text": "version 2"})

    # Verify fresh cache hit
    cached_v2 = cache.get_fusion_cached(doc_id, query)
    assert cached_v2 is not None
    assert cached_v2["version"] == "v2", "Should get fresh v2 results"
    assert cached_v2["version"] != "v1", "Should NOT get stale v1 results"


def test_cache_invalidation_integration_with_ingestion(
    redis_client, neo4j_driver, qdrant_client, config
):
    """
    Test that cache invalidation is automatically called during ingestion.

    Verifies the integration between GraphBuilder.upsert_document and
    _invalidate_caches_post_ingest.
    """
    os.environ["CACHE_MODE"] = "epoch"

    doc_id = "test_doc_integration_001"

    document = {
        "id": doc_id,
        "source_uri": "test://integration",
        "source_type": "test",
        "title": "Integration Test",
        "version": "1.0",
        "checksum": "checksum_int",
        "last_edited": "2025-01-01T00:00:00Z",
    }

    sections = [
        {
            "id": generate_chunk_id(doc_id, ["int_section_1"]),
            "document_id": doc_id,
            "level": 1,
            "order": 0,
            "heading": "Section 1",
            "text": "Content for integration test.",
            "is_combined": False,
            "is_split": False,
            "original_section_ids": ["int_section_1"],
            "boundaries_json": "{}",
            "token_count": 8,
        }
    ]

    # Ingest document
    builder = GraphBuilder(neo4j_driver, config, qdrant_client)
    stats = builder.upsert_document(document, sections, {}, [])

    # Verify cache invalidation was called
    assert "cache_invalidation" in stats
    assert stats["cache_invalidation"]["success"] is True
    assert stats["cache_invalidation"]["mode"] == "epoch"
    assert stats["cache_invalidation"]["document_id"] == doc_id
    assert stats["cache_invalidation"]["chunk_count"] == 1
    assert "doc_epoch" in stats["cache_invalidation"]

    # Verify epoch was actually bumped in Redis
    namespace = "rag:v1"
    doc_epoch = get_doc_epoch(redis_client, namespace, doc_id)
    assert doc_epoch > 0, "Document epoch should be bumped after ingestion"


def test_epoch_vs_scan_mode_both_work(redis_client, config):
    """
    Test that both epoch and scan modes successfully invalidate caches.

    Verifies both invalidation strategies work correctly.
    """
    namespace = "rag:v1"
    doc_id_epoch = "test_doc_mode_epoch"
    doc_id_scan = "test_doc_mode_scan"

    cache = TieredCache(
        config=config,
        redis_client=redis_client,
        schema_version="v2.1",
        embedding_version="jina-embeddings-v3",
    )

    # Test 1: Epoch mode
    os.environ["CACHE_MODE"] = "epoch"

    cache.put_fusion_cached(doc_id_epoch, "query1", {"data": "epoch_test"})
    assert cache.get_fusion_cached(doc_id_epoch, "query1") is not None

    # Bump epoch
    bump_doc_epoch(redis_client, namespace, doc_id_epoch)

    # Should miss after epoch bump
    assert cache.get_fusion_cached(doc_id_epoch, "query1") is None

    # Test 2: Scan mode
    os.environ["CACHE_MODE"] = "scan"

    # Write cache entry with explicit key
    scan_key = f"{namespace}:fusion:doc:{doc_id_scan}:epoch:0:q:test123"
    redis_client.setex(scan_key, 3600, '{"data": "scan_test"}')
    assert redis_client.get(scan_key) is not None

    # Run scan invalidation
    redis_url = os.getenv("CACHE_REDIS_URI", "redis://localhost:6379/0")
    deleted = invalidate(
        redis_url=redis_url,
        namespace=namespace,
        document_id=doc_id_scan,
        chunk_ids=[],
    )

    assert deleted > 0
    assert redis_client.get(scan_key) is None


def test_chunk_epoch_invalidation(redis_client, config):
    """
    Test chunk-level epoch invalidation for vector caches.

    Verifies that chunk epochs work independently of document epochs.
    """
    namespace = "rag:v1"
    chunk_id = "test_chunk_vector_001"

    cache = TieredCache(
        config=config,
        redis_client=redis_client,
        schema_version="v2.1",
        embedding_version="jina-embeddings-v3",
    )

    # Initial epoch should be 0
    assert get_chunk_epoch(redis_client, namespace, chunk_id) == 0

    # Cache vector for chunk
    cache.put_vector_cached(chunk_id, {"embedding": [0.1, 0.2, 0.3]})
    assert cache.get_vector_cached(chunk_id) is not None

    # Bump chunk epoch
    bump_chunk_epochs(redis_client, namespace, [chunk_id])

    # Vector cache should miss after epoch bump
    assert (
        cache.get_vector_cached(chunk_id) is None
    ), "Chunk epoch bump should invalidate vector cache"

    # Verify epoch was incremented
    assert get_chunk_epoch(redis_client, namespace, chunk_id) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

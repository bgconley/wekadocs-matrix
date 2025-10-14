"""
Phase 4, Task 4.3: Caching & Performance Tests (NO MOCKS)

Tests L1+L2 caching with version-prefixed keys, cache warmers,
invalidation on version rotation, and >80% hit rate under load.

See: /docs/implementation-plan.md → Task 4.3
See: /docs/expert-coder-guidance.md → Phase 4, Task 4.3
"""

import os
import time

import pytest
import redis

from src.ops.warmers import QueryWarmer
from src.shared.cache import L1Cache, L2Cache, TieredCache
from src.shared.config import load_config

# ============================================================================
# L1 Cache Tests
# ============================================================================


class TestL1Cache:
    """Test in-process L1 cache functionality."""

    def test_l1_basic_get_put(self):
        """L1 cache stores and retrieves values."""
        cache = L1Cache(max_size=10, ttl_seconds=60)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_l1_miss_returns_none(self):
        """L1 cache returns None for missing keys."""
        cache = L1Cache(max_size=10, ttl_seconds=60)

        assert cache.get("nonexistent") is None

        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.0

    def test_l1_ttl_expiration(self):
        """L1 cache entries expire after TTL."""
        cache = L1Cache(max_size=10, ttl_seconds=1)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        assert cache.get("key1") is None

    def test_l1_lru_eviction(self):
        """L1 cache evicts oldest entry when full."""
        cache = L1Cache(max_size=3, ttl_seconds=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Cache is now full; adding one more should evict key1
        cache.put("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_l1_invalidate_prefix(self):
        """L1 cache can invalidate entries by prefix."""
        cache = L1Cache(max_size=10, ttl_seconds=60)

        cache.put("prefix:key1", "value1")
        cache.put("prefix:key2", "value2")
        cache.put("other:key3", "value3")

        count = cache.invalidate_prefix("prefix:")
        assert count == 2

        assert cache.get("prefix:key1") is None
        assert cache.get("prefix:key2") is None
        assert cache.get("other:key3") == "value3"

    def test_l1_clear(self):
        """L1 cache can be cleared entirely."""
        cache = L1Cache(max_size=10, ttl_seconds=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.stats()["size"] == 0


# ============================================================================
# L2 Cache Tests (Redis)
# ============================================================================


class TestL2Cache:
    """Test Redis-backed L2 cache functionality."""

    @pytest.fixture
    def redis_client(self):
        """Real Redis client for testing."""
        redis_password = os.getenv("REDIS_PASSWORD", "testredis123")
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=1,
            password=redis_password,
            decode_responses=False,
        )
        client.flushdb()  # Clean slate
        yield client
        client.flushdb()  # Cleanup
        client.close()

    def test_l2_basic_get_put(self, redis_client):
        """L2 cache stores and retrieves values from Redis."""
        cache = L2Cache(redis_client, ttl_seconds=60)

        cache.put("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result == {"data": "value1"}

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_l2_miss_returns_none(self, redis_client):
        """L2 cache returns None for missing keys."""
        cache = L2Cache(redis_client, ttl_seconds=60)

        assert cache.get("nonexistent") is None

        stats = cache.stats()
        assert stats["misses"] == 1

    def test_l2_ttl_expiration(self, redis_client):
        """L2 cache entries expire after TTL."""
        cache = L2Cache(redis_client, ttl_seconds=1)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        assert cache.get("key1") is None

    def test_l2_invalidate_prefix(self, redis_client):
        """L2 cache can invalidate entries by prefix."""
        cache = L2Cache(redis_client, ttl_seconds=60)

        cache.put("prefix:key1", "value1")
        cache.put("prefix:key2", "value2")
        cache.put("other:key3", "value3")

        count = cache.invalidate_prefix("prefix:")
        assert count == 2

        assert cache.get("prefix:key1") is None
        assert cache.get("prefix:key2") is None
        assert cache.get("other:key3") == "value3"


# ============================================================================
# Tiered Cache Tests (L1 + L2 with version prefixes)
# ============================================================================


class TestTieredCache:
    """Test two-tier cache with version-prefixed keys."""

    @pytest.fixture
    def config(self):
        """Load config for cache settings."""
        config, _ = load_config()
        return config

    @pytest.fixture
    def redis_client(self):
        """Real Redis client."""
        redis_password = os.getenv("REDIS_PASSWORD", "testredis123")
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=1,
            password=redis_password,
            decode_responses=False,
        )
        client.flushdb()
        yield client
        client.flushdb()
        client.close()

    @pytest.fixture
    def tiered_cache(self, config, redis_client):
        """Create tiered cache instance."""
        return TieredCache(
            config=config,
            redis_client=redis_client,
            schema_version="v1",
            embedding_version="v1",
        )

    def test_tiered_cache_l1_hit(self, tiered_cache):
        """Tiered cache serves from L1 on hit."""
        params = {"query": "test"}

        # Put in cache
        tiered_cache.put("search", params, {"result": "data"})

        # Should hit L1
        result = tiered_cache.get("search", params)
        assert result == {"result": "data"}

        # Check L1 stats show hit
        stats = tiered_cache.stats()
        assert stats["l1"]["hits"] == 1

    def test_tiered_cache_l2_promotion(self, tiered_cache):
        """Tiered cache promotes L2 hits to L1."""
        params = {"query": "test"}

        # Put only in L2
        tiered_cache.l2.put(
            tiered_cache._make_cache_key("search", params), {"result": "data"}
        )

        # First get should hit L2 and promote to L1
        result = tiered_cache.get("search", params)
        assert result == {"result": "data"}

        # Second get should hit L1
        result2 = tiered_cache.get("search", params)
        assert result2 == {"result": "data"}

        stats = tiered_cache.stats()
        assert stats["l1"]["hits"] == 1
        assert stats["l2"]["hits"] == 1

    def test_tiered_cache_miss(self, tiered_cache):
        """Tiered cache returns None on complete miss."""
        result = tiered_cache.get("search", {"query": "missing"})
        assert result is None

    def test_tiered_cache_cached_function(self, tiered_cache):
        """Tiered cache.cached() computes on miss and caches."""
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"computed": call_count}

        params = {"query": "test"}

        # First call should compute
        result1 = tiered_cache.cached("search", params, compute)
        assert result1 == {"computed": 1}
        assert call_count == 1

        # Second call should use cache
        result2 = tiered_cache.cached("search", params, compute)
        assert result2 == {"computed": 1}
        assert call_count == 1  # Not called again

    def test_cache_key_includes_versions(self, tiered_cache):
        """Cache keys include schema and embedding versions."""
        params = {"query": "test"}
        key = tiered_cache._make_cache_key("search", params)

        assert "v1:v1" in key  # schema:embedding versions
        assert "search" in key

    def test_cache_invalidation_on_version_rotation(self, config, redis_client):
        """Cache invalidates entries when versions change."""
        # Cache with v1:v1
        cache_v1 = TieredCache(config, redis_client, "v1", "v1")
        cache_v1.put("search", {"query": "test"}, {"result": "old"})

        # Verify cached
        assert cache_v1.get("search", {"query": "test"}) == {"result": "old"}

        # Create cache with v1:v2 (embedding version changed)
        cache_v2 = TieredCache(config, redis_client, "v1", "v2")

        # Should miss (different version prefix)
        assert cache_v2.get("search", {"query": "test"}) is None

        # Put new value
        cache_v2.put("search", {"query": "test"}, {"result": "new"})

        # v1 cache should still have old value
        assert cache_v1.get("search", {"query": "test"}) == {"result": "old"}

        # v2 cache should have new value
        assert cache_v2.get("search", {"query": "test"}) == {"result": "new"}

    def test_explicit_version_invalidation(self, tiered_cache):
        """Can explicitly invalidate specific version."""
        tiered_cache.put("search", {"query": "test1"}, {"result": "data1"})
        tiered_cache.put("search", {"query": "test2"}, {"result": "data2"})

        # Invalidate current version
        counts = tiered_cache.invalidate_version()

        # Should have cleared entries
        assert counts["l1"] > 0 or counts["l2"] > 0

        # Cache misses now
        assert tiered_cache.get("search", {"query": "test1"}) is None
        assert tiered_cache.get("search", {"query": "test2"}) is None


# ============================================================================
# Cache Warmer Tests
# ============================================================================


class TestQueryWarmer:
    """Test cache warmer functionality."""

    @pytest.fixture
    def config(self):
        config, _ = load_config()
        return config

    @pytest.fixture
    def redis_client(self):
        redis_password = os.getenv("REDIS_PASSWORD", "testredis123")
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=1,
            password=redis_password,
            decode_responses=False,
        )
        client.flushdb()
        yield client
        client.flushdb()
        client.close()

    @pytest.fixture
    def tiered_cache(self, config, redis_client):
        return TieredCache(config, redis_client, "v1", "v1")

    def test_warmer_basic_warming(self, tiered_cache):
        """Warmer populates cache with patterns."""

        def mock_executor(intent, params):
            return {"intent": intent, "params": params, "result": "mocked"}

        warmer = QueryWarmer(tiered_cache, mock_executor)

        patterns = [
            {"intent": "search", "params": {"query": "test1"}},
            {"intent": "search", "params": {"query": "test2"}},
        ]

        stats = warmer.warm_patterns(patterns)

        assert stats["warmed"] == 2
        assert stats["failed"] == 0

        # Verify cached
        assert tiered_cache.get("search", {"query": "test1"}) is not None
        assert tiered_cache.get("search", {"query": "test2"}) is not None

    def test_warmer_skips_already_cached(self, tiered_cache):
        """Warmer skips patterns already in cache."""

        def mock_executor(intent, params):
            return {"result": "mocked"}

        # Pre-populate one pattern
        tiered_cache.put("search", {"query": "test1"}, {"result": "existing"})

        warmer = QueryWarmer(tiered_cache, mock_executor)

        patterns = [
            {"intent": "search", "params": {"query": "test1"}},
            {"intent": "search", "params": {"query": "test2"}},
        ]

        stats = warmer.warm_patterns(patterns)

        assert stats["warmed"] == 1  # Only test2 warmed
        assert stats["failed"] == 0

    def test_warmer_handles_failures(self, tiered_cache):
        """Warmer records failures but continues."""

        def failing_executor(intent, params):
            if params.get("query") == "fail":
                raise Exception("Simulated failure")
            return {"result": "success"}

        warmer = QueryWarmer(tiered_cache, failing_executor)

        patterns = [
            {"intent": "search", "params": {"query": "fail"}},
            {"intent": "search", "params": {"query": "succeed"}},
        ]

        stats = warmer.warm_patterns(patterns)

        assert stats["warmed"] == 1
        assert stats["failed"] == 1


# ============================================================================
# Performance Tests (Hit Rate)
# ============================================================================


class TestCachePerformance:
    """Test cache hit rate under load."""

    @pytest.fixture
    def config(self):
        config, _ = load_config()
        return config

    @pytest.fixture
    def redis_client(self):
        redis_password = os.getenv("REDIS_PASSWORD", "testredis123")
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=1,
            password=redis_password,
            decode_responses=False,
        )
        client.flushdb()
        yield client
        client.flushdb()
        client.close()

    @pytest.fixture
    def tiered_cache(self, config, redis_client):
        return TieredCache(config, redis_client, "v1", "v1")

    def test_steady_state_hit_rate_exceeds_80_percent(self, tiered_cache):
        """
        Cache achieves >80% hit rate under realistic load.

        Simulates repeated queries with 80/20 distribution:
        - 80% of requests hit 20% of query patterns (hot queries)
        - 20% of requests hit remaining 80% (cold queries)
        """
        import random

        # Hot queries (20% of unique queries, get 80% of traffic)
        hot_queries = [f"hot_query_{i}" for i in range(20)]

        # Cold queries (80% of unique queries, get 20% of traffic)
        cold_queries = [f"cold_query_{i}" for i in range(80)]

        def mock_compute():
            return {"result": "computed"}

        # Warm cache with hot queries
        for query in hot_queries:
            tiered_cache.cached("search", {"query": query}, mock_compute)

        # Reset stats after warming
        tiered_cache.l1.clear()
        tiered_cache.l1 = type(tiered_cache.l1)(
            max_size=tiered_cache.l1.max_size, ttl_seconds=tiered_cache.l1.ttl_seconds
        )

        # Simulate load: 1000 requests
        num_requests = 1000
        for _ in range(num_requests):
            # 80% hot, 20% cold
            if random.random() < 0.8:
                query = random.choice(hot_queries)
            else:
                query = random.choice(cold_queries)

            tiered_cache.cached("search", {"query": query}, mock_compute)

        # Check hit rate
        stats = tiered_cache.stats()

        # Combined hit rate should exceed 80%
        combined_rate = stats.get("combined_hit_rate", 0.0)

        print("\nCache performance stats:")
        print(f"  L1 stats: {stats['l1']}")
        print(f"  L2 stats: {stats['l2']}")
        print(f"  Combined hit rate: {combined_rate:.2%}")

        assert (
            combined_rate > 0.80
        ), f"Hit rate {combined_rate:.2%} does not meet 80% threshold"

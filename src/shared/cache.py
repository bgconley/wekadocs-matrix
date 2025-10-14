"""
L1 (in-process) + L2 (Redis) caching with version-prefixed keys.

Implements Phase 4, Task 4.3 (Caching & Performance)
See: /docs/spec.md §4.3 (Caching)
See: /docs/implementation-plan.md → Task 4.3
See: /docs/pseudocode-reference.md → Phase 4, Task 4.3

Cache keys are prefixed with {schema_version}:{embedding_version} to ensure
automatic invalidation when the model or schema changes.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

from src.shared.config import Config


class L1Cache:
    """In-process LRU cache with TTL support and size limits."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if present and not expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]
        if time.time() > expiry:
            # Expired
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with TTL."""
        expiry = time.time() + self.ttl_seconds

        if key in self._cache:
            # Update existing
            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)
        else:
            # Add new
            if len(self._cache) >= self.max_size:
                # Evict oldest
                self._cache.popitem(last=False)
            self._cache[key] = (value, expiry)

    def invalidate(self, key: str) -> None:
        """Remove specific key from cache."""
        if key in self._cache:
            del self._cache[key]

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys with given prefix. Returns count invalidated."""
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
        }


class L2Cache:
    """Redis-backed cache with version-prefixed keys."""

    def __init__(self, redis_client, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value_bytes = self.redis.get(key)
            if value_bytes is None:
                self._misses += 1
                return None

            self._hits += 1
            return json.loads(value_bytes)
        except Exception:
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Put value in Redis cache with TTL."""
        try:
            value_json = json.dumps(value)
            self.redis.setex(key, self.ttl_seconds, value_json)
        except Exception:
            # Log but don't fail if Redis unavailable
            pass

    def invalidate(self, key: str) -> None:
        """Remove specific key from cache."""
        try:
            self.redis.delete(key)
        except Exception:
            pass

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys with given prefix. Returns count invalidated."""
        try:
            # Use SCAN to find keys with prefix
            count = 0
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
                if keys:
                    self.redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception:
            return 0

    def clear_all(self) -> None:
        """Clear all keys (use with caution)."""
        try:
            self.redis.flushdb()
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class TieredCache:
    """
    Two-tier cache: L1 (in-process) + L2 (Redis).

    Keys are automatically prefixed with {schema_version}:{embedding_version}
    to ensure cache invalidation when versions change.
    """

    def __init__(
        self,
        config: Config,
        redis_client,
        schema_version: str,
        embedding_version: str,
    ):
        self.config = config
        self.schema_version = schema_version
        self.embedding_version = embedding_version

        # Initialize L1 cache
        l1_config = config.cache.l1
        self.l1 = (
            L1Cache(
                max_size=l1_config.max_size,
                ttl_seconds=l1_config.ttl_seconds,
            )
            if l1_config.enabled
            else None
        )

        # Initialize L2 cache
        l2_config = config.cache.l2
        self.l2 = (
            L2Cache(
                redis_client=redis_client,
                ttl_seconds=l2_config.ttl_seconds,
            )
            if l2_config.enabled and redis_client
            else None
        )

        self.key_prefix_base = l2_config.key_prefix

    def _make_cache_key(self, key_prefix: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key with version prefixes.

        Format: {base_prefix}:{schema_version}:{embedding_version}:{key_prefix}:{params_hash}
        """
        params_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:16]

        return f"{self.key_prefix_base}:{self.schema_version}:{self.embedding_version}:{key_prefix}:{params_hash}"

    def get(self, key_prefix: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get value from cache (L1 -> L2 -> None).

        Args:
            key_prefix: Logical cache key prefix (e.g., "search", "tool")
            params: Parameters that identify the cached computation

        Returns:
            Cached value if found, None otherwise
        """
        cache_key = self._make_cache_key(key_prefix, params)

        # Try L1 first
        if self.l1:
            value = self.l1.get(cache_key)
            if value is not None:
                return value

        # Try L2
        if self.l2:
            value = self.l2.get(cache_key)
            if value is not None:
                # Populate L1 for next time
                if self.l1:
                    self.l1.put(cache_key, value)
                return value

        return None

    def put(self, key_prefix: str, params: Dict[str, Any], value: Any) -> None:
        """
        Put value in both L1 and L2 caches.

        Args:
            key_prefix: Logical cache key prefix
            params: Parameters that identify the cached computation
            value: Value to cache
        """
        cache_key = self._make_cache_key(key_prefix, params)

        if self.l1:
            self.l1.put(cache_key, value)

        if self.l2:
            self.l2.put(cache_key, value)

    def cached(
        self,
        key_prefix: str,
        params: Dict[str, Any],
        compute_fn: Callable[[], Any],
    ) -> Any:
        """
        Get from cache or compute and cache.

        Args:
            key_prefix: Logical cache key prefix
            params: Parameters for the computation
            compute_fn: Function to compute value if not cached

        Returns:
            Cached or computed value
        """
        # Try cache first
        value = self.get(key_prefix, params)
        if value is not None:
            return value

        # Compute
        value = compute_fn()

        # Cache result
        self.put(key_prefix, params, value)

        return value

    def invalidate_version(
        self,
        schema_version: Optional[str] = None,
        embedding_version: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Invalidate all cache entries for specific version(s).

        Args:
            schema_version: Schema version to invalidate (or None for current)
            embedding_version: Embedding version to invalidate (or None for current)

        Returns:
            Dict with counts: {"l1": count, "l2": count}
        """
        sv = schema_version or self.schema_version
        ev = embedding_version or self.embedding_version

        prefix = f"{self.key_prefix_base}:{sv}:{ev}"

        l1_count = 0
        l2_count = 0

        if self.l1:
            l1_count = self.l1.invalidate_prefix(prefix)

        if self.l2:
            l2_count = self.l2.invalidate_prefix(prefix)

        return {"l1": l1_count, "l2": l2_count}

    def clear_all(self) -> None:
        """Clear all cache entries (use with caution)."""
        if self.l1:
            self.l1.clear()
        if self.l2:
            self.l2.clear_all()

    def stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        stats = {
            "schema_version": self.schema_version,
            "embedding_version": self.embedding_version,
        }

        if self.l1:
            stats["l1"] = self.l1.stats()

        if self.l2:
            stats["l2"] = self.l2.stats()

        # Combined hit rate
        if self.l1 and self.l2:
            total_hits = self.l1._hits + self.l2._hits
            total_requests = total_hits + self.l1._misses + self.l2._misses
            stats["combined_hit_rate"] = (
                total_hits / total_requests if total_requests > 0 else 0.0
            )

        return stats

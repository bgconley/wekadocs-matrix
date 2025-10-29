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

        # Initialize Prometheus metrics with layer label
        try:
            from src.shared.observability.metrics import (
                cache_hit_rate,
                cache_operations_total,
                cache_size_bytes,
            )

            cache_operations_total.labels(
                operation="get", layer="l1", result="hit"
            )  # Initialize label
            cache_hit_rate.labels(layer="l1").set(0.0)  # Initialize gauge
            cache_size_bytes.labels(layer="l1").set(0)  # Initialize gauge
        except Exception:
            pass  # Metrics not available in tests

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if present and not expired."""
        if key not in self._cache:
            self._misses += 1
            self._record_metrics("miss")
            return None

        value, expiry = self._cache[key]
        if time.time() > expiry:
            # Expired
            del self._cache[key]
            self._misses += 1
            self._record_metrics("miss")
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        self._record_metrics("hit")
        return value

    def _record_metrics(self, result: str):
        """Record cache operation metrics."""
        try:
            from src.shared.observability.metrics import (
                cache_hit_rate,
                cache_operations_total,
                cache_size_bytes,
            )

            cache_operations_total.labels(
                operation="get", layer="l1", result=result
            ).inc()
            total = self._hits + self._misses
            if total > 0:
                cache_hit_rate.labels(layer="l1").set(self._hits / total)
            cache_size_bytes.labels(layer="l1").set(len(self._cache))
        except Exception:
            pass

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

        # Initialize Prometheus metrics with layer label
        try:
            from src.shared.observability.metrics import (
                cache_hit_rate,
                cache_operations_total,
            )

            cache_operations_total.labels(
                operation="get", layer="l2", result="hit"
            )  # Initialize label
            cache_hit_rate.labels(layer="l2").set(0.0)  # Initialize gauge
        except Exception:
            pass  # Metrics not available in tests

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value_bytes = self.redis.get(key)
            if value_bytes is None:
                self._misses += 1
                self._record_metrics("miss")
                return None

            self._hits += 1
            self._record_metrics("hit")
            return json.loads(value_bytes)
        except Exception:
            self._misses += 1
            self._record_metrics("miss")
            return None

    def _record_metrics(self, result: str):
        """Record cache operation metrics."""
        try:
            from src.shared.observability.metrics import (
                cache_hit_rate,
                cache_operations_total,
            )

            cache_operations_total.labels(
                operation="get", layer="l2", result=result
            ).inc()
            total = self._hits + self._misses
            if total > 0:
                cache_hit_rate.labels(layer="l2").set(self._hits / total)
        except Exception:
            pass

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

        # Phase 7E-3: Epoch invalidation namespace (separate from cache key prefix)
        # Epochs are stored in simple HASHes: {namespace}:doc_epoch, {namespace}:chunk_epoch
        self.epoch_namespace = config.cache.invalidation.namespace

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

    def get_doc_epoch(self, document_id: str) -> str:
        """
        Get current document epoch from Redis (returns "0" if not set).

        Epoch-based invalidation (Phase 7E-3): Document epochs are stored in
        Redis hash {namespace}:doc_epoch with document_id as key.

        Args:
            document_id: Document identifier

        Returns:
            Current epoch as string (default "0")

        Reference: Canonical Spec L3230-3233
        """
        if not self.l2 or not self.l2.redis:
            return "0"

        try:
            epoch_key = f"{self.epoch_namespace}:doc_epoch"
            value = self.l2.redis.hget(epoch_key, document_id)
            if value is None:
                return "0"
            # Handle both bytes and str (depends on decode_responses setting)
            return value.decode() if isinstance(value, bytes) else str(value)
        except Exception:
            return "0"

    def get_chunk_epoch(self, chunk_id: str) -> str:
        """
        Get current chunk epoch from Redis (returns "0" if not set).

        Args:
            chunk_id: Chunk identifier

        Returns:
            Current epoch as string (default "0")

        Reference: Canonical Spec L3230-3233
        """
        if not self.l2 or not self.l2.redis:
            return "0"

        try:
            epoch_key = f"{self.epoch_namespace}:chunk_epoch"
            value = self.l2.redis.hget(epoch_key, chunk_id)
            if value is None:
                return "0"
            # Handle both bytes and str (depends on decode_responses setting)
            return value.decode() if isinstance(value, bytes) else str(value)
        except Exception:
            return "0"

    def make_fusion_cache_key(self, document_id: str, query_text: str) -> str:
        """
        Generate epoch-aware fusion cache key.

        Format: {ns}:fusion:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}

        Args:
            document_id: Document identifier
            query_text: Query text

        Returns:
            Epoch-aware cache key

        Reference: Canonical Spec L3198, L3234-3237
        """
        import hashlib

        epoch = self.get_doc_epoch(document_id)
        query_hash = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:16]

        return f"{self.key_prefix_base}:fusion:doc:{document_id}:epoch:{epoch}:q:{query_hash}"

    def make_answer_cache_key(self, document_id: str, query_text: str) -> str:
        """
        Generate epoch-aware answer cache key.

        Format: {ns}:answer:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}

        Args:
            document_id: Document identifier
            query_text: Query text

        Returns:
            Epoch-aware cache key

        Reference: Canonical Spec L3295
        """
        import hashlib

        epoch = self.get_doc_epoch(document_id)
        query_hash = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:16]

        return f"{self.key_prefix_base}:answer:doc:{document_id}:epoch:{epoch}:q:{query_hash}"

    def make_vector_cache_key(self, chunk_id: str) -> str:
        """
        Generate epoch-aware vector cache key.

        Format: {ns}:vector:chunk:{id}:epoch:{chunk_epoch}

        Args:
            chunk_id: Chunk identifier

        Returns:
            Epoch-aware cache key

        Reference: Canonical Spec L3207, L3298
        """
        epoch = self.get_chunk_epoch(chunk_id)
        return f"{self.key_prefix_base}:vector:chunk:{chunk_id}:epoch:{epoch}"

    def make_bm25_cache_key(self, document_id: str, query_text: str) -> str:
        """
        Generate epoch-aware BM25 cache key.

        Format: {ns}:bm25:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}

        Args:
            document_id: Document identifier
            query_text: Query text

        Returns:
            Epoch-aware cache key
        """
        import hashlib

        epoch = self.get_doc_epoch(document_id)
        query_hash = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:16]

        return f"{self.key_prefix_base}:bm25:doc:{document_id}:epoch:{epoch}:q:{query_hash}"

    def get_fusion_cached(self, document_id: str, query_text: str) -> Optional[Any]:
        """
        Get fusion results from epoch-aware cache.

        NOTE: Bypasses L1 cache to avoid stale reads. L1 (in-memory) cache
        cannot efficiently track epoch changes across distributed processes.
        L2 (Redis) is the source of truth for epoch-based invalidation.

        Args:
            document_id: Document identifier
            query_text: Query text

        Returns:
            Cached fusion results or None
        """
        cache_key = self.make_fusion_cache_key(document_id, query_text)

        # Skip L1 for epoch-based keys - epoch changes can't be efficiently
        # tracked in L1 across distributed processes
        if self.l2:
            return self.l2.get(cache_key)

        return None

    def put_fusion_cached(self, document_id: str, query_text: str, value: Any) -> None:
        """
        Put fusion results in epoch-aware cache.

        NOTE: Bypasses L1 cache to avoid stale reads. L1 (in-memory) cache
        cannot efficiently track epoch changes across distributed processes.
        L2 (Redis) is the source of truth for epoch-based invalidation.

        Args:
            document_id: Document identifier
            query_text: Query text
            value: Fusion results to cache
        """
        cache_key = self.make_fusion_cache_key(document_id, query_text)

        # Skip L1 for epoch-based keys - use L2 (Redis) only
        if self.l2:
            self.l2.put(cache_key, value)

    def get_vector_cached(self, chunk_id: str) -> Optional[Any]:
        """
        Get vector/embedding from epoch-aware cache.

        NOTE: Bypasses L1 cache to avoid stale reads. L1 (in-memory) cache
        cannot efficiently track epoch changes across distributed processes.
        L2 (Redis) is the source of truth for epoch-based invalidation.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Cached vector or None
        """
        cache_key = self.make_vector_cache_key(chunk_id)

        # Skip L1 for epoch-based keys - epoch changes can't be efficiently
        # tracked in L1 across distributed processes
        if self.l2:
            return self.l2.get(cache_key)

        return None

    def put_vector_cached(self, chunk_id: str, value: Any) -> None:
        """
        Put vector/embedding in epoch-aware cache.

        NOTE: Bypasses L1 cache to avoid stale reads. L1 (in-memory) cache
        cannot efficiently track epoch changes across distributed processes.
        L2 (Redis) is the source of truth for epoch-based invalidation.

        Args:
            chunk_id: Chunk identifier
            value: Vector to cache
        """
        cache_key = self.make_vector_cache_key(chunk_id)

        # Skip L1 for epoch-based keys - use L2 (Redis) only
        if self.l2:
            self.l2.put(cache_key, value)

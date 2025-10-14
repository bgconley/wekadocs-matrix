"""
Query cache warmer for preloading hot query patterns.

Implements Phase 4, Task 4.3 (Caching & Performance)
See: /docs/pseudocode-reference.md → Phase 4, Task 4.3 (warm_top_intents)
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryWarmer:
    """
    Warms cache by pre-executing common query patterns.

    Typical usage:
        warmer = QueryWarmer(cache, query_executor)
        warmer.warm_patterns([
            {"intent": "search", "query": "How to configure cluster?"},
            {"intent": "traverse", "start_id": "common_doc_id"},
        ])
    """

    def __init__(self, cache, query_executor: Callable[[str, Dict[str, Any]], Any]):
        """
        Initialize warmer.

        Args:
            cache: TieredCache instance
            query_executor: Function that executes queries (intent, params) -> result
        """
        self.cache = cache
        self.query_executor = query_executor
        self._warmed_count = 0
        self._failed_count = 0

    def warm_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Warm cache with list of query patterns.

        Args:
            patterns: List of dicts with query pattern info:
                - intent: Query intent (e.g., "search", "traverse")
                - params: Query parameters
                - key_prefix: Optional cache key prefix (defaults to intent)

        Returns:
            Stats dict with counts
        """
        started_at = datetime.utcnow()
        self._warmed_count = 0
        self._failed_count = 0

        logger.info(f"Starting cache warming with {len(patterns)} patterns")

        for pattern in patterns:
            intent = pattern.get("intent")
            params = pattern.get("params", {})
            key_prefix = pattern.get("key_prefix", intent)

            try:
                # Check if already cached
                cached_value = self.cache.get(key_prefix, params)
                if cached_value is not None:
                    logger.debug(f"Pattern {intent} already cached, skipping")
                    continue

                # Execute and cache
                result = self.query_executor(intent, params)
                self.cache.put(key_prefix, params, result)
                self._warmed_count += 1
                logger.debug(f"Warmed pattern {intent} with params {params}")

            except Exception as e:
                self._failed_count += 1
                logger.warning(f"Failed to warm pattern {intent}: {e}")

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        stats = {
            "warmed": self._warmed_count,
            "failed": self._failed_count,
            "total_patterns": len(patterns),
            "elapsed_seconds": elapsed,
        }

        logger.info(f"Cache warming complete: {stats}")
        return stats

    def warm_top_intents(
        self,
        top_n: int = 20,
        analytics_fn: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Warm cache with top N most frequent query intents.

        Args:
            top_n: Number of top intents to warm
            analytics_fn: Optional function that returns top intent patterns from analytics
                         If None, uses a default set of common patterns

        Returns:
            Stats dict
        """
        if analytics_fn:
            patterns = analytics_fn()[:top_n]
        else:
            # Default common patterns
            patterns = self._get_default_patterns()[:top_n]

        return self.warm_patterns(patterns)

    def _get_default_patterns(self) -> List[Dict[str, Any]]:
        """Get default common query patterns for warming."""
        return [
            {
                "intent": "search",
                "params": {"query": "cluster configuration", "filters": {}},
            },
            {
                "intent": "search",
                "params": {"query": "network setup", "filters": {}},
            },
            {
                "intent": "search",
                "params": {"query": "troubleshooting guide", "filters": {}},
            },
            {
                "intent": "search",
                "params": {"query": "installation steps", "filters": {}},
            },
            {
                "intent": "search",
                "params": {"query": "performance tuning", "filters": {}},
            },
        ]

    def stats(self) -> Dict[str, Any]:
        """Get warming statistics."""
        return {
            "warmed_count": self._warmed_count,
            "failed_count": self._failed_count,
        }

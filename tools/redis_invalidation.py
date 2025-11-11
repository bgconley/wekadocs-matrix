#!/usr/bin/env python3
"""
Pattern-scan cache invalidation (FALLBACK method).

Scans and deletes cache keys matching document/chunk patterns. Use this as
a fallback when epoch-based invalidation isn't available or for surgical
ops/debugging.

Usage:
  python tools/redis_invalidation.py --redis-url redis://localhost:6379/0 \
     --namespace rag:v1 --doc-id doc_abc123 \
     --chunks chunk_a1b2c3 chunk_d4e5f6 \
     --extra-pattern "{ns}:render:doc:{document_id}:*"

Recommended Patterns:
  - {ns}:search:doc:{document_id}:*
  - {ns}:fusion:doc:{document_id}:*
  - {ns}:answer:doc:{document_id}:*
  - {ns}:vector:chunk:{id}:*
  - {ns}:bm25:doc:{document_id}:*

Environment Variables:
  REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
  CACHE_NS: Cache namespace (default: rag:v1)

Reference: Canonical Spec L3116-3183 (Pattern-scan invalidation)
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def scan_delete(r: redis.Redis, pattern: str, batch: int = 1000) -> int:
    """
    Scan and delete keys matching pattern.

    Args:
        r: Redis client
        pattern: Pattern to match (e.g., "rag:v1:search:doc:*")
        batch: Batch size for SCAN operations

    Returns:
        Number of keys deleted
    """
    deleted = 0
    cursor = 0

    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=batch)

        if keys:
            deleted += r.delete(*keys)
            logger.debug(f"Deleted {len(keys)} keys matching {pattern}")

        if cursor == 0:
            break

    return deleted


def invalidate(
    redis_url: str,
    namespace: str,
    document_id: str,
    chunk_ids: Optional[List[str]] = None,
    extra_patterns: Optional[List[str]] = None,
) -> int:
    """
    Invalidate all cache entries for a document and its chunks.

    Args:
        redis_url: Redis connection URL
        namespace: Cache namespace
        document_id: Document identifier
        chunk_ids: Optional list of chunk identifiers
        extra_patterns: Optional custom patterns (use {ns} and {document_id} placeholders)

    Returns:
        Total number of keys deleted
    """
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    chunk_ids = chunk_ids or []

    # Build standard patterns
    patterns = [
        f"{namespace}:search:doc:{document_id}:*",
        f"{namespace}:bm25:doc:{document_id}:*",
        f"{namespace}:fusion:doc:{document_id}:*",
        f"{namespace}:answer:doc:{document_id}:*",
    ]

    # Add chunk-specific patterns
    for cid in chunk_ids:
        patterns.append(f"{namespace}:vector:chunk:{cid}:*")

    # Add custom patterns
    if extra_patterns:
        for p in extra_patterns:
            patterns.append(p.format(ns=namespace, document_id=document_id))

    # Deduplicate patterns
    unique_patterns = list(set(patterns))

    # Delete matching keys
    total = 0
    for pattern in unique_patterns:
        deleted = scan_delete(r, pattern)
        if deleted > 0:
            logger.debug(f"Pattern {pattern} → {deleted} keys deleted")
        total += deleted

    return total


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pattern-scan cache invalidation (fallback method)"
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis connection URL (env: REDIS_URL)",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("CACHE_NS", "rag:v1"),
        help="Cache namespace (env: CACHE_NS)",
    )
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document ID to invalidate",
    )
    parser.add_argument(
        "--chunks",
        nargs="*",
        default=[],
        help="Chunk IDs to invalidate (space-separated)",
    )
    parser.add_argument(
        "--extra-pattern",
        action="append",
        default=[],
        help="Additional patterns to match (can use {ns} and {document_id} placeholders)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Connect to Redis
        r = redis.Redis.from_url(args.redis_url, decode_responses=True)
        r.ping()  # Test connection
        logger.info(f"Connected to Redis at {args.redis_url}")

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    namespace = args.namespace
    document_id = args.doc_id
    chunk_ids = args.chunks
    extra_patterns = args.extra_pattern

    if args.dry_run:
        logger.info("DRY RUN MODE - no keys will be deleted")

        # Build patterns
        patterns = [
            f"{namespace}:search:doc:{document_id}:*",
            f"{namespace}:bm25:doc:{document_id}:*",
            f"{namespace}:fusion:doc:{document_id}:*",
            f"{namespace}:answer:doc:{document_id}:*",
        ]
        for cid in chunk_ids:
            patterns.append(f"{namespace}:vector:chunk:{cid}:*")
        if extra_patterns:
            for p in extra_patterns:
                patterns.append(p.format(ns=namespace, document_id=document_id))

        # Count keys without deleting
        logger.info("Would delete keys matching these patterns:")
        total_keys = 0
        for pattern in set(patterns):
            cursor = 0
            count = 0
            while True:
                cursor, keys = r.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            logger.info(f"  {pattern} → {count} keys")
            total_keys += count

        logger.info(f"Total keys that would be deleted: {total_keys}")
        return

    # Perform actual invalidation
    logger.info(f"Invalidating caches for document_id={document_id}")

    deleted = invalidate(
        args.redis_url, namespace, document_id, chunk_ids, extra_patterns
    )

    logger.info(f"✓ Deleted {deleted} keys for document_id={document_id}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Epoch-based cache invalidation (O(1) - PREFERRED method).

Bumps epoch counters for documents and chunks to invalidate all associated
cache entries without scanning or deleting keys. Old keys naturally become
stale when their epoch no longer matches the current epoch.

Usage:
  python tools/redis_epoch_bump.py --redis-url redis://localhost:6379/0 \
     --namespace rag:v1 --doc-id doc_abc123 \
     --chunks chunk_a1b2c3 chunk_d4e5f6

Environment Variables:
  REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
  CACHE_NS: Cache namespace (default: rag:v1)

Epoch Storage:
  - {namespace}:doc_epoch -> HSET {document_id} -> epoch_int
  - {namespace}:chunk_epoch -> HSET {chunk_id} -> epoch_int

Cache Key Format:
  - Fusion: {ns}:fusion:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}
  - Vector: {ns}:vector:chunk:{id}:epoch:{chunk_epoch}
  - Answer: {ns}:answer:doc:{document_id}:epoch:{doc_epoch}:q:{sha1}

Reference: Canonical Spec L3184-3281 (Epoch-based keys)
"""

import argparse
import logging
import os
import sys
from typing import List

import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def bump_doc_epoch(r: redis.Redis, ns: str, document_id: str) -> int:
    """
    Atomically increment document epoch counter.

    Args:
        r: Redis client
        ns: Cache namespace (e.g., "rag:v1")
        document_id: Document identifier

    Returns:
        New epoch value
    """
    epoch_key = f"{ns}:doc_epoch"
    new_epoch = r.hincrby(epoch_key, document_id, 1)
    logger.debug(f"Bumped doc_epoch[{document_id}] to {new_epoch}")
    return int(new_epoch)


def bump_chunk_epochs(r: redis.Redis, ns: str, chunk_ids: List[str]) -> int:
    """
    Atomically increment chunk epoch counters for multiple chunks.

    Args:
        r: Redis client
        ns: Cache namespace
        chunk_ids: List of chunk identifiers

    Returns:
        Sum of all new epoch values
    """
    if not chunk_ids:
        return 0

    epoch_key = f"{ns}:chunk_epoch"
    pipe = r.pipeline()
    for cid in chunk_ids:
        pipe.hincrby(epoch_key, cid, 1)

    results = pipe.execute()
    total = sum(int(x) for x in results)

    logger.debug(f"Bumped chunk_epoch for {len(chunk_ids)} chunks, total={total}")
    return total


def get_doc_epoch(r: redis.Redis, ns: str, document_id: str) -> int:
    """
    Get current document epoch (returns 0 if not set).

    Args:
        r: Redis client
        ns: Cache namespace
        document_id: Document identifier

    Returns:
        Current epoch value (0 if never set)
    """
    epoch_key = f"{ns}:doc_epoch"
    value = r.hget(epoch_key, document_id)
    return int(value) if value else 0


def get_chunk_epoch(r: redis.Redis, ns: str, chunk_id: str) -> int:
    """
    Get current chunk epoch (returns 0 if not set).

    Args:
        r: Redis client
        ns: Cache namespace
        chunk_id: Chunk identifier

    Returns:
        Current epoch value (0 if never set)
    """
    epoch_key = f"{ns}:chunk_epoch"
    value = r.hget(epoch_key, chunk_id)
    return int(value) if value else 0


def bump_global_epoch(r: redis.Redis, ns: str) -> int:
    """
    Bump global epoch for bulk invalidation.

    Use this for large batch re-ingests where you want to invalidate
    everything at once. Include global_epoch in cache keys to enable
    one-knob invalidation.

    Args:
        r: Redis client
        ns: Cache namespace

    Returns:
        New global epoch value
    """
    global_epoch_key = f"{ns}:global_epoch"
    new_epoch = r.incr(global_epoch_key)
    logger.info(f"Bumped global_epoch to {new_epoch}")
    return int(new_epoch)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bump epoch counters for cache invalidation (O(1) preferred method)"
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
        "--global",
        action="store_true",
        dest="bump_global",
        help="Also bump global epoch (for bulk invalidation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be bumped without actually bumping",
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

    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")
        current_doc_epoch = get_doc_epoch(r, namespace, document_id)
        logger.info(
            f"Would bump doc_epoch[{document_id}] from {current_doc_epoch} to {current_doc_epoch + 1}"
        )

        if chunk_ids:
            logger.info(f"Would bump chunk_epoch for {len(chunk_ids)} chunks:")
            for cid in chunk_ids[:5]:  # Show first 5
                current_chunk_epoch = get_chunk_epoch(r, namespace, cid)
                logger.info(
                    f"  - chunk_epoch[{cid}] from {current_chunk_epoch} to {current_chunk_epoch + 1}"
                )
            if len(chunk_ids) > 5:
                logger.info(f"  ... and {len(chunk_ids) - 5} more chunks")

        if args.bump_global:
            current_global = r.get(f"{namespace}:global_epoch") or 0
            logger.info(
                f"Would bump global_epoch from {current_global} to {int(current_global) + 1}"
            )

        return

    # Perform actual bumps
    logger.info(f"Invalidating caches for document_id={document_id}")

    # Bump document epoch
    doc_epoch = bump_doc_epoch(r, namespace, document_id)
    logger.info(f"✓ Bumped doc_epoch to {doc_epoch}")

    # Bump chunk epochs
    if chunk_ids:
        chunk_total = bump_chunk_epochs(r, namespace, chunk_ids)
        logger.info(
            f"✓ Bumped chunk_epoch for {len(chunk_ids)} chunks (total={chunk_total})"
        )
    else:
        logger.info("No chunks specified, skipping chunk_epoch bump")

    # Bump global epoch if requested
    if args.bump_global:
        global_epoch = bump_global_epoch(r, namespace)
        logger.info(f"✓ Bumped global_epoch to {global_epoch}")

    logger.info(
        f"Cache invalidation complete: doc_epoch={doc_epoch}, chunks={len(chunk_ids)}"
    )


if __name__ == "__main__":
    main()

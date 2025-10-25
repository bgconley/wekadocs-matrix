#!/usr/bin/env python3
"""
Session cleanup job for expired multi-turn conversation sessions.

Runs periodically to delete sessions older than TTL threshold (30 days).
Cascade deletes related Query and Answer nodes to prevent orphaned data.

Usage:
    # Dry run (show what would be deleted):
    python src/ops/session_cleanup_job.py --dry-run

    # Actual deletion:
    python src/ops/session_cleanup_job.py

    # Custom TTL:
    python src/ops/session_cleanup_job.py --ttl-days 60

This script is designed to be run as a cron job:
    # Add to crontab for daily cleanup at 2 AM:
    0 2 * * * cd /path/to/wekadocs-matrix && python src/ops/session_cleanup_job.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.query.session_tracker import SessionCleanupJob
from src.shared.config import get_config
from src.shared.connections import get_connection_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Run session cleanup job."""
    parser = argparse.ArgumentParser(
        description="Cleanup expired multi-turn conversation sessions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--ttl-days",
        type=int,
        default=30,
        help="Session TTL in days (default: 30)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (optional, uses default config if not provided)",
    )

    args = parser.parse_args()

    try:
        # Load configuration (initializes connection manager)
        if args.config:
            get_config(config_path=args.config)
        else:
            get_config()

        logger.info("Session cleanup job starting...")
        logger.info(f"  TTL: {args.ttl_days} days")
        logger.info(f"  Dry run: {args.dry_run}")

        # Get Neo4j connection
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()

        # Run cleanup
        cleanup = SessionCleanupJob(neo4j_driver, ttl_days=args.ttl_days)
        result = cleanup.run(dry_run=args.dry_run)

        # Report results
        if args.dry_run:
            logger.info("=" * 60)
            logger.info("DRY RUN COMPLETE - No data was deleted")
            logger.info("=" * 60)
            logger.info(
                f"  Sessions that would be deleted: {result['sessions_deleted']}"
            )
            logger.info(f"  Queries that would be deleted: {result['queries_deleted']}")
            logger.info(f"  Answers that would be deleted: {result['answers_deleted']}")
            logger.info(f"  Cutoff time: {result['cutoff_time']}")
        else:
            logger.info("=" * 60)
            logger.info("CLEANUP COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Sessions deleted: {result['sessions_deleted']}")
            logger.info(f"  Queries deleted: {result['queries_deleted']}")
            logger.info(f"  Answers deleted: {result['answers_deleted']}")
            logger.info(f"  Cutoff time: {result['cutoff_time']}")

        # Exit with appropriate code
        if result["sessions_deleted"] > 0:
            logger.info(
                f"Successfully cleaned up {result['sessions_deleted']} expired sessions"
            )
            sys.exit(0)
        else:
            logger.info("No expired sessions found")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Session cleanup job failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

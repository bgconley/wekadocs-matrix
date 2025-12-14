#!/usr/bin/env python3
"""
Migration Script: Deprecate :Section Label

Removes the :Section label from nodes that have both :Section and :Chunk labels,
and renames structural labels :CodeSection -> :CodeChunk, :TableSection -> :TableChunk.

This script is idempotent and safe to re-run.

Usage:
    python scripts/migrate_section_to_chunk.py --dry-run
    python scripts/migrate_section_to_chunk.py --execute

Environment Variables:
    NEO4J_URI: bolt://localhost:7687
    NEO4J_PASSWORD: testpassword123
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j package not installed. Run: pip install neo4j")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_driver():
    """Get Neo4j driver from environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "testpassword123")
    return GraphDatabase.driver(uri, auth=(user, password))


def get_migration_stats(driver) -> dict:
    """Get counts of nodes to be migrated."""
    with driver.session() as session:
        # Dual-labeled :Section:Chunk nodes
        dual_labeled = session.run(
            "MATCH (n:Section:Chunk) RETURN count(n) as count"
        ).single()["count"]

        # Orphan :Section nodes (no :Chunk label)
        orphan_sections = session.run(
            "MATCH (n:Section) WHERE NOT n:Chunk RETURN count(n) as count"
        ).single()["count"]

        # Structural labels to rename
        code_sections = session.run(
            "MATCH (n:CodeSection) RETURN count(n) as count"
        ).single()["count"]

        table_sections = session.run(
            "MATCH (n:TableSection) RETURN count(n) as count"
        ).single()["count"]

        return {
            "dual_labeled": dual_labeled,
            "orphan_sections": orphan_sections,
            "code_sections": code_sections,
            "table_sections": table_sections,
        }


def migrate_labels(driver, dry_run: bool = True) -> dict:
    """
    Migrate :Section labels to :Chunk.

    Steps:
    1. Remove :Section from dual-labeled nodes (:Section:Chunk -> :Chunk)
    2. Add :Chunk to orphan :Section nodes (:Section -> :Section:Chunk), then remove :Section
    3. Rename :CodeSection -> :CodeChunk
    4. Rename :TableSection -> :TableChunk

    Returns:
        dict with migration statistics
    """
    start_time = datetime.now()
    stats = get_migration_stats(driver)

    logger.info("=" * 70)
    logger.info("Section to Chunk Label Migration")
    logger.info("=" * 70)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    logger.info("")
    logger.info("Pre-migration state:")
    logger.info(f"  Dual-labeled (:Section:Chunk): {stats['dual_labeled']}")
    logger.info(f"  Orphan :Section nodes: {stats['orphan_sections']}")
    logger.info(f"  :CodeSection nodes: {stats['code_sections']}")
    logger.info(f"  :TableSection nodes: {stats['table_sections']}")
    logger.info("")

    if dry_run:
        logger.info("DRY RUN - No changes will be made")
        logger.info("")
        logger.info("Would perform:")
        logger.info(
            f"  1. Remove :Section from {stats['dual_labeled']} dual-labeled nodes"
        )
        logger.info(
            f"  2. Migrate {stats['orphan_sections']} orphan :Section nodes to :Chunk"
        )
        logger.info(f"  3. Rename {stats['code_sections']} :CodeSection -> :CodeChunk")
        logger.info(
            f"  4. Rename {stats['table_sections']} :TableSection -> :TableChunk"
        )
        return {
            "dry_run": True,
            "pre_stats": stats,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        }

    results = {"dry_run": False, "pre_stats": stats}

    with driver.session() as session:
        # Step 1: Remove :Section from dual-labeled nodes
        logger.info("Step 1: Removing :Section label from dual-labeled nodes...")
        result = session.run(
            """
            MATCH (n:Section:Chunk)
            REMOVE n:Section
            RETURN count(n) as migrated
        """
        )
        results["dual_labeled_migrated"] = result.single()["migrated"]
        logger.info(f"  Migrated: {results['dual_labeled_migrated']} nodes")

        # Step 2: Handle orphan :Section nodes (add :Chunk, then remove :Section)
        if stats["orphan_sections"] > 0:
            logger.info("Step 2: Migrating orphan :Section nodes...")
            result = session.run(
                """
                MATCH (n:Section)
                WHERE NOT n:Chunk
                SET n:Chunk
                REMOVE n:Section
                RETURN count(n) as migrated
            """
            )
            results["orphan_sections_migrated"] = result.single()["migrated"]
            logger.info(f"  Migrated: {results['orphan_sections_migrated']} nodes")
        else:
            results["orphan_sections_migrated"] = 0
            logger.info("Step 2: No orphan :Section nodes to migrate")

        # Step 3: Rename :CodeSection -> :CodeChunk
        if stats["code_sections"] > 0:
            logger.info("Step 3: Renaming :CodeSection -> :CodeChunk...")
            result = session.run(
                """
                MATCH (n:CodeSection)
                SET n:CodeChunk
                REMOVE n:CodeSection
                RETURN count(n) as renamed
            """
            )
            results["code_sections_renamed"] = result.single()["renamed"]
            logger.info(f"  Renamed: {results['code_sections_renamed']} nodes")
        else:
            results["code_sections_renamed"] = 0
            logger.info("Step 3: No :CodeSection nodes to rename")

        # Step 4: Rename :TableSection -> :TableChunk
        if stats["table_sections"] > 0:
            logger.info("Step 4: Renaming :TableSection -> :TableChunk...")
            result = session.run(
                """
                MATCH (n:TableSection)
                SET n:TableChunk
                REMOVE n:TableSection
                RETURN count(n) as renamed
            """
            )
            results["table_sections_renamed"] = result.single()["renamed"]
            logger.info(f"  Renamed: {results['table_sections_renamed']} nodes")
        else:
            results["table_sections_renamed"] = 0
            logger.info("Step 4: No :TableSection nodes to rename")

    # Verify migration
    logger.info("")
    logger.info("Verifying migration...")
    post_stats = get_migration_stats(driver)
    results["post_stats"] = post_stats

    logger.info("Post-migration state:")
    logger.info(f"  Dual-labeled (:Section:Chunk): {post_stats['dual_labeled']}")
    logger.info(f"  Orphan :Section nodes: {post_stats['orphan_sections']}")
    logger.info(f"  :CodeSection nodes: {post_stats['code_sections']}")
    logger.info(f"  :TableSection nodes: {post_stats['table_sections']}")

    # Check for success
    total_remaining = (
        post_stats["dual_labeled"]
        + post_stats["orphan_sections"]
        + post_stats["code_sections"]
        + post_stats["table_sections"]
    )

    if total_remaining == 0:
        logger.info("")
        logger.info("SUCCESS: All :Section labels have been migrated to :Chunk")
        results["success"] = True
    else:
        logger.warning("")
        logger.warning(f"WARNING: {total_remaining} nodes still need migration")
        results["success"] = False

    results["duration_seconds"] = (datetime.now() - start_time).total_seconds()
    logger.info(f"Duration: {results['duration_seconds']:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Migrate :Section labels to :Chunk in Neo4j"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration (required for actual changes)",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        logger.error("Must specify either --dry-run or --execute")
        parser.print_help()
        sys.exit(1)

    if args.dry_run and args.execute:
        logger.warning("Both --dry-run and --execute specified, treating as --execute")
        args.dry_run = False

    try:
        driver = get_driver()

        # Test connection
        with driver.session() as session:
            session.run("RETURN 1").single()
        logger.info("Connected to Neo4j")

        results = migrate_labels(driver, dry_run=args.dry_run)

        driver.close()

        if args.dry_run:
            sys.exit(0)
        else:
            sys.exit(0 if results.get("success", False) else 1)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

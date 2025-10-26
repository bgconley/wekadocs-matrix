#!/usr/bin/env python3
"""
Phase 7E.0 - Task 0.1: Document Token Backfill

Backfills Document.token_count by summing section token counts.
Critical prerequisite for validation queries in Task 0.2.

Usage:
    python scripts/backfill_document_tokens.py --dry-run
    python scripts/backfill_document_tokens.py --execute
    python scripts/backfill_document_tokens.py --execute --report reports/phase-7e/backfill-results.json

Features:
- Dry-run mode for safety
- Detailed logging with before/after stats
- JSON report generation
- Error handling with rollback
- Idempotent execution
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from neo4j import Driver

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.connections import get_connection_manager
from src.shared.observability.logging import setup_logging

logger = logging.getLogger(__name__)


class DocumentTokenBackfiller:
    """Backfills Document.token_count from section token counts"""

    def __init__(self, driver: Driver):
        self.driver = driver

    def get_documents_without_token_count(self) -> List[Dict]:
        """Find documents with null or zero token_count"""
        query = """
        MATCH (d:Document)
        WHERE d.token_count IS NULL OR d.token_count = 0
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        WITH d, count(s) as section_count
        RETURN d.id as doc_id,
               d.token_count as current_token_count,
               section_count
        ORDER BY doc_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_all_documents_stats(self) -> Dict:
        """Get statistics for all documents"""
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        WITH d,
             count(s) as section_count,
             sum(s.tokens) as calculated_tokens
        RETURN count(d) as total_documents,
               sum(CASE WHEN d.token_count IS NULL OR d.token_count = 0
                   THEN 1 ELSE 0 END) as missing_count,
               sum(CASE WHEN d.token_count > 0
                   THEN 1 ELSE 0 END) as has_count,
               avg(d.token_count) as avg_current_tokens,
               avg(calculated_tokens) as avg_calculated_tokens
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return dict(record) if record else {}

    def backfill_document_tokens(self, dry_run: bool = True) -> Dict:
        """
        Backfill Document.token_count from section sums.

        Args:
            dry_run: If True, only report what would be done

        Returns:
            Dict with execution statistics
        """
        start_time = datetime.now()

        # Get before stats
        logger.info("Gathering statistics before backfill...")
        before_stats = self.get_all_documents_stats()
        missing_docs = self.get_documents_without_token_count()

        logger.info(f"Total documents: {before_stats.get('total_documents', 0)}")
        logger.info(
            f"Documents missing token_count: {before_stats.get('missing_count', 0)}"
        )
        logger.info(f"Documents with token_count: {before_stats.get('has_count', 0)}")

        if not missing_docs:
            logger.info("✓ All documents already have token_count set")
            return {
                "success": True,
                "dry_run": dry_run,
                "documents_updated": 0,
                "before_stats": before_stats,
                "after_stats": before_stats,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        # Show what would be updated
        logger.info(f"Documents to update: {len(missing_docs)}")
        for doc in missing_docs[:5]:  # Show first 5
            logger.info(f"  - {doc['doc_id']}: sections={doc['section_count']}")
        if len(missing_docs) > 5:
            logger.info(f"  ... and {len(missing_docs) - 5} more")

        if dry_run:
            logger.info("DRY RUN: No changes made. Run with --execute to apply.")
            return {
                "success": True,
                "dry_run": True,
                "would_update": len(missing_docs),
                "documents_to_update": missing_docs,
                "before_stats": before_stats,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        # Execute backfill
        logger.info("Executing backfill...")
        backfill_query = """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
        WITH d, sum(s.tokens) AS section_tokens
        SET d.token_count = section_tokens
        RETURN d.id as doc_id,
               d.token_count as new_token_count
        """

        try:
            with self.driver.session() as session:
                result = session.run(backfill_query)
                updated_docs = [dict(record) for record in result]

            logger.info(f"✓ Successfully updated {len(updated_docs)} documents")

            # Get after stats
            after_stats = self.get_all_documents_stats()

            # Verify all documents now have token_count
            remaining_missing = after_stats.get("missing_count", 0)
            if remaining_missing > 0:
                logger.warning(
                    f"⚠ {remaining_missing} documents still missing token_count"
                )
            else:
                logger.info("✓ All documents now have token_count set")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Backfill completed in {duration:.2f}s")

            return {
                "success": True,
                "dry_run": False,
                "documents_updated": len(updated_docs),
                "updated_documents": updated_docs,
                "before_stats": before_stats,
                "after_stats": after_stats,
                "remaining_missing": remaining_missing,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error(f"Backfill failed: {e}", exc_info=True)
            return {
                "success": False,
                "dry_run": False,
                "error": str(e),
                "before_stats": before_stats,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Document.token_count from section sums"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the backfill (required for actual changes)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save JSON report (e.g., reports/phase-7e/backfill-results.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Validate arguments
    if not args.dry_run and not args.execute:
        logger.error("Must specify either --dry-run or --execute")
        parser.print_help()
        sys.exit(1)

    if args.dry_run and args.execute:
        logger.warning("Both --dry-run and --execute specified, treating as --execute")
        args.dry_run = False

    logger.info("=" * 80)
    logger.info("Phase 7E.0 - Task 0.1: Document Token Backfill")
    logger.info("=" * 80)
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    logger.info(f"Report: {args.report if args.report else 'None'}")
    logger.info("")

    try:
        # Get Neo4j connection
        conn_manager = get_connection_manager()
        driver = conn_manager.get_neo4j_driver()

        # Run backfill
        backfiller = DocumentTokenBackfiller(driver)
        results = backfiller.backfill_document_tokens(dry_run=args.dry_run)

        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Report saved to: {report_path}")

        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Success: {results['success']}")
        logger.info(f"Duration: {results['duration_seconds']:.2f}s")

        if results["dry_run"]:
            logger.info(f"Would update: {results.get('would_update', 0)} documents")
        else:
            logger.info(f"Documents updated: {results.get('documents_updated', 0)}")
            if results.get("remaining_missing", 0) == 0:
                logger.info("✓ All documents now have token_count")
            else:
                logger.warning(
                    f"⚠ {results['remaining_missing']} documents still missing token_count"
                )

        sys.exit(0 if results["success"] else 1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

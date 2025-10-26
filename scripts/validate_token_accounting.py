#!/usr/bin/env python3
"""
Phase 7E.0 - Task 0.2: Token Accounting Sanity Check

Validates that Document.token_count matches sum(Section.token_count) within 1% error.
Critical for ensuring data integrity before chunking implementation.

Usage:
    python scripts/validate_token_accounting.py
    python scripts/validate_token_accounting.py --threshold 0.01 --report reports/phase-7e/token-validation.json
    python scripts/validate_token_accounting.py --fail-on-violation

Features:
- Configurable error threshold (default 1%)
- Detailed per-document reporting
- Alert generation for violations
- Statistical summary
- JSON export for analysis
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


class TokenAccountingValidator:
    """Validates Document.token_count matches section sums"""

    def __init__(self, driver: Driver, threshold: float = 0.01):
        """
        Args:
            driver: Neo4j driver
            threshold: Maximum allowed error rate (default 1% = 0.01)
        """
        self.driver = driver
        self.threshold = threshold

    def validate_all_documents(self) -> Dict:
        """
        Validate token accounting for all documents.

        Returns:
            Dict with validation results and statistics
        """
        start_time = datetime.now()

        logger.info("Querying document token counts...")

        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        WITH d.id AS doc_id,
             d.token_count AS doc_tokens,
             sum(s.tokens) AS section_tokens_sum,
             count(s) AS section_count
        WHERE doc_tokens IS NOT NULL AND doc_tokens > 0
        WITH doc_id,
             doc_tokens,
             section_tokens_sum,
             section_count,
             doc_tokens - section_tokens_sum AS delta,
             CASE
                WHEN doc_tokens = 0 THEN 0.0
                ELSE abs(1.0 * (doc_tokens - section_tokens_sum) / doc_tokens)
             END AS error_rate
        RETURN doc_id,
               doc_tokens,
               section_tokens_sum,
               section_count,
               delta,
               error_rate
        ORDER BY error_rate DESC
        """

        with self.driver.session() as session:
            result = session.run(query)
            documents = [dict(record) for record in result]

        if not documents:
            logger.warning("No documents found with token_count set")
            return {
                "success": True,
                "documents_validated": 0,
                "violations": [],
                "statistics": {},
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        logger.info(f"Validated {len(documents)} documents")

        # Find violations
        violations = [doc for doc in documents if doc["error_rate"] > self.threshold]

        # Calculate statistics
        error_rates = [doc["error_rate"] for doc in documents]
        deltas = [doc["delta"] for doc in documents]

        statistics = {
            "total_documents": len(documents),
            "violations_count": len(violations),
            "violation_rate": len(violations) / len(documents) if documents else 0.0,
            "error_rate": {
                "min": min(error_rates) if error_rates else 0.0,
                "max": max(error_rates) if error_rates else 0.0,
                "avg": sum(error_rates) / len(error_rates) if error_rates else 0.0,
                "median": (
                    sorted(error_rates)[len(error_rates) // 2] if error_rates else 0.0
                ),
            },
            "delta": {
                "min": min(deltas) if deltas else 0,
                "max": max(deltas) if deltas else 0,
                "avg": sum(deltas) / len(deltas) if deltas else 0.0,
            },
            "threshold": self.threshold,
        }

        # Log summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Documents validated: {statistics['total_documents']}")
        logger.info(f"Violations found: {statistics['violations_count']}")
        logger.info(f"Violation rate: {statistics['violation_rate']:.2%}")
        logger.info(
            f"Error rate range: {statistics['error_rate']['min']:.4%} - {statistics['error_rate']['max']:.4%}"
        )
        logger.info(f"Average error rate: {statistics['error_rate']['avg']:.4%}")
        logger.info(f"Threshold: {self.threshold:.2%}")
        logger.info("")

        if violations:
            logger.warning(f"⚠ {len(violations)} VIOLATION(S) DETECTED")
            logger.warning("")
            logger.warning("Documents exceeding threshold:")
            for i, doc in enumerate(violations[:10], 1):  # Show first 10
                logger.warning(
                    f"  {i}. {doc['doc_id']}: "
                    f"doc_tokens={doc['doc_tokens']}, "
                    f"section_sum={doc['section_tokens_sum']}, "
                    f"delta={doc['delta']}, "
                    f"error={doc['error_rate']:.4%}"
                )
            if len(violations) > 10:
                logger.warning(f"  ... and {len(violations) - 10} more violations")
        else:
            logger.info("✓ All documents pass validation (within threshold)")

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Validation completed in {duration:.2f}s")

        return {
            "success": len(violations) == 0,
            "documents_validated": len(documents),
            "violations": violations,
            "all_documents": (
                documents if len(documents) <= 100 else None
            ),  # Only include full list if small
            "statistics": statistics,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }

    def get_documents_missing_token_count(self) -> List[Dict]:
        """Find documents without token_count set"""
        query = """
        MATCH (d:Document)
        WHERE d.token_count IS NULL OR d.token_count = 0
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        RETURN d.id as doc_id,
               d.token_count as token_count,
               count(s) as section_count
        ORDER BY doc_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]


def main():
    parser = argparse.ArgumentParser(
        description="Validate Document.token_count matches section sums"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Maximum allowed error rate (default: 0.01 = 1%%)",
    )
    parser.add_argument("--report", type=str, help="Path to save JSON report")
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit with error code if violations found",
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

    logger.info("=" * 80)
    logger.info("Phase 7E.0 - Task 0.2: Token Accounting Validation")
    logger.info("=" * 80)
    logger.info(f"Threshold: {args.threshold:.2%}")
    logger.info(f"Fail on violation: {args.fail_on_violation}")
    logger.info(f"Report: {args.report if args.report else 'None'}")
    logger.info("")

    try:
        # Get Neo4j connection
        conn_manager = get_connection_manager()
        driver = conn_manager.get_neo4j_driver()

        # Check for missing token_counts first
        validator = TokenAccountingValidator(driver, threshold=args.threshold)
        missing = validator.get_documents_missing_token_count()

        if missing:
            logger.warning(f"⚠ {len(missing)} documents missing token_count")
            logger.warning("Run backfill_document_tokens.py first:")
            logger.warning("  python scripts/backfill_document_tokens.py --execute")
            for doc in missing[:5]:
                logger.warning(f"  - {doc['doc_id']}")
            if len(missing) > 5:
                logger.warning(f"  ... and {len(missing) - 5} more")
            logger.warning("")

        # Run validation
        results = validator.validate_all_documents()

        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Report saved to: {report_path}")

        # Print final status
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL STATUS")
        logger.info("=" * 80)

        if results["success"]:
            logger.info("✓ PASS - All documents within threshold")
            exit_code = 0
        else:
            logger.error(f"✗ FAIL - {len(results['violations'])} violation(s) detected")
            exit_code = 1 if args.fail_on_violation else 0

        logger.info(f"Exit code: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

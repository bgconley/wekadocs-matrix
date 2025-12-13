#!/usr/bin/env python3
"""
One-time Neo4j structural edge migration script.

This script implements the migration from Section 2 of the Neo4j Overhaul Plan:
1. Create required indexes
2. Normalize parent_path → parent_path_norm for all chunks
3. Compute parent_chunk_id from parent_path_norm
4. Rebuild structural edges (HAS_CHUNK, NEXT_CHUNK, NEXT, hierarchy)
5. Normalize entity names
6. Run contract checks

Usage:
    # Dry run - show what would be done
    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
        python scripts/neo4j_structural_migration.py --dry-run

    # Execute migration
    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
        python scripts/neo4j_structural_migration.py --execute

    # Process specific documents
    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
        python scripts/neo4j_structural_migration.py --execute --doc-ids doc1,doc2
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = structlog.get_logger(__name__)


def get_neo4j_driver():
    """Create Neo4j driver from environment."""
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")

    return GraphDatabase.driver(uri, auth=(user, password))


def ensure_indexes(session) -> Dict[str, bool]:
    """Create required indexes if they don't exist."""
    indexes = {
        "chunk_doc_order": """
            CREATE RANGE INDEX chunk_doc_order IF NOT EXISTS
            FOR (c:Chunk) ON (c.document_id, c.order)
        """,
        "chunk_doc_parent_path_norm": """
            CREATE RANGE INDEX chunk_doc_parent_path_norm IF NOT EXISTS
            FOR (c:Chunk) ON (c.document_id, c.parent_path_norm)
        """,
        "chunk_parent_chunk_id": """
            CREATE RANGE INDEX chunk_parent_chunk_id IF NOT EXISTS
            FOR (c:Chunk) ON (c.parent_chunk_id)
        """,
        "entity_type_normalized_name": """
            CREATE RANGE INDEX entity_type_normalized_name IF NOT EXISTS
            FOR (e:Entity) ON (e.entity_type, e.normalized_name)
        """,
    }

    results = {}
    for name, query in indexes.items():
        try:
            session.run(query)
            results[name] = True
            logger.info("index_ensured", index=name)
        except Exception as e:
            results[name] = False
            logger.warning("index_creation_failed", index=name, error=str(e))

    return results


def get_all_document_ids(session) -> List[str]:
    """Get all document IDs with chunks."""
    query = """
    MATCH (c:Chunk)
    WHERE coalesce(c.document_id, c.doc_id) IS NOT NULL
    RETURN DISTINCT coalesce(c.document_id, c.doc_id) AS doc_id
    ORDER BY doc_id
    """
    result = session.run(query)
    return [record["doc_id"] for record in result]


def backfill_chunk_ids(session) -> Dict[str, int]:
    """Backfill Chunk.chunk_id from legacy Chunk.id where missing."""
    query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IS NULL AND c.id IS NOT NULL
    SET c.chunk_id = c.id
    RETURN count(c) AS updated
    """
    result = session.run(query)
    record = result.single()
    return {"updated": record["updated"] if record else 0}


def get_migration_stats(session) -> Dict[str, Any]:
    """Get current state statistics."""
    query = """
    MATCH (c:Chunk)
    WITH count(c) AS total_chunks
    MATCH (c:Chunk) WHERE c.parent_path_norm IS NOT NULL
    WITH total_chunks, count(c) AS has_parent_path_norm
    MATCH (c:Chunk) WHERE c.parent_chunk_id IS NOT NULL
    WITH total_chunks, has_parent_path_norm, count(c) AS has_parent_chunk_id
    OPTIONAL MATCH ()-[r:NEXT_CHUNK]->()
    WITH total_chunks, has_parent_path_norm, has_parent_chunk_id, count(r) AS next_chunk_edges
    OPTIONAL MATCH ()-[r:PARENT_HEADING]->()
    WITH total_chunks, has_parent_path_norm, has_parent_chunk_id, next_chunk_edges,
         count(r) AS parent_heading_edges
    OPTIONAL MATCH ()-[r:HAS_CHUNK]->()
    WITH total_chunks, has_parent_path_norm, has_parent_chunk_id, next_chunk_edges,
         parent_heading_edges, count(r) AS has_chunk_edges
    MATCH (e:Entity)
    WITH total_chunks, has_parent_path_norm, has_parent_chunk_id, next_chunk_edges,
         parent_heading_edges, has_chunk_edges, count(e) AS total_entities
    MATCH (e:Entity) WHERE e.normalized_name IS NOT NULL
    RETURN
        total_chunks,
        has_parent_path_norm,
        has_parent_chunk_id,
        next_chunk_edges,
        parent_heading_edges,
        has_chunk_edges,
        total_entities,
        count(e) AS normalized_entities
    """
    result = session.run(query)
    record = result.single()
    if record:
        return dict(record)
    return {}


def run_migration(
    driver,
    *,
    dry_run: bool = True,
    doc_ids: Optional[List[str]] = None,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the structural edge migration.

    Args:
        driver: Neo4j driver
        dry_run: If True, only show what would be done
        doc_ids: Optional list of specific document IDs to process
        report_path: Optional path to save JSON report

    Returns:
        Migration report dict
    """
    from src.neo.contract_checks import run_contract_checks
    from src.neo.entity_normalization import normalize_entities_backfill
    from src.neo.structural_builder import StructuralEdgeBuilder

    start_time = time.time()
    report = {
        "start_time": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "stats_before": {},
        "stats_after": {},
        "documents_processed": 0,
        "documents_failed": 0,
        "errors": [],
        "contract_check": None,
    }

    with driver.session() as session:
        # Get initial stats
        report["stats_before"] = get_migration_stats(session)
        logger.info("migration_starting", stats=report["stats_before"], dry_run=dry_run)

        if dry_run:
            logger.info("dry_run_mode", message="No changes will be made")
            if doc_ids:
                logger.info("would_process_documents", count=len(doc_ids))
            else:
                all_docs = get_all_document_ids(session)
                logger.info("would_process_documents", count=len(all_docs))
            return report

        # Step 1: Ensure indexes
        logger.info("step_1_creating_indexes")
        index_results = ensure_indexes(session)
        report["indexes_created"] = index_results

        # Step 1b: Backfill missing chunk_id values (legacy compatibility)
        logger.info("step_1b_backfilling_chunk_ids")
        chunk_id_stats = backfill_chunk_ids(session)
        report["chunk_id_backfill"] = chunk_id_stats

        # Step 2: Get documents to process
        if doc_ids:
            documents = doc_ids
        else:
            documents = get_all_document_ids(session)

        logger.info("documents_to_process", count=len(documents))

        # Step 3: Process each document
        builder = StructuralEdgeBuilder(session)
        for i, doc_id in enumerate(documents):
            try:
                result = builder.build_for_document(doc_id)
                if result.valid:
                    report["documents_processed"] += 1
                else:
                    report["documents_failed"] += 1
                    report["errors"].append(
                        {"document_id": doc_id, "violations": result.violations}
                    )

                if (i + 1) % 50 == 0:
                    logger.info(
                        "migration_progress",
                        processed=i + 1,
                        total=len(documents),
                        failed=report["documents_failed"],
                    )

            except Exception as e:
                report["documents_failed"] += 1
                report["errors"].append({"document_id": doc_id, "error": str(e)})
                logger.error(
                    "document_migration_failed", document_id=doc_id, error=str(e)
                )

        # Step 4: Normalize entities
        logger.info("step_4_normalizing_entities")
        entity_stats = normalize_entities_backfill(session)
        report["entity_normalization"] = entity_stats

        # Step 5: Get final stats
        report["stats_after"] = get_migration_stats(session)

        # Step 6: Run contract checks
        logger.info("step_6_running_contract_checks")
        contract_result = run_contract_checks(session, skip_expensive=False)
        report["contract_check"] = {
            "passed": contract_result.passed,
            "error_count": contract_result.error_count,
            "warning_count": contract_result.warning_count,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "message": c.message,
                }
                for c in contract_result.checks
            ],
        }

    report["end_time"] = datetime.utcnow().isoformat()
    report["duration_seconds"] = round(time.time() - start_time, 2)

    # Log summary
    logger.info(
        "migration_complete",
        documents_processed=report["documents_processed"],
        documents_failed=report["documents_failed"],
        contract_passed=report["contract_check"]["passed"],
        duration_seconds=report["duration_seconds"],
    )

    # Save report
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("report_saved", path=report_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Neo4j structural edge migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the migration",
    )
    parser.add_argument(
        "--doc-ids",
        type=str,
        help="Comma-separated list of document IDs to process",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save JSON report",
    )

    args = parser.parse_args()

    # Determine mode
    dry_run = not args.execute

    # Parse doc IDs
    doc_ids = None
    if args.doc_ids:
        doc_ids = [d.strip() for d in args.doc_ids.split(",")]

    # Default report path
    report_path = args.report
    if not report_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/migrations/structural_migration_{timestamp}.json"

    # Run migration
    driver = get_neo4j_driver()
    try:
        report = run_migration(
            driver,
            dry_run=dry_run,
            doc_ids=doc_ids,
            report_path=report_path if args.execute else None,
        )

        # Print summary
        print("\n" + "=" * 60)
        if dry_run:
            print("DRY RUN COMPLETE - No changes made")
        else:
            print("MIGRATION COMPLETE")
        print("=" * 60)
        print(f"Documents processed: {report.get('documents_processed', 'N/A')}")
        print(f"Documents failed: {report.get('documents_failed', 'N/A')}")

        if report.get("contract_check"):
            cc = report["contract_check"]
            status = "PASSED" if cc["passed"] else "FAILED"
            print(f"Contract check: {status}")
            print(f"  Errors: {cc['error_count']}, Warnings: {cc['warning_count']}")

        if not dry_run and report_path:
            print(f"\nReport saved to: {report_path}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()

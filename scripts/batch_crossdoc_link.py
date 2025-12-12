#!/usr/bin/env python3
"""
Batch cross-document linking script.

This script addresses the "early document linking gap" where documents ingested
before the corpus reached min_corpus_size (default: 3) never get outgoing
RELATED_TO edges.

Usage:
    # Show current edge status
    python scripts/batch_crossdoc_link.py --status

    # Preview which documents would be relinked
    python scripts/batch_crossdoc_link.py --missing --dry-run

    # Relink only documents missing outgoing edges
    python scripts/batch_crossdoc_link.py --missing --execute

    # Force relink all documents (idempotent, updates existing edges)
    python scripts/batch_crossdoc_link.py --all --execute

Reference:
    docs/cdx-outputs/2025-12-03-batch-crossdoc-relink-plan.md
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from src.services.cross_doc_linking import CrossDocLinker
from src.shared.config import load_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


@dataclass
class RelinkStats:
    """Statistics from a relink operation."""

    status: str = "pending"
    documents_found: int = 0
    documents_processed: int = 0
    edges_created: int = 0
    edges_updated: int = 0
    edges_pruned: int = 0
    skipped: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0
    error_details: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "documents_found": self.documents_found,
            "documents_processed": self.documents_processed,
            "edges_created": self.edges_created,
            "edges_updated": self.edges_updated,
            "edges_pruned": self.edges_pruned,
            "skipped": self.skipped,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def get_documents_missing_outgoing_edges(neo4j_driver) -> list[dict]:
    """
    Find documents with no outgoing RELATED_TO edges.

    These are documents that were ingested when corpus was < min_corpus_size,
    and therefore never had cross-doc linking run on them.

    Returns:
        List of dicts with 'id' and 'title' keys
    """
    query = """
    MATCH (d:Document)
    WHERE NOT (d)-[:RELATED_TO]->()
    RETURN d.doc_id as doc_id, d.title as title
    ORDER BY d.created_at
    """
    with neo4j_driver.session() as session:
        result = session.run(query)
        return [{"id": r["doc_id"], "title": r["title"] or ""} for r in result]


def get_documents_edge_summary(neo4j_driver) -> dict:
    """
    Get summary of documents with and without outgoing edges.

    Returns:
        Dict with 'with_edges', 'without_edges', 'total' counts
    """
    query = """
    MATCH (d:Document)
    OPTIONAL MATCH (d)-[r:RELATED_TO]->()
    WITH d, count(r) as outgoing_count
    RETURN
      sum(CASE WHEN outgoing_count = 0 THEN 1 ELSE 0 END) as without_edges,
      sum(CASE WHEN outgoing_count > 0 THEN 1 ELSE 0 END) as with_edges,
      count(d) as total
    """
    with neo4j_driver.session() as session:
        result = session.run(query).single()
        if result is None:
            return {"without_edges": 0, "with_edges": 0, "total": 0}
        return {
            "without_edges": result["without_edges"] or 0,
            "with_edges": result["with_edges"] or 0,
            "total": result["total"] or 0,
        }


def get_total_related_to_edges(neo4j_driver) -> int:
    """Get total count of RELATED_TO edges."""
    query = "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count"
    with neo4j_driver.session() as session:
        result = session.run(query).single()
        return result["count"] if result else 0


def relink_missing_documents(
    linker: CrossDocLinker,
    neo4j_driver,
    dry_run: bool = True,
) -> RelinkStats:
    """
    Relink only documents that have no outgoing RELATED_TO edges.

    Args:
        linker: CrossDocLinker instance
        neo4j_driver: Neo4j driver for queries
        dry_run: If True, only preview changes

    Returns:
        RelinkStats with operation results
    """
    stats = RelinkStats()
    docs = get_documents_missing_outgoing_edges(neo4j_driver)
    stats.documents_found = len(docs)

    logger.info(
        "found_documents_missing_edges",
        count=len(docs),
        dry_run=dry_run,
    )

    if len(docs) == 0:
        stats.status = "success"
        return stats

    if dry_run:
        # Preview mode - just list what would be done
        print(f"\n{'=' * 60}")
        print(f"DRY RUN: Would relink {len(docs)} documents")
        print(f"{'=' * 60}\n")

        for i, doc in enumerate(docs[:20]):  # Show first 20
            doc_id_short = doc["id"][:16] if doc["id"] else "unknown"
            title_short = (doc["title"] or "")[:50]
            print(f"  {i + 1:3}. {doc_id_short}... | {title_short}")

        if len(docs) > 20:
            print(f"  ... and {len(docs) - 20} more documents")

        print(f"\n{'=' * 60}")
        print("Run with --execute to perform relinking")
        print(f"{'=' * 60}\n")

        stats.status = "dry_run"
        return stats

    # Execute mode
    start_time = time.time()

    for i, doc in enumerate(docs):
        try:
            logger.info(
                "relinking_document",
                progress=f"{i + 1}/{len(docs)}",
                doc_id=doc["id"][:16] if doc["id"] else "unknown",
                title=(doc["title"] or "")[:40],
            )

            result = linker.link_document(
                doc_id=doc["id"],
                doc_title=doc.get("title", ""),
                dry_run=False,
            )

            stats.documents_processed += 1
            stats.edges_created += result.edges_created
            stats.edges_updated += result.edges_updated
            stats.edges_pruned += result.edges_pruned

            if result.skipped:
                stats.skipped += 1

        except Exception as e:
            logger.error(
                "relink_document_failed",
                doc_id=doc["id"],
                error=str(e),
            )
            stats.errors += 1
            stats.error_details.append({"doc_id": doc["id"], "error": str(e)})

    stats.elapsed_seconds = time.time() - start_time
    stats.status = "success" if stats.errors == 0 else "partial_success"

    return stats


def relink_all_documents(
    linker: CrossDocLinker,
    method: str = "rrf",
    dry_run: bool = True,
    limit: int | None = None,
) -> RelinkStats:
    """
    Relink all documents using link_all_documents().

    This is idempotent - existing edges will be updated with new scores.

    Args:
        linker: CrossDocLinker instance
        method: Linking method ('rrf' or 'dense')
        dry_run: If True, only preview changes
        limit: Limit number of documents (for testing)

    Returns:
        RelinkStats with operation results
    """
    stats = RelinkStats()

    logger.info(
        "relinking_all_documents",
        method=method,
        dry_run=dry_run,
        limit=limit,
    )

    start_time = time.time()

    results = linker.link_all_documents(
        method=method,
        dry_run=dry_run,
        limit=limit,
    )

    stats.documents_found = len(results)
    stats.documents_processed = len(results)

    for result in results:
        stats.edges_created += result.edges_created
        stats.edges_updated += result.edges_updated
        stats.edges_pruned += result.edges_pruned
        if result.skipped:
            stats.skipped += 1

    stats.elapsed_seconds = time.time() - start_time
    stats.status = "dry_run" if dry_run else "success"

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch cross-document linking for WekaDocs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current edge status
  python scripts/batch_crossdoc_link.py --status

  # Preview documents missing outgoing edges
  python scripts/batch_crossdoc_link.py --missing --dry-run

  # Relink only documents missing edges
  python scripts/batch_crossdoc_link.py --missing --execute

  # Force relink all documents (idempotent, updates existing)
  python scripts/batch_crossdoc_link.py --all --execute

  # Relink all with limit (for testing)
  python scripts/batch_crossdoc_link.py --all --execute --limit 10
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Relink all documents (idempotent)",
    )
    mode_group.add_argument(
        "--missing",
        action="store_true",
        help="Only relink documents missing outgoing edges",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current edge status without making changes",
    )

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    action_group.add_argument(
        "--execute",
        action="store_true",
        help="Execute the relinking operation",
    )

    parser.add_argument(
        "--method",
        default="rrf",
        choices=["rrf", "dense"],
        help="Linking method (default: rrf)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.status and not (args.dry_run or args.execute):
        parser.error("Must specify --dry-run or --execute (unless using --status)")

    # Load configuration
    try:
        config, settings = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Create database clients
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "testpassword123")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    neo4j_driver = GraphDatabase.driver(
        neo4j_uri,
        auth=("neo4j", neo4j_password),
    )

    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    try:
        # Test connections
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        qdrant_client.get_collections()

        if args.status:
            # Just show current status
            summary = get_documents_edge_summary(neo4j_driver)
            total_edges = get_total_related_to_edges(neo4j_driver)

            print(f"\n{'=' * 60}")
            print("Cross-Document Linking Status")
            print(f"{'=' * 60}")
            print(f"  Total documents:          {summary['total']}")
            print(f"  With outgoing edges:      {summary['with_edges']}")
            print(f"  Missing outgoing edges:   {summary['without_edges']}")
            print(f"  Total RELATED_TO edges:   {total_edges}")
            print(f"{'=' * 60}\n")

            if summary["without_edges"] > 0:
                print("Run with --missing --execute to relink missing documents")
                # Show sample of missing docs
                missing_docs = get_documents_missing_outgoing_edges(neo4j_driver)
                if missing_docs:
                    print("\nSample of documents missing edges:")
                    for doc in missing_docs[:5]:
                        doc_id = doc["id"][:20] if doc["id"] else "unknown"
                        title = (doc["title"] or "")[:40]
                        print(f"  - {doc_id}... | {title}")
                    if len(missing_docs) > 5:
                        print(f"  ... and {len(missing_docs) - 5} more")
            else:
                print("All documents have outgoing cross-doc links!")
            print()

            sys.exit(0)

        # Create linker with config (nested under ingestion)
        ingestion_config = getattr(config, "ingestion", None)
        cross_doc_config = (
            getattr(ingestion_config, "cross_doc_linking", None)
            if ingestion_config
            else None
        )
        if cross_doc_config is None:
            print(
                "Error: cross_doc_linking config not found (expected at config.ingestion.cross_doc_linking)"
            )
            sys.exit(1)

        linker = CrossDocLinker(
            neo4j_driver=neo4j_driver,
            qdrant_client=qdrant_client,
            config=cross_doc_config,
        )

        if args.missing:
            stats = relink_missing_documents(
                linker=linker,
                neo4j_driver=neo4j_driver,
                dry_run=args.dry_run,
            )
        elif args.all:
            stats = relink_all_documents(
                linker=linker,
                method=args.method,
                dry_run=args.dry_run,
                limit=args.limit,
            )

        # Print results
        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}")
        for key, value in stats.to_dict().items():
            print(f"  {key}: {value}")
        print(f"{'=' * 60}\n")

        # Exit with error code if there were failures
        if stats.errors > 0:
            sys.exit(1)

    except Exception as e:
        logger.error("batch_relink_failed", error=str(e))
        print(f"\nError: {e}")
        sys.exit(1)

    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    main()

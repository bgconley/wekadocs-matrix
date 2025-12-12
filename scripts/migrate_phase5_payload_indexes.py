#!/usr/bin/env python3
"""
Phase 5 Qdrant Payload Index Migration Script.

Adds the Phase 5 structural metadata indexes to existing Qdrant collections:
- line_end (INTEGER) - Source line end for complete range
- parent_path_depth (INTEGER) - Nesting level for depth filtering
- block_type (KEYWORD) - Dominant block type for filtering

This script is idempotent - safe to run multiple times.
Existing indexes will be skipped without error.

Usage:
    # Dry run (default) - shows what would be done
    python scripts/migrate_phase5_payload_indexes.py

    # Execute migration
    python scripts/migrate_phase5_payload_indexes.py --execute

    # Specify collection name
    python scripts/migrate_phase5_payload_indexes.py --collection chunks_multi_bge_m3 --execute

Environment:
    QDRANT_HOST: Qdrant host (default: localhost)
    QDRANT_PORT: Qdrant port (default: 6333)
"""

import argparse
import os
import sys
from typing import List, Tuple

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_phase5_indexes() -> List[Tuple[str, str]]:
    """Return the Phase 5 payload indexes to create."""
    return [
        ("line_end", "integer"),
        ("parent_path_depth", "integer"),
        ("block_type", "keyword"),
    ]


def migrate_indexes(
    collection_name: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    dry_run: bool = True,
) -> dict:
    """
    Add Phase 5 payload indexes to a Qdrant collection.

    Args:
        collection_name: Name of the collection to migrate
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        dry_run: If True, only show what would be done

    Returns:
        Dict with migration results
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PayloadSchemaType

    # Map string types to PayloadSchemaType
    type_map = {
        "integer": PayloadSchemaType.INTEGER,
        "float": PayloadSchemaType.FLOAT,
        "keyword": PayloadSchemaType.KEYWORD,
        "text": PayloadSchemaType.TEXT,
        "bool": PayloadSchemaType.BOOL,
    }

    results = {
        "collection": collection_name,
        "dry_run": dry_run,
        "indexes_created": [],
        "indexes_skipped": [],
        "errors": [],
    }

    print(f"\n{'=' * 60}")
    print("Phase 5 Qdrant Payload Index Migration")
    print(f"{'=' * 60}")
    print(f"Collection: {collection_name}")
    print(f"Host: {qdrant_host}:{qdrant_port}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"{'=' * 60}\n")

    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Verify collection exists
        try:
            collection_info = client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' found")
            print(f"  Points: {collection_info.points_count}")
        except Exception as e:
            error_msg = f"Collection '{collection_name}' not found: {e}"
            results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return results

        # Get existing payload schema
        existing_schema = {}
        if (
            hasattr(collection_info, "payload_schema")
            and collection_info.payload_schema
        ):
            existing_schema = collection_info.payload_schema
        print(f"  Existing payload indexes: {len(existing_schema)}")

        # Process each Phase 5 index
        indexes = get_phase5_indexes()
        print(f"\nPhase 5 indexes to create: {len(indexes)}")

        for field_name, field_type in indexes:
            schema_type = type_map.get(field_type)
            if not schema_type:
                error_msg = f"Unknown field type: {field_type}"
                results["errors"].append(error_msg)
                print(f"  ✗ {field_name}: {error_msg}")
                continue

            # Check if already exists
            if field_name in existing_schema:
                results["indexes_skipped"].append(field_name)
                print(f"  ⊘ {field_name}: Already exists (skipped)")
                continue

            if dry_run:
                print(f"  → {field_name}: Would create as {field_type}")
                results["indexes_created"].append(field_name)
            else:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                    results["indexes_created"].append(field_name)
                    print(f"  ✓ {field_name}: Created as {field_type}")
                except Exception as e:
                    # Handle "already exists" gracefully
                    if "already exists" in str(e).lower():
                        results["indexes_skipped"].append(field_name)
                        print(f"  ⊘ {field_name}: Already exists (skipped)")
                    else:
                        error_msg = f"Failed to create {field_name}: {e}"
                        results["errors"].append(error_msg)
                        print(f"  ✗ {field_name}: {error_msg}")

    except Exception as e:
        error_msg = f"Connection failed: {e}"
        results["errors"].append(error_msg)
        print(f"\n✗ {error_msg}")
        return results

    # Summary
    print(f"\n{'=' * 60}")
    print("Migration Summary")
    print(f"{'=' * 60}")
    print(f"Created:  {len(results['indexes_created'])}")
    print(f"Skipped:  {len(results['indexes_skipped'])}")
    print(f"Errors:   {len(results['errors'])}")

    if dry_run and results["indexes_created"]:
        print("\n⚠ This was a DRY RUN. Run with --execute to apply changes.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Add Phase 5 structural metadata indexes to Qdrant collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--collection",
        default="chunks_multi_bge_m3",
        help="Collection name (default: chunks_multi_bge_m3)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute migration (default is dry run)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("QDRANT_HOST", "localhost"),
        help="Qdrant host (default: localhost or QDRANT_HOST env)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QDRANT_PORT", "6333")),
        help="Qdrant port (default: 6333 or QDRANT_PORT env)",
    )

    args = parser.parse_args()

    results = migrate_indexes(
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        dry_run=not args.execute,
    )

    # Exit code based on errors
    sys.exit(1 if results["errors"] else 0)


if __name__ == "__main__":
    main()

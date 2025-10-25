#!/usr/bin/env python3
"""
Apply complete standalone v2.1 schema (Session 06-07).

This script applies the complete v2.1 schema in one atomic operation,
using the fixed multi-line parser from src/shared/schema.py.

Usage:
    python scripts/apply_complete_schema_v2_1.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger
from src.shared.schema import parse_cypher_statements

logger = get_logger(__name__)


def apply_complete_v2_1_schema():
    """Apply complete v2.1 schema to clean database."""

    logger.info("Starting complete v2.1 schema application")

    # Get Neo4j connection
    manager = get_connection_manager()
    driver = manager.get_neo4j_driver()

    # Read complete v2.1 schema
    schema_path = Path(__file__).parent / "neo4j" / "create_schema_v2_1_complete.cypher"

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return {"success": False, "error": f"File not found: {schema_path}"}

    logger.info(f"Reading schema from: {schema_path}")

    with open(schema_path, "r") as f:
        cypher_script = f.read()

    # Parse statements using fixed parser
    statements = parse_cypher_statements(cypher_script)

    logger.info(f"Parsed {len(statements)} statements from schema")

    # Execute statements
    results = {
        "success": False,
        "total_statements": len(statements),
        "executed": 0,
        "constraints_created": 0,
        "indexes_created": 0,
        "vector_indexes_created": 0,
        "schema_version_set": False,
        "dual_labeled_sections": 0,
        "errors": [],
    }

    with driver.session() as session:
        for idx, stmt in enumerate(statements, 1):
            try:
                result = session.run(stmt)
                results["executed"] += 1

                # Count by type
                if "CREATE CONSTRAINT" in stmt:
                    results["constraints_created"] += 1
                elif "CREATE VECTOR INDEX" in stmt:
                    results["vector_indexes_created"] += 1
                elif "CREATE INDEX" in stmt:
                    results["indexes_created"] += 1
                elif "MERGE (sv:SchemaVersion" in stmt:
                    results["schema_version_set"] = True
                elif "SET s:Chunk" in stmt:
                    summary = result.consume()
                    results["dual_labeled_sections"] = summary.counters.labels_added

                # Log progress every 10 statements
                if idx % 10 == 0:
                    logger.info(
                        f"Progress: {idx}/{len(statements)} statements executed"
                    )

            except Exception as e:
                error_msg = str(e)
                # Ignore "already exists" errors (idempotent)
                if (
                    "already exists" in error_msg.lower()
                    or "equivalent" in error_msg.lower()
                ):
                    logger.debug(
                        f"Statement {idx} already exists (idempotent): {error_msg[:100]}"
                    )
                    results["executed"] += 1
                else:
                    logger.error(f"Error executing statement {idx}: {error_msg}")
                    results["errors"].append(
                        {
                            "statement_num": idx,
                            "error": error_msg[:200],
                            "statement_preview": stmt[:100],
                        }
                    )

    results["success"] = results["executed"] == len(statements)

    # Log summary
    logger.info(
        "Schema application complete",
        success=results["success"],
        executed=results["executed"],
        total=results["total_statements"],
        constraints=results["constraints_created"],
        indexes=results["indexes_created"],
        vector_indexes=results["vector_indexes_created"],
        errors=len(results["errors"]),
    )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Applying Complete v2.1 Schema (Session 06-07)")
    print("=" * 60)
    print()

    results = apply_complete_v2_1_schema()

    print()
    print("=" * 60)
    print("Schema Application Results")
    print("=" * 60)
    print(f"Success: {results['success']}")
    print(f"Statements executed: {results['executed']}/{results['total_statements']}")
    print(f"Constraints created: {results['constraints_created']}")
    print(f"Regular indexes created: {results['indexes_created']}")
    print(f"Vector indexes created: {results['vector_indexes_created']}")
    print(f"SchemaVersion set: {results['schema_version_set']}")
    print(f"Dual-labeled sections: {results['dual_labeled_sections']}")

    if results["errors"]:
        print(f"\nErrors encountered: {len(results['errors'])}")
        for err in results["errors"][:5]:  # Show first 5 errors
            print(f"  Statement {err['statement_num']}: {err['error']}")

    print("=" * 60)

    sys.exit(0 if results["success"] else 1)

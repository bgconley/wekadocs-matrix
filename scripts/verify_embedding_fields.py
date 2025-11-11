#!/usr/bin/env python3
"""
Verify embedding field canonicalization across Neo4j and Qdrant.

This script ensures that all persisted data uses the canonical field names
and that no legacy `embedding_model` fields exist in either store.

Exit codes:
  0 - All checks passed
  1 - Violations found
  2 - Connection or configuration error
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict

import structlog
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Configure logging
logger = structlog.get_logger(__name__)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword123")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "weka_sections_v2")

# Canonical values
CANONICAL_VERSION = "jina-embeddings-v3"
CANONICAL_PROVIDER = "jina-ai"
CANONICAL_DIMENSIONS = 1024


class EmbeddingFieldVerifier:
    """Verifies embedding field canonicalization."""

    def __init__(self):
        self.neo4j_driver = None
        self.qdrant_client = None
        self.violations = []
        self.stats = {
            "neo4j_checked": 0,
            "neo4j_valid": 0,
            "neo4j_invalid": 0,
            "qdrant_checked": 0,
            "qdrant_valid": 0,
            "qdrant_invalid": 0,
        }

    def connect(self) -> bool:
        """Connect to databases."""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j", uri=NEO4J_URI)

            self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            # Test connection
            self.qdrant_client.get_collections()
            logger.info("Connected to Qdrant", host=QDRANT_HOST, port=QDRANT_PORT)

            return True
        except Exception as e:
            logger.error("Failed to connect", error=str(e))
            return False

    def verify_neo4j(self) -> bool:
        """Verify Neo4j nodes have canonical fields."""
        logger.info("Verifying Neo4j nodes...")

        with self.neo4j_driver.session() as session:
            # Check for nodes with legacy field (excluding SchemaVersion which is allowed)
            result = session.run(
                """
                MATCH (n)
                WHERE n.embedding_model IS NOT NULL
                  AND NOT 'SchemaVersion' IN labels(n)
                RETURN labels(n)[0] as label, n.id as id, n.embedding_model as model
                LIMIT 100
                """
            )
            legacy_nodes = list(result)

            if legacy_nodes:
                for node in legacy_nodes:
                    self.violations.append(
                        {
                            "store": "Neo4j",
                            "type": "legacy_field",
                            "label": node["label"],
                            "id": node["id"],
                            "field": "embedding_model",
                            "value": node["model"],
                        }
                    )
                    self.stats["neo4j_invalid"] += 1
                logger.error(
                    f"Found {len(legacy_nodes)} nodes with legacy embedding_model"
                )

            # Note about SchemaVersion exception
            result = session.run(
                """
                MATCH (n:SchemaVersion)
                WHERE n.embedding_model IS NOT NULL
                RETURN count(n) as count
                """
            )
            schema_version_count = result.single()["count"]
            if schema_version_count > 0:
                logger.info(
                    "SchemaVersion node has embedding_model (allowed as metadata)",
                    count=schema_version_count,
                )

            # Check Section and Chunk nodes have canonical fields
            for label in ["Section", "Chunk"]:
                # Count nodes with proper fields
                result = session.run(
                    f"""
                    MATCH (n:{label})
                    WHERE n.embedding_version IS NOT NULL
                    RETURN count(n) as count
                    """
                )
                with_version = result.single()["count"]

                # Count total nodes
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                total = result.single()["count"]

                self.stats["neo4j_checked"] += total

                # Check ALL nodes for detailed validation (no LIMIT for accurate stats)
                result = session.run(
                    f"""
                    MATCH (n:{label})
                    RETURN n.id as id,
                           n.embedding_version as version,
                           n.embedding_dimensions as dims,
                           n.embedding_provider as provider,
                           n.embedding_timestamp as timestamp,
                           n.embedding_model as legacy_model
                    """
                )

                for record in result:
                    node_valid = True
                    node_id = record["id"]

                    # Check no legacy field
                    if record["legacy_model"] is not None:
                        self.violations.append(
                            {
                                "store": "Neo4j",
                                "type": "legacy_field_present",
                                "label": label,
                                "id": node_id,
                                "field": "embedding_model",
                                "value": record["legacy_model"],
                            }
                        )
                        node_valid = False

                    # Check required canonical fields
                    if record["version"] != CANONICAL_VERSION:
                        self.violations.append(
                            {
                                "store": "Neo4j",
                                "type": "invalid_version",
                                "label": label,
                                "id": node_id,
                                "expected": CANONICAL_VERSION,
                                "actual": record["version"],
                            }
                        )
                        node_valid = False

                    if record["dims"] != CANONICAL_DIMENSIONS:
                        self.violations.append(
                            {
                                "store": "Neo4j",
                                "type": "invalid_dimensions",
                                "label": label,
                                "id": node_id,
                                "expected": CANONICAL_DIMENSIONS,
                                "actual": record["dims"],
                            }
                        )
                        node_valid = False

                    if record["provider"] != CANONICAL_PROVIDER:
                        self.violations.append(
                            {
                                "store": "Neo4j",
                                "type": "invalid_provider",
                                "label": label,
                                "id": node_id,
                                "expected": CANONICAL_PROVIDER,
                                "actual": record["provider"],
                            }
                        )
                        node_valid = False

                    if node_valid:
                        self.stats["neo4j_valid"] += 1
                    else:
                        self.stats["neo4j_invalid"] += 1

                logger.info(
                    f"{label} nodes verified",
                    total=total,
                    with_version=with_version,
                    percentage=(
                        f"{(with_version/total*100):.1f}%" if total > 0 else "N/A"
                    ),
                )

        return len([v for v in self.violations if v["store"] == "Neo4j"]) == 0

    def verify_qdrant(self) -> bool:
        """Verify Qdrant points have canonical fields."""
        logger.info("Verifying Qdrant points...")

        try:
            # Check collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if QDRANT_COLLECTION not in collection_names:
                logger.warning(
                    f"Collection {QDRANT_COLLECTION} not found",
                    available=collection_names,
                )
                return True  # Not an error if collection doesn't exist yet

            # Get collection info
            collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
            total_points = collection_info.points_count

            # Scroll through points and check payloads
            offset = None
            checked = 0
            batch_size = 100

            while checked < total_points:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    break

                for point in points:
                    self.stats["qdrant_checked"] += 1
                    point_valid = True
                    payload = point.payload

                    # Check no legacy field
                    if payload and "embedding_model" in payload:
                        self.violations.append(
                            {
                                "store": "Qdrant",
                                "type": "legacy_field_present",
                                "id": str(point.id),
                                "field": "embedding_model",
                                "value": payload["embedding_model"],
                            }
                        )
                        point_valid = False

                    # Check required canonical fields
                    if payload:
                        if payload.get("embedding_version") != CANONICAL_VERSION:
                            self.violations.append(
                                {
                                    "store": "Qdrant",
                                    "type": "invalid_version",
                                    "id": str(point.id),
                                    "expected": CANONICAL_VERSION,
                                    "actual": payload.get("embedding_version"),
                                }
                            )
                            point_valid = False

                        if payload.get("embedding_dimensions") != CANONICAL_DIMENSIONS:
                            self.violations.append(
                                {
                                    "store": "Qdrant",
                                    "type": "invalid_dimensions",
                                    "id": str(point.id),
                                    "expected": CANONICAL_DIMENSIONS,
                                    "actual": payload.get("embedding_dimensions"),
                                }
                            )
                            point_valid = False

                        if payload.get("embedding_provider") != CANONICAL_PROVIDER:
                            self.violations.append(
                                {
                                    "store": "Qdrant",
                                    "type": "invalid_provider",
                                    "id": str(point.id),
                                    "expected": CANONICAL_PROVIDER,
                                    "actual": payload.get("embedding_provider"),
                                }
                            )
                            point_valid = False

                    if point_valid:
                        self.stats["qdrant_valid"] += 1
                    else:
                        self.stats["qdrant_invalid"] += 1

                checked += len(points)
                offset = next_offset

                if not next_offset:
                    break

            logger.info(
                "Qdrant points verified",
                total=total_points,
                checked=self.stats["qdrant_checked"],
                valid=self.stats["qdrant_valid"],
                invalid=self.stats["qdrant_invalid"],
            )

        except Exception as e:
            logger.error("Failed to verify Qdrant", error=str(e))
            return False

        return len([v for v in self.violations if v["store"] == "Qdrant"]) == 0

    def generate_report(self) -> Dict:
        """Generate verification report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "passed": len(self.violations) == 0,
            "stats": self.stats,
            "violations": self.violations[:20],  # Limit to first 20
            "total_violations": len(self.violations),
            "summary": {
                "neo4j_compliance": (
                    f"{(self.stats['neo4j_valid'] / max(self.stats['neo4j_checked'], 1) * 100):.1f}%"
                ),
                "qdrant_compliance": (
                    f"{(self.stats['qdrant_valid'] / max(self.stats['qdrant_checked'], 1) * 100):.1f}%"
                ),
            },
        }

    def run(self) -> int:
        """Run verification and return exit code."""
        if not self.connect():
            return 2

        self.verify_neo4j()
        self.verify_qdrant()

        report = self.generate_report()

        # Print report
        print("\n" + "=" * 60)
        print("EMBEDDING FIELD VERIFICATION REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Status: {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
        print("\nStatistics:")
        print(
            f"  Neo4j:  {self.stats['neo4j_valid']}/{self.stats['neo4j_checked']} valid ({report['summary']['neo4j_compliance']})"
        )
        print(
            f"  Qdrant: {self.stats['qdrant_valid']}/{self.stats['qdrant_checked']} valid ({report['summary']['qdrant_compliance']})"
        )

        if self.violations:
            print(f"\nViolations found: {len(self.violations)}")
            print("\nFirst 5 violations:")
            for v in self.violations[:5]:
                print(f"  - {v['store']}/{v['type']}: {v.get('id', 'N/A')}")

        # Save report if requested
        report_file = os.getenv("REPORT_FILE")
        if report_file:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nFull report saved to: {report_file}")

        # Cleanup
        if self.neo4j_driver:
            self.neo4j_driver.close()

        return 0 if report["passed"] else 1


if __name__ == "__main__":
    verifier = EmbeddingFieldVerifier()
    sys.exit(verifier.run())

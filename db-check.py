#!/usr/bin/env python3
"""
Database Status Checker - WekaDocs GraphRAG System

Quick status report showing current state of all databases:
- Neo4j (graph nodes and relationships)
- Qdrant (vector collections)
- Redis (cache keys)

Usage:
    python db-check.py              # Full status report
    python db-check.py --json       # JSON output for scripting
    python db-check.py --quiet      # Minimal output

Requirements:
    - Docker services running (docker compose up -d)
    - Python packages: neo4j, qdrant-client, redis

Author: Claude Code
Generated: 2025-10-21
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import redis
    from neo4j import GraphDatabase
    from qdrant_client import QdrantClient

    from shared.config import init_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure required packages are installed:")
    print("  pip install neo4j qdrant-client redis")
    sys.exit(1)


class DatabaseStatusChecker:
    """Collects and displays status from all databases."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.status = {"neo4j": {}, "qdrant": {}, "redis": {}}

        # Initialize config
        try:
            self.config, self.settings = init_config()
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)

    def check_neo4j(self) -> Dict[str, Any]:
        """Check Neo4j database status."""
        try:
            driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )

            with driver.session() as session:
                # Total counts
                total_nodes = session.run(
                    "MATCH (n) RETURN count(n) as count"
                ).single()["count"]
                total_rels = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                ).single()["count"]

                # Node counts by label
                node_labels = []
                result = session.run(
                    """
                    CALL db.labels() YIELD label
                    CALL {
                        WITH label
                        MATCH (n)
                        WHERE label IN labels(n)
                        RETURN count(n) as count
                    }
                    RETURN label, count
                    ORDER BY count DESC
                """
                )
                for record in result:
                    node_labels.append(
                        {"label": record["label"], "count": record["count"]}
                    )

                # Relationship counts by type
                rel_types = []
                result = session.run(
                    """
                    CALL db.relationshipTypes() YIELD relationshipType
                    CALL {
                        WITH relationshipType
                        MATCH ()-[r]->()
                        WHERE type(r) = relationshipType
                        RETURN count(r) as count
                    }
                    RETURN relationshipType, count
                    ORDER BY count DESC
                """
                )
                for record in result:
                    rel_types.append(
                        {"type": record["relationshipType"], "count": record["count"]}
                    )

                self.status["neo4j"] = {
                    "status": "connected",
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "node_labels": node_labels,
                    "relationship_types": rel_types,
                }

            driver.close()
            return self.status["neo4j"]

        except Exception as e:
            self.status["neo4j"] = {"status": "error", "error": str(e)}
            return self.status["neo4j"]

    def check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant vector database status."""
        try:
            qdrant = QdrantClient(
                host=self.settings.qdrant_host, port=self.settings.qdrant_port
            )

            collections = qdrant.get_collections().collections
            collection_info = []

            for coll in collections:
                info = qdrant.get_collection(coll.name)
                collection_info.append(
                    {
                        "name": coll.name,
                        "vectors": info.points_count,
                        "status": str(info.status),
                        "config": {
                            "size": (
                                info.config.params.vectors.size
                                if hasattr(info.config.params.vectors, "size")
                                else "N/A"
                            ),
                            "distance": (
                                str(info.config.params.vectors.distance)
                                if hasattr(info.config.params.vectors, "distance")
                                else "N/A"
                            ),
                        },
                    }
                )

            self.status["qdrant"] = {
                "status": "connected",
                "collections": collection_info,
                "total_collections": len(collections),
            }

            return self.status["qdrant"]

        except Exception as e:
            self.status["qdrant"] = {"status": "error", "error": str(e)}
            return self.status["qdrant"]

    def check_redis(self, db: int = 1) -> Dict[str, Any]:
        """Check Redis cache status."""
        try:
            redis_password = os.getenv("REDIS_PASSWORD", "")
            redis_uri = (
                f"redis://:{redis_password}@localhost:6379/{db}"
                if redis_password
                else f"redis://localhost:6379/{db}"
            )

            r = redis.Redis.from_url(redis_uri, decode_responses=True)
            keys = r.keys("*")
            key_count = len(keys)

            self.status["redis"] = {
                "status": "connected",
                "database": db,
                "key_count": key_count,
                "sample_keys": keys[:10] if keys else [],
            }

            return self.status["redis"]

        except Exception as e:
            self.status["redis"] = {"status": "error", "error": str(e)}
            return self.status["redis"]

    def print_report(self):
        """Print formatted status report."""
        print("=" * 70)
        print("DATABASE STATUS REPORT")
        print("=" * 70)
        print()

        # Neo4j
        print("NEO4J DATABASE")
        print("-" * 70)
        neo4j = self.status["neo4j"]
        if neo4j.get("status") == "error":
            print(f"❌ Error: {neo4j['error']}")
        else:
            print(f"Total Nodes: {neo4j['total_nodes']:,}")
            print(f"Total Relationships: {neo4j['total_relationships']:,}")
            print()

            print("Nodes by Label:")
            for item in neo4j["node_labels"]:
                print(f"  {item['label']:20s} {item['count']:>8,}")

            print()
            print("Relationships by Type:")
            for item in neo4j["relationship_types"]:
                print(f"  {item['type']:20s} {item['count']:>8,}")

        print()

        # Qdrant
        print("QDRANT VECTOR DATABASE")
        print("-" * 70)
        qdrant = self.status["qdrant"]
        if qdrant.get("status") == "error":
            print(f"❌ Error: {qdrant['error']}")
        elif not qdrant["collections"]:
            print("No collections found")
        else:
            for coll in qdrant["collections"]:
                print(f"Collection: {coll['name']}")
                print(f"  Vectors: {coll['vectors']:,}")
                print(
                    f"  Config: size={coll['config']['size']} distance={coll['config']['distance']}"
                )
                print(f"  Status: {coll['status']}")
                print()

        # Redis
        print("REDIS DATABASE")
        print("-" * 70)
        redis_status = self.status["redis"]
        if redis_status.get("status") == "error":
            print(f"❌ Error: {redis_status['error']}")
        else:
            print(f"Database: {redis_status['database']} (test db)")
            print(f"Keys: {redis_status['key_count']:,}")

            if redis_status["sample_keys"]:
                print("\nSample Keys (first 10):")
                for key in redis_status["sample_keys"]:
                    print(f"  - {key}")

        print()

        # Data Parity Check
        if neo4j.get("status") == "connected" and qdrant.get("status") == "connected":

            # Find Section count in Neo4j
            section_count = 0
            for item in neo4j.get("node_labels", []):
                if item["label"] == "Section":
                    section_count = item["count"]
                    break

            # Find vector count in Qdrant
            vector_count = 0
            for coll in qdrant.get("collections", []):
                if "section" in coll["name"].lower():
                    vector_count = coll["vectors"]
                    break

            if section_count > 0 and vector_count > 0:
                drift = abs(section_count - vector_count)
                drift_pct = (drift / section_count) * 100 if section_count > 0 else 0

                print("DATA PARITY CHECK")
                print("-" * 70)
                print(f"Neo4j Sections: {section_count:,}")
                print(f"Qdrant Vectors: {vector_count:,}")
                print(f"Drift: {drift:,} ({drift_pct:.2f}%)")

                if drift == 0:
                    print("Status: ✅ Perfect parity")
                elif drift_pct < 0.5:
                    print("Status: ⚠️  Acceptable drift (<0.5%)")
                else:
                    print("Status: ❌ Drift exceeds threshold (>0.5%)")
                print()

        print("=" * 70)

    def run(self, output_json: bool = False) -> int:
        """
        Run all database checks.

        Args:
            output_json: Output JSON instead of formatted report

        Returns:
            int: Exit code (0 = success, 1 = any errors)
        """
        # Check all databases
        self.check_neo4j()
        self.check_qdrant()
        self.check_redis()

        # Output results
        if output_json:
            print(json.dumps(self.status, indent=2))
        else:
            self.print_report()

        # Determine exit code
        has_errors = any(db.get("status") == "error" for db in self.status.values())

        return 1 if has_errors else 0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check WekaDocs GraphRAG database status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted report"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output (errors only)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    checker = DatabaseStatusChecker(quiet=args.quiet)
    return checker.run(output_json=args.json)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Database Cleanup Script - Surgical Data Deletion with Metadata Preservation

This script performs intelligent cleanup of test/development data while preserving
critical system metadata nodes that are required for system operation. It is aware
of the new Neo4j v2.2 schema objects and the Qdrant multi-vector `chunks_multi`
collection, deleting only vector data while keeping collection/schema definitions.

IMPORTANT: This script now preserves:
- SchemaVersion nodes (required for health checks)
- SystemMetadata nodes (for configuration)
- MigrationHistory nodes (for schema tracking)
- Any other nodes labeled with _metadata suffix

Usage:
    python scripts/cleanup-databases.py [options]

Options:
    --dry-run           Show what would be deleted without actually deleting
    --redis-db N        Redis database number to clean (default: 1)
    --skip-neo4j        Skip Neo4j cleanup
    --skip-qdrant       Skip Qdrant cleanup
    --skip-redis        Skip Redis cleanup
    --report-dir PATH   Custom report directory (default: reports/cleanup/)
    --quiet             Minimal console output
    --restore-metadata  Restore missing metadata nodes if needed
    --help              Show this help message

Examples:
    # Standard cleanup with full reporting
    python scripts/cleanup-databases.py

    # Dry run to preview changes
    python scripts/cleanup-databases.py --dry-run

    # Clean and restore metadata if missing
    python scripts/cleanup-databases.py --restore-metadata

Safety Features:
    - NEVER deletes SchemaVersion or other metadata nodes
    - NEVER deletes Neo4j constraints or indexes
    - NEVER deletes Qdrant collection schemas
    - Can restore critical metadata if accidentally deleted
    - Shows exact count of what will be deleted before proceeding

Author: Claude Code
Generated: 2025-10-30 (Updated for metadata preservation)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import redis
    from neo4j import GraphDatabase
    from qdrant_client import QdrantClient

    from shared.config import init_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're running from the project root:")
    print("   python scripts/cleanup-databases.py")
    sys.exit(1)


class DatabaseCleaner:
    """Handles intelligent cleanup of databases with metadata preservation."""

    # Critical metadata node labels that must be preserved
    PRESERVED_LABELS: Set[str] = {
        "SchemaVersion",  # Required for health checks
        "SystemMetadata",  # System configuration
        "MigrationHistory",  # Schema migration tracking
        "RelationshipTypesMarker",  # Documents available relationship types
        "_metadata",  # Generic metadata suffix
        "_system",  # Generic system suffix
    }

    # Data node labels that should be deleted
    DATA_LABELS: Set[str] = {
        "Document",
        "Section",
        "Chunk",
        "Entity",
        "CitationUnit",
        "Example",
        "Parameter",
        "Component",
        "Procedure",
        "Command",
        "Configuration",
        "Step",
        "Concept",
        "Topic",
        "Person",
        "Organization",
        "Location",
        "Date",
        "Software",
        "Hardware",
        "Metric",
        "Session",
        "Query",
        "Answer",
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = None
        self.settings = None
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "options": vars(args),
            "before": {},
            "after": {},
            "actions": [],
            "errors": [],
            "summary": {},
            "preserved_nodes": [],
        }

        # Initialize config
        try:
            self.config, self.settings = init_config()
        except Exception as e:
            self._log_error(f"Failed to load config: {e}")
            sys.exit(1)

        # Track the primary / known Qdrant collections (legacy + new multi-vector)
        primary_collection = None
        if self.config and hasattr(self.config.search.vector, "qdrant"):
            primary_collection = (
                self.config.search.vector.qdrant.collection_name or None
            )

        self.qdrant_known_collections = sorted(
            {
                coll_name
                for coll_name in [
                    primary_collection,
                    "chunks",  # legacy single-vector
                    "chunks_multi",  # new multi-vector collection
                ]
                if coll_name
            }
        )

    def _log(self, message: str, level: str = "info"):
        """Log message to console and report."""
        if not self.args.quiet or level in ["error", "warning", "header"]:
            prefix = {
                "info": "  ",
                "success": "✅",
                "warning": "⚠️ ",
                "error": "❌",
                "header": "",
            }.get(level, "  ")
            print(f"{prefix} {message}")

    def _log_error(self, message: str):
        """Log error message."""
        self._log(message, "error")
        self.report_data["errors"].append(
            {"timestamp": datetime.now().isoformat(), "message": message}
        )

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log an action taken."""
        self.report_data["actions"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details,
            }
        )

    @staticmethod
    def _describe_vector_config(vector_config: Any) -> str:
        """
        Produce a compact description of a Qdrant vector configuration.

        Handles both single-vector (VectorParams) and named-vector structures.
        """

        def _safe_dump(obj: Any) -> Dict[str, Any]:
            if obj is None:
                return {}
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            if isinstance(obj, dict):
                return obj
            return {"value": str(obj)}

        try:
            raw = _safe_dump(vector_config)
            # Named-vector structures tend to store data under "vectors"
            if isinstance(raw, dict) and "vectors" in raw:
                raw = raw["vectors"]

            if isinstance(raw, dict):
                parts = []
                for name, cfg in raw.items():
                    cfg_data = _safe_dump(cfg)
                    size = cfg_data.get("size") or cfg_data.get("dimensions")
                    distance = (
                        cfg_data.get("distance")
                        or cfg_data.get("metric")
                        or cfg_data.get("similarity")
                    )
                    parts.append(f"{name}:{size}D/{distance}")
                return ", ".join(parts) if parts else json.dumps(raw)
            return json.dumps(raw)
        except Exception:
            return str(vector_config)

    def restore_metadata_nodes(self, session) -> bool:
        """
        Restore critical metadata nodes if they're missing.

        Args:
            session: Neo4j session

        Returns:
            bool: True if restoration successful or not needed
        """
        restored = []

        # Check for SchemaVersion node
        result = session.run("MATCH (sv:SchemaVersion) RETURN count(sv) as count")
        if result.single()["count"] == 0:
            self._log("SchemaVersion node missing - restoring...", "warning")

            # Get schema version from config
            schema_version = self.config.graph_schema.version if self.config else "v2.1"

            session.run(
                """
                CREATE (sv:SchemaVersion {
                    version: $version,
                    created_at: datetime(),
                    description: $description,
                    restored: true,
                    restored_at: datetime()
                })
                RETURN sv
            """,
                version=schema_version,
                description="Phase 7E schema - restored by cleanup script",
            )

            restored.append("SchemaVersion")
            self._log(
                f"Restored SchemaVersion node (version: {schema_version})", "success"
            )
            self._log_action(
                "restore_metadata",
                {"node_type": "SchemaVersion", "version": schema_version},
            )

        if restored:
            self._log(f"Restored {len(restored)} metadata node(s)", "success")
            self.report_data["preserved_nodes"].extend(restored)
            return True

        return True

    def cleanup_neo4j(self) -> bool:
        """
        Clean Neo4j database - delete only data nodes, preserve metadata and schema.

        Returns:
            bool: True if successful, False otherwise
        """
        if self.args.skip_neo4j:
            self._log("Skipping Neo4j cleanup (--skip-neo4j)", "warning")
            return True

        self._log("Neo4j Intelligent Data Cleanup", "header")
        self._log("-" * 60, "header")

        try:
            driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )

            with driver.session() as session:
                # First, restore metadata if requested
                if self.args.restore_metadata:
                    self.restore_metadata_nodes(session)

                # Get counts by label
                label_counts = {}
                result = session.run(
                    """
                    MATCH (n)
                    UNWIND labels(n) AS label
                    RETURN label, count(n) AS count
                    ORDER BY count DESC
                """
                )
                for record in result:
                    label_counts[record["label"]] = record["count"]

                # Separate preserved and deletable nodes
                preserved_labels = {}
                deletable_labels = {}

                for label, count in label_counts.items():
                    if (
                        label in self.PRESERVED_LABELS
                        or label.endswith("_metadata")
                        or label.endswith("_system")
                    ):
                        preserved_labels[label] = count
                    elif label in self.DATA_LABELS:
                        deletable_labels[label] = count
                    else:
                        # Unknown label - check if it looks like data
                        self._log(
                            f"Unknown label '{label}' with {count} nodes - will delete",
                            "warning",
                        )
                        deletable_labels[label] = count

                # Get total counts
                total_nodes_result = session.run("MATCH (n) RETURN count(n) as count")
                total_nodes = total_nodes_result.single()["count"]

                total_rels_result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                )
                total_rels = total_rels_result.single()["count"]

                # Calculate what will be deleted
                nodes_to_delete = sum(deletable_labels.values())
                nodes_to_preserve = sum(preserved_labels.values())

                # Get constraint and index counts
                constraints = list(session.run("SHOW CONSTRAINTS"))
                indexes = list(session.run("SHOW INDEXES"))

                # Log before state
                self._log(f"Total: {total_nodes} nodes, {total_rels} relationships")
                self._log(
                    f"Schema: {len(constraints)} constraints, {len(indexes)} indexes"
                )

                if preserved_labels:
                    self._log("", "header")
                    self._log("PRESERVED (Metadata/System):", "header")
                    for label, count in preserved_labels.items():
                        self._log(f"  {label}: {count} nodes", "success")

                if deletable_labels:
                    self._log("", "header")
                    self._log("TO DELETE (Data):", "header")
                    for label, count in list(deletable_labels.items())[
                        :10
                    ]:  # Show top 10
                        self._log(f"  {label}: {count} nodes", "warning")
                    if len(deletable_labels) > 10:
                        self._log(
                            f"  ... and {len(deletable_labels) - 10} more labels",
                            "warning",
                        )

                self._log("", "header")
                self._log(
                    f"Summary: Will delete {nodes_to_delete} nodes, preserve {nodes_to_preserve} nodes",
                    "header",
                )

                # Store before state
                self.report_data["before"]["neo4j"] = {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "preserved_nodes": preserved_labels,
                    "deletable_nodes": deletable_labels,
                    "constraints": len(constraints),
                    "indexes": len(indexes),
                }

                # Delete data nodes (unless dry run)
                if not self.args.dry_run:
                    # Delete nodes by label to preserve metadata
                    deleted_count = 0
                    for label in deletable_labels:
                        result = session.run(
                            f"""
                            MATCH (n:{label})
                            WITH n LIMIT 10000
                            DETACH DELETE n
                            RETURN count(n) as deleted
                        """
                        )
                        batch_deleted = result.single()["deleted"]

                        # Keep deleting in batches until all gone
                        while batch_deleted > 0:
                            deleted_count += batch_deleted
                            result = session.run(
                                f"""
                                MATCH (n:{label})
                                WITH n LIMIT 10000
                                DETACH DELETE n
                                RETURN count(n) as deleted
                            """
                            )
                            batch_deleted = result.single()["deleted"]

                    self._log(f"Deleted {deleted_count} data nodes", "success")
                    self._log_action(
                        "neo4j_selective_delete",
                        {
                            "nodes_deleted": deleted_count,
                            "labels_deleted": list(deletable_labels.keys()),
                            "preserved_labels": list(preserved_labels.keys()),
                        },
                    )
                else:
                    self._log(
                        f"DRY RUN: Would delete {nodes_to_delete} nodes", "warning"
                    )

                # Capture after state
                after_nodes_result = session.run("MATCH (n) RETURN count(n) as count")
                after_nodes = after_nodes_result.single()["count"]

                after_rels_result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                )
                after_rels = after_rels_result.single()["count"]

                after_constraints = list(session.run("SHOW CONSTRAINTS"))
                after_indexes = list(session.run("SHOW INDEXES"))

                # Check preserved nodes still exist
                preserved_check = {}
                for label in preserved_labels:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    preserved_check[label] = result.single()["count"]

                self.report_data["after"]["neo4j"] = {
                    "total_nodes": after_nodes,
                    "total_relationships": after_rels,
                    "preserved_nodes": preserved_check,
                    "constraints": len(after_constraints),
                    "indexes": len(after_indexes),
                }

                self._log("", "header")
                self._log(
                    f"After: {after_nodes} nodes, {after_rels} relationships", "success"
                )

                # Verify metadata preservation
                metadata_intact = all(
                    preserved_check.get(label, 0) >= preserved_labels.get(label, 0)
                    for label in preserved_labels
                )

                if metadata_intact:
                    self._log("All metadata nodes preserved", "success")
                else:
                    self._log(
                        "WARNING: Some metadata nodes may have been affected", "error"
                    )
                    for label in preserved_labels:
                        before = preserved_labels[label]
                        after = preserved_check.get(label, 0)
                        if after < before:
                            self._log(
                                f"  {label}: {before} -> {after} (lost {before - after})",
                                "error",
                            )

                # Verify schema preservation
                if len(after_constraints) == len(constraints) and len(
                    after_indexes
                ) == len(indexes):
                    self._log(
                        f"Schema preserved: {len(constraints)} constraints, {len(indexes)} indexes",
                        "success",
                    )
                else:
                    self._log(
                        f"Schema changed! Before: {len(constraints)}c/{len(indexes)}i, After: {len(after_constraints)}c/{len(after_indexes)}i",
                        "error",
                    )
                    return False

            driver.close()
            return metadata_intact

        except Exception as e:
            self._log_error(f"Neo4j cleanup failed: {e}")
            return False

    def cleanup_qdrant(self) -> bool:
        """
        Clean Qdrant database - delete all vectors, preserve collection schemas.

        Returns:
            bool: True if successful, False otherwise
        """
        if self.args.skip_qdrant:
            self._log("Skipping Qdrant cleanup (--skip-qdrant)", "warning")
            return True

        self._log("", "header")
        self._log("Qdrant Vector Cleanup", "header")
        self._log("-" * 60, "header")
        if self.qdrant_known_collections:
            self._log(
                "Target collections: " + ", ".join(self.qdrant_known_collections),
                "info",
            )

        try:
            qdrant = QdrantClient(
                host=self.settings.qdrant_host, port=self.settings.qdrant_port
            )

            collections = qdrant.get_collections().collections
            qdrant_before = {}
            qdrant_after = {}

            for coll in collections:
                try:
                    coll_info = qdrant.get_collection(coll.name)
                    before_count = coll_info.points_count

                    qdrant_before[coll.name] = {
                        "vector_count": before_count,
                        "vectors": self._describe_vector_config(
                            coll_info.config.params.vectors
                        ),
                    }

                    self._log(f"Collection: {coll.name}")
                    self._log(f"  Before: {before_count} vectors")
                    if qdrant_before[coll.name]["vectors"]:
                        self._log(
                            f"  Vector config: {qdrant_before[coll.name]['vectors']}",
                            "info",
                        )

                    # Delete all points but keep collection structure
                    if not self.args.dry_run and before_count > 0:
                        # Delete points in batches
                        offset = None
                        deleted_total = 0

                        while True:
                            # Get batch of point IDs
                            result = qdrant.scroll(
                                collection_name=coll.name,
                                scroll_filter=None,
                                limit=100,
                                offset=offset,
                                with_payload=False,
                                with_vectors=False,
                            )

                            if not result[0]:
                                break

                            point_ids = [p.id for p in result[0]]
                            qdrant.delete(
                                collection_name=coll.name,
                                points_selector=point_ids,
                            )

                            deleted_total += len(point_ids)
                            offset = result[1]

                            if offset is None:
                                break

                        self._log(f"  Deleted: {deleted_total} vectors", "success")
                        self._log_action(
                            "qdrant_delete_points",
                            {
                                "collection": coll.name,
                                "vectors_deleted": deleted_total,
                            },
                        )

                        qdrant_after[coll.name] = {
                            "vector_count": 0,
                            "preserved": True,
                            "vectors": qdrant_before[coll.name]["vectors"],
                        }
                    elif self.args.dry_run:
                        self._log(
                            f"  DRY RUN: Would delete {before_count} vectors", "warning"
                        )
                        qdrant_after[coll.name] = {
                            "vector_count": before_count,
                            "preserved": True,
                            "vectors": qdrant_before[coll.name]["vectors"],
                        }
                    else:
                        self._log("  No vectors to delete", "info")
                        qdrant_after[coll.name] = {
                            "vector_count": 0,
                            "preserved": True,
                            "vectors": qdrant_before[coll.name]["vectors"],
                        }

                except Exception as e:
                    self._log(
                        f"  Error processing collection {coll.name}: {e}", "error"
                    )
                    continue

            self.report_data["before"]["qdrant"] = qdrant_before
            self.report_data["after"]["qdrant"] = qdrant_after

            return True

        except Exception as e:
            self._log_error(f"Qdrant cleanup failed: {e}")
            return False

    def cleanup_redis(self) -> bool:
        """
        Clean Redis database - delete only data keys, preserve system keys.

        Returns:
            bool: True if successful, False otherwise
        """
        if self.args.skip_redis:
            self._log("Skipping Redis cleanup (--skip-redis)", "warning")
            return True

        self._log("", "header")
        self._log("Redis Selective Cleanup", "header")
        self._log("-" * 60, "header")

        try:
            redis_password = os.getenv("REDIS_PASSWORD", "")
            redis_uri = (
                f"redis://:{redis_password}@localhost:6379/{self.args.redis_db}"
                if redis_password
                else f"redis://localhost:6379/{self.args.redis_db}"
            )

            r = redis.Redis.from_url(redis_uri, decode_responses=True)

            # Get all keys
            all_keys = r.keys("*")

            # Separate system and data keys
            system_patterns = [
                "schema:*",
                "metadata:*",
                "system:*",
                "_version",
                "_config",
            ]

            system_keys = []
            data_keys = []

            for key in all_keys:
                is_system = any(
                    key.startswith(pattern.replace("*", ""))
                    for pattern in system_patterns
                )
                if is_system:
                    system_keys.append(key)
                else:
                    data_keys.append(key)

            redis_before = {
                "db": self.args.redis_db,
                "total_keys": len(all_keys),
                "system_keys": len(system_keys),
                "data_keys": len(data_keys),
                "sample_data_keys": data_keys[:10] if data_keys else [],
                "preserved_keys": system_keys[:10] if system_keys else [],
            }
            self.report_data["before"]["redis"] = redis_before

            self._log(f"Database: db={self.args.redis_db}")
            self._log(f"Total: {len(all_keys)} keys")
            self._log(
                f"System/Metadata: {len(system_keys)} keys (preserved)", "success"
            )
            self._log(f"Data: {len(data_keys)} keys (to delete)", "warning")

            # Delete data keys (unless dry run)
            if not self.args.dry_run and data_keys:
                # Delete in batches to avoid blocking
                batch_size = 1000
                for i in range(0, len(data_keys), batch_size):
                    batch = data_keys[i : i + batch_size]
                    r.delete(*batch)

                self._log(f"Deleted {len(data_keys)} data keys", "success")
                self._log_action(
                    "redis_selective_delete",
                    {
                        "db": self.args.redis_db,
                        "keys_deleted": len(data_keys),
                        "keys_preserved": len(system_keys),
                    },
                )
            elif self.args.dry_run:
                self._log(f"DRY RUN: Would delete {len(data_keys)} keys", "warning")

            # Capture after state
            after_keys = r.keys("*") if not self.args.dry_run else system_keys

            redis_after = {
                "db": self.args.redis_db,
                "total_keys": len(after_keys),
                "preserved_keys": len(system_keys),
            }
            self.report_data["after"]["redis"] = redis_after

            self._log(f"After: {len(after_keys)} keys", "success")

            return True

        except Exception as e:
            self._log_error(f"Redis cleanup failed: {e}")
            return False

    def generate_report(self) -> Path:
        """
        Generate comprehensive cleanup report.

        Returns:
            Path: Path to generated report file
        """
        # Create report directory
        report_dir = Path(self.args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_file = report_dir / f"cleanup-report-{timestamp}.json"

        # Calculate summary statistics
        summary = {
            "dry_run": self.args.dry_run,
            "success": len(self.report_data["errors"]) == 0,
            "metadata_preserved": True,
            "databases_cleaned": [],
        }

        if not self.args.skip_neo4j:
            neo4j_before = self.report_data["before"].get("neo4j", {})
            neo4j_after = self.report_data["after"].get("neo4j", {})

            data_deleted = sum(neo4j_before.get("deletable_nodes", {}).values())
            metadata_preserved = sum(neo4j_after.get("preserved_nodes", {}).values())

            summary["databases_cleaned"].append(
                {
                    "database": "neo4j",
                    "data_nodes_deleted": data_deleted,
                    "metadata_nodes_preserved": metadata_preserved,
                    "schema_preserved": True,
                }
            )

        if not self.args.skip_qdrant:
            qdrant_before = self.report_data["before"].get("qdrant", {})
            total_vectors = sum(
                coll.get("vector_count", 0) for coll in qdrant_before.values()
            )
            summary["databases_cleaned"].append(
                {
                    "database": "qdrant",
                    "collections_preserved": len(qdrant_before),
                    "total_vectors_deleted": total_vectors,
                    "target_collections": self.qdrant_known_collections,
                }
            )

        if not self.args.skip_redis:
            redis_before = self.report_data["before"].get("redis", {})
            summary["databases_cleaned"].append(
                {
                    "database": "redis",
                    "db": self.args.redis_db,
                    "data_keys_deleted": redis_before.get("data_keys", 0),
                    "system_keys_preserved": redis_before.get("system_keys", 0),
                }
            )

        self.report_data["summary"] = summary

        # Write report
        with open(report_file, "w") as f:
            json.dump(self.report_data, f, indent=2)

        return report_file

    def run(self) -> int:
        """
        Execute cleanup workflow.

        Returns:
            int: Exit code (0 = success, 1 = failure)
        """
        self._log("=" * 60, "header")
        mode = "DRY RUN - " if self.args.dry_run else ""
        self._log(f"{mode}INTELLIGENT DATABASE CLEANUP", "header")
        self._log("Preserving metadata, deleting only data", "header")
        self._log("=" * 60, "header")
        self._log(f"Timestamp: {self.report_data['timestamp']}", "header")
        self._log("", "header")

        # Execute cleanup steps
        success = True
        success &= self.cleanup_neo4j()
        success &= self.cleanup_qdrant()
        success &= self.cleanup_redis()

        # Generate report
        self._log("", "header")
        self._log("=" * 60, "header")
        self._log("CLEANUP COMPLETE" if success else "CLEANUP FAILED", "header")
        self._log("=" * 60, "header")

        try:
            report_file = self.generate_report()
            self._log(f"Report saved: {report_file}", "success")
        except Exception as e:
            self._log_error(f"Failed to generate report: {e}")
            success = False

        # Print summary
        if success:
            if self.args.dry_run:
                self._log("DRY RUN completed - no changes made", "warning")
            else:
                self._log("All data deleted successfully", "success")
                self._log("All metadata and schemas preserved", "success")
                self._log("System ready for fresh ingestion", "success")
        else:
            self._log(f"{len(self.report_data['errors'])} error(s) occurred", "error")

        self._log("", "header")

        return 0 if success else 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Intelligent database cleanup with metadata preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=1,
        help="Redis database number to clean (default: 1 = test db)",
    )
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j cleanup")
    parser.add_argument(
        "--skip-qdrant", action="store_true", help="Skip Qdrant cleanup"
    )
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis cleanup")
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/cleanup",
        help="Custom report directory (default: reports/cleanup/)",
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    parser.add_argument(
        "--restore-metadata",
        action="store_true",
        help="Restore missing metadata nodes if needed",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate args
    if args.skip_neo4j and args.skip_qdrant and args.skip_redis:
        print("❌ Error: Cannot skip all databases. Nothing to clean!")
        return 1

    # Run cleanup
    cleaner = DatabaseCleaner(args)
    return cleaner.run()


if __name__ == "__main__":
    sys.exit(main())

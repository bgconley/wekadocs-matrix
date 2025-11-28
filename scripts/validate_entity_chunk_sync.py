#!/usr/bin/env python3
"""
Entity-Chunk Sync Validator and Repair Tool.

This script validates and optionally repairs the synchronization between:
- Entity→Chunk relationships in Neo4j
- Chunk vectors in Qdrant

The root cause of graph reranker failure is often Entity nodes pointing to
Chunk IDs that don't exist in the Qdrant vector store.

Usage:
    # Dry-run validation (no changes)
    python scripts/validate_entity_chunk_sync.py --dry-run

    # Full validation with report
    python scripts/validate_entity_chunk_sync.py --report reports/sync_validation.json

    # Repair mode (delete orphan relationships)
    python scripts/validate_entity_chunk_sync.py --repair

    # Repair mode with backup
    python scripts/validate_entity_chunk_sync.py --repair --backup
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SyncValidationResult:
    """Results from entity-chunk sync validation."""

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Counts
    total_entities: int = 0
    connected_entities: int = 0
    orphan_entities: int = 0

    total_chunks_neo4j: int = 0
    total_chunks_qdrant: int = 0

    total_entity_chunk_rels: int = 0
    valid_rels: int = 0
    orphan_rels: int = 0  # Rels pointing to non-existent Qdrant chunks

    # Detailed orphan info
    orphan_chunk_ids: List[str] = field(default_factory=list)
    orphan_entity_ids: List[str] = field(default_factory=list)
    sample_orphan_rels: List[Dict] = field(default_factory=list)

    # Repair info
    repaired: bool = False
    rels_deleted: int = 0
    backup_file: Optional[str] = None

    # Sync health
    sync_health_percent: float = 0.0  # valid_rels / total_entity_chunk_rels * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "counts": {
                "total_entities": self.total_entities,
                "connected_entities": self.connected_entities,
                "orphan_entities": self.orphan_entities,
                "total_chunks_neo4j": self.total_chunks_neo4j,
                "total_chunks_qdrant": self.total_chunks_qdrant,
                "total_entity_chunk_rels": self.total_entity_chunk_rels,
                "valid_rels": self.valid_rels,
                "orphan_rels": self.orphan_rels,
            },
            "orphan_chunk_ids_sample": self.orphan_chunk_ids[:10],
            "orphan_entity_ids_sample": self.orphan_entity_ids[:10],
            "sample_orphan_rels": self.sample_orphan_rels[:5],
            "repair": {
                "repaired": self.repaired,
                "rels_deleted": self.rels_deleted,
                "backup_file": self.backup_file,
            },
            "sync_health_percent": round(self.sync_health_percent, 2),
        }


class EntityChunkSyncValidator:
    """Validates and repairs Entity→Chunk sync between Neo4j and Qdrant."""

    def __init__(self, neo4j_driver, qdrant_client, config):
        self.driver = neo4j_driver
        self.qdrant = qdrant_client
        self.config = config
        self.collection = config.search.vector.qdrant.collection_name

    def validate(self) -> SyncValidationResult:
        """Run full sync validation."""
        result = SyncValidationResult()

        logger.info("Starting entity-chunk sync validation")

        # Step 1: Count entities
        with self.driver.session() as session:
            # Total entities
            r = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            result.total_entities = r.single()["cnt"]

            # Connected vs orphan entities
            r = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) AS rel_count
                RETURN
                    sum(CASE WHEN rel_count > 0 THEN 1 ELSE 0 END) AS connected,
                    sum(CASE WHEN rel_count = 0 THEN 1 ELSE 0 END) AS orphan
            """
            )
            rec = r.single()
            result.connected_entities = rec["connected"]
            result.orphan_entities = rec["orphan"]

            # Total chunks in Neo4j
            r = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt")
            result.total_chunks_neo4j = r.single()["cnt"]

        # Step 2: Count chunks in Qdrant
        try:
            info = self.qdrant.get_collection(self.collection)
            result.total_chunks_qdrant = info.points_count
        except Exception as e:
            logger.warning("Failed to get Qdrant collection info", error=str(e))
            result.total_chunks_qdrant = 0

        # Step 3: Get all Entity→Chunk relationship chunk IDs
        entity_chunk_ids: Set[str] = set()
        with self.driver.session() as session:
            r = session.run(
                """
                MATCH (e:Entity)-[r:MENTIONED_IN|DEFINES]->(c:Chunk)
                RETURN DISTINCT c.id AS chunk_id
            """
            )
            for rec in r:
                if rec["chunk_id"]:
                    entity_chunk_ids.add(rec["chunk_id"])

            # Total entity-chunk relationships
            r = session.run(
                """
                MATCH (e:Entity)-[r:MENTIONED_IN|DEFINES]->(c:Chunk)
                RETURN count(r) AS cnt
            """
            )
            result.total_entity_chunk_rels = r.single()["cnt"]

        logger.info(
            "Found entity-chunk relationships",
            total_rels=result.total_entity_chunk_rels,
            unique_chunks=len(entity_chunk_ids),
        )

        # Step 4: Check which chunk IDs exist in Qdrant
        qdrant_chunk_ids = self._get_qdrant_chunk_ids()

        # Step 5: Find orphan chunk IDs (in Neo4j relationships but not in Qdrant)
        orphan_chunks = entity_chunk_ids - qdrant_chunk_ids
        result.orphan_chunk_ids = list(orphan_chunks)
        result.orphan_rels = len(orphan_chunks)
        result.valid_rels = result.total_entity_chunk_rels - result.orphan_rels

        # Step 6: Get sample orphan relationships for debugging
        if orphan_chunks:
            sample_ids = list(orphan_chunks)[:10]
            with self.driver.session() as session:
                r = session.run(
                    """
                    MATCH (e:Entity)-[r:MENTIONED_IN|DEFINES]->(c:Chunk)
                    WHERE c.id IN $chunk_ids
                    RETURN e.name AS entity_name, e.id AS entity_id,
                           type(r) AS rel_type, c.id AS chunk_id
                    LIMIT 20
                """,
                    chunk_ids=sample_ids,
                )

                for rec in r:
                    result.sample_orphan_rels.append(
                        {
                            "entity_name": rec["entity_name"],
                            "entity_id": rec["entity_id"],
                            "rel_type": rec["rel_type"],
                            "chunk_id": rec["chunk_id"],
                        }
                    )
                    if rec["entity_id"] not in result.orphan_entity_ids:
                        result.orphan_entity_ids.append(rec["entity_id"])

        # Calculate sync health
        if result.total_entity_chunk_rels > 0:
            result.sync_health_percent = (
                result.valid_rels / result.total_entity_chunk_rels * 100
            )
        else:
            result.sync_health_percent = 100.0

        logger.info(
            "Validation complete",
            sync_health=f"{result.sync_health_percent:.1f}%",
            orphan_rels=result.orphan_rels,
            valid_rels=result.valid_rels,
        )

        return result

    def repair(self, backup: bool = True) -> SyncValidationResult:
        """
        Repair sync by removing orphan Entity→Chunk relationships.

        Args:
            backup: If True, backup orphan relationships before deletion
        """
        # First validate to find orphans
        result = self.validate()

        if not result.orphan_chunk_ids:
            logger.info("No orphan relationships to repair")
            return result

        # Backup if requested
        if backup:
            backup_data = self._backup_orphan_rels(result.orphan_chunk_ids)
            backup_file = f"entity_chunk_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = Path("reports") / backup_file
            backup_path.parent.mkdir(exist_ok=True)

            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2)

            result.backup_file = str(backup_path)
            logger.info("Backup created", file=result.backup_file)

        # Delete orphan relationships
        with self.driver.session() as session:
            r = session.run(
                """
                MATCH (e:Entity)-[r:MENTIONED_IN|DEFINES]->(c:Chunk)
                WHERE c.id IN $chunk_ids
                DELETE r
                RETURN count(r) AS deleted
            """,
                chunk_ids=result.orphan_chunk_ids,
            )
            result.rels_deleted = r.single()["deleted"]

        result.repaired = True

        logger.info(
            "Repair complete",
            rels_deleted=result.rels_deleted,
            backup_file=result.backup_file,
        )

        return result

    def _get_qdrant_chunk_ids(self) -> Set[str]:
        """Get all chunk IDs from Qdrant."""
        chunk_ids: Set[str] = set()

        try:
            # Scroll through all points to get their IDs
            offset = None
            batch_size = 1000

            while True:
                result, next_offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=["id", "node_id"],
                )

                for point in result:
                    # Try both 'id' and 'node_id' payload fields
                    chunk_id = (
                        (point.payload.get("id") or point.payload.get("node_id"))
                        if point.payload
                        else None
                    )

                    if chunk_id:
                        chunk_ids.add(chunk_id)

                if next_offset is None or len(result) < batch_size:
                    break

                offset = next_offset

            logger.info("Retrieved Qdrant chunk IDs", count=len(chunk_ids))

        except Exception as e:
            logger.error("Failed to get Qdrant chunk IDs", error=str(e))

        return chunk_ids

    def _backup_orphan_rels(self, orphan_chunk_ids: List[str]) -> Dict:
        """Backup orphan relationships before deletion."""
        backup = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "orphan_chunk_ids": orphan_chunk_ids,
            "relationships": [],
        }

        with self.driver.session() as session:
            r = session.run(
                """
                MATCH (e:Entity)-[r:MENTIONED_IN|DEFINES]->(c:Chunk)
                WHERE c.id IN $chunk_ids
                RETURN
                    e.id AS entity_id,
                    e.name AS entity_name,
                    type(r) AS rel_type,
                    properties(r) AS rel_props,
                    c.id AS chunk_id,
                    c.document_id AS document_id
            """,
                chunk_ids=orphan_chunk_ids,
            )

            for rec in r:
                backup["relationships"].append(
                    {
                        "entity_id": rec["entity_id"],
                        "entity_name": rec["entity_name"],
                        "rel_type": rec["rel_type"],
                        "rel_props": dict(rec["rel_props"]) if rec["rel_props"] else {},
                        "chunk_id": rec["chunk_id"],
                        "document_id": rec["document_id"],
                    }
                )

        return backup


def main():
    parser = argparse.ArgumentParser(
        description="Validate and repair Entity→Chunk sync between Neo4j and Qdrant"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, don't make any changes",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Delete orphan Entity→Chunk relationships",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup orphan relationships before repair (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup before repair",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output validation report to JSON file",
    )

    args = parser.parse_args()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    # Initialize connections
    from neo4j import GraphDatabase

    from src.shared.config import get_config, get_settings
    from src.shared.connections import CompatQdrantClient

    config = get_config()
    settings = get_settings()

    neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    qdrant_client = CompatQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    try:
        validator = EntityChunkSyncValidator(neo4j_driver, qdrant_client, config)

        if args.repair and not args.dry_run:
            result = validator.repair(backup=not args.no_backup)
        else:
            result = validator.validate()

        # Print summary
        print("\n" + "=" * 60)
        print("Entity-Chunk Sync Validation Report")
        print("=" * 60)
        print(f"Timestamp: {result.timestamp}")
        print("\nEntity Statistics:")
        print(f"  Total entities:     {result.total_entities}")
        print(f"  Connected entities: {result.connected_entities}")
        print(
            f"  Orphan entities:    {result.orphan_entities} ({result.orphan_entities/max(result.total_entities,1)*100:.1f}%)"
        )
        print("\nChunk Statistics:")
        print(f"  Neo4j chunks:  {result.total_chunks_neo4j}")
        print(f"  Qdrant chunks: {result.total_chunks_qdrant}")
        print(
            f"  Gap:           {result.total_chunks_neo4j - result.total_chunks_qdrant}"
        )
        print("\nEntity→Chunk Relationships:")
        print(f"  Total:  {result.total_entity_chunk_rels}")
        print(f"  Valid:  {result.valid_rels}")
        print(f"  Orphan: {result.orphan_rels}")
        print(f"\n  Sync Health: {result.sync_health_percent:.1f}%")

        if result.sample_orphan_rels:
            print("\nSample Orphan Relationships:")
            for rel in result.sample_orphan_rels[:5]:
                print(
                    f"  - Entity '{rel['entity_name']}' -[{rel['rel_type']}]-> Chunk {rel['chunk_id'][:16]}..."
                )

        if result.repaired:
            print("\nRepair Results:")
            print(f"  Relationships deleted: {result.rels_deleted}")
            if result.backup_file:
                print(f"  Backup file: {result.backup_file}")

        print("=" * 60)

        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nReport saved to: {args.report}")

        # Exit with error code if sync health is poor
        if result.sync_health_percent < 90:
            print("\n⚠️  WARNING: Sync health is below 90%!")
            sys.exit(1)

    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    main()

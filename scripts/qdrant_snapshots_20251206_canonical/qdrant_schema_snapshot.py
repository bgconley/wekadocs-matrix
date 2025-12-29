#!/usr/bin/env python3
"""
Qdrant Schema Snapshot & Restoration Tool

This script captures the current Qdrant collection schema as a snapshot and can
restore it idempotently. It's designed for:
- Disaster recovery (recreate collection from scratch)
- Environment parity (ensure dev/staging/prod have identical schemas)
- Schema version control (track schema changes as code)

Usage:
    # Capture current schema to a snapshot file
    python scripts/qdrant_schema_snapshot.py snapshot --collection chunks_multi_<dense_profile>

    # Restore schema from snapshot (idempotent - safe to run multiple times)
    python scripts/qdrant_schema_snapshot.py restore --collection chunks_multi_<dense_profile>

    # Validate current schema matches snapshot
    python scripts/qdrant_schema_snapshot.py validate --collection chunks_multi_<dense_profile>

    # Show diff between current schema and snapshot
    python scripts/qdrant_schema_snapshot.py diff --collection chunks_multi_<dense_profile>

Options:
    --host          Qdrant host (default: localhost, env: QDRANT_HOST)
    --port          Qdrant port (default: 6333, env: QDRANT_PORT)
    --collection    Collection name (required)
    --snapshot-dir  Directory for snapshot files (default: scripts/qdrant_snapshots/)
    --force         Force recreation even if collection exists (restore only)
    --dry-run       Show what would be done without making changes

Author: Claude Code
Generated: 2025-12-06
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        MultiVectorComparator,
        MultiVectorConfig,
        PayloadSchemaType,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )
except ImportError:
    print("Error: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)


# Default snapshot directory
DEFAULT_SNAPSHOT_DIR = Path(__file__).parent / "qdrant_snapshots"


class QdrantSchemaManager:
    """Manages Qdrant collection schema snapshots and restoration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        snapshot_dir: Optional[Path] = None,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.snapshot_dir = snapshot_dir or DEFAULT_SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _get_snapshot_path(self, collection_name: str) -> Path:
        """Get the snapshot file path for a collection."""
        return self.snapshot_dir / f"{collection_name}_schema.json"

    def _extract_schema(self, collection_name: str) -> Dict[str, Any]:
        """Extract complete schema from a live Qdrant collection."""
        info = self.client.get_collection(collection_name)
        config = info.config
        params = config.params

        # Extract vectors config
        vectors_config = {}
        vectors = params.vectors
        if isinstance(vectors, dict):
            for name, vp in vectors.items():
                dist_val = (
                    vp.distance.value
                    if hasattr(vp.distance, "value")
                    else str(vp.distance)
                )
                vec_dict = {
                    "size": vp.size,
                    "distance": dist_val,
                }
                if vp.multivector_config:
                    vec_dict["multivector_config"] = {
                        "comparator": (
                            vp.multivector_config.comparator.value
                            if hasattr(vp.multivector_config.comparator, "value")
                            else str(vp.multivector_config.comparator)
                        )
                    }
                if hasattr(vp, "hnsw_config") and vp.hnsw_config:
                    hnsw = vp.hnsw_config
                    vec_dict["hnsw_config"] = {
                        "m": getattr(hnsw, "m", None),
                    }
                vectors_config[name] = vec_dict

        # Extract sparse vectors config
        sparse_vectors_config = {}
        sparse = params.sparse_vectors
        if sparse:
            for name, sp in sparse.items():
                sparse_dict = {}
                if sp.index:
                    sparse_dict["index"] = {
                        "on_disk": getattr(sp.index, "on_disk", None),
                    }
                sparse_vectors_config[name] = sparse_dict

        # Extract HNSW config
        hnsw_config = {}
        if config.hnsw_config:
            hc = config.hnsw_config
            hnsw_config = {
                "m": hc.m,
                "ef_construct": hc.ef_construct,
                "full_scan_threshold": hc.full_scan_threshold,
                "on_disk": getattr(hc, "on_disk", False),
            }

        # Extract payload schema (indexed fields)
        payload_indexes = []
        if info.payload_schema:
            payload_indexes = sorted(info.payload_schema.keys())

        return {
            "_schema_version": "1.0",
            "_generated_at": datetime.now(timezone.utc).isoformat(),
            "_source": "live_qdrant_collection",
            "_collection_name": collection_name,
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config,
            "hnsw_config": hnsw_config,
            "on_disk_payload": getattr(params, "on_disk_payload", True),
            "payload_indexes": payload_indexes,
        }

    def snapshot(self, collection_name: str) -> Path:
        """Capture current schema and save to snapshot file."""
        schema = self._extract_schema(collection_name)
        snapshot_path = self._get_snapshot_path(collection_name)

        with open(snapshot_path, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"✅ Schema snapshot saved: {snapshot_path}")
        print(f"   Dense vectors: {list(schema['vectors_config'].keys())}")
        print(f"   Sparse vectors: {list(schema['sparse_vectors_config'].keys())}")
        print(f"   Payload indexes: {len(schema['payload_indexes'])} fields")

        return snapshot_path

    def load_snapshot(self, collection_name: str) -> Dict[str, Any]:
        """Load schema from snapshot file."""
        snapshot_path = self._get_snapshot_path(collection_name)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"No snapshot found: {snapshot_path}")

        with open(snapshot_path) as f:
            return json.load(f)

    def _build_vectors_config(self, schema: Dict) -> Dict[str, VectorParams]:
        """Build VectorParams from snapshot schema."""
        result = {}
        for name, cfg in schema.get("vectors_config", {}).items():
            distance = Distance.COSINE
            if cfg.get("distance") == "Dot":
                distance = Distance.DOT
            elif cfg.get("distance") == "Euclid":
                distance = Distance.EUCLID

            multivector = None
            if cfg.get("multivector_config"):
                comparator = MultiVectorComparator.MAX_SIM
                if cfg["multivector_config"].get("comparator") == "avg":
                    comparator = MultiVectorComparator.AVG
                multivector = MultiVectorConfig(comparator=comparator)

            hnsw = None
            if cfg.get("hnsw_config"):
                hnsw = HnswConfigDiff(m=cfg["hnsw_config"].get("m", 0))

            result[name] = VectorParams(
                size=cfg["size"],
                distance=distance,
                multivector_config=multivector,
                hnsw_config=hnsw,
            )

        return result

    def _build_sparse_config(self, schema: Dict) -> Dict[str, SparseVectorParams]:
        """Build SparseVectorParams from snapshot schema."""
        result = {}
        for name, cfg in schema.get("sparse_vectors_config", {}).items():
            index_cfg = cfg.get("index", {})
            result[name] = SparseVectorParams(
                index=SparseIndexParams(on_disk=index_cfg.get("on_disk", True))
            )
        return result

    def _schemas_match(
        self, current: Dict[str, Any], snapshot: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Compare current schema with snapshot. Returns (match, differences)."""
        differences = []

        # Compare vectors
        current_vecs = set(current.get("vectors_config", {}).keys())
        snapshot_vecs = set(snapshot.get("vectors_config", {}).keys())
        if current_vecs != snapshot_vecs:
            missing = snapshot_vecs - current_vecs
            extra = current_vecs - snapshot_vecs
            if missing:
                differences.append(f"Missing dense vectors: {missing}")
            if extra:
                differences.append(f"Extra dense vectors: {extra}")

        # Compare sparse vectors
        current_sparse = set(current.get("sparse_vectors_config", {}).keys())
        snapshot_sparse = set(snapshot.get("sparse_vectors_config", {}).keys())
        if current_sparse != snapshot_sparse:
            missing = snapshot_sparse - current_sparse
            extra = current_sparse - snapshot_sparse
            if missing:
                differences.append(f"Missing sparse vectors: {missing}")
            if extra:
                differences.append(f"Extra sparse vectors: {extra}")

        # Compare vector dimensions
        for vec_name in current_vecs & snapshot_vecs:
            curr_size = current["vectors_config"][vec_name].get("size")
            snap_size = snapshot["vectors_config"][vec_name].get("size")
            if curr_size != snap_size:
                differences.append(
                    f"Vector '{vec_name}' size mismatch: {curr_size} vs {snap_size}"
                )

        return len(differences) == 0, differences

    def validate(self, collection_name: str) -> bool:
        """Validate current schema matches snapshot."""
        try:
            snapshot = self.load_snapshot(collection_name)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False

        try:
            current = self._extract_schema(collection_name)
        except Exception as e:
            print(f"❌ Could not fetch current schema: {e}")
            return False

        match, differences = self._schemas_match(current, snapshot)

        if match:
            print(f"✅ Schema matches snapshot for '{collection_name}'")
            return True
        else:
            print(f"❌ Schema mismatch for '{collection_name}':")
            for diff in differences:
                print(f"   - {diff}")
            return False

    def diff(self, collection_name: str) -> None:
        """Show differences between current schema and snapshot."""
        try:
            snapshot = self.load_snapshot(collection_name)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

        try:
            current = self._extract_schema(collection_name)
        except Exception as e:
            print(f"❌ Could not fetch current schema: {e}")
            return

        print(f"\n=== Schema Diff: {collection_name} ===\n")
        print("Dense Vectors:")
        print(f"  Current:  {sorted(current.get('vectors_config', {}).keys())}")
        print(f"  Snapshot: {sorted(snapshot.get('vectors_config', {}).keys())}")

        print("\nSparse Vectors:")
        print(f"  Current:  {sorted(current.get('sparse_vectors_config', {}).keys())}")
        print(f"  Snapshot: {sorted(snapshot.get('sparse_vectors_config', {}).keys())}")

        print("\nPayload Indexes:")
        current_idx = set(current.get("payload_indexes", []))
        snapshot_idx = set(snapshot.get("payload_indexes", []))
        if current_idx == snapshot_idx:
            print(f"  Match: {len(current_idx)} fields")
        else:
            print(f"  Missing in current: {snapshot_idx - current_idx}")
            print(f"  Extra in current: {current_idx - snapshot_idx}")

    def restore(
        self,
        collection_name: str,
        force: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Restore collection schema from snapshot (idempotent)."""
        try:
            snapshot = self.load_snapshot(collection_name)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False

        # Check if collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        exists = collection_name in collections

        if exists and not force:
            # Validate existing schema
            current = self._extract_schema(collection_name)
            match, differences = self._schemas_match(current, snapshot)

            if match:
                print(
                    f"✅ Collection '{collection_name}' already exists "
                    "with correct schema"
                )
                return True
            else:
                print("⚠️  Collection exists but schema differs:")
                for diff in differences:
                    print(f"   - {diff}")
                print("   Use --force to recreate the collection")
                return False

        # Build configs from snapshot
        vectors_config = self._build_vectors_config(snapshot)
        sparse_config = self._build_sparse_config(snapshot)

        if dry_run:
            action = "recreate" if exists else "create"
            print(f"DRY RUN: Would {action} collection '{collection_name}'")
            print(f"  Dense vectors: {list(vectors_config.keys())}")
            print(f"  Sparse vectors: {list(sparse_config.keys())}")
            return True

        # Create or recreate collection
        if exists:
            print(f"⚠️  Deleting existing collection '{collection_name}'...")
            self.client.delete_collection(collection_name)

        print(f"Creating collection '{collection_name}'...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config if sparse_config else None,
            on_disk_payload=snapshot.get("on_disk_payload", True),
            hnsw_config=HnswConfigDiff(
                m=snapshot.get("hnsw_config", {}).get("m", 48),
                ef_construct=snapshot.get("hnsw_config", {}).get("ef_construct", 256),
            ),
        )

        # Create payload indexes
        integer_fields = ("order", "token_count", "embedding_dimensions", "updated_at")
        for field in snapshot.get("payload_indexes", []):
            try:
                # Infer schema type from field name conventions
                if field in integer_fields:
                    schema_type = PayloadSchemaType.INTEGER
                elif field in ("is_microdoc",):
                    schema_type = PayloadSchemaType.BOOL
                elif field in ("heading", "doc_title"):
                    schema_type = PayloadSchemaType.TEXT
                else:
                    schema_type = PayloadSchemaType.KEYWORD

                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception as e:
                print(f"   Warning: Could not create index for '{field}': {e}")

        print(f"✅ Collection '{collection_name}' restored from snapshot")
        print(f"   Dense vectors: {list(vectors_config.keys())}")
        print(f"   Sparse vectors: {list(sparse_config.keys())}")
        print(f"   Payload indexes: {len(snapshot.get('payload_indexes', []))} fields")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Qdrant Schema Snapshot & Restoration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=["snapshot", "restore", "validate", "diff"],
        help="Command to execute",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection name",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help=f"Snapshot directory (default: {DEFAULT_SNAPSHOT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation even if collection exists (restore only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    manager = QdrantSchemaManager(
        host=args.host,
        port=args.port,
        snapshot_dir=args.snapshot_dir,
    )

    if args.command == "snapshot":
        manager.snapshot(args.collection)
    elif args.command == "restore":
        success = manager.restore(
            args.collection,
            force=args.force,
            dry_run=args.dry_run,
        )
        sys.exit(0 if success else 1)
    elif args.command == "validate":
        success = manager.validate(args.collection)
        sys.exit(0 if success else 1)
    elif args.command == "diff":
        manager.diff(args.collection)


if __name__ == "__main__":
    main()

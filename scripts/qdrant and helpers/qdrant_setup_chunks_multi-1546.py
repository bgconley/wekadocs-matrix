# -*- coding: utf-8 -*-
"""
qdrant_setup_chunks_multi.py  (v2.2, idempotent, no providers)

Purpose:
  Create/ensure a side-by-side named-vector collection `chunks_multi`
  for hybrid retrieval (content + title [+ optional entity] embeddings).

Notes:
  - Idempotent: no destructive deletes. Safe to re-run.
  - Keeps your legacy single-vector collection `chunks` untouched.
  - Adds payload indexes used by filters / scoring.
  - Canonical ID: `document_id` (with `doc_id` kept as an alias).

Usage:
  python qdrant_setup_chunks_multi.py --host localhost --port 6333 --collection chunks_multi

Optional flags:
  --add-entity         Include an 'entity' named vector (1024-D) at creation time.
  --dims 1024          Dimensionality for vectors (default 1024).
"""

import argparse
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)


def ensure_collection(
    client: QdrantClient, name: str, dims: int = 1024, add_entity: bool = False
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        vectors = {
            "content": VectorParams(size=dims, distance=Distance.COSINE),
            "title": VectorParams(size=dims, distance=Distance.COSINE),
        }
        if add_entity:
            vectors["entity"] = VectorParams(size=dims, distance=Distance.COSINE)
        client.create_collection(
            collection_name=name,
            vectors_config=vectors,
            hnsw_config=HnswConfigDiff(
                m=48,
                ef_construct=256,
                full_scan_threshold=10000,
                max_indexing_threads=0,
                on_disk=False,
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                indexing_threshold=20000,
                deleted_threshold=0.2,
                vacuum_min_vector_number=2000,
                max_optimization_threads=1,
                flush_interval_sec=5,
            ),
            shard_number=1,
            replication_factor=1,
            write_consistency_factor=1,
            on_disk_payload=True,
        )
        print(f"[OK] Created collection '{name}'.")
    else:
        # Apply best-effort, non-destructive tuning updates if supported by this Qdrant version.
        try:
            client.update_collection(
                collection_name=name,
                hnsw_config=HnswConfigDiff(
                    m=48, ef_construct=256, full_scan_threshold=10000
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000,
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=2000,
                    max_optimization_threads=1,
                    flush_interval_sec=5,
                ),
            )
            print(
                f"[OK] Collection '{name}' already exists; applied non-destructive config updates."
            )
        except Exception:
            print(f"[OK] Collection '{name}' already exists; leaving as-is.")


def ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    fields = [
        # Identifiers (canonical + alias)
        ("id", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),  # canonical
        ("doc_id", PayloadSchemaType.KEYWORD),  # legacy alias
        ("parent_section_id", PayloadSchemaType.KEYWORD),
        # Structure & ordering
        ("order", PayloadSchemaType.INTEGER),
        ("heading", PayloadSchemaType.TEXT),
        # Filters / metadata
        ("updated_at", PayloadSchemaType.INTEGER),  # epoch seconds (recommended)
        ("doc_tag", PayloadSchemaType.KEYWORD),
        ("is_microdoc", PayloadSchemaType.BOOL),
        ("token_count", PayloadSchemaType.INTEGER),
        ("tenant", PayloadSchemaType.KEYWORD),
        ("lang", PayloadSchemaType.KEYWORD),
        ("version", PayloadSchemaType.KEYWORD),
        ("source_path", PayloadSchemaType.KEYWORD),
        # Embedding/version audit
        ("embedding_version", PayloadSchemaType.KEYWORD),
        ("embedding_provider", PayloadSchemaType.KEYWORD),
        ("embedding_dimensions", PayloadSchemaType.INTEGER),
        # Dedup / near-dup
        ("text_hash", PayloadSchemaType.KEYWORD),
        ("shingle_hash", PayloadSchemaType.KEYWORD),
    ]
    for name_field, schema in fields:
        try:
            client.create_payload_index(
                collection_name=name, field_name=name_field, field_schema=schema
            )
            print(f"[OK] Ensured payload index: {name_field}")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    ap.add_argument("--collection", default="chunks_multi")
    ap.add_argument("--dims", type=int, default=int(os.getenv("QDRANT_DIMS", "1024")))
    ap.add_argument("--add-entity", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port)
    ensure_collection(
        client, args.collection, dims=args.dims, add_entity=args.add_entity
    )
    ensure_payload_indexes(client, args.collection)
    print(f"[READY] Qdrant collection '{args.collection}' is ready.")


if __name__ == "__main__":
    main()

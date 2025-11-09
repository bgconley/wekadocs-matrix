# -*- coding: utf-8 -*-
"""
qdrant_setup_chunks_multi.py

Purpose:
  Create/ensure a side-by-side named-vector collection `chunks_multi`
  for hybrid retrieval (content + title [+ optional entity] embeddings).

Notes:
  - Idempotent: no destructive deletes. Safe to re-run.
  - Leaves legacy single-vector collection `chunks` untouched.
  - Adds payload indexes used by filters/scoring.
  - Canonical: `document_id`; `doc_id` is a legacy alias (both indexed).

Usage:
  python qdrant_setup_chunks_multi.py --host localhost --port 6333
  # or rely on env: QDRANT_HOST, QDRANT_PORT
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

COLL = "chunks_multi"


def ensure_collection(client: QdrantClient, dims: int = 1024) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLL not in existing:
        client.create_collection(
            collection_name=COLL,
            vectors_config={
                "content": VectorParams(size=dims, distance=Distance.COSINE),
                "title": VectorParams(size=dims, distance=Distance.COSINE),
                # Enable later when you start storing entity-name vectors:
                # "entity":  VectorParams(size=dims, distance=Distance.COSINE),
            },
            hnsw_config=HnswConfigDiff(
                m=48,
                ef_construct=256,
                full_scan_threshold=10000,
                max_indexing_threads=0,
                on_disk=False,
            ),
            optimizer_config=OptimizersConfigDiff(
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
        print(f"[qdrant] Created collection '{COLL}'.")
    else:
        # Best-effort tuning updates (non-destructive). If unsupported, Qdrant will ignore.
        client.update_collection(
            collection_name=COLL,
            hnsw_config=HnswConfigDiff(
                m=48, ef_construct=256, full_scan_threshold=10000
            ),
            optimizer_config=OptimizersConfigDiff(
                default_segment_number=2,
                indexing_threshold=20000,
                deleted_threshold=0.2,
                vacuum_min_vector_number=2000,
                max_optimization_threads=1,
                flush_interval_sec=5,
            ),
        )
        print(
            f"[qdrant] Collection '{COLL}' exists; applied non-destructive config updates."
        )


def ensure_payload_indexes(client: QdrantClient) -> None:
    fields: Iterable[Tuple[str, PayloadSchemaType]] = [
        # Identifiers (canonical + alias)
        ("id", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),  # canonical
        ("doc_id", PayloadSchemaType.KEYWORD),  # legacy alias
        # Structure & ordering
        ("parent_section_id", PayloadSchemaType.KEYWORD),
        ("order", PayloadSchemaType.INTEGER),
        ("heading", PayloadSchemaType.TEXT),
        # Filters / metadata
        ("updated_at", PayloadSchemaType.INTEGER),  # epoch seconds
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
    for name, schema in fields:
        try:
            client.create_payload_index(
                collection_name=COLL, field_name=name, field_schema=schema
            )
            print(f"[qdrant] Ensured payload index: {name}")
        except Exception:
            # Likely already exists; ignore
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    p.add_argument("--dims", type=int, default=int(os.getenv("QDRANT_DIMS", "1024")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = QdrantClient(host=args.host, port=args.port)
    ensure_collection(client, dims=args.dims)
    ensure_payload_indexes(client)
    print(f"[qdrant] Collection '{COLL}' is ready.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
qdrant_setup_chunks_multi.py

Purpose:
  Create/ensure a side-by-side named-vector collection `chunks_multi`
  for hybrid retrieval (content + title [+ optional entity] embeddings).

Notes:
  - Idempotent: no destructive deletes. Safe to re-run.
  - Keeps your legacy single-vector collection `chunks` untouched.
  - Adds payload indexes used by filters / scoring.
Canonical:
  document_id is canonical. doc_id is a legacy alias kept for compatibility.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

HOST = "localhost"
PORT = 6333
COLL = "chunks_multi"


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLL not in existing:
        client.create_collection(
            collection_name=COLL,
            vectors_config={
                "content": VectorParams(size=1024, distance=Distance.COSINE),
                "title": VectorParams(size=1024, distance=Distance.COSINE),
                # Enable when you start storing entity-name vectors:
                # "entity":  VectorParams(size=1024, distance=Distance.COSINE),
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
        print(f"Created collection '{COLL}'.")
    else:
        print(f"Collection '{COLL}' already exists; leaving as-is.")


def ensure_payload_indexes(client: QdrantClient) -> None:
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
        ("provider", PayloadSchemaType.KEYWORD),
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
            print(f"Ensured payload index: {name}")
        except Exception:
            # Likely already exists; ignore
            pass


def main() -> None:
    client = QdrantClient(host=HOST, port=PORT)
    ensure_collection(client)
    ensure_payload_indexes(client)
    print(f"Qdrant collection '{COLL}' is ready.")


if __name__ == "__main__":
    main()

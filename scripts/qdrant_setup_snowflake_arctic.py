#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qdrant_setup_snowflake_arctic.py

Create the chunks_multi_snowflake_arctic_v2l collection for the Snowflake Arctic
dense embedder. Idempotent - safe to re-run.

Usage:
    python scripts/qdrant_setup_snowflake_arctic.py
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    OptimizersConfigDiff,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

HOST = "localhost"
PORT = 6333
COLLECTION_NAME = "chunks_multi_snowflake_arctic_v2l"


def ensure_collection(client: QdrantClient) -> bool:
    """Create collection if it doesn't exist. Returns True if created."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists; leaving as-is.")
        return False

    # Dense vectors (1024 dims, cosine)
    vectors_config = {
        "content": VectorParams(size=1024, distance=Distance.COSINE),
        "title": VectorParams(size=1024, distance=Distance.COSINE),
        "doc_title": VectorParams(size=1024, distance=Distance.COSINE),
        "late-interaction": VectorParams(
            size=1024,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
        ),
    }

    # Sparse vectors (for hybrid search via BGE-M3)
    sparse_vectors_config = {
        "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True)),
        "title-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True)),
        "doc_title-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True)),
        "entity-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True)),
    }

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
        hnsw_config=HnswConfigDiff(
            m=48,
            ef_construct=256,
            full_scan_threshold=10000,
            max_indexing_threads=0,
            on_disk=False,
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=0,
            indexing_threshold=10000,
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            max_optimization_threads=None,
            flush_interval_sec=5,
        ),
        shard_number=1,
        replication_factor=1,
        write_consistency_factor=1,
        on_disk_payload=True,
    )
    print(f"Created collection '{COLLECTION_NAME}'.")
    return True


def ensure_payload_indexes(client: QdrantClient) -> None:
    """Create payload indexes matching existing collections."""
    indexes = [
        # Identifiers
        ("id", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("parent_section_id", PayloadSchemaType.KEYWORD),
        ("parent_section_original_id", PayloadSchemaType.KEYWORD),
        ("node_id", PayloadSchemaType.KEYWORD),
        ("kg_id", PayloadSchemaType.KEYWORD),
        # Structure & ordering
        ("order", PayloadSchemaType.INTEGER),
        ("heading", PayloadSchemaType.TEXT),
        ("doc_title", PayloadSchemaType.TEXT),
        ("parent_path", PayloadSchemaType.TEXT),
        ("parent_path_depth", PayloadSchemaType.INTEGER),
        ("line_start", PayloadSchemaType.INTEGER),
        ("line_end", PayloadSchemaType.INTEGER),
        # Content classification
        ("block_type", PayloadSchemaType.KEYWORD),
        ("has_code", PayloadSchemaType.BOOL),
        ("has_table", PayloadSchemaType.BOOL),
        ("code_ratio", PayloadSchemaType.FLOAT),
        # Entity metadata
        ("entity_metadata.entity_count", PayloadSchemaType.INTEGER),
        ("entity_metadata.entity_types", PayloadSchemaType.KEYWORD),
        ("entity_metadata.entity_values", PayloadSchemaType.KEYWORD),
        ("entity_metadata.entity_values_normalized", PayloadSchemaType.KEYWORD),
        # Filters / metadata
        ("updated_at", PayloadSchemaType.INTEGER),
        ("doc_tag", PayloadSchemaType.KEYWORD),
        ("is_microdoc", PayloadSchemaType.BOOL),
        ("token_count", PayloadSchemaType.INTEGER),
        ("tenant", PayloadSchemaType.KEYWORD),
        ("lang", PayloadSchemaType.KEYWORD),
        ("version", PayloadSchemaType.KEYWORD),
        ("source_path", PayloadSchemaType.KEYWORD),
        ("snapshot_scope", PayloadSchemaType.KEYWORD),
        # Embedding audit
        ("embedding_version", PayloadSchemaType.KEYWORD),
        ("embedding_provider", PayloadSchemaType.KEYWORD),
        ("embedding_dimensions", PayloadSchemaType.INTEGER),
        # Deduplication
        ("text_hash", PayloadSchemaType.KEYWORD),
        ("shingle_hash", PayloadSchemaType.KEYWORD),
    ]

    created = 0
    skipped = 0
    for name, schema in indexes:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=name,
                field_schema=schema,
            )
            created += 1
        except Exception:
            # Already exists
            skipped += 1

    print(f"Payload indexes: {created} created, {skipped} already existed.")


def main() -> None:
    print(f"Connecting to Qdrant at {HOST}:{PORT}...")
    client = QdrantClient(host=HOST, port=PORT)

    ensure_collection(client)
    ensure_payload_indexes(client)

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection '{COLLECTION_NAME}' ready:")
    print(f"  Status: {info.status}")
    print(f"  Vectors: {list(info.config.params.vectors.keys())}")
    print(f"  Sparse: {list(info.config.params.sparse_vectors.keys())}")
    print(f"  Points: {info.points_count}")


if __name__ == "__main__":
    main()

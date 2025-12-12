#!/usr/bin/env python3
"""Quick test script to verify sparse title/entity vector ingestion."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.ingestion.build_graph import DocumentGraphBuilder
from src.providers.embedding_factory import get_embedding_provider
from src.shared.config import get_config


def main():
    print("=" * 60)
    print(" SPARSE VECTOR INGESTION TEST")
    print("=" * 60)

    # Load config
    config = get_config()

    # Initialize embedder
    print("\n1. Initializing embedder...")
    embedder = get_embedding_provider()
    print(f"   Provider: {type(embedder).__name__}")
    print(f"   Supports sparse: {hasattr(embedder, 'embed_sparse')}")

    # Initialize Qdrant client
    print("\n2. Connecting to Qdrant...")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_client = QdrantClient(host=qdrant_host, port=6333)

    # Check if collection exists (should have been deleted)
    collections = [c.name for c in qdrant_client.get_collections().collections]
    collection_name = config.search.vector.qdrant.collection_name
    print(f"   Collection '{collection_name}' exists: {collection_name in collections}")

    # Initialize builder
    print("\n3. Initializing DocumentGraphBuilder...")
    builder = DocumentGraphBuilder(
        embedder=embedder,
        include_entity_vector=False,  # We removed dense entity
    )

    # Load a test document
    test_doc_path = Path("data/ingest/additional-protocols_nfs-support_nfs-support.md")
    if not test_doc_path.exists():
        print(f"   ❌ Test doc not found: {test_doc_path}")
        return 1

    print(f"\n4. Loading test document: {test_doc_path.name}")
    content = test_doc_path.read_text()
    print(f"   Content length: {len(content)} chars")

    # Create document dict
    doc_dict = {
        "doc_id": "test_sparse_vectors_001",
        "document_id": "test_sparse_vectors_001",
        "title": "NFS Support Overview",
        "content": content,
        "source_path": str(test_doc_path),
        "tenant": "test",
    }

    # Run ingestion
    print("\n5. Running ingestion...")
    try:
        result = builder.upsert_document(doc_dict)
        print("   ✅ Ingestion complete!")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Ingestion failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Verify Qdrant collection schema
    print("\n6. Verifying Qdrant collection schema...")
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        vectors = collection_info.config.params.vectors
        sparse = collection_info.config.params.sparse_vectors

        print(f"   Points: {collection_info.points_count}")
        print(f"   Dense vectors: {list(vectors.keys()) if vectors else 'None'}")
        print(f"   Sparse vectors: {list(sparse.keys()) if sparse else 'None'}")

        # Check for new sparse vectors
        has_title_sparse = sparse and "title-sparse" in sparse
        has_entity_sparse = sparse and "entity-sparse" in sparse

        print(f"\n   title-sparse present: {'✅' if has_title_sparse else '❌'}")
        print(f"   entity-sparse present: {'✅' if has_entity_sparse else '❌'}")

        if has_title_sparse and has_entity_sparse:
            print("\n✅ SUCCESS: New sparse vectors are in schema!")
        else:
            print("\n❌ FAIL: Missing new sparse vectors")
            return 1

    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print(" TEST COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Quick integration test for doc_title vector query integration.

Tests that the sec_001 query ("Enforce security and compliance in WEKA")
now ranks better with doc_title vectors enabled.

Usage:
    BGE_M3_API_URL=http://127.0.0.1:9000 \
    NEO4J_URI=bolt://localhost:7687 \
    NEO4J_PASSWORD=testpassword123 \
    QDRANT_HOST=localhost \
    python scripts/test_doc_title_integration.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sec_001_query():
    """Test that security query now ranks better with doc_title integration."""
    from qdrant_client import QdrantClient

    from src.providers.factory import ProviderFactory
    from src.query.hybrid_retrieval import QdrantMultiVectorRetriever
    from src.shared.config import get_config
    from src.shared.connections import get_connection_manager

    # Get config and connections
    get_config()  # Ensure config is loaded (required for connection manager)
    cm = get_connection_manager()
    neo4j_driver = cm.get_neo4j_driver()

    # Create embedding provider
    embedder = ProviderFactory.create_embedding_provider()

    # Create Qdrant client
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    # Get doc_id -> title mapping
    with neo4j_driver.session() as session:
        result = session.run(
            "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.title AS title"
        )
        doc_titles = {r["doc_id"]: r["title"] for r in result}

    # Create retriever with doc_title in field_weights (should be default now)
    field_weights = {
        "content": 1.0,
        "title": 0.35,
        "doc_title": 0.2,  # This is the new integration
        "entity": 0.2,
    }

    retriever = QdrantMultiVectorRetriever(
        qdrant_client=qdrant,
        embedder=embedder,
        collection_name="chunks_multi_bge_m3",
        field_weights=field_weights,
        schema_supports_sparse=True,
        schema_supports_doc_title_sparse=True,
        use_query_api=True,
        query_api_weighted_fusion=True,  # Use weighted path for per-field scoring
    )

    # Test query
    query = "Enforce security and compliance in WEKA"
    expected_title = "Enforce security and compliance"

    print(f"\n{'=' * 70}")
    print("doc_title Vector Integration Test")
    print(f"{'=' * 70}")
    print(f"Query: {query}")
    print(f"Expected doc title: {expected_title}")
    print(f"Field weights: {field_weights}")
    print()

    # Run search
    results = retriever.search(query, top_k=20)

    print("Top 20 results:")
    print("-" * 70)

    target_rank = None
    for i, chunk in enumerate(results, start=1):
        doc_id = chunk.document_id
        doc_title = doc_titles.get(doc_id, "Unknown")
        score = chunk.fused_score or chunk.vector_score or 0.0
        doc_title_score = chunk.doc_title_vec_score or 0.0

        # Check if this is the target document
        is_target = expected_title.lower() in doc_title.lower()
        marker = " <-- TARGET" if is_target else ""
        if is_target and target_rank is None:
            target_rank = i

        # Show detailed scores for top 10
        if i <= 10:
            print(
                f"  {i:2}. [{score:.4f}] doc_title_score={doc_title_score:.4f} | {doc_title[:50]}{marker}"
            )

    print("-" * 70)
    print()

    # Report result
    if target_rank is not None:
        if target_rank <= 5:
            print(f"SUCCESS! Target document ranked #{target_rank} (top 5)")
            status = "PASS"
        elif target_rank <= 10:
            print(f"IMPROVED! Target document ranked #{target_rank} (top 10)")
            status = "MARGINAL"
        else:
            print(f"NEEDS WORK: Target document ranked #{target_rank} (not in top 10)")
            status = "FAIL"
    else:
        print("FAIL: Target document not found in top 20 results")
        status = "FAIL"

    print()

    # Show doc_title contribution
    print("Analysis:")
    print("-" * 70)
    if results:
        top_result = results[0]
        print(
            f"  Top result doc_title_score: {top_result.doc_title_vec_score or 0.0:.4f}"
        )
        print(f"  Top result fused_score: {top_result.fused_score or 0.0:.4f}")

        if target_rank and target_rank <= len(results):
            target_result = results[target_rank - 1]
            print(
                f"  Target result doc_title_score: {target_result.doc_title_vec_score or 0.0:.4f}"
            )
            print(
                f"  Target result fused_score: {target_result.fused_score or 0.0:.4f}"
            )

    print(f"\n{'=' * 70}")
    print(f"TEST STATUS: {status}")
    print(f"{'=' * 70}\n")

    return status == "PASS" or status == "MARGINAL"


def test_doc_title_prefetch_active():
    """Verify doc_title is in dense_vector_names when configured."""
    from src.query.hybrid_retrieval import QdrantMultiVectorRetriever

    class MockClient:
        pass

    class MockEmbedder:
        def embed_sparse(self, text):
            return {"indices": [1], "values": [1.0]}

    retriever = QdrantMultiVectorRetriever(
        qdrant_client=MockClient(),
        embedder=MockEmbedder(),
        field_weights={
            "content": 1.0,
            "title": 0.35,
            "doc_title": 0.2,
            "entity": 0.2,
        },
        schema_supports_sparse=True,
        schema_supports_doc_title_sparse=True,
    )

    assert (
        "doc_title" in retriever.dense_vector_names
    ), "doc_title not in dense_vector_names"
    assert (
        retriever.schema_supports_doc_title_sparse
    ), "schema_supports_doc_title_sparse not set"

    print("Prefetch config test: PASS")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("doc_title Vector Integration Tests")
    print("=" * 70 + "\n")

    # Test 1: Config test
    print("Test 1: Verify prefetch configuration")
    test_doc_title_prefetch_active()
    print()

    # Test 2: Live query test
    print("Test 2: Live query ranking test")
    success = test_sec_001_query()

    sys.exit(0 if success else 1)

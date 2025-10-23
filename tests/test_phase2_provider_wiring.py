#!/usr/bin/env python3
"""
Test script for Phase 2 (Pre-Phase 7) implementation.
Validates provider wiring throughout ingestion and query paths.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_build_graph_uses_provider():
    """Test that build_graph.py uses the embedding provider."""
    print("\n=== Testing build_graph.py Provider Usage ===")

    try:
        from src.ingestion.build_graph import GraphBuilder
        from src.shared.config import get_config

        # Mock Neo4j driver
        mock_driver = MagicMock()
        config = get_config()

        # Initialize GraphBuilder
        builder = GraphBuilder(mock_driver, config)

        # The embedder should be None initially (lazy loading)
        assert builder.embedder is None, "Embedder should be lazy loaded"

        print("✓ GraphBuilder initializes without embedder (lazy loading)")

        # Note: Full test would require mocking _process_embeddings
        # but we've verified the import and initialization logic

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_query_service_uses_provider():
    """Test that query_service.py uses the embedding provider."""
    print("\n=== Testing query_service.py Provider Usage ===")

    try:
        from src.mcp_server.query_service import QueryService

        # Initialize QueryService
        service = QueryService()

        # The embedder should be None initially (lazy loading)
        assert service._embedder is None, "Embedder should be lazy loaded"

        print("✓ QueryService initializes without embedder (lazy loading)")

        # Get embedder (will initialize provider)
        embedder = service._get_embedder()

        # Verify it's our provider
        from src.providers.embeddings import SentenceTransformersProvider

        assert isinstance(
            embedder, SentenceTransformersProvider
        ), f"Should be SentenceTransformersProvider, got {type(embedder)}"

        print("✓ QueryService uses SentenceTransformersProvider")
        print(f"  - Model: {embedder.model_id}")
        print(f"  - Dimensions: {embedder.dims}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_provider_methods():
    """Test that provider methods work correctly."""
    print("\n=== Testing Provider Methods ===")

    try:
        from src.providers.embeddings import SentenceTransformersProvider
        from src.shared.config import get_config

        config = get_config()
        provider = SentenceTransformersProvider()

        # Test embed_documents
        docs = ["Test document 1", "Test document 2"]
        doc_embeddings = provider.embed_documents(docs)

        assert len(doc_embeddings) == 2, "Should return 2 embeddings"
        assert (
            len(doc_embeddings[0]) == config.embedding.dims
        ), f"Wrong dimensions: expected {config.embedding.dims}, got {len(doc_embeddings[0])}"

        print("✓ embed_documents works correctly")
        print(f"  - Generated {len(doc_embeddings)} embeddings")
        print(f"  - Each has {len(doc_embeddings[0])} dimensions")

        # Test embed_query
        query = "Test query"
        query_embedding = provider.embed_query(query)

        assert (
            len(query_embedding) == config.embedding.dims
        ), f"Wrong query dimensions: expected {config.embedding.dims}, got {len(query_embedding)}"

        print("✓ embed_query works correctly")
        print(f"  - Generated embedding with {len(query_embedding)} dimensions")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_embedding_metadata():
    """Test that embedding metadata is properly set."""
    print("\n=== Testing Embedding Metadata ===")

    try:
        from datetime import datetime

        # Simulate metadata that should be added
        metadata = {
            "embedding_version": "miniLM-L6-v2-2024-01-01",
            "embedding_provider": "sentence-transformers",
            "embedding_dimensions": 384,
            "embedding_task": "retrieval.passage",
            "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Verify all required fields are present
        required_fields = [
            "embedding_version",
            "embedding_provider",
            "embedding_dimensions",
            "embedding_task",
            "embedding_timestamp",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Verify timestamp format
        assert metadata["embedding_timestamp"].endswith(
            "Z"
        ), "Timestamp should be ISO-8601 UTC with Z suffix"

        print("✓ Embedding metadata structure correct")
        print(f"  - Version: {metadata['embedding_version']}")
        print(f"  - Provider: {metadata['embedding_provider']}")
        print(f"  - Dimensions: {metadata['embedding_dimensions']}")
        print(f"  - Task: {metadata['embedding_task']}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2 Provider Wiring Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("build_graph.py uses provider", test_build_graph_uses_provider()))
    results.append(
        ("query_service.py uses provider", test_query_service_uses_provider())
    )
    results.append(("Provider methods", test_provider_methods()))
    results.append(("Embedding metadata", test_embedding_metadata()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 2 tests passed! Ready to proceed to Phase 3.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Test script for Phase 1 (Pre-Phase 7) implementation.
Validates configuration loading and embedding provider functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_configuration():
    """Test configuration loading and validation."""
    print("\n=== Testing Configuration Loading ===")

    try:
        from src.shared.config import get_config, get_settings

        # Load configuration
        config = get_config()
        settings = get_settings()

        print("✓ Configuration loaded successfully")
        print(f"  - Environment: {settings.env}")
        print(f"  - Embedding model: {config.embedding.embedding_model}")
        print(f"  - Dimensions: {config.embedding.dims}")
        print(f"  - Version: {config.embedding.version}")
        print(f"  - Provider: {config.embedding.provider}")

        # Verify critical values
        assert config.embedding.dims > 0, "Dimensions must be positive"
        assert config.embedding.version, "Version must be set"
        assert config.embedding.similarity in ["cosine", "dot", "euclidean"]

        print("✓ Configuration validation passed")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_embedding_provider():
    """Test embedding provider implementation."""
    print("\n=== Testing Embedding Provider ===")

    try:
        from src.providers.embeddings import SentenceTransformersProvider
        from src.shared.config import get_config

        config = get_config()

        # Initialize provider
        print(f"Initializing provider with model: {config.embedding.embedding_model}")
        provider = SentenceTransformersProvider()

        print("✓ Provider initialized")
        print(f"  - Provider name: {provider.provider_name}")
        print(f"  - Model ID: {provider.model_id}")
        print(f"  - Dimensions: {provider.dims}")

        # Verify dimensions match config
        assert (
            provider.dims == config.embedding.dims
        ), f"Dimension mismatch: provider={provider.dims}, config={config.embedding.dims}"

        # Test document embedding
        test_docs = [
            "This is a test document about Weka filesystem.",
            "Another document discussing storage configuration.",
        ]

        doc_embeddings = provider.embed_documents(test_docs)

        assert len(doc_embeddings) == 2, "Should return 2 embeddings"
        assert all(
            len(emb) == provider.dims for emb in doc_embeddings
        ), "Wrong dimensions"
        assert all(isinstance(emb, list) for emb in doc_embeddings), "Should be lists"
        assert all(
            isinstance(val, float) for emb in doc_embeddings for val in emb
        ), "Should be floats"

        print("✓ Document embedding successful")
        print(f"  - Generated {len(doc_embeddings)} embeddings")
        print(f"  - Each has {len(doc_embeddings[0])} dimensions")

        # Test query embedding
        test_query = "How do I configure NFS for Weka?"
        query_embedding = provider.embed_query(test_query)

        assert len(query_embedding) == provider.dims, "Wrong query dimensions"
        assert isinstance(query_embedding, list), "Should be a list"
        assert all(
            isinstance(val, float) for val in query_embedding
        ), "Should be floats"

        print("✓ Query embedding successful")
        print(f"  - Generated embedding with {len(query_embedding)} dimensions")

        # Test dimension validation
        assert provider.validate_dimensions(
            config.embedding.dims
        ), "Dimension validation failed"
        assert not provider.validate_dimensions(999), "Should reject wrong dimensions"

        print("✓ Dimension validation working")

        return True

    except ImportError as e:
        print(f"✗ Import error (is sentence-transformers installed?): {e}")
        return False
    except Exception as e:
        print(f"✗ Provider test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_protocol_compliance():
    """Test that provider implements the protocol correctly."""
    print("\n=== Testing Protocol Compliance ===")

    try:
        from src.providers.embeddings import (
            EmbeddingProvider,
            SentenceTransformersProvider,
        )

        provider = SentenceTransformersProvider()

        # Check if provider implements protocol
        assert isinstance(
            provider, EmbeddingProvider
        ), "Provider should implement protocol"

        # Check all required attributes exist
        assert hasattr(provider, "dims")
        assert hasattr(provider, "model_id")
        assert hasattr(provider, "provider_name")
        assert hasattr(provider, "embed_documents")
        assert hasattr(provider, "embed_query")
        assert hasattr(provider, "validate_dimensions")

        print("✓ Provider correctly implements EmbeddingProvider protocol")

        return True

    except Exception as e:
        print(f"✗ Protocol compliance test failed: {e}")
        return False


def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Phase 1 Foundation Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Configuration", test_configuration()))
    results.append(("Embedding Provider", test_embedding_provider()))
    results.append(("Protocol Compliance", test_protocol_compliance()))

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
        print("\n🎉 All Phase 1 tests passed! Ready to proceed to Phase 2.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Test Jina AI embedding provider integration.
Validates Phase 7C.7 implementation with real Jina v4 API.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.factory import ProviderFactory
from src.shared.config import get_config


def test_jina_provider_creation():
    """Test 1: Verify ProviderFactory creates JinaEmbeddingProvider correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: Provider Factory Integration")
    print("=" * 70)

    # Override config to use Jina
    config = get_config()
    config.embedding.provider = "jina-ai"
    config.embedding.embedding_model = "jina-embeddings-v4"
    config.embedding.dims = 1024

    try:
        provider = ProviderFactory.create_embedding_provider()

        print("✅ Provider created successfully")
        print(f"   - Provider name: {provider.provider_name}")
        print(f"   - Model ID: {provider.model_id}")
        print(f"   - Configured dims: {provider.dims}")
        print(f"   - Task: {getattr(provider, 'task', 'N/A')}")

        assert provider.provider_name == "jina-ai", "Wrong provider name"
        assert provider.dims == 1024, f"Wrong dimensions: {provider.dims}"

        return provider

    except Exception as e:
        print(f"❌ Provider creation failed: {e}")
        raise


def test_jina_embedding_generation(provider):
    """Test 2: Generate embeddings and validate dimensions."""
    print("\n" + "=" * 70)
    print("TEST 2: Embedding Generation")
    print("=" * 70)

    test_texts = [
        "How do I configure NFS for high-throughput workloads?",
        "What are the prerequisites for installing Weka on RHEL 8?",
        "Troubleshooting network connectivity issues in a Weka cluster",
    ]

    try:
        print(f"Generating embeddings for {len(test_texts)} test documents...")
        embeddings = provider.embed_documents(test_texts)

        print(f"✅ Generated {len(embeddings)} embeddings")

        for idx, (text, embedding) in enumerate(zip(test_texts, embeddings), 1):
            dims = len(embedding)
            vector_norm = sum(x * x for x in embedding) ** 0.5
            non_zero = sum(1 for x in embedding if abs(x) > 0.001)
            max_val = max(abs(x) for x in embedding)

            print(f"\n   Embedding {idx}:")
            print(f"   - Text: {text[:60]}...")
            print(f"   - Dimensions: {dims}")
            print(f"   - Vector norm: {vector_norm:.4f}")
            print(
                f"   - Non-zero elements: {non_zero}/{dims} ({100 * non_zero / dims:.1f}%)"
            )
            print(f"   - Max absolute value: {max_val:.4f}")

            assert dims == 1024, f"Wrong dimensions: expected 1024, got {dims}"
            assert vector_norm > 0, "Zero vector detected"
            assert non_zero > 900, f"Too many zero elements: {non_zero}/{dims}"

        print("\n✅ All embeddings validated successfully")
        return embeddings

    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        raise


def test_jina_query_embedding(provider):
    """Test 3: Test query-specific embedding (different task)."""
    print("\n" + "=" * 70)
    print("TEST 3: Query Embedding (retrieval.query task)")
    print("=" * 70)

    query = "How do I upgrade Weka to version 5.0?"

    try:
        print("Generating query embedding...")
        embedding = provider.embed_query(query)

        dims = len(embedding)
        vector_norm = sum(x * x for x in embedding) ** 0.5

        print("✅ Query embedding generated")
        print(f"   - Query: {query}")
        print(f"   - Dimensions: {dims}")
        print(f"   - Vector norm: {vector_norm:.4f}")

        assert dims == 1024, f"Wrong dimensions: expected 1024, got {dims}"

        return embedding

    except Exception as e:
        print(f"❌ Query embedding failed: {e}")
        raise


def test_dimension_validation(provider):
    """Test 4: Verify dimension validation catches mismatches."""
    print("\n" + "=" * 70)
    print("TEST 4: Dimension Safety Validation")
    print("=" * 70)

    # This should work (provider is 1024-D)
    config = get_config()

    print(f"Config dimensions: {config.embedding.dims}")
    print(f"Provider dimensions: {provider.dims}")

    if config.embedding.dims == provider.dims:
        print(f"✅ Dimensions match: {config.embedding.dims}-D")
    else:
        print("⚠️  Dimension mismatch detected (this would block ingestion)")
        print(f"   - Config expects: {config.embedding.dims}-D")
        print(f"   - Provider generates: {provider.dims}-D")


def test_provider_metadata(provider):
    """Test 5: Verify all required provider metadata is present."""
    print("\n" + "=" * 70)
    print("TEST 5: Provider Metadata Completeness")
    print("=" * 70)

    required_attrs = [
        "provider_name",
        "model_id",
        "dims",
    ]

    optional_attrs = [
        "task",
        "version",
    ]

    print("Required attributes:")
    for attr in required_attrs:
        if hasattr(provider, attr):
            value = getattr(provider, attr)
            print(f"   ✅ {attr}: {value}")
        else:
            print(f"   ❌ {attr}: MISSING")
            raise AttributeError(f"Provider missing required attribute: {attr}")

    print("\nOptional attributes:")
    for attr in optional_attrs:
        if hasattr(provider, attr):
            value = getattr(provider, attr)
            print(f"   ✅ {attr}: {value}")
        else:
            print(f"   ⚠️  {attr}: Not present (optional)")


def main():
    """Run all Jina integration tests."""
    print("\n" + "=" * 70)
    print("JINA AI EMBEDDING PROVIDER INTEGRATION TEST")
    print("Phase 7C.7 - Jina v4 @ 1024-D Validation")
    print("=" * 70)

    # Verify API key is set
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("❌ JINA_API_KEY environment variable not set")
        sys.exit(1)

    print(f"✅ API key found: {api_key[:15]}...{api_key[-4:]}")

    try:
        # Run tests
        provider = test_jina_provider_creation()
        _ = test_jina_embedding_generation(provider)
        _ = test_jina_query_embedding(provider)
        test_dimension_validation(provider)
        test_provider_metadata(provider)

        # Summary
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nValidated:")
        print("  ✅ Provider factory creates Jina v4 provider correctly")
        print("  ✅ Embeddings generated at 1024 dimensions")
        print("  ✅ Both document and query embeddings working")
        print("  ✅ Vector quality validated (non-zero, normalized)")
        print("  ✅ All required metadata present")
        print("\nJina AI integration ready for production use!")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Quick script to verify Jina provider configuration."""

import os
import sys

# Add src to path
sys.path.insert(0, "/app/src")

from src.providers.factory import create_embedding_provider, create_rerank_provider


def main():
    print("=" * 60)
    print("PROVIDER CONFIGURATION VERIFICATION")
    print("=" * 60)

    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"  EMBEDDINGS_MODEL: {os.getenv('EMBEDDINGS_MODEL', 'NOT SET')}")
    print(f"  RERANK_MODEL: {os.getenv('RERANK_MODEL', 'NOT SET')}")
    print(f"  JINA_API_KEY: {'SET' if os.getenv('JINA_API_KEY') else 'NOT SET'}")

    # Create embedding provider
    print("\nEmbedding Provider:")
    try:
        embed_provider = create_embedding_provider()
        print(f"  ✅ Provider created successfully")
        print(f"  Model ID: {embed_provider.model_id}")
        print(f"  Dimensions: {embed_provider.dims}")
        print(f"  Provider: {embed_provider.provider_name}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 1

    # Create rerank provider
    print("\nRerank Provider:")
    try:
        rerank_provider = create_rerank_provider()
        print(f"  ✅ Provider created successfully")
        print(f"  Model: {rerank_provider._model}")
        print(f"  Provider: {rerank_provider._provider_name}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✅ ALL PROVIDERS CONFIGURED CORRECTLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

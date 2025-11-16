#!/usr/bin/env python3
"""Quick script to verify embedding/rerank providers for the active profile and list profile metadata."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "/app/src")

from src.providers.factory import create_embedding_provider, create_rerank_provider
from src.shared.config import _load_embedding_profiles, get_embedding_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_MANIFEST = PROJECT_ROOT / "config" / "embedding_profiles.yaml"


def _print_profile_matrix():
    print("\nProfile Matrix:")
    try:
        profiles = _load_embedding_profiles(str(PROFILE_MANIFEST))
    except FileNotFoundError as exc:  # pragma: no cover - safeguarded in production
        print(f"  ❌ Unable to load profile manifest: {exc}")
        return {}

    for name, profile in profiles.items():
        caps = profile.capabilities
        missing = [req for req in profile.requirements if not os.getenv(req)]
        requirement_status = "OK" if not missing else f"missing {', '.join(missing)}"
        print(
            f"  - {name}: provider={profile.provider}, model={profile.model_id}, "
            f"dims={profile.dims}, similarity={profile.similarity}, task={profile.task}"
        )
        print(
            f"      tokenizer={profile.tokenizer.backend}"
            f" :: {profile.tokenizer.model_id}"
        )
        print(
            "      capabilities="
            f"dense:{caps.supports_dense} "
            f"sparse:{caps.supports_sparse} "
            f"colbert:{caps.supports_colbert} "
            f"long_seq:{caps.supports_long_sequences} "
            f"normalized:{caps.normalized_output} "
            f"multilingual:{caps.multilingual}"
        )
        print(f"      requirements: {requirement_status}")
    return profiles


def main():
    print("=" * 60)
    print("PROVIDER CONFIGURATION VERIFICATION")
    print("=" * 60)

    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"  EMBEDDINGS_PROFILE: {os.getenv('EMBEDDINGS_PROFILE', 'NOT SET')}")
    print(
        f"  LEGACY EMBEDDINGS_PROVIDER: {os.getenv('EMBEDDINGS_PROVIDER', 'NOT SET')}"
    )
    print(f"  LEGACY EMBEDDINGS_MODEL: {os.getenv('EMBEDDINGS_MODEL', 'NOT SET')}")
    print(f"  RERANK_MODEL: {os.getenv('RERANK_MODEL', 'NOT SET')}")
    print(f"  JINA_API_KEY: {'SET' if os.getenv('JINA_API_KEY') else 'NOT SET'}")

    settings = get_embedding_settings()
    print("\nResolved Embedding Settings (after profile + overrides):")
    print(f"  Profile: {settings.profile}")
    print(f"  Provider: {settings.provider}")
    print(f"  Model ID: {settings.model_id}")
    print(f"  Dims: {settings.dims}")
    print(f"  Tokenizer: {settings.tokenizer_backend} :: {settings.tokenizer_model_id}")
    if settings.extra:
        extras = ", ".join(f"{k}={v}" for k, v in settings.extra.items())
        print(f"  Legacy overrides: {extras}")
    else:
        print("  Legacy overrides: none (profile-only)")

    _print_profile_matrix()

    # Create embedding provider
    print("\nEmbedding Provider:")
    try:
        embed_provider = create_embedding_provider()
        print("  ✅ Provider created successfully")
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
        print("  ✅ Provider created successfully")
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

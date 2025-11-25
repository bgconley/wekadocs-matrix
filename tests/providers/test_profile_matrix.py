import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.providers.factory import ProviderFactory
from src.shared import config as config_module
from src.shared.config import get_embedding_settings

PROFILE_CASES = [  # pragma: allowlist secret
    ("jina_v3", "jina-ai", 1024),
    ("bge_m3", "bge-m3-service", 1024),
    ("st_minilm", "sentence-transformers", 384),
]


@pytest.mark.parametrize("profile,provider,dims", PROFILE_CASES)
def test_get_embedding_settings_resolves_profiles(monkeypatch, profile, provider, dims):
    env = {
        "NEO4J_PASSWORD": "placeholder",  # pragma: allowlist secret
        "EMBEDDINGS_PROFILE": profile,  # pragma: allowlist secret
    }
    if profile == "bge_m3":
        env.update(
            {
                "BGE_M3_API_URL": "http://127.0.0.1:9000",  # pragma: allowlist secret
                "BGE_M3_CLIENT_PATH": "/tmp",  # pragma: allowlist secret
            }
        )
    with patch.dict(os.environ, env, clear=True):
        config_module._config = None
        config_module._settings = None
        settings = get_embedding_settings()
        assert settings.profile == profile
        assert settings.provider == provider
        assert settings.dims == dims

        stub = SimpleNamespace(
            provider_name=provider,
            model_id=settings.model_id,
            dims=dims,
        )
        creators_backup = ProviderFactory._EMBEDDING_PROVIDER_CREATORS.copy()
        try:
            ProviderFactory._EMBEDDING_PROVIDER_CREATORS[provider] = (
                lambda s: stub  # type: ignore[assignment]
            )
            provider_instance = ProviderFactory.create_embedding_provider(
                settings=settings
            )
            assert provider_instance is stub
        finally:
            ProviderFactory._EMBEDDING_PROVIDER_CREATORS = creators_backup

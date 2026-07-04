import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.providers.factory import ProviderFactory
from src.shared import config as config_module
from src.shared.config import get_embedding_settings

PROFILE_REQUIREMENTS = {
    "jina_v3": ["JINA_API_KEY"],
    "bge_m3": ["BGE_M3_API_URL", "BGE_M3_CLIENT_PATH"],
}


def _should_run():
    return os.getenv("RUN_PROFILE_MATRIX_INTEGRATION") == "1"


def _missing_env(vars_):
    return [var for var in vars_ if not os.getenv(var)]


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.parametrize(
    "profile,provider_hint",
    [
        ("jina_v3", "jina-ai"),
        ("bge_m3", "bge-m3-service"),
    ],
)
def test_real_profile_embeddings(profile, provider_hint):
    if not _should_run():
        pytest.skip(
            "RUN_PROFILE_MATRIX_INTEGRATION=1 is required to hit real providers"
        )

    requirements = PROFILE_REQUIREMENTS.get(profile, [])
    missing = _missing_env(requirements)
    if missing:
        pytest.skip(f"Profile {profile} missing env vars: {missing}")

    patch_env = {"EMBEDDINGS_PROFILE": profile}  # pragma: allowlist secret
    if "NEO4J_PASSWORD" not in os.environ:
        patch_env["NEO4J_PASSWORD"] = "placeholder"  # pragma: allowlist secret
    temp_manifest = None
    if "EMBEDDING_PROFILES_PATH" not in os.environ:
        manifest_path = (
            Path(__file__).resolve().parents[2] / "config" / "embedding_profiles.yaml"
        )
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        data.pop("plan", None)
        temp_manifest = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        temp_manifest.write(yaml.safe_dump(data).encode("utf-8"))
        temp_manifest.flush()
        patch_env["EMBEDDING_PROFILES_PATH"] = temp_manifest.name

    try:
        with patch.dict(os.environ, patch_env, clear=False):
            config_module._config = None
            config_module._settings = None
            settings = get_embedding_settings()
            assert settings.profile == profile
            assert settings.provider == provider_hint

            embedder = ProviderFactory.create_embedding_provider(settings=settings)

            vectors = embedder.embed_documents(["real provider ping"])
            assert len(vectors) == 1
            assert len(vectors[0]) == settings.dims
    finally:
        if temp_manifest:
            try:
                os.unlink(temp_manifest.name)
            except OSError:
                pass

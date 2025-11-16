from pathlib import Path

import pytest
import yaml

from src.shared.config import (
    Config,
    Settings,
    _load_embedding_profiles,
    apply_embedding_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_CONFIG_PATH = PROJECT_ROOT / "config" / "development.yaml"
MANIFEST_PATH = PROJECT_ROOT / "config" / "embedding_profiles.yaml"


def _load_dev_config() -> Config:
    with DEV_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return Config(**yaml.safe_load(fh))


def test_manifest_contains_expected_profiles():
    manifest = _load_embedding_profiles(str(MANIFEST_PATH))
    assert "jina_v3" in manifest
    assert manifest["jina_v3"].dims == 1024
    assert "bge_m3" in manifest


def test_apply_embedding_profile_defaults_when_missing_profile(monkeypatch):
    config = _load_dev_config()
    config.embedding.profile = None
    settings = Settings(NEO4J_PASSWORD="placeholder")

    apply_embedding_profile(config, settings, DEV_CONFIG_PATH)

    assert config.embedding.profile == "jina_v3"
    assert config.embedding.embedding_model == "jina-embeddings-v3"
    assert config.embedding.provider == "jina-ai"


def test_apply_embedding_profile_honors_env_override(monkeypatch):
    config = _load_dev_config()
    settings = Settings(NEO4J_PASSWORD="placeholder", EMBEDDINGS_PROFILE="bge_m3")

    apply_embedding_profile(config, settings, DEV_CONFIG_PATH)

    assert config.embedding.profile == "bge_m3"
    assert config.embedding.embedding_model == "bge-m3"
    assert config.embedding.provider == "bge-m3-service"
    assert config.embedding.dims == 1024


def test_load_embedding_profiles_raises_on_invalid_manifest(tmp_path: Path):
    manifest_path = tmp_path / "bad_profiles.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "broken": {
                        "provider": "",
                        "model_id": "",
                        "dims": 0,
                        "tokenizer": {"backend": "", "model_id": ""},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        _load_embedding_profiles(str(manifest_path))

    assert "broken" in str(exc.value)
    assert "invalid" in str(exc.value)

import pytest

from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.shared.config import get_expected_namespace_suffix


def _settings(profile=None, version=None, model_id=None):
    caps = EmbeddingCapabilities()
    return EmbeddingSettings(
        profile=profile,
        provider="test-provider",
        model_id=model_id or "test-model",
        version=version,
        dims=128,
        similarity="cosine",
        task="retrieval.passage",
        tokenizer_backend="hf",
        tokenizer_model_id=None,
        service_url=None,
        capabilities=caps,
        extra={},
    )


@pytest.mark.parametrize(
    "mode, profile, version, model, expected",
    [
        ("profile", "bge_m3", "BAAI/bge-m3", None, "bge_m3"),
        ("version", "bge_m3", "BAAI/bge-m3", None, "baai_bge_m3"),
        ("model", "bge_m3", "BAAI/bge-m3", "Custom/Model", "custom_model"),
        ("none", "bge_m3", "BAAI/bge-m3", None, ""),
        ("", "bge_m3", "BAAI/bge-m3", None, ""),
        ("disabled", "bge_m3", "BAAI/bge-m3", None, ""),
    ],
)
def test_expected_namespace_suffix_explicit(mode, profile, version, model, expected):
    settings = _settings(profile=profile, version=version, model_id=model)
    assert get_expected_namespace_suffix(settings, mode) == expected


def test_expected_namespace_suffix_conflict_profile_wins_fallback():
    settings = _settings(profile="profileA", version="versionB", model_id="modelC")
    # With mode=profile missing or unknown, fallback prefers profile
    assert get_expected_namespace_suffix(settings, "unknown") == "profilea"


def test_expected_namespace_suffix_fallback_version_when_no_profile():
    settings = _settings(profile=None, version="versionB", model_id="modelC")
    assert get_expected_namespace_suffix(settings, "unknown") == "versionb"


def test_expected_namespace_suffix_fallback_model_when_no_profile_or_version():
    settings = _settings(profile=None, version=None, model_id="ModelC")
    assert get_expected_namespace_suffix(settings, "unknown") == "modelc"

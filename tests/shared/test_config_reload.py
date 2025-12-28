from src.shared.config import get_config, reload_config


def test_reload_config_respects_env_overrides(monkeypatch):
    # Baseline: use namespaced BGE-M3
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("EMBEDDINGS_PROFILE", "bge_m3")
    monkeypatch.setenv("BGE_M3_API_URL", "http://127.0.0.1:9000")
    reload_config()
    assert get_config().embedding.profile == "bge_m3"

    # Plan overrides EMBEDDINGS_PROFILE when present
    monkeypatch.setenv("EMBEDDINGS_PROFILE", "jina_v3")
    reload_config()
    assert get_config().embedding.profile == "bge_m3"

    # Clean up env for downstream tests
    monkeypatch.delenv("EMBEDDINGS_PROFILE", raising=False)
    monkeypatch.delenv("BGE_M3_API_URL", raising=False)

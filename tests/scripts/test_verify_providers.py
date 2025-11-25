import os
from types import SimpleNamespace
from unittest.mock import patch

from scripts import verify_providers


def test_verify_providers_lists_profiles(monkeypatch, capsys):
    env = {
        "NEO4J_PASSWORD": "placeholder",  # pragma: allowlist secret
        "EMBEDDINGS_PROFILE": "jina_v3",  # pragma: allowlist secret
        "JINA_API_KEY": "test-key",  # pragma: allowlist secret
        "BGE_M3_API_URL": "http://127.0.0.1:9000",  # pragma: allowlist secret
        "BGE_M3_CLIENT_PATH": "/tmp",  # pragma: allowlist secret
    }
    with patch.dict(os.environ, env, clear=True):
        fake_embed = SimpleNamespace(
            provider_name="jina-ai",
            model_id="jina-embeddings-v3",
            dims=1024,
        )
        fake_rerank = SimpleNamespace(
            _provider_name="noop",
            _model="noop",
        )
        with patch(
            "scripts.verify_providers.create_embedding_provider",
            return_value=fake_embed,
        ), patch(
            "scripts.verify_providers.create_rerank_provider", return_value=fake_rerank
        ):
            rc = verify_providers.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Profile Matrix:" in out
    assert "jina_v3" in out and "bge_m3" in out and "st_minilm" in out
    assert "requirements: OK" in out or "missing" in out

import types

import pytest

from src.ingestion.build_graph import GraphBuilder
from src.shared.config import reload_config


def _make_stub_embedder():
    stub = types.SimpleNamespace()
    stub.provider_name = "stub-provider"
    stub.task = "symmetric"
    return stub


def test_qdrant_namespace_enforced_on_upsert(monkeypatch):
    from src.shared.config import Settings

    monkeypatch.setenv("EMBEDDING_NAMESPACE_MODE", "profile")
    # Ensure get_settings returns a namespaced mode
    monkeypatch.setattr(
        "src.ingestion.build_graph.get_settings",
        lambda: Settings(
            EMBEDDING_NAMESPACE_MODE="profile",
            NEO4J_PASSWORD="placeholder",  # pragma: allowlist secret
        ),
        raising=False,
    )

    config, _ = reload_config()
    # Force a mismatched collection name vs. namespaced expected suffix
    monkeypatch.setattr(
        config.search.vector.qdrant, "collection_name", "chunks_mismatch", raising=False
    )

    gb = GraphBuilder(driver=None, config=config, qdrant_client=None)
    gb.embedder = _make_stub_embedder()

    with pytest.raises(RuntimeError):
        gb._upsert_to_qdrant(
            node_id="n1",
            vectors={"content": [0.1, 0.2]},
            section={"id": "s1", "document_id": "d1", "doc_id": "d1"},
            document={"id": "d1"},
            label="Chunk",
        )

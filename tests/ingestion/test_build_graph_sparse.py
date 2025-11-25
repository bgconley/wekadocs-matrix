from __future__ import annotations

from types import SimpleNamespace

from src.ingestion.build_graph import GraphBuilder


class _DummyClient:
    def __init__(self) -> None:
        self.last_points = None
        self.last_expected_dim = None

    def upsert_validated(self, collection_name, points, expected_dim, wait=True):
        self.collection_name = collection_name
        self.last_points = points
        self.last_expected_dim = expected_dim


def test_upsert_to_qdrant_attaches_sparse_vectors(monkeypatch):
    builder = GraphBuilder.__new__(GraphBuilder)
    builder.qdrant_client = _DummyClient()
    builder.config = SimpleNamespace(
        search=SimpleNamespace(
            vector=SimpleNamespace(
                qdrant=SimpleNamespace(
                    collection_name="chunks_multi",
                    enable_sparse=True,
                    enable_colbert=True,
                )
            )
        )
    )
    builder.embedding_settings = SimpleNamespace(
        version="bge-m3",
        profile="bge_m3",
        provider="bge-m3-service",
        task="symmetric",
    )
    builder.embedder = SimpleNamespace(provider_name="bge-m3-service")
    builder.embedding_settings.profile = "bge_m3"

    section = {
        "id": "section-1",
        "text": "sample text",
        "title": "Heading",
    }
    document = {
        "id": "doc-1",
        "source_uri": "file://doc",
    }

    builder._extract_semantic_metadata = lambda section: {"keywords": []}

    builder._upsert_to_qdrant(
        node_id="section-1",
        vectors={"content": [0.1, 0.2], "title": [0.3, 0.4]},
        section=section,
        document=document,
        label="Section",
        sparse_vector={"indices": [0, 1], "values": [0.5, 0.25]},
        colbert_vectors=[[0.1, 0.2], [0.3, 0.4]],
    )

    point = builder.qdrant_client.last_points[0]
    vector = point.vector
    assert "text-sparse" in vector
    assert vector["text-sparse"].indices == [0, 1]
    assert "late-interaction" in vector
    assert len(vector["late-interaction"]) == 2
    assert point.payload["colbert_vector_count"] == 2

from __future__ import annotations

from types import SimpleNamespace

from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.query.hybrid_retrieval import QdrantMultiVectorRetriever


class _FakeEmbedder:
    def __init__(self):
        self.dims = 4
        self.provider_name = "fake"
        self.model_id = "fake-model"

    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_sparse(self, texts):
        return [{"indices": [0, 1], "values": [0.5, 0.25]} for _ in texts]


class _FakeClient:
    def __init__(self):
        self.requests = []

    def search(self, **kwargs):
        self.requests.append(kwargs)
        hit = SimpleNamespace(id="dense-hit", score=0.42, payload={"id": "dense-hit"})
        if kwargs.get("query_sparse_vector"):
            hit = SimpleNamespace(
                id="lexical-hit",
                score=0.38,
                payload={"id": "lexical-hit"},
            )
        return [hit]

    def query_points(self, **kwargs):
        self.requests.append(kwargs)
        point = SimpleNamespace(
            id="qp-hit",
            score=0.73,
            payload={"id": "qp-hit", "document_id": "doc", "parent_section_id": ""},
        )
        return SimpleNamespace(points=[point])


class _FakeBundleEmbedder(_FakeEmbedder):
    def embed_query_all(self, text: str):
        return QueryEmbeddingBundle(dense=[0.1, 0.2, 0.3, 0.4])


def test_build_query_vectors_includes_sparse_leg():
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=_FakeClient(),
        embedder=_FakeEmbedder(),
        field_weights={"content": 1.0, "lexical": 0.5},
    )
    vectors = retriever._build_query_vectors("hello world")
    kinds = {name: kind for name, kind, _ in vectors}
    assert kinds["content"] == "dense"
    assert kinds["lexical"] == "sparse"


def test_search_dispatches_sparse_branch(monkeypatch):
    client = _FakeClient()
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=client,
        embedder=_FakeEmbedder(),
        field_weights={"content": 1.0, "lexical": 0.5},
    )

    def fake_dense(vector_name, vector, top_k, qdrant_filter, ef):
        return [
            SimpleNamespace(
                id="dense-id",
                score=0.9,
                payload={
                    "id": "dense-id",
                    "document_id": "doc",
                    "parent_section_id": "",
                },
            )
        ]

    def fake_sparse(vector, top_k, qdrant_filter):
        return [
            SimpleNamespace(
                id="lex-id",
                score=0.8,
                payload={"id": "lex-id", "document_id": "doc", "parent_section_id": ""},
            )
        ]

    monkeypatch.setattr(retriever, "_search_single", fake_dense)
    monkeypatch.setattr(retriever, "_search_sparse", fake_sparse)

    results = retriever.search("hello world", top_k=2)
    chunk_ids = {chunk.chunk_id for chunk in results}
    assert "lex-id" in chunk_ids
    assert "dense-id" in chunk_ids
    assert retriever.last_stats.get("path") == "legacy"


def test_qdrant_filter_includes_embedding_version():
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=_FakeClient(),
        embedder=_FakeEmbedder(),
        field_weights={"content": 1.0},
        embedding_settings=_fake_settings(version="bge-m3"),
    )

    qdrant_filter = retriever._build_filter({})
    assert qdrant_filter is not None
    assert any(
        cond.key == "embedding_version" and cond.match.value == "bge-m3"
        for cond in qdrant_filter.must
    )


def test_qdrant_filter_appends_user_filters():
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=_FakeClient(),
        embedder=_FakeEmbedder(),
        field_weights={"content": 1.0},
        embedding_settings=_fake_settings(),
    )
    qdrant_filter = retriever._build_filter({"tenant": "acme"})
    keys = [cond.key for cond in qdrant_filter.must]
    assert "embedding_version" in keys
    assert "tenant" in keys


def test_query_api_path_returns_results():
    client = _FakeClient()
    embedder = _FakeBundleEmbedder()
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=client,
        embedder=embedder,
        field_weights={"content": 1.0},
        use_query_api=True,
    )
    results = retriever.search("hello query", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "qp-hit"
    assert any(
        isinstance(request, dict) and "query" in request for request in client.requests
    )
    assert retriever.last_stats.get("path") == "query_api"


def _fake_settings(version: str = "jina-embeddings-v3") -> EmbeddingSettings:
    return EmbeddingSettings(
        profile="jina_v3",
        provider="jina-ai",
        model_id="jina-embeddings-v3",
        version=version,
        dims=1024,
        similarity="cosine",
        task="retrieval.passage",
        tokenizer_backend="hf",
        tokenizer_model_id="jinaai/jina-embeddings-v3",
        service_url=None,
        capabilities=EmbeddingCapabilities(),
    )

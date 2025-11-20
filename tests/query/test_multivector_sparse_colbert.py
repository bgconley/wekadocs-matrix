from types import SimpleNamespace

from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.query.hybrid_retrieval import QdrantMultiVectorRetriever


class StubSparse:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


def _stub_embedder():
    def embed_sparse(*args, **kwargs):
        return None

    def embed_colbert(*args, **kwargs):
        return None

    return SimpleNamespace(
        embed_sparse=embed_sparse,
        embed_colbert=embed_colbert,
        dims=3,
        provider_name="stub",
        task="symmetric",
    )


def test_build_prefetch_includes_sparse_when_supported():
    # Monkeypatch Prefetch to bypass pydantic validation for test isolation
    import src.query.hybrid_retrieval as hr

    hr.Prefetch = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore

    caps = EmbeddingCapabilities(
        supports_dense=True, supports_sparse=True, supports_colbert=False
    )
    settings = EmbeddingSettings(
        profile="stub",
        provider="stub",
        model_id="stub-model",
        version="stub-model",
        dims=3,
        similarity="cosine",
        task="symmetric",
        tokenizer_backend="hf",
        tokenizer_model_id="stub-tokenizer",
        service_url="http://127.0.0.1:9000",
        capabilities=caps,
    )
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=None,
        embedder=_stub_embedder(),
        collection_name="stub",
        embedding_settings=settings,
        schema_supports_sparse=True,
    )
    bundle = QueryEmbeddingBundle(
        dense=[0.1, 0.2, 0.3],
        sparse=StubSparse(indices=[0, 2], values=[0.5, 0.7]),
        multivector=None,
    )
    prefetch = retriever._build_prefetch_entries(
        bundle, None, retriever.query_api_dense_limit
    )
    names = [getattr(p.query, "name", None) for p in prefetch]
    assert "text-sparse" in names


def test_build_query_api_query_prefers_colbert_when_supported():
    caps = EmbeddingCapabilities(
        supports_dense=True, supports_sparse=False, supports_colbert=True
    )
    settings = EmbeddingSettings(
        profile="stub",
        provider="stub",
        model_id="stub-model",
        version="stub-model",
        dims=3,
        similarity="cosine",
        task="symmetric",
        tokenizer_backend="hf",
        tokenizer_model_id="stub-tokenizer",
        service_url="http://127.0.0.1:9000",
        capabilities=caps,
    )
    retriever = QdrantMultiVectorRetriever(
        qdrant_client=None,
        embedder=_stub_embedder(),
        collection_name="stub",
        embedding_settings=settings,
        schema_supports_colbert=True,
    )
    bundle = QueryEmbeddingBundle(
        dense=[0.1, 0.2, 0.3],
        sparse=None,
        multivector=SimpleNamespace(vectors=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
    )
    query, name = retriever._build_query_api_query(bundle)
    assert name == "late-interaction"
    assert isinstance(query, list)

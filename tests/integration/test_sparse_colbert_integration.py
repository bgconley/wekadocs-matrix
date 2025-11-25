import os
import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorConfig,
)
from qdrant_client.http.models import SparseVector as HttpSparseVector
from qdrant_client.http.models import (
    VectorParams,
)

from src.providers.embeddings.contracts import QueryEmbeddingBundle
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.query.hybrid_retrieval import QdrantMultiVectorRetriever


def _should_run():
    return os.getenv("RUN_SPARSE_PROFILE_MATRIX") == "1"


def _stub_embedder():
    def embed_sparse(*args, **kwargs):
        return None

    def embed_colbert(*args, **kwargs):
        return None

    def embed_query(*args, **kwargs):
        return QueryEmbeddingBundle(
            dense=[0.1, 0.2, 0.3],
            sparse=SimpleNamespace(indices=[0, 2], values=[0.5, 0.7]),
            multivector=SimpleNamespace(vectors=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
        )

    return SimpleNamespace(
        embed_sparse=embed_sparse,
        embed_colbert=embed_colbert,
        embed_query=embed_query,
        dims=3,
        provider_name="stub",
        task="symmetric",
    )


@pytest.mark.integration
def test_sparse_colbert_retrieval_smoke(qdrant_client: QdrantClient):
    if not _should_run():
        pytest.skip("RUN_SPARSE_PROFILE_MATRIX=1 required")
    # Ensure qdrant is reachable
    try:
        qdrant_client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant not reachable: {exc}")

    collection_name = f"sparse_colbert_{uuid.uuid4().hex[:8]}"
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "content": VectorParams(size=3, distance=Distance.COSINE),
            "title": VectorParams(size=3, distance=Distance.COSINE),
            "late-interaction": VectorParams(
                size=3,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(),
                hnsw_config=HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={"text-sparse": HttpSparseVector(indices=[], values=[])},
    )

    point_id = uuid.uuid4().hex
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": point_id,
                "vector": {
                    "content": [0.9, 0.1, 0.2],
                    "title": [0.8, 0.2, 0.2],
                    "late-interaction": [[0.9, 0.1, 0.2], [0.2, 0.1, 0.9]],
                },
                "sparse_vectors": {
                    "text-sparse": HttpSparseVector(indices=[0, 2], values=[0.6, 0.4])
                },
                "payload": {
                    "document_id": "doc1",
                    "embedding_version": "stub-model",
                    "tenant": "default",
                },
            }
        ],
        wait=True,
    )

    caps = EmbeddingCapabilities(
        supports_dense=True, supports_sparse=True, supports_colbert=True
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
        qdrant_client=qdrant_client,
        embedder=_stub_embedder(),
        collection_name=collection_name,
        embedding_settings=settings,
        schema_supports_sparse=True,
        schema_supports_colbert=True,
        use_query_api=False,
        field_weights={"content": 1.0, "text-sparse": 0.5},
    )

    results = retriever.search("network connectivity", top_k=3)
    assert len(results) > 0
    assert any(getattr(r, "vector_score", 0) > 0 for r in results)

    # Cleanup
    qdrant_client.delete_collection(collection_name=collection_name)

import pytest
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.shared.qdrant_schema import validate_qdrant_schema


def _test_settings(dims: int = 3) -> EmbeddingSettings:
    return EmbeddingSettings(
        profile="test_profile",
        provider="test-provider",
        model_id="test-model",
        version="v1",
        dims=dims,
        similarity="cosine",
        task="symmetric",
        tokenizer_backend="hf",
        tokenizer_model_id="test-model",
        service_url="http://127.0.0.1:9000",
        capabilities=EmbeddingCapabilities(
            supports_dense=True,
            supports_sparse=True,
            supports_colbert=False,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
    )


@pytest.mark.integration
def test_validate_qdrant_schema_strict_passes(qdrant_client):
    collection_name = "test_validate_qdrant_schema_strict"
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "content": VectorParams(size=3, distance=Distance.COSINE),
            "title": VectorParams(size=3, distance=Distance.COSINE),
            "entity": VectorParams(size=3, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
        },
    )
    for field, schema_type in [
        ("embedding_version", PayloadSchemaType.KEYWORD),
        ("embedding_provider", PayloadSchemaType.KEYWORD),
        ("embedding_dimensions", PayloadSchemaType.INTEGER),
        ("tenant", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),
    ]:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema_type,
        )

    settings = _test_settings()
    validate_qdrant_schema(
        qdrant_client,
        collection_name,
        settings,
        require_sparse=True,
        include_entity=True,
        require_payload_fields=[
            "embedding_version",
            "embedding_provider",
            "embedding_dimensions",
            "tenant",
            "document_id",
        ],
        strict=True,
    )


@pytest.mark.integration
def test_validate_qdrant_schema_detects_wrong_dims(qdrant_client):
    collection_name = "test_validate_qdrant_schema_wrong_dims"
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "content": VectorParams(size=4, distance=Distance.COSINE),
            "title": VectorParams(size=4, distance=Distance.COSINE),
        },
    )

    settings = _test_settings(dims=3)
    with pytest.raises(RuntimeError):
        validate_qdrant_schema(
            qdrant_client,
            collection_name,
            settings,
            require_sparse=False,
            include_entity=False,
            strict=True,
        )

from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.shared.config import (
    EmbeddingPlan,
    EmbeddingProfileCapabilities,
    EmbeddingProfileDefinition,
    EmbeddingProfileTokenizer,
    EmbeddingRolePlan,
)
from src.shared.qdrant_schema import build_qdrant_schema


def _profile(
    name: str, *, dims: int, supports_colbert: bool
) -> EmbeddingProfileDefinition:
    return EmbeddingProfileDefinition(
        description=f"{name} profile",
        provider="test-provider",
        model_id=name,
        version=name,
        dims=dims,
        similarity="cosine",
        task="retrieval.passage",
        tokenizer=EmbeddingProfileTokenizer(backend="hf", model_id="test-tokenizer"),
        capabilities=EmbeddingProfileCapabilities(
            supports_dense=True,
            supports_sparse=False,
            supports_colbert=supports_colbert,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
    )


def test_build_qdrant_schema_uses_colbert_dims_from_plan():
    dense_profile = _profile("dense-model", dims=1024, supports_colbert=False)
    colbert_profile = _profile("colbert-model", dims=768, supports_colbert=True)
    plan = EmbeddingPlan(
        dense=EmbeddingRolePlan(
            role="dense", profile_name="dense-model", profile=dense_profile
        ),
        colbert=EmbeddingRolePlan(
            role="colbert", profile_name="colbert-model", profile=colbert_profile
        ),
        sparse=None,
    )
    settings = EmbeddingSettings(
        profile="dense-model",
        provider="test-provider",
        model_id="dense-model",
        version="v1",
        dims=1024,
        similarity="cosine",
        task="retrieval.passage",
        tokenizer_backend="hf",
        tokenizer_model_id="test-tokenizer",
        service_url=None,
        capabilities=EmbeddingCapabilities(
            supports_dense=True,
            supports_sparse=False,
            supports_colbert=True,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
        extra={},
    )

    schema = build_qdrant_schema(
        settings,
        embedding_plan=plan,
        enable_colbert=True,
        enable_sparse=False,
    )
    assert schema.vectors_config["late-interaction"].size == 768

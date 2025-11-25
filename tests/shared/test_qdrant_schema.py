from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.shared.qdrant_schema import build_qdrant_schema


def _settings(
    profile: str, supports_sparse: bool, supports_colbert: bool, dims: int = 1024
):
    return EmbeddingSettings(
        profile=profile,
        provider="test-provider",
        model_id="test-model",
        version="test",
        dims=dims,
        similarity="cosine",
        task="retrieval",
        tokenizer_backend="hf",
        tokenizer_model_id=None,
        service_url=None,
        capabilities=EmbeddingCapabilities(
            supports_dense=True,
            supports_sparse=supports_sparse,
            supports_colbert=supports_colbert,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
    )


def test_schema_plan_dense_only_profile():
    settings = _settings("jina_v3", supports_sparse=False, supports_colbert=False)
    plan = build_qdrant_schema(settings)
    assert set(plan.vectors_config.keys()) == {"content", "title"}
    assert plan.sparse_vectors_config == {}
    assert any("Sparse vector support disabled" in note for note in plan.notes)


def test_schema_plan_bge_profile_includes_sparse_and_colbert():
    settings = _settings("bge_m3", supports_sparse=True, supports_colbert=True)
    plan = build_qdrant_schema(settings, include_entity=True)
    assert {"content", "title", "entity", "late-interaction"}.issubset(
        plan.vectors_config.keys()
    )
    assert "text-sparse" in plan.sparse_vectors_config
    late_vector = plan.vectors_config["late-interaction"]
    assert late_vector.multivector_config is not None

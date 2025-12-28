from src.shared.config import (
    EmbeddingPlan,
    EmbeddingProfileCapabilities,
    EmbeddingProfileDefinition,
    EmbeddingProfileTokenizer,
    EmbeddingRolePlan,
    _embedding_plan_fingerprint,
)


def _profile(name: str, *, output_dtype: str) -> EmbeddingProfileDefinition:
    return EmbeddingProfileDefinition(
        description=f"{name} profile",
        provider="test-provider",
        model_id=name,
        version=name,
        dims=128,
        similarity="cosine",
        task="retrieval.passage",
        output_dimension=128,
        output_dtype=output_dtype,
        tokenizer=EmbeddingProfileTokenizer(backend="hf", model_id="test-tokenizer"),
        capabilities=EmbeddingProfileCapabilities(
            supports_dense=True,
            supports_sparse=False,
            supports_colbert=False,
            supports_long_sequences=True,
            normalized_output=True,
            multilingual=True,
        ),
    )


def test_embedding_plan_fingerprint_deterministic():
    profile = _profile("model-a", output_dtype="float")
    plan = EmbeddingPlan(
        dense=EmbeddingRolePlan(role="dense", profile_name="model-a", profile=profile),
        sparse=None,
        colbert=None,
    )
    fingerprint_a = _embedding_plan_fingerprint(plan)
    fingerprint_b = _embedding_plan_fingerprint(plan)
    assert fingerprint_a == fingerprint_b


def test_embedding_plan_fingerprint_changes_on_output_dtype():
    plan_float = EmbeddingPlan(
        dense=EmbeddingRolePlan(
            role="dense",
            profile_name="model-a",
            profile=_profile("model-a", output_dtype="float"),
        ),
        sparse=None,
        colbert=None,
    )
    plan_int8 = EmbeddingPlan(
        dense=EmbeddingRolePlan(
            role="dense",
            profile_name="model-a",
            profile=_profile("model-a", output_dtype="int8"),
        ),
        sparse=None,
        colbert=None,
    )
    assert _embedding_plan_fingerprint(plan_float) != _embedding_plan_fingerprint(
        plan_int8
    )

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
    # Dense vectors: content, title, doc_title (always present)
    assert set(plan.vectors_config.keys()) == {"content", "title", "doc_title"}
    assert plan.sparse_vectors_config == {}
    assert any("Sparse vector support disabled" in note for note in plan.notes)


def test_schema_plan_bge_profile_includes_sparse_and_colbert():
    settings = _settings("bge_m3", supports_sparse=True, supports_colbert=True)
    plan = build_qdrant_schema(settings, include_entity=True)
    # Note: include_entity is DEPRECATED - dense entity vector was removed
    # We now use entity-sparse for lexical entity matching instead
    assert {"content", "title", "doc_title", "late-interaction"}.issubset(
        plan.vectors_config.keys()
    )
    assert "text-sparse" in plan.sparse_vectors_config
    assert "entity-sparse" in plan.sparse_vectors_config  # Replaced dense entity
    late_vector = plan.vectors_config["late-interaction"]
    assert late_vector.multivector_config is not None


def test_schema_plan_includes_entity_metadata_indexes():
    """Phase 3: GLiNER entity metadata payload indexes are included in schema."""
    settings = _settings("bge_m3", supports_sparse=True, supports_colbert=True)
    plan = build_qdrant_schema(settings)

    # Extract field names from payload_indexes tuples
    index_field_names = [field for field, _ in plan.payload_indexes]

    # Verify all 4 entity metadata indexes are present
    expected_entity_indexes = [
        "entity_metadata.entity_types",
        "entity_metadata.entity_values",
        "entity_metadata.entity_values_normalized",
        "entity_metadata.entity_count",
    ]

    for field in expected_entity_indexes:
        assert field in index_field_names, f"Missing entity index: {field}"


def test_schema_plan_entity_metadata_index_types():
    """Verify entity metadata indexes have correct PayloadSchemaType."""
    from qdrant_client.models import PayloadSchemaType

    settings = _settings("bge_m3", supports_sparse=True, supports_colbert=True)
    plan = build_qdrant_schema(settings)

    # Convert to dict for easier lookup
    index_types = {field: schema for field, schema in plan.payload_indexes}

    # KEYWORD type for string arrays
    assert index_types.get("entity_metadata.entity_types") == PayloadSchemaType.KEYWORD
    assert index_types.get("entity_metadata.entity_values") == PayloadSchemaType.KEYWORD
    assert (
        index_types.get("entity_metadata.entity_values_normalized")
        == PayloadSchemaType.KEYWORD
    )

    # INTEGER type for count
    assert index_types.get("entity_metadata.entity_count") == PayloadSchemaType.INTEGER

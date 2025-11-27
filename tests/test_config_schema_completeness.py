"""
Test to prevent config schema drift.

This test ensures that all YAML config keys have corresponding Pydantic model fields.
The bug this prevents: Adding a YAML config value but forgetting to add the Pydantic
field, causing Pydantic v2 to silently drop the value.

Reference: Phase C fix - graph_channel_enabled was in YAML but not in HybridSearchConfig,
causing graph_as_reranker to never execute despite correct YAML configuration.
"""

from pathlib import Path
from typing import Any, Dict, Set

import pytest
import yaml

from src.shared.config import (
    BM25Config,
    EmbeddingConfig,
    ExpansionConfig,
    FeatureFlagsConfig,
    HybridSearchConfig,
    QdrantVectorConfig,
    RerankerConfig,
)


def get_pydantic_fields(model_class) -> Set[str]:
    """Get all field names defined in a Pydantic model."""
    return set(model_class.model_fields.keys())


def get_yaml_keys(yaml_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Set[str]]:
    """Recursively extract all keys from YAML dict, grouped by path."""
    result = {}
    if not isinstance(yaml_dict, dict):
        return result

    current_keys = set(yaml_dict.keys())
    if prefix:
        result[prefix] = current_keys
    else:
        result["root"] = current_keys

    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            nested_prefix = f"{prefix}.{key}" if prefix else key
            result.update(get_yaml_keys(value, nested_prefix))

    return result


# Map YAML paths to their Pydantic model classes
YAML_PATH_TO_MODEL = {
    "search.hybrid": HybridSearchConfig,
    "search.hybrid.reranker": RerankerConfig,
    "search.hybrid.bm25": BM25Config,
    "search.hybrid.expansion": ExpansionConfig,
    "search.vector.qdrant": QdrantVectorConfig,
    "search.bm25": BM25Config,
    "embedding": EmbeddingConfig,
    "feature_flags": FeatureFlagsConfig,
}

# Known exceptions - ONLY for nested config objects that have their own Pydantic models
# NOTE: Do NOT add exceptions for missing fields - fix them in the Pydantic model instead!
# This test exists precisely to prevent the silent field drop bug (Pydantic v2 extra="ignore").
KNOWN_EXCEPTIONS = {
    "search.hybrid": {
        # These are nested config objects with their own Pydantic models (RerankerConfig, BM25Config, ExpansionConfig)
        "reranker",
        "bm25",
        "expansion",
    },
    "search.hybrid.expansion": {
        # These are nested config objects with their own models (ExpansionRescoringConfig, ExpansionStructureConfig)
        "rescoring",
        "structure",
    },
    "search.vector": {"neo4j"},  # Optional nested config
    "search.vector.qdrant": set(),
    "embedding": {"tokenizer"},  # Nested config
    "feature_flags": set(),  # All flags must be in FeatureFlagsConfig - no exceptions!
}


@pytest.fixture
def development_config() -> Dict[str, Any]:
    """Load the development config YAML."""
    config_path = Path(__file__).parent.parent / "config" / "development.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_hybrid_search_config_completeness(development_config):
    """Ensure all YAML keys under search.hybrid are defined in HybridSearchConfig."""
    yaml_keys = set(development_config.get("search", {}).get("hybrid", {}).keys())
    model_fields = get_pydantic_fields(HybridSearchConfig)
    exceptions = KNOWN_EXCEPTIONS.get("search.hybrid", set())

    # Remove nested config objects from comparison
    yaml_keys_flat = yaml_keys - exceptions

    missing_in_model = yaml_keys_flat - model_fields

    assert not missing_in_model, (
        f"YAML keys in search.hybrid not defined in HybridSearchConfig: {missing_in_model}\n"
        f"Add these fields to HybridSearchConfig in src/shared/config.py to prevent silent config drops.\n"
        f"YAML keys: {sorted(yaml_keys_flat)}\n"
        f"Model fields: {sorted(model_fields)}"
    )


def test_feature_flags_config_completeness(development_config):
    """Ensure all YAML keys under feature_flags are defined in FeatureFlagsConfig."""
    yaml_keys = set(development_config.get("feature_flags", {}).keys())
    model_fields = get_pydantic_fields(FeatureFlagsConfig)
    exceptions = KNOWN_EXCEPTIONS.get("feature_flags", set())

    yaml_keys_flat = yaml_keys - exceptions
    missing_in_model = yaml_keys_flat - model_fields

    assert not missing_in_model, (
        f"YAML keys in feature_flags not defined in FeatureFlagsConfig: {missing_in_model}\n"
        f"Add these fields to FeatureFlagsConfig in src/shared/config.py to prevent silent config drops.\n"
        f"YAML keys: {sorted(yaml_keys_flat)}\n"
        f"Model fields: {sorted(model_fields)}"
    )


def test_qdrant_config_completeness(development_config):
    """Ensure all YAML keys under search.vector.qdrant are defined in QdrantVectorConfig."""
    yaml_keys = set(
        development_config.get("search", {}).get("vector", {}).get("qdrant", {}).keys()
    )
    model_fields = get_pydantic_fields(QdrantVectorConfig)
    exceptions = KNOWN_EXCEPTIONS.get("search.vector.qdrant", set())

    yaml_keys_flat = yaml_keys - exceptions
    missing_in_model = yaml_keys_flat - model_fields

    assert not missing_in_model, (
        f"YAML keys in search.vector.qdrant not defined in QdrantVectorConfig: {missing_in_model}\n"
        f"Add these fields to QdrantVectorConfig in src/shared/config.py to prevent silent config drops."
    )


def test_expansion_config_completeness(development_config):
    """Ensure all YAML keys under search.hybrid.expansion are defined in ExpansionConfig."""
    yaml_keys = set(
        development_config.get("search", {})
        .get("hybrid", {})
        .get("expansion", {})
        .keys()
    )
    model_fields = get_pydantic_fields(ExpansionConfig)
    exceptions = KNOWN_EXCEPTIONS.get("search.hybrid.expansion", set())

    yaml_keys_flat = yaml_keys - exceptions
    missing_in_model = yaml_keys_flat - model_fields

    assert not missing_in_model, (
        f"YAML keys in search.hybrid.expansion not defined in ExpansionConfig: {missing_in_model}\n"
        f"Add these fields to ExpansionConfig in src/shared/config.py to prevent silent config drops."
    )


def test_reranker_config_completeness(development_config):
    """Ensure all YAML keys under search.hybrid.reranker are defined in RerankerConfig."""
    yaml_keys = set(
        development_config.get("search", {})
        .get("hybrid", {})
        .get("reranker", {})
        .keys()
    )
    model_fields = get_pydantic_fields(RerankerConfig)

    missing_in_model = yaml_keys - model_fields

    assert not missing_in_model, (
        f"YAML keys in search.hybrid.reranker not defined in RerankerConfig: {missing_in_model}\n"
        f"Add these fields to RerankerConfig in src/shared/config.py to prevent silent config drops."
    )


def test_config_loads_without_warning():
    """Ensure config loads cleanly without Pydantic validation warnings.

    Note: Deprecation warnings (like .copy() -> .model_copy()) are filtered out
    as they are separate migration issues and don't affect config loading correctness.
    """
    import warnings

    from src.shared.config import load_config

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config, settings = load_config()

        # Filter for Pydantic validation warnings, excluding deprecation warnings
        pydantic_warnings = [
            warning
            for warning in w
            if (
                "pydantic" in str(warning.category).lower()
                or "validation" in str(warning.message).lower()
            )
            and "deprecated" not in str(warning.message).lower()
        ]

        assert (
            not pydantic_warnings
        ), f"Config loading produced Pydantic warnings: {[str(w.message) for w in pydantic_warnings]}"

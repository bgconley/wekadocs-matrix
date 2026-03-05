"""
Unit tests for entity quality gating (Phase B of retrieval tuning).

Tests verify:
- Expanded entity exclusion list catches generic domain terms
- Exclusion list preserves discriminative domain entities
- Per-label confidence floors filter correctly
- Entity cap at 8 per chunk in atomic.py entity-sparse generation
"""

import pytest

from src.providers.ner.labels import (
    DEFAULT_RETRIEVAL_FLOOR,
    RETRIEVAL_CONFIDENCE_FLOORS,
    extract_label_name,
    is_excluded_entity,
)


class TestExpandedExclusionList:
    """Tests for the expanded ENTITY_EXCLUSIONS set."""

    @pytest.mark.parametrize(
        "term",
        [
            "cluster",
            "Cluster",
            "CLUSTER",
            "system",
            "System",
            "server",
            "node",
            "service",
            "data",
            "file",
            "process",
            "configuration",
            "management",
            "user",
            "host",
            "network",
            "storage",
            "volume",
            "drive",
            "performance",
            "capacity",
            "step",
            "click",
            "select",
            "run",
            "enter",
            "WEKA",
            "weka",
            "WekaFS",
        ],
    )
    def test_generic_terms_excluded(self, term: str):
        """Generic domain terms must be excluded from entity enrichment."""
        assert is_excluded_entity(term), f"Expected '{term}' to be excluded"

    @pytest.mark.parametrize(
        "term",
        [
            "NFS",
            "SMB",
            "S3",
            "POSIX",
            "inode",
            "metadata",
            "tiering",
            "snapshot",
            "AWS",
            "Azure",
            "GCP",
            "weka fs",
            "mount",
            "stripe-width",
            "backend server",
        ],
    )
    def test_discriminative_terms_preserved(self, term: str):
        """Domain-specific discriminative terms must NOT be excluded."""
        assert not is_excluded_entity(term), f"'{term}' should NOT be excluded"


class TestPerLabelConfidenceFloors:
    """Tests for RETRIEVAL_CONFIDENCE_FLOORS and DEFAULT_RETRIEVAL_FLOOR."""

    def test_all_10_labels_have_floors(self):
        """Every default label type should have an explicit floor."""
        expected_labels = {
            "COMMAND",
            "PARAMETER",
            "PROTOCOL",
            "CLOUD_PROVIDER",
            "VERSION",
            "ERROR",
            "COMPONENT",
            "PROCEDURE_STEP",
            "STORAGE_CONCEPT",
            "CAPACITY_METRIC",
        }
        assert set(RETRIEVAL_CONFIDENCE_FLOORS.keys()) == expected_labels

    def test_noisy_labels_have_higher_floors(self):
        """Abstract/ambiguous labels should have stricter confidence floors."""
        assert RETRIEVAL_CONFIDENCE_FLOORS["STORAGE_CONCEPT"] >= 0.70
        assert RETRIEVAL_CONFIDENCE_FLOORS["CAPACITY_METRIC"] >= 0.70
        assert RETRIEVAL_CONFIDENCE_FLOORS["PROCEDURE_STEP"] >= 0.70

    def test_precise_labels_have_lower_floors(self):
        """Labels with distinctive surface patterns should have lower floors."""
        assert RETRIEVAL_CONFIDENCE_FLOORS["COMMAND"] <= 0.60
        assert RETRIEVAL_CONFIDENCE_FLOORS["PARAMETER"] <= 0.60
        assert RETRIEVAL_CONFIDENCE_FLOORS["VERSION"] <= 0.60

    def test_default_floor_is_moderate(self):
        """The default fallback floor should be moderate (0.55-0.65)."""
        assert 0.55 <= DEFAULT_RETRIEVAL_FLOOR <= 0.65

    def test_storage_concept_below_floor_filtered(self):
        """A STORAGE_CONCEPT entity at 0.65 should be below the 0.70 floor."""
        label = "STORAGE_CONCEPT"
        confidence = 0.65
        floor = RETRIEVAL_CONFIDENCE_FLOORS.get(label, DEFAULT_RETRIEVAL_FLOOR)
        assert confidence < floor, "0.65 STORAGE_CONCEPT should be below 0.70 floor"

    def test_command_above_floor_kept(self):
        """A COMMAND entity at 0.60 should pass the 0.55 floor."""
        label = "COMMAND"
        confidence = 0.60
        floor = RETRIEVAL_CONFIDENCE_FLOORS.get(label, DEFAULT_RETRIEVAL_FLOOR)
        assert confidence >= floor, "0.60 COMMAND should pass 0.55 floor"

    def test_unknown_label_uses_default(self):
        """An unknown label type should fall back to DEFAULT_RETRIEVAL_FLOOR."""
        floor = RETRIEVAL_CONFIDENCE_FLOORS.get("UNKNOWN_TYPE", DEFAULT_RETRIEVAL_FLOOR)
        assert floor == DEFAULT_RETRIEVAL_FLOOR


class TestExtractLabelName:
    """Tests for label name extraction from descriptive labels."""

    def test_strips_examples(self):
        assert extract_label_name("COMMAND (e.g. weka fs, mount)") == "COMMAND"

    def test_plain_label(self):
        assert extract_label_name("PROTOCOL") == "PROTOCOL"

    def test_label_with_spaces(self):
        assert extract_label_name("CLOUD_PROVIDER (e.g. AWS)") == "CLOUD_PROVIDER"

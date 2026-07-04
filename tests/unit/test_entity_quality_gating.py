"""
Unit tests for entity quality gating (Phase B of retrieval tuning).

Tests verify:
- Expanded entity exclusion list catches generic domain terms
- Exclusion list preserves discriminative domain entities
- Per-label confidence floors filter correctly
- Entity cap at 8 per chunk in atomic.py entity-sparse generation
"""

import pytest

from src.ingestion.extract.commands import extract_commands
from src.ingestion.extract.configs import extract_configurations
from src.ingestion.extract.procedures import extract_procedures
from src.providers.ner.labels import (
    DEFAULT_RETRIEVAL_FLOOR,
    RETRIEVAL_CONFIDENCE_FLOORS,
    extract_label_name,
    is_excluded_entity,
    is_excluded_structural_entity,
    normalize_entity_name,
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
            "Nutanix",
            "nutanix",
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
            "Prism Central",
            "NCI",
            "AOS",
            "AHV",
            "metadata",
            "tiering",
            "snapshot",
            "AWS",
            "Azure",
            "GCP",
            "ncli cluster",
            "mount",
            "stripe-width",
            "Nutanix Files",
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
        assert extract_label_name("COMMAND (e.g. ncli, acli)") == "COMMAND"

    def test_plain_label(self):
        assert extract_label_name("PROTOCOL") == "PROTOCOL"

    def test_label_with_spaces(self):
        assert extract_label_name("CLOUD_PROVIDER (e.g. AWS)") == "CLOUD_PROVIDER"


class TestStructuralEntityQualityGate:
    def test_normalize_strips_markdown_bold(self):
        assert normalize_entity_name("**Before you begin**") == "Before you begin"

    def test_normalize_strips_backticks(self):
        assert normalize_entity_name("`ncli cluster`") == "ncli cluster"

    def test_normalize_collapses_whitespace(self):
        assert normalize_entity_name("  foo   bar \n baz ") == "foo bar baz"

    def test_normalize_preserves_underscores_and_hyphens(self):
        assert normalize_entity_name("memory_mb") == "memory_mb"
        assert normalize_entity_name("filter-color") == "filter-color"

    @pytest.mark.parametrize(
        "term",
        [
            "color",
            "profile",
            "output",
            "format",
            "filter",
            "sort",
            "**Procedure**",
            "`json`",
        ],
    )
    def test_structural_noise_terms_excluded(self, term: str):
        assert is_excluded_structural_entity(term), f"Expected '{term}' to be excluded"

    @pytest.mark.parametrize(
        "term",
        [
            "inode",
            "metadata",
            "S3",
            "s3",
            "NFS",
            "stripe-width",
        ],
    )
    def test_discriminative_terms_not_excluded(self, term: str):
        assert not is_excluded_structural_entity(
            term
        ), f"Expected '{term}' to NOT be excluded"


class TestStructuralExtractorQuality:
    def test_flag_exclusion_scoped_to_flag_pattern(self):
        section = {
            "id": "s1",
            "text": '--color auto\n`color`: "blue"\n',
            "code_blocks": ['{"color": "blue"}'],
        }
        configs, _ = extract_configurations(section)
        names = {c["name"] for c in configs}
        color_entities = [c for c in configs if c["name"] == "color"]

        # --color should be blocked by flag-specific gate
        assert "color" in names, "YAML/JSON key extraction should remain intact"
        assert color_entities
        assert all(c.get("category") != "flag" for c in color_entities)

    def test_structural_entities_have_type_and_source(self):
        cmd_section = {
            "id": "s2",
            "text": "`ncli cluster status`",
            "code_blocks": ["ncli cluster status"],
        }
        cmds, _ = extract_commands(cmd_section)
        assert cmds, "Expected at least one command entity"
        assert cmds[0]["entity_type"] == "COMMAND"
        assert cmds[0]["source"] == "structural"

        proc_section = {
            "id": "s3",
            "title": "How to configure",
            "text": "1. Run command\n2. Verify output",
            "code_blocks": [],
        }
        procedures, steps, _ = extract_procedures(proc_section)
        assert procedures, "Expected procedure extraction"
        assert steps, "Expected step extraction"
        assert procedures[0]["entity_type"] == "PROCEDURE"
        assert procedures[0]["source"] == "structural"
        assert steps[0]["entity_type"] == "STEP"
        assert steps[0]["source"] == "structural"

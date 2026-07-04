"""Tests for query intent classification."""

import pytest

from src.query.query_intent import (
    SIZING_ANCHORS,
    SIZING_MODIFIERS,
    SUBSYSTEM_ANCHORS,
    SUBSYSTEM_MODIFIERS,
    classify_query_intent,
)


class TestSubsystemArchitecture:
    @pytest.mark.parametrize(
        "query",
        [
            "how is metadata managed and architected on a nutanix cluster",
            "explain Nutanix inode management internals",
            "Nutanix filesystem internals and tiering",
            "how does Nutanix handle snapshots limitations",
            "describe the Nutanix data protection scheme and rebuild process",
            "Nutanix metadata architecture overview",
            "how does data placement work in Nutanix",
        ],
    )
    def test_subsystem_queries_classified_correctly(self, query):
        intent = classify_query_intent(query)
        assert intent.query_type == "subsystem_architecture"
        assert intent.precision_mode is True
        assert len(intent.subsystem_terms) >= 1

    def test_subsystem_with_cloud_cue_still_subsystem(self):
        """Cloud cues are orthogonal — should NOT disqualify subsystem_architecture."""
        intent = classify_query_intent("how does Nutanix metadata work on Azure")
        assert intent.query_type == "subsystem_architecture"
        assert intent.has_cloud_cues is True
        assert intent.precision_mode is True
        assert "metadata" in intent.subsystem_terms

    def test_subsystem_with_aws_still_subsystem(self):
        intent = classify_query_intent("Nutanix tiering architecture on AWS")
        assert intent.query_type == "subsystem_architecture"
        assert intent.has_cloud_cues is True


class TestResourceSizing:
    @pytest.mark.parametrize(
        "query",
        [
            "How do I appropriately size the nutanix drives, compute, and frontends containers?",
            "Nutanix frontend sizing requirements",
            "minimum CPU cores for Nutanix containers",
            "Nutanix capacity planning drives compute frontend",
            "how much memory do Nutanix containers need",
            "Nutanix ram requirements for compute containers",
        ],
    )
    def test_sizing_queries_classified_correctly(self, query):
        intent = classify_query_intent(query)
        assert intent.query_type == "resource_sizing"
        assert intent.precision_mode is True
        assert len(intent.sizing_terms) >= 1

    def test_sizing_with_cloud_cue_still_sizing(self):
        """Cloud cues are orthogonal — should NOT disqualify resource_sizing."""
        intent = classify_query_intent("Nutanix container sizing on GCP")
        assert intent.query_type == "resource_sizing"
        assert intent.has_cloud_cues is True
        assert intent.precision_mode is True


class TestExistingTypesStable:
    def test_cli_still_works(self):
        intent = classify_query_intent("nutanix cluster run command --force")
        assert intent.query_type == "cli"

    def test_config_still_works(self):
        intent = classify_query_intent("configure tiering policy setting")
        assert intent.query_type == "config"

    def test_procedural_still_works(self):
        intent = classify_query_intent("how to install Nutanix cluster")
        assert intent.query_type == "procedural"

    def test_troubleshooting_still_works(self):
        intent = classify_query_intent("Nutanix cluster error failed to start")
        assert intent.query_type == "troubleshooting"

    def test_reference_still_works(self):
        intent = classify_query_intent("what is Nutanix deduplication")
        assert intent.query_type == "reference"

    def test_default_conceptual(self):
        intent = classify_query_intent("Nutanix cluster overview benefits")
        assert intent.query_type == "conceptual"


class TestCloudCueDetection:
    @pytest.mark.parametrize(
        "cue",
        [
            "aws",
            "azure",
            "gcp",
            "slurm",
            "cyclecloud",
            "terraform",
            "cloudformation",
            "parallelcluster",
            "sagemaker",
        ],
    )
    def test_cloud_cues_detected(self, cue):
        intent = classify_query_intent(f"deploy Nutanix on {cue}")
        assert intent.has_cloud_cues is True


class TestEdgeCases:
    def test_empty_query_is_conceptual(self):
        intent = classify_query_intent("")
        assert intent.query_type == "conceptual"

    def test_none_query_is_conceptual(self):
        intent = classify_query_intent(None)
        assert intent.query_type == "conceptual"

    def test_mixed_subsystem_and_sizing_subsystem_wins(self):
        """When both terms present, subsystem wins (earlier in chain)."""
        intent = classify_query_intent("Nutanix metadata architecture sizing capacity")
        assert intent.query_type == "subsystem_architecture"
        assert len(intent.subsystem_terms) >= 1
        assert len(intent.sizing_terms) >= 1

    def test_config_overrides_subsystem(self):
        """Config patterns take priority over subsystem terms."""
        intent = classify_query_intent("configure metadata tiering.yaml")
        assert intent.query_type == "config"

    def test_cli_overrides_all(self):
        """CLI with enough signals overrides everything."""
        intent = classify_query_intent("nutanix metadata --tiering command")
        assert intent.query_type == "cli"


class TestQueryIntentImmutability:
    def test_frozen_dataclass(self):
        intent = classify_query_intent("test query")
        with pytest.raises(AttributeError):
            intent.query_type = "changed"


class TestRegressionQueries:
    """Exact regression tests for the two queries that motivated this work."""

    def test_metadata_architecture_query(self):
        intent = classify_query_intent(
            "how is metadata managed and architected on a nutanix cluster"
        )
        assert intent.query_type == "subsystem_architecture"
        assert intent.precision_mode is True
        assert "metadata" in intent.subsystem_terms
        assert (
            "architected" in intent.subsystem_terms
            or "architecture" in intent.subsystem_terms
        )

    def test_container_sizing_query(self):
        intent = classify_query_intent(
            "How do I appropriately size the nutanix drives, compute, and frontends containers?"
        )
        assert intent.query_type == "resource_sizing"
        assert intent.precision_mode is True
        assert "size" in intent.sizing_terms or "sizing" in intent.sizing_terms
        assert "drives" in intent.sizing_terms
        assert "compute" in intent.sizing_terms


class TestAnchorModifierSplit:
    """Phase A2: Anchor/modifier partitioning for precision intents."""

    def test_metadata_managed_split(self):
        intent = classify_query_intent("how is metadata managed")
        assert intent.primary_anchors == ("metadata",)
        assert intent.generic_modifiers == ("managed",)

    def test_drives_compute_frontend_sizing(self):
        intent = classify_query_intent("Nutanix drives compute frontend sizing")
        assert "drives" in intent.primary_anchors
        assert "compute" in intent.primary_anchors
        assert "frontend" in intent.primary_anchors
        assert "sizing" in intent.generic_modifiers

    def test_subsystem_anchors_are_subset_of_terms(self):
        intent = classify_query_intent(
            "how is metadata managed and architected on a nutanix cluster"
        )
        for a in intent.primary_anchors:
            assert a in intent.subsystem_terms
        for m in intent.generic_modifiers:
            assert m in intent.subsystem_terms

    def test_sizing_anchors_are_subset_of_terms(self):
        intent = classify_query_intent(
            "How do I appropriately size the nutanix drives, compute, and frontends containers?"
        )
        for a in intent.primary_anchors:
            assert a in intent.sizing_terms
        for m in intent.generic_modifiers:
            assert m in intent.sizing_terms

    def test_empty_anchors_for_modifiers_only(self):
        """A query with only modifier terms still classifies but has empty anchors."""
        intent = classify_query_intent("Nutanix architecture internals backend")
        assert intent.query_type == "subsystem_architecture"
        # "architecture", "internals", "backend" are all modifiers
        assert len(intent.generic_modifiers) >= 1
        # Some terms are modifiers, but primary_anchors may still be empty

    def test_non_precision_query_has_empty_anchors(self):
        intent = classify_query_intent("how to install Nutanix cluster")
        assert intent.primary_anchors == ()
        assert intent.generic_modifiers == ()

    def test_anchor_modifier_sets_disjoint(self):
        """Anchor and modifier sets should not overlap."""
        assert not (SUBSYSTEM_ANCHORS & SUBSYSTEM_MODIFIERS)
        assert not (SIZING_ANCHORS & SIZING_MODIFIERS)

    def test_tiering_is_anchor_not_modifier(self):
        intent = classify_query_intent("Nutanix tiering architecture")
        assert "tiering" in intent.primary_anchors
        assert "architecture" in intent.generic_modifiers

    def test_snapshots_limitations(self):
        intent = classify_query_intent("Nutanix snapshots limitations")
        assert "snapshots" in intent.primary_anchors
        assert "limitations" in intent.generic_modifiers

    def test_ram_cpu_cores_are_sizing_anchors(self):
        intent = classify_query_intent("Nutanix ram cpu cores capacity")
        assert "ram" in intent.primary_anchors
        assert "cpu" in intent.primary_anchors
        assert "cores" in intent.primary_anchors
        assert "capacity" in intent.generic_modifiers


class TestWordBoundaryMatching:
    """Regression: short terms like 'ram', 'cpu', 'size' must not match
    inside unrelated words like 'program', 'diagram', 'resize'."""

    @pytest.mark.parametrize(
        "query",
        [
            "program overview",
            "How to program Nutanix",
            "diagram of system",
            "dramatic changes to Nutanix",
            "resize the window",
        ],
    )
    def test_short_terms_no_false_positive(self, query):
        intent = classify_query_intent(query)
        assert (
            intent.query_type != "resource_sizing"
        ), f"{query!r} falsely classified as resource_sizing"
        assert len(intent.sizing_terms) == 0

    @pytest.mark.parametrize(
        "query,expected_term",
        [
            ("Nutanix ram requirements", "ram"),
            ("how much cpu for containers", "cpu"),
            ("appropriately size the nutanix drives", "size"),
        ],
    )
    def test_short_terms_true_positive(self, query, expected_term):
        intent = classify_query_intent(query)
        assert intent.query_type == "resource_sizing"
        assert expected_term in intent.sizing_terms

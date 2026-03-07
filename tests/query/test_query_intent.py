"""Tests for query intent classification."""

import pytest

from src.query.query_intent import classify_query_intent


class TestSubsystemArchitecture:
    @pytest.mark.parametrize(
        "query",
        [
            "how is metadata managed and architected on a weka cluster",
            "explain WEKA inode management internals",
            "WEKA filesystem internals and tiering",
            "how does WEKA handle snapshots limitations",
            "describe the WEKA data protection scheme and rebuild process",
            "WEKA metadata architecture overview",
            "how does data placement work in WEKA",
        ],
    )
    def test_subsystem_queries_classified_correctly(self, query):
        intent = classify_query_intent(query)
        assert intent.query_type == "subsystem_architecture"
        assert intent.precision_mode is True
        assert len(intent.subsystem_terms) >= 1

    def test_subsystem_with_cloud_cue_still_subsystem(self):
        """Cloud cues are orthogonal — should NOT disqualify subsystem_architecture."""
        intent = classify_query_intent("how does WEKA metadata work on Azure")
        assert intent.query_type == "subsystem_architecture"
        assert intent.has_cloud_cues is True
        assert intent.precision_mode is True
        assert "metadata" in intent.subsystem_terms

    def test_subsystem_with_aws_still_subsystem(self):
        intent = classify_query_intent("WEKA tiering architecture on AWS")
        assert intent.query_type == "subsystem_architecture"
        assert intent.has_cloud_cues is True


class TestResourceSizing:
    @pytest.mark.parametrize(
        "query",
        [
            "How do I appropriately size the weka drives, compute, and frontends containers?",
            "WEKA frontend sizing requirements",
            "minimum CPU cores for WEKA containers",
            "WEKA capacity planning drives compute frontend",
            "how much memory do WEKA containers need",
            "WEKA ram requirements for compute containers",
        ],
    )
    def test_sizing_queries_classified_correctly(self, query):
        intent = classify_query_intent(query)
        assert intent.query_type == "resource_sizing"
        assert intent.precision_mode is True
        assert len(intent.sizing_terms) >= 1

    def test_sizing_with_cloud_cue_still_sizing(self):
        """Cloud cues are orthogonal — should NOT disqualify resource_sizing."""
        intent = classify_query_intent("WEKA container sizing on GCP")
        assert intent.query_type == "resource_sizing"
        assert intent.has_cloud_cues is True
        assert intent.precision_mode is True


class TestExistingTypesStable:
    def test_cli_still_works(self):
        intent = classify_query_intent("weka cluster run command --force")
        assert intent.query_type == "cli"

    def test_config_still_works(self):
        intent = classify_query_intent("configure tiering policy setting")
        assert intent.query_type == "config"

    def test_procedural_still_works(self):
        intent = classify_query_intent("how to install WEKA cluster")
        assert intent.query_type == "procedural"

    def test_troubleshooting_still_works(self):
        intent = classify_query_intent("WEKA cluster error failed to start")
        assert intent.query_type == "troubleshooting"

    def test_reference_still_works(self):
        intent = classify_query_intent("what is WEKA deduplication")
        assert intent.query_type == "reference"

    def test_default_conceptual(self):
        intent = classify_query_intent("WEKA cluster overview benefits")
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
        intent = classify_query_intent(f"deploy WEKA on {cue}")
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
        intent = classify_query_intent("WEKA metadata architecture sizing capacity")
        assert intent.query_type == "subsystem_architecture"
        assert len(intent.subsystem_terms) >= 1
        assert len(intent.sizing_terms) >= 1

    def test_config_overrides_subsystem(self):
        """Config patterns take priority over subsystem terms."""
        intent = classify_query_intent("configure metadata tiering.yaml")
        assert intent.query_type == "config"

    def test_cli_overrides_all(self):
        """CLI with enough signals overrides everything."""
        intent = classify_query_intent("weka metadata --tiering command")
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
            "how is metadata managed and architected on a weka cluster"
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
            "How do I appropriately size the weka drives, compute, and frontends containers?"
        )
        assert intent.query_type == "resource_sizing"
        assert intent.precision_mode is True
        assert "size" in intent.sizing_terms or "sizing" in intent.sizing_terms
        assert "drives" in intent.sizing_terms
        assert "compute" in intent.sizing_terms

"""Tests for retrieval plan module."""

import types
import warnings

import pytest

from src.query.retrieval_plan import (
    RetrievalProfile,
    resolve_retrieval_plan,
)


def _make_hybrid_config(**overrides):
    defaults = {
        "colbert_rerank_enabled": True,
        "graph_channel_enabled": False,
        "graph_enrichment_enabled": False,
        "neo4j_disabled": False,
        "graph_adaptive_enabled": True,
        "signal_pool": types.SimpleNamespace(enabled=True),
        "profile": None,
        "profile_overrides": {},
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_ff(**overrides):
    defaults = {
        "signal_diverse_rerank_pool": True,
        "query_api_weighted_fusion": True,
        "signal_pool_before_colbert": True,
        "precision_focused_rerank_text": True,
        "precision_specificity_adjustment": True,
        "graph_garbage_filter": True,
        "graph_score_normalized": True,
        "graph_as_reranker": False,
        "structure_aware_expansion": True,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestProfileResolution:
    def test_vector_only_disables_all_graph(self):
        plan = resolve_retrieval_plan("vector_only", _make_hybrid_config(), _make_ff())
        assert plan.profile == RetrievalProfile.VECTOR_ONLY
        assert plan.use_related_to_expansion is False
        assert plan.use_related_to_blending is False
        assert plan.use_entity_graph_channel is False
        assert plan.use_graph_enrichment is False
        assert plan.use_signal_pool is False
        assert plan.use_colbert is True
        assert plan.use_structure_expansion is False

    def test_precision_vector_enables_pool_graph_free(self):
        plan = resolve_retrieval_plan(
            "precision_vector", _make_hybrid_config(), _make_ff()
        )
        assert plan.profile == RetrievalProfile.PRECISION_VECTOR
        assert plan.use_signal_pool is True
        assert plan.signal_pool_before_colbert is True
        assert plan.use_weighted_fusion is True
        assert plan.use_focused_rerank_text is True
        assert plan.use_structure_expansion is True
        # Graph-free
        assert plan.use_related_to_expansion is False
        assert plan.use_related_to_blending is False
        assert plan.use_entity_graph_channel is False
        # Specificity off by default
        assert plan.use_specificity_adjustment is False

    def test_graph_assisted_enables_blending_and_channel(self):
        plan = resolve_retrieval_plan(
            "graph_assisted", _make_hybrid_config(), _make_ff()
        )
        assert plan.profile == RetrievalProfile.GRAPH_ASSISTED
        assert plan.use_related_to_expansion is True
        assert plan.use_related_to_blending is True
        assert plan.use_entity_graph_channel is True
        assert plan.use_graph_enrichment is False
        assert plan.graph_garbage_filter_on is True
        assert plan.graph_score_normalized_on is True

    def test_graph_full_enables_enrichment(self):
        plan = resolve_retrieval_plan("graph_full", _make_hybrid_config(), _make_ff())
        assert plan.profile == RetrievalProfile.GRAPH_FULL
        assert plan.use_graph_enrichment is True
        assert plan.use_entity_graph_channel is True
        assert plan.use_related_to_blending is True

    def test_specificity_off_in_all_default_profiles(self):
        for profile_name in [
            "vector_only",
            "precision_vector",
            "graph_assisted",
            "graph_full",
        ]:
            plan = resolve_retrieval_plan(
                profile_name, _make_hybrid_config(), _make_ff()
            )
            assert (
                plan.use_specificity_adjustment is False
            ), f"{profile_name} should have specificity off by default"

    def test_graph_score_override_off_in_all_profiles(self):
        for profile_name in [
            "vector_only",
            "precision_vector",
            "graph_assisted",
            "graph_full",
        ]:
            plan = resolve_retrieval_plan(
                profile_name, _make_hybrid_config(), _make_ff()
            )
            assert plan.use_graph_score_override is False


class TestProfileOverrides:
    def test_specificity_override(self):
        hc = _make_hybrid_config(profile_overrides={"use_specificity_adjustment": True})
        plan = resolve_retrieval_plan("precision_vector", hc, _make_ff())
        assert plan.use_specificity_adjustment is True

    def test_unrecognized_override_ignored(self):
        hc = _make_hybrid_config(
            profile_overrides={"use_signal_pool": False}  # not in allowlist
        )
        plan = resolve_retrieval_plan("precision_vector", hc, _make_ff())
        assert plan.use_signal_pool is True  # override ignored


class TestNeo4jDisabledZerosGraph:
    def test_graph_assisted_with_neo4j_disabled(self):
        hc = _make_hybrid_config(neo4j_disabled=True)
        plan = resolve_retrieval_plan("graph_assisted", hc, _make_ff())
        assert plan.profile == RetrievalProfile.GRAPH_ASSISTED
        assert plan.use_related_to_expansion is False
        assert plan.use_related_to_blending is False
        assert plan.use_entity_graph_channel is False
        assert plan.use_graph_enrichment is False
        assert plan.graph_garbage_filter_on is False

    def test_legacy_with_neo4j_disabled(self):
        hc = _make_hybrid_config(
            graph_channel_enabled=True,
            graph_enrichment_enabled=True,
            neo4j_disabled=True,
        )
        plan = resolve_retrieval_plan(None, hc, _make_ff())
        assert plan.use_entity_graph_channel is False
        assert plan.use_graph_enrichment is False
        assert plan.use_related_to_expansion is False


class TestLegacyInference:
    def test_no_profile_infers_precision_vector(self):
        plan = resolve_retrieval_plan(None, _make_hybrid_config(), _make_ff())
        assert plan.profile == RetrievalProfile.PRECISION_VECTOR
        assert plan.use_signal_pool is True
        assert plan.use_weighted_fusion is True

    def test_no_pool_infers_vector_only(self):
        hc = _make_hybrid_config(signal_pool=types.SimpleNamespace(enabled=False))
        ff = _make_ff(signal_diverse_rerank_pool=False, query_api_weighted_fusion=False)
        plan = resolve_retrieval_plan(None, hc, ff)
        assert plan.profile == RetrievalProfile.VECTOR_ONLY

    def test_graph_channel_infers_graph_assisted(self):
        hc = _make_hybrid_config(graph_channel_enabled=True)
        plan = resolve_retrieval_plan(None, hc, _make_ff())
        assert plan.profile == RetrievalProfile.GRAPH_ASSISTED

    def test_graph_enrichment_infers_graph_full(self):
        hc = _make_hybrid_config(
            graph_enrichment_enabled=True, graph_channel_enabled=True
        )
        plan = resolve_retrieval_plan(None, hc, _make_ff())
        assert plan.profile == RetrievalProfile.GRAPH_FULL

    def test_legacy_preserves_blending_bug(self):
        """With profile=None and graph_channel_enabled=False,
        use_related_to_blending should be False (the current bug preserved)."""
        plan = resolve_retrieval_plan(None, _make_hybrid_config(), _make_ff())
        assert plan.use_related_to_blending is False

    def test_legacy_related_to_expansion_on(self):
        """RELATED_TO expansion is on by default in legacy (neo4j not disabled)."""
        plan = resolve_retrieval_plan(None, _make_hybrid_config(), _make_ff())
        assert plan.use_related_to_expansion is True


class TestPlanImmutability:
    def test_frozen_dataclass(self):
        plan = resolve_retrieval_plan("vector_only", _make_hybrid_config(), _make_ff())
        with pytest.raises(AttributeError):
            plan.use_signal_pool = True  # type: ignore


class TestDeprecationWarnings:
    def test_contradicting_legacy_flag_emits_warning(self):
        # graph_channel_enabled=True contradicts vector_only profile
        hc = _make_hybrid_config(graph_channel_enabled=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = resolve_retrieval_plan("vector_only", hc, _make_ff())
            assert plan.use_entity_graph_channel is False  # profile wins
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert any(
                "graph_channel_enabled" in str(x.message) for x in deprecation_warnings
            )

    def test_no_warning_when_flags_match_profile(self):
        # All flags off matches vector_only
        hc = _make_hybrid_config()
        ff = _make_ff(
            signal_pool_before_colbert=False,
            precision_focused_rerank_text=False,
            precision_specificity_adjustment=False,
            graph_garbage_filter=False,
            graph_score_normalized=False,
            structure_aware_expansion=False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_retrieval_plan("vector_only", hc, ff)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

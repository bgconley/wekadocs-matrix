"""Tests for Stage 6 — structural priors materialization.

Validates that _compute_structural_priors correctly computes prior_reference,
prior_entity, and prior_taxonomy from the graph, and handles config disabling
and individual failures gracefully.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.services.cross_doc_linking import CrossDocLinker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Return a mock CrossDocLinkingConfig with sensible defaults."""
    defaults = {
        "enabled": True,
        "method": "rrf",
        "dense_threshold": 0.70,
        "rrf_threshold": 0.025,
        "discovery_threshold": 0.50,
        "chunk_limit": 100,
        "max_edges_per_doc": 5,
        "rrf_k": 60,
        "min_corpus_size": 3,
        "collection_name": "chunks",
        "colbert_rerank": False,
        "colbert_rerank_before_write": False,
        "colbert_threshold": 0.40,
        "colbert_max_chunks": 3,
        "colbert_max_tokens": 200,
        "quality_tier_high": 0.040,
        "quality_tier_medium": 0.028,
        "compute_reciprocity": False,
        "compute_priors": True,
        "entity_hub_threshold": 20,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_linker(config=None, neo4j_driver=None, qdrant_client=None):
    """Build a CrossDocLinker with mocked dependencies."""
    config = config or _make_config()
    neo4j_driver = neo4j_driver or MagicMock()
    qdrant_client = qdrant_client or MagicMock()
    return CrossDocLinker(neo4j_driver, qdrant_client, config)


def _make_session_with_sequential_results(results_sequence):
    """Create a mock Neo4j driver that returns different results on successive session() calls.

    Each entry in results_sequence is a dict representing a single record,
    or None to simulate an exception.

    Returns (mock_driver, list_of_sessions) so tests can inspect individual sessions.
    """
    sessions = []

    for result_data in results_sequence:
        mock_result = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        if isinstance(result_data, Exception):
            mock_session.run.side_effect = result_data
        else:
            mock_result.single.return_value = result_data
            mock_session.run.return_value = mock_result

        sessions.append(mock_session)

    mock_driver = MagicMock()
    mock_driver.session.side_effect = sessions

    return mock_driver, sessions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPriorReferenceFromReferences:
    """Test prior_reference computation from REFERENCES edges."""

    def test_prior_reference_from_references(self):
        """When REFERENCES edges exist, prior_reference should be the max confidence."""
        mock_driver, _ = _make_session_with_sequential_results(
            [
                {"prior_reference": 0.92},  # reference query
                {"prior_entity": 0.5},  # entity query
                {"prior_taxonomy": 0.0},  # taxonomy query
            ]
        )
        linker = _make_linker(neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_reference == pytest.approx(0.92)


class TestPriorEntityWithHubSuppression:
    """Test prior_entity computation with hub-suppressed entity sharing."""

    def test_prior_entity_with_hub_suppression(self):
        """Hub-suppressed entity overlap should be returned as prior_entity."""
        mock_driver, _ = _make_session_with_sequential_results(
            [
                {"prior_reference": None},  # no REFERENCES edges
                {"prior_entity": 0.35},  # 35% entity overlap after hub suppression
                {"prior_taxonomy": 0.0},  # taxonomy query
            ]
        )
        linker = _make_linker(neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_reference is None
        assert priors.prior_entity == pytest.approx(0.35)


class TestPriorTaxonomySameTag:
    """Test prior_taxonomy returns 1.0 when doc_tag matches."""

    def test_prior_taxonomy_same_tag(self):
        """Same doc_tag should yield prior_taxonomy=1.0."""
        mock_driver, _ = _make_session_with_sequential_results(
            [
                {"prior_reference": None},
                {"prior_entity": None},
                {"prior_taxonomy": 1.0},
            ]
        )
        linker = _make_linker(neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_taxonomy == pytest.approx(1.0)


class TestPriorTaxonomySameCategory:
    """Test prior_taxonomy returns 0.5 when only doc_category matches."""

    def test_prior_taxonomy_same_category(self):
        """Same doc_category but different doc_tag should yield 0.5."""
        mock_driver, _ = _make_session_with_sequential_results(
            [
                {"prior_reference": None},
                {"prior_entity": None},
                {"prior_taxonomy": 0.5},
            ]
        )
        linker = _make_linker(neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_taxonomy == pytest.approx(0.5)


class TestPriorTaxonomyNoMatch:
    """Test prior_taxonomy returns 0.0 when neither tag nor category matches."""

    def test_prior_taxonomy_no_match(self):
        """No taxonomy alignment should yield 0.0."""
        mock_driver, _ = _make_session_with_sequential_results(
            [
                {"prior_reference": None},
                {"prior_entity": None},
                {"prior_taxonomy": 0.0},
            ]
        )
        linker = _make_linker(neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_taxonomy == pytest.approx(0.0)


class TestPriorsDisabledByConfig:
    """Test that no Neo4j calls are made when compute_priors is False."""

    def test_priors_disabled_by_config(self):
        """With compute_priors=False, should return empty StructuralPriors without Neo4j calls."""
        mock_driver = MagicMock()
        config = _make_config(compute_priors=False)
        linker = _make_linker(config=config, neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_reference is None
        assert priors.prior_entity is None
        assert priors.prior_taxonomy is None
        mock_driver.session.assert_not_called()


class TestIndividualPriorFailureGraceful:
    """Test that failure in one prior does not prevent others from computing."""

    def test_individual_prior_failure_graceful(self):
        """When the entity query fails, reference and taxonomy should still compute."""
        mock_driver, _ = _make_session_with_sequential_results(
            [
                {"prior_reference": 0.85},  # reference succeeds
                Exception("entity index unavailable"),  # entity fails
                {"prior_taxonomy": 0.5},  # taxonomy succeeds
            ]
        )
        linker = _make_linker(neo4j_driver=mock_driver)

        priors = linker._compute_structural_priors("src-doc", "tgt-doc")

        assert priors.prior_reference == pytest.approx(0.85)
        assert priors.prior_entity is None  # failed, stays None
        assert priors.prior_taxonomy == pytest.approx(0.5)

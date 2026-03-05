"""Tests for Stage 5 — reciprocity reconciliation.

Validates that _reconcile_reciprocity correctly sets is_mutual and mutual_score
on RELATED_TO edges, and handles config disabling and failures gracefully.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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
        "compute_reciprocity": True,
        "compute_priors": False,
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


def _mock_neo4j_run(return_record):
    """Create a mock Neo4j driver whose session().run() returns the given record."""
    mock_result = MagicMock()
    mock_result.single.return_value = return_record

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    return mock_driver, mock_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMutualEdgesDetected:
    """Test that mutual edges are correctly detected and updated."""

    def test_mutual_edges_detected(self):
        """When both A->B and B->A edges exist, reciprocity should return the updated count."""
        mock_driver, mock_session = _mock_neo4j_run({"updated": 4})
        linker = _make_linker(neo4j_driver=mock_driver)

        count = linker._reconcile_reciprocity("doc-abc-123-456")

        assert count == 4
        mock_session.run.assert_called_once()
        # Verify the Cypher query was passed with the correct doc_id parameter
        args, kwargs = mock_session.run.call_args
        assert kwargs["doc_id"] == "doc-abc-123-456"


class TestOneWayNotMutual:
    """Test that one-way edges (no reciprocal) still update but report correctly."""

    def test_one_way_not_mutual(self):
        """When only A->B exists (no B->A), the query still runs and returns count."""
        # The Cypher query sets is_mutual=false for one-way edges;
        # the returned count reflects how many r1 edges were processed.
        mock_driver, mock_session = _mock_neo4j_run({"updated": 2})
        linker = _make_linker(neo4j_driver=mock_driver)

        count = linker._reconcile_reciprocity("doc-one-way")

        assert count == 2
        mock_session.run.assert_called_once()


class TestReciprocityDisabledByConfig:
    """Test that no Neo4j calls are made when compute_reciprocity is False."""

    def test_reciprocity_disabled_by_config(self):
        """With compute_reciprocity=False, should return 0 without calling Neo4j."""
        mock_driver = MagicMock()
        config = _make_config(compute_reciprocity=False)
        linker = _make_linker(config=config, neo4j_driver=mock_driver)

        count = linker._reconcile_reciprocity("doc-disabled")

        assert count == 0
        mock_driver.session.assert_not_called()


class TestReciprocityFailureGraceful:
    """Test that Neo4j exceptions are caught and return 0."""

    def test_reciprocity_failure_graceful(self):
        """When Neo4j raises an exception, should log warning and return 0."""
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("connection lost")
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        linker = _make_linker(neo4j_driver=mock_driver)

        count = linker._reconcile_reciprocity("doc-fail-test")

        assert count == 0

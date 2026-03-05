"""Tests for the cross_doc_linking v2 edge model integration.

Validates that the refactored CrossDocLinker correctly uses CandidateSignals,
EdgePayload, and the rerank-before-write architecture.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.services.cross_doc_edge_model import (
    EDGE_MODEL_VERSION,
    CandidateSignals,
    EdgePayload,
)
from src.services.cross_doc_linking import (
    LinkingResult,
    aggregate_chunks_to_documents,
    reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(point_id: str, document_id: str, score: float):
    """Create a mock Qdrant ScoredPoint-like object."""
    return SimpleNamespace(
        id=point_id,
        payload={"document_id": document_id},
        score=score,
    )


# ---------------------------------------------------------------------------
# Test 1: EdgePayload.from_candidate has all fields
# ---------------------------------------------------------------------------


class TestEdgePayloadFromCandidateHasAllFields:
    def test_edge_payload_from_candidate_has_all_fields(self):
        """Mock candidate produces EdgePayload with all expected fields populated."""
        candidate = CandidateSignals(
            target_doc_id="doc-target-001",
            score_dense=0.85,
            score_sparse=0.62,
            score_rrf=0.038,
            score_colbert=0.90,
            score_title_ft=None,
            dense_rank=1,
            sparse_rank=3,
        )

        thresholds = {"high": 0.040, "medium": 0.028}

        payload = EdgePayload.from_candidate(
            signals=candidate,
            method="rrf_fusion+colbert",
            phase="3.5b",
            quality_thresholds=thresholds,
        )

        # Signal scores propagated
        assert payload.score_final == 0.90  # colbert wins via score_final
        assert payload.score == 0.90  # legacy alias
        assert payload.score_dense == 0.85
        assert payload.score_sparse == 0.62
        assert payload.score_rrf == 0.038
        assert payload.score_colbert == 0.90
        assert payload.score_title_ft is None

        # Provenance
        assert payload.method == "rrf_fusion+colbert"
        assert payload.method_version == EDGE_MODEL_VERSION
        assert payload.phase == "3.5b"

        # Quality tier (0.90 >= 0.040 threshold for rrf -> "high",
        # but score_final=0.90, and method contains "rrf" so uses rrf thresholds)
        assert payload.quality_tier == "high"

        # Reciprocity fields exist with defaults
        assert payload.is_mutual is None
        assert payload.mutual_score is None

    def test_edge_payload_quality_tier_low(self):
        """Low-scoring candidate gets quality_tier='low'."""
        candidate = CandidateSignals(
            target_doc_id="doc-low",
            score_rrf=0.020,
        )
        payload = EdgePayload.from_candidate(
            signals=candidate,
            method="rrf_fusion",
            phase="3.5b",
            quality_thresholds={"high": 0.040, "medium": 0.028},
        )
        assert payload.quality_tier == "low"

    def test_edge_payload_quality_tier_medium(self):
        """Medium-scoring candidate gets quality_tier='medium'."""
        candidate = CandidateSignals(
            target_doc_id="doc-med",
            score_rrf=0.032,
        )
        payload = EdgePayload.from_candidate(
            signals=candidate,
            method="rrf_fusion",
            phase="3.5b",
            quality_thresholds={"high": 0.040, "medium": 0.028},
        )
        assert payload.quality_tier == "medium"


# ---------------------------------------------------------------------------
# Test 2: _create_edge uses EdgePayload params
# ---------------------------------------------------------------------------


class TestCreateEdgeUsesPayloadParams:
    def test_create_edge_uses_payload_params(self):
        """Mock Neo4j session captures Cypher params with v2 payload fields."""
        from src.services.cross_doc_linking import CrossDocLinker
        from src.shared.config import CrossDocLinkingConfig

        # Build config
        config = CrossDocLinkingConfig()

        # Mock Neo4j driver + session
        mock_record = MagicMock()
        mock_record.get.return_value = True  # was_created = True

        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        mock_qdrant = MagicMock()

        linker = CrossDocLinker(mock_driver, mock_qdrant, config)

        # Build payload
        candidate = CandidateSignals(
            target_doc_id="target-abc",
            score_dense=0.82,
            score_rrf=0.035,
            score_colbert=0.88,
        )
        payload = EdgePayload.from_candidate(
            signals=candidate,
            method="rrf_fusion+colbert",
            phase="3.5b",
        )

        created, updated = linker._create_edge(
            source_id="source-xyz",
            target_id="target-abc",
            payload=payload,
            dry_run=False,
        )

        assert created is True
        assert updated is False

        # Verify session.run was called with correct parameters
        call_args = mock_session.run.call_args
        assert call_args is not None

        # Check named params
        kwargs = call_args.kwargs if call_args.kwargs else {}
        # Fall back to positional if kwargs empty
        if not kwargs and len(call_args.args) > 1:
            # session.run(cypher, source_id=..., target_id=..., props=...)
            pass

        # The run call should have source_id, target_id, props kwargs
        call_kwargs = call_args[1] if len(call_args) > 1 else call_args.kwargs
        assert call_kwargs["source_id"] == "source-xyz"
        assert call_kwargs["target_id"] == "target-abc"

        props = call_kwargs["props"]
        assert props["score_final"] == 0.88  # colbert wins
        assert props["score"] == 0.88  # legacy alias
        assert props["score_dense"] == 0.82
        assert props["score_rrf"] == 0.035
        assert props["score_colbert"] == 0.88
        assert props["method"] == "rrf_fusion+colbert"
        assert props["method_version"] == EDGE_MODEL_VERSION
        assert props["phase"] == "3.5b"
        assert "quality_tier" in props

    def test_create_edge_dry_run_returns_true_false(self):
        """Dry run returns (True, False) without touching Neo4j."""
        from src.services.cross_doc_linking import CrossDocLinker
        from src.shared.config import CrossDocLinkingConfig

        config = CrossDocLinkingConfig()
        mock_driver = MagicMock()
        mock_qdrant = MagicMock()
        linker = CrossDocLinker(mock_driver, mock_qdrant, config)

        payload = EdgePayload(score_final=0.5, method="test")
        created, updated = linker._create_edge("src", "tgt", payload, dry_run=True)
        assert created is True
        assert updated is False

        # Neo4j session should not have been opened
        mock_driver.session.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: LinkingResult has reciprocity_updated field
# ---------------------------------------------------------------------------


class TestLinkingResultReciprocityField:
    def test_linking_result_has_reciprocity_field(self):
        """reciprocity_updated field exists with default 0."""
        result = LinkingResult(source_document_id="doc-001")
        assert hasattr(result, "reciprocity_updated")
        assert result.reciprocity_updated == 0

    def test_linking_result_reciprocity_serializes(self):
        """reciprocity_updated appears in to_dict() output."""
        result = LinkingResult(
            source_document_id="doc-002",
            edges_created=3,
            reciprocity_updated=2,
        )
        d = result.to_dict()
        assert "reciprocity_updated" in d
        assert d["reciprocity_updated"] == 2

    def test_linking_result_reciprocity_default_in_dict(self):
        """Default reciprocity_updated=0 appears in to_dict()."""
        result = LinkingResult(source_document_id="doc-003")
        d = result.to_dict()
        assert d["reciprocity_updated"] == 0


# ---------------------------------------------------------------------------
# Test 4: Deprecated aggregate_chunks_to_documents still works
# ---------------------------------------------------------------------------


class TestDeprecatedAggregateStillWorks:
    def test_deprecated_aggregate_still_works(self):
        """Old aggregate_chunks_to_documents returns List[Tuple[str, float]]."""
        hits = [
            _make_hit("c1", "doc-A", 0.90),
            _make_hit("c2", "doc-A", 0.70),
            _make_hit("c3", "doc-B", 0.85),
            _make_hit("c4", "doc-C", 0.60),
        ]
        result = aggregate_chunks_to_documents(hits, exclude_document_id="doc-X")

        # Should return list of (doc_id, score) tuples
        assert isinstance(result, list)
        assert len(result) == 3

        # Each element is a tuple
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

        # Sorted by score descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

        # Max-score wins for doc-A
        doc_a = [s for d, s in result if d == "doc-A"]
        assert doc_a[0] == 0.90

    def test_deprecated_aggregate_excludes_self(self):
        """Old function excludes the source document."""
        hits = [
            _make_hit("c1", "self-doc", 0.95),
            _make_hit("c2", "other-doc", 0.80),
        ]
        result = aggregate_chunks_to_documents(hits, exclude_document_id="self-doc")
        doc_ids = [d for d, _ in result]
        assert "self-doc" not in doc_ids
        assert "other-doc" in doc_ids

    def test_deprecated_rrf_still_works(self):
        """Old reciprocal_rank_fusion returns List[Tuple[str, float]]."""
        dense = [("doc-A", 0.90), ("doc-B", 0.80)]
        sparse = [("doc-B", 0.70), ("doc-C", 0.60)]

        result = reciprocal_rank_fusion(dense, sparse, k=60)

        assert isinstance(result, list)
        assert len(result) == 3

        # Each element is a tuple
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

        # doc-B should be first (appears in both lists)
        doc_ids = [d for d, _ in result]
        assert doc_ids[0] == "doc-B"

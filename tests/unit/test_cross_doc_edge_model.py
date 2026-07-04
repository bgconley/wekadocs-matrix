"""Tests for the shared RELATED_TO v2 edge data model."""

from __future__ import annotations

from types import SimpleNamespace

from src.services.cross_doc_edge_model import (
    CandidateSignals,
    EdgePayload,
    StructuralPriors,
    aggregate_chunks_to_candidates,
    build_edge_merge_cypher,
    reciprocal_rank_fusion_v2,
)

# ---------------------------------------------------------------------------
# CandidateSignals.score_final
# ---------------------------------------------------------------------------


class TestCandidateScoreFinal:
    def test_candidate_score_final_colbert_wins(self):
        """score_final prefers colbert when present."""
        c = CandidateSignals(
            target_doc_id="d1",
            score_dense=0.80,
            score_rrf=0.035,
            score_colbert=0.92,
        )
        assert c.score_final == 0.92

    def test_candidate_score_final_rrf_fallback(self):
        """score_final uses rrf when no colbert."""
        c = CandidateSignals(
            target_doc_id="d2",
            score_dense=0.80,
            score_rrf=0.035,
        )
        assert c.score_final == 0.035

    def test_candidate_score_final_dense_fallback(self):
        """score_final uses dense when no rrf/colbert."""
        c = CandidateSignals(
            target_doc_id="d3",
            score_dense=0.75,
        )
        assert c.score_final == 0.75

    def test_candidate_score_final_zero_default(self):
        """score_final is 0.0 when all scores are None."""
        c = CandidateSignals(target_doc_id="d4")
        assert c.score_final == 0.0


# ---------------------------------------------------------------------------
# EdgePayload.to_neo4j_params
# ---------------------------------------------------------------------------


class TestEdgePayloadParams:
    def test_edge_payload_to_neo4j_params_omits_none(self):
        """None fields are excluded from the output dict."""
        ep = EdgePayload(
            score_final=0.5,
            score_dense=0.8,
            method="dense",
        )
        params = ep.to_neo4j_params()
        assert "score_sparse" not in params
        assert "score_rrf" not in params
        assert "score_colbert" not in params
        assert "score_title_ft" not in params
        assert "prior_reference" not in params
        assert "is_mutual" not in params
        assert "mutual_score" not in params
        assert "quality_tier" not in params

    def test_edge_payload_score_legacy_alias(self):
        """The ``score`` field in params equals score_final."""
        ep = EdgePayload(score_final=0.42)
        params = ep.to_neo4j_params()
        assert params["score"] == params["score_final"]
        assert params["score"] == 0.42


# ---------------------------------------------------------------------------
# EdgePayload.from_candidate
# ---------------------------------------------------------------------------


class TestEdgePayloadFromCandidate:
    def test_edge_payload_from_candidate(self):
        """All fields propagate correctly from CandidateSignals."""
        signals = CandidateSignals(
            target_doc_id="doc-A",
            score_dense=0.82,
            score_sparse=0.60,
            score_rrf=0.038,
            score_colbert=0.91,
            score_title_ft=3.5,
            dense_rank=2,
            sparse_rank=5,
        )
        priors = StructuralPriors(
            prior_reference=0.1,
            prior_entity=0.05,
            prior_taxonomy=0.03,
        )
        ep = EdgePayload.from_candidate(
            signals=signals,
            method="rrf_fusion",
            phase="3.5b",
            priors=priors,
        )

        # score_final should be colbert (highest priority non-None)
        assert ep.score_final == 0.91
        assert ep.score == 0.91
        assert ep.score_dense == 0.82
        assert ep.score_sparse == 0.60
        assert ep.score_rrf == 0.038
        assert ep.score_colbert == 0.91
        assert ep.score_title_ft == 3.5
        assert ep.method == "rrf_fusion"
        assert ep.phase == "3.5b"
        assert ep.prior_reference == 0.1
        assert ep.prior_entity == 0.05
        assert ep.prior_taxonomy == 0.03
        assert ep.quality_tier is not None


# ---------------------------------------------------------------------------
# EdgePayload._compute_quality_tier
# ---------------------------------------------------------------------------


class TestQualityTier:
    def test_quality_tier_rrf_high(self):
        """score_final >= 0.040 with rrf method -> 'high'."""
        ep = EdgePayload(score_final=0.045, method="rrf_fusion")
        assert ep._compute_quality_tier() == "high"

    def test_quality_tier_rrf_medium(self):
        """0.028 <= score_final < 0.040 with rrf method -> 'medium'."""
        ep = EdgePayload(score_final=0.032, method="rrf_fusion")
        assert ep._compute_quality_tier() == "medium"

    def test_quality_tier_rrf_low(self):
        """score_final < 0.028 with rrf method -> 'low'."""
        ep = EdgePayload(score_final=0.020, method="rrf_fusion")
        assert ep._compute_quality_tier() == "low"


# ---------------------------------------------------------------------------
# aggregate_chunks_to_candidates
# ---------------------------------------------------------------------------


def _make_hit(point_id: str, document_id: str, score: float):
    """Create a mock Qdrant ScoredPoint-like object."""
    return SimpleNamespace(
        id=point_id,
        payload={"document_id": document_id},
        score=score,
    )


class TestAggregateChunks:
    def test_aggregate_chunks_excludes_self(self):
        """Chunks belonging to exclude_doc_id are filtered out."""
        hits = [
            _make_hit("c1", "self-doc", 0.95),
            _make_hit("c2", "other-doc", 0.80),
        ]
        result = aggregate_chunks_to_candidates(hits, exclude_doc_id="self-doc")
        doc_ids = [c.target_doc_id for c in result]
        assert "self-doc" not in doc_ids
        assert "other-doc" in doc_ids

    def test_aggregate_chunks_max_score_wins(self):
        """When multiple chunks share a document_id, the highest score wins."""
        hits = [
            _make_hit("c1", "doc-X", 0.60),
            _make_hit("c2", "doc-X", 0.90),
            _make_hit("c3", "doc-X", 0.75),
        ]
        result = aggregate_chunks_to_candidates(hits, exclude_doc_id="doc-Z")
        assert len(result) == 1
        assert result[0].target_doc_id == "doc-X"
        assert result[0].score_dense == 0.90


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion_v2
# ---------------------------------------------------------------------------


class TestRRFv2:
    def test_rrf_v2_preserves_component_scores(self):
        """Dense and sparse scores survive fusion."""
        dense = [
            CandidateSignals(target_doc_id="d1", score_dense=0.90),
            CandidateSignals(target_doc_id="d2", score_dense=0.80),
        ]
        sparse = [
            CandidateSignals(target_doc_id="d2", score_sparse=0.70),
            CandidateSignals(target_doc_id="d3", score_sparse=0.60),
        ]
        merged = reciprocal_rank_fusion_v2(dense, sparse, k=60)
        by_id = {c.target_doc_id: c for c in merged}

        # d2 appears in both lists — should have both scores
        assert by_id["d2"].score_dense == 0.80
        assert by_id["d2"].score_sparse == 0.70
        assert by_id["d2"].score_rrf is not None

        # d1 only in dense — sparse should be None
        assert by_id["d1"].score_dense == 0.90
        assert by_id["d1"].score_sparse is None
        assert by_id["d1"].score_rrf is not None

        # d3 only in sparse — dense should be None
        assert by_id["d3"].score_dense is None
        assert by_id["d3"].score_sparse == 0.60
        assert by_id["d3"].score_rrf is not None

    def test_rrf_v2_sets_ranks(self):
        """dense_rank and sparse_rank are populated correctly."""
        dense = [
            CandidateSignals(target_doc_id="d1", score_dense=0.90),
            CandidateSignals(target_doc_id="d2", score_dense=0.80),
        ]
        sparse = [
            CandidateSignals(target_doc_id="d2", score_sparse=0.70),
            CandidateSignals(target_doc_id="d1", score_sparse=0.50),
        ]
        merged = reciprocal_rank_fusion_v2(dense, sparse, k=60)
        by_id = {c.target_doc_id: c for c in merged}

        assert by_id["d1"].dense_rank == 1
        assert by_id["d1"].sparse_rank == 2
        assert by_id["d2"].dense_rank == 2
        assert by_id["d2"].sparse_rank == 1


# ---------------------------------------------------------------------------
# build_edge_merge_cypher
# ---------------------------------------------------------------------------


class TestBuildEdgeMergeCypher:
    def test_build_edge_merge_cypher(self):
        """Returns non-empty string with MERGE, ON CREATE, ON MATCH."""
        cypher = build_edge_merge_cypher()
        assert isinstance(cypher, str)
        assert len(cypher) > 0
        assert "MERGE" in cypher
        assert "ON CREATE" in cypher
        assert "ON MATCH" in cypher
        assert "$source_id" in cypher
        assert "$target_id" in cypher
        assert "$props" in cypher
        assert "$now" in cypher
        assert "was_created" in cypher

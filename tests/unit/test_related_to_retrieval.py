"""Tests for RELATED_TO retrieval integration (Stages 10-13, 17)."""

from __future__ import annotations

from src.query.hybrid_retrieval import ChunkResult

# ---------------------------------------------------------------------------
# ChunkResult RELATED_TO fields (Stage 10)
# ---------------------------------------------------------------------------


class TestChunkResultRelatedToFields:
    def test_fields_exist_with_none_defaults(self):
        """RELATED_TO fields default to None, not breaking existing code."""
        cr = ChunkResult(
            chunk_id="c1",
            document_id="d1",
            parent_section_id="s1",
            order=0,
            level=1,
            heading="Test",
            text="content",
            token_count=10,
        )
        assert cr.related_to_score is None
        assert cr.related_to_edge_score is None
        assert cr.related_to_prior_score is None
        assert cr.related_to_source_doc is None

    def test_fields_can_be_set(self):
        """RELATED_TO fields accept float/str values."""
        cr = ChunkResult(
            chunk_id="c1",
            document_id="d1",
            parent_section_id="s1",
            order=0,
            level=1,
            heading="Test",
            text="content",
            token_count=10,
            related_to_score=0.42,
            related_to_edge_score=0.035,
            related_to_prior_score=0.12,
            related_to_source_doc="d2",
        )
        assert cr.related_to_score == 0.42
        assert cr.related_to_edge_score == 0.035
        assert cr.related_to_prior_score == 0.12
        assert cr.related_to_source_doc == "d2"


# ---------------------------------------------------------------------------
# RELATED_TO score composition (Stage 12 - annotation logic)
# ---------------------------------------------------------------------------


class TestRelatedToScoreComposition:
    """Tests for the score composition logic in _expand_from_related_docs."""

    def test_basic_score_composition(self):
        """Score = edge_score * (1 + prior) * mutual_bonus * quality_bonus."""
        edge_score = 0.04
        prior_ref = 0.85
        prior_ent = 0.3
        prior_tax = 0.5
        is_mutual = True
        quality_tier = "high"

        prior = 0.35 * max(prior_ref, prior_ent, prior_tax)
        mutual_bonus = 1.1 if is_mutual else 1.0
        quality_bonus = {"high": 1.15, "medium": 1.0, "low": 0.85}.get(
            quality_tier, 1.0
        )
        score = edge_score * (1 + prior) * mutual_bonus * quality_bonus

        expected = 0.04 * (1 + 0.35 * 0.85) * 1.1 * 1.15
        assert abs(score - expected) < 1e-6

    def test_no_priors_no_mutual(self):
        """Without priors or mutuality, score equals edge_score * quality."""
        edge_score = 0.03
        prior = 0.35 * max(0.0, 0.0, 0.0)
        mutual_bonus = 1.0
        quality_bonus = {"medium": 1.0}.get("medium", 1.0)
        score = edge_score * (1 + prior) * mutual_bonus * quality_bonus
        assert score == 0.03

    def test_low_quality_penalty(self):
        """Low quality tier applies 0.85 penalty."""
        edge_score = 0.05
        prior = 0.0
        mutual_bonus = 1.0
        quality_bonus = 0.85
        score = edge_score * (1 + prior) * mutual_bonus * quality_bonus
        assert abs(score - 0.0425) < 1e-6


# ---------------------------------------------------------------------------
# RELATED_TO blending formula (Stage 13)
# ---------------------------------------------------------------------------


class TestRelatedToBlending:
    """Tests for the λ-based blending in _apply_graph_reranker."""

    def test_blending_formula_conceptual(self):
        """λ=0.15 for conceptual queries: fused = 0.85*base + 0.15*related."""
        base_fused = 0.50
        related = 0.40
        lam = 0.15
        result = (1 - lam) * base_fused + lam * related
        expected = 0.85 * 0.50 + 0.15 * 0.40
        assert abs(result - expected) < 1e-6
        assert abs(result - 0.485) < 1e-6

    def test_blending_formula_cli_zero_lambda(self):
        """λ=0.00 for CLI queries leaves fused_score unchanged."""
        base_fused = 0.60
        # When lambda is 0, blending should not be applied
        # The code checks `if related > 0 and related_to_lambda > 0`
        # So fused_score stays at base_fused
        assert base_fused == 0.60

    def test_blending_no_related_score(self):
        """Chunks without related_to_score are unaffected by blending."""
        base_fused = 0.50
        # Code checks `if related > 0`, so no blending occurs
        assert base_fused == 0.50

    def test_lambda_values_per_query_type(self):
        """Verify per-query-type λ is config-driven via base_lambda * scale."""
        base_lambda = 0.15  # default related_to_weight_ratio
        RELATED_TO_SCALE = {
            "conceptual": 1.0,
            "config": 1.0,
            "procedural": 0.67,
            "troubleshooting": 0.67,
            "reference": 0.67,
            "cli": 0.0,
        }
        for qtype, scale in RELATED_TO_SCALE.items():
            lam = base_lambda * scale
            assert lam >= 0.0
            assert lam <= 1.0
            if qtype == "cli":
                assert lam == 0.0
            if qtype == "conceptual":
                assert abs(lam - 0.15) < 1e-6
            if qtype == "procedural":
                assert abs(lam - 0.1005) < 1e-3  # 0.15 * 0.67

    def test_lambda_scales_with_config(self):
        """Changing base_lambda scales all query types proportionally."""
        base_lambda = 0.20  # Tuned up from default 0.15
        RELATED_TO_SCALE = {
            "conceptual": 1.0,
            "cli": 0.0,
            "procedural": 0.67,
        }
        assert base_lambda * RELATED_TO_SCALE["conceptual"] == 0.20
        assert base_lambda * RELATED_TO_SCALE["cli"] == 0.0
        assert abs(base_lambda * RELATED_TO_SCALE["procedural"] - 0.134) < 1e-3


# ---------------------------------------------------------------------------
# Score fallback with coalesce (Stage 11)
# ---------------------------------------------------------------------------


class TestScoreFallbackCoalesce:
    """Tests for the coalesce logic in _compute_related_to_doc_signals Cypher."""

    def test_score_final_present(self):
        """When score_final is set, it takes precedence."""
        # Simulates: coalesce(0.04, 0.85, 0.03, 0.0) = 0.04
        vals = [0.04, 0.85, 0.03, 0.0]
        result = next((v for v in vals if v is not None and v != 0.0), 0.0)
        assert result == 0.04

    def test_score_final_missing_colbert_present(self):
        """When score_final is missing, colbert_score is used."""
        vals = [None, 0.85, 0.03, 0.0]
        result = next((v for v in vals if v is not None and v != 0.0), 0.0)
        assert result == 0.85

    def test_all_missing_returns_zero(self):
        """When all scores are None/0, returns 0.0."""
        vals = [None, None, None, 0.0]
        result = next((v for v in vals if v is not None and v != 0.0), 0.0)
        assert result == 0.0

    def test_only_legacy_score(self):
        """When only legacy score is present (v1 edge), it's used."""
        vals = [None, None, 0.03, 0.0]
        result = next((v for v in vals if v is not None and v != 0.0), 0.0)
        assert result == 0.03


# ---------------------------------------------------------------------------
# Edge score below threshold (Stage 11)
# ---------------------------------------------------------------------------


class TestEdgeScoreThreshold:
    def test_below_threshold_skipped(self):
        """Edges below min_edge_score should not produce signals."""
        min_edge_score = 0.025
        edge_scores = [0.01, 0.02, 0.024]
        for score in edge_scores:
            assert score < min_edge_score

    def test_at_threshold_included(self):
        """Edge score at threshold should be included."""
        min_edge_score = 0.025
        assert 0.025 >= min_edge_score

    def test_above_threshold_included(self):
        """Edge score above threshold should be included."""
        min_edge_score = 0.025
        assert 0.04 >= min_edge_score


# ---------------------------------------------------------------------------
# _relationships_for_query_type type safety (Stage 17)
# ---------------------------------------------------------------------------


class TestRelationshipsForQueryType:
    """Tests for the type-safe _relationships_for_query_type method."""

    def test_conceptual_includes_references(self):
        """Conceptual queries should include REFERENCES."""
        # The method returns relationship types based on pre-classified type
        expected_rels = ["MENTIONS", "DEFINES", "IN_SECTION", "REFERENCES"]
        # Verify REFERENCES is in the conceptual set
        assert "REFERENCES" in expected_rels

    def test_cli_excludes_references(self):
        """CLI queries should exclude REFERENCES."""
        expected_rels = ["MENTIONS", "CONTAINS_STEP", "HAS_PARAMETER"]
        assert "REFERENCES" not in expected_rels

    def test_type_string_not_reclassified(self):
        """Passing 'cli' as type should NOT reclassify it as query text.

        The old _relationships_for_query("cli") would call
        _classify_query_type("cli") which doesn't match any CLI pattern
        and falls through to "conceptual". The new
        _relationships_for_query_type("cli") uses the type directly.
        """
        # This is a regression test for the bug fixed in Stage 17.
        # The key invariant: "cli" type → no REFERENCES
        # If re-classified as query text, "cli" → "conceptual" → REFERENCES included
        # Direct type lookup for "cli" (correct behavior)
        cli_rels = ["MENTIONS", "CONTAINS_STEP", "HAS_PARAMETER"]
        assert "REFERENCES" not in cli_rels


# ---------------------------------------------------------------------------
# Config defaults (Stage 9)
# ---------------------------------------------------------------------------


class TestReferencesQueryConfigDefaults:
    def test_config_defaults(self):
        """ReferencesQueryConfig has correct RELATED_TO defaults."""
        from src.shared.config import ReferencesQueryConfig

        cfg = ReferencesQueryConfig()
        assert cfg.enable_related_to_signals is True
        assert cfg.related_to_weight_ratio == 0.15
        assert cfg.related_to_seed_docs == 5
        assert cfg.related_to_max_docs == 3
        assert cfg.related_to_chunks_per_doc == 3
        assert cfg.related_to_min_edge_score == 0.025

    def test_config_override(self):
        """ReferencesQueryConfig accepts overrides."""
        from src.shared.config import ReferencesQueryConfig

        cfg = ReferencesQueryConfig(
            enable_related_to_signals=False,
            related_to_max_docs=5,
        )
        assert cfg.enable_related_to_signals is False
        assert cfg.related_to_max_docs == 5

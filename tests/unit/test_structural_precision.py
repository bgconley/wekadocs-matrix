"""Tests for structural boost behavior with precision query types."""

import pytest

from src.query.structural_retrieval import (
    DEFAULT_QUERY_TYPE_RRF_WEIGHTS,
    DEFAULT_STRUCTURAL_BOOSTS,
    apply_structural_boost,
    get_query_type_rrf_weights,
)


class TestSubsystemArchitectureNoDepthPenalty:
    """The critical fix: subsystem_architecture must NOT penalize deep nesting."""

    def test_depth_4_no_penalty(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 4,
                    "has_code": False,
                    "has_table": False,
                },
            }
        ]
        boosted = apply_structural_boost(results, "subsystem_architecture")
        assert boosted[0]["score"] == 0.8  # Unchanged

    def test_depth_10_no_penalty(self):
        results = [
            {
                "score": 0.5,
                "payload": {
                    "parent_path_depth": 10,
                    "has_code": False,
                    "has_table": False,
                },
            }
        ]
        boosted = apply_structural_boost(results, "subsystem_architecture")
        assert boosted[0]["score"] == 0.5  # Unchanged

    def test_code_gets_small_boost(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 3,
                    "has_code": True,
                    "has_table": False,
                },
            }
        ]
        boosted = apply_structural_boost(results, "subsystem_architecture")
        assert boosted[0]["score"] == pytest.approx(0.8 * 1.05)

    def test_table_gets_boost(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 3,
                    "has_code": False,
                    "has_table": True,
                },
            }
        ]
        boosted = apply_structural_boost(results, "subsystem_architecture")
        assert boosted[0]["score"] == pytest.approx(0.8 * 1.10)


class TestResourceSizingTableBoost:
    def test_table_gets_25_percent_boost(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 2,
                    "has_code": False,
                    "has_table": True,
                },
            }
        ]
        boosted = apply_structural_boost(results, "resource_sizing")
        assert boosted[0]["score"] == pytest.approx(0.8 * 1.25)

    def test_depth_no_penalty(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 5,
                    "has_code": False,
                    "has_table": False,
                },
            }
        ]
        boosted = apply_structural_boost(results, "resource_sizing")
        assert boosted[0]["score"] == 0.8  # Unchanged


class TestConceptualPenaltyStillApplies:
    """Regression: conceptual type must still penalize depth > 2."""

    def test_depth_3_gets_penalty(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 3,
                    "has_code": False,
                    "has_table": False,
                },
            }
        ]
        boosted = apply_structural_boost(results, "conceptual")
        assert boosted[0]["score"] == pytest.approx(0.8 * 0.85)

    def test_depth_2_no_penalty(self):
        results = [
            {
                "score": 0.8,
                "payload": {
                    "parent_path_depth": 2,
                    "has_code": False,
                    "has_table": False,
                },
            }
        ]
        boosted = apply_structural_boost(results, "conceptual")
        assert boosted[0]["score"] == 0.8  # At threshold, no penalty


class TestNewQueryTypeWeights:
    def test_subsystem_architecture_weights_exist(self):
        weights = get_query_type_rrf_weights("subsystem_architecture")
        assert weights["text-sparse"] == 2.5
        assert weights["content"] == 1.0
        assert weights["entity-sparse"] == 0.8

    def test_resource_sizing_weights_exist(self):
        weights = get_query_type_rrf_weights("resource_sizing")
        assert weights["text-sparse"] == 2.0
        assert weights["content"] == 1.2
        assert weights["entity-sparse"] == 0.8

    def test_new_types_in_default_dict(self):
        assert "subsystem_architecture" in DEFAULT_QUERY_TYPE_RRF_WEIGHTS
        assert "resource_sizing" in DEFAULT_QUERY_TYPE_RRF_WEIGHTS

    def test_new_types_in_boost_dict(self):
        assert "subsystem_architecture" in DEFAULT_STRUCTURAL_BOOSTS
        assert "resource_sizing" in DEFAULT_STRUCTURAL_BOOSTS

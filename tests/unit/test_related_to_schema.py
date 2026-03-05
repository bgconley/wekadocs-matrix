"""
Unit tests for RELATED_TO schema prerequisites (Stage 4).

Verifies:
- Guard DDL contains 4 RELATED_TO relationship indexes
- RelationshipTypesMarker includes RELATED_TO
- Health checker expects the new indexes at DEGRADED level
"""

from pathlib import Path

import pytest

GUARD_DDL_PATH = Path(__file__).resolve().parents[2] / (
    "scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher"
)

REQUIRED_INDEXES = [
    "related_to_score_final_idx",
    "related_to_method_idx",
    "related_to_quality_tier_idx",
    "related_to_is_mutual_idx",
]


class TestGuardDDL:
    """Tests for the Neo4j guard DDL file."""

    @pytest.fixture()
    def ddl_content(self) -> str:
        return GUARD_DDL_PATH.read_text()

    def test_guard_ddl_contains_related_to_indexes(self, ddl_content: str):
        """All 4 RELATED_TO relationship indexes must be in the guard DDL."""
        for idx_name in REQUIRED_INDEXES:
            assert idx_name in ddl_content, f"Guard DDL missing index: {idx_name}"

    def test_guard_ddl_uses_if_not_exists(self, ddl_content: str):
        """All RELATED_TO indexes use IF NOT EXISTS for idempotency."""
        for idx_name in REQUIRED_INDEXES:
            # Find the CREATE INDEX line for this index
            for line in ddl_content.split("\n"):
                if idx_name in line and "CREATE INDEX" in line:
                    assert (
                        "IF NOT EXISTS" in line
                    ), f"Index {idx_name} missing IF NOT EXISTS"
                    break

    def test_marker_contains_related_to(self, ddl_content: str):
        """RelationshipTypesMarker must include RELATED_TO in its types array."""
        # Find the MERGE line for the marker
        assert (
            "'RELATED_TO'" in ddl_content
        ), "RelationshipTypesMarker missing RELATED_TO in types array"

    def test_schema_version_v41(self, ddl_content: str):
        """SchemaVersion must be set to v4.1."""
        assert (
            "sv.version = 'v4.1'" in ddl_content
        ), "SchemaVersion not set to v4.1 in guard DDL"


class TestHealthCheckerExpectsIndexes:
    """Tests that the health checker validates RELATED_TO indexes."""

    def test_health_checker_has_gds_indexes(self):
        """Health checker must list the 4 RELATED_TO indexes."""
        health_path = Path(__file__).resolve().parents[2] / "src/monitoring/health.py"
        content = health_path.read_text()
        for idx_name in REQUIRED_INDEXES:
            assert idx_name in content, f"health.py missing GDS index: {idx_name}"

    def test_health_checker_schema_v41(self):
        """Health checker REQUIRED_SCHEMA_VERSION must be v4.1."""
        health_path = Path(__file__).resolve().parents[2] / "src/monitoring/health.py"
        content = health_path.read_text()
        assert (
            'REQUIRED_SCHEMA_VERSION = "v4.1"' in content
        ), "health.py REQUIRED_SCHEMA_VERSION not v4.1"

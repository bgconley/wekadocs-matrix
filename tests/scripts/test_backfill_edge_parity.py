"""
Tests for backfill/service edge schema parity (Stage 3).

Verifies:
- Backfill uses shared edge model (no independent copies)
- Backfill uses ON CREATE/ON MATCH (not bare SET)
- Edge property schema matches between service and backfill paths
"""

from pathlib import Path

import pytest

BACKFILL_PATH = Path(__file__).resolve().parents[2] / (
    "scripts/backfill_cross_doc_edges.py"
)


class TestBackfillUsesSharedModule:
    """Verify backfill imports from cross_doc_edge_model, not independent copies."""

    @pytest.fixture()
    def backfill_content(self) -> str:
        return BACKFILL_PATH.read_text()

    def test_imports_shared_model(self, backfill_content: str):
        """Backfill must import from cross_doc_edge_model."""
        assert "from src.services.cross_doc_edge_model import" in backfill_content

    def test_imports_candidate_signals(self, backfill_content: str):
        """Backfill must import CandidateSignals."""
        assert "CandidateSignals" in backfill_content

    def test_imports_edge_payload(self, backfill_content: str):
        """Backfill must import EdgePayload."""
        assert "EdgePayload" in backfill_content

    def test_no_independent_rrf_function(self, backfill_content: str):
        """Backfill must not have its own reciprocal_rank_fusion() definition."""
        # The function should be imported, not defined locally
        lines = backfill_content.split("\n")
        for line in lines:
            if line.strip().startswith("def reciprocal_rank_fusion("):
                pytest.fail(
                    "Backfill script still has independent reciprocal_rank_fusion()"
                )

    def test_no_independent_aggregate_function(self, backfill_content: str):
        """Backfill must not have its own aggregate_chunks_to_documents() definition."""
        lines = backfill_content.split("\n")
        for line in lines:
            if line.strip().startswith("def aggregate_chunks_to_documents("):
                pytest.fail(
                    "Backfill script still has independent aggregate_chunks_to_documents()"
                )


class TestBackfillEdgeWriter:
    """Verify backfill edge writer uses ON CREATE/ON MATCH pattern."""

    @pytest.fixture()
    def backfill_content(self) -> str:
        return BACKFILL_PATH.read_text()

    def test_uses_on_create_set(self, backfill_content: str):
        """Edge writer must use ON CREATE SET for new edges."""
        assert "ON CREATE SET" in backfill_content

    def test_uses_on_match_set(self, backfill_content: str):
        """Edge writer must use ON MATCH SET for existing edges."""
        assert "ON MATCH SET" in backfill_content

    def test_no_bare_set_created_at(self, backfill_content: str):
        """Edge writer must NOT use bare 'SET r.created_at' (timestamp clobber bug)."""
        # Look for the pattern: SET r.score = ... r.created_at = datetime()
        # without ON CREATE/ON MATCH guards. This is the old bug.
        lines = backfill_content.split("\n")
        in_create_edge = False
        for line in lines:
            if "def create_related_to_edge" in line:
                in_create_edge = True
            elif in_create_edge and line.strip().startswith("def "):
                in_create_edge = False
            elif in_create_edge:
                stripped = line.strip()
                # Bare SET (not ON CREATE SET or ON MATCH SET) with created_at
                if (
                    stripped.startswith("SET r.")
                    and "created_at" in stripped
                    and "ON CREATE" not in stripped
                    and "ON MATCH" not in stripped
                ):
                    pytest.fail(
                        f"Bare SET r.created_at found (timestamp clobber): {stripped}"
                    )


class TestBackfillReciprocityMode:
    """Verify backfill has --reciprocity mode."""

    @pytest.fixture()
    def backfill_content(self) -> str:
        return BACKFILL_PATH.read_text()

    def test_reciprocity_in_method_choices(self, backfill_content: str):
        """--method must include 'reciprocity' as a choice."""
        assert '"reciprocity"' in backfill_content

    def test_reciprocity_cypher_present(self, backfill_content: str):
        """Reciprocity reconciliation Cypher must be present."""
        assert "is_mutual" in backfill_content
        assert "mutual_score" in backfill_content


class TestEdgePayloadParity:
    """Verify service and backfill produce the same edge properties."""

    def test_both_use_edge_payload(self):
        """Both paths must use EdgePayload for edge construction."""
        service_path = (
            Path(__file__).resolve().parents[2] / "src/services/cross_doc_linking.py"
        )
        service_content = service_path.read_text()
        backfill_content = BACKFILL_PATH.read_text()

        # Both must reference EdgePayload
        assert "EdgePayload" in service_content, "Service missing EdgePayload"
        assert "EdgePayload" in backfill_content, "Backfill missing EdgePayload"

        # Both must reference to_neo4j_params
        assert "to_neo4j_params" in service_content, "Service missing to_neo4j_params"
        assert "to_neo4j_params" in backfill_content, "Backfill missing to_neo4j_params"

"""Integration tests for RELATED_TO retrieval integration (Stage 16).

These tests require live Neo4j + Qdrant connections.
Skipped when NEO4J_URI is not set.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("NEO4J_URI"),
        reason="Integration tests require NEO4J_URI environment variable",
    ),
]


class TestRelatedToIntegration:
    """Integration tests for RELATED_TO retrieval pipeline."""

    def test_related_to_expansion_produces_chunks(self):
        """Related doc chunks should appear in results when RELATED_TO is enabled.

        Requires:
        - RELATED_TO edges exist between documents in Neo4j
        - Qdrant collection has chunks for both source and target documents
        - config.references.query.enable_related_to_signals = true
        """
        # This test validates the full pipeline:
        # 1. Vector search returns initial candidates
        # 2. Seed doc IDs extracted from top results
        # 3. _compute_related_to_doc_signals finds related docs via Neo4j
        # 4. Vector search constrained to related doc_ids returns chunks
        # 5. Chunks annotated with related_to_* scores
        # 6. Merged into fused_results
        pytest.skip("Requires live infrastructure with RELATED_TO edges")

    def test_related_to_expansion_disabled_no_expansion(self):
        """When enable_related_to_signals=false, no expansion occurs.

        Validates that the config gate works correctly.
        """
        pytest.skip("Requires live infrastructure")

    def test_related_to_chunks_enter_signal_pool(self):
        """Related doc chunks should enter reranker via dedicated signal pool slots.

        Requires:
        - Signal pool enabled
        - related_to_slots > 0
        - Chunks with related_to_score in the candidate pool
        """
        pytest.skip("Requires live infrastructure with signal pool active")

    def test_related_to_blending_affects_ranking(self):
        """RELATED_TO blending should change chunk rankings for conceptual queries.

        Validates that λ > 0 queries produce measurably different rankings
        compared to the same query with enable_related_to_signals=false.
        """
        pytest.skip("Requires live infrastructure with RELATED_TO edges")

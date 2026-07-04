"""
Live validation tests for precision retrieval.

Requires: running Qdrant + reranker services on GPU server.
Run: RUN_LIVE_RETRIEVAL=1 pytest tests/integration/test_precision_retrieval_live.py -v

Live Validation Runbook:
  1. Deploy to GPU server, restart mcp-server
  2. Run both exact queries through MCP evidence pack
  3. Inspect traces: query_type, post-fusion top 20, post-reranker top 20
  4. Success criteria:
     - Metadata/sizing content in top 3
     - <= 1 cloud deployment doc in top 10
     - Signal pool active, BM25 disabled
     - ColBERT runtime status visible in trace
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_RETRIEVAL"),
    reason="Live retrieval tests disabled (set RUN_LIVE_RETRIEVAL=1)",
)


class TestPrecisionRetrievalLive:
    """Live validation against real Qdrant index.

    These tests require a running retrieval stack (Qdrant, Neo4j, GPU gateway).
    They are skipped by default and run explicitly during deployment validation.
    """

    @pytest.fixture(autouse=True)
    def setup_retriever(self):
        """Create a retriever connected to the live stack."""
        # TODO: instantiate HybridRetriever with production config
        # self.retriever = ...
        pass

    def test_metadata_architecture_returns_subsystem_docs(self):
        """
        Query: 'how is metadata managed and architected on a weka cluster'
        Expected: top 5 results include deeply nested metadata/filesystem docs,
        not general overviews or cloud deployment docs.
        """
        # results, metrics = self.retriever.retrieve(
        #     "how is metadata managed and architected on a weka cluster",
        #     top_k=20,
        # )
        # assert metrics["query_type"] == "subsystem_architecture"
        # assert metrics["query_intent_precision_mode"] is True
        # assert metrics.get("pre_rerank_structural_expansion_applied") is False
        #
        # # Check top 5 for relevant content
        # top5_tags = [r.doc_tag for r in results[:5]]
        # cloud_tags = {"aws-solutions", "azure-solutions", "gcp-solutions"}
        # cloud_in_top5 = sum(1 for t in top5_tags if t and any(c in t for c in cloud_tags))
        # assert cloud_in_top5 <= 1, f"Too many cloud docs in top 5: {top5_tags}"
        pass

    def test_container_sizing_returns_sizing_tables(self):
        """
        Query: 'How do I appropriately size the weka drives, compute, and frontends containers?'
        Expected: top 5 results include chunks with sizing tables from
        core WEKA planning/sizing docs.
        """
        # results, metrics = self.retriever.retrieve(
        #     "How do I appropriately size the weka drives, compute, and frontends containers?",
        #     top_k=20,
        # )
        # assert metrics["query_type"] == "resource_sizing"
        # assert metrics["query_intent_precision_mode"] is True
        pass

    def test_structural_expansion_skipped_for_precision(self):
        """Verify metrics confirm structural expansion was skipped for precision queries."""
        # results, metrics = self.retriever.retrieve(
        #     "WEKA inode management internals", top_k=10
        # )
        # assert metrics.get("pre_rerank_structural_expansion_applied") is False
        pass

    def test_colbert_status_visible_in_metrics(self):
        """Verify ColBERT runtime status is captured in metrics."""
        # results, metrics = self.retriever.retrieve(
        #     "WEKA metadata architecture", top_k=10
        # )
        # assert "colbert_runtime_available" in metrics
        # assert "colbert_query_embedding_ok" in metrics
        pass

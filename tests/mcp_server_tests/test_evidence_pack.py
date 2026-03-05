# =============================================================================
# @status: ACTIVE
# @tests: mcp_app.py evidence pack (kb.retrieve_evidence)
# =============================================================================
"""
Tests for the enriched evidence pack: retrieval-score-based ranking,
decoupled retrieval depth, coverage metadata, and enriched quote schema.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from src.mcp_server.mcp_app import (
    KB_EVIDENCE_INTERNAL_FETCH_K,
    KB_EVIDENCE_MAX_FETCH_K,
    _extract_evidence_from_passages,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_scratch_entry(
    section_id: str = "chunk_1",
    doc_tag: str = "weka_docs/4.3/admin/s3",
    title: str = "Bucket Settings",
    text: str = "Set the IAM role ARN in the weka_s3_iam_role field for S3 access control.",
    rerank_score: Optional[float] = 0.95,
    fused_score: Optional[float] = 0.7,
    vector_score: Optional[float] = 0.65,
    bm25_score: Optional[float] = None,
    parent_path_norm: Optional[str] = "Configuration > S3 Backend > Bucket Settings",
    source: str = "reranked",
    **overrides: Any,
) -> Dict[str, Any]:
    entry = {
        "section_id": section_id,
        "doc_tag": doc_tag,
        "title": title,
        "text": text,
        "source_uri": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rerank_score": rerank_score,
        "fused_score": fused_score,
        "vector_score": vector_score,
        "bm25_score": bm25_score,
        "graph_score": None,
        "parent_path_norm": parent_path_norm,
        "rerank_rank": None,
        "fusion_method": "rrf",
        "is_expanded": False,
        "expansion_source": None,
        "source": source,
    }
    entry.update(overrides)
    return entry


class FakeScratch:
    """In-memory scratch store for testing."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    async def put(self, session: str, pid: str, payload: dict) -> int:
        self._store[pid] = payload
        return len(str(payload))

    async def get(self, session: str, pid: str) -> Optional[dict]:
        return self._store.get(pid)

    @staticmethod
    def build_uri(session: str, pid: str) -> str:
        return f"wekadocs://scratch/{session}/{pid}"


class FakeDeps:
    def __init__(self, scratch: FakeScratch) -> None:
        self.scratch = scratch
        self.query = None
        self.graph = None
        self.text = None
        self.summarizer = None
        self.assembler = None


# ── Tests: Evidence extraction with retrieval scores ─────────────────


class TestEvidenceExtractionScoring:
    """Verify that evidence extraction uses retrieval scores, not just keyword overlap."""

    @pytest.fixture
    def scratch(self) -> FakeScratch:
        return FakeScratch()

    @pytest.fixture
    def deps(self, scratch: FakeScratch) -> FakeDeps:
        return FakeDeps(scratch)

    @pytest.mark.asyncio
    async def test_high_rerank_score_beats_high_keyword_overlap(self, scratch, deps):
        """A passage with high rerank_score but low keyword overlap should rank
        above a passage with low rerank_score but high keyword overlap."""
        await scratch.put(
            "s",
            "p1",
            _make_scratch_entry(
                section_id="chunk_rerank_high",
                text="Configure the ARN identifier in the IAM settings panel for object storage.",
                rerank_score=0.95,
                fused_score=0.8,
            ),
        )
        await scratch.put(
            "s",
            "p2",
            _make_scratch_entry(
                section_id="chunk_keyword_high",
                text="S3 bucket configuration S3 bucket S3 bucket settings S3 access S3 IAM role.",
                rerank_score=0.3,
                fused_score=0.4,
            ),
        )

        quotes = await _extract_evidence_from_passages(
            question="S3 bucket IAM role configuration",
            passage_ids=["p1", "p2"],
            max_quotes=2,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert len(quotes) == 2
        assert quotes[0]["section_id"] == "chunk_rerank_high"
        assert quotes[0]["confidence"] > quotes[1]["confidence"]

    @pytest.mark.asyncio
    async def test_enriched_quote_fields(self, scratch, deps):
        """Quotes should include doc_tag, parent_path, source, and rank."""
        await scratch.put(
            "s",
            "p1",
            _make_scratch_entry(
                doc_tag="weka_docs/4.3/admin/s3",
                parent_path_norm="Configuration > S3 Backend",
                source="reranked",
            ),
        )

        quotes = await _extract_evidence_from_passages(
            question="how to configure S3",
            passage_ids=["p1"],
            max_quotes=1,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert len(quotes) == 1
        q = quotes[0]
        assert q["doc_tag"] == "weka_docs/4.3/admin/s3"
        assert q["parent_path"] == "Configuration > S3 Backend"
        assert q["source"] == "reranked"
        assert q["rank"] == 1
        assert "confidence" in q
        assert "quote" in q
        assert "passage_id" in q

    @pytest.mark.asyncio
    async def test_fallback_to_fused_score_when_no_rerank(self, scratch, deps):
        """When rerank_score is None, fused_score should be used for ranking."""
        await scratch.put(
            "s",
            "p1",
            _make_scratch_entry(
                section_id="high_fused",
                rerank_score=None,
                fused_score=0.9,
            ),
        )
        await scratch.put(
            "s",
            "p2",
            _make_scratch_entry(
                section_id="low_fused",
                rerank_score=None,
                fused_score=0.3,
            ),
        )

        quotes = await _extract_evidence_from_passages(
            question="test query",
            passage_ids=["p1", "p2"],
            max_quotes=2,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert quotes[0]["section_id"] == "high_fused"

    @pytest.mark.asyncio
    async def test_max_quotes_limits_output(self, scratch, deps):
        """Output should be capped at max_quotes even with more passages."""
        for i in range(10):
            await scratch.put(
                "s",
                f"p{i}",
                _make_scratch_entry(
                    section_id=f"chunk_{i}",
                    rerank_score=0.9 - (i * 0.05),
                ),
            )

        quotes = await _extract_evidence_from_passages(
            question="test query",
            passage_ids=[f"p{i}" for i in range(10)],
            max_quotes=3,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert len(quotes) == 3

    @pytest.mark.asyncio
    async def test_max_quotes_supports_more_than_12(self, scratch, deps):
        """Evidence extraction should not hard-clamp to 12 quotes."""
        for i in range(15):
            await scratch.put(
                "s",
                f"p{i}",
                _make_scratch_entry(
                    section_id=f"chunk_{i}",
                    rerank_score=0.99 - (i * 0.01),
                    source="reranked",
                ),
            )

        quotes = await _extract_evidence_from_passages(
            question="weka metadata architecture",
            passage_ids=[f"p{i}" for i in range(15)],
            max_quotes=20,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert len(quotes) == 15

    @pytest.mark.asyncio
    async def test_graph_expanded_only_backfills_after_primary(self, scratch, deps):
        """Graph-expanded passages should never displace strong primary hits."""
        await scratch.put(
            "s",
            "p_primary_1",
            _make_scratch_entry(
                section_id="primary_1",
                title="WEKA metadata architecture overview",
                text="Metadata services coordinate inode updates across the cluster.",
                rerank_score=0.91,
                fused_score=0.88,
                source="reranked",
            ),
        )
        await scratch.put(
            "s",
            "p_primary_2",
            _make_scratch_entry(
                section_id="primary_2",
                title="Cluster architecture internals",
                text="Control-plane metadata components synchronize allocation state.",
                rerank_score=None,
                fused_score=0.44,
                source="rrf_fusion",
            ),
        )
        await scratch.put(
            "s",
            "p_graph_1",
            _make_scratch_entry(
                section_id="graph_1",
                title="Neighbor chunk from graph",
                text="Sibling content that should only appear as backfill.",
                rerank_score=None,
                fused_score=0.95,
                source="graph_expanded",
            ),
        )
        await scratch.put(
            "s",
            "p_graph_2",
            _make_scratch_entry(
                section_id="graph_2",
                title="Another graph neighbor",
                text="Additional structural context candidate.",
                rerank_score=None,
                fused_score=0.80,
                source="graph_expanded",
            ),
        )

        quotes = await _extract_evidence_from_passages(
            question="how is metadata managed and architected on a weka cluster",
            passage_ids=["p_graph_1", "p_primary_2", "p_graph_2", "p_primary_1"],
            max_quotes=4,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert len(quotes) == 4
        assert quotes[0]["source"] != "graph_expanded"
        assert quotes[1]["source"] != "graph_expanded"

    @pytest.mark.asyncio
    async def test_metadata_limitations_forced_before_graph_backfill(
        self, scratch, deps
    ):
        """Metadata-limitations reranked evidence should be force-kept for the known failure query."""
        await scratch.put(
            "s",
            "p_overview_1",
            _make_scratch_entry(
                section_id="overview_1",
                title="WEKA cluster architecture overview",
                text="General architecture overview of the cluster.",
                rerank_score=0.95,
                source="reranked",
            ),
        )
        await scratch.put(
            "s",
            "p_overview_2",
            _make_scratch_entry(
                section_id="overview_2",
                title="Cluster metadata services summary",
                text="General summary content without deep metadata detail.",
                rerank_score=0.85,
                source="reranked",
            ),
        )
        await scratch.put(
            "s",
            "p_metadata_limits",
            _make_scratch_entry(
                section_id="metadata_limits",
                title="Metadata limitations in WEKA filesystems",
                text="Metadata limitations and inode management constraints are documented here.",
                rerank_score=0.20,
                source="reranked",
            ),
        )
        await scratch.put(
            "s",
            "p_graph",
            _make_scratch_entry(
                section_id="graph_neighbor",
                title="Graph-expanded neighboring chunk",
                text="Neighbor context that should only backfill.",
                rerank_score=None,
                fused_score=0.30,
                source="graph_expanded",
            ),
        )

        quotes = await _extract_evidence_from_passages(
            question="how is metadata managed and architected on a weka cluster",
            passage_ids=[
                "p_overview_1",
                "p_overview_2",
                "p_metadata_limits",
                "p_graph",
            ],
            max_quotes=2,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )

        assert len(quotes) == 2
        assert any(
            "metadata limitations" in (q.get("title") or "").lower() for q in quotes
        )
        assert all(q["source"] != "graph_expanded" for q in quotes[:2])

    @pytest.mark.asyncio
    async def test_empty_passage_ids_returns_empty(self, scratch, deps):
        quotes = await _extract_evidence_from_passages(
            question="test",
            passage_ids=[],
            max_quotes=6,
            max_quote_tokens=80,
            include_context_tokens=10,
            deps=deps,
            effective_session="s",
        )
        assert quotes == []


# ── Tests: Constants ─────────────────────────────────────────────────


class TestEvidencePackConstants:
    def test_internal_fetch_k_default(self):
        assert KB_EVIDENCE_INTERNAL_FETCH_K == 60

    def test_max_fetch_k_cap(self):
        assert KB_EVIDENCE_MAX_FETCH_K == 150
        assert KB_EVIDENCE_MAX_FETCH_K > KB_EVIDENCE_INTERNAL_FETCH_K

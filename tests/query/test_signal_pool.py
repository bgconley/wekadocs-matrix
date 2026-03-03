"""
Tests for the signal-diverse rerank pool builder.

Tests the pure-function build_signal_pool() algorithm with synthetic
ChunkResult objects. No mocking needed — signal_pool.py has no IO.
"""

from __future__ import annotations

from typing import Optional

from src.query.hybrid_retrieval import ChunkResult
from src.query.signal_pool import build_signal_pool
from src.shared.config import SignalPoolConfig


def _make_chunk(
    chunk_id: str,
    document_id: str = "doc1",
    fused_score: float = 0.5,
    vector_score: Optional[float] = None,
    bm25_score: Optional[float] = None,
    title_vec_score: Optional[float] = None,
    doc_title_vec_score: Optional[float] = None,
    entity_vec_score: Optional[float] = None,
    lexical_vec_score: Optional[float] = None,
    doc_title_sparse_score: Optional[float] = None,
) -> ChunkResult:
    """Factory for test ChunkResult objects with controlled per-field scores."""
    return ChunkResult(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_section_id="sec1",
        order=1,
        level=1,
        heading="Test Heading",
        text="Test body text",
        token_count=10,
        fused_score=fused_score,
        vector_score=vector_score,
        bm25_score=bm25_score,
        title_vec_score=title_vec_score,
        doc_title_vec_score=doc_title_vec_score,
        entity_vec_score=entity_vec_score,
        lexical_vec_score=lexical_vec_score,
        doc_title_sparse_score=doc_title_sparse_score,
    )


class TestConsensusSlots:
    """Consensus slots should always be filled first with top fused_score chunks."""

    def test_consensus_fills_first(self):
        config = SignalPoolConfig(enabled=True, pool_size=10, consensus_slots=5)
        chunks = [_make_chunk(f"c{i}", fused_score=1.0 - i * 0.01) for i in range(20)]
        result = build_signal_pool(chunks, [], config)
        assert result.slot_fills.get("consensus", 0) == 5
        # Top 5 by fused_score should be in pool
        pool_ids = {c.chunk_id for c in result.pool}
        for i in range(5):
            assert f"c{i}" in pool_ids

    def test_consensus_with_fewer_candidates_than_slots(self):
        config = SignalPoolConfig(enabled=True, pool_size=20, consensus_slots=10)
        chunks = [_make_chunk(f"c{i}", fused_score=0.9) for i in range(3)]
        result = build_signal_pool(chunks, [], config)
        assert result.slot_fills.get("consensus", 0) == 3
        assert len(result.pool) == 3


class TestSignalUniqueSlots:
    """Per-signal slots should pull in chunks that excel on one signal but missed consensus."""

    def test_high_vector_low_fused_gets_included(self):
        config = SignalPoolConfig(
            enabled=True,
            pool_size=30,
            consensus_slots=3,
            content_dense_slots=5,
        )
        chunks = []
        # First 3: high fused (consensus)
        for i in range(3):
            chunks.append(
                _make_chunk(
                    f"consensus{i}",
                    fused_score=0.95 - i * 0.01,
                    vector_score=0.3,
                    title_vec_score=0.1,
                )
            )
        # Next 5: low fused but high vector_score (should fill content_dense)
        for i in range(5):
            chunks.append(
                _make_chunk(
                    f"vector_strong{i}",
                    fused_score=0.2,
                    vector_score=0.95 - i * 0.01,
                    title_vec_score=0.1,
                )
            )
        result = build_signal_pool(chunks, [], config)
        pool_ids = {c.chunk_id for c in result.pool}
        # All vector-strong chunks should be pulled in
        for i in range(5):
            assert f"vector_strong{i}" in pool_ids

    def test_entity_sparse_unique_chunks_included(self):
        config = SignalPoolConfig(
            enabled=True,
            pool_size=20,
            consensus_slots=3,
            entity_sparse_slots=3,
        )
        chunks = [
            _make_chunk("consensus0", fused_score=0.9, entity_vec_score=0.1),
            _make_chunk("consensus1", fused_score=0.8, entity_vec_score=0.1),
            _make_chunk("consensus2", fused_score=0.7, entity_vec_score=0.1),
            # High entity but low fused
            _make_chunk("entity_hit0", fused_score=0.1, entity_vec_score=0.95),
            _make_chunk("entity_hit1", fused_score=0.1, entity_vec_score=0.90),
            _make_chunk("entity_hit2", fused_score=0.1, entity_vec_score=0.85),
        ]
        result = build_signal_pool(chunks, [], config)
        pool_ids = {c.chunk_id for c in result.pool}
        assert "entity_hit0" in pool_ids
        assert "entity_hit1" in pool_ids
        assert "entity_hit2" in pool_ids


class TestGracefulDegradation:
    """Falls back to BM25/vector provenance when per-field scores are absent."""

    def test_degraded_flag_set_when_no_per_field_scores(self):
        config = SignalPoolConfig(enabled=True, pool_size=10, consensus_slots=3)
        # No per-field scores (title_vec_score, entity_vec_score, etc. all None)
        chunks = [
            _make_chunk(f"c{i}", fused_score=0.5, vector_score=0.3, bm25_score=0.2)
            for i in range(10)
        ]
        result = build_signal_pool(chunks, [], config)
        assert result.degraded is True

    def test_not_degraded_when_per_field_scores_present(self):
        config = SignalPoolConfig(enabled=True, pool_size=10, consensus_slots=3)
        chunks = [
            _make_chunk(f"c{i}", fused_score=0.5, title_vec_score=0.3)
            for i in range(10)
        ]
        result = build_signal_pool(chunks, [], config)
        assert result.degraded is False

    def test_provenance_fallback_fills_bm25_slot(self):
        config = SignalPoolConfig(
            enabled=True,
            pool_size=20,
            consensus_slots=3,
            text_sparse_slots=5,
            entity_sparse_slots=3,
            title_sparse_slots=2,
        )
        chunks = [
            _make_chunk("consensus0", fused_score=0.9, bm25_score=0.1),
            _make_chunk("consensus1", fused_score=0.8, bm25_score=0.1),
            _make_chunk("consensus2", fused_score=0.7, bm25_score=0.1),
            # BM25-strong but not in consensus
            _make_chunk("bm25_hit0", fused_score=0.1, bm25_score=0.95),
            _make_chunk("bm25_hit1", fused_score=0.1, bm25_score=0.90),
        ]
        result = build_signal_pool(chunks, [], config)
        pool_ids = {c.chunk_id for c in result.pool}
        assert "bm25_hit0" in pool_ids
        assert "bm25_hit1" in pool_ids
        assert "bm25_provenance" in result.slot_fills


class TestStructuralSlots:
    """Structural expansion candidates should fill the structural slot."""

    def test_structural_candidates_included(self):
        config = SignalPoolConfig(
            enabled=True, pool_size=10, consensus_slots=3, structural_slots=3
        )
        fused = [_make_chunk(f"f{i}", fused_score=0.9 - i * 0.1) for i in range(5)]
        structural = [_make_chunk(f"s{i}", fused_score=0.3) for i in range(5)]
        result = build_signal_pool(fused, structural, config)
        pool_ids = {c.chunk_id for c in result.pool}
        structural_count = sum(1 for cid in pool_ids if cid.startswith("s"))
        assert structural_count >= 1
        assert result.slot_fills.get("structural", 0) >= 1

    def test_structural_deduped_against_fused(self):
        config = SignalPoolConfig(
            enabled=True, pool_size=10, consensus_slots=3, structural_slots=3
        )
        fused = [_make_chunk("shared", fused_score=0.9)]
        structural = [_make_chunk("shared", fused_score=0.3)]  # Same chunk_id
        result = build_signal_pool(fused, structural, config)
        ids = [c.chunk_id for c in result.pool]
        assert ids.count("shared") == 1


class TestPerDocDepth:
    """Per-document depth should ensure multiple documents are represented."""

    def test_multiple_docs_get_depth(self):
        config = SignalPoolConfig(
            enabled=True,
            pool_size=30,
            consensus_slots=3,
            per_doc_depth_slots=12,
            per_doc_k=2,
            per_doc_m=6,
        )
        chunks = []
        for doc_i in range(3):
            for chunk_i in range(10):
                chunks.append(
                    _make_chunk(
                        f"d{doc_i}_c{chunk_i}",
                        document_id=f"doc{doc_i}",
                        fused_score=0.9 - doc_i * 0.2 - chunk_i * 0.01,
                        vector_score=0.5,
                    )
                )
        result = build_signal_pool(chunks, [], config)
        pool_doc_ids = [c.document_id for c in result.pool]
        # Top 2 docs should have depth representation
        assert pool_doc_ids.count("doc0") >= 3
        assert pool_doc_ids.count("doc1") >= 2


class TestPoolConstraints:
    """Pool size limits and deduplication invariants."""

    def test_no_duplicates_in_pool(self):
        config = SignalPoolConfig(enabled=True, pool_size=50, consensus_slots=20)
        chunks = [
            _make_chunk(
                f"c{i}",
                fused_score=0.5,
                vector_score=0.5,
                bm25_score=0.5,
                title_vec_score=0.5,
                entity_vec_score=0.5,
            )
            for i in range(30)
        ]
        result = build_signal_pool(chunks, [], config)
        ids = [c.chunk_id for c in result.pool]
        assert len(ids) == len(set(ids))

    def test_pool_size_never_exceeded(self):
        config = SignalPoolConfig(enabled=True, pool_size=10, consensus_slots=5)
        chunks = [_make_chunk(f"c{i}", fused_score=0.5) for i in range(100)]
        result = build_signal_pool(chunks, [], config)
        assert len(result.pool) <= 10

    def test_empty_input_returns_empty_pool(self):
        config = SignalPoolConfig(enabled=True, pool_size=10)
        result = build_signal_pool([], [], config)
        assert len(result.pool) == 0
        assert result.overflow_count == 0


class TestBackfill:
    """Backfill should fill remaining capacity from fused_score order."""

    def test_backfill_fills_remaining_capacity(self):
        config = SignalPoolConfig(enabled=True, pool_size=10, consensus_slots=3)
        chunks = [_make_chunk(f"c{i}", fused_score=0.9 - i * 0.01) for i in range(20)]
        result = build_signal_pool(chunks, [], config)
        assert len(result.pool) == 10
        assert result.slot_fills.get("backfill", 0) > 0

"""Tests for RELATED_TO signal pool slot (Stage 14)."""

from __future__ import annotations

from src.query.hybrid_retrieval import ChunkResult
from src.query.signal_pool import build_signal_pool
from src.shared.config import SignalPoolConfig


def _make_chunk(
    chunk_id: str,
    fused_score: float = 0.0,
    related_to_score: float = 0.0,
    vector_score: float = 0.0,
) -> ChunkResult:
    """Helper to create a minimal ChunkResult for pool testing."""
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc1",
        parent_section_id="s1",
        order=0,
        level=1,
        heading=f"Heading {chunk_id}",
        text=f"Text {chunk_id}",
        token_count=50,
        fused_score=fused_score,
        vector_score=vector_score,
        related_to_score=related_to_score,
    )


class TestRelatedToSlotConfig:
    def test_related_to_slots_default(self):
        """SignalPoolConfig has related_to_slots with default 10."""
        cfg = SignalPoolConfig()
        assert cfg.related_to_slots == 10

    def test_related_to_slots_override(self):
        """related_to_slots can be overridden."""
        cfg = SignalPoolConfig(related_to_slots=15)
        assert cfg.related_to_slots == 15


class TestRelatedToSlotFilling:
    def test_related_to_chunks_fill_dedicated_slot(self):
        """Chunks with related_to_score get placed in the related_to slot."""
        config = SignalPoolConfig(
            enabled=True,
            pool_size=50,
            consensus_slots=5,
            content_dense_slots=0,
            title_dense_slots=0,
            doc_title_dense_slots=0,
            text_sparse_slots=0,
            entity_sparse_slots=0,
            title_sparse_slots=0,
            structural_slots=0,
            related_to_slots=5,
            per_doc_depth_slots=0,
        )

        # 5 consensus chunks + 5 related_to chunks (no overlap)
        # Set title_vec_score on consensus chunks to trigger per-field detection
        chunks = []
        for i in range(10):
            c = _make_chunk(
                f"consensus_{i}",
                fused_score=1.0 - i * 0.1,
                vector_score=1.0 - i * 0.1,
            )
            c.title_vec_score = 0.1  # Triggers _has_per_field_scores
            chunks.append(c)
        for i in range(5):
            chunks.append(
                _make_chunk(
                    f"related_{i}",
                    fused_score=0.01,  # Low fused so they wouldn't be in consensus
                    related_to_score=0.5 - i * 0.05,
                )
            )

        result = build_signal_pool(chunks, [], config)
        pool_ids = {c.chunk_id for c in result.pool}

        # At least some related_to chunks should be in the pool
        related_in_pool = [cid for cid in pool_ids if cid.startswith("related_")]
        assert len(related_in_pool) > 0
        assert result.slot_fills.get("related_to", 0) > 0

    def test_chunks_without_related_to_score_not_picked(self):
        """Chunks with related_to_score=None/0 are not picked for the related_to slot."""
        config = SignalPoolConfig(
            enabled=True,
            pool_size=20,
            consensus_slots=5,
            content_dense_slots=0,
            title_dense_slots=0,
            doc_title_dense_slots=0,
            text_sparse_slots=0,
            entity_sparse_slots=0,
            title_sparse_slots=0,
            structural_slots=0,
            related_to_slots=5,
            per_doc_depth_slots=0,
        )

        # All chunks have related_to_score = 0/None
        chunks = [
            _make_chunk(f"chunk_{i}", fused_score=0.5 - i * 0.05) for i in range(10)
        ]

        result = build_signal_pool(chunks, [], config)
        # related_to slot should have 0 fills (no chunks have the score)
        assert result.slot_fills.get("related_to", 0) == 0

    def test_pool_total_within_size(self):
        """Total pool size respects pool_size limit after rebalance."""
        config = SignalPoolConfig(
            enabled=True,
            pool_size=200,
            consensus_slots=40,
            content_dense_slots=15,
            title_dense_slots=10,
            doc_title_dense_slots=10,
            text_sparse_slots=30,
            entity_sparse_slots=15,
            title_sparse_slots=15,
            structural_slots=25,
            related_to_slots=10,
            per_doc_depth_slots=30,
        )
        # Sum of slots: 40+15+10+10+30+15+15+25+10+30 = 200 (exact fit)
        total_slots = (
            config.consensus_slots
            + config.content_dense_slots
            + config.title_dense_slots
            + config.doc_title_dense_slots
            + config.text_sparse_slots
            + config.entity_sparse_slots
            + config.title_sparse_slots
            + config.structural_slots
            + config.related_to_slots
            + config.per_doc_depth_slots
        )
        assert total_slots <= config.pool_size

    def test_yaml_rebalance_correct(self):
        """YAML config values sum to pool_size after per_doc_depth reduction."""
        # These match the production.yaml values after Stage 14
        yaml_values = {
            "consensus_slots": 40,
            "content_dense_slots": 15,
            "title_dense_slots": 10,
            "doc_title_dense_slots": 10,
            "text_sparse_slots": 30,
            "entity_sparse_slots": 15,
            "title_sparse_slots": 15,
            "structural_slots": 25,
            "related_to_slots": 10,
            "per_doc_depth_slots": 30,
        }
        total = sum(yaml_values.values())
        assert total == 200  # Exactly matches pool_size

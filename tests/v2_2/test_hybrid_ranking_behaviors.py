"""Focused unit tests for hybrid ranking helpers."""

from src.query.hybrid_retrieval import ChunkResult, HybridRetriever


def _make_retriever():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.context_max_tokens = 4500
    retriever.context_group_cap = 3
    retriever.vector_field_weights = {}
    return retriever


def _chunk(chunk_id: str, doc_id: str, fused: float) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        parent_section_id=f"parent-{chunk_id}",
        order=0,
        level=2,
        heading="Heading",
        text="Example text",
        token_count=50,
        fused_score=fused,
        citation_labels=[(0, "Heading", 2)],
    )


def test_doc_continuity_boost_prefers_dominant_document():
    retriever = _make_retriever()
    chunks = [
        _chunk("a1", "docA", 0.42),
        _chunk("a2", "docA", 0.38),
        _chunk("b1", "docB", 0.44),
    ]

    baseline = chunks[0].fused_score
    boosted = retriever._apply_doc_continuity_boost([c for c in chunks], alpha=0.2)

    assert boosted[0].document_id == "docA"
    assert boosted[0].fused_score > baseline
    docb = next(c for c in boosted if c.document_id == "docB")
    assert boosted[0].fused_score >= docb.fused_score


def test_neighbor_score_is_capped_below_source():
    retriever = _make_retriever()
    score = retriever._neighbor_score(1.0)
    assert 0.0 <= score <= 0.5
    assert score < 1.0

    zero_score = retriever._neighbor_score(0.0)
    assert zero_score == 0.0

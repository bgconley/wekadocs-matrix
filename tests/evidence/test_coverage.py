from src.evidence.coverage import build_coverage, identify_gaps, mark_budget_state
from src.evidence.models import EvidenceCoverage, EvidenceQuote


def test_build_coverage_counts_unique_docs_and_flags():
    coverage = build_coverage(
        search_results=[
            {"doc_tag": "nci/aos"},
            {"doc_tag": "nci/aos"},
            {"doc_tag": "ncm/ops"},
            {"doc_tag": None},
        ],
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nci/aos",
                text="e",
                confidence=0.8,
            )
        ],
        metrics={"reranker_applied": True, "signal_pool_used": True},
        graph_expansion_applied=True,
    )

    assert coverage.documents_searched == 2
    assert coverage.documents_with_evidence == 1
    assert coverage.retrieval_depth == 4
    assert coverage.reranker_applied is True
    assert coverage.signal_pool_active is True
    assert coverage.graph_expansion_applied is True
    assert coverage.partial is False
    assert coverage.limit_reason == "none"


def test_build_coverage_uses_signal_pool_enabled_fallback():
    coverage = build_coverage(
        search_results=[],
        quotes=[],
        metrics={"signal_pool_enabled": True},
        graph_expansion_applied=False,
    )

    assert coverage.signal_pool_active is True


def test_identify_gaps_flags_empty_and_missing_live_validation():
    gaps = identify_gaps(
        quotes=[],
        documents_searched=4,
        live_validation_available=False,
    )

    assert [g.kind for g in gaps] == [
        "insufficient_evidence",
        "missing_live_validation",
    ]
    assert "4 searched documents" in gaps[0].message
    assert gaps[1].severity == "info"


def test_identify_gaps_omits_empty_gap_when_quotes_exist():
    gaps = identify_gaps(
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                text="e",
                confidence=0.8,
            )
        ],
        documents_searched=1,
        live_validation_available=True,
    )

    assert gaps == []


def test_mark_budget_state_sets_coverage_and_appends_gap():
    coverage = EvidenceCoverage(
        documents_searched=2,
        documents_with_evidence=1,
        retrieval_depth=10,
        reranker_applied=True,
        signal_pool_active=True,
        graph_expansion_applied=False,
    )
    gaps = []

    mark_budget_state(coverage, gaps, partial=True, limit_reason="token_cap")

    assert coverage.partial is True
    assert coverage.limit_reason == "token_cap"
    assert gaps[-1].kind == "budget_exceeded"
    assert "token_cap" in gaps[-1].message


def test_mark_budget_state_clears_limit_reason_when_not_partial():
    coverage = EvidenceCoverage(
        documents_searched=2,
        documents_with_evidence=1,
        retrieval_depth=10,
        reranker_applied=True,
        signal_pool_active=True,
        graph_expansion_applied=False,
        partial=True,
        limit_reason="token_cap",
    )
    gaps = []

    mark_budget_state(coverage, gaps, partial=False, limit_reason="token_cap")

    assert coverage.partial is False
    assert coverage.limit_reason == "none"
    assert gaps == []

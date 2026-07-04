import pytest
from pydantic import ValidationError

from src.evidence.models import (
    EvidenceAnswerDraft,
    EvidenceClaim,
    EvidenceCoverage,
    EvidenceGap,
    EvidencePackage,
    EvidenceQuote,
    EvidenceRequest,
)


def _coverage(**over):
    base = dict(
        documents_searched=1,
        documents_with_evidence=1,
        retrieval_depth=20,
        reranker_applied=True,
        signal_pool_active=True,
        graph_expansion_applied=False,
        partial=False,
        limit_reason="none",
    )
    base.update(over)
    return EvidenceCoverage(**base)


def test_request_defaults_are_the_transport_resolved_contract():
    request = EvidenceRequest(question="How does NC2 work?", session_id="s1")

    assert request.top_k == 5
    assert request.max_quotes == 6
    assert request.max_quote_tokens == 80
    assert request.include_context_tokens == 20
    assert request.retrieval_depth == 60
    assert request.graph_enrichment is False
    assert request.scope is None
    assert request.filters is None
    assert request.options == {}
    assert request.response_mode == "evidence_only"


def test_quote_coverage_and_gap_dump_match_contract_shape():
    quote = EvidenceQuote(
        quote_id="q_0001",
        rank=1,
        passage_id="p1",
        section_id="s1",
        doc_tag="nci/aos",
        title="AOS Storage",
        parent_path=["NCI", "AOS Storage"],
        source_uri="nutanixdocs://scratch/s1/p1",
        text="AOS provides distributed storage.",
        confidence=0.81,
        score=0.9,
        retrieval_signals={"score": 0.9},
    )
    coverage = _coverage()
    gap = EvidenceGap(kind="missing_live_validation", message="Live pass deferred.")

    quote_dump = quote.model_dump()
    assert quote_dump == {
        "quote_id": "q_0001",
        "rank": 1,
        "passage_id": "p1",
        "section_id": "s1",
        "doc_tag": "nci/aos",
        "title": "AOS Storage",
        "parent_path": ["NCI", "AOS Storage"],
        "source_uri": "nutanixdocs://scratch/s1/p1",
        "text": "AOS provides distributed storage.",
        "confidence": 0.81,
        "score": 0.9,
        "source": "retrieval",
        "retrieval_signals": {"score": 0.9},
        "context_before": None,
        "context_after": None,
    }
    assert coverage.model_dump() == {
        "documents_searched": 1,
        "documents_with_evidence": 1,
        "retrieval_depth": 20,
        "reranker_applied": True,
        "signal_pool_active": True,
        "graph_expansion_applied": False,
        "partial": False,
        "limit_reason": "none",
    }
    assert gap.model_dump() == {
        "kind": "missing_live_validation",
        "message": "Live pass deferred.",
        "severity": "warning",
        "quote_ids": [],
    }


def test_package_round_trips_and_carries_internal_telemetry_without_diagnostic_leak():
    package = EvidencePackage(
        request=EvidenceRequest(question="q", session_id="s"),
        normalized_query="q",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                text="t",
                confidence=0.8,
            )
        ],
        coverage=_coverage(),
        retrieval_metrics={"reranker_model": "qwen3", "signal_pool_size": 30},
        diagnostic_context={"private": "adapter-only"},
    )
    data = package.model_dump()
    assert data["package_id"].startswith("ep_")
    assert data["request"]["response_mode"] == "evidence_only"
    assert data["retrieval_metrics"]["reranker_model"] == "qwen3"
    assert "diagnostic_context" not in data


def test_evidence_only_forbids_answer_draft():
    with pytest.raises(ValidationError):
        EvidencePackage(
            request=EvidenceRequest(
                question="q", session_id="s", response_mode="evidence_only"
            ),
            normalized_query="q",
            quotes=[
                EvidenceQuote(
                    quote_id="q_0001",
                    rank=1,
                    passage_id="p1",
                    text="t",
                    confidence=0.8,
                )
            ],
            coverage=_coverage(),
            answer_draft=EvidenceAnswerDraft(
                markdown="t [q_0001]",
                claims=[
                    EvidenceClaim(
                        claim_id="c1",
                        text="t",
                        quote_ids=["q_0001"],
                        confidence=0.8,
                    )
                ],
            ),
        )


def test_draft_claims_must_cite_in_package_quotes():
    with pytest.raises(ValidationError):
        EvidencePackage(
            request=EvidenceRequest(
                question="q", session_id="s", response_mode="evidence_plus_draft"
            ),
            normalized_query="q",
            quotes=[
                EvidenceQuote(
                    quote_id="q_0001",
                    rank=1,
                    passage_id="p1",
                    text="t",
                    confidence=0.8,
                )
            ],
            coverage=_coverage(),
            answer_draft=EvidenceAnswerDraft(
                markdown="hallucinated [q_9999]",
                claims=[
                    EvidenceClaim(
                        claim_id="c1",
                        text="x",
                        quote_ids=["q_9999"],
                        confidence=0.8,
                    )
                ],
            ),
        )


def test_draft_claims_must_include_at_least_one_quote_id():
    with pytest.raises(ValidationError):
        EvidencePackage(
            request=EvidenceRequest(
                question="q", session_id="s", response_mode="evidence_plus_draft"
            ),
            normalized_query="q",
            quotes=[
                EvidenceQuote(
                    quote_id="q_0001",
                    rank=1,
                    passage_id="p1",
                    text="t",
                    confidence=0.8,
                )
            ],
            coverage=_coverage(),
            answer_draft=EvidenceAnswerDraft(
                markdown="uncited claim",
                claims=[
                    EvidenceClaim(
                        claim_id="c1",
                        text="x",
                        quote_ids=[],
                        confidence=0.8,
                    )
                ],
            ),
        )


def test_validate_on_assignment_blocks_uncited_draft_added_later():
    package = EvidencePackage(
        request=EvidenceRequest(
            question="q", session_id="s", response_mode="evidence_plus_draft"
        ),
        normalized_query="q",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                text="t",
                confidence=0.8,
            )
        ],
        coverage=_coverage(),
    )
    with pytest.raises(ValidationError):
        package.answer_draft = EvidenceAnswerDraft(
            markdown="x [q_9999]",
            claims=[
                EvidenceClaim(
                    claim_id="c1",
                    text="x",
                    quote_ids=["q_9999"],
                    confidence=0.8,
                )
            ],
        )

import pytest

from src.evidence.enhancer import FakeEvidenceEnhancer, UncitedDraftEnhancer
from src.evidence.models import (
    EvidenceCoverage,
    EvidencePackage,
    EvidenceQuote,
    EvidenceRequest,
)
from src.evidence.service import EvidenceService


def _pkg():
    return EvidencePackage(
        request=EvidenceRequest(
            question="Deploy with NAI?",
            session_id="s1",
            response_mode="evidence_plus_draft",
        ),
        normalized_query="deploy with NAI",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nai/deploy",
                text="NAI serves models.",
                confidence=0.86,
            )
        ],
        coverage=EvidenceCoverage(
            documents_searched=3,
            documents_with_evidence=1,
            retrieval_depth=20,
            reranker_applied=False,
            signal_pool_active=False,
            graph_expansion_applied=False,
        ),
    )


@pytest.mark.asyncio
async def test_fake_enhancer_adds_cited_draft():
    enhanced = await FakeEvidenceEnhancer().enhance(_pkg())

    assert enhanced.answer_draft.claims[0].quote_ids == ["q_0001"]
    assert "[q_0001]" in enhanced.answer_draft.markdown


@pytest.mark.asyncio
async def test_service_refuses_to_launder_uncited_draft():
    service = EvidenceService(
        search_candidates=None,
        extract_quotes=None,
        expand_with_graph=None,
        live_validation_available=False,
        enhancer=UncitedDraftEnhancer(),
    )

    package = await service._safe_enhance(_pkg())

    assert package.answer_draft is None
    assert any(gap.kind == "uncited_draft_rejected" for gap in package.gaps)

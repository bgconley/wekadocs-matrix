from __future__ import annotations

from typing import Protocol

from src.evidence.models import EvidenceAnswerDraft, EvidenceClaim, EvidencePackage


class EvidenceEnhancer(Protocol):
    async def enhance(self, package: EvidencePackage) -> EvidencePackage: ...


class FakeEvidenceEnhancer:
    async def enhance(self, package: EvidencePackage) -> EvidencePackage:
        if not package.quotes:
            return package
        quote = package.quotes[0]
        package.answer_draft = EvidenceAnswerDraft(
            markdown=f"{quote.text} [{quote.quote_id}]",
            claims=[
                EvidenceClaim(
                    claim_id="claim_0001",
                    text=quote.text,
                    quote_ids=[quote.quote_id],
                    confidence=quote.confidence,
                )
            ],
            enhancer_model="fake",
            enhancer_latency_ms=0.0,
        )
        return package


class UncitedDraftEnhancer:
    async def enhance(self, package: EvidencePackage) -> EvidencePackage:
        package.answer_draft = EvidenceAnswerDraft(
            markdown="Trust me, NAI does everything.",
            claims=[
                EvidenceClaim(
                    claim_id="c1",
                    text="everything",
                    quote_ids=["q_9999"],
                    confidence=0.99,
                )
            ],
        )
        return package

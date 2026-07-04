from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceResponseMode = Literal["evidence_only", "evidence_plus_draft"]
EvidenceGapKind = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
    "missing_live_validation",
    "source_unavailable",
    "budget_exceeded",
    "uncited_draft_rejected",
]
EvidenceGapSeverity = Literal["info", "warning", "error"]


class EvidenceRequest(BaseModel):
    question: str
    session_id: str
    top_k: int = 5
    max_quotes: int = 6
    max_quote_tokens: int = 80
    include_context_tokens: int = 20
    retrieval_depth: int = 60
    graph_enrichment: bool = False
    scope: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    response_mode: EvidenceResponseMode = "evidence_only"


class EvidenceQuote(BaseModel):
    quote_id: str
    rank: int
    passage_id: str
    section_id: Optional[str] = None
    doc_tag: Optional[str] = None
    title: Optional[str] = None
    parent_path: List[str] = Field(default_factory=list)
    source_uri: Optional[str] = None
    text: str
    confidence: float
    score: Optional[float] = None
    source: str = "retrieval"
    retrieval_signals: Dict[str, Any] = Field(default_factory=dict)
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class EvidenceCoverage(BaseModel):
    documents_searched: int
    documents_with_evidence: int
    retrieval_depth: int
    reranker_applied: bool
    signal_pool_active: bool
    graph_expansion_applied: bool
    partial: bool = False
    limit_reason: str = "none"


class EvidenceGap(BaseModel):
    kind: EvidenceGapKind
    message: str
    severity: EvidenceGapSeverity = "warning"
    quote_ids: List[str] = Field(default_factory=list)


class EvidenceClaim(BaseModel):
    claim_id: str
    text: str
    quote_ids: List[str]
    confidence: float


class EvidenceAnswerDraft(BaseModel):
    markdown: str
    claims: List[EvidenceClaim]
    enhancer_model: Optional[str] = None
    enhancer_latency_ms: Optional[float] = None


class EvidencePackage(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    package_id: str = Field(default_factory=lambda: f"ep_{uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request: EvidenceRequest
    normalized_query: str
    quotes: List[EvidenceQuote]
    coverage: EvidenceCoverage
    gaps: List[EvidenceGap] = Field(default_factory=list)
    answer_draft: Optional[EvidenceAnswerDraft] = None
    trace_id: Optional[str] = None
    diagnostic_id: Optional[str] = None
    diagnostic_uri: Optional[str] = None
    retrieval_metrics: Dict[str, Any] = Field(default_factory=dict)
    diagnostic_context: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _enforce_citation_provenance(self) -> "EvidencePackage":
        if self.answer_draft is None:
            return self
        if self.request.response_mode != "evidence_plus_draft":
            raise ValueError("answer_draft present but response_mode is evidence_only")

        quote_ids = {q.quote_id for q in self.quotes}
        for claim in self.answer_draft.claims:
            if not claim.quote_ids:
                raise ValueError(f"claim {claim.claim_id} has no citations")
            missing = [qid for qid in claim.quote_ids if qid not in quote_ids]
            if missing:
                raise ValueError(
                    f"claim {claim.claim_id} cites unknown quotes: {missing}"
                )
        return self

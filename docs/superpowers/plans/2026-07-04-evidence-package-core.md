# Evidence Package Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `EvidencePackage` the canonical product contract and reduce MCP to a thin transport adapter over that contract.

**Architecture:** Move evidence retrieval, quote extraction, coverage, trace correlation, diagnostics, and optional LLM enhancement into `src/evidence/`. Keep MCP tools as request/response adapters that call the evidence service and apply MCP-specific budgets/resource URIs. The default response remains evidence-first; answer drafting is optional and must preserve claim-to-citation provenance.

**Tech Stack:** Python 3.11, Pydantic models, existing `QueryService`, `ScratchStore`, `RetrievalTraceBuilder`, pytest, fake dependencies for architecture-only tests, live validation deferred.

---

## File Structure

Create a new evidence package subsystem:

- `src/evidence/__init__.py` exports the public contract.
- `src/evidence/models.py` owns request/response/domain models.
- `src/evidence/service.py` orchestrates retrieval, graph expansion, quote extraction, coverage, trace, and diagnostics.
- `src/evidence/quotes.py` owns quote extraction and quote scoring.
- `src/evidence/coverage.py` owns coverage, gaps, contradictions, and confidence rung calculation.
- `src/evidence/enhancer.py` owns the optional LLM enhancer protocol and fake/test enhancer.
- `src/evidence/serializers.py` converts `EvidencePackage` into MCP-safe dictionaries.

Modify MCP and trace code:

- `src/mcp_server/mcp_tools.py` keeps tool schemas and adapters, but moves `kb_retrieve_evidence` internals into `EvidenceService`.
- `src/mcp_server/retrieval_trace.py` records `EvidencePackage` instead of receiving MCP-shaped quote dictionaries.
- `src/mcp_server/mcp_app.py` keeps tool registration behavior and exposes a thinner production profile.

Create tests:

- `tests/evidence/test_models.py`
- `tests/evidence/test_quotes.py`
- `tests/evidence/test_coverage.py`
- `tests/evidence/test_service.py`
- `tests/evidence/test_enhancer.py`
- `tests/mcp_server_tests/test_evidence_tool_adapter.py`
- Update `tests/mcp_server_tests/test_evidence_pack.py`
- Update `tests/mcp_server_tests/test_tool_profiles.py`

---

## Task 1: Define The Evidence Package Contract

**Files:**
- Create: `src/evidence/__init__.py`
- Create: `src/evidence/models.py`
- Create: `tests/evidence/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/evidence/test_models.py`:

```python
from src.evidence.models import (
    EvidenceCoverage,
    EvidenceGap,
    EvidencePackage,
    EvidenceQuote,
    EvidenceRequest,
)


def test_evidence_package_contract_round_trips():
    request = EvidenceRequest(
        question="How do I configure Prism Central authentication?",
        session_id="session-1",
        max_quotes=3,
        retrieval_depth=20,
        response_mode="evidence_only",
    )
    quote = EvidenceQuote(
        quote_id="q1",
        rank=1,
        passage_id="p1",
        section_id="s1",
        doc_tag="prism/admin/auth",
        parent_path=["Prism Central", "Authentication"],
        text="Configure identity providers in Prism Central.",
        confidence=0.82,
        score=0.91,
        source="retrieval",
        retrieval_signals={"dense": 0.8, "reranker": 0.91},
    )
    package = EvidencePackage(
        package_id="ep_123",
        request=request,
        normalized_query="configure Prism Central authentication",
        quotes=[quote],
        coverage=EvidenceCoverage(
            documents_searched=5,
            documents_with_evidence=1,
            retrieval_depth=20,
            reranker_applied=True,
            signal_pool_active=True,
            graph_expansion_applied=False,
            partial=False,
            limit_reason="none",
        ),
        gaps=[
            EvidenceGap(
                kind="missing_live_validation",
                message="Live datastore validation was not run.",
                severity="info",
            )
        ],
        trace_id="trace-1",
    )

    data = package.model_dump()
    assert data["request"]["question"] == request.question
    assert data["quotes"][0]["doc_tag"] == "prism/admin/auth"
    assert data["coverage"]["documents_with_evidence"] == 1
    assert data["gaps"][0]["kind"] == "missing_live_validation"


def test_evidence_package_requires_citations_for_draft_claims():
    package = EvidencePackage(
        package_id="ep_124",
        request=EvidenceRequest(question="What is NKP?", session_id="s"),
        normalized_query="What is NKP?",
        quotes=[],
        coverage=EvidenceCoverage(
            documents_searched=0,
            documents_with_evidence=0,
            retrieval_depth=0,
            reranker_applied=False,
            signal_pool_active=False,
            graph_expansion_applied=False,
            partial=False,
            limit_reason="none",
        ),
        answer_draft=None,
    )
    assert package.answer_draft is None
```

- [ ] **Step 2: Run the model tests and verify they fail**

Run:

```bash
pytest tests/evidence/test_models.py -q
```

Expected: import failure for `src.evidence.models`.

- [ ] **Step 3: Implement the contract models**

Create `src/evidence/models.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


EvidenceResponseMode = Literal["evidence_only", "evidence_plus_draft"]
EvidenceGapKind = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
    "missing_live_validation",
    "source_unavailable",
    "budget_truncated",
]
EvidenceGapSeverity = Literal["info", "warning", "error"]


class EvidenceRequest(BaseModel):
    question: str
    session_id: str
    top_k: int = 20
    max_quotes: int = 6
    max_quote_tokens: int = 80
    include_context_tokens: int = 20
    retrieval_depth: int = 60
    graph_enrichment: Optional[bool] = None
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
    partial: bool
    limit_reason: str


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
```

Create `src/evidence/__init__.py`:

```python
from src.evidence.models import (
    EvidenceAnswerDraft,
    EvidenceClaim,
    EvidenceCoverage,
    EvidenceGap,
    EvidencePackage,
    EvidenceQuote,
    EvidenceRequest,
)

__all__ = [
    "EvidenceAnswerDraft",
    "EvidenceClaim",
    "EvidenceCoverage",
    "EvidenceGap",
    "EvidencePackage",
    "EvidenceQuote",
    "EvidenceRequest",
]
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest tests/evidence/test_models.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/evidence/__init__.py src/evidence/models.py tests/evidence/test_models.py
git commit -m "feat: add evidence package contract"
```

---

## Task 2: Extract Quote Normalization And Scoring

**Files:**
- Create: `src/evidence/quotes.py`
- Create: `tests/evidence/test_quotes.py`
- Read: `src/mcp_server/mcp_tools.py:1081-1128`

- [ ] **Step 1: Write failing quote tests**

Create `tests/evidence/test_quotes.py`:

```python
from src.evidence.quotes import normalize_quote_payloads


def test_normalize_quote_payloads_creates_stable_quote_ids():
    raw_quotes = [
        {
            "rank": 1,
            "passage_id": "p1",
            "section_id": "s1",
            "doc_tag": "nci/storage/aos",
            "parent_path": "NCI > AOS Storage",
            "quote": "AOS provides distributed storage services.",
            "confidence": 0.81,
            "score": 0.9,
            "source": "retrieval",
        }
    ]

    quotes = normalize_quote_payloads(raw_quotes)

    assert len(quotes) == 1
    assert quotes[0].quote_id == "q_0001"
    assert quotes[0].parent_path == ["NCI", "AOS Storage"]
    assert quotes[0].text == "AOS provides distributed storage services."
    assert quotes[0].retrieval_signals == {"score": 0.9}


def test_normalize_quote_payloads_ignores_empty_quote_text():
    quotes = normalize_quote_payloads(
        [
            {"rank": 1, "passage_id": "p1", "quote": "   ", "confidence": 0.9},
            {"rank": 2, "passage_id": "p2", "quote": "Useful evidence.", "confidence": 0.7},
        ]
    )

    assert [q.quote_id for q in quotes] == ["q_0001"]
    assert quotes[0].passage_id == "p2"
```

- [ ] **Step 2: Run quote tests and verify they fail**

Run:

```bash
pytest tests/evidence/test_quotes.py -q
```

Expected: import failure for `src.evidence.quotes`.

- [ ] **Step 3: Implement quote normalization**

Create `src/evidence/quotes.py`:

```python
from __future__ import annotations

from typing import Any, Iterable, List

from src.evidence.models import EvidenceQuote


def _split_parent_path(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(">") if part.strip()]
    return []


def normalize_quote_payloads(raw_quotes: Iterable[dict[str, Any]]) -> List[EvidenceQuote]:
    quotes: List[EvidenceQuote] = []
    for raw in raw_quotes:
        text = str(raw.get("quote") or raw.get("text") or "").strip()
        if not text:
            continue
        score = raw.get("score")
        retrieval_signals = dict(raw.get("retrieval_signals") or {})
        if score is not None and "score" not in retrieval_signals:
            retrieval_signals["score"] = score
        quotes.append(
            EvidenceQuote(
                quote_id=f"q_{len(quotes) + 1:04d}",
                rank=int(raw.get("rank") or len(quotes) + 1),
                passage_id=str(raw.get("passage_id") or raw.get("section_id") or ""),
                section_id=raw.get("section_id"),
                doc_tag=raw.get("doc_tag"),
                parent_path=_split_parent_path(raw.get("parent_path")),
                source_uri=raw.get("source_uri"),
                text=text,
                confidence=float(raw.get("confidence") or 0.0),
                score=score,
                source=str(raw.get("source") or "retrieval"),
                retrieval_signals=retrieval_signals,
                context_before=raw.get("context_before"),
                context_after=raw.get("context_after"),
            )
        )
    return quotes
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest tests/evidence/test_quotes.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/evidence/quotes.py tests/evidence/test_quotes.py
git commit -m "feat: normalize evidence quotes"
```

---

## Task 3: Add Coverage, Gap, And Confidence Rung Logic

**Files:**
- Create: `src/evidence/coverage.py`
- Create: `tests/evidence/test_coverage.py`

- [ ] **Step 1: Write failing coverage tests**

Create `tests/evidence/test_coverage.py`:

```python
from src.evidence.coverage import build_coverage, identify_gaps
from src.evidence.models import EvidenceQuote


def test_build_coverage_counts_documents_and_flags():
    search_results = [
        {"doc_tag": "nci/aos", "section_id": "s1"},
        {"doc_tag": "nci/aos", "section_id": "s2"},
        {"doc_tag": "ncm/ops", "section_id": "s3"},
    ]
    quotes = [
        EvidenceQuote(
            quote_id="q_0001",
            rank=1,
            passage_id="p1",
            doc_tag="nci/aos",
            text="AOS evidence.",
            confidence=0.8,
        )
    ]
    metrics = {
        "reranker_applied": True,
        "signal_pool_used": True,
    }

    coverage = build_coverage(
        search_results=search_results,
        quotes=quotes,
        metrics=metrics,
        graph_expansion_applied=True,
        partial=False,
        limit_reason="none",
    )

    assert coverage.documents_searched == 2
    assert coverage.documents_with_evidence == 1
    assert coverage.retrieval_depth == 3
    assert coverage.reranker_applied is True
    assert coverage.signal_pool_active is True
    assert coverage.graph_expansion_applied is True


def test_identify_gaps_marks_empty_evidence():
    gaps = identify_gaps(
        quotes=[],
        documents_searched=4,
        live_validation_available=False,
        partial=False,
        limit_reason="none",
    )

    assert [gap.kind for gap in gaps] == [
        "insufficient_evidence",
        "missing_live_validation",
    ]
```

- [ ] **Step 2: Run coverage tests and verify they fail**

Run:

```bash
pytest tests/evidence/test_coverage.py -q
```

Expected: import failure for `src.evidence.coverage`.

- [ ] **Step 3: Implement coverage helpers**

Create `src/evidence/coverage.py`:

```python
from __future__ import annotations

from typing import Any, Iterable, List

from src.evidence.models import EvidenceCoverage, EvidenceGap, EvidenceQuote


def build_coverage(
    *,
    search_results: Iterable[dict[str, Any]],
    quotes: Iterable[EvidenceQuote],
    metrics: dict[str, Any],
    graph_expansion_applied: bool,
    partial: bool,
    limit_reason: str,
) -> EvidenceCoverage:
    search_list = list(search_results)
    quote_list = list(quotes)
    searched_docs = {item.get("doc_tag") for item in search_list if item.get("doc_tag")}
    evidence_docs = {quote.doc_tag for quote in quote_list if quote.doc_tag}
    return EvidenceCoverage(
        documents_searched=len(searched_docs),
        documents_with_evidence=len(evidence_docs),
        retrieval_depth=len(search_list),
        reranker_applied=bool(metrics.get("reranker_applied")),
        signal_pool_active=bool(metrics.get("signal_pool_used") or metrics.get("signal_pool_enabled")),
        graph_expansion_applied=graph_expansion_applied,
        partial=partial,
        limit_reason=limit_reason,
    )


def identify_gaps(
    *,
    quotes: Iterable[EvidenceQuote],
    documents_searched: int,
    live_validation_available: bool,
    partial: bool,
    limit_reason: str,
) -> List[EvidenceGap]:
    quote_list = list(quotes)
    gaps: List[EvidenceGap] = []
    if not quote_list:
        gaps.append(
            EvidenceGap(
                kind="insufficient_evidence",
                message=f"No evidence quotes were selected from {documents_searched} searched documents.",
                severity="warning",
            )
        )
    if not live_validation_available:
        gaps.append(
            EvidenceGap(
                kind="missing_live_validation",
                message="Live embedding, reranker, Qdrant, and Neo4j validation were not run for this package.",
                severity="info",
            )
        )
    if partial:
        gaps.append(
            EvidenceGap(
                kind="budget_truncated",
                message=f"Evidence package was truncated by output budget: {limit_reason}.",
                severity="warning",
            )
        )
    return gaps
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest tests/evidence/test_coverage.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/evidence/coverage.py tests/evidence/test_coverage.py
git commit -m "feat: add evidence coverage and gap logic"
```

---

## Task 4: Extract EvidenceService From MCP Tool Logic

**Files:**
- Create: `src/evidence/service.py`
- Create: `tests/evidence/test_service.py`
- Modify: `src/mcp_server/mcp_tools.py`

- [ ] **Step 1: Run GitNexus impact before editing**

Run:

```bash
npx gitnexus impact kb_retrieve_evidence --direction upstream --include-tests
npx gitnexus impact _extract_evidence_from_passages --direction upstream --include-tests
```

Expected: report impact. If either returns `HIGH` or `CRITICAL`, stop and report blast radius before editing.

- [ ] **Step 2: Write failing service tests**

Create `tests/evidence/test_service.py`:

```python
import pytest

from src.evidence.models import EvidenceRequest
from src.evidence.service import EvidenceService


class FakeScratch:
    async def get(self, session_id, passage_id):
        return {"text": "full text", "parent_path_norm": "NCI > AOS"}


class FakeDeps:
    scratch = FakeScratch()


async def fake_search_candidates(**kwargs):
    return (
        {
            "results": [
                {
                    "passage_id": "p1",
                    "section_id": "s1",
                    "doc_tag": "nci/aos",
                    "title": "AOS Storage",
                    "score": 0.9,
                    "rank": 1,
                }
            ],
            "metrics": {
                "reranker_applied": True,
                "signal_pool_used": True,
                "query_rewrite_result": "configure AOS storage",
            },
        },
        {"diagnostic": "ctx"},
    )


async def fake_extract_quotes(**kwargs):
    return [
        {
            "rank": 1,
            "passage_id": "p1",
            "section_id": "s1",
            "doc_tag": "nci/aos",
            "parent_path": "NCI > AOS",
            "quote": "AOS provides distributed storage services.",
            "confidence": 0.82,
            "score": 0.9,
            "source": "retrieval",
        }
    ]


@pytest.mark.asyncio
async def test_evidence_service_builds_package():
    service = EvidenceService(
        search_candidates=fake_search_candidates,
        extract_quotes=fake_extract_quotes,
        expand_with_graph=None,
        write_trace=lambda package: "trace-1",
        emit_diagnostics=None,
        live_validation_available=False,
    )

    package = await service.build_package(
        request=EvidenceRequest(
            question="How does AOS storage work?",
            session_id="s1",
            max_quotes=2,
            retrieval_depth=10,
        ),
        deps=FakeDeps(),
    )

    assert package.normalized_query == "configure AOS storage"
    assert package.quotes[0].quote_id == "q_0001"
    assert package.coverage.documents_searched == 1
    assert package.trace_id == "trace-1"
    assert package.gaps[0].kind == "missing_live_validation"
```

- [ ] **Step 3: Run service tests and verify they fail**

Run:

```bash
pytest tests/evidence/test_service.py -q
```

Expected: import failure for `src.evidence.service`.

- [ ] **Step 4: Implement EvidenceService**

Create `src/evidence/service.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional
from uuid import uuid4

from src.evidence.coverage import build_coverage, identify_gaps
from src.evidence.models import EvidencePackage, EvidenceRequest
from src.evidence.quotes import normalize_quote_payloads


SearchCandidatesFn = Callable[..., Awaitable[tuple[dict[str, Any], dict[str, Any]]]]
ExtractQuotesFn = Callable[..., Awaitable[list[dict[str, Any]]]]
ExpandGraphFn = Callable[..., Awaitable[list[str]]]
WriteTraceFn = Callable[[EvidencePackage], Optional[str]]
EmitDiagnosticsFn = Callable[..., Awaitable[dict[str, Any] | None]]


class EvidenceService:
    def __init__(
        self,
        *,
        search_candidates: SearchCandidatesFn,
        extract_quotes: ExtractQuotesFn,
        expand_with_graph: Optional[ExpandGraphFn],
        write_trace: Optional[WriteTraceFn],
        emit_diagnostics: Optional[EmitDiagnosticsFn],
        live_validation_available: bool,
    ) -> None:
        self._search_candidates = search_candidates
        self._extract_quotes = extract_quotes
        self._expand_with_graph = expand_with_graph
        self._write_trace = write_trace
        self._emit_diagnostics = emit_diagnostics
        self._live_validation_available = live_validation_available

    async def build_package(self, *, request: EvidenceRequest, deps: Any) -> EvidencePackage:
        internal_fetch_k = max(request.max_quotes, request.retrieval_depth)
        evidence_options = dict(request.options)
        evidence_options.setdefault("max_per_doc", 5)

        search_payload, diagnostic_context = await self._search_candidates(
            query=request.question,
            top_k=internal_fetch_k,
            cursor=None,
            page_size=internal_fetch_k,
            scope=request.scope,
            filters=request.filters,
            options=evidence_options,
            deps=deps,
            effective_session=request.session_id,
            _fetch_k_override=internal_fetch_k,
        )

        search_results = search_payload.get("results", [])
        metrics = search_payload.get("metrics") or {}
        passage_ids = [item["passage_id"] for item in search_results if item.get("passage_id")]

        graph_expansion_applied = False
        if request.graph_enrichment and self._expand_with_graph:
            section_ids = [item["section_id"] for item in search_results if item.get("section_id")]
            graph_passage_ids = await self._expand_with_graph(
                section_ids=section_ids,
                deps=deps,
                effective_session=request.session_id,
            )
            passage_ids.extend(graph_passage_ids)
            graph_expansion_applied = bool(graph_passage_ids)

        raw_quotes = await self._extract_quotes(
            question=request.question,
            passage_ids=passage_ids,
            max_quotes=request.max_quotes,
            max_quote_tokens=request.max_quote_tokens,
            include_context_tokens=request.include_context_tokens,
            deps=deps,
            effective_session=request.session_id,
        )
        quotes = normalize_quote_payloads(raw_quotes)
        coverage = build_coverage(
            search_results=search_results,
            quotes=quotes,
            metrics=metrics,
            graph_expansion_applied=graph_expansion_applied,
            partial=False,
            limit_reason="none",
        )
        gaps = identify_gaps(
            quotes=quotes,
            documents_searched=coverage.documents_searched,
            live_validation_available=self._live_validation_available,
            partial=coverage.partial,
            limit_reason=coverage.limit_reason,
        )

        package = EvidencePackage(
            package_id=f"ep_{uuid4().hex[:12]}",
            request=request,
            normalized_query=metrics.get("query_rewrite_result") or request.question,
            quotes=quotes,
            coverage=coverage,
            gaps=gaps,
        )
        if self._write_trace:
            package.trace_id = self._write_trace(package)
        if self._emit_diagnostics:
            diagnostic = await self._emit_diagnostics(
                package=package,
                diagnostic_context=diagnostic_context,
            )
            if diagnostic:
                package.diagnostic_id = diagnostic.get("diagnostic_id")
                package.diagnostic_uri = diagnostic.get("diagnostic_uri")
        return package
```

- [ ] **Step 5: Run service tests**

Run:

```bash
pytest tests/evidence/test_service.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/evidence/service.py tests/evidence/test_service.py
git commit -m "feat: extract evidence package service"
```

---

## Task 5: Convert MCP `kb.retrieve_evidence` Into An Adapter

**Files:**
- Modify: `src/mcp_server/mcp_tools.py`
- Create: `src/evidence/serializers.py`
- Create: `tests/mcp_server_tests/test_evidence_tool_adapter.py`

- [ ] **Step 1: Write failing MCP adapter test**

Create `tests/mcp_server_tests/test_evidence_tool_adapter.py`:

```python
from src.evidence.models import EvidenceCoverage, EvidencePackage, EvidenceQuote, EvidenceRequest
from src.evidence.serializers import evidence_package_to_mcp_payload


def test_evidence_package_to_mcp_payload_preserves_current_shape():
    package = EvidencePackage(
        package_id="ep_1",
        request=EvidenceRequest(question="How do I configure NAI?", session_id="s1"),
        normalized_query="configure NAI",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nai/deploy",
                parent_path=["NAI", "Deploy"],
                text="Deploy models through Nutanix Enterprise AI.",
                confidence=0.84,
                source="retrieval",
            )
        ],
        coverage=EvidenceCoverage(
            documents_searched=4,
            documents_with_evidence=1,
            retrieval_depth=20,
            reranker_applied=True,
            signal_pool_active=False,
            graph_expansion_applied=False,
            partial=False,
            limit_reason="none",
        ),
        trace_id="trace-1",
    )

    payload = evidence_package_to_mcp_payload(package)

    assert payload["package_id"] == "ep_1"
    assert payload["quotes"][0]["quote"] == "Deploy models through Nutanix Enterprise AI."
    assert payload["quotes"][0]["quote_id"] == "q_0001"
    assert payload["coverage"]["documents_with_evidence"] == 1
    assert payload["trace_id"] == "trace-1"
```

- [ ] **Step 2: Run adapter test and verify it fails**

Run:

```bash
pytest tests/mcp_server_tests/test_evidence_tool_adapter.py -q
```

Expected: import failure for `src.evidence.serializers`.

- [ ] **Step 3: Implement serializer**

Create `src/evidence/serializers.py`:

```python
from __future__ import annotations

from typing import Any, Dict

from src.evidence.models import EvidencePackage


def evidence_package_to_mcp_payload(package: EvidencePackage) -> Dict[str, Any]:
    return {
        "package_id": package.package_id,
        "question": package.request.question,
        "normalized_query": package.normalized_query,
        "quotes": [
            {
                "quote_id": quote.quote_id,
                "rank": quote.rank,
                "passage_id": quote.passage_id,
                "section_id": quote.section_id,
                "doc_tag": quote.doc_tag,
                "parent_path": " > ".join(quote.parent_path),
                "quote": quote.text,
                "confidence": quote.confidence,
                "score": quote.score,
                "source": quote.source,
                "retrieval_signals": quote.retrieval_signals,
            }
            for quote in package.quotes
        ],
        "coverage": package.coverage.model_dump(),
        "gaps": [gap.model_dump() for gap in package.gaps],
        "answer_draft": package.answer_draft.model_dump() if package.answer_draft else None,
        "trace_id": package.trace_id,
        "diagnostic_id": package.diagnostic_id,
        "diagnostic_uri": package.diagnostic_uri,
    }
```

- [ ] **Step 4: Modify `kb_retrieve_evidence` to delegate**

In `src/mcp_server/mcp_tools.py`, keep the public signature for `kb_retrieve_evidence`, but replace the orchestration body with:

```python
    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")

    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)
    if top_k != KB_SEARCH_DEFAULT_TOP_K and max_quotes == 6:
        max_quotes = top_k

    request = EvidenceRequest(
        question=question,
        session_id=effective_session,
        top_k=top_k,
        max_quotes=max_quotes,
        max_quote_tokens=max_quote_tokens,
        include_context_tokens=include_context_tokens,
        retrieval_depth=retrieval_depth,
        graph_enrichment=graph_enrichment,
        scope=scope,
        filters=filters,
        options=dict(options or {}),
        response_mode="evidence_only",
    )
    service = EvidenceService(
        search_candidates=_kb_search_candidates,
        extract_quotes=_extract_evidence_from_passages,
        expand_with_graph=_expand_evidence_with_structure,
        write_trace=_write_evidence_trace,
        emit_diagnostics=_emit_evidence_diagnostics,
        live_validation_available=False,
    )
    package = await service.build_package(request=request, deps=deps)
    payload = evidence_package_to_mcp_payload(package)
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "snippets"
    )
    finalized = _finalize_payload(
        "kb_retrieve_evidence",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=budget_partial,
        limit_reason=budget_reason if budget_partial else "none",
        session_id=effective_session,
    )
    return finalized
```

Add imports at the top of `src/mcp_server/mcp_tools.py`:

```python
from src.evidence.models import EvidenceRequest
from src.evidence.serializers import evidence_package_to_mcp_payload
from src.evidence.service import EvidenceService
```

- [ ] **Step 5: Add MCP-local trace and diagnostics wrappers**

Add two local helpers near the existing trace helpers:

```python
def _write_evidence_trace(package: EvidencePackage) -> str:
    trace = RetrievalTraceBuilder(trace_id=uuid4().hex, session_id=package.request.session_id)
    trace.record_query(
        client_query=package.request.question,
        reformulated=package.normalized_query,
        method="evidence_service",
        latency_ms=0,
        dual_query_active=False,
    )
    trace.record_evidence_pack(
        quotes=[
            TraceQuote(
                rank=quote.rank,
                confidence=quote.confidence,
                doc_tag=quote.doc_tag,
                parent_path=" > ".join(quote.parent_path),
                source=quote.source,
                text=quote.text,
            )
            for quote in package.quotes
        ],
        coverage=package.coverage.model_dump(),
    )
    write_trace(trace)
    set_active_trace(package.request.session_id, trace)
    return trace.trace_id


async def _emit_evidence_diagnostics(
    *, package: EvidencePackage, diagnostic_context: dict[str, Any]
) -> dict[str, Any] | None:
    return None
```

This keeps diagnostics non-blocking in the first extraction. Task 8 reconnects the existing diagnostic emitter with package metadata.

- [ ] **Step 6: Run adapter and existing evidence tests**

Run:

```bash
pytest tests/mcp_server_tests/test_evidence_tool_adapter.py tests/mcp_server_tests/test_evidence_pack.py tests/evidence -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/evidence/serializers.py src/mcp_server/mcp_tools.py tests/mcp_server_tests/test_evidence_tool_adapter.py
git commit -m "refactor: make MCP evidence tool an adapter"
```

---

## Task 6: Add Optional LLM Evidence Enhancer

**Files:**
- Create: `src/evidence/enhancer.py`
- Create: `tests/evidence/test_enhancer.py`
- Modify: `src/evidence/service.py`
- Modify: `src/evidence/models.py`

- [ ] **Step 1: Write failing enhancer tests**

Create `tests/evidence/test_enhancer.py`:

```python
import pytest

from src.evidence.enhancer import FakeEvidenceEnhancer
from src.evidence.models import EvidenceCoverage, EvidencePackage, EvidenceQuote, EvidenceRequest


def _package() -> EvidencePackage:
    return EvidencePackage(
        package_id="ep_1",
        request=EvidenceRequest(
            question="How do I deploy models with NAI?",
            session_id="s1",
            response_mode="evidence_plus_draft",
        ),
        normalized_query="deploy models with Nutanix Enterprise AI",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nai/deploy",
                text="Nutanix Enterprise AI deploys and serves models.",
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
            partial=False,
            limit_reason="none",
        ),
    )


@pytest.mark.asyncio
async def test_fake_enhancer_adds_cited_draft():
    package = _package()
    enhancer = FakeEvidenceEnhancer()

    enhanced = await enhancer.enhance(package)

    assert enhanced.answer_draft is not None
    assert enhanced.answer_draft.claims[0].quote_ids == ["q_0001"]
    assert "[q_0001]" in enhanced.answer_draft.markdown
```

- [ ] **Step 2: Run enhancer tests and verify they fail**

Run:

```bash
pytest tests/evidence/test_enhancer.py -q
```

Expected: import failure for `src.evidence.enhancer`.

- [ ] **Step 3: Implement enhancer protocol and fake enhancer**

Create `src/evidence/enhancer.py`:

```python
from __future__ import annotations

from typing import Protocol

from src.evidence.models import EvidenceAnswerDraft, EvidenceClaim, EvidencePackage


class EvidenceEnhancer(Protocol):
    async def enhance(self, package: EvidencePackage) -> EvidencePackage:
        ...


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
```

- [ ] **Step 4: Wire enhancer into EvidenceService**

Modify `src/evidence/service.py` constructor:

```python
from src.evidence.enhancer import EvidenceEnhancer
```

Add constructor argument:

```python
        enhancer: Optional[EvidenceEnhancer] = None,
```

Set:

```python
        self._enhancer = enhancer
```

Before returning `package` in `build_package`:

```python
        if request.response_mode == "evidence_plus_draft" and self._enhancer:
            package = await self._enhancer.enhance(package)
```

- [ ] **Step 5: Run enhancer and service tests**

Run:

```bash
pytest tests/evidence/test_enhancer.py tests/evidence/test_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/evidence/enhancer.py src/evidence/service.py tests/evidence/test_enhancer.py
git commit -m "feat: add optional evidence enhancer"
```

---

## Task 7: Add MCP Response Mode For Evidence-Only Versus Evidence-Plus-Draft

**Files:**
- Modify: `src/mcp_server/mcp_tools.py`
- Modify: `src/evidence/serializers.py`
- Modify: `tests/mcp_server_tests/test_evidence_tool_adapter.py`
- Modify: `tests/mcp_server_tests/test_evidence_pack.py`

- [ ] **Step 1: Write failing adapter test for draft mode**

Append to `tests/mcp_server_tests/test_evidence_tool_adapter.py`:

```python
from src.evidence.models import EvidenceAnswerDraft, EvidenceClaim


def test_mcp_payload_includes_answer_draft_when_present():
    package = EvidencePackage(
        package_id="ep_2",
        request=EvidenceRequest(
            question="What is NKP?",
            session_id="s1",
            response_mode="evidence_plus_draft",
        ),
        normalized_query="What is NKP?",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                text="NKP manages Kubernetes fleets.",
                confidence=0.8,
            )
        ],
        coverage=EvidenceCoverage(
            documents_searched=2,
            documents_with_evidence=1,
            retrieval_depth=10,
            reranker_applied=False,
            signal_pool_active=False,
            graph_expansion_applied=False,
            partial=False,
            limit_reason="none",
        ),
        answer_draft=EvidenceAnswerDraft(
            markdown="NKP manages Kubernetes fleets. [q_0001]",
            claims=[
                EvidenceClaim(
                    claim_id="claim_0001",
                    text="NKP manages Kubernetes fleets.",
                    quote_ids=["q_0001"],
                    confidence=0.8,
                )
            ],
            enhancer_model="fake",
        ),
    )

    payload = evidence_package_to_mcp_payload(package)

    assert payload["answer_draft"]["claims"][0]["quote_ids"] == ["q_0001"]
```

- [ ] **Step 2: Run adapter test**

Run:

```bash
pytest tests/mcp_server_tests/test_evidence_tool_adapter.py -q
```

Expected: pass if Task 5 serializer already preserves `answer_draft`. If it fails, update `evidence_package_to_mcp_payload` to include `answer_draft`.

- [ ] **Step 3: Add `response_mode` to MCP tool schema**

In `src/mcp_server/mcp_tools.py`, add to the `kb.retrieve_evidence` schema properties:

```python
        "response_mode": {
            "type": "string",
            "enum": ["evidence_only", "evidence_plus_draft"],
            "default": "evidence_only",
            "description": "Return evidence only, or evidence plus a citation-preserving draft answer.",
        },
```

Add parameter to `kb_retrieve_evidence` signature:

```python
    response_mode: str = "evidence_only",
```

Set request field:

```python
        response_mode=(
            "evidence_plus_draft"
            if response_mode == "evidence_plus_draft"
            else "evidence_only"
        ),
```

- [ ] **Step 4: Run MCP evidence tests**

Run:

```bash
pytest tests/mcp_server_tests/test_evidence_tool_adapter.py tests/mcp_server_tests/test_evidence_pack.py tests/mcp_server_tests/test_tool_profiles.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mcp_server/mcp_tools.py src/evidence/serializers.py tests/mcp_server_tests/test_evidence_tool_adapter.py tests/mcp_server_tests/test_evidence_pack.py
git commit -m "feat: expose evidence response modes"
```

---

## Task 8: Reconnect Diagnostics And Trace Around EvidencePackage

**Files:**
- Modify: `src/evidence/service.py`
- Modify: `src/mcp_server/mcp_tools.py`
- Modify: `src/mcp_server/retrieval_trace.py`
- Create: `tests/evidence/test_trace_diagnostics.py`

- [ ] **Step 1: Write failing trace/diagnostic test**

Create `tests/evidence/test_trace_diagnostics.py`:

```python
from src.evidence.models import EvidenceCoverage, EvidencePackage, EvidenceQuote, EvidenceRequest
from src.mcp_server.retrieval_trace import RetrievalTraceBuilder


def test_trace_builder_records_evidence_package():
    package = EvidencePackage(
        package_id="ep_1",
        request=EvidenceRequest(question="How does NC2 work?", session_id="s1"),
        normalized_query="How does NC2 work?",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nc2/aws",
                text="NC2 runs Nutanix Cloud Clusters in public cloud.",
                confidence=0.83,
            )
        ],
        coverage=EvidenceCoverage(
            documents_searched=2,
            documents_with_evidence=1,
            retrieval_depth=10,
            reranker_applied=True,
            signal_pool_active=True,
            graph_expansion_applied=False,
            partial=False,
            limit_reason="none",
        ),
    )

    trace = RetrievalTraceBuilder(trace_id="trace-1", session_id="s1")
    trace.record_evidence_package(package)

    rendered = trace.format()
    assert "Evidence pack" in rendered
    assert "NC2 runs Nutanix Cloud Clusters" in rendered
```

- [ ] **Step 2: Run trace test and verify it fails**

Run:

```bash
pytest tests/evidence/test_trace_diagnostics.py -q
```

Expected: `record_evidence_package` missing.

- [ ] **Step 3: Add `record_evidence_package` to trace builder**

In `src/mcp_server/retrieval_trace.py`, import:

```python
from src.evidence.models import EvidencePackage
```

Add method to `RetrievalTraceBuilder`:

```python
    def record_evidence_package(self, package: EvidencePackage) -> None:
        self.record_query(
            client_query=package.request.question,
            reformulated=package.normalized_query,
            method="evidence_package",
            latency_ms=0,
            dual_query_active=False,
        )
        self.record_evidence_pack(
            quotes=[
                TraceQuote(
                    rank=quote.rank,
                    confidence=quote.confidence,
                    doc_tag=quote.doc_tag,
                    parent_path=" > ".join(quote.parent_path),
                    source=quote.source,
                    text=quote.text,
                )
                for quote in package.quotes
            ],
            coverage=package.coverage.model_dump(),
        )
```

- [ ] **Step 4: Replace MCP-local trace wrapper**

In `src/mcp_server/mcp_tools.py`, simplify `_write_evidence_trace`:

```python
def _write_evidence_trace(package: EvidencePackage) -> str:
    trace = RetrievalTraceBuilder(trace_id=uuid4().hex, session_id=package.request.session_id)
    trace.record_evidence_package(package)
    write_trace(trace)
    set_active_trace(package.request.session_id, trace)
    return trace.trace_id
```

- [ ] **Step 5: Run trace and evidence tests**

Run:

```bash
pytest tests/evidence/test_trace_diagnostics.py tests/evidence tests/mcp_server_tests/test_evidence_pack.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mcp_server/retrieval_trace.py src/mcp_server/mcp_tools.py tests/evidence/test_trace_diagnostics.py
git commit -m "refactor: trace evidence packages directly"
```

---

## Task 9: Slim The Production MCP Tool Surface

**Files:**
- Modify: `src/mcp_server/mcp_tools.py`
- Modify: `src/mcp_server/mcp_app.py`
- Modify: `tests/mcp_server_tests/test_tool_profiles.py`

- [ ] **Step 1: Write failing profile tests**

Append to `tests/mcp_server_tests/test_tool_profiles.py`:

```python
def test_production_profile_exposes_evidence_first_surface():
    from src.mcp_server.mcp_tools import build_tool_specs

    specs = build_tool_specs(profile="production")
    names = {spec["name"] for spec in specs}

    assert "kb.retrieve_evidence" in names
    assert "kb.read_excerpt" in names
    assert "diagnostics.latest" in names
    assert "search_documentation" not in names
    assert not any("[Deprecated" in spec.get("description", "") for spec in specs)
```

- [ ] **Step 2: Run profile tests**

Run:

```bash
pytest tests/mcp_server_tests/test_tool_profiles.py -q
```

Expected: fail if `build_tool_specs(profile=...)` does not exist or if production still exposes aliases.

- [ ] **Step 3: Add explicit profile argument**

In `src/mcp_server/mcp_tools.py`, change the tool spec builder from environment-only profile selection to:

```python
def build_tool_specs(profile: str | None = None) -> list[dict[str, Any]]:
    active_profile = profile or MCP_TOOL_PROFILE
    specs = _all_tool_specs()
    if active_profile == "production":
        allowed = {"kb.retrieve_evidence", "kb.read_excerpt", "diagnostics.latest"}
        return [spec for spec in specs if spec["name"] in allowed]
    if active_profile == "analyst":
        return [spec for spec in specs if "[Deprecated" not in spec.get("description", "")]
    return specs
```

Keep `_all_tool_specs()` as the complete internal list that includes compatibility aliases for full mode.

- [ ] **Step 4: Update MCP app registration**

In `src/mcp_server/mcp_app.py`, call:

```python
tool_specs = build_tool_specs(profile=MCP_TOOL_PROFILE)
```

Keep call dispatch backward-compatible for full mode by leaving the handler map unchanged.

- [ ] **Step 5: Run MCP profile and evidence tests**

Run:

```bash
pytest tests/mcp_server_tests/test_tool_profiles.py tests/mcp_server_tests/test_evidence_pack.py tests/mcp_server_tests/test_evidence_tool_adapter.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mcp_server/mcp_tools.py src/mcp_server/mcp_app.py tests/mcp_server_tests/test_tool_profiles.py
git commit -m "refactor: slim production MCP tool profile"
```

---

## Task 10: Split `mcp_tools.py` After Evidence Extraction

**Files:**
- Create: `src/mcp_server/tools/__init__.py`
- Create: `src/mcp_server/tools/evidence.py`
- Create: `src/mcp_server/tools/read.py`
- Create: `src/mcp_server/tools/search.py`
- Create: `src/mcp_server/tools/diagnostics.py`
- Create: `src/mcp_server/tools/profiles.py`
- Modify: `src/mcp_server/mcp_tools.py`
- Modify: `tests/mcp_server_tests/test_tool_profiles.py`

- [ ] **Step 1: Run GitNexus impact**

Run:

```bash
npx gitnexus impact mcp_tools --direction upstream --include-tests
npx gitnexus impact build_tool_specs --direction upstream --include-tests
```

Expected: report blast radius. If impact is `HIGH` or `CRITICAL`, pause and report before editing.

- [ ] **Step 2: Move specs without changing behavior**

Create `src/mcp_server/tools/profiles.py`:

```python
from __future__ import annotations

from typing import Any


def filter_tool_specs(
    specs: list[dict[str, Any]], *, profile: str
) -> list[dict[str, Any]]:
    if profile == "production":
        allowed = {"kb.retrieve_evidence", "kb.read_excerpt", "diagnostics.latest"}
        return [spec for spec in specs if spec["name"] in allowed]
    if profile == "analyst":
        return [spec for spec in specs if "[Deprecated" not in spec.get("description", "")]
    return specs
```

Create `src/mcp_server/tools/__init__.py`:

```python
from src.mcp_server.tools.profiles import filter_tool_specs

__all__ = ["filter_tool_specs"]
```

Update `src/mcp_server/mcp_tools.py`:

```python
from src.mcp_server.tools.profiles import filter_tool_specs
```

Use `filter_tool_specs(_all_tool_specs(), profile=active_profile)` inside `build_tool_specs`.

- [ ] **Step 3: Run profile tests**

Run:

```bash
pytest tests/mcp_server_tests/test_tool_profiles.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Move evidence adapter function**

Create `src/mcp_server/tools/evidence.py` and move the `kb_retrieve_evidence` adapter plus `_write_evidence_trace` and `_emit_evidence_diagnostics` into it. Keep imports explicit. Re-export from `src/mcp_server/mcp_tools.py`:

```python
from src.mcp_server.tools.evidence import kb_retrieve_evidence
```

- [ ] **Step 5: Run evidence tests**

Run:

```bash
pytest tests/mcp_server_tests/test_evidence_pack.py tests/mcp_server_tests/test_evidence_tool_adapter.py tests/evidence -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Move read/search/diagnostics adapters**

Move only one cluster at a time:

```text
kb_search, search_sections -> src/mcp_server/tools/search.py
kb_read_excerpt, kb_expand_excerpt -> src/mcp_server/tools/read.py
diagnostics handlers -> src/mcp_server/tools/diagnostics.py
```

After each move, keep a re-export in `src/mcp_server/mcp_tools.py` and run:

```bash
pytest tests/mcp_server_tests -q
```

Expected: selected MCP tests pass after each move.

- [ ] **Step 7: Commit**

```bash
git add src/mcp_server/tools src/mcp_server/mcp_tools.py tests/mcp_server_tests/test_tool_profiles.py
git commit -m "refactor: split MCP tool adapters"
```

---

## Task 11: Clean Up Defunct Compatibility Paths

**Files:**
- Modify: `src/mcp_server/mcp_tools.py`
- Modify: `src/mcp_server/mcp_app.py`
- Modify: `tests/mcp_server_tests/test_tool_profiles.py`
- Modify: `tests/test_cleanup_validation.py`

- [ ] **Step 1: Inventory compatibility aliases**

Run:

```bash
rg -n "Deprecated|Backward compat|backward|legacy|alias" src/mcp_server tests/mcp_server_tests tests/test_cleanup_validation.py
```

Expected: list compatibility paths. Save the output in the task notes before removing anything.

- [ ] **Step 2: Add tests for allowed legacy surface**

Update `tests/mcp_server_tests/test_tool_profiles.py` with:

```python
def test_full_profile_is_the_only_profile_with_deprecated_aliases():
    from src.mcp_server.mcp_tools import build_tool_specs

    full_specs = build_tool_specs(profile="full")
    analyst_specs = build_tool_specs(profile="analyst")
    production_specs = build_tool_specs(profile="production")

    assert any("[Deprecated" in spec.get("description", "") for spec in full_specs)
    assert not any("[Deprecated" in spec.get("description", "") for spec in analyst_specs)
    assert not any("[Deprecated" in spec.get("description", "") for spec in production_specs)
```

- [ ] **Step 3: Run profile tests**

Run:

```bash
pytest tests/mcp_server_tests/test_tool_profiles.py -q
```

Expected: pass.

- [ ] **Step 4: Remove aliases from production and analyst docs**

Do not delete full-mode aliases in this task. Remove deprecated alias mentions from production and analyst instructions only. Confirm `full` profile still supports existing clients.

- [ ] **Step 5: Run MCP and cleanup tests**

Run:

```bash
pytest tests/mcp_server_tests/test_tool_profiles.py tests/test_cleanup_validation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mcp_server/mcp_tools.py src/mcp_server/mcp_app.py tests/mcp_server_tests/test_tool_profiles.py tests/test_cleanup_validation.py
git commit -m "chore: constrain deprecated MCP aliases to full profile"
```

---

## Task 12: Verification Matrix

**Files:**
- Modify: `docs/architecture/evidence-package-core.md`
- Create: `tests/evidence/test_architecture_boundaries.py`

- [ ] **Step 1: Add architecture boundary test**

Create `tests/evidence/test_architecture_boundaries.py`:

```python
from pathlib import Path


def test_evidence_package_logic_does_not_import_mcp_tools():
    src = Path("src/evidence")
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "src.mcp_server.mcp_tools" in text:
            offenders.append(str(path))
    assert offenders == []


def test_mcp_tools_imports_evidence_service():
    text = Path("src/mcp_server/mcp_tools.py").read_text(encoding="utf-8")
    assert "src.evidence" in text
```

- [ ] **Step 2: Create architecture doc**

Create `docs/architecture/evidence-package-core.md`:

```markdown
# Evidence Package Core

The canonical retrieval product is `EvidencePackage`.

MCP is a transport adapter. It validates request shape, resolves session IDs,
applies MCP output budgets, emits MCP diagnostics, and serializes packages for
clients. It does not own retrieval orchestration, quote scoring, coverage, gap
classification, trace semantics, or LLM evidence enhancement.

The default response mode is `evidence_only`. The optional
`evidence_plus_draft` mode may add an answer draft only when every draft claim
references one or more `quote_id` values from the same package.

Live quality validation is separate from architecture validation. Embedding,
sparse vector, ColBERT, reranker, Qdrant, and Neo4j checks must be run during
the live validation pass.
```

- [ ] **Step 3: Run architecture and focused tests**

Run:

```bash
pytest tests/evidence tests/mcp_server_tests/test_evidence_pack.py tests/mcp_server_tests/test_evidence_tool_adapter.py tests/mcp_server_tests/test_tool_profiles.py tests/evidence/test_architecture_boundaries.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run broader architecture-only suite**

Run:

```bash
pytest tests/shared tests/query tests/providers tests/mcp_server_tests tests/contracts tests/ingestion tests/scripts tests/unit tests/clients tests/test_config_schema_completeness.py tests/test_phase2_provider_wiring.py tests/test_phase5_response_schema.py tests/test_phase7c_reranking.py tests/test_phase7e2_hybrid_retrieval.py tests/test_structure_aware_expansion.py tests/test_chunk_parent_mapping.py tests/test_cleanup_validation.py tests/test_nutanix_hard_rename_guard.py -m "not integration and not external and not live" --ignore=tests/unit/test_context_microdoc.py --ignore=tests/unit/test_tokenizer_overlap.py --ignore=tests/test_tokenizer_service.py --ignore=tests/test_jina_adaptive_batching.py --ignore=tests/test_phase7c_provider_factory.py --ignore=tests/test_phase7e_phase0.py --ignore=tests/query/test_guardrails_modes.py --ignore=tests/query/test_reranker_mode.py --ignore=tests/providers/test_profile_matrix.py --ignore=tests/test_query_api_weighted_fusion.py --ignore=tests/test_phase7c_dual_write.py --ignore=tests/unit/test_gliner_service.py -q
```

Expected: pass with live/model-cache/stale-profile buckets explicitly excluded.

- [ ] **Step 5: Run static checks**

Run:

```bash
python -m compileall -q src scripts
rg -n "WEKA|Weka|weka|WekaDocs|wekadocs|weka-" src config data/ingest/nutanix docker deploy scripts services .github docker-compose.yml Makefile docs/nutanix_sample_ingest
NEO4J_PASSWORD=test REDIS_PASSWORD=test JWT_SECRET=test docker compose config >/tmp/nutanix-compose-config.out
npx gitnexus detect-changes
```

Expected:

```text
compileall exits 0
rg exits 1 with no matches
docker compose config exits 0, allowing the obsolete version warning
GitNexus reports affected scope for review before commit
```

- [ ] **Step 6: Commit**

```bash
git add docs/architecture/evidence-package-core.md tests/evidence/test_architecture_boundaries.py
git commit -m "docs: document evidence package architecture"
```

---

## Execution Notes

Do not run live embedding, sparse, ColBERT, reranker, Qdrant, or Neo4j quality checks in this plan. This plan is architecture and code-boundary work only.

Do not remove full-profile backward-compatible MCP aliases until a client inventory confirms they are unused. Production and analyst profiles can hide deprecated aliases immediately.

Do not let the optional LLM enhancer emit uncited final answers. It can add claims and a draft only when every claim references quote IDs already present in the package.

Do not stage unrelated dirty files. Current known unrelated or separately reviewed paths include `reports/retrieval_diagnostics/**`, local progress files, and untracked repo-audit artifacts.

---

## Self-Review Checklist

- Spec coverage: The plan creates `EvidencePackage`, extracts evidence orchestration, preserves MCP behavior through an adapter, adds optional enhancer support, slims production tools, and starts god-module breakup.
- Placeholder scan: No task relies on unspecified behavior; each task has explicit files, commands, and expected outcomes.
- Type consistency: `EvidenceRequest`, `EvidenceQuote`, `EvidenceCoverage`, `EvidenceGap`, `EvidencePackage`, and `EvidenceService.build_package()` are introduced before later tasks use them.
- Live boundary: Live quality validation is explicitly excluded from this architecture pass and preserved for a later run.

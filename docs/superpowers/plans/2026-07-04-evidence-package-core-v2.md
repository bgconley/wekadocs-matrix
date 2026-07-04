# Evidence Package Core Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This is a revision of `2026-07-04-evidence-package-core.md`. It keeps the original's spine (an `EvidencePackage` contract, an `EvidenceService`, MCP demoted to an adapter, an optional citation-preserving enhancer) but corrects the module map it was written against and removes five behavior regressions the original smuggled into "behavior-preserving" tasks. **Do not execute the v1 plan.**

**Goal:** Make `EvidencePackage` the canonical product contract and reduce MCP to a thin transport adapter over that contract — **without changing any observable production behavior of `kb.retrieve_evidence` until it is explicitly, separately, and test-guardedly intended.**

**Architecture:** Move retrieval orchestration, quote normalization, coverage/gap logic, telemetry assembly, and optional LLM enhancement into a **pure** `src/evidence/` package that has **no dependency on `src/mcp_server/`**. MCP (`mcp_tools.py`) becomes an adapter that (a) normalizes the MCP request — backward-compat, fetch-depth clamp, graph-enrichment default resolution — into a fully-resolved `EvidenceRequest`, (b) calls `EvidenceService`, (c) applies MCP output budgets and reconciles the budget/partial state into the package, (d) emits MCP diagnostics with post-budget values, (e) persists the retrieval trace, and (f) serializes the package for clients. Trace *semantics* move into `retrieval_trace.py`; MCP only triggers them.

**Tech Stack:** Python 3.11, Pydantic v2 models, existing `QueryService`/`ScratchStore`/`RetrievalTraceBuilder`, pytest with the repo's existing fake-deps fixtures. **Live embedder/reranker/Qdrant/Neo4j/ColBERT/SPLADE validation is out of scope for this plan** and deferred to a later live pass.

---

## v2.5 Amendments (post sixth-pass, final polish)

Two one-line fixes + one historical-wording tidy; no code-behavior change:

1. **Dropped the unsanctioned `question` field.** The serializer emitted `question`, which the parity checklist did not sanction and the legacy payload never included. Removed, so the additive surface is exactly `{package_id, normalized_query, gaps}`.
2. **Task 8 commit stages the maybe-updated legacy test.** `git add` now also includes `tests/mcp_server_tests/test_evidence_pack.py` (a no-op if Step 5 left it unchanged), so an updated legacy-key assertion is committed alongside its adapter change.
3. *(tidy)* Reworded the last historical "truncation" mention in the v2.1 note to "budget/partial state."

---

## v2.4 Amendments (post fifth-pass, polish)

A fifth read confirmed no architectural or behavioral defects remain; the following polish landed (plus a self-review typo fix):

1. **Reranker parity regression test (Task 6).** The trace fixture now sets `_result_count = 42` with a 1-item `_result_snapshot`, and the test asserts `output_count == 42` — proving the trace uses the full count, not the capped snapshot.
2. **Softened "byte-for-byte" wording.** The behavior-parity checklist now says "behavior-equivalent except the sanctioned additive evidence-package fields," since Task 8 intentionally enriches the payload.
3. **Precise budget-state language.** Remaining loose "truncation" mentions are reworded to "budget/partial state," and the helper `mark_truncation` is renamed `mark_budget_state` (nothing is trimmed — `_apply_budget` flags only).
4. **Rename guard covers the fixture surface.** Task 11's WEKA guard now also scans `tests/mcp_server_tests/conftest.py`, which Task 1 creates/edits.
5. **Changelog typo fixed.** A v2.2 `replace_all` had mangled its own note to "`budget_exceeded` renamed `budget_exceeded`"; corrected to "`budget_truncated` → `budget_exceeded`."

---

## v2.3 Amendments (post fourth-pass, line-by-line review)

A fourth line-by-line read found five residual issues — mostly test-robustness and wording, no architectural defects. All validated against code and folded in:

1. **Reranker `output_count` parity (Tasks 5–6).** Current trace reports `len(results)` uncapped (`mcp_tools.py:1237`); the plan reported `len(results[:20])`, under-counting when the evidence path fetches >20 (up to 150). Service now stores `_result_count = len(results)` and the trace uses it.
2. **Non-vacuous pins (Tasks 1, 8).** Both pins looped over `payload["quotes"]` to assert `title`/`uri` but never asserted the list is non-empty — an empty result would pass vacuously. Added `assert payload["quotes"]` to both, and the fixture note now requires the fake to return ≥1 deterministic `ChunkResult`.
3. **Honest pin-coverage claim (Task 8).** The Step 6 "Expected" text claimed the enriched pin asserts fetch clamp / graph default / trace fidelity / diagnostics; the shown test only checks payload+quote shape. Reworded to state what the pin actually guards and where the behavioral parity items are guarded instead.
4. **Consistent pin language (Task 8 + execution note).** "Must leave the Task 1 pin green" contradicted the (correct) Step 5 supersession. Both now say Task 8 performs the one sanctioned pin replacement and the enriched pin is kept green afterward.
5. **Complete Task 9 commit scope.** `git add` now includes `src/mcp_server/mcp_utils.py` (the `KB_EVIDENCE_DRAFT_ENABLED` flag) and `tests/mcp_server_tests/test_evidence_tool_adapter.py` (the gate test).

---

## v2.2 Amendments (post third-pass, line-by-line review)

A full line-by-line read + code cross-check found five more concrete blockers — two of them **introduced by the v2.1 amendments** (the pin now asserted the future shape; a stale `wekadocs://` literal in a test). All five are validated and folded in:

1. **Task 1 is a true pre-edit pin again.** v2.1 wrongly made the Task 1 pin assert the *post-refactor* keys (`gaps`, `package_id`, `normalized_query`, `coverage.partial`), so it could not pass against current code. Task 1 now characterizes the **legacy** shape (`{quotes, coverage}` + finalize metadata + `trace_id`, 6-key coverage, no enrichment keys) and passes today; Task 8 **supersedes** it with the enriched pin as the single sanctioned pin change.
2. **Quote `title`/`uri` no longer dropped.** Current quotes come from `_quote_from_passage` (`mcp_utils.py:467`) and include `title` and `uri`. `EvidenceQuote` gains `title`; normalization maps `uri`→`source_uri`; the serializer emits both legacy keys; both pins assert them. (Prevents a client-visible quote-shape regression.)
3. **Stale WEKA URI removed.** The Task 7 test literal `wekadocs://…` is replaced with `nutanixdocs://…` (`domain.py:173` → `uri_scheme="nutanixdocs"`), and the Task 11 rename guard now also scans the new test files.
4. **Honest budget wording.** `_apply_budget` (`mcp_utils.py:139`) only estimates size and flags `partial` — it never trims. The `budget_truncated` gap is renamed `budget_exceeded`, messages/docs no longer claim it truncates, and the Task 8 note is corrected.
5. **Graph-seed telemetry parity.** Current trace records `min(10, len(section_ids))` seeds (`mcp_tools.py:1307`); the service now matches that instead of `len(section_ids)`.

---

## v2.1 Amendments (post second-pass review)

A full sequential re-read plus live-code checks surfaced ten concentrated correctness gaps in the first v2 draft. All were validated and are folded into the tasks below:

1. **Budget-before-trace (Task 8):** the budget/partial state is reconciled into the package *before* the trace is written, so trace coverage, `coverage.partial`, and top-level `partial` never disagree. The adapter now uses `mark_budget_state` (Task 4) instead of ad-hoc dict mutation.
2. **Honest serializer (Task 7):** relabeled from "preserve exactly" to "additive + no null leaks." `answer_draft`/`diagnostic_id`/`diagnostic_uri` are **omitted when absent** — emitting them as `null` violates `KB_EVIDENCE_OUTPUT_SCHEMA` (typed `string`). `package_id`/`normalized_query`/`gaps` are declared *intentional* additive enrichment.
3. **Pin locks the shape (Task 1):** the pin asserts the full default key set, forbids any top-level `None` value, and requires `partial == coverage.partial`.
4. **Lossless appendix telemetry (Tasks 5–6):** appendix entries are rebuilt from the result rows *and* scratch, preserving `rerank_score`, `heading`, and order (the earlier draft dropped them — the same trace-lossiness v2 criticized in v1).
5. **Graph neighbor count (Task 5):** `_graph_neighbors_added` is set, so the graph-enrichment facet no longer reports `0` after real expansion.
6. **Nine-facet trace test (Task 6):** asserts all nine facets (adds RELATED_TO, graph enrichment with nonzero neighbors, and stage snapshots).
7. **Explicit pin fixture (Task 1):** `evidence_fake_ctx` is specified to satisfy `_get_deps` (request context + pre-initialized `Deps` so `ensure_initialized()` is a no-op).
8. **Gated draft mode (Task 9):** `evidence_plus_draft` is wired only behind `KB_EVIDENCE_DRAFT_ENABLED` (default **off**) so the deterministic fake enhancer is never exposed as a user-facing answer before the real LLM enhancer exists.
9. **No transport-in-search (Task 10):** the optional relocation of the adapter into `mcp_search.py` is dropped; only the behavior-neutral schema extraction remains.
10. **Failing rename guard (Task 11):** the WEKA-token guard fails on match instead of always exiting 0.

---

## Why v2 differs from v1 (read before starting)

The v1 plan was written against a stale map of `src/mcp_server/`. Verified facts about the **current** code:

| v1 assumption | Reality (verified) |
|---|---|
| `_kb_search_candidates`, `_extract_evidence_from_passages`, `_expand_evidence_with_structure` are internals of `mcp_tools.py` | They live in **`src/mcp_server/mcp_search.py`** and are already imported into `mcp_tools.py:18-24`. |
| Budget/session/scope helpers live in `mcp_tools.py` | They live in **`src/mcp_server/mcp_utils.py`** (`_apply_budget`, `_finalize_payload`, `_resolve_session_id`, `_normalize_scope`, `_get_deps`, `KB_EVIDENCE_MAX_FETCH_K=150`, `KB_EVIDENCE_GRAPH_EXPANSION_ENABLED`). |
| Profile selection lives in a `build_tool_specs(profile=...)` in `mcp_tools.py`; `_all_tool_specs()` exists | **Neither symbol exists.** `_tool_specs()` (`mcp_tools.py:2122`) returns *all* specs; filtering happens in **`mcp_app.py:build_mcp_server()`** via the `TOOL_PROFILES` dict (`mcp_app.py:163-191`). |
| Production profile = `{kb.retrieve_evidence, kb.read_excerpt, diagnostics.latest}` | Real production = **`{kb.retrieve_evidence, kb.read_excerpt, graph.expand}`**. `diagnostics.latest` **is not a tool** — diagnostics is a *resource template*. `graph.expand` is required by `PRODUCTION_INSTRUCTIONS`. |
| The god-module lives entirely in `mcp_tools.py`; create a new `src/mcp_server/tools/` package | The god-module was **already partially split** into `mcp_search.py`, `mcp_utils.py`, `query_service.py`, `models.py`, `scratch_store.py`, `retrieval_trace.py`. Creating `tools/` would be a **third** parallel scheme. |
| `EvidenceCoverage` can own `partial`/`limit_reason` set at service time | Budget runs in the **adapter after** the service returns. Hardcoding `partial=False` in the service produces a response where top-level `partial=true` and `coverage.partial=false` **contradict** each other. |
| Trace can be rebuilt from the package | The current tool drives **9** `record_*` trace facets from a rich `search_metrics` dict. The v1 `EvidencePackage` has **no field** for that telemetry, so its trace collapses to 2 facets. |
| GitNexus `impact` gates the risky edits | `gitnexus impact kb_retrieve_evidence` returns **LOW / 0 callers** because MCP tools dispatch through a dynamic `full_tool_map[name]` map, invisible to the static call graph. **The golden pin in Task 1 — not GitNexus — is the real safety gate for the MCP surface.** |

### Ownership after this refactor (the architectural spine)

| Concern | Owner | Notes |
|---|---|---|
| Request normalization (backward-compat `top_k`→`max_quotes`, fetch-depth clamp to `KB_EVIDENCE_MAX_FETCH_K`, graph-enrichment default via `KB_EVIDENCE_GRAPH_EXPANSION_ENABLED`) | **MCP adapter** (`mcp_tools.py`) | Uses `mcp_utils` constants. Produces a fully-resolved `EvidenceRequest`. Keeps `src/evidence/` free of any `mcp_server` import. |
| Retrieval orchestration, quote normalization, coverage, gaps, telemetry assembly, optional enhancement | **`EvidenceService`** (`src/evidence/`) | Pure. Depends only on injected callables + duck-typed `deps`. |
| Output budget + partial-state truthfulness | **MCP adapter** | `partial`/`limit_reason` are transport facts; the adapter writes them into `coverage` + appends the `budget_exceeded` gap **after** `_apply_budget`. |
| Diagnostics emission | **MCP adapter** | Needs `ctx` + post-budget values. Never disabled. |
| Trace **semantics** | **`retrieval_trace.py`** (`record_evidence_package`) | Reconstructs all 9 facets from `package.retrieval_metrics`. MCP only calls it + persists. |
| Tool profile / surface | **`TOOL_PROFILES` in `mcp_app.py`** | Single source of truth. Production surface is **unchanged** by this plan. |

### Behavior-parity checklist (MUST hold through Task 8)

The following must hold before/after the adapter swap — **behavior-equivalent except the sanctioned additive evidence-package fields** (`package_id`, `normalized_query`, `gaps`), guarded by the Task 1 → enriched pin:

1. **Fetch depth clamp:** `internal_fetch_k = max(max_quotes, min(retrieval_depth or KB_EVIDENCE_INTERNAL_FETCH_K, KB_EVIDENCE_MAX_FETCH_K))` (cap = 150).
2. **Graph-enrichment default:** `None` → `_coerce_bool(options.get("graph_enrichment"), default=KB_EVIDENCE_GRAPH_EXPANSION_ENABLED)`; explicit value → `_coerce_bool(value, default=False)`.
3. **Backward-compat:** `top_k != KB_SEARCH_DEFAULT_TOP_K and max_quotes == 6` → `max_quotes = top_k`.
4. **Response payload keys:** `quotes[]` (with `quote` text key), `coverage{documents_searched, documents_with_evidence, retrieval_depth, reranker_applied, signal_pool_active, graph_expansion_applied}`, `trace_id`, and — when diagnostics fire — `diagnostic_id` / `diagnostic_hint` / `diagnostic_uri`.
5. **Trace fidelity:** all 9 `record_*` facets (`query`, `signal_pool`, `reranker`, `related_to_expansion`, `graph_enrichment`, `appendix_chunks`, `stage_snapshots`, `colbert`, `evidence_pack`) still populated.

---

## File Structure

Pure evidence core (no `src.mcp_server` imports allowed):

- `src/evidence/__init__.py` — public contract exports.
- `src/evidence/models.py` — request/response/domain models + provenance invariant.
- `src/evidence/quotes.py` — quote normalization + stable IDs.
- `src/evidence/coverage.py` — coverage, gap classification, `mark_budget_state`.
- `src/evidence/service.py` — orchestration over injected callables.
- `src/evidence/enhancer.py` — optional LLM enhancer protocol + fake.
- `src/evidence/serializers.py` — `EvidencePackage` → MCP-safe dict.

MCP + trace (may import `src.evidence`; never the reverse):

- `src/mcp_server/mcp_tools.py` — `kb_retrieve_evidence` becomes a thin adapter; schemas + `_tool_specs()` stay.
- `src/mcp_server/retrieval_trace.py` — gains `record_evidence_package(package)`.
- `src/mcp_server/mcp_app.py` — `TOOL_PROFILES` remains the single profile source (surface unchanged).

Tests (pins + contract invariants; reuse existing fake-deps fixtures):

- `tests/mcp_server_tests/test_evidence_pack_pin.py` (new — golden pins)
- `tests/evidence/test_models.py`, `test_quotes.py`, `test_coverage.py`, `test_service.py`, `test_enhancer.py`, `test_trace_diagnostics.py`, `test_architecture_boundaries.py` (new)
- `tests/mcp_server_tests/test_evidence_tool_adapter.py` (new)
- Existing `tests/mcp_server_tests/test_evidence_pack.py` and `test_tool_profiles.py` are **updated only if a name/key genuinely changed** — they should otherwise keep passing untouched (that is the point of the pin).

---

## Task 1: Pin Current Behavior Before Touching Anything

**Rationale:** GitNexus reports LOW risk for `kb_retrieve_evidence` because it is dynamically dispatched (`mcp_app.py:244`, `full_tool_map[name]`). The static graph cannot protect this refactor. A characterization pin is the real gate.

**Files:**
- Create: `tests/mcp_server_tests/test_evidence_pack_pin.py`

- [ ] **Step 1: Pin the production tool surface (self-contained, no deps)**

```python
# tests/mcp_server_tests/test_evidence_pack_pin.py
from src.mcp_server.mcp_app import TOOL_PROFILES


def test_production_surface_is_evidence_first_and_stable():
    assert TOOL_PROFILES["production"] == {
        "kb.retrieve_evidence",
        "kb.read_excerpt",
        "graph.expand",
    }


def test_no_deprecated_alias_leaks_into_curated_profiles():
    # Underscore aliases carry "[Deprecated" and must never surface in
    # production/analyst list_tools. `full` is the only alias-bearing profile.
    from src.mcp_server.mcp_tools import _tool_specs

    specs = {s["name"]: s for s in _tool_specs()}
    for profile in ("production", "analyst"):
        for name in TOOL_PROFILES[profile]:
            assert name in specs, f"{name} missing from specs"
            assert "[Deprecated" not in specs[name].get("description", "")
```

- [ ] **Step 2: Pin the `kb.retrieve_evidence` response contract**

Exercise the **real** `kb_retrieve_evidence` through a fake context and lock the observable payload shape. Do not assert live retrieval quality — only structural keys and the partial-state invariant.

```python
import pytest

from src.mcp_server import mcp_tools


@pytest.mark.asyncio
async def test_retrieve_evidence_current_shape(evidence_fake_ctx):
    # CHARACTERIZATION of the CURRENT tool (pre-refactor). This MUST pass against
    # today's code — it is the pre-edit baseline. Task 8 intentionally supersedes
    # it with the enriched shape (`test_retrieve_evidence_enriched_shape`).
    payload = await mcp_tools.kb_retrieve_evidence(
        question="How do I configure authentication?",
        ctx=evidence_fake_ctx,
        session_id="pin-session",
    )

    # Legacy top-level keys present today.
    assert {"quotes", "coverage", "session_id", "partial", "limit_reason",
            "meta", "trace_id"} <= payload.keys()
    # The enrichment keys do NOT exist yet — proves this is the pre-edit baseline.
    assert "gaps" not in payload
    assert "package_id" not in payload
    assert "normalized_query" not in payload
    # Legacy coverage has exactly 6 keys and no partial/limit_reason fields yet.
    assert set(payload["coverage"].keys()) == {
        "documents_searched", "documents_with_evidence", "retrieval_depth",
        "reranker_applied", "signal_pool_active", "graph_expansion_applied",
    }
    # Legacy quote fields the refactor MUST preserve (title + uri included today).
    assert payload["quotes"], "fixture must return >=1 quote or these asserts are vacuous"
    for quote in payload["quotes"]:
        assert {"quote", "passage_id", "doc_tag", "title", "uri", "parent_path",
                "confidence", "source", "rank"} <= quote.keys()
    # Internal telemetry must never surface (true before and after).
    assert "retrieval_metrics" not in payload
    assert "diagnostic_context" not in payload
```

> **Fixture requirement (do not skip):** `_get_deps(ctx)` raises `RuntimeError("MCP request context is required")` when `ctx` is `None` and then calls `deps.ensure_initialized()` (see `mcp_utils.py:488`). So `evidence_fake_ctx` MUST provide a request context whose `lifespan_context` is a `Deps` with `query`/`scratch` fakes already attached and `_initialized = True` (so `ensure_initialized()` is a no-op — no live services start). The fake `deps.query.search_sections_light` MUST return at least one deterministic `ChunkResult` (with `text`, `heading`/`title`, `doc_tag`, and `parent_path_norm`) so the handler yields **≥1 quote** — otherwise the quote-field assertions loop over an empty list and pass vacuously. Build it in `tests/mcp_server_tests/conftest.py`, mirroring exactly how `tests/mcp_server_tests/test_evidence_pack.py` already constructs its context. Do not stub `kb_retrieve_evidence` itself — the pin must run the real handler.

- [ ] **Step 3: Run the pins and record baseline**

```bash
pytest tests/mcp_server_tests/test_evidence_pack_pin.py -q
```

Expected: `PASS`. These tests are the acceptance gate for Task 8. Save their output in the task notes.

- [ ] **Step 4: Commit**

```bash
git add tests/mcp_server_tests/test_evidence_pack_pin.py tests/mcp_server_tests/conftest.py
git commit -m "test(evidence): pin kb.retrieve_evidence contract and production surface"
```

---

## Task 2: Define The Evidence Package Contract (with telemetry + provenance invariant)

**Files:**
- Create: `src/evidence/__init__.py`
- Create: `src/evidence/models.py`
- Create: `tests/evidence/test_models.py`

- [ ] **Step 1: Write failing model tests**

```python
# tests/evidence/test_models.py
import pytest
from pydantic import ValidationError

from src.evidence.models import (
    EvidenceAnswerDraft,
    EvidenceClaim,
    EvidenceCoverage,
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


def test_package_round_trips_and_carries_telemetry():
    package = EvidencePackage(
        request=EvidenceRequest(question="q", session_id="s"),
        normalized_query="q",
        quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", text="t", confidence=0.8)],
        coverage=_coverage(),
        retrieval_metrics={"reranker_model": "qwen3", "signal_pool_size": 30},
    )
    data = package.model_dump()
    assert data["retrieval_metrics"]["reranker_model"] == "qwen3"
    # diagnostic_context is internal-only and must never serialize.
    assert "diagnostic_context" not in data


def test_evidence_only_forbids_answer_draft():
    with pytest.raises(ValidationError):
        EvidencePackage(
            request=EvidenceRequest(question="q", session_id="s", response_mode="evidence_only"),
            normalized_query="q",
            quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", text="t", confidence=0.8)],
            coverage=_coverage(),
            answer_draft=EvidenceAnswerDraft(
                markdown="t [q_0001]",
                claims=[EvidenceClaim(claim_id="c1", text="t", quote_ids=["q_0001"], confidence=0.8)],
            ),
        )


def test_draft_claims_must_cite_in_package_quotes():
    with pytest.raises(ValidationError):
        EvidencePackage(
            request=EvidenceRequest(question="q", session_id="s", response_mode="evidence_plus_draft"),
            normalized_query="q",
            quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", text="t", confidence=0.8)],
            coverage=_coverage(),
            answer_draft=EvidenceAnswerDraft(
                markdown="hallucinated [q_9999]",
                claims=[EvidenceClaim(claim_id="c1", text="x", quote_ids=["q_9999"], confidence=0.8)],
            ),
        )


def test_validate_on_assignment_blocks_uncited_draft_added_later():
    package = EvidencePackage(
        request=EvidenceRequest(question="q", session_id="s", response_mode="evidence_plus_draft"),
        normalized_query="q",
        quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", text="t", confidence=0.8)],
        coverage=_coverage(),
    )
    with pytest.raises(ValidationError):
        package.answer_draft = EvidenceAnswerDraft(
            markdown="x [q_9999]",
            claims=[EvidenceClaim(claim_id="c1", text="x", quote_ids=["q_9999"], confidence=0.8)],
        )
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/evidence/test_models.py -q
```

Expected: import failure for `src.evidence.models`.

- [ ] **Step 3: Implement the contract**

```python
# src/evidence/models.py
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
    retrieval_depth: int = 60          # already clamped by the adapter
    graph_enrichment: bool = False      # already resolved by the adapter
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
    title: Optional[str] = None          # legacy quote field (from _quote_from_passage)
    parent_path: List[str] = Field(default_factory=list)
    source_uri: Optional[str] = None     # legacy `uri` field
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
    # validate_assignment ensures the provenance invariant re-runs when the
    # enhancer assigns answer_draft after construction.
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

    # Internal telemetry superset — lets retrieval_trace.py rebuild the full
    # 9-facet trace. Serialized in model_dump() but NOT emitted to MCP clients
    # (the serializer picks fields explicitly).
    retrieval_metrics: Dict[str, Any] = Field(default_factory=dict)
    # Never serialized anywhere; carried so the MCP adapter can emit diagnostics.
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
                raise ValueError(f"claim {claim.claim_id} cites unknown quotes: {missing}")
        return self
```

```python
# src/evidence/__init__.py
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

- [ ] **Step 4: Run and commit**

```bash
pytest tests/evidence/test_models.py -q      # expect: 4 passed
git add src/evidence/__init__.py src/evidence/models.py tests/evidence/test_models.py
git commit -m "feat(evidence): add package contract with telemetry + citation invariant"
```

---

## Task 3: Quote Normalization And Stable IDs

**Files:**
- Create: `src/evidence/quotes.py`
- Create: `tests/evidence/test_quotes.py`
- Read (for parity): `src/mcp_server/mcp_search.py:250-360` (`_extract_evidence_from_passages` output shape)

- [ ] **Step 1: Write failing tests**

```python
# tests/evidence/test_quotes.py
from src.evidence.quotes import normalize_quote_payloads


def test_stable_ids_parent_path_and_legacy_fields():
    quotes = normalize_quote_payloads([
        {
            "rank": 1, "passage_id": "p1", "section_id": "s1", "doc_tag": "nci/aos",
            "title": "AOS Storage", "uri": "nutanixdocs://scratch/sess/p1",
            "parent_path": "NCI > AOS Storage",
            "quote": "AOS provides distributed storage.",
            "confidence": 0.81, "score": 0.9, "source": "retrieval",
        }
    ])
    assert quotes[0].quote_id == "q_0001"
    assert quotes[0].parent_path == ["NCI", "AOS Storage"]
    assert quotes[0].retrieval_signals == {"score": 0.9}
    # Legacy quote fields must survive normalization (no client-visible loss).
    assert quotes[0].title == "AOS Storage"
    assert quotes[0].source_uri == "nutanixdocs://scratch/sess/p1"


def test_empty_text_is_skipped_and_ids_stay_contiguous():
    quotes = normalize_quote_payloads([
        {"rank": 1, "passage_id": "p1", "quote": "   ", "confidence": 0.9},
        {"rank": 2, "passage_id": "p2", "quote": "Useful.", "confidence": 0.7},
    ])
    assert [q.quote_id for q in quotes] == ["q_0001"]
    assert quotes[0].passage_id == "p2"
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/evidence/test_quotes.py -q
```

- [ ] **Step 3: Implement**

```python
# src/evidence/quotes.py
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
        signals = dict(raw.get("retrieval_signals") or {})
        if score is not None and "score" not in signals:
            signals["score"] = score
        quotes.append(
            EvidenceQuote(
                quote_id=f"q_{len(quotes) + 1:04d}",
                rank=int(raw.get("rank") or len(quotes) + 1),
                passage_id=str(raw.get("passage_id") or raw.get("section_id") or ""),
                section_id=raw.get("section_id"),
                doc_tag=raw.get("doc_tag"),
                title=raw.get("title"),
                parent_path=_split_parent_path(raw.get("parent_path")),
                source_uri=raw.get("source_uri") or raw.get("uri"),
                text=text,
                confidence=float(raw.get("confidence") or 0.0),
                score=score,
                source=str(raw.get("source") or "retrieval"),
                retrieval_signals=signals,
                context_before=raw.get("context_before"),
                context_after=raw.get("context_after"),
            )
        )
    return quotes
```

- [ ] **Step 4: Run and commit**

```bash
pytest tests/evidence/test_quotes.py -q      # expect: 2 passed
git add src/evidence/quotes.py tests/evidence/test_quotes.py
git commit -m "feat(evidence): normalize quotes with stable ids"
```

---

## Task 4: Coverage, Gaps, And Post-Budget Partial-State Marking

**Files:**
- Create: `src/evidence/coverage.py`
- Create: `tests/evidence/test_coverage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/evidence/test_coverage.py
from src.evidence.coverage import build_coverage, identify_gaps, mark_budget_state
from src.evidence.models import EvidenceCoverage, EvidenceQuote


def test_build_coverage_counts_docs_and_flags():
    coverage = build_coverage(
        search_results=[
            {"doc_tag": "nci/aos"}, {"doc_tag": "nci/aos"}, {"doc_tag": "ncm/ops"},
        ],
        quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", doc_tag="nci/aos", text="e", confidence=0.8)],
        metrics={"reranker_applied": True, "signal_pool_used": True},
        graph_expansion_applied=True,
    )
    assert coverage.documents_searched == 2
    assert coverage.documents_with_evidence == 1
    assert coverage.retrieval_depth == 3
    assert coverage.partial is False and coverage.limit_reason == "none"


def test_identify_gaps_flags_empty_and_missing_live_validation():
    gaps = identify_gaps(quotes=[], documents_searched=4, live_validation_available=False)
    assert [g.kind for g in gaps] == ["insufficient_evidence", "missing_live_validation"]


def test_mark_budget_state_sets_coverage_and_appends_gap():
    coverage = EvidenceCoverage(
        documents_searched=2, documents_with_evidence=1, retrieval_depth=10,
        reranker_applied=True, signal_pool_active=True, graph_expansion_applied=False,
    )
    gaps = []
    mark_budget_state(coverage, gaps, partial=True, limit_reason="token_cap")
    assert coverage.partial is True and coverage.limit_reason == "token_cap"
    assert gaps[-1].kind == "budget_exceeded"
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/evidence/test_coverage.py -q
```

- [ ] **Step 3: Implement**

```python
# src/evidence/coverage.py
from __future__ import annotations

from typing import Any, Iterable, List

from src.evidence.models import EvidenceCoverage, EvidenceGap, EvidenceQuote


def build_coverage(
    *,
    search_results: Iterable[dict[str, Any]],
    quotes: Iterable[EvidenceQuote],
    metrics: dict[str, Any],
    graph_expansion_applied: bool,
) -> EvidenceCoverage:
    search_list = list(search_results)
    quote_list = list(quotes)
    searched = {r.get("doc_tag") for r in search_list if r.get("doc_tag")}
    with_evidence = {q.doc_tag for q in quote_list if q.doc_tag}
    return EvidenceCoverage(
        documents_searched=len(searched),
        documents_with_evidence=len(with_evidence),
        retrieval_depth=len(search_list),
        reranker_applied=bool(metrics.get("reranker_applied")),
        signal_pool_active=bool(
            metrics.get("signal_pool_used") or metrics.get("signal_pool_enabled")
        ),
        graph_expansion_applied=graph_expansion_applied,
        # partial/limit_reason intentionally left at neutral defaults;
        # the MCP adapter is the sole writer of the truthful value.
    )


def identify_gaps(
    *,
    quotes: Iterable[EvidenceQuote],
    documents_searched: int,
    live_validation_available: bool,
) -> List[EvidenceGap]:
    gaps: List[EvidenceGap] = []
    if not list(quotes):
        gaps.append(EvidenceGap(
            kind="insufficient_evidence",
            message=f"No evidence quotes were selected from {documents_searched} searched documents.",
            severity="warning",
        ))
    if not live_validation_available:
        gaps.append(EvidenceGap(
            kind="missing_live_validation",
            message="Live embedding, reranker, Qdrant, and Neo4j validation were not run for this package.",
            severity="info",
        ))
    return gaps


def mark_budget_state(
    coverage: EvidenceCoverage,
    gaps: List[EvidenceGap],
    *,
    partial: bool,
    limit_reason: str,
) -> None:
    """Called by the MCP adapter AFTER output budgeting, so coverage and the
    top-level payload never disagree about the output-budget state.

    Note: `_apply_budget` only *estimates* size and flags `partial`; it does NOT
    trim the payload (matching the legacy tool). This marks a budget-exceeded
    state, not an actual truncation of quotes."""
    coverage.partial = partial
    coverage.limit_reason = limit_reason if partial else "none"
    if partial:
        gaps.append(EvidenceGap(
            kind="budget_exceeded",
            message=f"Response exceeds the configured output budget ({limit_reason}).",
            severity="warning",
        ))
```

- [ ] **Step 4: Run and commit**

```bash
pytest tests/evidence/test_coverage.py -q     # expect: 3 passed
git add src/evidence/coverage.py tests/evidence/test_coverage.py
git commit -m "feat(evidence): coverage, gaps, and post-budget partial-state marking"
```

---

## Task 5: Extract The (Pure) EvidenceService

**Files:**
- Create: `src/evidence/service.py`
- Create: `tests/evidence/test_service.py`

- [ ] **Step 1: GitNexus impact (with caveat)**

```bash
npx gitnexus impact kb_retrieve_evidence --direction upstream --include-tests
npx gitnexus impact _extract_evidence_from_passages --direction upstream --include-tests
```

> **Caveat:** `kb_retrieve_evidence` will report LOW/0 because it is dynamically dispatched. That is expected and does **not** mean the change is safe. The Task 1 pin is the gate. Record the blast radius for `_extract_evidence_from_passages` (a real static callee) and report if HIGH/CRITICAL.

- [ ] **Step 2: Write failing service test**

```python
# tests/evidence/test_service.py
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
            "results": [{"passage_id": "p1", "section_id": "s1", "doc_tag": "nci/aos", "title": "AOS", "score": 0.9, "rank": 1}],
            "metrics": {"reranker_applied": True, "signal_pool_used": True, "query_rewrite_result": "configure AOS"},
        },
        {"metrics": {"reranker_applied": True}, "chunks": []},
    )


async def fake_extract_quotes(**kwargs):
    return [{
        "rank": 1, "passage_id": "p1", "section_id": "s1", "doc_tag": "nci/aos",
        "parent_path": "NCI > AOS", "quote": "AOS provides distributed storage.",
        "confidence": 0.82, "score": 0.9, "source": "retrieval",
    }]


@pytest.mark.asyncio
async def test_service_builds_package_with_telemetry_and_diag_context():
    service = EvidenceService(
        search_candidates=fake_search_candidates,
        extract_quotes=fake_extract_quotes,
        expand_with_graph=None,
        live_validation_available=False,
        enhancer=None,
    )
    package = await service.build_package(
        request=EvidenceRequest(question="How does AOS work?", session_id="s1", max_quotes=2, retrieval_depth=10),
        deps=FakeDeps(),
    )
    assert package.normalized_query == "configure AOS"
    assert package.quotes[0].quote_id == "q_0001"
    assert package.coverage.documents_searched == 1
    assert package.retrieval_metrics.get("reranker_applied") is True
    assert package.diagnostic_context is not None       # carried for the adapter
    assert package.gaps[0].kind == "missing_live_validation"


@pytest.mark.asyncio
async def test_service_uses_prekclamped_request_depth_verbatim():
    seen = {}

    async def capture_search(**kwargs):
        seen["top_k"] = kwargs["top_k"]
        return ({"results": [], "metrics": {}}, {})

    service = EvidenceService(
        search_candidates=capture_search, extract_quotes=fake_extract_quotes,
        expand_with_graph=None, live_validation_available=False, enhancer=None,
    )
    # The adapter is responsible for clamping; the service must NOT re-clamp.
    await service.build_package(
        request=EvidenceRequest(question="q", session_id="s", max_quotes=3, retrieval_depth=150),
        deps=FakeDeps(),
    )
    assert seen["top_k"] == 150      # max(3, 150); no hidden re-clamp
```

- [ ] **Step 3: Run and verify failure**

```bash
pytest tests/evidence/test_service.py -q
```

- [ ] **Step 4: Implement the pure service**

```python
# src/evidence/service.py
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional

from src.evidence.coverage import build_coverage, identify_gaps
from src.evidence.models import EvidencePackage, EvidenceRequest
from src.evidence.quotes import normalize_quote_payloads

SearchCandidatesFn = Callable[..., Awaitable[tuple[dict[str, Any], dict[str, Any]]]]
ExtractQuotesFn = Callable[..., Awaitable[list[dict[str, Any]]]]
ExpandGraphFn = Callable[..., Awaitable[list[str]]]


class EvidenceService:
    """Pure orchestration. No src.mcp_server imports. All I/O is injected.

    The request handed in is already fully resolved by the caller (the MCP
    adapter): retrieval_depth is clamped, graph_enrichment is a concrete bool.
    """

    def __init__(
        self,
        *,
        search_candidates: SearchCandidatesFn,
        extract_quotes: ExtractQuotesFn,
        expand_with_graph: Optional[ExpandGraphFn],
        live_validation_available: bool,
        enhancer: Optional["EvidenceEnhancer"] = None,  # noqa: F821 (Task 9)
    ) -> None:
        self._search_candidates = search_candidates
        self._extract_quotes = extract_quotes
        self._expand_with_graph = expand_with_graph
        self._live_validation_available = live_validation_available
        self._enhancer = enhancer

    async def build_package(self, *, request: EvidenceRequest, deps: Any) -> EvidencePackage:
        internal_fetch_k = max(request.max_quotes, request.retrieval_depth)
        options = dict(request.options)
        options.setdefault("max_per_doc", 5)

        search_payload, diagnostic_context = await self._search_candidates(
            query=request.question,
            top_k=internal_fetch_k,
            cursor=None,
            page_size=internal_fetch_k,
            scope=request.scope,
            filters=request.filters,
            options=options,
            deps=deps,
            effective_session=request.session_id,
            _fetch_k_override=internal_fetch_k,
        )

        results = search_payload.get("results", [])
        metrics = dict(search_payload.get("metrics") or {})
        passage_ids = [r["passage_id"] for r in results if r.get("passage_id")]

        graph_expansion_applied = False
        graph_seed_count = 0
        graph_neighbors_added = 0
        if request.graph_enrichment and self._expand_with_graph:
            section_ids = [r["section_id"] for r in results if r.get("section_id")]
            graph_seed_count = min(10, len(section_ids))  # match legacy trace seed cap
            graph_passage_ids = await self._expand_with_graph(
                section_ids=section_ids, deps=deps, effective_session=request.session_id,
            )
            passage_ids.extend(graph_passage_ids)
            graph_neighbors_added = len(graph_passage_ids)
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
            search_results=results, quotes=quotes, metrics=metrics,
            graph_expansion_applied=graph_expansion_applied,
        )
        gaps = identify_gaps(
            quotes=quotes,
            documents_searched=coverage.documents_searched,
            live_validation_available=self._live_validation_available,
        )

        # Telemetry superset: everything retrieval_trace.py needs for 9 facets.
        # Appendix mirrors the legacy tool: the skeleton (rerank_score, heading,
        # order) comes from the result rows; text/parent_path_norm are hydrated
        # from scratch. This preserves fields the earlier draft dropped.
        appendix = []
        for result in results[:20]:
            pid = result.get("passage_id")
            entry = await deps.scratch.get(request.session_id, pid) if pid else None
            appendix.append({
                "chunk_id": result.get("section_id", ""),
                "rerank_score": result.get("score"),
                "doc_tag": result.get("doc_tag"),
                "heading": result.get("title", ""),
                "parent_path_norm": (entry or {}).get("parent_path_norm"),
                "text": (entry or {}).get("text", ""),
            })
        metrics["_appendix_chunks"] = appendix
        metrics["_result_snapshot"] = results[:20]      # top-20 detail for top_results
        metrics["_result_count"] = len(results)         # full count for reranker output_count
        metrics["_graph_expansion_applied"] = graph_expansion_applied
        metrics["_graph_seed_count"] = graph_seed_count
        metrics["_graph_neighbors_added"] = graph_neighbors_added

        package = EvidencePackage(
            request=request,
            normalized_query=metrics.get("query_rewrite_result") or request.question,
            quotes=quotes,
            coverage=coverage,
            gaps=gaps,
            retrieval_metrics=metrics,
            diagnostic_context=diagnostic_context,
        )

        if request.response_mode == "evidence_plus_draft" and self._enhancer:
            package = await self._safe_enhance(package)
        return package

    async def _safe_enhance(self, package: EvidencePackage) -> EvidencePackage:
        # Filled in by Task 9. Placeholder keeps evidence_only behavior intact.
        return package
```

> **Parity note for the implementer:** the service deliberately does **not** clamp `retrieval_depth` or resolve the graph-enrichment default. Those are the adapter's job (Task 8). The service trusts a resolved request. This is what keeps `src/evidence/` free of `mcp_utils` imports.

- [ ] **Step 5: Run and commit**

```bash
pytest tests/evidence/test_service.py -q      # expect: 2 passed
git add src/evidence/service.py tests/evidence/test_service.py
git commit -m "feat(evidence): pure evidence service over injected retrieval callables"
```

---

## Task 6: Teach The Trace Builder To Record An EvidencePackage (full 9 facets)

**Files:**
- Modify: `src/mcp_server/retrieval_trace.py` (add method near existing `record_*`, ~line 178)
- Create: `tests/evidence/test_trace_diagnostics.py`

- [ ] **Step 1: GitNexus impact**

```bash
npx gitnexus impact record_evidence_pack --direction upstream --include-tests
```

Report blast radius. `record_evidence_pack` is called by the current `kb_retrieve_evidence`; the new method wraps the existing `record_*` methods without changing them.

- [ ] **Step 2: Write failing test**

```python
# tests/evidence/test_trace_diagnostics.py
from src.evidence.models import EvidenceCoverage, EvidencePackage, EvidenceQuote, EvidenceRequest
from src.mcp_server.retrieval_trace import RetrievalTraceBuilder


def _package():
    return EvidencePackage(
        request=EvidenceRequest(question="How does NC2 work?", session_id="s1"),
        normalized_query="How does NC2 work?",
        quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", doc_tag="nc2/aws", text="NC2 runs Nutanix Cloud Clusters.", confidence=0.83)],
        coverage=EvidenceCoverage(
            documents_searched=2, documents_with_evidence=1, retrieval_depth=10,
            reranker_applied=True, signal_pool_active=True, graph_expansion_applied=False,
        ),
        retrieval_metrics={
            "reranker_model": "qwen3-reranker", "reranker_input_count": 10,
            "signal_pool_used": True, "signal_pool_size": 30,
            "colbert_rerank_applied": True,
            "related_to_docs_found": 3, "related_to_chunks_added": 5,
            "_graph_seed_count": 4, "_graph_neighbors_added": 6,
            "snapshot_post_reranker": [{"chunk_id": "s1", "score": 0.9}],
            "_result_snapshot": [{"section_id": "s1", "title": "NC2", "score": 0.9, "rank": 1}],
            "_result_count": 42,   # deliberately > len(_result_snapshot) to prove output_count parity
            "_appendix_chunks": [{"chunk_id": "s1", "rerank_score": 0.9, "heading": "NC2",
                                  "text": "NC2 runs Nutanix Cloud Clusters.", "doc_tag": "nc2/aws",
                                  "parent_path_norm": None}],
        },
    )


def test_record_evidence_package_populates_all_facets():
    trace = RetrievalTraceBuilder(trace_id="trace-1", session_id="s1")
    trace.record_evidence_package(_package())

    rendered = trace.format()
    assert "NC2 runs Nutanix Cloud Clusters" in rendered       # evidence pack facet
    assert "qwen3-reranker" in rendered                         # reranker facet
    # All nine facets must be recorded (proves nothing collapsed to 2).
    assert trace._query is not None                            # 1 query
    assert trace._signal_pool is not None                      # 2 signal pool
    assert trace._reranker is not None                         # 3 reranker
    assert trace._reranker["output_count"] == 42               #   full count, NOT the capped snapshot (len 1)
    assert trace._related_to is not None                       # 4 RELATED_TO
    assert trace._graph_enrichment is not None                 # 5 graph enrichment
    assert trace._graph_enrichment["neighbors_added"] == 6     #   nonzero neighbors
    assert trace._appendix_chunks                              # 6 appendix
    assert trace._appendix_chunks[0]["rerank_score"] == 0.9    #   score preserved
    assert trace._stage_snapshots is not None                 # 7 stage snapshots
    assert trace._colbert is not None                          # 8 ColBERT
    assert trace._evidence_pack is not None                    # 9 evidence pack
```

- [ ] **Step 3: Run and verify failure**

```bash
pytest tests/evidence/test_trace_diagnostics.py -q
```

- [ ] **Step 4: Implement `record_evidence_package`**

Add to `RetrievalTraceBuilder` in `src/mcp_server/retrieval_trace.py`. Import `EvidencePackage` at the top (`from src.evidence.models import EvidencePackage` — this is the allowed mcp_server → evidence direction). Reconstruct every facet from `package.retrieval_metrics`, mirroring the current inline calls in `mcp_tools.py:1206-1377`:

```python
    def record_evidence_package(self, package: "EvidencePackage") -> None:
        m = package.retrieval_metrics or {}
        self.record_query(
            client_query=m.get("query_rewrite_original", package.request.question),
            reformulated=package.normalized_query,
            method=m.get("query_rewrite_method", "passthrough") if m.get("query_rewrite_applied") else "passthrough",
            latency_ms=m.get("query_rewrite_latency_ms", 0),
            dual_query_active=bool(m.get("dual_query_active")),
        )
        self.record_signal_pool(
            enabled=bool(m.get("signal_pool_used", m.get("signal_pool_enabled", False))),
            pool_size=int(m.get("signal_pool_size", 0)),
            slot_fills=m.get("signal_pool_slot_fills", {}),
            degraded=bool(m.get("signal_pool_degraded")),
        )
        snapshot = m.get("_result_snapshot", [])
        self.record_reranker(
            model=m.get("reranker_model", "unknown"),
            instruction=m.get("reranker_instruction"),
            input_count=int(m.get("reranker_input_count", 0)),
            output_count=int(m.get("_result_count", len(snapshot))),  # full count, not capped
            latency_ms=m.get("reranker_time_ms", m.get("rerank_time_ms", 0)),
            top_results=[
                {"chunk_id": r.get("section_id", ""), "score": r.get("score", 0),
                 "heading": r.get("title", ""), "rank": r.get("rank", 0), "original_rank": i + 1}
                for i, r in enumerate(snapshot[:10])
            ],
        )
        self.record_appendix_chunks([
            {"chunk_id": c.get("chunk_id", ""), "rerank_score": c.get("rerank_score"),
             "doc_tag": c.get("doc_tag"), "parent_path_norm": c.get("parent_path_norm"),
             "heading": c.get("heading", ""), "text": c.get("text", "")}
            for c in m.get("_appendix_chunks", [])
        ])
        self.record_related_to_expansion(
            seed_docs=int(m.get("related_to_seed_docs", 0)),
            related_docs_found=int(m.get("related_to_docs_found", 0)),
            chunks_added=int(m.get("related_to_chunks_added", 0)),
            avg_edge_score=float(m.get("related_to_avg_edge_score", 0)),
            blended_count=int(m.get("related_to_blended", 0)),
            blend_lambda=float(m.get("related_to_lambda", 0)),
        )
        self.record_graph_enrichment(
            seeds=int(m.get("_graph_seed_count", 0)),
            neighbors_added=int(m.get("_graph_neighbors_added", 0)),
            neighbor_details=[],
        )
        snapshots = {
            k.replace("snapshot_", ""): m[k]
            for k in ("snapshot_post_fusion", "snapshot_post_entity_boost",
                      "snapshot_post_structural_boost", "snapshot_post_colbert",
                      "snapshot_post_reranker")
            if k in m
        }
        if snapshots:
            self.record_stage_snapshots(snapshots)
        self.record_colbert(
            applied=bool(m.get("colbert_rerank_applied")),
            runtime_available=bool(m.get("colbert_runtime_available", False)),
            query_embedding_ok=bool(m.get("colbert_query_embedding_ok", False)),
            rank_deltas=m.get("colbert_rank_delta_top10"),
            candidates=int(m.get("colbert_candidates", 0)),
            hydrated=int(m.get("colbert_hydrated", 0)),
            latency_ms=m.get("colbert_rerank_time_ms", 0),
        )
        self.record_evidence_pack(
            quotes=[
                TraceQuote(rank=q.rank, confidence=q.confidence, doc_tag=q.doc_tag,
                           parent_path=" > ".join(q.parent_path), source=q.source, text=q.text)
                for q in package.quotes
            ],
            coverage=package.coverage.model_dump(),
        )
```

- [ ] **Step 5: Run and commit**

```bash
pytest tests/evidence/test_trace_diagnostics.py -q
npx gitnexus detect-changes
git add src/mcp_server/retrieval_trace.py tests/evidence/test_trace_diagnostics.py
git commit -m "feat(trace): record full retrieval trace from EvidencePackage telemetry"
```

---

## Task 7: MCP Serializer (additive enrichment, no null leaks; shape locked by the pin)

**Files:**
- Create: `src/evidence/serializers.py`
- Create: `tests/mcp_server_tests/test_evidence_tool_adapter.py`

- [ ] **Step 1: Write failing serializer test**

```python
# tests/mcp_server_tests/test_evidence_tool_adapter.py
from src.evidence.models import EvidenceCoverage, EvidencePackage, EvidenceQuote, EvidenceRequest
from src.evidence.serializers import evidence_package_to_mcp_payload


def _package(**over):
    base = dict(
        request=EvidenceRequest(question="How do I configure NAI?", session_id="s1"),
        normalized_query="configure NAI",
        quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", doc_tag="nai/deploy",
                              title="Deploy", parent_path=["NAI", "Deploy"],
                              source_uri="nutanixdocs://scratch/s1/p1",
                              text="Deploy models through NAI.", confidence=0.84)],
        coverage=EvidenceCoverage(documents_searched=4, documents_with_evidence=1, retrieval_depth=20,
                                  reranker_applied=True, signal_pool_active=False, graph_expansion_applied=False),
        trace_id="trace-1",
    )
    base.update(over)
    return EvidencePackage(**base)


def test_payload_is_additive_and_omits_null_fields():
    payload = evidence_package_to_mcp_payload(_package())
    assert payload["quotes"][0]["quote"] == "Deploy models through NAI."
    assert payload["quotes"][0]["parent_path"] == "NAI > Deploy"
    # Legacy quote fields preserved (no client-visible regression).
    assert payload["quotes"][0]["title"] == "Deploy"
    assert payload["quotes"][0]["uri"] == "nutanixdocs://scratch/s1/p1"
    assert payload["coverage"]["documents_with_evidence"] == 1
    assert payload["trace_id"] == "trace-1"
    # Declared additive enrichment (intentional; locked by the Task 1 pin).
    assert payload["package_id"].startswith("ep_")
    assert payload["normalized_query"] == "configure NAI"
    assert payload["gaps"] == []
    # No null leaks: absent optional fields are OMITTED, not emitted as None.
    # (diagnostic_id/diagnostic_uri are typed `string` in KB_EVIDENCE_OUTPUT_SCHEMA.)
    assert "answer_draft" not in payload
    assert "diagnostic_id" not in payload
    assert "diagnostic_uri" not in payload
    # Internal telemetry must never leak to clients.
    assert "retrieval_metrics" not in payload
    assert "diagnostic_context" not in payload


def test_payload_includes_diagnostics_and_draft_only_when_present():
    package = _package()
    package.diagnostic_id = "diag-1"
    package.diagnostic_uri = "nutanixdocs://diagnostics/2026-07-04/diag-1"
    payload = evidence_package_to_mcp_payload(package)
    assert payload["diagnostic_id"] == "diag-1"
    assert payload["diagnostic_uri"].endswith("diag-1")
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/mcp_server_tests/test_evidence_tool_adapter.py -q
```

- [ ] **Step 3: Implement**

```python
# src/evidence/serializers.py
from __future__ import annotations

from typing import Any, Dict

from src.evidence.models import EvidencePackage


def evidence_package_to_mcp_payload(package: EvidencePackage) -> Dict[str, Any]:
    # Additive over the legacy payload (quotes/coverage/trace_id). package_id,
    # normalized_query, and gaps are intentional enrichments. Optional fields are
    # OMITTED when absent — never emitted as null (they are typed `string`/object
    # in KB_EVIDENCE_OUTPUT_SCHEMA, so a null value would violate the contract).
    payload: Dict[str, Any] = {
        "package_id": package.package_id,
        "normalized_query": package.normalized_query,
        "quotes": [
            {
                "quote_id": q.quote_id,
                "rank": q.rank,
                "passage_id": q.passage_id,
                "section_id": q.section_id,
                "doc_tag": q.doc_tag,
                "title": q.title,                       # legacy field — preserved
                "parent_path": " > ".join(q.parent_path),
                "uri": q.source_uri,                    # legacy field — preserved
                "quote": q.text,
                "confidence": q.confidence,
                "score": q.score,
                "source": q.source,
                "retrieval_signals": q.retrieval_signals,
            }
            for q in package.quotes
        ],
        "coverage": package.coverage.model_dump(),
        "gaps": [g.model_dump() for g in package.gaps],
    }
    if package.answer_draft is not None:
        payload["answer_draft"] = package.answer_draft.model_dump()
    if package.trace_id is not None:
        payload["trace_id"] = package.trace_id
    if package.diagnostic_id is not None:
        payload["diagnostic_id"] = package.diagnostic_id
    if package.diagnostic_uri is not None:
        payload["diagnostic_uri"] = package.diagnostic_uri
    return payload
```

- [ ] **Step 4: Run and commit**

```bash
pytest tests/mcp_server_tests/test_evidence_tool_adapter.py -q
git add src/evidence/serializers.py tests/mcp_server_tests/test_evidence_tool_adapter.py
git commit -m "feat(evidence): additive MCP serializer with no null-field leaks"
```

---

## Task 8: Convert `kb_retrieve_evidence` Into A Thin Adapter (the linchpin)

**This is the only task that changes production behavior of `kb.retrieve_evidence`. It performs the one sanctioned pin replacement: the legacy Task 1 pin is expected to fail here and is superseded by the enriched pin (Step 5). Keep the enriched pin green from this point on.**

**Files:**
- Modify: `src/mcp_server/mcp_tools.py:1131-1420` (`kb_retrieve_evidence`)
- Modify: `src/mcp_server/mcp_tools.py` imports (add `src.evidence.*`)

- [ ] **Step 1: GitNexus impact + re-read the pin**

```bash
npx gitnexus impact kb_retrieve_evidence --direction upstream --include-tests
pytest tests/mcp_server_tests/test_evidence_pack_pin.py -q   # must be green BEFORE editing
```

- [ ] **Step 2: Add imports at the top of `mcp_tools.py`**

```python
from src.evidence.coverage import mark_budget_state
from src.evidence.models import EvidenceRequest
from src.evidence.serializers import evidence_package_to_mcp_payload
from src.evidence.service import EvidenceService
```

- [ ] **Step 3: Add two adapter-local helpers**

Place near the existing trace helpers. `_write_evidence_trace` now delegates trace *semantics* to `retrieval_trace.py`:

```python
def _write_evidence_trace(package) -> str:
    trace = RetrievalTraceBuilder(trace_id=uuid4().hex, session_id=package.request.session_id)
    trace.record_evidence_package(package)
    write_trace(trace)
    set_active_trace(package.request.session_id, trace)
    return trace.trace_id


def _build_evidence_service() -> EvidenceService:
    return EvidenceService(
        search_candidates=_kb_search_candidates,
        extract_quotes=_extract_evidence_from_passages,
        expand_with_graph=_expand_evidence_with_structure,
        live_validation_available=False,
        enhancer=None,          # wired in Task 9
    )
```

- [ ] **Step 4: Replace the body of `kb_retrieve_evidence` (keep the signature)**

Keep the exact public signature (`question, top_k, max_quotes, ..., ctx`). Replace the orchestration body. **Preserve behavior-parity items 1–3 here in the adapter:**

```python
    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")
    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)

    # Parity #3: backward-compat top_k -> max_quotes
    if top_k != KB_SEARCH_DEFAULT_TOP_K and max_quotes == 6:
        max_quotes = top_k

    # Parity #1: fetch-depth clamp (adapter owns mcp_utils constants)
    internal_fetch_k = max(
        max_quotes,
        min(int(retrieval_depth or KB_EVIDENCE_INTERNAL_FETCH_K), KB_EVIDENCE_MAX_FETCH_K),
    )

    # Parity #2: graph-enrichment default resolution
    evidence_options = dict(options or {})
    if graph_enrichment is None:
        graph_enabled = _coerce_bool(
            evidence_options.get("graph_enrichment"),
            default=KB_EVIDENCE_GRAPH_EXPANSION_ENABLED,
        )
    else:
        graph_enabled = _coerce_bool(graph_enrichment, default=False)

    request = EvidenceRequest(
        question=question,
        session_id=effective_session,
        top_k=top_k,
        max_quotes=max_quotes,
        max_quote_tokens=max_quote_tokens,
        include_context_tokens=include_context_tokens,
        retrieval_depth=internal_fetch_k,     # already clamped
        graph_enrichment=graph_enabled,        # already resolved
        scope=scope,
        filters=filters,
        options=evidence_options,
        response_mode="evidence_only",         # Task 9 makes this a parameter
    )

    package = await _build_evidence_service().build_package(request=request, deps=deps)

    # Serialize a provisional payload, then apply the MCP output budget.
    payload = evidence_package_to_mcp_payload(package)
    budget = _new_budget()
    tokens_estimate, bytes_estimate, partial, reason = _apply_budget(payload, budget, "snippets")
    limit_reason = reason if partial else "none"

    # Reconcile the budget/partial state INTO the package BEFORE tracing, so the trace, the
    # coverage view, and the top-level flags all report the same thing.
    mark_budget_state(package.coverage, package.gaps, partial=partial, limit_reason=limit_reason)
    payload["coverage"] = package.coverage.model_dump()
    payload["gaps"] = [g.model_dump() for g in package.gaps]

    # Trace now records the truthful (post-budget) coverage.
    package.trace_id = _write_evidence_trace(package)

    # Diagnostics stay LIVE, with post-budget values.
    diagnostic = await _emit_diagnostics(
        tool_name="kb_retrieve_evidence",
        ctx=ctx,
        session_id=effective_session,
        diagnostic_context=package.diagnostic_context or {},
        tokens_estimate=tokens_estimate,
        bytes_estimate=bytes_estimate,
        partial=partial,
        limit_reason=limit_reason,
    )
    if diagnostic and diagnostic.get("diagnostic_id"):
        payload["diagnostic_id"] = diagnostic["diagnostic_id"]
        payload["diagnostic_hint"] = f"See retrieval diagnostics {diagnostic['diagnostic_id']}"
        if DIAGNOSTICS_RESOURCES_ENABLED and diagnostic.get("date"):
            payload["diagnostic_uri"] = _diagnostics_uri(diagnostic["date"], diagnostic["diagnostic_id"])

    finalized = _finalize_payload(
        "kb_retrieve_evidence", payload,
        tokens=tokens_estimate, bytes_=bytes_estimate,
        partial=partial, limit_reason=limit_reason, session_id=effective_session,
    )
    finalized["trace_id"] = package.trace_id
    return finalized
```

> **Note on ordering:** budget runs on the serialized bytes, so it must happen after serialization — but the budget state is reconciled into the *package* (via `mark_budget_state`, Task 4) *before* the trace is written, and coverage/gaps are then re-synced into the budgeted `payload`. This guarantees the trace, `coverage.partial`, and the top-level `partial` never disagree. Important: `_apply_budget` only *estimates* size and flags `partial` — it does **not** trim `payload["quotes"]` (this matches the legacy tool), so re-emitting the small coverage/gaps objects afterward changes nothing else. If real trimming is ever wanted, that is a deliberate behavior change for a separate task.

- [ ] **Step 5: Supersede the pin with the enriched shape (the one sanctioned pin change)**

The Task 1 pin (`test_retrieve_evidence_current_shape`) characterizes the *legacy* payload, so it will now fail — the adapter intentionally enriches the shape. This is the **only** sanctioned pin change in the whole plan. Replace it with the enriched characterization, which becomes the guard from here on. It still asserts every field that must NOT regress (legacy quote `title`/`uri`, no telemetry leak, single partial-state signal):

```python
@pytest.mark.asyncio
async def test_retrieve_evidence_enriched_shape(evidence_fake_ctx):
    payload = await mcp_tools.kb_retrieve_evidence(
        question="How do I configure authentication?",
        ctx=evidence_fake_ctx,
        session_id="pin-session",
    )
    # Legacy keys preserved + declared additive enrichment.
    assert {
        "quotes", "coverage", "gaps", "package_id", "normalized_query",
        "trace_id", "session_id", "partial", "limit_reason", "meta",
    } <= payload.keys()
    # No null leaks (diagnostic_id/uri are typed `string` in the output schema).
    assert [k for k, v in payload.items() if v is None] == []
    # Coverage now carries the partial-state fields; single truthful signal.
    cov = payload["coverage"]
    assert {"partial", "limit_reason"} <= cov.keys()
    assert payload["partial"] == cov["partial"]
    # Legacy quote fields MUST still be present — no client-visible regression.
    assert payload["quotes"], "fixture must return >=1 quote or these asserts are vacuous"
    for quote in payload["quotes"]:
        assert {"quote", "passage_id", "doc_tag", "title", "uri", "parent_path",
                "confidence", "source", "rank", "quote_id"} <= quote.keys()
    # Internal telemetry still never surfaces.
    assert "retrieval_metrics" not in payload
    assert "diagnostic_context" not in payload
```

Also update any *pre-existing* test (e.g. `test_evidence_pack.py`) that pinned the exact legacy key set, in this same commit — that is the intended, documented shape change.

- [ ] **Step 6: Run the acceptance gate**

```bash
pytest tests/mcp_server_tests/test_evidence_pack_pin.py tests/mcp_server_tests/test_evidence_pack.py tests/evidence tests/mcp_server_tests/test_evidence_tool_adapter.py -q
```

Expected: **all green.** The enriched pin guards the payload/quote **shape** (key set, `title`/`uri` quote fields, no null leak, no telemetry leak, and top-level `partial == coverage.partial`). The *behavioral* parity items — fetch-depth clamp, graph-enrichment default, 9-facet trace fidelity, and live diagnostics — are not observable in the tool payload, so they are guarded separately: trace fidelity by the Task 6 unit test, and the clamp/default/diagnostics by the behavior-parity checklist verified against the adapter code (which unconditionally applies the clamp/default and calls `_emit_diagnostics`). Re-run the Task 5/6 tests in this step too; if a shape item regresses, fix the adapter — do not weaken the pin.

- [ ] **Step 7: `detect-changes` + commit**

```bash
npx gitnexus detect-changes
git add src/mcp_server/mcp_tools.py tests/mcp_server_tests/test_evidence_pack_pin.py \
        tests/mcp_server_tests/test_evidence_pack.py   # (no-op if Step 5 left it unchanged)
git commit -m "refactor(mcp): make kb.retrieve_evidence a thin adapter over EvidenceService"
```

---

## Task 9: Optional Citation-Preserving Enhancer + `response_mode`

**Files:**
- Create: `src/evidence/enhancer.py`
- Create: `tests/evidence/test_enhancer.py`
- Modify: `src/evidence/service.py` (`_safe_enhance`, constructor already accepts `enhancer`)
- Modify: `src/mcp_server/mcp_tools.py` (schema `KB_RETRIEVE_INPUT_SCHEMA:326` + signature + request field + service wiring)

- [ ] **Step 1: Write failing enhancer tests (happy path + laundering-refusal)**

```python
# tests/evidence/test_enhancer.py
import pytest

from src.evidence.enhancer import FakeEvidenceEnhancer, UncitedDraftEnhancer
from src.evidence.models import EvidenceCoverage, EvidencePackage, EvidenceQuote, EvidenceRequest
from src.evidence.service import EvidenceService


def _pkg():
    return EvidencePackage(
        request=EvidenceRequest(question="Deploy with NAI?", session_id="s1", response_mode="evidence_plus_draft"),
        normalized_query="deploy with NAI",
        quotes=[EvidenceQuote(quote_id="q_0001", rank=1, passage_id="p1", doc_tag="nai/deploy", text="NAI serves models.", confidence=0.86)],
        coverage=EvidenceCoverage(documents_searched=3, documents_with_evidence=1, retrieval_depth=20,
                                  reranker_applied=False, signal_pool_active=False, graph_expansion_applied=False),
    )


@pytest.mark.asyncio
async def test_fake_enhancer_adds_cited_draft():
    enhanced = await FakeEvidenceEnhancer().enhance(_pkg())
    assert enhanced.answer_draft.claims[0].quote_ids == ["q_0001"]
    assert "[q_0001]" in enhanced.answer_draft.markdown


@pytest.mark.asyncio
async def test_service_refuses_to_launder_uncited_draft():
    # An enhancer that emits an uncited claim must NOT produce a final answer.
    service = EvidenceService(
        search_candidates=None, extract_quotes=None, expand_with_graph=None,
        live_validation_available=False, enhancer=UncitedDraftEnhancer(),
    )
    package = await service._safe_enhance(_pkg())
    assert package.answer_draft is None
    assert any(g.kind == "uncited_draft_rejected" for g in package.gaps)
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/evidence/test_enhancer.py -q
```

- [ ] **Step 3: Implement the enhancer protocol + fakes**

```python
# src/evidence/enhancer.py
from __future__ import annotations

from typing import Protocol

from src.evidence.models import EvidenceAnswerDraft, EvidenceClaim, EvidencePackage


class EvidenceEnhancer(Protocol):
    async def enhance(self, package: EvidencePackage) -> EvidencePackage: ...


class FakeEvidenceEnhancer:
    async def enhance(self, package: EvidencePackage) -> EvidencePackage:
        if not package.quotes:
            return package
        q = package.quotes[0]
        package.answer_draft = EvidenceAnswerDraft(
            markdown=f"{q.text} [{q.quote_id}]",
            claims=[EvidenceClaim(claim_id="claim_0001", text=q.text, quote_ids=[q.quote_id], confidence=q.confidence)],
            enhancer_model="fake",
            enhancer_latency_ms=0.0,
        )
        return package


class UncitedDraftEnhancer:
    """Test double that tries to launder an uncited answer. The package
    validator must reject the assignment."""
    async def enhance(self, package: EvidencePackage) -> EvidencePackage:
        package.answer_draft = EvidenceAnswerDraft(
            markdown="Trust me, NAI does everything.",
            claims=[EvidenceClaim(claim_id="c1", text="everything", quote_ids=["q_9999"], confidence=0.99)],
        )
        return package
```

- [ ] **Step 4: Implement `_safe_enhance` in `service.py`**

```python
    async def _safe_enhance(self, package: EvidencePackage) -> EvidencePackage:
        from pydantic import ValidationError
        from src.evidence.models import EvidenceGap
        try:
            return await self._enhancer.enhance(package)
        except ValidationError:
            # The enhancer produced a draft whose claims are not grounded in
            # the retrieved quotes. Refuse to surface an uncited final answer.
            package.answer_draft = None
            package.gaps.append(EvidenceGap(
                kind="uncited_draft_rejected",
                message="Draft answer rejected: one or more claims were not grounded in retrieved quotes.",
                severity="warning",
            ))
            return package
```

- [ ] **Step 5: Expose `response_mode` through MCP (default stays `evidence_only`)**

In `KB_RETRIEVE_INPUT_SCHEMA` (`mcp_tools.py:326`), add:

```python
        "response_mode": {
            "type": "string",
            "enum": ["evidence_only", "evidence_plus_draft"],
            "default": "evidence_only",
            "description": "Return evidence only (default), or evidence plus a citation-preserving draft answer. Draft mode is experimental and active only when the server sets KB_EVIDENCE_DRAFT_ENABLED; otherwise the server returns evidence only.",
        },
```

Add `response_mode: str = "evidence_only"` to the `kb_retrieve_evidence` signature and set the request field accordingly:

```python
        response_mode=("evidence_plus_draft" if response_mode == "evidence_plus_draft" else "evidence_only"),
```

**Gate draft mode behind a server flag.** No real LLM enhancer exists yet, so the deterministic `FakeEvidenceEnhancer` must never reach a client through MCP by default. Add to `mcp_utils.py` (after `_coerce_bool` is defined):

```python
KB_EVIDENCE_DRAFT_ENABLED = _coerce_bool(os.getenv("KB_EVIDENCE_DRAFT_ENABLED"), default=False)
```

Import it plus `FakeEvidenceEnhancer` into `mcp_tools.py`, and make the factory inject an enhancer only when drafting is both requested **and** enabled:

```python
from src.evidence.enhancer import FakeEvidenceEnhancer
from src.mcp_server.mcp_utils import KB_EVIDENCE_DRAFT_ENABLED


def _build_evidence_service(response_mode: str) -> EvidenceService:
    drafting = response_mode == "evidence_plus_draft" and KB_EVIDENCE_DRAFT_ENABLED
    return EvidenceService(
        search_candidates=_kb_search_candidates,
        extract_quotes=_extract_evidence_from_passages,
        expand_with_graph=_expand_evidence_with_structure,
        live_validation_available=False,
        enhancer=FakeEvidenceEnhancer() if drafting else None,
    )
```

**Also update the call site** inside `kb_retrieve_evidence` (Task 8, Step 4) from `_build_evidence_service()` to `_build_evidence_service(request.response_mode)`.

Add an adapter test to this task asserting the gate both ways: with `KB_EVIDENCE_DRAFT_ENABLED` **false** (default), a client requesting `evidence_plus_draft` gets a payload with **no `answer_draft`**; with it **true**, the payload includes an `answer_draft` whose every claim cites an in-package `quote_id`.

> When drafting is disabled (the default) the service never enhances — no enhancer is injected — so a draft request transparently degrades to evidence-only. The real LLM enhancer (Qwen2.5-1.5B-Instruct, per the model stack) replaces `FakeEvidenceEnhancer` in the later live pass; the package validator guarantees no uncited final answer can escape regardless of which enhancer is wired.

- [ ] **Step 6: Run the pin + enhancer + evidence tests**

```bash
pytest tests/mcp_server_tests/test_evidence_pack_pin.py tests/evidence tests/mcp_server_tests/test_evidence_tool_adapter.py -q
```

Expected: all green. The pin still passes because `evidence_only` is the default and unchanged.

- [ ] **Step 7: `detect-changes` + commit**

```bash
npx gitnexus detect-changes
git add src/evidence/enhancer.py src/evidence/service.py src/mcp_server/mcp_tools.py \
        src/mcp_server/mcp_utils.py tests/evidence/test_enhancer.py \
        tests/mcp_server_tests/test_evidence_tool_adapter.py
git commit -m "feat(evidence): optional citation-preserving enhancer with laundering refusal"
```

---

## Task 10: Reconcile Module Boundaries (extend the existing split; no third scheme)

**Goal:** shrink `mcp_tools.py` (2321 lines) using the decomposition that already exists (`mcp_search.py`, `mcp_utils.py`, …). **Do not create `src/mcp_server/tools/`.** All moves are mechanical + re-export-shimmed and guarded by the pin.

**Files:**
- Create: `src/mcp_server/tool_schemas.py`
- Modify: `src/mcp_server/mcp_tools.py`

- [ ] **Step 1: GitNexus impact + confirm profile source of truth**

```bash
npx gitnexus impact _tool_specs --direction upstream --include-tests
rg -n "TOOL_PROFILES|build_tool_specs|_all_tool_specs" src/mcp_server
```

Expected: `TOOL_PROFILES` in `mcp_app.py` is the only profile mechanism. Confirm `build_tool_specs`/`_all_tool_specs` do **not** exist and are **not** introduced.

- [ ] **Step 2: Move the large JSON-schema constants into `tool_schemas.py`**

Move `KB_*_SCHEMA`, `GENERIC_GRAPH_*`, `SCOPE_SCHEMA`, `FILTERS_SCHEMA`, `BASE_META_SCHEMA`, `_error_schema`, `_with_error`, etc. (the schema block, roughly `mcp_tools.py:175-670`) into `src/mcp_server/tool_schemas.py`. Re-export from `mcp_tools.py`:

```python
from src.mcp_server.tool_schemas import *  # noqa: F401,F403  (compat re-export)
```

- [ ] **Step 3: Run the pin after the schema move**

```bash
pytest tests/mcp_server_tests/test_evidence_pack_pin.py tests/mcp_server_tests -q
```

Expected: green. This move changes no behavior.

- [ ] **Step 4: Do NOT relocate the adapter into `mcp_search.py`**

The schema extraction in Step 2 is the size reduction for this task. Do **not** move `kb_retrieve_evidence`/`_write_evidence_trace`/`_build_evidence_service` into `mcp_search.py`: that module owns *search* helpers, and folding the MCP transport adapter into it would mix transport concerns back into a retrieval-helper module — the exact god-module coupling this plan is unwinding. The adapter stays in `mcp_tools.py`. If a dedicated adapter home is wanted later, that is a separate, explicitly-scoped task — not this one.

- [ ] **Step 5: `detect-changes` + commit**

```bash
npx gitnexus detect-changes
git add src/mcp_server/tool_schemas.py src/mcp_server/mcp_tools.py
git commit -m "refactor(mcp): extract tool schemas; keep single profile source of truth"
```

---

## Task 11: Architecture Boundary Tests + Docs + Static Checks

**Files:**
- Create: `tests/evidence/test_architecture_boundaries.py`
- Create: `docs/architecture/evidence-package-core.md`

- [ ] **Step 1: Write boundary tests (enforce the layering)**

```python
# tests/evidence/test_architecture_boundaries.py
from pathlib import Path


def test_evidence_core_never_imports_mcp_server():
    offenders = [
        str(p) for p in Path("src/evidence").rglob("*.py")
        if "src.mcp_server" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"evidence core must not depend on mcp_server: {offenders}"


def test_mcp_tools_depends_on_evidence_core():
    text = Path("src/mcp_server/mcp_tools.py").read_text(encoding="utf-8")
    assert "src.evidence" in text


def test_trace_semantics_live_in_retrieval_trace_not_mcp_tools():
    trace = Path("src/mcp_server/retrieval_trace.py").read_text(encoding="utf-8")
    assert "def record_evidence_package" in trace
```

- [ ] **Step 2: Run boundary tests**

```bash
pytest tests/evidence/test_architecture_boundaries.py -q
```

- [ ] **Step 3: Write the architecture doc (accurate ownership)**

```markdown
# Evidence Package Core

`EvidencePackage` is the canonical retrieval product contract. It carries the
evidence quotes, coverage, gaps, an optional citation-preserving answer draft,
and an internal `retrieval_metrics` telemetry superset (never sent to clients).

## Layering
- `src/evidence/` is pure. It must never import `src/mcp_server/`.
- `src/mcp_server/` depends on `src/evidence/`, never the reverse.

## Ownership
- MCP (`mcp_tools.py`) is a transport adapter: it normalizes the request
  (backward-compat, fetch-depth clamp to KB_EVIDENCE_MAX_FETCH_K, graph-enrichment
  default), calls `EvidenceService`, applies output budgets, reconciles the partial state
  into `coverage` + a `budget_exceeded` gap, emits diagnostics with post-budget
  values, persists the trace, and serializes. It does NOT own retrieval
  orchestration, quote scoring, coverage/gap logic, trace semantics, or LLM
  enhancement.
- Trace semantics live in `retrieval_trace.py::record_evidence_package`, which
  rebuilds all nine facets from `package.retrieval_metrics`.

## Invariants
- Default response mode is `evidence_only`.
- `evidence_plus_draft` may add a draft only when every claim cites `quote_id`s
  present in the same package. The `EvidencePackage` validator enforces this at
  construction and on assignment; the service drops any draft that violates it
  and records an `uncited_draft_rejected` gap. LLM enhancement cannot launder an
  uncited final answer.
- Partial/budget state is reported once: top-level `partial`/`limit_reason` and
  `coverage.partial`/`coverage.limit_reason` always agree.

## Out of scope (deferred live pass)
Embedding, sparse (SPLADE), ColBERT, reranker, Qdrant, and Neo4j quality checks
are validated live, separately from this architecture pass.
```

- [ ] **Step 4: Static checks + full architecture-only suite**

```bash
python -m compileall -q src scripts
# Nutanix hard-rename guard (worktree is mid-migration): FAIL if any WEKA token
# survives in the code this plan touched. `rg` exits 0 on a match, so gate on it.
if rg -n "WEKA|Weka|weka|WekaDocs|wekadocs" src/evidence src/mcp_server \
      tests/evidence tests/mcp_server_tests/test_evidence_pack_pin.py \
      tests/mcp_server_tests/test_evidence_tool_adapter.py \
      tests/mcp_server_tests/conftest.py; then
  echo "ERROR: WEKA tokens found in touched code" >&2
  exit 1
fi
npx gitnexus detect-changes
pytest tests/evidence tests/mcp_server_tests -q
```

Expected: `compileall` exits 0; rename guard clean for touched files; `detect-changes` shows only the evidence + mcp_server symbols this plan intends; evidence + mcp_server tests green.

- [ ] **Step 5: Commit**

```bash
git add tests/evidence/test_architecture_boundaries.py docs/architecture/evidence-package-core.md
git commit -m "docs+test(evidence): document and enforce evidence-package boundaries"
```

---

## Execution Notes

- **The Task 1 pin — not GitNexus — is the acceptance gate for the MCP surface.** `gitnexus impact` reports LOW/0 for dynamically-dispatched tool handlers; run it (CLAUDE.md requires it) but do not treat LOW as "safe." Task 8 performs the one sanctioned pin replacement (legacy → enriched); keep the **enriched** pin green through Tasks 9–10.
- **Do not change the production tool surface.** It is already the correct 3-tool evidence-first set (`kb.retrieve_evidence`, `kb.read_excerpt`, `graph.expand`). There is no `diagnostics.latest` tool; diagnostics is a resource template.
- **Do not disable diagnostics at any commit.** The adapter emits them with post-budget values throughout.
- **The optional enhancer must never launder an uncited answer.** The contract validator + `_safe_enhance` guarantee this regardless of the enhancer implementation.
- **Live quality validation is out of scope.** Do not run embedder/sparse/ColBERT/reranker/Qdrant/Neo4j checks; the real LLM enhancer is wired in the later live pass.
- **Ignore the dirty worktree.** The WEKA→Nutanix rename churn (`config/**`, `deploy/**`, Grafana dashboards, `reports/retrieval_diagnostics/**` deletions, `docker-compose.yml`, `Makefile`, CI) is unrelated. Do not stage it. Stage only the files each task lists.

---

## Self-Review Checklist

- **Spine preserved:** `EvidencePackage` contract, pure `EvidenceService`, MCP-as-adapter, optional enhancer — all present (Tasks 2, 5, 8, 9).
- **No stale symbols:** every referenced symbol (`_kb_search_candidates`, `_extract_evidence_from_passages`, `_expand_evidence_with_structure`, `_emit_diagnostics`, `_apply_budget`, `_finalize_payload`, `TOOL_PROFILES`, `_tool_specs`, `KB_EVIDENCE_MAX_FETCH_K`, `KB_EVIDENCE_GRAPH_EXPANSION_ENABLED`) exists in the current tree at the cited location. No `build_tool_specs`/`_all_tool_specs`/`diagnostics.latest`.
- **Layering:** `src/evidence/` imports nothing from `src/mcp_server/` (enforced by Task 11).
- **Behavior parity:** the 5 checklist items are preserved in the adapter (Task 8) and guarded by the Task 1 pin.
- **Telemetry completeness:** `retrieval_metrics` carries the 9-facet superset; the trace is rebuilt from it (Task 6), not thrown away.
- **Partial-state truthfulness:** `partial`/`limit_reason` are written once, post-budget, and agree between top-level and coverage.
- **Citation integrity:** provenance is a hard contract invariant (construction + assignment), with a service-level laundering refusal.
- **God-module breakup without risky rewrites:** mechanical schema extraction + re-export shims, each guarded by the pin (Task 10); no third module scheme.
- **Live boundary:** explicitly deferred and stated in the doc and execution notes.

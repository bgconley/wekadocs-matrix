"""
Shared MCP server factory + tool implementations for STDIO and HTTP transports.
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4
from weakref import WeakKeyDictionary

import mcp.types as types
from mcp.server.lowlevel.server import Server, request_ctx

try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    StatusCode = None  # type: ignore

from src.mcp_server.query_service import QueryService, get_query_service
from src.mcp_server.scratch_store import ScratchStore
from src.query.hybrid_retrieval import ChunkResult
from src.query.traversal import TraversalService
from src.services import ContextBudgetManager, GraphService, TextService
from src.services.context_assembler import (
    ContextAssemblerService,
    SummarizationService,
)
from src.services.context_budget_manager import BudgetExceeded
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
from src.shared.observability import get_correlation_id, get_logger
from src.shared.observability.metrics import (
    cursor_returned_total,
    duplicates_suppressed_total,
    excerpt_truncations_total,
    over_budget_attempts_total,
    partial_responses_total,
    summary_calls_total,
    tool_response_bytes,
)
from src.shared.observability.retrieval_diagnostics import RetrievalDiagnosticEmitter

logger = get_logger(__name__)

_DIAGNOSTIC_EMITTER = RetrievalDiagnosticEmitter()

MAX_TOKENS_PER_TURN = 14_000
MAX_RESPONSE_BYTES = 524_288
MAX_TEXT_BYTES_PER_CALL = 32_768
DEFAULT_PAGE_SIZE = 25
KB_SEARCH_DEFAULT_TOP_K = 5
KB_SEARCH_MAX_TOP_K = 20
KB_SEARCH_MAX_PAGE_SIZE = 20
KB_SEARCH_DEFAULT_SNIPPET_CHARS = 280
KB_SEARCH_MAX_SNIPPET_CHARS = 500
SCRATCH_TTL_SECONDS = int(os.getenv("MCP_SCRATCH_TTL_SECONDS", "1800"))
SCRATCH_MAX_BYTES = int(os.getenv("MCP_SCRATCH_MAX_BYTES", str(256 * 1024 * 1024)))
_SESSION_IDS: "WeakKeyDictionary[Any, str]" = WeakKeyDictionary()
LEGACY_SEARCH_DOCUMENTATION_ENABLED = os.getenv(
    "ENABLE_LEGACY_SEARCH_DOCUMENTATION", "false"
).lower() in {"1", "true", "yes", "on"}
DIAGNOSTICS_RESOURCES_ENABLED = os.getenv(
    "MCP_DIAGNOSTICS_RESOURCES_ENABLED", "false"
).lower() in {"1", "true", "yes", "on"}


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("utf-8")).decode("utf-8")


def _decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    try:
        return int(base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8"))
    except Exception:
        return 0


def _new_budget(
    token_budget: int = MAX_TOKENS_PER_TURN, byte_budget: int = MAX_RESPONSE_BYTES
) -> ContextBudgetManager:
    return ContextBudgetManager(token_budget=token_budget, byte_budget=byte_budget)


def _apply_budget(
    payload: dict, budget: ContextBudgetManager, phase: str
) -> tuple[int, int, bool, str]:
    body = json.dumps(payload)
    bytes_estimate = len(body.encode("utf-8"))
    tokens_estimate = budget.estimate_tokens(body)
    try:
        budget.consume(tokens_estimate, bytes_estimate, phase)
        return tokens_estimate, bytes_estimate, False, "none"
    except BudgetExceeded as exc:
        return (
            exc.usage.get("tokens", tokens_estimate),
            exc.usage.get("bytes", bytes_estimate),
            True,
            exc.limit_reason,
        )


async def _report_progress(
    ctx: Any | None, *, progress: float, total: float, message: str
) -> None:
    report = getattr(ctx, "report_progress", None)
    if callable(report):
        await report(progress=progress, total=total, message=message)


def _get_request_context(ctx: Any | None) -> Optional[Any]:
    if ctx is not None and hasattr(ctx, "request_context"):
        return ctx.request_context
    try:
        return request_ctx.get()
    except LookupError:
        return None


def _detect_transport(ctx: Any | None) -> str:
    request_context = _get_request_context(ctx)
    if request_context and request_context.request is not None:
        return "http"
    return "stdio"


def _resolve_session_id(ctx: Any | None, provided: Optional[str]) -> str:
    if provided:
        return provided
    request_context = _get_request_context(ctx)
    if not request_context:
        return f"server-{uuid4()}"
    session = request_context.session
    existing = _SESSION_IDS.get(session)
    if not existing:
        existing = f"server-{uuid4()}"
        _SESSION_IDS[session] = existing
    return existing


def _finalize_payload(
    tool_name: str,
    payload: dict,
    *,
    tokens: int,
    bytes_: int,
    partial: bool,
    limit_reason: str,
    session_id: str,
    duplicates: int = 0,
) -> dict:
    payload["session_id"] = session_id
    payload["partial"] = partial
    payload["limit_reason"] = limit_reason
    payload["meta"] = {
        "usage": {
            "tokens_estimate": tokens,
            "bytes_returned": bytes_,
            "duplicates_suppressed": duplicates,
        }
    }
    tool_response_bytes.labels(tool_name).observe(bytes_)
    if payload.get("next_cursor"):
        cursor_returned_total.labels(tool_name).inc()
    if partial:
        partial_responses_total.labels(tool_name, limit_reason).inc()
    if limit_reason in {"token_cap", "byte_cap"}:
        over_budget_attempts_total.labels(tool_name, limit_reason).inc()
    if duplicates:
        duplicates_suppressed_total.labels(tool_name).inc(duplicates)
    return payload


def _graph_response(tool_name: str, session_id: str, result) -> dict:
    payload = dict(result.payload)
    return _finalize_payload(
        tool_name,
        payload,
        tokens=result.tokens_estimate,
        bytes_=result.bytes_estimate,
        partial=result.partial,
        limit_reason=result.limit_reason,
        session_id=session_id,
        duplicates=result.duplicates_suppressed,
    )


_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _error_payload(code: str, message: str, details: Optional[dict] = None) -> dict:
    payload = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return payload


def _graph_disabled_payload(extra: Optional[dict[str, Any]] = None) -> dict:
    payload = _error_payload(
        "BACKEND_UNAVAILABLE",
        "Graph traversal is disabled (neo4j_disabled=true). Use search_sections for vector search instead.",
    )
    payload["neo4j_disabled"] = True
    if extra:
        payload.update(extra)
    return payload


def _normalize_scope(scope: Optional[dict]) -> dict:
    normalized = dict(scope or {})
    default_project = getattr(_config.app, "name", "wekadocs-matrix")
    default_env = getattr(_config.app, "environment", "development")

    project_id = normalized.get("project_id")
    if project_id and project_id != default_project:
        raise ValueError(
            f"Scope project_id '{project_id}' does not match server project '{default_project}'"
        )
    environment = normalized.get("environment")
    if environment and environment != default_env:
        raise ValueError(
            f"Scope environment '{environment}' does not match server environment '{default_env}'"
        )

    normalized["project_id"] = project_id or default_project
    normalized["environment"] = environment or default_env

    doc_tags = normalized.get("doc_tags")
    if isinstance(doc_tags, str):
        doc_tags = [doc_tags]
    if doc_tags is not None:
        normalized["doc_tags"] = list(doc_tags)

    repositories = normalized.get("repositories")
    if isinstance(repositories, str):
        repositories = [repositories]
    if repositories is not None:
        normalized["repositories"] = list(repositories)

    return normalized


def _merge_scope_filters(filters: Optional[dict], scope: dict) -> dict:
    merged = dict(filters or {})
    doc_tags = scope.get("doc_tags") or []
    if doc_tags:
        merged["doc_tag"] = doc_tags
    return merged


def _tokenize_query(query: str) -> list[str]:
    if not query:
        return []
    tokens = [t for t in _QUERY_TOKEN_RE.findall(query.lower()) if len(t) >= 3]
    return tokens


def _split_spans(text: str) -> list[str]:
    if not text:
        return []
    spans: list[str] = []
    parts = text.split("```")
    for idx, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if idx % 2 == 1:
            spans.append(part)
            continue
        for paragraph in re.split(r"\n{2,}", part):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for sentence in re.split(r"(?<=[.!?])\\s+", paragraph):
                sentence = sentence.strip()
                if sentence:
                    spans.append(sentence)
    return spans


def _build_preview(text: str, query: str, max_chars: int) -> str:
    spans = _split_spans(text or "")
    if not spans:
        return (text or "")[:max_chars]
    tokens = _tokenize_query(query)
    if not tokens:
        preview = spans[0]
        return preview[:max_chars]
    scored = []
    for idx, span in enumerate(spans):
        lowered = span.lower()
        hits = sum(1 for t in tokens if t in lowered)
        score = hits / max(1, len(tokens))
        scored.append((score, len(span), idx, span))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [item[3] for item in scored[:3] if item[0] > 0]
    if not selected:
        selected = [scored[0][3]]
    preview = " ".join(selected)
    return preview[:max_chars]


def _dedupe_by_doc(chunks, max_per_doc: int) -> tuple[list, int]:
    if max_per_doc <= 0:
        return list(chunks), 0
    deduped = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.document_id or chunk.doc_tag or chunk.chunk_id
        count = counts.get(key, 0)
        if count >= max_per_doc:
            continue
        counts[key] = count + 1
        deduped.append(chunk)
    return deduped, len(chunks) - len(deduped)


def _split_spans_with_offsets(text: str) -> list[tuple[str, int, int]]:
    if not text:
        return []
    spans: list[tuple[str, int, int]] = []

    code_block_re = re.compile(r"```.*?```", re.DOTALL)
    pos = 0
    for match in code_block_re.finditer(text):
        spans.extend(_split_plain_spans_with_offsets(text[pos : match.start()], pos))
        span_text = match.group(0).strip()
        if span_text:
            spans.append((span_text, match.start(), match.end()))
        pos = match.end()
    spans.extend(_split_plain_spans_with_offsets(text[pos:], pos))
    return spans


def _split_plain_spans_with_offsets(
    text: str, offset: int
) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for para_match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        paragraph = para_match.group(0)
        base = offset + para_match.start()
        for sent_match in re.finditer(r"[^.!?]+[.!?]?", paragraph):
            sentence = sent_match.group(0).strip()
            if not sentence:
                continue
            spans.append(
                (
                    sentence,
                    base + sent_match.start(),
                    base + sent_match.end(),
                )
            )
    return spans


def _format_bullets(text: str) -> str:
    spans = _split_spans(text)
    if not spans:
        return text
    return "\n".join(f"- {span}" for span in spans)


async def _kb_search_candidates(
    *,
    query: str,
    top_k: int,
    cursor: Optional[str],
    page_size: int,
    scope: Optional[dict[str, Any]],
    filters: Optional[dict[str, Any]],
    options: Optional[dict[str, Any]],
    deps: Deps,
    effective_session: str,
) -> tuple[dict, dict]:
    normalized_scope = _normalize_scope(scope)
    merged_filters = _merge_scope_filters(filters, normalized_scope)
    options = options or {}

    mode = str(options.get("mode", "auto")).lower()
    include_scores = bool(options.get("include_scores", False))
    include_debug = bool(options.get("include_debug", False))
    max_snippet_chars = int(
        options.get("max_snippet_chars", KB_SEARCH_DEFAULT_SNIPPET_CHARS)
    )
    max_snippet_chars = max(60, min(max_snippet_chars, KB_SEARCH_MAX_SNIPPET_CHARS))
    max_per_doc = int(options.get("max_per_doc", 1))
    max_per_doc = max(1, min(max_per_doc, 5))

    top_k = max(1, min(int(top_k or KB_SEARCH_DEFAULT_TOP_K), KB_SEARCH_MAX_TOP_K))
    page = max(1, min(int(page_size or top_k), KB_SEARCH_MAX_PAGE_SIZE))

    offset = _decode_cursor(cursor)
    effective_limit = min(page, top_k)
    total_cap = max(page, top_k)
    fetch_k = min(total_cap, offset + effective_limit + 1)

    if mode == "vector_only":
        expand = False
    elif mode == "hybrid_local":
        expand = True
    else:
        expand = not _neo4j_disabled

    chunks, metrics = deps.query.search_sections_light(
        query=query,
        fetch_k=fetch_k,
        filters=merged_filters,
        expand=expand,
    )
    deduped, duplicates = _dedupe_by_doc(chunks, max_per_doc)

    sliced = deduped[offset : offset + effective_limit]
    more = len(deduped) > offset + len(sliced)
    next_cursor = _encode_cursor(offset + len(sliced)) if more else None

    results = []
    for idx, chunk in enumerate(sliced):
        score = (
            chunk.rerank_score
            if chunk.rerank_score is not None
            else (chunk.fused_score or chunk.vector_score or chunk.bm25_score or 0.0)
        )
        source = "hybrid"
        graph_distance = getattr(chunk, "graph_distance", 0) or 0
        graph_score = getattr(chunk, "graph_score", 0.0) or 0.0
        if chunk.rerank_score is not None:
            source = "reranked"
        elif graph_distance > 0 or graph_score > 0:
            source = "graph_expanded"
        elif chunk.fusion_method == "rrf":
            source = "rrf_fusion"
        elif chunk.vector_score is not None and chunk.bm25_score is None:
            source = "vector"
        elif chunk.bm25_score is not None and chunk.vector_score is None:
            source = "bm25"

        preview = _build_preview(chunk.text or "", query, max_snippet_chars)
        passage_id = uuid4().hex
        scratch_payload = {
            "section_id": chunk.chunk_id,
            "doc_tag": chunk.doc_tag,
            "title": chunk.heading,
            "text": chunk.text,
            "source_uri": getattr(chunk, "source_path", None),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        size_bytes = await deps.scratch.put(
            effective_session, passage_id, scratch_payload
        )

        result = {
            "passage_id": passage_id,
            "section_id": chunk.chunk_id,
            "doc_tag": chunk.doc_tag,
            "title": chunk.heading,
            "rank": offset + idx + 1,
            "score": float(score),
            "source": source,
            "preview": preview,
            "scratch_uri": ScratchStore.build_uri(effective_session, passage_id),
            "size_bytes": size_bytes,
            "anchor": getattr(chunk, "anchor", None),
        }

        if include_scores and not include_debug:
            result["score_breakdown"] = {
                "fused_score": (
                    float(chunk.fused_score) if chunk.fused_score is not None else None
                ),
                "vector_score": (
                    float(chunk.vector_score)
                    if chunk.vector_score is not None
                    else None
                ),
                "bm25_score": (
                    float(chunk.bm25_score) if chunk.bm25_score is not None else None
                ),
                "rerank_score": (
                    float(chunk.rerank_score)
                    if chunk.rerank_score is not None
                    else None
                ),
                "graph_score": float(graph_score),
            }

        if include_debug:
            result["debug"] = {
                "fusion_method": chunk.fusion_method,
                "fused_score": (
                    float(chunk.fused_score) if chunk.fused_score is not None else None
                ),
                "vector_score": (
                    float(chunk.vector_score)
                    if chunk.vector_score is not None
                    else None
                ),
                "bm25_score": (
                    float(chunk.bm25_score) if chunk.bm25_score is not None else None
                ),
                "rerank_score": (
                    float(chunk.rerank_score)
                    if chunk.rerank_score is not None
                    else None
                ),
                "graph_score": float(graph_score),
                "graph_distance": int(graph_distance),
                "connection_count": int(getattr(chunk, "connection_count", 0) or 0),
                "mention_count": int(getattr(chunk, "mention_count", 0) or 0),
                "entity_boost_applied": getattr(chunk, "entity_boost_applied", False),
            }

        results.append(result)

    payload = {
        "results": results,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "duplicates": duplicates,
        "metrics": metrics,
        "partial": bool(next_cursor),
        "limit_reason": "page_size" if next_cursor else "none",
        "include_debug": include_debug,
    }
    diagnostic_context = {
        "query": query,
        "scope": normalized_scope,
        "filters": merged_filters,
        "metrics": metrics,
        "chunks": list(sliced),
        "duplicates": duplicates,
        "deduped_count": len(deduped),
        "mode": mode,
        "top_k": top_k,
    }
    return payload, diagnostic_context


async def _extract_evidence_from_passages(
    *,
    question: str,
    passage_ids: list[str],
    max_quotes: int,
    max_quote_tokens: int,
    include_context_tokens: int,
    deps: Deps,
    effective_session: str,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize_query(question)
    if not passage_ids:
        return []

    spans = []
    for passage_id in passage_ids:
        entry = await deps.scratch.get(effective_session, passage_id)
        if not entry:
            continue
        text = entry.get("text") or ""
        section_id = entry.get("section_id")
        title = entry.get("title")
        for idx, (span, start, end) in enumerate(_split_spans_with_offsets(text)):
            lowered = span.lower()
            hits = sum(1 for t in query_tokens if t in lowered)
            score = hits / max(1, len(query_tokens)) if query_tokens else 0.0
            spans.append(
                {
                    "score": score,
                    "length": len(span),
                    "position": idx,
                    "span": span,
                    "start": start,
                    "end": end,
                    "passage_id": passage_id,
                    "section_id": section_id,
                    "title": title,
                    "text": text,
                }
            )

    if not spans:
        return []

    spans.sort(key=lambda item: (-item["score"], item["length"], item["position"]))
    has_positive = any(item["score"] > 0 for item in spans)
    if has_positive:
        spans = [item for item in spans if item["score"] > 0]

    max_quotes = max(1, min(int(max_quotes or 6), 12))
    max_quote_tokens = max(1, min(int(max_quote_tokens or 80), 200))
    include_context_tokens = max(0, min(int(include_context_tokens or 20), 200))

    context_chars = include_context_tokens * 4
    max_quote_chars = min(max_quote_tokens * 4, 500)

    quotes = []
    for item in spans[:max_quotes]:
        text = item["text"]
        start = max(0, item["start"] - context_chars)
        end = min(len(text), item["end"] + context_chars)
        quote = text[start:end].strip()
        quote = quote[:max_quote_chars]
        quotes.append(
            {
                "quote": quote,
                "passage_id": item["passage_id"],
                "section_id": item["section_id"],
                "title": item["title"],
                "uri": ScratchStore.build_uri(effective_session, item["passage_id"]),
                "confidence": round(float(item["score"]), 3),
            }
        )

    return quotes


def _infer_source(chunk: ChunkResult) -> str:
    source = "hybrid"
    graph_distance = getattr(chunk, "graph_distance", 0) or 0
    graph_score = getattr(chunk, "graph_score", 0.0) or 0.0
    if chunk.rerank_score is not None:
        source = "reranked"
    elif graph_distance > 0 or graph_score > 0:
        source = "graph_expanded"
    elif chunk.fusion_method == "rrf":
        source = "rrf_fusion"
    elif chunk.vector_score is not None and chunk.bm25_score is None:
        source = "vector"
    elif chunk.bm25_score is not None and chunk.vector_score is None:
        source = "bm25"
    return source


def _diagnostic_results_from_chunks(chunks: list[ChunkResult]) -> list[dict[str, Any]]:
    results = []
    for idx, chunk in enumerate(chunks):
        results.append(
            {
                "rank": idx + 1,
                "chunk_id": chunk.chunk_id,
                "doc_tag": chunk.doc_tag,
                "token_count": chunk.token_count,
                "source": _infer_source(chunk),
                "scores": {
                    "bm25": float(chunk.bm25_score or 0.0),
                    "vector": float(chunk.vector_score or 0.0),
                    "fused": float(chunk.fused_score or 0.0),
                    "rerank": float(chunk.rerank_score or 0.0),
                    "graph": float(getattr(chunk, "graph_score", 0.0) or 0.0),
                },
                "explain": {
                    "rrf_field_contributions": getattr(
                        chunk, "rrf_field_contributions", None
                    ),
                    "entity_boost_applied": getattr(
                        chunk, "entity_boost_applied", False
                    ),
                    "entity_metadata": getattr(chunk, "entity_metadata", None),
                    "graph_distance": int(getattr(chunk, "graph_distance", 0) or 0),
                    "graph_path": getattr(chunk, "graph_path", None),
                },
            }
        )
    return results


async def _emit_diagnostics(
    *,
    tool_name: str,
    ctx: Any | None,
    session_id: str,
    diagnostic_context: dict[str, Any],
    tokens_estimate: int,
    bytes_estimate: int,
    partial: bool,
    limit_reason: str,
) -> Optional[dict[str, Any]]:
    metrics = diagnostic_context.get("metrics") or {}
    scope = diagnostic_context.get("scope") or {}
    filters = diagnostic_context.get("filters") or {}
    chunks = diagnostic_context.get("chunks") or []
    duplicates = diagnostic_context.get("duplicates") or 0
    deduped_count = diagnostic_context.get("deduped_count") or len(chunks)
    doc_tag = None
    raw_doc_tag = filters.get("doc_tag")
    if isinstance(raw_doc_tag, list) and raw_doc_tag:
        doc_tag = raw_doc_tag[0]
    elif isinstance(raw_doc_tag, str):
        doc_tag = raw_doc_tag

    results = _diagnostic_results_from_chunks(chunks)
    candidates_initial = int(metrics.get("vec_count", 0) or 0) + int(
        metrics.get("bm25_count", 0) or 0
    )
    candidates_post_filter = int(metrics.get("primary_count", 0) or 0)
    candidates_post_dedupe = int(metrics.get("final_count", 0) or deduped_count)

    record = {
        "schema_version": 1,
        "diagnostic_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transport": _detect_transport(ctx),
        "tool": tool_name,
        "session_id": session_id,
        "correlation_id": get_correlation_id(),
        "otel": {"trace_id": None, "span_id": None},
        "scope": {
            "project_id": scope.get("project_id"),
            "environment": scope.get("environment"),
            "doc_tag": doc_tag,
            "snapshot_scope": filters.get("snapshot_scope"),
            "embedding_version": metrics.get("embedding_version"),
        },
        "query": {
            "raw": diagnostic_context.get("query"),
            "normalized": metrics.get(
                "query_rewrite_result", diagnostic_context.get("query")
            ),
            "rewritten": {
                "applied": bool(metrics.get("query_rewrite_applied")),
                "result": metrics.get("query_rewrite_result"),
                "reason": metrics.get("query_rewrite_reason"),
            },
        },
        "config": {
            "neo4j_disabled": _neo4j_disabled,
            "hybrid_enabled": getattr(_config.search.hybrid, "enabled", True),
            "reranker": {
                "enabled": bool(
                    getattr(
                        getattr(_config.search.hybrid, "reranker", None),
                        "enabled",
                        False,
                    )
                ),
                "provider": getattr(
                    getattr(_config.search.hybrid, "reranker", None), "provider", None
                ),
                "model": getattr(
                    getattr(_config.search.hybrid, "reranker", None), "model", None
                ),
            },
            "colbert": {
                "enabled": bool(
                    getattr(
                        getattr(_config.search.vector, "qdrant", None),
                        "enable_colbert",
                        False,
                    )
                ),
                "scoring": "maxsim",
            },
            "fusion": {
                "method": metrics.get("fusion_method")
                or getattr(_config.search.hybrid, "method", None),
                "params": {"k": getattr(_config.search.hybrid, "rrf_k", 60)},
            },
        },
        "timing_ms": {
            "bm25": metrics.get("bm25_time_ms", 0.0),
            "vector_search": metrics.get("vec_time_ms", 0.0),
            "fusion": metrics.get("fusion_time_ms", 0.0),
            "rerank": metrics.get("reranker_time_ms", 0.0),
            "graph_expansion": metrics.get("expansion_time_ms", 0.0),
            "context_assembly": metrics.get("context_assembly_ms", 0.0),
            "total": metrics.get("total_time_ms", 0.0),
        },
        "counts": {
            "candidates_initial": candidates_initial,
            "candidates_post_filter": candidates_post_filter,
            "candidates_post_dedupe": candidates_post_dedupe,
            "returned": len(results),
            "dropped": {
                "dedupe": duplicates,
                "veto": 0,
                "scope_mismatch": 0,
            },
        },
        "results": results,
        "budgets": {
            "response_bytes": bytes_estimate,
            "tokens_estimate": tokens_estimate,
            "partial": partial,
            "limit_reason": limit_reason,
        },
    }

    diag = await _DIAGNOSTIC_EMITTER.emit(record)
    if not diag:
        return None

    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("retrieval.diagnostic_id", diag["diagnostic_id"])
            span.set_attribute("retrieval.transport", record["transport"])
            span.set_attribute("retrieval.mode", diagnostic_context.get("mode"))
            span.set_attribute("retrieval.top_k", diagnostic_context.get("top_k"))
            span.set_attribute("retrieval.returned_count", len(results))
            span.set_attribute("retrieval.dedupe_dropped", duplicates)
            span.set_attribute("retrieval.partial", partial)
            span.set_attribute("retrieval.limit_reason", limit_reason)
            span.set_attribute("retrieval.bm25_ms", record["timing_ms"]["bm25"])
            span.set_attribute(
                "retrieval.vector_ms", record["timing_ms"]["vector_search"]
            )
            span.set_attribute("retrieval.fusion_ms", record["timing_ms"]["fusion"])
            span.set_attribute("retrieval.rerank_ms", record["timing_ms"]["rerank"])
            span.set_attribute(
                "retrieval.expansion_ms", record["timing_ms"]["graph_expansion"]
            )
            span.set_attribute("retrieval.total_ms", record["timing_ms"]["total"])
            for item in results[:3]:
                scores = item.get("scores", {}) or {}
                span.add_event(
                    "retrieval.candidate",
                    {
                        "rank": item.get("rank"),
                        "chunk_id": item.get("chunk_id"),
                        "source": item.get("source"),
                        "final_score": scores.get("fused"),
                        "rerank_score": scores.get("rerank"),
                        "doc_tag": item.get("doc_tag"),
                    },
                )

    return diag


def _get_deps(ctx: Any | None) -> Deps:
    request_context = _get_request_context(ctx)
    if request_context is None:
        raise RuntimeError("MCP request context is required")
    return request_context.lifespan_context


@dataclass
class Deps:
    """Dependencies shared across MCP tool calls via lifespan context."""

    query: Optional[QueryService] = None
    graph: Optional[GraphService] = None
    text: Optional[TextService] = None
    summarizer: Optional[SummarizationService] = None
    assembler: Optional[ContextAssemblerService] = None
    scratch: Optional[ScratchStore] = None


@asynccontextmanager
async def lifespan(server: Server) -> AsyncIterator[Deps]:
    """
    Lifespan context manager for dependency injection.
    Initializes QueryService once and shares across all tool calls.
    Prevents embedder cold-start penalty (2.5s) on every request.
    """
    deps = Deps()
    logger.info("STDIO server lifespan: initializing dependencies")
    _DIAGNOSTIC_EMITTER.cleanup()

    try:
        # Initialize QueryService (embedder loaded on first search)
        deps.query = get_query_service()
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()

        # Validate Neo4j schema at startup (Phase 3 hardening)
        from src.neo.schema_validator import validate_neo4j_schema

        schema_result = validate_neo4j_schema(neo4j_driver, strict=False)
        if not schema_result.valid:
            logger.error(
                "Neo4j schema validation failed - queries may return empty results",
                errors=schema_result.errors,
                node_counts=schema_result.node_counts,
            )
        elif schema_result.warnings:
            logger.warning(
                "Neo4j schema has warnings",
                warnings=schema_result.warnings,
            )

        deps.graph = GraphService(neo4j_driver)
        deps.text = TextService(neo4j_driver)
        deps.summarizer = SummarizationService(deps.graph)
        deps.assembler = ContextAssemblerService(deps.graph, deps.text)
        deps.scratch = ScratchStore(
            max_bytes=SCRATCH_MAX_BYTES,
            ttl_seconds=SCRATCH_TTL_SECONDS,
            logger=logger,
        )
        logger.info("STDIO server lifespan: QueryService ready")
        yield deps
    finally:
        # Cleanup connections if needed
        if deps.query:
            close = getattr(deps.query, "close", None)
            if callable(close):
                await close()
                logger.info("STDIO server lifespan: connections closed")


# Create FastMCP instance with lifespan and mode-appropriate instructions
GRAPH_FIRST_INSTRUCTIONS = (
    "You are connected to the Weka docs graph via MCP tools. "
    "Always start with kb.search to collect seed IDs (use search_sections only for legacy callers); "
    "then explore the neighborhood with expand_neighbors, get_paths_between, describe_nodes, "
    "list_children, list_parents, get_entities_for_sections, get_sections_for_entities, and compute_context_bundle. "
    "Only after mapping the graph should you call get_section_text for a small number of high-value sections, "
    "using conservative max_bytes_per (4–8KB) and multiple small calls. Avoid unbounded text dumps and prefer cursors."
)

VECTOR_ONLY_INSTRUCTIONS = (
    "You are connected to the Weka docs via MCP vector search tools. "
    "Graph traversal is DISABLED - do NOT use expand_neighbors, get_paths_between, list_children, list_parents, or traverse_relationships. "
    "Use kb.search (or search_sections for legacy callers) to find relevant documentation, then get_section_text to retrieve content. "
    "Complete your response in 2-3 tool calls maximum. Do not loop or retry failed graph operations."
)

# Select instructions based on neo4j_disabled config
_config = get_config()
_neo4j_disabled = getattr(getattr(_config, "hybrid", None), "neo4j_disabled", False)
_instructions = (
    VECTOR_ONLY_INSTRUCTIONS if _neo4j_disabled else GRAPH_FIRST_INSTRUCTIONS
)
logger.info(
    "MCP instructions mode",
    neo4j_disabled=_neo4j_disabled,
    mode="vector_only" if _neo4j_disabled else "graph_first",
)

KB_SEARCH_DESCRIPTION = (
    "Use when you need a short list of candidate passages. "
    "Do not use when you already have passage_ids; use kb.read_excerpt or kb.extract_evidence instead. "
    "Returns at most page_size results with previews and scratch URIs (no full text). "
    "If you need more text, call kb.read_excerpt or kb.expand_excerpt. "
    "Defaults: top_k=5, page_size=5, max_snippet_chars=280, max_per_doc=1. "
    "Max: top_k=20, page_size=20, max_snippet_chars=500."
)
KB_READ_EXCERPT_DESCRIPTION = (
    "Use when you already have a passage_id and need a bounded excerpt. "
    "Do not use for discovery; use kb.search. "
    "Returns a capped excerpt (default 300 tokens, max 800, 32KB per call). "
    "If you need more context, call kb.expand_excerpt."
)
KB_EXPAND_EXCERPT_DESCRIPTION = (
    "Use to expand around the last excerpt for a passage_id. "
    "Do not use for discovery; use kb.search first. "
    "Returns a bounded expansion before/after the last excerpt window."
)
KB_EXTRACT_EVIDENCE_DESCRIPTION = (
    "Use to extract minimal quotes that answer a question from known passage_ids. "
    "Do not use for discovery; use kb.search to get passage_ids first. "
    "Returns at most max_quotes short quotes with citations."
)
KB_RETRIEVE_EVIDENCE_DESCRIPTION = (
    "Use when you want the best evidence in one call. "
    "This runs kb.search then kb.extract_evidence, returning quotes only. "
    "Do not use when you need exploratory graph traversal."
)

RETRIEVAL_PLAYBOOK_URI = "wekadocs://retrieval_playbook"
SCRATCH_RESOURCE_TEMPLATE = "wekadocs://scratch/{session_id}/{passage_id}"
DIAGNOSTICS_RESOURCE_TEMPLATE = "wekadocs://diagnostics/{date}/{diagnostic_id}"

PROMPT_DEFINITIONS = [
    {
        "name": "graph.neighborhood_summary",
        "description": "Explore local graph neighborhood and fetch small excerpts only as needed.",
        "content": (
            "Use search_sections to seed, expand with expand_neighbors (1–2 hops), "
            "summarize with describe_nodes, and only then fetch small text via "
            "get_section_text (4–8KB)."
        ),
    },
    {
        "name": "graph.connect_concepts",
        "description": "Explain how two concepts/sections are related using graph paths.",
        "content": (
            "Find seeds for A and B via search_sections, call get_paths_between, "
            "summarize nodes with describe_nodes, then fetch minimal text for pivotal sections."
        ),
    },
    {
        "name": "graph.task_context_bundle",
        "description": "Assemble a budgeted context bundle for a downstream task.",
        "content": (
            "Identify candidate sections via search + graph tools, then call "
            "compute_context_bundle with an explicit budget; avoid large unstructured text dumps."
        ),
    },
]


def _error_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "error": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": "object"},
                },
                "required": ["code", "message"],
                "additionalProperties": True,
            }
        },
        "required": ["error"],
        "additionalProperties": True,
    }


def _with_error(normal_schema: dict) -> dict:
    return {"oneOf": [normal_schema, _error_schema()]}


BASE_META_SCHEMA = {
    "type": "object",
    "properties": {
        "usage": {
            "type": "object",
            "properties": {
                "tokens_estimate": {"type": "number"},
                "bytes_returned": {"type": "number"},
                "duplicates_suppressed": {"type": "number"},
            },
            "additionalProperties": True,
        }
    },
    "additionalProperties": True,
}

SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "project_id": {"type": "string"},
        "environment": {"type": "string"},
        "doc_tags": {
            "oneOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "string"},
            ]
        },
        "repositories": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

FILTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_tag": {"type": "array", "items": {"type": "string"}},
        "path_prefix": {"type": ["string", "null"]},
        "updated_after": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

KB_SEARCH_OPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "max_snippet_chars": {"type": "integer"},
        "max_per_doc": {"type": "integer"},
        "include_scores": {"type": "boolean"},
        "include_debug": {"type": "boolean"},
        "mode": {"type": "string"},
    },
    "additionalProperties": True,
}

KB_SEARCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": KB_SEARCH_DEFAULT_TOP_K},
        "cursor": {"type": ["string", "null"]},
        "page_size": {"type": "integer", "default": KB_SEARCH_DEFAULT_TOP_K},
        "scope": SCOPE_SCHEMA,
        "filters": FILTERS_SCHEMA,
        "options": KB_SEARCH_OPTIONS_SCHEMA,
        "session_id": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": True,
}

KB_EXCERPT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "passage_id": {"type": "string"},
        "max_tokens": {"type": "integer", "default": 300},
        "start_char": {"type": "integer", "default": 0},
        "scope": SCOPE_SCHEMA,
        "options": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "include_citation": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "session_id": {"type": "string"},
    },
    "required": ["passage_id"],
    "additionalProperties": True,
}

KB_EXPAND_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "passage_id": {"type": "string"},
        "before_tokens": {"type": "integer", "default": 150},
        "after_tokens": {"type": "integer", "default": 150},
        "scope": SCOPE_SCHEMA,
        "options": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "include_citation": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "session_id": {"type": "string"},
    },
    "required": ["passage_id"],
    "additionalProperties": True,
}

KB_EXTRACT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "passage_ids": {"type": "array", "items": {"type": "string"}},
        "max_quotes": {"type": "integer", "default": 6},
        "max_quote_tokens": {"type": "integer", "default": 80},
        "include_context_tokens": {"type": "integer", "default": 20},
        "scope": SCOPE_SCHEMA,
        "session_id": {"type": "string"},
    },
    "required": ["question", "passage_ids"],
    "additionalProperties": True,
}

KB_RETRIEVE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "top_k": {"type": "integer", "default": KB_SEARCH_DEFAULT_TOP_K},
        "max_quotes": {"type": "integer", "default": 6},
        "max_quote_tokens": {"type": "integer", "default": 80},
        "include_context_tokens": {"type": "integer", "default": 20},
        "scope": SCOPE_SCHEMA,
        "filters": FILTERS_SCHEMA,
        "options": KB_SEARCH_OPTIONS_SCHEMA,
        "session_id": {"type": "string"},
    },
    "required": ["question"],
    "additionalProperties": True,
}

KB_SEARCH_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": {"type": "object"}},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
            "diagnostic_id": {"type": "string"},
            "diagnostic_hint": {"type": "string"},
            "diagnostic_uri": {"type": "string"},
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

KB_EXCERPT_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string"},
            "excerpt": {"type": "string"},
            "truncated": {"type": "boolean"},
            "next_start_char": {"type": "integer"},
            "citation": {"type": ["object", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": [
            "passage_id",
            "excerpt",
            "truncated",
            "next_start_char",
            "partial",
            "limit_reason",
            "session_id",
            "meta",
        ],
        "additionalProperties": True,
    }
)

KB_EVIDENCE_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "quotes": {"type": "array", "items": {"type": "object"}},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
            "diagnostic_id": {"type": "string"},
            "diagnostic_hint": {"type": "string"},
            "diagnostic_uri": {"type": "string"},
        },
        "required": ["quotes", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

SEARCH_SECTIONS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 20},
        "filters": {"type": "object"},
        "cursor": {"type": ["string", "null"]},
        "page_size": {"type": "integer"},
        "session_id": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": True,
}

GENERIC_GRAPH_INPUT_SCHEMA = {"type": "object", "additionalProperties": True}

GENERIC_GRAPH_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "label": {"type": "string"},
        "title": {"type": "string"},
        "level": {"type": ["integer", "null"]},
        "tokens": {"type": ["integer", "null"]},
        "doc_tag": {"type": ["string", "null"]},
        "anchor": {"type": ["string", "null"]},
        "snippet": {"type": "string"},
    },
    "required": ["id"],
    "additionalProperties": True,
}

GRAPH_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "src": {"type": "string"},
        "dst": {"type": "string"},
        "type": {"type": "string"},
    },
    "required": ["src", "dst", "type"],
    "additionalProperties": True,
}

GRAPH_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {"type": "array", "items": {"type": "string"}},
        "types": {"type": "array", "items": {"type": "string"}},
        "length": {"type": "integer"},
    },
    "required": ["nodes", "types", "length"],
    "additionalProperties": True,
}

GRAPH_DESCRIBE_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": GRAPH_NODE_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_EXPAND_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "items": GRAPH_NODE_SCHEMA},
            "edges": {"type": "array", "items": GRAPH_EDGE_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "dedupe_applied": {"type": "boolean"},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["nodes", "edges", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_PATHS_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": GRAPH_PATH_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["paths", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_CHILDREN_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "children": {"type": "array", "items": GRAPH_NODE_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["children", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_PARENTS_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "parent_id": {"type": "string"},
                        "parent_title": {"type": "string"},
                    },
                    "required": ["section_id", "parent_id", "parent_title"],
                    "additionalProperties": True,
                },
            },
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_ENTITIES_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                                "required": ["id", "label", "name"],
                                "additionalProperties": True,
                            },
                        },
                    },
                    "required": ["section_id", "entities"],
                    "additionalProperties": True,
                },
            },
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_SECTIONS_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "section_id": {"type": "string"},
                                    "title": {"type": "string"},
                                },
                                "required": ["section_id", "title"],
                                "additionalProperties": True,
                            },
                        },
                    },
                    "required": ["entity_id", "sections"],
                    "additionalProperties": True,
                },
            },
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)


async def search_documentation(
    query: str,
    top_k: int = 20,
    verbosity: str = "graph",
    ctx: Any | None = None,
) -> dict:
    """
    Search documentation using hybrid retrieval (vector + graph).

    Args:
        query: Natural language search query
        top_k: Maximum number of results to return (default: 20)
        verbosity: Response detail level - "full" (complete text) or "graph" (text + relationships). For backward compatibility, "snippet" maps to "graph".
        ctx: MCP context for progress reporting and dependency access

    Returns:
        Dictionary with answer_markdown and answer_json (evidence, confidence, diagnostics)
    """
    logger.info(
        f"STDIO tool called: search_documentation(query='{query}', top_k={top_k}, verbosity='{verbosity}')"
    )

    await _report_progress(ctx, progress=0.1, total=1.0, message="Encoding query")

    try:
        deps = _get_deps(ctx)
        query_service = deps.query
        if not query_service:
            raise RuntimeError("QueryService not initialized")

        await _report_progress(
            ctx, progress=0.3, total=1.0, message="Searching vectors"
        )

        # Map legacy/short verbosity tokens
        verb = (verbosity or "graph").strip().lower()
        if verb in {"snippet", "snip", "short"}:
            verb = "graph"

        # Execute search via existing Phase 2 pipeline
        response = query_service.search(
            query=query,
            top_k=top_k,
            expand_graph=True,
            find_paths=False,
            verbosity=verb,
        )

        await _report_progress(
            ctx, progress=0.9, total=1.0, message="Building response"
        )

        # Convert to JSON-serializable dict
        result = response.to_dict()

        await _report_progress(ctx, progress=1.0, total=1.0, message="Complete")

        logger.info(
            f"STDIO tool completed: {len(result.get('answer_json', {}).get('evidence', []))} evidence items, "
            f"confidence={result.get('answer_json', {}).get('confidence', 0):.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"STDIO tool failed: {e}", exc_info=True)
        # Return error in MCP-compatible format
        return {
            "answer_markdown": f"Error: {str(e)}",
            "answer_json": {
                "answer": f"Search failed: {str(e)}",
                "evidence": [],
                "confidence": 0.0,
                "diagnostics": {"error": str(e)},
            },
        }


async def kb_search(
    query: str,
    top_k: int = KB_SEARCH_DEFAULT_TOP_K,
    cursor: Optional[str] = None,
    page_size: int = KB_SEARCH_DEFAULT_TOP_K,
    scope: Optional[dict[str, Any]] = None,
    filters: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Search for candidate passages without returning full text."""

    logger.info(
        "STDIO tool called: kb.search(query='%s', top_k=%s, cursor=%s)",
        query,
        top_k,
        cursor,
    )
    deps = _get_deps(ctx)
    if not deps.query:
        raise RuntimeError("QueryService not initialized")
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")
    effective_session = _resolve_session_id(ctx, session_id)
    payload, diagnostic_context = await _kb_search_candidates(
        query=query,
        top_k=top_k,
        cursor=cursor,
        page_size=page_size,
        scope=scope,
        filters=filters,
        options=options,
        deps=deps,
        effective_session=effective_session,
    )
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "seeds"
    )
    limit_reason = payload.get("limit_reason", "none")
    if budget_partial:
        limit_reason = budget_reason
    partial = bool(payload.get("partial")) or budget_partial
    finalized = _finalize_payload(
        "kb.search",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=partial,
        limit_reason=limit_reason,
        session_id=effective_session,
        duplicates=payload.get("duplicates", 0),
    )
    diagnostic = await _emit_diagnostics(
        tool_name="kb.search",
        ctx=ctx,
        session_id=effective_session,
        diagnostic_context=diagnostic_context,
        tokens_estimate=tokens_estimate,
        bytes_estimate=bytes_estimate,
        partial=partial,
        limit_reason=limit_reason,
    )
    if diagnostic:
        diagnostic_id = diagnostic.get("diagnostic_id")
        if diagnostic_id:
            finalized["diagnostic_id"] = diagnostic_id
            finalized["diagnostic_hint"] = f"See retrieval diagnostics {diagnostic_id}"
            if DIAGNOSTICS_RESOURCES_ENABLED and diagnostic.get("date"):
                finalized["diagnostic_uri"] = _diagnostics_uri(
                    diagnostic["date"], diagnostic_id
                )
    if payload.get("include_debug"):
        finalized["meta"]["usage"]["source"] = payload.get("metrics")
    return finalized


async def kb_read_excerpt(
    passage_id: str,
    max_tokens: int = 300,
    start_char: int = 0,
    scope: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Read a bounded excerpt from scratch storage."""

    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")

    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)
    entry = await deps.scratch.get(effective_session, passage_id)
    if not entry:
        return _error_payload(
            "INVALID_ARGUMENT",
            f"Unknown passage_id '{passage_id}' for this session.",
        )

    max_tokens = max(1, min(int(max_tokens or 300), 800))
    start_char = max(0, int(start_char or 0))
    char_budget = min(max_tokens * 4, MAX_TEXT_BYTES_PER_CALL)

    text = entry.get("text") or ""
    excerpt = text[start_char : start_char + char_budget]
    truncated = start_char + len(excerpt) < len(text)
    next_start_char = start_char + len(excerpt)
    if truncated:
        excerpt_truncations_total.labels("kb.read_excerpt").inc()

    options = options or {}
    if str(options.get("format", "text")).lower() == "bullets":
        excerpt = _format_bullets(excerpt)

    include_citation = options.get("include_citation", True)
    citation = None
    if include_citation:
        citation = {
            "section_id": entry.get("section_id"),
            "doc_tag": entry.get("doc_tag"),
            "title": entry.get("title"),
            "uri": ScratchStore.build_uri(effective_session, passage_id),
        }

    await deps.scratch.update(
        effective_session,
        passage_id,
        {"last_start_char": start_char, "last_end_char": next_start_char},
    )

    payload = {
        "passage_id": passage_id,
        "excerpt": excerpt,
        "truncated": truncated,
        "next_start_char": next_start_char,
        "citation": citation,
    }
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "excerpt"
    )
    limit_reason = budget_reason if budget_partial else "none"
    finalized = _finalize_payload(
        "kb.read_excerpt",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=budget_partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    return finalized


async def kb_expand_excerpt(
    passage_id: str,
    before_tokens: int = 150,
    after_tokens: int = 150,
    scope: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Expand around the last excerpt window for a passage."""

    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")

    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)
    entry = await deps.scratch.get(effective_session, passage_id)
    if not entry:
        return _error_payload(
            "INVALID_ARGUMENT",
            f"Unknown passage_id '{passage_id}' for this session.",
        )

    before_tokens = max(0, min(int(before_tokens or 150), 800))
    after_tokens = max(0, min(int(after_tokens or 150), 800))
    before_chars = before_tokens * 4
    after_chars = after_tokens * 4

    text = entry.get("text") or ""
    last_start = int(entry.get("last_start_char") or 0)
    last_end = int(entry.get("last_end_char") or 0)
    if last_end <= last_start:
        last_start = 0
        last_end = 0

    start_char = max(0, last_start - before_chars)
    end_char = min(len(text), last_end + after_chars)
    if end_char <= start_char:
        end_char = min(len(text), start_char + before_chars + after_chars)
    if end_char - start_char > MAX_TEXT_BYTES_PER_CALL:
        end_char = min(len(text), start_char + MAX_TEXT_BYTES_PER_CALL)

    excerpt = text[start_char:end_char]
    truncated = start_char > 0 or end_char < len(text)
    next_start_char = end_char
    if truncated:
        excerpt_truncations_total.labels("kb.expand_excerpt").inc()

    options = options or {}
    if str(options.get("format", "text")).lower() == "bullets":
        excerpt = _format_bullets(excerpt)

    include_citation = options.get("include_citation", True)
    citation = None
    if include_citation:
        citation = {
            "section_id": entry.get("section_id"),
            "doc_tag": entry.get("doc_tag"),
            "title": entry.get("title"),
            "uri": ScratchStore.build_uri(effective_session, passage_id),
        }

    await deps.scratch.update(
        effective_session,
        passage_id,
        {"last_start_char": start_char, "last_end_char": end_char},
    )

    payload = {
        "passage_id": passage_id,
        "excerpt": excerpt,
        "truncated": truncated,
        "next_start_char": next_start_char,
        "citation": citation,
    }
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "excerpt"
    )
    limit_reason = budget_reason if budget_partial else "none"
    finalized = _finalize_payload(
        "kb.expand_excerpt",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=budget_partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    return finalized


async def kb_extract_evidence(
    question: str,
    passage_ids: list[str],
    max_quotes: int = 6,
    max_quote_tokens: int = 80,
    include_context_tokens: int = 20,
    scope: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Extract minimal evidence spans from known passages."""

    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")

    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)
    quotes = await _extract_evidence_from_passages(
        question=question,
        passage_ids=passage_ids,
        max_quotes=max_quotes,
        max_quote_tokens=max_quote_tokens,
        include_context_tokens=include_context_tokens,
        deps=deps,
        effective_session=effective_session,
    )

    payload = {"quotes": quotes}
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "evidence"
    )
    limit_reason = budget_reason if budget_partial else "none"
    finalized = _finalize_payload(
        "kb.extract_evidence",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=budget_partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    return finalized


async def kb_retrieve_evidence(
    question: str,
    top_k: int = KB_SEARCH_DEFAULT_TOP_K,
    max_quotes: int = 6,
    max_quote_tokens: int = 80,
    include_context_tokens: int = 20,
    scope: Optional[dict[str, Any]] = None,
    filters: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Search for candidates then extract minimal evidence quotes."""

    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")

    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)
    search_payload, diagnostic_context = await _kb_search_candidates(
        query=question,
        top_k=top_k,
        cursor=None,
        page_size=top_k,
        scope=scope,
        filters=filters,
        options=options,
        deps=deps,
        effective_session=effective_session,
    )
    passage_ids = [item["passage_id"] for item in search_payload["results"]]
    quotes = await _extract_evidence_from_passages(
        question=question,
        passage_ids=passage_ids,
        max_quotes=max_quotes,
        max_quote_tokens=max_quote_tokens,
        include_context_tokens=include_context_tokens,
        deps=deps,
        effective_session=effective_session,
    )

    payload = {"quotes": quotes}
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "evidence"
    )
    limit_reason = budget_reason if budget_partial else "none"
    finalized = _finalize_payload(
        "kb.retrieve_evidence",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=budget_partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    diagnostic = await _emit_diagnostics(
        tool_name="kb.retrieve_evidence",
        ctx=ctx,
        session_id=effective_session,
        diagnostic_context=diagnostic_context,
        tokens_estimate=tokens_estimate,
        bytes_estimate=bytes_estimate,
        partial=budget_partial,
        limit_reason=limit_reason,
    )
    if diagnostic:
        diagnostic_id = diagnostic.get("diagnostic_id")
        if diagnostic_id:
            finalized["diagnostic_id"] = diagnostic_id
            finalized["diagnostic_hint"] = f"See retrieval diagnostics {diagnostic_id}"
            if DIAGNOSTICS_RESOURCES_ENABLED and diagnostic.get("date"):
                finalized["diagnostic_uri"] = _diagnostics_uri(
                    diagnostic["date"], diagnostic_id
                )
    return finalized


async def search_sections(
    query: str,
    top_k: int = 20,
    filters: Optional[dict[str, Any]] = None,
    cursor: Optional[str] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Compact section search returning ids + metadata only."""

    logger.info(
        "STDIO tool called: search_sections(query='%s', top_k=%s, cursor=%s)",
        query,
        top_k,
        cursor,
    )
    deps = _get_deps(ctx)
    if not deps.query:
        raise RuntimeError("QueryService not initialized")

    query_service = deps.query
    effective_session = _resolve_session_id(ctx, session_id)
    offset = _decode_cursor(cursor)
    page = max(1, min(page_size or DEFAULT_PAGE_SIZE, 100))
    # top_k is the hard limit on results; page_size controls pagination within that limit
    effective_limit = min(page, top_k) if top_k and top_k > 0 else page
    total_cap = max(page, top_k or page)
    fetch_k = min(total_cap, offset + effective_limit + 1)
    if fetch_k <= 0:
        rows = []
        metrics = {}
    else:
        rows, metrics = query_service.search_sections_light(
            query=query, fetch_k=fetch_k, filters=filters
        )
    # Enforce top_k as hard ceiling on returned results
    sliced = rows[offset : offset + effective_limit]
    more = len(rows) > offset + len(sliced)
    next_cursor = _encode_cursor(offset + len(sliced)) if more else None
    results = []
    for idx, chunk in enumerate(sliced):
        # Prefer rerank_score (from ColBERT/BGE cross-encoder) over pre-rerank fusion score
        score = (
            chunk.rerank_score
            if chunk.rerank_score is not None
            else (chunk.fused_score or chunk.vector_score or chunk.bm25_score or 0.0)
        )
        # Source detection: use explicit 'is not None' checks to handle 0.0 scores correctly
        # Priority order: reranked > graph_expanded > rrf_fusion > vector > bm25 > hybrid
        source = "hybrid"
        graph_distance = getattr(chunk, "graph_distance", 0) or 0
        graph_score = getattr(chunk, "graph_score", 0.0) or 0.0
        if chunk.rerank_score is not None:
            source = "reranked"
        elif graph_distance > 0 or graph_score > 0:
            # Graph-expanded results bypass RRF fusion, so no per-signal scores
            source = "graph_expanded"
        elif chunk.fusion_method == "rrf":
            source = "rrf_fusion"
        elif chunk.vector_score is not None and chunk.bm25_score is None:
            source = "vector"
        elif chunk.bm25_score is not None and chunk.vector_score is None:
            source = "bm25"
        results.append(
            {
                "section_id": chunk.chunk_id,
                "title": chunk.heading,
                "tokens": chunk.token_count,
                "doc_tag": chunk.doc_tag,
                "score": float(score),
                "rank": offset + idx + 1,
                "source": source,
                # RRF fusion metadata - enables Agent to understand retrieval method
                "fusion_method": chunk.fusion_method,
                "fused_score": (
                    float(chunk.fused_score) if chunk.fused_score is not None else None
                ),
                # Per-signal scores - enables Agent to understand WHY chunk matched
                "title_vec_score": (
                    float(chunk.title_vec_score)
                    if chunk.title_vec_score is not None
                    else None
                ),
                "entity_vec_score": (
                    float(chunk.entity_vec_score)
                    if chunk.entity_vec_score is not None
                    else None
                ),
                "doc_title_sparse_score": (
                    float(chunk.doc_title_sparse_score)
                    if chunk.doc_title_sparse_score is not None
                    else None
                ),
                "lexical_vec_score": (
                    float(chunk.lexical_vec_score)
                    if chunk.lexical_vec_score is not None
                    else None
                ),
                "rerank_score": (
                    float(chunk.rerank_score)
                    if chunk.rerank_score is not None
                    else None
                ),
                # Graph enrichment scores
                "graph_score": float(getattr(chunk, "graph_score", 0.0) or 0.0),
                "graph_distance": int(getattr(chunk, "graph_distance", 0) or 0),
                "connection_count": int(getattr(chunk, "connection_count", 0) or 0),
                "mention_count": int(getattr(chunk, "mention_count", 0) or 0),
                # GLiNER entity boosting metadata (Phase 4) - enables Agent to see entity-aware ranking
                "entity_boost_applied": getattr(chunk, "entity_boost_applied", False),
                "entity_metadata": getattr(chunk, "entity_metadata", None),
                # RRF per-field contributions (when rrf_debug_logging=true)
                # Shows how each vector field contributed to the fused_score
                "rrf_field_contributions": getattr(
                    chunk, "rrf_field_contributions", None
                ),
            }
        )

    payload = {
        "results": results,
        "cursor": cursor,
        "next_cursor": next_cursor,
    }
    budget = _new_budget()
    tokens_estimate, bytes_estimate, budget_partial, budget_reason = _apply_budget(
        payload, budget, "seeds"
    )
    limit_reason = "page_size" if next_cursor else "none"
    if budget_partial:
        limit_reason = budget_reason
    partial = bool(next_cursor) or budget_partial
    finalized = _finalize_payload(
        "search_sections",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    finalized["meta"]["usage"]["source"] = metrics
    return finalized


async def traverse_relationships(
    start_ids: list[str],
    rel_types: list[str] | None = None,
    max_depth: int = 2,
    include_text: bool = False,
    ctx: Any | None = None,
) -> dict:
    """
    Traverse graph relationships from given nodes for multi-turn exploration.

    Args:
        start_ids: Starting section/entity IDs to traverse from
        rel_types: Relationship types to follow (default: ["MENTIONS", "CONTAINS_STEP"])
        max_depth: Maximum traversal depth, 1-3 hops (default: 2)
        include_text: Include node text field in results (default: False; when True, truncated to ~1K chars)
        ctx: MCP context for progress reporting

    Returns:
        Dictionary with nodes, relationships, and paths discovered during traversal
    """
    logger.info(
        f"STDIO tool called: traverse_relationships(start_ids={start_ids}, "
        f"rel_types={rel_types}, max_depth={max_depth})"
    )

    await _report_progress(
        ctx, progress=0.1, total=1.0, message="Initializing traversal"
    )

    try:
        # Validate inputs
        if not start_ids:
            return {
                "error": "start_ids is required and cannot be empty",
                "nodes": [],
                "relationships": [],
                "paths": [],
            }

        # Check if graph is disabled
        if _neo4j_disabled:
            logger.info(
                "traverse_relationships called but neo4j_disabled=true, returning early"
            )
            return _graph_disabled_payload(
                {"nodes": [], "relationships": [], "paths": []}
            )

        # Get Neo4j driver from connection manager
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()

        await _report_progress(ctx, progress=0.3, total=1.0, message="Traversing graph")

        # Create traversal service and execute
        traversal_svc = TraversalService(neo4j_driver)
        result = traversal_svc.traverse(
            start_ids=start_ids,
            rel_types=rel_types,
            max_depth=max_depth,
            include_text=include_text,
        )

        await _report_progress(
            ctx, progress=0.9, total=1.0, message="Formatting results"
        )

        # Convert to dict
        result_dict = result.to_dict()

        await _report_progress(ctx, progress=1.0, total=1.0, message="Complete")

        logger.info(
            f"STDIO tool completed: {len(result.nodes)} nodes, "
            f"{len(result.relationships)} relationships found"
        )

        return result_dict

    except ValueError as e:
        # Validation errors
        logger.warning(f"STDIO tool validation error: {e}")
        return {
            "error": f"Validation error: {str(e)}",
            "nodes": [],
            "relationships": [],
            "paths": [],
        }
    except Exception as e:
        logger.error(f"STDIO tool failed: {e}", exc_info=True)
        return {
            "error": f"Traversal failed: {str(e)}",
            "nodes": [],
            "relationships": [],
            "paths": [],
        }


async def describe_nodes(
    node_ids: list[str],
    fields: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    Return projection-only node metadata without heavy text.

    When to use:
    - After search_sections or expand_neighbors to triage nodes cheaply
    - Before calling get_section_text, to choose a few high-value sections

    Recipe:
    - search_sections → expand_neighbors (1–2 hops) → describe_nodes → get_section_text (few)
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info("describe_nodes called but neo4j_disabled=true, returning early")
        return _graph_disabled_payload({"results": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    result = deps.graph.describe_nodes(
        node_ids=node_ids, fields=fields, budget=budget, phase="neighbors"
    )
    effective_session = _resolve_session_id(ctx, session_id)
    return _graph_response("describe_nodes", effective_session, result)


async def expand_neighbors(
    node_ids: list[str],
    rel_types: Optional[list[str]] = None,
    direction: str = "both",
    max_hops: int = 1,
    include_snippet: bool = False,
    cursor: Optional[str] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    Expand to directly connected nodes with cursor pagination and de-duplication.

    When to use:
    - After search_sections to discover related sections/entities before fetching text
    - To build a local neighborhood for summarization or path finding

    Tips:
    - Keep max_hops to 1–2; filter rel_types when possible; use cursor for paging
    - Pair with describe_nodes to rank which nodes to read

    Recipe:
    - search_sections (seeds) → expand_neighbors → describe_nodes → get_section_text (few)
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info("expand_neighbors called but neo4j_disabled=true, returning early")
        return _graph_disabled_payload({"nodes": [], "edges": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.expand_neighbors(
        node_ids=node_ids,
        rel_types=rel_types,
        direction=direction,
        max_hops=max_hops,
        include_snippet=include_snippet,
        page_size=page_size,
        cursor=cursor,
        session_id=effective_session,
        budget=budget,
    )
    return _graph_response("expand_neighbors", effective_session, result)


async def get_paths_between(
    a_ids: list[str],
    b_ids: list[str],
    rel_types: Optional[list[str]] = None,
    max_hops: int = 3,
    max_paths: int = 10,
    cursor: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    Find connecting paths between two node sets to explain relationships.

    When to use:
    - The user asks how/why concepts are related, or you need a chain

    Tips:
    - Keep max_hops ≤ 3 and max_paths small (≤ 10) for concise results

    Recipe:
    - search_sections(A,B) → get_paths_between → describe_nodes → get_section_text (pivots)
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info("get_paths_between called but neo4j_disabled=true, returning early")
        return _graph_disabled_payload({"paths": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.get_paths_between(
        a_ids=a_ids,
        b_ids=b_ids,
        rel_types=rel_types,
        max_hops=max_hops,
        max_paths=max_paths,
        cursor=cursor,
        budget=budget,
    )
    return _graph_response("get_paths_between", effective_session, result)


async def list_children(
    parent_id: str,
    cursor: Optional[str] = None,
    page_size: int = 50,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    List child sections for a parent, with cursor pagination.

    When to use:
    - Navigate hierarchy without fetching full text; find nearby relevant sections

    Recipe:
    - search_sections → list_parents / list_children → describe_nodes → selective get_section_text
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info("list_children called but neo4j_disabled=true, returning early")
        return _graph_disabled_payload({"children": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.list_children(
        parent_id=parent_id,
        page_size=page_size,
        cursor=cursor,
        budget=budget,
    )
    return _graph_response("list_children", effective_session, result)


async def list_parents(
    section_ids: list[str],
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    List parent sections for given sections.

    When to use:
    - Move up the hierarchy to situate a section before fetching text

    Recipe:
    - search_sections → list_parents → describe_nodes → get_section_text (few)
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info("list_parents called but neo4j_disabled=true, returning early")
        return _graph_disabled_payload({"results": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.list_parents(section_ids=section_ids, budget=budget)
    return _graph_response("list_parents", effective_session, result)


async def get_entities_for_sections(
    section_ids: list[str],
    labels: Optional[list[str]] = None,
    max_per_section: int = 20,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    Pivot from sections to entities (and back) to broaden/narrow context.

    When to use:
    - Extract key entities from candidate sections before reading text

    Recipe:
    - search_sections → get_entities_for_sections → get_sections_for_entities → describe_nodes → get_section_text
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info(
            "get_entities_for_sections called but neo4j_disabled=true, returning early"
        )
        return _graph_disabled_payload({"results": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.get_entities_for_sections(
        section_ids=section_ids,
        labels=labels,
        max_per_section=max_per_section,
        budget=budget,
    )
    return _graph_response("get_entities_for_sections", effective_session, result)


async def get_sections_for_entities(
    entity_ids: list[str],
    max_per: int = 20,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    Pivot from entities to sections to target a small set for text fetches.

    When to use:
    - After extracting entities, find the most relevant sections linked to them

    Recipe:
    - search_sections → get_entities_for_sections → get_sections_for_entities → describe_nodes → get_section_text
    """
    # Check if graph is disabled
    if _neo4j_disabled:
        logger.info(
            "get_sections_for_entities called but neo4j_disabled=true, returning early"
        )
        return _graph_disabled_payload({"results": []})

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.get_sections_for_entities(
        entity_ids=entity_ids,
        max_per=max_per,
        budget=budget,
    )
    return _graph_response("get_sections_for_entities", effective_session, result)


async def get_section_text(
    section_ids: list[str],
    max_bytes_per: int = 8192,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """
    Fetch small excerpts of section text with strict byte limits (default 8KB per section).

    Guidance:
    - Use only after graph exploration has narrowed candidates
    - Prefer multiple small calls over one large fetch; you can override max_bytes_per when essential
    """
    deps = _get_deps(ctx)
    if not deps.text:
        raise RuntimeError("TextService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    text_result = deps.text.get_section_text(
        section_ids=section_ids,
        max_bytes_per=min(max_bytes_per, MAX_TEXT_BYTES_PER_CALL),
        budget=budget,
    )
    truncated_count = sum(1 for item in text_result.results if item.get("truncated"))
    if truncated_count:
        excerpt_truncations_total.labels("get_section_text").inc(truncated_count)
    payload = {
        "results": text_result.results,
        "hints": {"next_tools": ["expand_neighbors", "describe_nodes"]},
    }
    return _finalize_payload(
        "get_section_text",
        payload,
        tokens=text_result.tokens_estimate,
        bytes_=text_result.bytes_estimate,
        partial=text_result.partial,
        limit_reason=text_result.limit_reason,
        session_id=effective_session,
    )


async def summarize_neighborhood(
    node_ids: list[str],
    token_budget: int = 400,
    ctx: Any | None = None,
) -> dict:
    deps = _get_deps(ctx)
    if not deps.summarizer:
        raise RuntimeError("Summarizer not initialized")
    effective_session = _resolve_session_id(ctx, None)
    try:
        summary = deps.summarizer.summarize_neighborhood(
            node_ids=node_ids, token_budget=token_budget
        )
        summary_calls_total.labels("summarize_neighborhood", "success").inc()
    except Exception:
        summary_calls_total.labels("summarize_neighborhood", "error").inc()
        raise
    payload = {
        "bullets": summary["bullets"],
        "citations": summary["citations"],
    }
    body = json.dumps(payload)
    bytes_estimate = len(body.encode("utf-8"))
    tokens_estimate = max(1, len(body) // 4)
    return _finalize_payload(
        "summarize_neighborhood",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=False,
        limit_reason="none",
        session_id=effective_session,
    )


async def compute_context_bundle(
    seeds: list[str],
    strategy: str = "hybrid",
    token_budget: int = 2_000,
    ctx: Any | None = None,
) -> dict:
    deps = _get_deps(ctx)
    if not deps.assembler:
        raise RuntimeError("Context assembler not initialized")
    effective_session = _resolve_session_id(ctx, None)
    try:
        bundle = deps.assembler.compute_context_bundle(
            seeds=seeds, strategy=strategy, token_budget=token_budget
        )
        summary_calls_total.labels("compute_context_bundle", "success").inc()
    except Exception:
        summary_calls_total.labels("compute_context_bundle", "error").inc()
        raise
    payload = {
        "bundle": bundle["bundle"],
        "usage": bundle["usage"],
    }
    body = json.dumps(payload)
    bytes_estimate = len(body.encode("utf-8"))
    tokens_estimate = max(1, len(body) // 4)
    return _finalize_payload(
        "compute_context_bundle",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=False,
        limit_reason="none",
        session_id=effective_session,
    )


def _tool_description(func, fallback: str) -> str:
    return inspect.getdoc(func) or fallback


def _parse_scratch_uri(uri: str) -> Optional[tuple[str, str]]:
    prefix = "wekadocs://scratch/"
    if not uri.startswith(prefix):
        return None
    remainder = uri[len(prefix) :]
    parts = remainder.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _parse_diagnostics_uri(uri: str) -> Optional[tuple[str, str]]:
    prefix = "wekadocs://diagnostics/"
    if not uri.startswith(prefix):
        return None
    remainder = uri[len(prefix) :]
    parts = remainder.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _diagnostics_uri(date: str, diagnostic_id: str) -> str:
    return f"wekadocs://diagnostics/{date}/{diagnostic_id}"


def _retrieval_playbook_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(root, "docs", "mcp", "retrieval_playbook.md")


def _read_retrieval_playbook() -> str:
    path = _retrieval_playbook_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        return f"Playbook unavailable: {exc}"


def _tool_specs() -> list[dict[str, Any]]:
    readonly = types.ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=False,
        idempotentHint=True,
        destructiveHint=False,
    )
    specs = [
        {
            "name": "kb.search",
            "handler": kb_search,
            "description": KB_SEARCH_DESCRIPTION,
            "input_schema": KB_SEARCH_INPUT_SCHEMA,
            "output_schema": KB_SEARCH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "kb.read_excerpt",
            "handler": kb_read_excerpt,
            "description": KB_READ_EXCERPT_DESCRIPTION,
            "input_schema": KB_EXCERPT_INPUT_SCHEMA,
            "output_schema": KB_EXCERPT_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "kb.expand_excerpt",
            "handler": kb_expand_excerpt,
            "description": KB_EXPAND_EXCERPT_DESCRIPTION,
            "input_schema": KB_EXPAND_INPUT_SCHEMA,
            "output_schema": KB_EXCERPT_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "kb.extract_evidence",
            "handler": kb_extract_evidence,
            "description": KB_EXTRACT_EVIDENCE_DESCRIPTION,
            "input_schema": KB_EXTRACT_INPUT_SCHEMA,
            "output_schema": KB_EVIDENCE_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "kb.retrieve_evidence",
            "handler": kb_retrieve_evidence,
            "description": KB_RETRIEVE_EVIDENCE_DESCRIPTION,
            "input_schema": KB_RETRIEVE_INPUT_SCHEMA,
            "output_schema": KB_EVIDENCE_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.describe",
            "handler": describe_nodes,
            "description": _tool_description(describe_nodes, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_DESCRIBE_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.expand",
            "handler": expand_neighbors,
            "description": _tool_description(expand_neighbors, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_EXPAND_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.paths",
            "handler": get_paths_between,
            "description": _tool_description(get_paths_between, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_PATHS_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.parents",
            "handler": list_parents,
            "description": _tool_description(list_parents, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_PARENTS_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.children",
            "handler": list_children,
            "description": _tool_description(list_children, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_CHILDREN_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.entities_for_sections",
            "handler": get_entities_for_sections,
            "description": _tool_description(get_entities_for_sections, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_ENTITIES_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.sections_for_entities",
            "handler": get_sections_for_entities,
            "description": _tool_description(get_sections_for_entities, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_SECTIONS_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "search_sections",
            "handler": search_sections,
            "description": _tool_description(search_sections, ""),
            "input_schema": SEARCH_SECTIONS_INPUT_SCHEMA,
            "output_schema": KB_SEARCH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "get_section_text",
            "handler": get_section_text,
            "description": _tool_description(get_section_text, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GENERIC_GRAPH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "describe_nodes",
            "handler": describe_nodes,
            "description": _tool_description(describe_nodes, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_DESCRIBE_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "expand_neighbors",
            "handler": expand_neighbors,
            "description": _tool_description(expand_neighbors, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_EXPAND_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "get_paths_between",
            "handler": get_paths_between,
            "description": _tool_description(get_paths_between, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_PATHS_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "list_parents",
            "handler": list_parents,
            "description": _tool_description(list_parents, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_PARENTS_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "list_children",
            "handler": list_children,
            "description": _tool_description(list_children, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_CHILDREN_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "get_entities_for_sections",
            "handler": get_entities_for_sections,
            "description": _tool_description(get_entities_for_sections, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_ENTITIES_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "get_sections_for_entities",
            "handler": get_sections_for_entities,
            "description": _tool_description(get_sections_for_entities, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GRAPH_SECTIONS_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "traverse_relationships",
            "handler": traverse_relationships,
            "description": _tool_description(traverse_relationships, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": {"type": "object", "additionalProperties": True},
            "annotations": readonly,
        },
        {
            "name": "summarize_neighborhood",
            "handler": summarize_neighborhood,
            "description": _tool_description(summarize_neighborhood, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GENERIC_GRAPH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "compute_context_bundle",
            "handler": compute_context_bundle,
            "description": _tool_description(compute_context_bundle, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GENERIC_GRAPH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
    ]

    if LEGACY_SEARCH_DOCUMENTATION_ENABLED:
        specs.append(
            {
                "name": "search_documentation",
                "handler": search_documentation,
                "description": _tool_description(search_documentation, ""),
                "input_schema": {"type": "object", "additionalProperties": True},
                "output_schema": {"type": "object", "additionalProperties": True},
                "annotations": readonly,
            }
        )

    return specs


def _summary_for_tool(name: str, result: dict) -> str:
    if isinstance(result, dict) and "error" in result:
        message = result.get("error", {}).get("message", "error")
        return f"{name} error: {message}"
    if "results" in result and isinstance(result["results"], list):
        return f"{name} returned {len(result['results'])} results."
    if "quotes" in result and isinstance(result["quotes"], list):
        return f"{name} returned {len(result['quotes'])} quotes."
    if "nodes" in result and isinstance(result["nodes"], list):
        return f"{name} returned {len(result['nodes'])} nodes."
    if "paths" in result and isinstance(result["paths"], list):
        return f"{name} returned {len(result['paths'])} paths."
    if "children" in result and isinstance(result["children"], list):
        return f"{name} returned {len(result['children'])} children."
    if "parents" in result and isinstance(result["parents"], list):
        return f"{name} returned {len(result['parents'])} parents."
    return f"{name} completed."


async def _invoke_tool(handler, arguments: dict[str, Any]) -> dict:
    sig = inspect.signature(handler)
    kwargs = {k: v for k, v in (arguments or {}).items() if k in sig.parameters}
    return await handler(**kwargs)


def build_mcp_server() -> Server:
    server = Server("wekadocs", instructions=_instructions, lifespan=lifespan)
    tool_specs = _tool_specs()
    tool_map = {spec["name"]: spec["handler"] for spec in tool_specs}

    @server.list_tools()
    async def _list_tools():
        tools: list[types.Tool] = []
        for spec in tool_specs:
            tools.append(
                types.Tool(
                    name=spec["name"],
                    description=spec["description"],
                    inputSchema=spec["input_schema"],
                    outputSchema=spec["output_schema"],
                    annotations=spec.get("annotations"),
                )
            )
        return tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None):
        handler = tool_map.get(name)
        if handler is None:
            payload = _error_payload("INVALID_ARGUMENT", f"Unknown tool '{name}'")
            summary = _summary_for_tool(name, payload)
            return ([types.TextContent(type="text", text=summary)], payload)
        if name == "search_documentation" and not LEGACY_SEARCH_DOCUMENTATION_ENABLED:
            payload = _error_payload(
                "INVALID_ARGUMENT", "search_documentation is disabled"
            )
            summary = _summary_for_tool(name, payload)
            return ([types.TextContent(type="text", text=summary)], payload)

        result = await _invoke_tool(handler, arguments or {})
        summary = _summary_for_tool(name, result)
        return ([types.TextContent(type="text", text=summary)], result)

    @server.list_resources()
    async def _list_resources():
        size = None
        path = _retrieval_playbook_path()
        if os.path.exists(path):
            size = os.path.getsize(path)
        return [
            types.Resource(
                uri=RETRIEVAL_PLAYBOOK_URI,
                name="retrieval_playbook",
                title="Retrieval playbook",
                description="Graph-first retrieval playbook (markdown).",
                mimeType="text/markdown",
                size=size,
                annotations=types.Annotations(audience=["assistant"], priority=0.2),
            )
        ]

    @server.list_resource_templates()
    async def _list_resource_templates():
        templates = [
            types.ResourceTemplate(
                uriTemplate=SCRATCH_RESOURCE_TEMPLATE,
                name="scratch",
                title="Scratch passage",
                description="Scratch storage for bounded retrieval passages.",
                mimeType="text/plain",
                annotations=types.Annotations(audience=["assistant"], priority=0.1),
            )
        ]
        if DIAGNOSTICS_RESOURCES_ENABLED:
            templates.append(
                types.ResourceTemplate(
                    uriTemplate=DIAGNOSTICS_RESOURCE_TEMPLATE,
                    name="diagnostics",
                    title="Retrieval diagnostics summary",
                    description="Operator-only diagnostics summary (markdown).",
                    mimeType="text/markdown",
                    annotations=types.Annotations(
                        audience=["assistant"], priority=0.05
                    ),
                )
            )
        return templates

    @server.read_resource()
    async def _read_resource(uri: str):
        uri = str(uri)
        if uri == RETRIEVAL_PLAYBOOK_URI:
            return _read_retrieval_playbook()

        parsed = _parse_scratch_uri(uri)
        if parsed:
            session_id, passage_id = parsed
            deps = _get_deps(None)
            if not deps.scratch:
                return "Scratch store unavailable"
            effective_session = _resolve_session_id(None, None)
            if session_id != effective_session:
                return "Scratch access denied for this session"
            entry = await deps.scratch.get(session_id, passage_id)
            if not entry:
                return "Scratch entry not found"
            return entry.get("text", "")

        parsed = _parse_diagnostics_uri(uri)
        if parsed:
            if not DIAGNOSTICS_RESOURCES_ENABLED:
                return "Diagnostics resources are disabled"
            date, diagnostic_id = parsed
            try:
                return _DIAGNOSTIC_EMITTER.read_markdown(
                    date=date, diagnostic_id=diagnostic_id
                )
            except Exception as exc:
                return f"Diagnostics unavailable: {exc}"

        return f"Unknown resource: {uri}"

    @server.list_prompts()
    async def _list_prompts():
        return [
            types.Prompt(name=item["name"], description=item.get("description"))
            for item in PROMPT_DEFINITIONS
        ]

    @server.get_prompt()
    async def _get_prompt(name: str, arguments: dict[str, str] | None):
        for item in PROMPT_DEFINITIONS:
            if item["name"] == name:
                return types.GetPromptResult(
                    description=item.get("description"),
                    messages=[
                        types.PromptMessage(
                            role="assistant",
                            content=types.TextContent(
                                type="text", text=item["content"]
                            ),
                        )
                    ],
                )
        raise ValueError(f"Unknown prompt '{name}'")

    return server

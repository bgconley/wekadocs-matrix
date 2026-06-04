# =============================================================================
# Utility functions for MCP server
# =============================================================================
"""
Pure utility functions: cursors, budgets, payloads, text processing, URI helpers,
and the shared Deps dataclass.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Optional
from uuid import uuid4

from mcp.server.lowlevel.server import request_ctx

from src.mcp_server.scratch_store import ScratchStore
from src.services import ContextBudgetManager, GraphService, TextService
from src.services.context_assembler import (
    ContextAssemblerService,
    SummarizationService,
)
from src.services.context_budget_manager import BudgetExceeded
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    cursor_returned_total,
    duplicates_suppressed_total,
    over_budget_attempts_total,
    partial_responses_total,
    tool_response_bytes,
)

try:
    from src.neo.schema_validator import validate_neo4j_schema
except Exception:
    validate_neo4j_schema = None  # type: ignore

try:
    from src.mcp_server.query_service import QueryService, get_query_service
except Exception:
    QueryService = None  # type: ignore
    get_query_service = None  # type: ignore

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
KB_EVIDENCE_INTERNAL_FETCH_K = 60  # How deep the evidence pack searches internally
KB_EVIDENCE_MAX_FETCH_K = 150  # Hard cap on internal retrieval depth
KB_EVIDENCE_MAX_QUOTES = int(os.getenv("MCP_EVIDENCE_MAX_QUOTES", "50"))
KB_EVIDENCE_GRAPH_EXPANSION_ENABLED = os.getenv(
    "MCP_EVIDENCE_GRAPH_EXPANSION_ENABLED", "false"
).lower() in {"1", "true", "yes", "on"}
KB_EVIDENCE_GRAPH_SYNTHETIC_FUSED_SCORE = float(
    os.getenv("MCP_EVIDENCE_GRAPH_SYNTHETIC_FUSED_SCORE", "0.08")
)
KB_EVIDENCE_FORCE_METADATA_LIMITATIONS = os.getenv(
    "MCP_EVIDENCE_FORCE_METADATA_LIMITATIONS", "true"
).lower() in {"1", "true", "yes", "on"}
KB_EVIDENCE_METADATA_LIMITATIONS_MIN_SCORE = float(
    os.getenv("MCP_EVIDENCE_METADATA_LIMITATIONS_MIN_SCORE", "0.15")
)
SCRATCH_TTL_SECONDS = int(os.getenv("MCP_SCRATCH_TTL_SECONDS", "1800"))
SCRATCH_MAX_BYTES = int(os.getenv("MCP_SCRATCH_MAX_BYTES", str(256 * 1024 * 1024)))
# _SESSION_IDS removed - now using single _GLOBAL_SESSION_ID for stable scratch storage
LEGACY_SEARCH_DOCUMENTATION_ENABLED = os.getenv(
    "ENABLE_LEGACY_SEARCH_DOCUMENTATION", "false"
).lower() in {"1", "true", "yes", "on"}
DIAGNOSTICS_RESOURCES_ENABLED = os.getenv(
    "MCP_DIAGNOSTICS_RESOURCES_ENABLED", "false"
).lower() in {"1", "true", "yes", "on"}
MCP_TOOL_PROFILE = os.getenv("MCP_TOOL_PROFILE", "production")


# ── Module-level config ───────────────────────────────────────────────
_config = get_config()
_neo4j_disabled = getattr(getattr(_config, "hybrid", None), "neo4j_disabled", False)

RETRIEVAL_PLAYBOOK_URI = "wekadocs://retrieval_playbook"
SCRATCH_RESOURCE_TEMPLATE = "wekadocs://scratch/{session_id}/{passage_id}"
DIAGNOSTICS_RESOURCE_TEMPLATE = "wekadocs://diagnostics/{date}/{diagnostic_id}"


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("utf-8")).decode("utf-8")


def _decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    try:
        return int(base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8"))
    except Exception:
        return 0


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


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


# Single global session ID - MCP SDK doesn't provide stable session identifiers,
# so we use one session ID for all operations to ensure scratch storage works
_GLOBAL_SESSION_ID: Optional[str] = None


def _resolve_session_id(ctx: Any | None, provided: Optional[str]) -> str:
    """
    Resolve session ID for scratch storage operations.

    IMPORTANT: The MCP SDK's request_context.session object changes between calls,
    making it unreliable as a dict key. We use a single global session ID to ensure
    passage_ids stored by kb_search can be retrieved by kb_read_excerpt.
    """
    global _GLOBAL_SESSION_ID
    if provided:
        return provided
    # Use single global session ID for all operations
    if _GLOBAL_SESSION_ID is None:
        _GLOBAL_SESSION_ID = f"server-{uuid4()}"
    return _GLOBAL_SESSION_ID


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


def _is_graph_expanded_source(source: Any) -> bool:
    return str(source or "").strip().lower() == "graph_expanded"


def _is_metadata_limitations_candidate(
    entry: dict[str, Any], query_tokens: list[str]
) -> bool:
    if "metadata" not in query_tokens:
        return False
    haystack = f"{entry.get('title') or ''}\n{entry.get('text') or ''}".lower()
    return "metadata" in haystack and "limitation" in haystack


def _quote_from_passage(
    *,
    passage: dict[str, Any],
    query_tokens: list[str],
    max_quote_chars: int,
    context_chars: int,
    effective_session: str,
) -> Optional[dict[str, Any]]:
    entry = passage["entry"]
    text = entry.get("text") or ""
    if not text:
        return None

    # Score each span with blended retrieval + lexical
    retrieval_score = passage["retrieval_score"]
    retrieval_weight = 0.7
    lexical_weight = 0.3
    best_span = None
    best_blended = -1.0
    for span, start, end in _split_spans_with_offsets(text):
        lowered = span.lower()
        hits = sum(1 for t in query_tokens if t in lowered)
        lexical = hits / max(1, len(query_tokens)) if query_tokens else 0.0
        blended = (retrieval_weight * retrieval_score) + (lexical_weight * lexical)
        if blended > best_blended:
            best_blended = blended
            best_span = (start, end, blended)

    if best_span is None:
        return None

    start, end, blended_score = best_span
    start = max(0, start - context_chars)
    end = min(len(text), end + context_chars)
    quote = text[start:end].strip()[:max_quote_chars]

    return {
        "quote": quote,
        "passage_id": passage["passage_id"],
        "section_id": entry.get("section_id"),
        "doc_tag": entry.get("doc_tag"),
        "title": entry.get("title"),
        "parent_path": entry.get("parent_path_norm"),
        "uri": ScratchStore.build_uri(effective_session, passage["passage_id"]),
        "confidence": round(blended_score, 3),
        "source": entry.get("source", "hybrid"),
        "rank": passage["rank"],
    }


def _format_bullets(text: str) -> str:
    spans = _split_spans(text)
    if not spans:
        return text
    return "\n".join(f"- {span}" for span in spans)


def _get_deps(ctx: Any | None) -> Deps:
    request_context = _get_request_context(ctx)
    if request_context is None:
        raise RuntimeError("MCP request context is required")
    deps = request_context.lifespan_context
    # Lazy initialization on first tool call
    deps.ensure_initialized()
    return deps


class Deps:
    """Dependencies shared across MCP tool calls via lifespan context.

    Uses lazy initialization to ensure fast MCP handshake response.
    Heavy initialization (Neo4j, QueryService) is deferred until first tool call.
    """

    query: Optional[QueryService] = None
    graph: Optional[GraphService] = None
    text: Optional[TextService] = None
    summarizer: Optional[SummarizationService] = None
    assembler: Optional[ContextAssemblerService] = None
    scratch: Optional[ScratchStore] = None
    _initialized: bool = False

    def ensure_initialized(self) -> None:
        """Lazy initialization of heavy dependencies.

        Called on first tool invocation to avoid blocking MCP handshake.
        This allows Claude Desktop to complete the initialize/tools/list
        sequence quickly, then do heavy work on first actual tool call.
        """
        if self._initialized:
            return

        logger.info("Deps: lazy-initializing dependencies on first tool call")
        _DIAGNOSTIC_EMITTER.cleanup()

        # Initialize QueryService (embedder loaded on first search)
        self.query = get_query_service()
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()

        # Validate Neo4j schema (Phase 3 hardening)
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

        self.graph = GraphService(neo4j_driver)
        self.text = TextService(neo4j_driver)
        self.summarizer = SummarizationService(self.graph)
        self.assembler = ContextAssemblerService(self.graph, self.text)
        self.scratch = ScratchStore(
            max_bytes=SCRATCH_MAX_BYTES,
            ttl_seconds=SCRATCH_TTL_SECONDS,
            logger=logger,
        )
        self._initialized = True
        logger.info("Deps: lazy initialization complete, QueryService ready")


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

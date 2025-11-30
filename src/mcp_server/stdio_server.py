"""
STDIO MCP Server for Claude Desktop
Implements Phase 1 STDIO transport using FastMCP SDK with lifespan pattern.
See: /docs/spec.md §9 (Interfaces)
See: docs/cdx-outputs/retrieval_fix.json for v2 tool contract.
"""

from __future__ import annotations

import base64
import builtins
import json
import logging

# --- STDIO-SAFE BOOTSTRAP: Must be at the very top ---
import os
import sys
import warnings
from uuid import uuid4
from weakref import WeakKeyDictionary

# Unbuffered, UTF-8 (avoids partial writes / encoding surprises)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# 1) Route all Python logging to STDERR (never STDOUT)
root = logging.getLogger()
# Remove any handlers installed during module import
for h in list(root.handlers):
    root.removeHandler(h)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(
    logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
)
root.addHandler(stderr_handler)
root.setLevel(logging.INFO)

# Silence noisy libraries that might spam
for noisy in ("httpx", "urllib3", "botocore", "boto3", "opentelemetry", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# 2) Redirect accidental print() calls to STDERR
_builtin_print = builtins.print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    return _builtin_print(*args, **kwargs)


builtins.print = _stderr_print

# 3) If you use Rich or Loguru, force them to STDERR (examples below):
try:
    from rich.console import Console
    from rich.logging import RichHandler

    root.handlers.clear()
    root.addHandler(
        RichHandler(console=Console(stderr=True), markup=False, show_path=False)
    )
except Exception:
    pass

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=False, diagnose=False)
except Exception:
    pass

# Optional: demote Deprecation/Runtime warnings to STDERR (default)
warnings.simplefilter("default")

# 4) Configure structlog to use STDERR (our app uses structlog via get_logger)
try:
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),  # Human-readable for stderr
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
except Exception:
    pass
# --- END STDIO-SAFE BOOTSTRAP ---

# Now safe to import application modules
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Optional  # noqa: E402

from mcp.server.fastmcp import Context, FastMCP  # noqa: E402
from mcp.server.session import ServerSession  # noqa: E402

from src.mcp_server.query_service import QueryService, get_query_service  # noqa: E402
from src.query.traversal import TraversalService  # noqa: E402
from src.services import ContextBudgetManager, GraphService, TextService  # noqa: E402
from src.services.context_assembler import (  # noqa: E402
    ContextAssemblerService,
    SummarizationService,
)
from src.services.context_budget_manager import BudgetExceeded  # noqa: E402
from src.shared.connections import get_connection_manager  # noqa: E402
from src.shared.observability import get_logger  # noqa: E402
from src.shared.observability.metrics import (  # noqa: E402
    cursor_returned_total,
    duplicates_suppressed_total,
    over_budget_attempts_total,
    partial_responses_total,
    summary_calls_total,
    tool_response_bytes,
)

logger = get_logger(__name__)

MAX_TOKENS_PER_TURN = 14_000
MAX_RESPONSE_BYTES = 524_288
MAX_TEXT_BYTES_PER_CALL = 32_768
DEFAULT_PAGE_SIZE = 25
_SESSION_IDS: "WeakKeyDictionary[Any, str]" = WeakKeyDictionary()
LEGACY_SEARCH_DOCUMENTATION_ENABLED = os.getenv(
    "ENABLE_LEGACY_SEARCH_DOCUMENTATION", "false"
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


def _resolve_session_id(
    ctx: Context[ServerSession, Deps] | None, provided: Optional[str]
) -> str:
    if provided:
        return provided
    if ctx is None:
        return f"server-{uuid4()}"
    session = ctx.session
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


def _get_deps(ctx: Context[ServerSession, Deps] | None) -> Deps:
    if ctx is None:
        raise RuntimeError("MCP context is required")
    return ctx.request_context.lifespan_context


@dataclass
class Deps:
    """Dependencies shared across MCP tool calls via lifespan context."""

    query: Optional[QueryService] = None
    graph: Optional[GraphService] = None
    text: Optional[TextService] = None
    summarizer: Optional[SummarizationService] = None
    assembler: Optional[ContextAssemblerService] = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[Deps]:
    """
    Lifespan context manager for dependency injection.
    Initializes QueryService once and shares across all tool calls.
    Prevents embedder cold-start penalty (2.5s) on every request.
    """
    deps = Deps()
    logger.info("STDIO server lifespan: initializing dependencies")

    try:
        # Initialize QueryService (embedder loaded on first search)
        deps.query = get_query_service()
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()
        deps.graph = GraphService(neo4j_driver)
        deps.text = TextService(neo4j_driver)
        deps.summarizer = SummarizationService(deps.graph)
        deps.assembler = ContextAssemblerService(deps.graph, deps.text)
        logger.info("STDIO server lifespan: QueryService ready")
        yield deps
    finally:
        # Cleanup connections if needed
        if deps.query:
            close = getattr(deps.query, "close", None)
            if callable(close):
                await close()
                logger.info("STDIO server lifespan: connections closed")


# Create FastMCP instance with lifespan and graph-first instructions
GRAPH_FIRST_INSTRUCTIONS = (
    "You are connected to the Weka docs graph via MCP tools. "
    'Always start with search_sections using verbosity="snippet" to collect seed IDs; '
    "then explore the neighborhood with expand_neighbors, get_paths_between, describe_nodes, "
    "list_children, list_parents, get_entities_for_sections, get_sections_for_entities, and compute_context_bundle. "
    "Only after mapping the graph should you call get_section_text for a small number of high-value sections, "
    "using conservative max_bytes_per (4–8KB) and multiple small calls. Avoid unbounded text dumps and prefer cursors."
)

mcp = FastMCP("wekadocs", instructions=GRAPH_FIRST_INSTRUCTIONS, lifespan=lifespan)

# Register a retrieval playbook resource (read-only)
try:

    @mcp.resource("retrieval_playbook")
    def retrieval_playbook():
        """Graph-first retrieval playbook (markdown)."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "docs",
            "mcp",
            "retrieval_playbook.md",
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"mimeType": "text/markdown", "text": f.read()}
        except Exception as e:
            return {"mimeType": "text/plain", "text": f"Playbook unavailable: {e}"}

except Exception:
    logger.warning(
        "Failed to register retrieval_playbook resource; continuing without it"
    )


async def search_documentation(
    query: str,
    top_k: int = 20,
    verbosity: str = "graph",
    ctx: Context[ServerSession, Deps] | None = None,
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

    # Report progress to Claude Desktop UI
    if ctx:
        await ctx.report_progress(progress=0.1, total=1.0, message="Encoding query")

    try:
        # Access shared QueryService from lifespan context
        query_service = ctx.request_context.lifespan_context.query

        if ctx:
            await ctx.report_progress(
                progress=0.3, total=1.0, message="Searching vectors"
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

        if ctx:
            await ctx.report_progress(
                progress=0.9, total=1.0, message="Building response"
            )

        # Convert to JSON-serializable dict
        result = response.to_dict()

        if ctx:
            await ctx.report_progress(progress=1.0, total=1.0, message="Complete")

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


if LEGACY_SEARCH_DOCUMENTATION_ENABLED:
    search_documentation = mcp.tool()(search_documentation)
    logger.info(
        "search_documentation MCP tool enabled (graph-aware hybrid responses available)"
    )
else:
    logger.warning(
        "search_documentation MCP tool disabled via ENABLE_LEGACY_SEARCH_DOCUMENTATION; "
        "LLMs must orchestrate low-level graph tools manually"
    )


@mcp.tool()
async def search_sections(
    query: str,
    top_k: int = 20,
    filters: Optional[dict[str, Any]] = None,
    cursor: Optional[str] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """Compact section search returning ids + metadata only."""

    logger.info(
        "STDIO tool called: search_sections(query='%s', top_k=%s, cursor=%s)",
        query,
        top_k,
        cursor,
    )
    if not ctx or not ctx.request_context.lifespan_context.query:
        raise RuntimeError("QueryService not initialized")

    query_service = ctx.request_context.lifespan_context.query
    effective_session = _resolve_session_id(ctx, session_id)
    offset = _decode_cursor(cursor)
    page = max(1, min(page_size or DEFAULT_PAGE_SIZE, 100))
    total_cap = max(page, top_k or page)
    fetch_k = min(total_cap, offset + page + 1)
    if fetch_k <= 0:
        rows = []
        metrics = {}
    else:
        rows, metrics = query_service.search_sections_light(
            query=query, fetch_k=fetch_k, filters=filters
        )
    sliced = rows[offset : offset + page]
    more = len(rows) > offset + len(sliced)
    next_cursor = _encode_cursor(offset + len(sliced)) if more else None
    results = []
    for idx, chunk in enumerate(sliced):
        score = chunk.fused_score or chunk.vector_score or chunk.bm25_score or 0.0
        source = "hybrid"
        if chunk.vector_score and not chunk.bm25_score:
            source = "vector"
        elif chunk.bm25_score and not chunk.vector_score:
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
                "graph_score": float(getattr(chunk, "graph_score", 0.0) or 0.0),
                "graph_distance": int(getattr(chunk, "graph_distance", 0) or 0),
                "connection_count": int(getattr(chunk, "connection_count", 0) or 0),
                "mention_count": int(getattr(chunk, "mention_count", 0) or 0),
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


@mcp.tool()
async def traverse_relationships(
    start_ids: list[str],
    rel_types: list[str] | None = None,
    max_depth: int = 2,
    include_text: bool = False,
    ctx: Context[ServerSession, Deps] | None = None,
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

    if ctx:
        await ctx.report_progress(
            progress=0.1, total=1.0, message="Initializing traversal"
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

        # Get Neo4j driver from connection manager
        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()

        if ctx:
            await ctx.report_progress(
                progress=0.3, total=1.0, message="Traversing graph"
            )

        # Create traversal service and execute
        traversal_svc = TraversalService(neo4j_driver)
        result = traversal_svc.traverse(
            start_ids=start_ids,
            rel_types=rel_types,
            max_depth=max_depth,
            include_text=include_text,
        )

        if ctx:
            await ctx.report_progress(
                progress=0.9, total=1.0, message="Formatting results"
            )

        # Convert to dict
        result_dict = result.to_dict()

        if ctx:
            await ctx.report_progress(progress=1.0, total=1.0, message="Complete")

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


@mcp.tool()
async def describe_nodes(
    node_ids: list[str],
    fields: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    Return projection-only node metadata without heavy text.

    When to use:
    - After search_sections or expand_neighbors to triage nodes cheaply
    - Before calling get_section_text, to choose a few high-value sections

    Recipe:
    - search_sections → expand_neighbors (1–2 hops) → describe_nodes → get_section_text (few)
    """

    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    result = deps.graph.describe_nodes(
        node_ids=node_ids, fields=fields, budget=budget, phase="neighbors"
    )
    effective_session = _resolve_session_id(ctx, session_id)
    return _graph_response("describe_nodes", effective_session, result)


@mcp.tool()
async def expand_neighbors(
    node_ids: list[str],
    rel_types: Optional[list[str]] = None,
    direction: str = "both",
    max_hops: int = 1,
    include_snippet: bool = False,
    cursor: Optional[str] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
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


@mcp.tool()
async def get_paths_between(
    a_ids: list[str],
    b_ids: list[str],
    rel_types: Optional[list[str]] = None,
    max_hops: int = 3,
    max_paths: int = 10,
    cursor: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
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


@mcp.tool()
async def list_children(
    parent_id: str,
    cursor: Optional[str] = None,
    page_size: int = 50,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    List child sections for a parent, with cursor pagination.

    When to use:
    - Navigate hierarchy without fetching full text; find nearby relevant sections

    Recipe:
    - search_sections → list_parents / list_children → describe_nodes → selective get_section_text
    """
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


@mcp.tool()
async def list_parents(
    section_ids: list[str],
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    List parent sections for given sections.

    When to use:
    - Move up the hierarchy to situate a section before fetching text

    Recipe:
    - search_sections → list_parents → describe_nodes → get_section_text (few)
    """
    deps = _get_deps(ctx)
    if not deps.graph:
        raise RuntimeError("GraphService not initialized")
    budget = _new_budget()
    effective_session = _resolve_session_id(ctx, session_id)
    result = deps.graph.list_parents(section_ids=section_ids, budget=budget)
    return _graph_response("list_parents", effective_session, result)


@mcp.tool()
async def get_entities_for_sections(
    section_ids: list[str],
    labels: Optional[list[str]] = None,
    max_per_section: int = 20,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    Pivot from sections to entities (and back) to broaden/narrow context.

    When to use:
    - Extract key entities from candidate sections before reading text

    Recipe:
    - search_sections → get_entities_for_sections → get_sections_for_entities → describe_nodes → get_section_text
    """
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


@mcp.tool()
async def get_sections_for_entities(
    entity_ids: list[str],
    max_per: int = 20,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    Pivot from entities to sections to target a small set for text fetches.

    When to use:
    - After extracting entities, find the most relevant sections linked to them

    Recipe:
    - search_sections → get_entities_for_sections → get_sections_for_entities → describe_nodes → get_section_text
    """
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


@mcp.tool()
async def get_section_text(
    section_ids: list[str],
    max_bytes_per: int = 8192,
    session_id: Optional[str] = None,
    ctx: Context[ServerSession, Deps] | None = None,
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


@mcp.tool()
async def summarize_neighborhood(
    node_ids: list[str],
    token_budget: int = 400,
    ctx: Context[ServerSession, Deps] | None = None,
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


@mcp.tool()
async def compute_context_bundle(
    seeds: list[str],
    strategy: str = "hybrid",
    token_budget: int = 2_000,
    ctx: Context[ServerSession, Deps] | None = None,
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


# Register curated prompts (graph-first recipes)
try:
    mcp.add_prompt(
        name="graph.neighborhood_summary",
        description="Explore local graph neighborhood and fetch small excerpts only as needed.",
        messages=[
            {
                "role": "system",
                "content": "Use search_sections to seed, expand with expand_neighbors (1–2 hops), summarize with describe_nodes, and only then fetch small text via get_section_text (4–8KB).",
            }
        ],
    )
    mcp.add_prompt(
        name="graph.connect_concepts",
        description="Explain how two concepts/sections are related using graph paths.",
        messages=[
            {
                "role": "system",
                "content": "Find seeds for A and B via search_sections, call get_paths_between, summarize nodes with describe_nodes, then fetch minimal text for pivotal sections.",
            }
        ],
    )
    mcp.add_prompt(
        name="graph.task_context_bundle",
        description="Assemble a budgeted context bundle for a downstream task.",
        messages=[
            {
                "role": "system",
                "content": "Identify candidate sections via search + graph tools, then call compute_context_bundle with an explicit budget; avoid large unstructured text dumps.",
            }
        ],
    )
except Exception:
    logger.warning("Failed to register MCP prompts; continuing without them")

if __name__ == "__main__":
    # Run MCP server with STDIO transport
    # Claude Desktop will launch this via: docker exec -i weka-mcp-server python -m src.mcp_server.stdio_server
    logger.info("Starting wekadocs MCP server with STDIO transport")
    mcp.run(transport="stdio")

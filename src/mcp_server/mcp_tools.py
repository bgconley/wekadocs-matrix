# =============================================================================
# MCP tool implementations and schema definitions
# =============================================================================
"""
All 18 tool implementations (kb_search, search_sections, graph tools, etc.),
input/output JSON schemas, and tool spec builder.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Optional
from uuid import uuid4

import mcp.types as types

from src.evidence.coverage import mark_budget_state
from src.evidence.models import EvidenceRequest
from src.evidence.serializers import evidence_package_to_mcp_payload
from src.evidence.service import EvidenceService
from src.mcp_server.mcp_search import (
    _emit_diagnostics,
    _expand_evidence_with_structure,
    _extract_evidence_from_passages,
    _infer_source_tags,
    _kb_search_candidates,
)
from src.mcp_server.mcp_utils import (
    DEFAULT_PAGE_SIZE,
    DIAGNOSTICS_RESOURCES_ENABLED,
    KB_EVIDENCE_GRAPH_EXPANSION_ENABLED,
    KB_EVIDENCE_INTERNAL_FETCH_K,
    KB_EVIDENCE_MAX_FETCH_K,
    KB_EVIDENCE_MAX_QUOTES,
    KB_SEARCH_DEFAULT_TOP_K,
    LEGACY_SEARCH_DOCUMENTATION_ENABLED,
    MAX_TEXT_BYTES_PER_CALL,
    MCP_TOOL_PROFILE,
    ScratchStore,
    _apply_budget,
    _coerce_bool,
    _decode_cursor,
    _diagnostics_uri,
    _encode_cursor,
    _error_payload,
    _finalize_payload,
    _format_bullets,
    _get_deps,
    _graph_disabled_payload,
    _graph_response,
    _neo4j_disabled,
    _new_budget,
    _normalize_scope,
    _report_progress,
    _resolve_session_id,
)
from src.mcp_server.retrieval_trace import (
    RetrievalTraceBuilder,
    set_active_trace,
    write_trace,
)
from src.query.traversal import TraversalService
from src.shared.connections import get_connection_manager
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    excerpt_truncations_total,
    summary_calls_total,
)

logger = get_logger(__name__)

# Production: evidence-pack-first workflow with 3 tools.
# Analyst: full tool access with evidence pack as recommended start.
# Legacy instructions retained for reference but no longer active.

PRODUCTION_INSTRUCTIONS = (
    "You are connected to the WEKA documentation knowledge base. "
    "Start with kb.retrieve_evidence to get an evidence pack for any question. "
    "The evidence pack includes quotes with confidence scores, document context "
    "(doc_tag, parent_path), and coverage metadata showing retrieval depth. "
    "Use kb.read_excerpt to read the full text of a passage if you need more context "
    "beyond the quote snippet. "
    "Use graph.expand only for follow-up navigation when the user asks about related "
    "content near a specific section (e.g., 'what else is in that configuration page?'). "
    "Do NOT use graph.expand for initial research — kb.retrieve_evidence handles that "
    "with server-side graph enrichment and deep retrieval. "
    "CRITICAL: Only state that a feature, API, or capability is supported if it is "
    "EXPLICITLY listed in the documentation. If something is not explicitly documented, "
    "clearly state that you could not find documentation for it."
)

ANALYST_INSTRUCTIONS = (
    "You are connected to the WEKA documentation knowledge base with full tool access. "
    "For most queries, start with kb.retrieve_evidence for a server-built evidence pack "
    "with retrieval-score-based confidence and coverage metadata. "
    "Use kb.search for browsing candidates, graph.* tools for structural exploration "
    "(graph.expand, graph.describe, graph.paths, graph.parents, graph.children), "
    "and kb.get_section_text for full text retrieval. "
    "kb.extract_evidence extracts quotes from known passage_ids (post-search). "
    "CRITICAL: Only state that a feature, API, or capability is supported if it is "
    "EXPLICITLY listed in the documentation. If something is not explicitly documented, "
    "clearly state that you could not find documentation for it."
)

# Select instructions based on tool profile
_instructions = (
    PRODUCTION_INSTRUCTIONS
    if MCP_TOOL_PROFILE == "production"
    else ANALYST_INSTRUCTIONS
)
logger.info(
    "MCP instructions mode",
    neo4j_disabled=_neo4j_disabled,
    profile=MCP_TOOL_PROFILE,
    mode="production" if MCP_TOOL_PROFILE == "production" else "analyst",
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
    "Use when you already have a passage_id from kb_search and need a bounded excerpt. "
    "IMPORTANT: You MUST use the exact passage_id UUID returned by kb_search (e.g., '022160622467452ba0ccf501cbf771c1'), "
    "NOT a constructed document path. Do not use for discovery; use kb.search first. "
    "Returns a capped excerpt (default 300 tokens, max 800, 32KB per call). "
    "If you need more context, call kb.expand_excerpt."
)
KB_EXPAND_EXCERPT_DESCRIPTION = (
    "Use to expand around the last excerpt for a passage_id from kb_search. "
    "IMPORTANT: You MUST use the exact passage_id UUID returned by kb_search. "
    "Do not use for discovery; use kb.search first. "
    "Returns a bounded expansion before/after the last excerpt window."
)
KB_EXTRACT_EVIDENCE_DESCRIPTION = (
    "Use to extract minimal quotes that answer a question from known passage_ids. "
    "IMPORTANT: You MUST use the exact passage_id UUIDs returned by kb_search. "
    "Do not use for discovery; use kb.search to get passage_ids first. "
    "Returns at most max_quotes short quotes with citations."
)
KB_RETRIEVE_EVIDENCE_DESCRIPTION = (
    "Use when you want the best evidence in one call. "
    "This runs kb.search then kb.extract_evidence, returning quotes only. "
    "Do not use when you need exploratory graph traversal."
)


PROMPT_DEFINITIONS = [
    {
        "name": "graph_neighborhood_summary",
        "description": "Explore local graph neighborhood and fetch small excerpts only as needed.",
        "content": (
            "Use search_sections to seed, expand with expand_neighbors (1–2 hops), "
            "summarize with describe_nodes, and only then fetch small text via "
            "get_section_text (4–8KB)."
        ),
    },
    {
        "name": "graph_connect_concepts",
        "description": "Explain how two concepts/sections are related using graph paths.",
        "content": (
            "Find seeds for A and B via search_sections, call get_paths_between, "
            "summarize nodes with describe_nodes, then fetch minimal text for pivotal sections."
        ),
    },
    {
        "name": "graph_task_context_bundle",
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
        "question": {
            "type": "string",
            "description": "The question to find evidence for",
        },
        "max_quotes": {
            "type": "integer",
            "default": 6,
            "description": f"Max evidence quotes to return (1-{KB_EVIDENCE_MAX_QUOTES})",
        },
        "max_quote_tokens": {
            "type": "integer",
            "default": 80,
            "description": "Max tokens per quote",
        },
        "include_context_tokens": {
            "type": "integer",
            "default": 20,
            "description": "Context tokens around each quote",
        },
        "retrieval_depth": {
            "type": "integer",
            "default": KB_EVIDENCE_INTERNAL_FETCH_K,
            "description": "Internal search depth (default 60, max 150). Searches this many candidates to find the best quotes.",
        },
        "graph_enrichment": {
            "type": "boolean",
            "description": "Enable evidence-only structural graph expansion (default false for precision mode).",
        },
        "top_k": {
            "type": "integer",
            "default": KB_SEARCH_DEFAULT_TOP_K,
            "description": "Backward compat alias for max_quotes",
        },
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
        "kb_search",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=partial,
        limit_reason=limit_reason,
        session_id=effective_session,
        duplicates=payload.get("duplicates", 0),
    )
    diagnostic = await _emit_diagnostics(
        tool_name="kb_search",
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
    """Read a bounded excerpt from scratch storage.

    Note: passage_id should be the UUID returned by kb_search, but if the AI
    passes a section_id or constructed path instead, we'll try to find it.
    """

    deps = _get_deps(ctx)
    if not deps.scratch:
        raise RuntimeError("ScratchStore not initialized")

    try:
        _normalize_scope(scope)
    except ValueError as exc:
        return _error_payload("SCOPE_VIOLATION", str(exc))

    effective_session = _resolve_session_id(ctx, session_id)
    entry = await deps.scratch.get(effective_session, passage_id)

    # Fallback 1: if passage_id not found, search by section_id in all session entries
    if not entry:
        entry = await deps.scratch.find_by_section_id(effective_session, passage_id)

    # Fallback 2: if still not found, try fetching directly from Neo4j via TextService
    # This handles the search_sections → kb_read_excerpt path where scratch isn't populated
    if not entry and deps.text:
        try:
            budget = _new_budget()
            text_result = deps.text.get_section_text(
                section_ids=[passage_id],
                max_bytes_per=MAX_TEXT_BYTES_PER_CALL,
                budget=budget,
            )
            if text_result.results and len(text_result.results) > 0:
                neo4j_entry = text_result.results[0]
                if neo4j_entry.get("text"):
                    # Create a synthetic entry matching scratch format
                    entry = {
                        "text": neo4j_entry.get("text", ""),
                        "section_id": passage_id,
                        "doc_tag": neo4j_entry.get("doc_tag", ""),
                        "heading": neo4j_entry.get("heading", ""),
                    }
                    logger.info(
                        f"kb_read_excerpt: fetched section '{passage_id}' directly from Neo4j"
                    )
        except Exception as e:
            logger.warning(f"kb_read_excerpt: Neo4j fallback failed: {e}")

    if not entry:
        return _error_payload(
            "INVALID_ARGUMENT",
            f"Unknown passage_id '{passage_id}' for this session. "
            f"Use passage_id from kb_search, or section_id from search_sections.",
        )

    max_tokens = max(1, min(int(max_tokens or 300), 800))
    start_char = max(0, int(start_char or 0))
    char_budget = min(max_tokens * 4, MAX_TEXT_BYTES_PER_CALL)

    text = entry.get("text") or ""
    excerpt = text[start_char : start_char + char_budget]
    truncated = start_char + len(excerpt) < len(text)
    next_start_char = start_char + len(excerpt)
    if truncated:
        excerpt_truncations_total.labels("kb_read_excerpt").inc()

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
        payload, budget, "snippets"  # excerpts use snippets budget phase
    )
    limit_reason = budget_reason if budget_partial else "none"
    finalized = _finalize_payload(
        "kb_read_excerpt",
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

    # Fallback 1: if passage_id not found, search by section_id in scratch
    if not entry:
        entry = await deps.scratch.find_by_section_id(effective_session, passage_id)

    # Fallback 2: if still not found, try fetching directly from Neo4j via TextService
    if not entry and deps.text:
        try:
            budget = _new_budget()
            text_result = deps.text.get_section_text(
                section_ids=[passage_id],
                max_bytes_per=MAX_TEXT_BYTES_PER_CALL,
                budget=budget,
            )
            if text_result.results and len(text_result.results) > 0:
                neo4j_entry = text_result.results[0]
                if neo4j_entry.get("text"):
                    entry = {
                        "text": neo4j_entry.get("text", ""),
                        "section_id": passage_id,
                        "doc_tag": neo4j_entry.get("doc_tag", ""),
                        "heading": neo4j_entry.get("heading", ""),
                        "last_start_char": 0,
                        "last_end_char": 0,
                    }
                    logger.info(
                        f"kb_expand_excerpt: fetched section '{passage_id}' directly from Neo4j"
                    )
        except Exception as e:
            logger.warning(f"kb_expand_excerpt: Neo4j fallback failed: {e}")

    if not entry:
        return _error_payload(
            "INVALID_ARGUMENT",
            f"Unknown passage_id '{passage_id}' for this session. "
            f"Use passage_id from kb_search, or section_id from search_sections.",
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
        excerpt_truncations_total.labels("kb_expand_excerpt").inc()

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
        payload, budget, "snippets"  # excerpts use snippets budget phase
    )
    limit_reason = budget_reason if budget_partial else "none"
    finalized = _finalize_payload(
        "kb_expand_excerpt",
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
        "kb_extract_evidence",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=budget_partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    return finalized


def _write_evidence_trace(package) -> str:
    trace = RetrievalTraceBuilder(
        trace_id=uuid4().hex,
        session_id=package.request.session_id,
    )
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
        enhancer=None,
    )


async def kb_retrieve_evidence(
    question: str,
    top_k: int = KB_SEARCH_DEFAULT_TOP_K,
    max_quotes: int = 6,
    max_quote_tokens: int = 80,
    include_context_tokens: int = 20,
    retrieval_depth: int = KB_EVIDENCE_INTERNAL_FETCH_K,
    graph_enrichment: Optional[bool] = None,
    scope: Optional[dict[str, Any]] = None,
    filters: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    ctx: Any | None = None,
) -> dict:
    """Search deeply then extract the best evidence quotes.

    Unlike kb_search (which returns ranked passages), this tool returns
    minimal evidence quotes with retrieval-score-based confidence. It
    searches much deeper than the number of quotes returned — retrieval_depth
    controls internal search depth while max_quotes controls output size.
    """
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

    internal_fetch_k = max(
        max_quotes,
        min(
            int(retrieval_depth or KB_EVIDENCE_INTERNAL_FETCH_K),
            KB_EVIDENCE_MAX_FETCH_K,
        ),
    )

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
        retrieval_depth=internal_fetch_k,
        graph_enrichment=graph_enabled,
        scope=scope,
        filters=filters,
        options=evidence_options,
        response_mode="evidence_only",
    )

    package = await _build_evidence_service().build_package(request=request, deps=deps)

    payload = evidence_package_to_mcp_payload(package)
    budget = _new_budget()
    tokens_estimate, bytes_estimate, partial, reason = _apply_budget(
        payload, budget, "snippets"
    )
    limit_reason = reason if partial else "none"

    mark_budget_state(
        package.coverage,
        package.gaps,
        partial=partial,
        limit_reason=limit_reason,
    )
    payload["coverage"] = package.coverage.model_dump()
    payload["gaps"] = [gap.model_dump() for gap in package.gaps]

    package.trace_id = _write_evidence_trace(package)

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
        payload["diagnostic_hint"] = (
            f"See retrieval diagnostics {diagnostic['diagnostic_id']}"
        )
        if DIAGNOSTICS_RESOURCES_ENABLED and diagnostic.get("date"):
            payload["diagnostic_uri"] = _diagnostics_uri(
                diagnostic["date"],
                diagnostic["diagnostic_id"],
            )

    finalized = _finalize_payload(
        "kb_retrieve_evidence",
        payload,
        tokens=tokens_estimate,
        bytes_=bytes_estimate,
        partial=partial,
        limit_reason=limit_reason,
        session_id=effective_session,
    )
    finalized["trace_id"] = package.trace_id

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
                "source_tags": _infer_source_tags(chunk),
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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(node_ids, str):
        import json as _json

        node_ids = _json.loads(node_ids)
    if isinstance(fields, str):
        import json as _json

        fields = _json.loads(fields)

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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(node_ids, str):
        import json as _json

        node_ids = _json.loads(node_ids)
    if isinstance(max_hops, str):
        max_hops = int(max_hops)
    if isinstance(page_size, str):
        page_size = int(page_size)
    if isinstance(include_snippet, str):
        include_snippet = include_snippet.lower() in ("true", "1", "yes")

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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(a_ids, str):
        import json as _json

        a_ids = _json.loads(a_ids)
    if isinstance(b_ids, str):
        import json as _json

        b_ids = _json.loads(b_ids)
    if isinstance(rel_types, str):
        import json as _json

        rel_types = _json.loads(rel_types)
    if isinstance(max_hops, str):
        max_hops = int(max_hops)
    if isinstance(max_paths, str):
        max_paths = int(max_paths)

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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(page_size, str):
        page_size = int(page_size)

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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(section_ids, str):
        import json as _json

        section_ids = _json.loads(section_ids)

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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(section_ids, str):
        import json as _json

        section_ids = _json.loads(section_ids)
    if isinstance(labels, str):
        import json as _json

        labels = _json.loads(labels)
    if isinstance(max_per_section, str):
        max_per_section = int(max_per_section)

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
    # Coerce types - MCP clients may send strings instead of proper types
    if isinstance(entity_ids, str):
        import json as _json

        entity_ids = _json.loads(entity_ids)
    if isinstance(max_per, str):
        max_per = int(max_per)

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
    # MCP framework may pass arguments as strings - coerce to proper types
    if isinstance(section_ids, str):
        try:
            section_ids = json.loads(section_ids)
        except json.JSONDecodeError:
            section_ids = [section_ids]  # Single ID passed as string
    if isinstance(max_bytes_per, str):
        max_bytes_per = int(max_bytes_per)
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


def _tool_specs() -> list[dict[str, Any]]:
    readonly = types.ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=False,
        idempotentHint=True,
        destructiveHint=False,
    )

    # ── Canonical tool definitions (dot notation) ─────────────────────
    # These are the primary names matching api-contracts.md and
    # contract tests. 7 bare-name graph duplicates (describe_nodes,
    # expand_neighbors, etc.) have been removed.
    specs = [
        # ── kb.* tools ──
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
            "name": "kb.search_sections",
            "handler": search_sections,
            "description": _tool_description(search_sections, ""),
            "input_schema": SEARCH_SECTIONS_INPUT_SCHEMA,
            "output_schema": KB_SEARCH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "kb.get_section_text",
            "handler": get_section_text,
            "description": _tool_description(get_section_text, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GENERIC_GRAPH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        # ── graph.* tools ──
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
            "name": "graph.traverse",
            "handler": traverse_relationships,
            "description": _tool_description(traverse_relationships, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": {"type": "object", "additionalProperties": True},
            "annotations": readonly,
        },
        {
            "name": "graph.summarize",
            "handler": summarize_neighborhood,
            "description": _tool_description(summarize_neighborhood, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GENERIC_GRAPH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
        {
            "name": "graph.context_bundle",
            "handler": compute_context_bundle,
            "description": _tool_description(compute_context_bundle, ""),
            "input_schema": GENERIC_GRAPH_INPUT_SCHEMA,
            "output_schema": GENERIC_GRAPH_OUTPUT_SCHEMA,
            "annotations": readonly,
        },
    ]

    # ── Backward-compat aliases (underscore names) ────────────────────
    # Temporary: will be removed after one release cycle. Both names
    # point to the same handler; underscore versions are deprecated.
    _UNDERSCORE_ALIASES = {
        "kb_search": "kb.search",
        "kb_read_excerpt": "kb.read_excerpt",
        "kb_expand_excerpt": "kb.expand_excerpt",
        "kb_extract_evidence": "kb.extract_evidence",
        "kb_retrieve_evidence": "kb.retrieve_evidence",
        "search_sections": "kb.search_sections",
        "get_section_text": "kb.get_section_text",
        "graph_describe": "graph.describe",
        "graph_expand": "graph.expand",
        "graph_paths": "graph.paths",
        "graph_parents": "graph.parents",
        "graph_children": "graph.children",
        "graph_entities_for_sections": "graph.entities_for_sections",
        "graph_sections_for_entities": "graph.sections_for_entities",
        "traverse_relationships": "graph.traverse",
        "summarize_neighborhood": "graph.summarize",
        "compute_context_bundle": "graph.context_bundle",
    }
    canonical_by_name = {s["name"]: s for s in specs}
    for alias_name, canonical_name in _UNDERSCORE_ALIASES.items():
        canonical = canonical_by_name.get(canonical_name)
        if canonical:
            specs.append(
                {
                    **canonical,
                    "name": alias_name,
                    "description": f"[Deprecated: use {canonical_name}] {canonical['description']}",
                }
            )

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

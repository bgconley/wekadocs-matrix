# =============================================================================
# @status: ACTIVE
# @called-by: main.py
# =============================================================================
"""
Shared MCP server factory + tool registration for STDIO and HTTP transports.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import AsyncIterator
from typing import Any, Optional

import mcp.types as types
from mcp.server.lowlevel.server import Server

from src.mcp_server.mcp_tools import (
    DIAGNOSTICS_RESOURCE_TEMPLATE,
    PROMPT_DEFINITIONS,
    _tool_specs,
)
from src.mcp_server.mcp_utils import (
    _DIAGNOSTIC_EMITTER,
    DIAGNOSTICS_RESOURCES_ENABLED,
    LEGACY_SEARCH_DOCUMENTATION_ENABLED,
    MCP_TOOL_PROFILE,
    RETRIEVAL_PLAYBOOK_URI,
    SCRATCH_RESOURCE_TEMPLATE,
    Deps,
    _error_payload,
    _get_deps,
    _neo4j_disabled,
    _parse_diagnostics_uri,
    _parse_scratch_uri,
    _read_retrieval_playbook,
    _resolve_session_id,
    _retrieval_playbook_path,
)
from src.mcp_server.retrieval_trace import (
    append_followup_and_write,
)
from src.shared.observability import get_logger

logger = get_logger(__name__)


async def lifespan(server: Server) -> AsyncIterator[Deps]:
    """
    Lifespan context manager for dependency injection.

    Returns immediately with empty Deps to ensure fast MCP handshake.
    Actual initialization is deferred to first tool call via ensure_initialized().
    This prevents Claude Desktop from timing out during the initialize sequence.
    """
    deps = Deps()
    logger.info("STDIO server lifespan: returning immediately (lazy init enabled)")

    try:
        yield deps
    finally:
        # Cleanup connections if needed
        if deps.query:
            close = getattr(deps.query, "close", None)
            if callable(close):
                await close()
                logger.info("STDIO server lifespan: connections closed")


# ── MCP instructions (profile-aware) ──────────────────────────────────

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


def _summary_for_tool(name: str, result: dict) -> str:
    if isinstance(result, dict) and "error" in result:
        message = result.get("error", {}).get("message", "error")
        return f"{name} error: {message}"
    if "results" in result and isinstance(result["results"], list):
        results = result["results"]
        count = len(results)
        # For kb.search, explicitly list passage_ids to help the AI use the correct IDs
        if name in {"kb.search", "kb_search"} and count > 0:
            passage_ids = [r.get("passage_id") for r in results if r.get("passage_id")]
            if passage_ids:
                ids_str = ", ".join(passage_ids[:5])
                return (
                    f"{name} returned {count} results. "
                    f"Use these passage_ids with kb.read_excerpt: [{ids_str}]"
                )
        # For kb.search_sections, explicitly list section_ids
        if name in {"kb.search_sections", "search_sections"} and count > 0:
            section_ids = [r.get("section_id") for r in results if r.get("section_id")]
            if section_ids:
                ids_str = ", ".join(section_ids[:3])
                return (
                    f"{name} returned {count} results. "
                    f"Use these section_ids with kb.get_section_text: [{ids_str[:150]}...]"
                )
        return f"{name} returned {count} results."
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


# ── Tool profiles ─────────────────────────────────────────────────────
# Config-driven via MCP_TOOL_PROFILE env var.  Default: "production"
# (minimal 3-tool surface for LLM QA clients).
TOOL_PROFILES: dict[str, Optional[set[str]]] = {
    "production": {
        "kb.retrieve_evidence",
        "kb.read_excerpt",
        "graph.expand",
    },
    "analyst": {
        # All kb.* tools
        "kb.search",
        "kb.read_excerpt",
        "kb.expand_excerpt",
        "kb.extract_evidence",
        "kb.retrieve_evidence",
        "kb.search_sections",
        "kb.get_section_text",
        # All graph.* tools
        "graph.describe",
        "graph.expand",
        "graph.paths",
        "graph.parents",
        "graph.children",
        "graph.entities_for_sections",
        "graph.sections_for_entities",
        "graph.traverse",
        "graph.summarize",
        "graph.context_bundle",
    },
    "full": None,  # No filtering — all tools including backward aliases
}


def build_mcp_server() -> Server:
    server = Server("wekadocs", instructions=_instructions, lifespan=lifespan)
    all_specs = _tool_specs()

    # Apply tool profile filtering
    profile_name = MCP_TOOL_PROFILE
    allowed = TOOL_PROFILES.get(profile_name)
    if allowed is None and profile_name not in TOOL_PROFILES:
        logger.warning(
            "unknown_tool_profile",
            profile=profile_name,
            fallback="production",
        )
        allowed = TOOL_PROFILES["production"]

    if allowed is not None:
        tool_specs = [s for s in all_specs if s["name"] in allowed]
    else:
        tool_specs = all_specs

    logger.info(
        "mcp_tool_profile_applied",
        profile=profile_name,
        tools_available=len(tool_specs),
        tool_names=[s["name"] for s in tool_specs],
    )

    # Allow calling tools by their canonical/alias name even if not listed
    # (the call_tool handler uses the full map for backward compat)
    full_tool_map = {spec["name"]: spec["handler"] for spec in all_specs}

    @server.list_tools()
    async def _list_tools():
        tools: list[types.Tool] = []
        for spec in tool_specs:
            tools.append(
                types.Tool(
                    name=spec["name"],
                    description=spec["description"],
                    inputSchema=spec["input_schema"],
                    # outputSchema removed - Claude Desktop doesn't support it
                    annotations=spec.get("annotations"),
                )
            )
        return tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None):
        # Use full_tool_map so backward-compat aliases work even when
        # not listed (profile filtering only affects list_tools, not call_tool)
        handler = full_tool_map.get(name)
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

        # Trace follow-up: record non-evidence tool calls on the active trace.
        # Try multiple session ID sources to handle the edge case where
        # kb.retrieve_evidence used an explicit session_id but the follow-up omits it.
        if name not in {"kb.retrieve_evidence", "kb_retrieve_evidence"}:
            from src.mcp_server.retrieval_trace import get_active_trace

            session_id = (arguments or {}).get("session_id") or _resolve_session_id(
                None, None
            )
            # Also try the result's session_id (set by _finalize_payload)
            if isinstance(result, dict) and not get_active_trace(session_id):
                result_session = result.get("session_id")
                if result_session and get_active_trace(result_session):
                    session_id = result_session
            append_followup_and_write(
                session_id=session_id,
                tool_name=name,
                arguments_summary=json.dumps(arguments or {}, default=str)[:200],
                result_summary=summary[:200],
            )

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

"""
STDIO MCP Server for Claude Desktop
Implements Phase 1 STDIO transport using FastMCP SDK with lifespan pattern.
See: /docs/spec.md §9 (Interfaces)
See: mcp-compatibility-enhanced-gpt5pro analysis in neo4j memory
See: comprehensive-mcp-stdio-phase1-fix-20251020 for STDIO safety details

CRITICAL: This file uses STDIO transport where stdout is sacred (JSON-RPC only).
ALL logs must go to stderr. The bootstrap below ensures this.
"""

import builtins
import logging

# --- STDIO-SAFE BOOTSTRAP: Must be at the very top ---
import os
import sys
import warnings

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
from typing import Optional  # noqa: E402

from mcp.server.fastmcp import Context, FastMCP  # noqa: E402
from mcp.server.session import ServerSession  # noqa: E402

from src.mcp_server.query_service import QueryService, get_query_service  # noqa: E402
from src.query.traversal import TraversalService  # noqa: E402
from src.shared.connections import get_connection_manager  # noqa: E402
from src.shared.observability import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class Deps:
    """Dependencies shared across MCP tool calls via lifespan context."""

    query: Optional[QueryService] = None


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
        logger.info("STDIO server lifespan: QueryService ready")
        yield deps
    finally:
        # Cleanup connections if needed
        if deps.query:
            close = getattr(deps.query, "close", None)
            if callable(close):
                await close()
                logger.info("STDIO server lifespan: connections closed")


# Create FastMCP instance with lifespan
mcp = FastMCP("wekadocs", lifespan=lifespan)


@mcp.tool()
async def search_documentation(
    query: str,
    top_k: int = 20,
    verbosity: str = "snippet",
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    Search documentation using hybrid retrieval (vector + graph).

    Args:
        query: Natural language search query
        top_k: Maximum number of results to return (default: 20)
        verbosity: Response detail level - "snippet" (200 chars, fast), "full" (complete text), or "graph" (text + relationships) (default: "snippet")
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

        # Execute search via existing Phase 2 pipeline
        response = query_service.search(
            query=query,
            top_k=top_k,
            expand_graph=True,
            find_paths=False,
            verbosity=verbosity,
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


@mcp.tool()
async def traverse_relationships(
    start_ids: list[str],
    rel_types: list[str] | None = None,
    max_depth: int = 2,
    include_text: bool = True,
    ctx: Context[ServerSession, Deps] | None = None,
) -> dict:
    """
    Traverse graph relationships from given nodes for multi-turn exploration.

    Args:
        start_ids: Starting section/entity IDs to traverse from
        rel_types: Relationship types to follow (default: ["MENTIONS", "CONTAINS_STEP", "REQUIRES", "AFFECTS"])
        max_depth: Maximum traversal depth, 1-3 hops (default: 2)
        include_text: Include full text of nodes in results (default: True)
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


if __name__ == "__main__":
    # Run MCP server with STDIO transport
    # Claude Desktop will launch this via: docker exec -i weka-mcp-server python -m src.mcp_server.stdio_server
    logger.info("Starting wekadocs MCP server with STDIO transport")
    mcp.run(transport="stdio")

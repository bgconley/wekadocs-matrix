"""
Service layer modules for the agentic retrieval stack.

Each service encapsulates a narrowly scoped concern (graph, text, budgeting,
delta caching, etc.) so that MCP handlers can remain thin wrappers that enforce
schemas and budgeting.  See docs/cdx-outputs/retrieval_fix.json for the
approved architecture.
"""

# Re-export CrossDocLinkingConfig from config for convenience
from src.shared.config import CrossDocLinkingConfig  # noqa: F401

from .context_budget_manager import BudgetExceeded, ContextBudgetManager  # noqa: F401
from .cross_doc_linking import CrossDocLinker, LinkingResult  # noqa: F401
from .delta_cache import SessionDeltaCache  # noqa: F401
from .graph_service import GraphService  # noqa: F401
from .text_service import TextService  # noqa: F401

__all__ = [
    "BudgetExceeded",
    "ContextBudgetManager",
    "CrossDocLinker",
    "CrossDocLinkingConfig",
    "GraphService",
    "LinkingResult",
    "SessionDeltaCache",
    "TextService",
]

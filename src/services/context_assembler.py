"""
ContextAssembler and SummarizationService for MCP v2 tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.services.context_budget_manager import ContextBudgetManager
from src.services.graph_service import GraphService
from src.services.text_service import TextService


class SummarizationService:
    def __init__(self, graph: GraphService) -> None:
        self.graph = graph

    def summarize_neighborhood(
        self,
        node_ids: Sequence[str],
        *,
        token_budget: int = 400,
    ) -> Dict[str, List[str]]:
        budget = ContextBudgetManager(
            token_budget=token_budget,
            byte_budget=token_budget * 4,
        )
        graph_result = self.graph.describe_nodes(
            node_ids=node_ids, budget=budget, phase="neighbors"
        )
        nodes = graph_result.payload.get("results", [])
        bullets = []
        citations = []
        for node in nodes[:5]:
            title = node.get("title") or node.get("id")
            label = node.get("label") or "Node"
            tokens = node.get("tokens") or 0
            bullets.append(f"{title} — {label} (≈{tokens} tokens)")
            citations.append(node.get("id"))
            if len(bullets) >= 5:
                break
        return {"bullets": bullets, "citations": citations}


class ContextAssemblerService:
    def __init__(self, graph: GraphService, text: TextService) -> None:
        self.graph = graph
        self.text = text

    def compute_context_bundle(
        self,
        seeds: Sequence[str],
        *,
        strategy: str = "hybrid",
        token_budget: int = 2_000,
    ) -> Dict[str, Any]:
        budget = ContextBudgetManager(
            token_budget=token_budget,
            byte_budget=token_budget * 4,
        )
        graph = self.graph.describe_nodes(seeds, budget=budget, phase="seeds")
        text_result = self.text.get_section_text(
            seeds, max_bytes_per=16_384, budget=budget
        )
        bundle = {
            "nodes": graph.payload.get("results", []),
            "snippets": text_result.results,
            "strategy": strategy,
        }
        usage = {
            "tokens": budget.usage["tokens"],
            "bytes": budget.usage["bytes"],
        }
        return {"bundle": bundle, "usage": usage}

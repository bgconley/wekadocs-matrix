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
    def __init__(
        self,
        *,
        search_candidates: SearchCandidatesFn,
        extract_quotes: ExtractQuotesFn,
        expand_with_graph: Optional[ExpandGraphFn],
        live_validation_available: bool,
        enhancer: Any = None,
    ) -> None:
        self._search_candidates = search_candidates
        self._extract_quotes = extract_quotes
        self._expand_with_graph = expand_with_graph
        self._live_validation_available = live_validation_available
        self._enhancer = enhancer

    async def build_package(
        self, *, request: EvidenceRequest, deps: Any
    ) -> EvidencePackage:
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
            graph_seed_count = min(10, len(section_ids))
            graph_passage_ids = await self._expand_with_graph(
                section_ids=section_ids,
                deps=deps,
                effective_session=request.session_id,
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
            search_results=results,
            quotes=quotes,
            metrics=metrics,
            graph_expansion_applied=graph_expansion_applied,
        )
        gaps = identify_gaps(
            quotes=quotes,
            documents_searched=coverage.documents_searched,
            live_validation_available=self._live_validation_available,
        )

        appendix = []
        for result in results[:20]:
            pid = result.get("passage_id")
            entry = await deps.scratch.get(request.session_id, pid) if pid else None
            appendix.append(
                {
                    "chunk_id": result.get("section_id", ""),
                    "rerank_score": result.get("score"),
                    "doc_tag": result.get("doc_tag"),
                    "heading": result.get("title", ""),
                    "parent_path_norm": (entry or {}).get("parent_path_norm"),
                    "text": (entry or {}).get("text", ""),
                }
            )
        metrics["_appendix_chunks"] = appendix
        metrics["_result_snapshot"] = results[:20]
        metrics["_result_count"] = len(results)
        metrics["_graph_expansion_applied"] = graph_expansion_applied
        metrics["_graph_seed_count"] = graph_seed_count
        metrics["_graph_neighbors_added"] = graph_neighbors_added

        return EvidencePackage(
            request=request,
            normalized_query=metrics.get("query_rewrite_result") or request.question,
            quotes=quotes,
            coverage=coverage,
            gaps=gaps,
            retrieval_metrics=metrics,
            diagnostic_context=diagnostic_context,
        )

    async def _safe_enhance(self, package: EvidencePackage) -> EvidencePackage:
        return package

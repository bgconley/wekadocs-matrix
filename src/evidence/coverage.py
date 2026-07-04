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
    )


def identify_gaps(
    *,
    quotes: Iterable[EvidenceQuote],
    documents_searched: int,
    live_validation_available: bool,
) -> List[EvidenceGap]:
    quote_list = list(quotes)
    gaps: List[EvidenceGap] = []
    if not quote_list:
        gaps.append(
            EvidenceGap(
                kind="insufficient_evidence",
                message=(
                    "No evidence quotes were selected from "
                    f"{documents_searched} searched documents."
                ),
                severity="warning",
            )
        )
    if not live_validation_available:
        gaps.append(
            EvidenceGap(
                kind="missing_live_validation",
                message=(
                    "Live embedding, reranker, Qdrant, and Neo4j validation "
                    "were not run for this package."
                ),
                severity="info",
            )
        )
    return gaps


def mark_budget_state(
    coverage: EvidenceCoverage,
    gaps: List[EvidenceGap],
    *,
    partial: bool,
    limit_reason: str,
) -> None:
    coverage.partial = partial
    coverage.limit_reason = limit_reason if partial else "none"
    if partial:
        gaps.append(
            EvidenceGap(
                kind="budget_exceeded",
                message=f"Response exceeds the configured output budget ({limit_reason}).",
                severity="warning",
            )
        )

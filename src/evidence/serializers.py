from __future__ import annotations

from typing import Any, Dict

from src.evidence.models import EvidencePackage


def evidence_package_to_mcp_payload(package: EvidencePackage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "package_id": package.package_id,
        "normalized_query": package.normalized_query,
        "quotes": [
            {
                "quote_id": quote.quote_id,
                "rank": quote.rank,
                "passage_id": quote.passage_id,
                "section_id": quote.section_id,
                "doc_tag": quote.doc_tag,
                "title": quote.title,
                "parent_path": " > ".join(quote.parent_path),
                "uri": quote.source_uri,
                "quote": quote.text,
                "confidence": quote.confidence,
                "score": quote.score,
                "source": quote.source,
                "retrieval_signals": quote.retrieval_signals,
            }
            for quote in package.quotes
        ],
        "coverage": package.coverage.model_dump(),
        "gaps": [gap.model_dump() for gap in package.gaps],
    }
    if package.answer_draft is not None:
        payload["answer_draft"] = package.answer_draft.model_dump()
    if package.trace_id is not None:
        payload["trace_id"] = package.trace_id
    if package.diagnostic_id is not None:
        payload["diagnostic_id"] = package.diagnostic_id
    if package.diagnostic_uri is not None:
        payload["diagnostic_uri"] = package.diagnostic_uri
    return payload

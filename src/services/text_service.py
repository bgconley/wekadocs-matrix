"""
TextService fetches section text with strict truncation and byte accounting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from neo4j import Driver

from src.services.context_budget_manager import BudgetExceeded, ContextBudgetManager


@dataclass
class TextFetchResult:
    results: List[Dict[str, Any]]
    partial: bool
    limit_reason: str
    tokens_estimate: int
    bytes_estimate: int


class TextService:
    def __init__(
        self,
        driver: Driver,
        *,
        default_max_bytes: int = 16_384,
    ) -> None:
        self.driver = driver
        self.default_max_bytes = default_max_bytes

    def get_section_text(
        self,
        section_ids: Sequence[str],
        *,
        max_bytes_per: Optional[int] = None,
        budget: Optional[ContextBudgetManager] = None,
    ) -> TextFetchResult:
        if not section_ids:
            return TextFetchResult(
                results=[],
                partial=False,
                limit_reason="none",
                tokens_estimate=0,
                bytes_estimate=0,
            )

        query = """
        MATCH (s:Section)
        WHERE s.id IN $section_ids
        RETURN s.id AS id,
               coalesce(s.title, s.name, s.heading, '') AS title,
               s.text AS text,
               s.doc_tag AS doc_tag,
               s.blob_id AS blob_id
        """
        with self.driver.session() as session:
            rows = list(session.run(query, section_ids=list(section_ids)))

        max_bytes = max_bytes_per or self.default_max_bytes
        results: List[Dict[str, Any]] = []
        for row in rows:
            text_value = row.get("text") or ""
            encoded = text_value.encode("utf-8")
            truncated = False
            if len(encoded) > max_bytes:
                truncated = True
                encoded = encoded[:max_bytes]
                text_value = encoded.decode("utf-8", errors="ignore")
            bytes_len = len(encoded)
            results.append(
                {
                    "section_id": row.get("id"),
                    "title": row.get("title"),
                    "doc_tag": row.get("doc_tag"),
                    "text_truncated": text_value,
                    "bytes": bytes_len,
                    "truncated": truncated,
                    "source_blob_id": row.get("blob_id"),
                }
            )

        body = json.dumps({"results": results})
        bytes_estimate = len(body.encode("utf-8"))
        tokens_estimate = (
            budget.estimate_tokens(body) if budget else max(1, len(body) // 4)
        )
        partial = False
        limit_reason = "none"
        if budget:
            try:
                budget.consume(tokens_estimate, bytes_estimate, "snippets")
            except BudgetExceeded as exc:
                partial = True
                limit_reason = exc.limit_reason
                tokens_estimate = exc.usage.get("tokens", tokens_estimate)
                bytes_estimate = exc.usage.get("bytes", bytes_estimate)

        return TextFetchResult(
            results=results,
            partial=partial,
            limit_reason=limit_reason,
            tokens_estimate=tokens_estimate,
            bytes_estimate=bytes_estimate,
        )

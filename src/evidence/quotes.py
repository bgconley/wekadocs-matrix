from __future__ import annotations

from typing import Any, Iterable, List

from src.evidence.models import EvidenceQuote


def _split_parent_path(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(">") if part.strip()]
    return []


def normalize_quote_payloads(
    raw_quotes: Iterable[dict[str, Any]],
) -> List[EvidenceQuote]:
    quotes: List[EvidenceQuote] = []
    for raw in raw_quotes:
        text = str(raw.get("quote") or raw.get("text") or "").strip()
        if not text:
            continue

        score = raw.get("score")
        signals = dict(raw.get("retrieval_signals") or {})
        if score is not None and "score" not in signals:
            signals["score"] = score

        quotes.append(
            EvidenceQuote(
                quote_id=f"q_{len(quotes) + 1:04d}",
                rank=int(raw.get("rank") or len(quotes) + 1),
                passage_id=str(raw.get("passage_id") or raw.get("section_id") or ""),
                section_id=raw.get("section_id"),
                doc_tag=raw.get("doc_tag"),
                title=raw.get("title"),
                parent_path=_split_parent_path(raw.get("parent_path")),
                source_uri=raw.get("source_uri") or raw.get("uri"),
                text=text,
                confidence=float(raw.get("confidence") or 0.0),
                score=score,
                source=str(raw.get("source") or "retrieval"),
                retrieval_signals=signals,
                context_before=raw.get("context_before"),
                context_after=raw.get("context_after"),
            )
        )
    return quotes

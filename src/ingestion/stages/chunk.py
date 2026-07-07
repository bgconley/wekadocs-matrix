"""Chunk assembly stage for atomic ingestion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def assemble_chunks(
    document: Dict[str, Any],
    sections: List[Dict[str, Any]],
    config,
    *,
    assembler: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Assemble parser sections into ingestion chunks."""
    if assembler is None:
        from src.ingestion.chunk_assembler import get_chunk_assembler

        assembler = get_chunk_assembler(
            getattr(getattr(config, "ingestion", None), "chunk_assembly", None)
        )

    assembled = assembler.assemble(document["id"], sections)

    doc_total_tokens = sum(int(section.get("token_count", 0)) for section in assembled)
    document["total_tokens"] = doc_total_tokens
    document.setdefault("doc_id", document.get("id"))

    for section in assembled:
        section.setdefault("document_id", document["id"])
        section.setdefault("doc_id", document.get("doc_id"))
        section["document_total_tokens"] = doc_total_tokens

    return assembled

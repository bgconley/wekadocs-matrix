import json
from typing import List

import pytest

from src.ingestion.chunk_assembler import GreedyCombinerV2
from src.providers.tokenizer_service import TokenizerService
from src.query.context_assembly import ContextAssembler
from src.query.hybrid_retrieval import ChunkResult


@pytest.fixture()
def tuned_chunking_env(monkeypatch):
    """Force small token thresholds so tests run quickly while exercising split logic."""
    monkeypatch.setenv("EMBED_TARGET_TOKENS", "200")
    monkeypatch.setenv("EMBED_MAX_TOKENS", "256")
    monkeypatch.setenv("EMBED_OVERLAP_TOKENS", "32")
    monkeypatch.setenv("SPLIT_MIN_TOKENS", "50")
    monkeypatch.setenv("COMBINE_TARGET_TOKENS", "200")
    monkeypatch.setenv("COMBINE_MAX_TOKENS", "256")
    monkeypatch.setenv("COMBINE_MIN_TOKENS", "150")
    monkeypatch.setenv("COMBINE_DOC_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("COMBINE_DOC_FALLBACK_DOC_TOKEN_MAX", "150")
    yield


def _build_large_section(tokenizer: TokenizerService, target_tokens: int) -> str:
    """Create a single-section body that safely exceeds the requested token target."""
    body_tokens: List[str] = []
    token_budget = target_tokens + 150  # overshoot to guarantee a split
    while tokenizer.count_tokens(" ".join(body_tokens) or "") < token_budget:
        body_tokens.extend([f"token{len(body_tokens) % 997}"] * 10)
    return "Massive Single Section\n\n" + " ".join(body_tokens)


def _reassemble_text(chunks, tokenizer: TokenizerService) -> str:
    """Reconstruct original text using overlap information and the shared tokenizer."""
    token_ids: List[int] = []
    overlap = tokenizer.overlap_tokens
    backend = tokenizer.backend
    for index, chunk in enumerate(sorted(chunks, key=lambda c: c["order"])):
        ids = backend.encode(chunk["text"])
        if index > 0:
            ids = ids[overlap:]
        token_ids.extend(ids)
    return backend.decode(token_ids)


def test_single_section_chunking_splits_and_roundtrips(tuned_chunking_env):
    tokenizer = TokenizerService()
    section_text = _build_large_section(tokenizer, tokenizer.target_tokens * 2)
    section = {
        "id": "sec-single",
        "level": 2,
        "order": 100,
        "title": "Massive Single Section",
        "text": section_text,
        "anchor": "sec-single",
        "doc_tag": "DOC-IMP-001",
    }
    assembler = GreedyCombinerV2()

    chunks = assembler.assemble("doc-large-001", [section])

    # Split must have occurred and every chunk must respect provider bounds.
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["is_split"] is True
        assert chunk["token_count"] <= assembler.hard_max
        # Split metadata should reference the seed section deterministically.
        assert chunk["original_section_ids"][0] == section["id"]
        assert len(chunk["original_section_ids"]) == 2
        payload = json.loads(chunk["boundaries_json"])
        assert payload["total_chunks"] == len(chunks)
        assert payload["chunk_index"] < len(chunks)

    # Rebuild the original body using overlap semantics.
    rebuilt = _reassemble_text(chunks, tokenizer)
    rebuilt_body = rebuilt.split("\n\n", 1)[-1].strip()
    original_body = section_text.split("\n\n", 1)[-1].strip()
    assert original_body in rebuilt_body


def test_split_chunks_flow_through_context_assembler(tuned_chunking_env):
    tokenizer = TokenizerService()
    section_text = _build_large_section(tokenizer, tokenizer.target_tokens * 2)
    section = {
        "id": "sec-single",
        "level": 2,
        "order": 100,
        "title": "Massive Single Section",
        "text": section_text,
        "anchor": "sec-single",
        "doc_tag": "DOC-IMP-001",
    }
    assembler = GreedyCombinerV2()
    chunk_dicts = assembler.assemble("doc-large-001", [section])

    chunk_results: List[ChunkResult] = []
    for data in chunk_dicts:
        chunk_results.append(
            ChunkResult(
                chunk_id=data["id"],
                document_id=data["document_id"],
                parent_section_id=data["parent_section_id"],
                order=data["order"],
                level=data["level"],
                heading=data["heading"],
                text=data["text"],
                token_count=data["token_count"],
                is_combined=data["is_combined"],
                is_split=data["is_split"],
                original_section_ids=data["original_section_ids"],
                boundaries_json=data["boundaries_json"],
                doc_tag=data.get("doc_tag"),
            )
        )

    assembler_ctx = ContextAssembler(tokenizer=TokenizerService())
    assembled = assembler_ctx.assemble(chunk_results)

    # Ordering and count should be preserved for downstream retrieval.
    assert len(assembled.chunks) == len(chunk_results)
    assert [c.order for c in assembled.chunks] == sorted(
        [c.order for c in chunk_results]
    )

    # Citation formatting should fall back once per chunk when explicit labels are absent.
    formatted = assembler_ctx.format_with_citations(assembled)
    citation_lines = [line for line in formatted.splitlines() if line.startswith("[")]
    assert len(citation_lines) == len(chunk_results)


def test_small_doc_collapses_to_single_chunk(tuned_chunking_env):
    section_a = {
        "id": "sec-a",
        "level": 2,
        "order": 10,
        "title": "Overview",
        "text": "word " * 40,
        "anchor": "sec-a",
    }
    section_b = {
        "id": "sec-b",
        "level": 2,
        "order": 20,
        "title": "Details",
        "text": "word " * 30,
        "anchor": "sec-b",
    }
    assembler = GreedyCombinerV2()
    chunks = assembler.assemble("doc-small-001", [section_a, section_b])
    assert len(chunks) == 1
    chunk = chunks[0]
    boundaries = json.loads(chunk["boundaries_json"])
    assert len(boundaries.get("sections", [])) == 2
    assert chunk["token_count"] <= assembler.hard_max

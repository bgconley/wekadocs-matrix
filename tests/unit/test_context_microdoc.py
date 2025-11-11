import pytest

from src.providers.tokenizer_service import TokenizerService
from src.query.context_assembly import ContextAssembler
from src.query.hybrid_retrieval import ChunkResult


@pytest.fixture()
def tokenizer():
    return TokenizerService()


def test_context_assembler_renders_microdoc_extra(tokenizer):
    assembler = ContextAssembler(tokenizer=tokenizer)
    micro_chunk = ChunkResult(
        chunk_id="micro-1",
        document_id="doc-related",
        parent_section_id="micro-1",
        order=1,
        level=2,
        heading="Quick Start",
        text="word " * 40,
        token_count=tokenizer.count_tokens("word " * 40),
        is_microdoc_extra=True,
    )

    context = assembler.assemble([micro_chunk])
    assert "Related:" in context.text

    citations = assembler.format_with_citations(context)
    assert "Related:" in citations

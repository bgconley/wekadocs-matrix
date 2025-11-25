from src.query.context_assembly import ContextAssembler
from src.query.hybrid_retrieval import ChunkResult


class RecordingTokenizer:
    supports_decode = False

    def __init__(self):
        self.calls = []

    def count_tokens(self, text: str) -> int:
        self.calls.append(text)
        return max(1, len(text.split()))


def test_partial_group_uses_cached_chunk_token_counts():
    tokenizer = RecordingTokenizer()
    assembler = ContextAssembler(tokenizer=tokenizer)

    chunk = ChunkResult(
        chunk_id="c1",
        document_id="doc1",
        parent_section_id="parent",
        order=0,
        level=1,
        heading="",
        text="LONG TEXT THAT SHOULD NOT BE TOKENIZED AGAIN",
        token_count=120,
    )

    text, included = assembler._fit_partial_group(
        [chunk],
        budget=500,
        current_document=None,
        current_parent=None,
    )

    assert included == [chunk]
    assert "LONG TEXT THAT SHOULD NOT BE TOKENIZED AGAIN" not in tokenizer.calls
    assert text.strip().startswith("LONG TEXT")

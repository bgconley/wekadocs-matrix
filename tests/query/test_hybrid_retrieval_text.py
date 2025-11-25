from src.query.hybrid_retrieval import HybridRetriever


class NoDecodeTokenizer:
    supports_decode = False
    backend_name = "segmenter"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class DecodeCapableTokenizer(NoDecodeTokenizer):
    supports_decode = True

    def encode(self, text: str):
        return text.split()

    def decode_tokens(self, tokens):
        return " ".join(tokens)


def _bootstrap_retriever_with_tokenizer(tokenizer):
    retriever = object.__new__(HybridRetriever)
    retriever.tokenizer = tokenizer
    return retriever


def test_truncate_text_without_decode_falls_back_to_chars():
    tokenizer = NoDecodeTokenizer()
    retriever = _bootstrap_retriever_with_tokenizer(tokenizer)

    truncated, used = retriever._truncate_text("one two three four", 2)

    assert used == 2
    assert len(truncated) < len("one two three four")


def test_truncate_text_with_decode_is_lossless():
    tokenizer = DecodeCapableTokenizer()
    retriever = _bootstrap_retriever_with_tokenizer(tokenizer)

    truncated, used = retriever._truncate_text("alpha beta gamma", 2)

    assert used == 2
    assert truncated == "alpha beta"

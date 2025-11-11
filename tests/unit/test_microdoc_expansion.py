from dataclasses import replace

from src.providers.tokenizer_service import TokenizerService
from src.query.hybrid_retrieval import ChunkResult, HybridRetriever


class DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *args, **kwargs):
        return []


class DummyDriver:
    def session(self):
        return DummySession()


class DummyVectorClient:
    def search(self, **kwargs):
        return []


class DummyEmbedder:
    def embed_documents(self, texts):
        return [[0.0 for _ in range(4)]]


class DummyVectorRetriever:
    def __init__(self):
        self.client = DummyVectorClient()
        self.embedder = DummyEmbedder()
        self.collection_name = "chunks"


def _bootstrap_hybrid_retriever() -> HybridRetriever:
    hr = object.__new__(HybridRetriever)
    hr.neo4j_driver = DummyDriver()
    hr.vector_retriever = DummyVectorRetriever()
    hr.tokenizer = TokenizerService()
    hr.micro_max_neighbors = 1
    hr.microdoc_enabled = True
    hr.micro_min_tokens = 600
    hr.micro_doc_max = 2000
    hr.micro_dir_depth = 2
    hr.micro_knn_limit = 5
    hr.micro_sim_threshold = 0.5
    hr.micro_per_doc_budget = 250
    hr.micro_total_budget = 500
    return hr


def test_microdoc_expansion_marks_extras():
    hr = _bootstrap_hybrid_retriever()

    base = ChunkResult(
        chunk_id="seed",
        document_id="doc-base",
        parent_section_id="seed",
        order=1,
        level=2,
        heading="Base Doc",
        text="word " * 80,
        token_count=80,
        fused_score=1.0,
        document_total_tokens=320,
        source_path="/guides/base",
    )

    fused_extra = replace(
        base,
        chunk_id="extra",
        document_id="doc-extra",
        text="word " * 70,
        token_count=70,
        fused_score=0.8,
        document_total_tokens=300,
        source_path="/guides/extra",
    )

    hr._microdoc_from_fused = lambda base_chunk, pool, used, needed: [
        replace(fused_extra)
    ]
    hr._microdoc_from_directory = lambda base_chunk, used, needed: []
    hr._microdoc_from_knn = lambda base_chunk, used, needed, filters: []

    extras, tokens = hr._expand_microdoc_results(
        "query", [base, fused_extra], [base], {}
    )

    assert len(extras) == 1
    extra = extras[0]
    assert extra.is_microdoc_extra is True
    assert extra.document_id == "doc-extra"
    assert tokens <= hr.micro_total_budget

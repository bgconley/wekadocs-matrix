"""
Tests for bge_reranker mode: ensures BM25 is skipped, reranker drives ordering,
and confidence comes from rerank_score via sigmoid.
"""

import math
from types import SimpleNamespace

from src.query.hybrid_retrieval import HybridRetriever
from src.query.ranking import Ranker
from src.shared.config import get_config


class DummyRerankProvider:
    def __init__(self):
        self.model_id = "dummy-reranker"
        self.provider_name = "dummy"

    def rerank(self, query, candidates, top_k=10):
        # assign descending scores by original order
        results = []
        for i, cand in enumerate(candidates):
            results.append(
                {
                    **cand,
                    "rerank_score": float(len(candidates) - i),
                    "original_rank": i + 1,
                    "reranker": self.model_id,
                }
            )
        return results[:top_k]


def test_bge_reranker_mode_skips_bm25(monkeypatch):
    config = get_config()
    # set mode and toggles; these fields now exist on the model
    config.search.hybrid.mode = "bge_reranker"
    config.search.hybrid.reranker.enabled = True
    config.search.hybrid.reranker.top_n = 3
    config.search.hybrid.bm25.enabled = False
    # top-level bm25 config is still used in HybridRetriever init
    config.search.bm25.enabled = False

    # stub vector retriever to return fake chunks
    fake_chunk = SimpleNamespace(
        chunk_id="c1",
        document_id="d1",
        parent_section_id="",
        order=0,
        level=2,
        heading="h",
        text="t",
        token_count=10,
        is_combined=False,
        is_split=False,
        original_section_ids=[],
        boundaries_json="{}",
        doc_tag=None,
        snapshot_scope=None,
        document_total_tokens=0,
        source_path=None,
        bm25_rank=None,
        bm25_score=None,
        vector_rank=None,
        vector_score=0.5,
        fused_score=0.5,
        vector_score_kind="late-interaction",
        fusion_method=None,
        rerank_score=None,
        rerank_rank=None,
        rerank_original_rank=None,
        reranker=None,
        graph_distance=0,
        graph_score=0.0,
        graph_path=None,
        connection_count=0,
        mention_count=0,
        citation_labels=[],
        is_microdoc=False,
        is_microdoc_stub=False,
        doc_is_microdoc=False,
        is_microdoc_extra=False,
    )

    dummy_vec_retriever = SimpleNamespace(
        search=lambda q, k, f: [fake_chunk],
        last_stats={"duration_ms": 1, "path": "query_api"},
        supports_sparse=True,
        supports_colbert=True,
        schema_supports_sparse=True,
        schema_supports_colbert=True,
        sparse_field_name="text-sparse",
        field_weights={"content": 1.0},
        use_query_api=True,
    )

    # monkeypatch HybridRetriever internals to avoid real deps
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.QdrantMultiVectorRetriever",
        lambda *a, **k: dummy_vec_retriever,
    )
    # prevent real BM25; return dummy with index_name and search
    dummy_bm25 = SimpleNamespace(index_name=None, search=lambda *a, **k: [])
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.BM25Retriever", lambda *a, **k: dummy_bm25
    )
    # skip schema check
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.ensure_schema_version", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.HybridRetriever._hydrate_missing_citations",
        lambda self, chunks: None,
    )
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.HybridRetriever._expand_microdoc_results",
        lambda self, query, fused_results, seeds, filters: ([], 0),
    )
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.HybridRetriever._apply_graph_enrichment",
        lambda self, seeds, results, doc_tag: (
            [],
            {"graph_time_ms": 0, "graph_count": 0},
        ),
    )
    monkeypatch.setattr(
        "src.query.hybrid_retrieval.HybridRetriever._annotate_coverage",
        lambda self, results: None,
    )

    hr = HybridRetriever(
        neo4j_driver=None,
        qdrant_client=None,
        embedder=None,
        tokenizer=None,
        embedding_settings=None,
    )

    # inject dummy reranker
    monkeypatch.setattr(hr, "_get_reranker", lambda: DummyRerankProvider())

    def fake_apply(query, seeds, metrics):
        for idx, c in enumerate(seeds):
            c.rerank_score = float(len(seeds) - idx)
            c.reranker = "dummy"
            c.rerank_rank = idx + 1
        metrics["reranker_applied"] = True
        return seeds

    monkeypatch.setattr(hr, "_apply_reranker", fake_apply)

    results, metrics = hr.retrieve("q", top_k=1, expand=False)

    assert metrics["bm25_count"] == 0
    assert len(results) == 1
    assert results[0].rerank_score == float(len([fake_chunk]))
    # confidence should be sigmoid(rerank_score)
    # Build a SearchResult to pass to Ranker
    from src.query.hybrid_search import SearchResult

    res = SearchResult(
        node_id=results[0].chunk_id,
        node_label="Chunk",
        score=results[0].rerank_score,
        distance=0,
        metadata={
            "rerank_score": results[0].rerank_score,
            "vector_score": results[0].vector_score,
            "vector_score_kind": results[0].vector_score_kind,
        },
    )
    ranker = Ranker()
    ranked = ranker.rank([res], {})
    features = ranked[0].features
    rerank_prob = 1.0 / (1.0 + math.exp(-results[0].rerank_score))
    expected_semantic = (
        ranker.semantic_recall_weight * 1.0
        + ranker.semantic_rerank_weight * rerank_prob
    )
    assert abs(features.semantic_score - expected_semantic) < 1e-6

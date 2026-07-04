from src.evidence.models import (
    EvidenceCoverage,
    EvidencePackage,
    EvidenceQuote,
    EvidenceRequest,
)
from src.mcp_server.retrieval_trace import RetrievalTraceBuilder


def _package():
    return EvidencePackage(
        request=EvidenceRequest(question="How does NC2 work?", session_id="s1"),
        normalized_query="How does NC2 work?",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nc2/aws",
                text="NC2 runs Nutanix Cloud Clusters.",
                confidence=0.83,
            )
        ],
        coverage=EvidenceCoverage(
            documents_searched=2,
            documents_with_evidence=1,
            retrieval_depth=10,
            reranker_applied=True,
            signal_pool_active=True,
            graph_expansion_applied=False,
        ),
        retrieval_metrics={
            "reranker_model": "qwen3-reranker",
            "reranker_input_count": 10,
            "signal_pool_used": True,
            "signal_pool_size": 30,
            "colbert_rerank_applied": True,
            "related_to_docs_found": 3,
            "related_to_chunks_added": 5,
            "_graph_seed_count": 4,
            "_graph_neighbors_added": 6,
            "snapshot_post_reranker": [{"chunk_id": "s1", "score": 0.9}],
            "_result_snapshot": [
                {"section_id": "s1", "title": "NC2", "score": 0.9, "rank": 1}
            ],
            "_result_count": 42,
            "_appendix_chunks": [
                {
                    "chunk_id": "s1",
                    "rerank_score": 0.9,
                    "heading": "NC2",
                    "text": "NC2 runs Nutanix Cloud Clusters.",
                    "doc_tag": "nc2/aws",
                    "parent_path_norm": None,
                }
            ],
        },
    )


def test_record_evidence_package_populates_all_facets():
    trace = RetrievalTraceBuilder(trace_id="trace-1", session_id="s1")

    trace.record_evidence_package(_package())

    rendered = trace.format()
    assert "NC2 runs Nutanix Cloud Clusters" in rendered
    assert "qwen3-reranker" in rendered
    assert trace._query is not None
    assert trace._signal_pool is not None
    assert trace._reranker is not None
    assert trace._reranker["output_count"] == 42
    assert trace._related_to is not None
    assert trace._graph_enrichment is not None
    assert trace._graph_enrichment["neighbors_added"] == 6
    assert trace._appendix_chunks
    assert trace._appendix_chunks[0]["rerank_score"] == 0.9
    assert trace._stage_snapshots is not None
    assert trace._colbert is not None
    assert trace._evidence_pack is not None

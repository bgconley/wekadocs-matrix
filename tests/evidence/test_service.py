import pytest

from src.evidence.models import EvidenceRequest
from src.evidence.service import EvidenceService


class FakeScratch:
    def __init__(self) -> None:
        self.entries = {
            "p1": {"text": "full text p1", "parent_path_norm": "NCI > AOS"},
            "p2": {"text": "full text p2", "parent_path_norm": "NCM > Ops"},
        }

    async def get(self, session_id, passage_id):
        return self.entries.get(passage_id)


class FakeDeps:
    scratch = FakeScratch()


async def fake_search_candidates(**kwargs):
    return (
        {
            "results": [
                {
                    "passage_id": "p1",
                    "section_id": "s1",
                    "doc_tag": "nci/aos",
                    "title": "AOS",
                    "score": 0.9,
                    "rank": 1,
                },
                {
                    "passage_id": "p2",
                    "section_id": "s2",
                    "doc_tag": "ncm/ops",
                    "title": "Ops",
                    "score": 0.7,
                    "rank": 2,
                },
            ],
            "metrics": {
                "reranker_applied": True,
                "signal_pool_used": True,
                "query_rewrite_result": "configure AOS",
            },
        },
        {"metrics": {"reranker_applied": True}, "chunks": []},
    )


async def fake_extract_quotes(**kwargs):
    return [
        {
            "rank": 1,
            "passage_id": "p1",
            "section_id": "s1",
            "doc_tag": "nci/aos",
            "parent_path": "NCI > AOS",
            "quote": "AOS provides distributed storage.",
            "confidence": 0.82,
            "score": 0.9,
            "source": "retrieval",
        }
    ]


@pytest.mark.asyncio
async def test_service_builds_package_with_telemetry_and_diag_context():
    service = EvidenceService(
        search_candidates=fake_search_candidates,
        extract_quotes=fake_extract_quotes,
        expand_with_graph=None,
        live_validation_available=False,
        enhancer=None,
    )

    package = await service.build_package(
        request=EvidenceRequest(
            question="How does AOS work?",
            session_id="s1",
            max_quotes=2,
            retrieval_depth=10,
        ),
        deps=FakeDeps(),
    )

    assert package.normalized_query == "configure AOS"
    assert package.quotes[0].quote_id == "q_0001"
    assert package.quotes[0].parent_path == ["NCI", "AOS"]
    assert package.coverage.documents_searched == 2
    assert package.coverage.documents_with_evidence == 1
    assert package.coverage.retrieval_depth == 2
    assert package.retrieval_metrics["reranker_applied"] is True
    assert package.retrieval_metrics["_result_count"] == 2
    assert package.retrieval_metrics["_result_snapshot"][0]["section_id"] == "s1"
    assert package.retrieval_metrics["_appendix_chunks"][0]["text"] == "full text p1"
    assert package.diagnostic_context is not None
    assert [gap.kind for gap in package.gaps] == ["missing_live_validation"]


@pytest.mark.asyncio
async def test_service_uses_preclamped_request_depth_verbatim():
    seen = {}

    async def capture_search(**kwargs):
        seen["top_k"] = kwargs["top_k"]
        seen["page_size"] = kwargs["page_size"]
        seen["_fetch_k_override"] = kwargs["_fetch_k_override"]
        return {"results": [], "metrics": {}}, {}

    service = EvidenceService(
        search_candidates=capture_search,
        extract_quotes=fake_extract_quotes,
        expand_with_graph=None,
        live_validation_available=False,
        enhancer=None,
    )

    await service.build_package(
        request=EvidenceRequest(
            question="q",
            session_id="s",
            max_quotes=3,
            retrieval_depth=150,
        ),
        deps=FakeDeps(),
    )

    assert seen == {"top_k": 150, "page_size": 150, "_fetch_k_override": 150}


@pytest.mark.asyncio
async def test_service_expands_graph_and_records_graph_telemetry():
    seen = {}

    async def capture_extract(**kwargs):
        seen["passage_ids"] = kwargs["passage_ids"]
        return []

    async def fake_graph(**kwargs):
        seen["section_ids"] = kwargs["section_ids"]
        return ["graph-p1", "graph-p2"]

    service = EvidenceService(
        search_candidates=fake_search_candidates,
        extract_quotes=capture_extract,
        expand_with_graph=fake_graph,
        live_validation_available=True,
        enhancer=None,
    )

    package = await service.build_package(
        request=EvidenceRequest(
            question="q",
            session_id="s1",
            max_quotes=2,
            retrieval_depth=10,
            graph_enrichment=True,
        ),
        deps=FakeDeps(),
    )

    assert seen["section_ids"] == ["s1", "s2"]
    assert seen["passage_ids"] == ["p1", "p2", "graph-p1", "graph-p2"]
    assert package.coverage.graph_expansion_applied is True
    assert package.retrieval_metrics["_graph_seed_count"] == 2
    assert package.retrieval_metrics["_graph_neighbors_added"] == 2

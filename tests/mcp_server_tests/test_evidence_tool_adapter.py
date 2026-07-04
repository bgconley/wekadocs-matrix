from src.evidence.models import (
    EvidenceCoverage,
    EvidencePackage,
    EvidenceQuote,
    EvidenceRequest,
)
from src.evidence.serializers import evidence_package_to_mcp_payload


def _package(**over):
    base = dict(
        request=EvidenceRequest(question="How do I configure NAI?", session_id="s1"),
        normalized_query="configure NAI",
        quotes=[
            EvidenceQuote(
                quote_id="q_0001",
                rank=1,
                passage_id="p1",
                doc_tag="nai/deploy",
                title="Deploy",
                parent_path=["NAI", "Deploy"],
                source_uri="nutanixdocs://scratch/s1/p1",
                text="Deploy models through NAI.",
                confidence=0.84,
            )
        ],
        coverage=EvidenceCoverage(
            documents_searched=4,
            documents_with_evidence=1,
            retrieval_depth=20,
            reranker_applied=True,
            signal_pool_active=False,
            graph_expansion_applied=False,
        ),
        trace_id="trace-1",
    )
    base.update(over)
    return EvidencePackage(**base)


def test_payload_is_additive_and_omits_null_fields():
    payload = evidence_package_to_mcp_payload(_package())

    assert payload["quotes"][0]["quote"] == "Deploy models through NAI."
    assert payload["quotes"][0]["parent_path"] == "NAI > Deploy"
    assert payload["quotes"][0]["title"] == "Deploy"
    assert payload["quotes"][0]["uri"] == "nutanixdocs://scratch/s1/p1"
    assert payload["coverage"]["documents_with_evidence"] == 1
    assert payload["trace_id"] == "trace-1"
    assert payload["package_id"].startswith("ep_")
    assert payload["normalized_query"] == "configure NAI"
    assert payload["gaps"] == []
    assert "answer_draft" not in payload
    assert "diagnostic_id" not in payload
    assert "diagnostic_uri" not in payload
    assert "retrieval_metrics" not in payload
    assert "diagnostic_context" not in payload


def test_payload_includes_diagnostics_and_draft_only_when_present():
    package = _package()
    package.diagnostic_id = "diag-1"
    package.diagnostic_uri = "nutanixdocs://diagnostics/2026-07-04/diag-1"

    payload = evidence_package_to_mcp_payload(package)

    assert payload["diagnostic_id"] == "diag-1"
    assert payload["diagnostic_uri"].endswith("diag-1")

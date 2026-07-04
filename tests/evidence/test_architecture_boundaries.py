from pathlib import Path


def test_evidence_core_never_imports_mcp_server():
    offenders = [
        str(path)
        for path in Path("src/evidence").rglob("*.py")
        if "src.mcp_server" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"evidence core must not depend on mcp_server: {offenders}"


def test_mcp_tools_depends_on_evidence_core():
    text = Path("src/mcp_server/mcp_tools.py").read_text(encoding="utf-8")
    assert "src.evidence" in text


def test_trace_semantics_live_in_retrieval_trace_not_mcp_tools():
    trace = Path("src/mcp_server/retrieval_trace.py").read_text(encoding="utf-8")
    assert "def record_evidence_package" in trace


def test_architecture_doc_records_ownership_and_deferred_live_boundary():
    doc = Path("docs/architecture/evidence-package-core.md").read_text(encoding="utf-8")
    for required in (
        "retrieval_metrics",
        "KB_EVIDENCE_MAX_FETCH_K",
        "budget_exceeded",
        "record_evidence_package",
        "uncited_draft_rejected",
        "Embedding, sparse (SPLADE), ColBERT, reranker, Qdrant, and Neo4j",
    ):
        assert required in doc

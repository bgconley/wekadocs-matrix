import pytest

from src.mcp_server import mcp_tools
from src.mcp_server.mcp_app import TOOL_PROFILES


def test_production_surface_is_evidence_first_and_stable():
    assert TOOL_PROFILES["production"] == {
        "kb.retrieve_evidence",
        "kb.read_excerpt",
        "graph.expand",
    }


def test_no_deprecated_alias_leaks_into_curated_profiles():
    # Underscore aliases carry "[Deprecated" and must never surface in
    # production/analyst list_tools. `full` is the only alias-bearing profile.
    specs = {s["name"]: s for s in mcp_tools._tool_specs()}
    for profile in ("production", "analyst"):
        for name in TOOL_PROFILES[profile]:
            assert name in specs, f"{name} missing from specs"
            assert "[Deprecated" not in specs[name].get("description", "")


@pytest.mark.asyncio
async def test_retrieve_evidence_current_shape(evidence_fake_ctx):
    # CHARACTERIZATION of the CURRENT tool (pre-refactor). This MUST pass against
    # today's code; Task 8 intentionally supersedes it with the enriched shape.
    payload = await mcp_tools.kb_retrieve_evidence(
        question="How do I configure authentication?",
        ctx=evidence_fake_ctx,
        session_id="pin-session",
    )

    assert {
        "quotes",
        "coverage",
        "session_id",
        "partial",
        "limit_reason",
        "meta",
        "trace_id",
    } <= payload.keys()
    assert "gaps" not in payload
    assert "package_id" not in payload
    assert "normalized_query" not in payload
    assert set(payload["coverage"].keys()) == {
        "documents_searched",
        "documents_with_evidence",
        "retrieval_depth",
        "reranker_applied",
        "signal_pool_active",
        "graph_expansion_applied",
    }
    assert payload[
        "quotes"
    ], "fixture must return >=1 quote or these asserts are vacuous"
    for quote in payload["quotes"]:
        assert {
            "quote",
            "passage_id",
            "doc_tag",
            "title",
            "uri",
            "parent_path",
            "confidence",
            "source",
            "rank",
        } <= quote.keys()
    assert "retrieval_metrics" not in payload
    assert "diagnostic_context" not in payload

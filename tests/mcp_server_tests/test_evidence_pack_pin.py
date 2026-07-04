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
async def test_retrieve_evidence_enriched_shape(evidence_fake_ctx):
    payload = await mcp_tools.kb_retrieve_evidence(
        question="How do I configure authentication?",
        ctx=evidence_fake_ctx,
        session_id="pin-session",
    )

    assert {
        "quotes",
        "coverage",
        "gaps",
        "package_id",
        "normalized_query",
        "trace_id",
        "session_id",
        "partial",
        "limit_reason",
        "meta",
    } <= payload.keys()
    assert [key for key, value in payload.items() if value is None] == []
    coverage = payload["coverage"]
    assert {"partial", "limit_reason"} <= coverage.keys()
    assert payload["partial"] == coverage["partial"]
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
            "quote_id",
        } <= quote.keys()
    assert "retrieval_metrics" not in payload
    assert "diagnostic_context" not in payload

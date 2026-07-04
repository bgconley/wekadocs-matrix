"""Nutanix-specific query reformulation behavior."""

from src.mcp_server.query_service import QueryService


def test_heuristic_reformulation_uses_nutanix_domain():
    rewritten = QueryService._heuristic_reformulate(
        "configure Prism Central authentication",
        ["configure", "prism", "central", "authentication"],
    )

    assert rewritten == "How do I configure configure Prism Central authentication in Nutanix?"
    assert "WEKA" not in rewritten
    assert "weka" not in rewritten

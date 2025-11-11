import pytest

from src.neo.explain_guard import ExplainGuard, PlanRejected


def make_guard():
    # Driver unused in the relationship check; pass None for simplicity.
    return ExplainGuard(driver=None)  # type: ignore[arg-type]


def test_relationship_whitelist_allows_schema_edges():
    guard = make_guard()
    metadata = {
        "details": [
            "Expand(All) | type: 'CONTAINS_STEP'",
            "Expand(All) | type: 'HAS_SECTION'",
        ]
    }

    # Should not raise for schema-supported relationships.
    guard._check_relationship_types(metadata)


def test_relationship_whitelist_rejects_unknown_edges():
    guard = make_guard()
    metadata = {"details": ["Expand(All) | type: 'FORBIDDEN_EDGE'"]}

    with pytest.raises(PlanRejected):
        guard._check_relationship_types(metadata)

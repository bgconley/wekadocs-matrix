import pytest


@pytest.mark.skip(
    reason="_upsert_to_qdrant removed from GraphBuilder in Phase 5; functionality moved to qdrant_writers.py"
)
def test_qdrant_namespace_enforced_on_upsert(monkeypatch):
    pass

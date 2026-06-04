from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="_upsert_to_qdrant removed from GraphBuilder in Phase 5; functionality moved to qdrant_writers.py"
)
def test_upsert_to_qdrant_attaches_sparse_vectors(monkeypatch):
    pass

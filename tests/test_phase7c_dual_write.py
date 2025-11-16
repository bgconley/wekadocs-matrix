"""
Phase 7C strict-mode guard tests.
Ensure GraphBuilder catches embedding drift instead of relying on the deprecated
dual-write migration scaffolding.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, List

import pytest

from src.ingestion.build_graph import GraphBuilder
from src.shared.config import get_config
from src.shared.observability.metrics import embedding_profile_guard_events_total


@dataclass
class _FakeSession(contextlib.AbstractContextManager):
    """Minimal Neo4j session stub that yields canned Section IDs."""

    ids: List[str]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *_args, **_kwargs) -> Iterable[dict]:
        return [{"id": section_id} for section_id in self.ids]


@dataclass
class _FakeDriver:
    """Provides sessions that return predetermined Section IDs."""

    ids: List[str]

    def session(self):
        return _FakeSession(self.ids)


@pytest.fixture(autouse=True)
def _reset_guard_metrics():
    """Clear embedding guard counters before/after each test."""
    embedding_profile_guard_events_total._metrics.clear()
    yield
    embedding_profile_guard_events_total._metrics.clear()


def _get_metric_value(profile: str, outcome: str) -> float:
    return embedding_profile_guard_events_total.labels(
        profile=profile, outcome=outcome
    )._value.get()


def test_profile_guard_clean_path_records_metric():
    """No drift → strict mode stays quiet but records a clean event."""
    config = get_config()
    builder = GraphBuilder(
        driver=_FakeDriver(ids=[]), config=config, qdrant_client=None, strict_mode=True
    )

    builder._enforce_profile_guard()

    assert (
        _get_metric_value(builder.embedding_settings.profile or "legacy", "clean") == 1
    )


def test_profile_guard_strict_mode_blocks_on_drift():
    """When drift exists and strict mode is disabled, ingestion must fail."""
    config = get_config()
    builder = GraphBuilder(
        driver=_FakeDriver(ids=["sec-1"]),
        config=config,
        qdrant_client=None,
        strict_mode=True,
    )

    with pytest.raises(RuntimeError):
        builder._enforce_profile_guard()

    assert (
        _get_metric_value(builder.embedding_settings.profile or "legacy", "blocked")
        == 1
    )


def test_profile_guard_warns_when_strict_mode_disabled():
    """Strict mode off → ingestion continues but drift warning metric fires."""
    config = get_config()
    builder = GraphBuilder(
        driver=_FakeDriver(ids=["sec-2"]),
        config=config,
        qdrant_client=None,
        strict_mode=False,
    )

    builder._enforce_profile_guard()

    assert (
        _get_metric_value(builder.embedding_settings.profile or "legacy", "warned") == 1
    )

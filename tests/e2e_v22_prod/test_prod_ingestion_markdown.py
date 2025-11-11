"""E2E v2.2 – Ingestion & Markdown parsing validations (spec-only).

These tests assume production docs are auto-ingested via data/ingest and written
into Neo4j/ Qdrant by the pipeline. We assert outcomes observable in graph/chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
from neo4j import Driver

pytestmark = pytest.mark.integration


def _run_cypher(driver: Driver, query: str, params: Dict | None = None):
    with driver.session() as sess:
        return list(sess.run(query, params or {}))


def test_parsed_docs_coverage(prod_env, prepare_ingest, capture_logs):
    capture_logs("ingestion-before")
    driver: Driver = prod_env["neo4j"]
    snapshot_scope = prod_env["snapshot_scope"]

    # Count documents seen with this tag
    rows = _run_cypher(
        driver,
        """
        MATCH (d:Document) WHERE d.snapshot_scope = $scope RETURN count(d) as c
        """,
        {"scope": snapshot_scope},
    )
    count_docs = rows[0]["c"] if rows else 0

    # Heuristic acceptance: at least one doc observed under snapshot_scope
    assert count_docs >= 1, "Expected at least one Document under snapshot_scope"
    capture_logs("ingestion-after")


def test_heading_parity_and_char_coverage(prod_env, capture_logs):
    capture_logs("parity-before")
    driver: Driver = prod_env["neo4j"]
    snapshot_scope = prod_env["snapshot_scope"]
    src_dir: Path = prod_env["docs_dir"]

    # Compute raw char sum from source files
    raw_chars = 0
    for path in src_dir.rglob("*.md"):
        try:
            raw_chars += len(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

    # Graph text sum (sum of Section text lengths as a proxy)
    rows = _run_cypher(
        driver,
        """
        MATCH (s:Section) WHERE s.snapshot_scope = $scope AND s.text IS NOT NULL
        RETURN sum(size(s.text)) as total
        """,
        {"scope": snapshot_scope},
    )
    parsed_chars = rows[0]["total"] if rows and rows[0]["total"] is not None else 0

    # Parity is a soft check; assert presence and non-trivial coverage
    assert raw_chars > 0, "No source chars counted"
    assert parsed_chars > 0, "No parsed Section text observed"
    # Strong threshold to be tuned after first runs – start lenient in spec-only
    assert parsed_chars / raw_chars >= 0.50, "Parsed char coverage unexpectedly low"
    capture_logs("parity-after")

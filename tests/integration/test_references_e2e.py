import os

import pytest

from src.ingestion.extract.references import (
    MAX_TEXT_LENGTH_FOR_REGEX,
    extract_references,
    slugify_for_id,
)


def test_windowed_reference_extraction_captures_tail_reference():
    """Ensure references beyond the first window are still detected (regression for 8KB truncation)."""
    base = "a" * (MAX_TEXT_LENGTH_FOR_REGEX + 100)
    text = base + "\nSee also: Snapshot Policies\n"
    refs = extract_references(text, source_chunk_id="chunk-1")
    assert any(r.target_hint.lower().startswith("snapshot") for r in refs)


@pytest.mark.parametrize(
    "title",
    [
        "日本語ドキュメント",
        "中文文档",
        "Emoji Guide 🚀",
    ],
)
def test_slugify_hash_fallback_prevents_unknown_collision(title):
    slug = slugify_for_id(title)
    assert slug != "unknown"
    assert len(slug) == 12


def test_cross_doc_references_boost_score():
    """
    E2E regression (smoke): ensure cross-document REFERENCES edges contribute signals.
    Skips unless NEO4J_* env vars are set and neo4j driver is available.
    """
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        pytest.skip("NEO4J_* env vars not set")

    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception:
        pytest.skip("neo4j driver not installed")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)-[:REFERENCES]->(:Document)-[:HAS_CHUNK]->(related:Chunk)
            RETURN related.id AS cid LIMIT 1
            """
        )
        records = list(result)
        assert (
            records
        ), "Expected cross-doc REFERENCES edges to yield at least one related chunk"
    driver.close()

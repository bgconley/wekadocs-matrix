"""
Test to ensure embedding field canonicalization is maintained.

This test acts as a CI guardrail to prevent regression where
legacy `embedding_model` fields might be reintroduced in persisted data.
"""

import os

import pytest
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from src.shared.config import get_config

# Skip if not in CI or integration test environment
pytestmark = pytest.mark.integration


def test_no_legacy_embedding_model_in_neo4j():
    """Ensure no data nodes have legacy embedding_model field."""
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "testpassword123")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        with driver.session() as session:
            # Check for any nodes with embedding_model (excluding SchemaVersion)
            result = session.run(
                """
                MATCH (n)
                WHERE n.embedding_model IS NOT NULL
                  AND NOT 'SchemaVersion' IN labels(n)
                RETURN count(n) as count
                """
            )
            count = result.single()["count"]

            assert count == 0, (
                f"Found {count} nodes with legacy embedding_model field. "
                "All data nodes must use embedding_version instead."
            )

            # Verify Section and Chunk nodes have canonical fields
            for label in ["Section", "Chunk"]:
                result = session.run(
                    f"""
                    MATCH (n:{label})
                    WHERE n.embedding_version IS NULL
                    RETURN count(n) as count
                    """
                )
                missing_version = result.single()["count"]

                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                total = result.single()["count"]

                if total > 0:  # Only check if nodes exist
                    assert missing_version == 0, (
                        f"Found {missing_version}/{total} {label} nodes without embedding_version. "
                        "All nodes must have canonical embedding fields."
                    )
    finally:
        driver.close()


def test_no_legacy_embedding_model_in_qdrant():
    """Ensure no Qdrant points have legacy embedding_model field."""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "weka_sections_v2")

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        pytest.skip(f"Collection {collection_name} not found")

    # Sample points to check for legacy fields
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    legacy_count = 0
    missing_version = 0

    for point in points:
        if point.payload:
            if "embedding_model" in point.payload:
                legacy_count += 1
            if "embedding_version" not in point.payload:
                missing_version += 1

    assert legacy_count == 0, (
        f"Found {legacy_count} Qdrant points with legacy embedding_model field. "
        "All points must use embedding_version instead."
    )

    assert missing_version == 0, (
        f"Found {missing_version} Qdrant points without embedding_version. "
        "All points must have canonical embedding fields."
    )


def test_canonical_embedding_values():
    """Verify that all embeddings use canonical values."""
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "testpassword123")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    config = get_config()
    canonical_version = config.embedding.version
    canonical_provider = config.embedding.provider
    canonical_dimensions = config.embedding.dims

    try:
        with driver.session() as session:
            # Check a sample of nodes for canonical values
            result = session.run(
                """
                MATCH (n:Section)
                WHERE n.embedding_version <> $version
                   OR n.embedding_provider <> $provider
                   OR n.embedding_dimensions <> $dims
                RETURN count(n) as count
                """,
                version=canonical_version,
                provider=canonical_provider,
                dims=canonical_dimensions,
            )
            non_canonical = result.single()["count"]

            assert non_canonical == 0, (
                f"Found {non_canonical} nodes with non-canonical embedding values. "
                f"Expected: version={canonical_version}, provider={canonical_provider}, "
                f"dimensions={canonical_dimensions}"
            )
    finally:
        driver.close()


if __name__ == "__main__":
    # Run tests locally
    test_no_legacy_embedding_model_in_neo4j()
    test_no_legacy_embedding_model_in_qdrant()
    test_canonical_embedding_values()
    print("✅ All embedding field canonicalization tests passed!")

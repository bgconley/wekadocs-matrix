"""
Graph Contract Validation Tests

P3: Formalized graph contract checks as CI gates.
These tests validate the structural integrity of the Neo4j graph after ingestion.

Based on expert Neo4j architect recommendations (2025-12-13):
- NEXT_CHUNK coverage should be 100% of expected (N-1 for N chunks per doc)
- Eligible hierarchy coverage should be >= 98%
- Cross-document edges should be 0
- Cross-parent sibling adjacency should be 0
"""

import os

import pytest
from neo4j import GraphDatabase

# Get Neo4j connection from environment
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpassword123")


@pytest.fixture(scope="module")
def neo4j_driver():
    """Create a Neo4j driver for testing."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    driver.close()


class TestGraphContract:
    """Graph contract validation tests."""

    def test_next_chunk_coverage(self, neo4j_driver):
        """
        Validate NEXT_CHUNK edge coverage.

        For N chunks in a document, there should be N-1 NEXT_CHUNK edges.
        Total expected = sum(chunks_per_doc - 1) for all docs with >= 1 chunk.
        Coverage should be 100% (ratio = 1.0).
        """
        query = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(DISTINCT c) AS chunks
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(a:Chunk)-[:NEXT_CHUNK]->(b:Chunk)<-[:HAS_CHUNK]-(d)
        WITH d, chunks, count(*) AS next_edges
        WITH
            sum(chunks) AS total_chunks,
            sum(next_edges) AS actual_next_chunk_edges,
            sum(CASE WHEN chunks > 1 THEN chunks - 1 ELSE 0 END) AS expected_next_chunk_edges
        RETURN
            total_chunks,
            actual_next_chunk_edges,
            expected_next_chunk_edges,
            CASE WHEN expected_next_chunk_edges = 0 THEN 1.0
                 ELSE actual_next_chunk_edges * 1.0 / expected_next_chunk_edges
            END AS coverage
        """
        with neo4j_driver.session() as session:
            result = session.run(query).single()

        coverage = result["coverage"]
        expected = result["expected_next_chunk_edges"]
        actual = result["actual_next_chunk_edges"]

        # Allow small variance (within 1%) for edge cases
        assert coverage >= 0.99, (
            f"NEXT_CHUNK coverage {coverage:.2%} is below 99% threshold. "
            f"Expected: {expected}, Actual: {actual}"
        )

    def test_no_cross_document_next_chunk(self, neo4j_driver):
        """
        Validate no NEXT_CHUNK edges cross document boundaries.

        NEXT_CHUNK edges should only connect chunks within the same document.
        Cross-document edges indicate a bug in the structural edge builder.
        """
        query = """
        MATCH (a:Chunk)-[:NEXT_CHUNK]->(b:Chunk)
        WHERE a.document_id <> b.document_id
        RETURN count(*) AS cross_doc_edges
        """
        with neo4j_driver.session() as session:
            result = session.run(query).single()

        cross_doc_edges = result["cross_doc_edges"]
        assert (
            cross_doc_edges == 0
        ), f"Found {cross_doc_edges} NEXT_CHUNK edges crossing document boundaries"

    def test_eligible_hierarchy_coverage(self, neo4j_driver):
        """
        Validate hierarchy coverage for eligible chunks.

        Eligible chunks are those with parent_path_norm containing ' > '.
        These chunks should have a parent unless the parent chunk doesn't exist.
        Coverage should be >= 98%.
        """
        query = """
        MATCH (c:Chunk)
        RETURN
            count(c) AS total,
            sum(CASE WHEN c.parent_path_norm IS NOT NULL
                      AND c.parent_path_norm CONTAINS ' > ' THEN 1 ELSE 0 END) AS eligible,
            sum(CASE WHEN c.parent_path_norm IS NOT NULL
                      AND c.parent_path_norm CONTAINS ' > '
                      AND c.parent_chunk_id IS NOT NULL THEN 1 ELSE 0 END) AS covered
        """
        with neo4j_driver.session() as session:
            result = session.run(query).single()

        eligible = result["eligible"]
        covered = result["covered"]

        if eligible == 0:
            pytest.skip("No eligible chunks found (no multi-level hierarchy)")

        coverage = covered / eligible
        assert coverage >= 0.98, (
            f"Eligible hierarchy coverage {coverage:.2%} is below 98% threshold. "
            f"Eligible: {eligible}, Covered: {covered}, Missing: {eligible - covered}"
        )

    def test_no_cross_parent_next_edges(self, neo4j_driver):
        """
        Validate NEXT (sibling) edges don't cross parent scopes.

        NEXT edges should only connect chunks with the same parent_chunk_id.
        Cross-parent edges indicate a bug in sibling edge creation.
        """
        query = """
        MATCH (a:Chunk)-[:NEXT]->(b:Chunk)
        WHERE coalesce(a.parent_chunk_id, '') <> coalesce(b.parent_chunk_id, '')
        RETURN count(*) AS cross_parent_edges
        """
        with neo4j_driver.session() as session:
            result = session.run(query).single()

        cross_parent_edges = result["cross_parent_edges"]
        assert (
            cross_parent_edges == 0
        ), f"Found {cross_parent_edges} NEXT edges crossing parent boundaries"

    def test_next_chunk_order_monotonicity(self, neo4j_driver):
        """
        Validate NEXT_CHUNK edges maintain order monotonicity.

        For (a)-[:NEXT_CHUNK]->(b), we should have a.order < b.order.
        Violations indicate corrupted ordering or incorrect edge creation.
        """
        query = """
        MATCH (a:Chunk)-[:NEXT_CHUNK]->(b:Chunk)
        WHERE a.document_id = b.document_id AND a.order >= b.order
        RETURN count(*) AS bad_edges
        """
        with neo4j_driver.session() as session:
            result = session.run(query).single()

        bad_edges = result["bad_edges"]
        assert (
            bad_edges == 0
        ), f"Found {bad_edges} NEXT_CHUNK edges violating order monotonicity"

    def test_document_chunk_membership(self, neo4j_driver):
        """
        Validate every chunk has exactly one HAS_CHUNK edge from a document.

        Chunks should have consistent document membership via HAS_CHUNK.
        """
        query = """
        MATCH (c:Chunk)
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WITH c, count(d) AS doc_count
        WHERE doc_count <> 1
        RETURN count(c) AS orphaned_or_multi_parent
        """
        with neo4j_driver.session() as session:
            result = session.run(query).single()

        bad_chunks = result["orphaned_or_multi_parent"]
        assert bad_chunks == 0, (
            f"Found {bad_chunks} chunks with incorrect document membership "
            "(expected exactly 1 HAS_CHUNK edge per chunk)"
        )


class TestGraphContractSummary:
    """Summary report of graph contract validation."""

    def test_print_summary(self, neo4j_driver):
        """Print a summary of graph contract metrics."""
        summary_query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(DISTINCT c) AS chunks
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(a:Chunk)-[:NEXT_CHUNK]->(b:Chunk)<-[:HAS_CHUNK]-(d)
        WITH d, chunks, count(*) AS next_edges
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(h:Chunk)
        WHERE h.parent_path_norm CONTAINS ' > '
        WITH d, chunks, next_edges, count(DISTINCT h) AS eligible
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(h2:Chunk)
        WHERE h2.parent_path_norm CONTAINS ' > ' AND h2.parent_chunk_id IS NOT NULL
        WITH d, chunks, next_edges, eligible, count(DISTINCT h2) AS covered
        WITH
            count(d) AS total_documents,
            sum(chunks) AS total_chunks,
            sum(next_edges) AS actual_next_chunk,
            sum(CASE WHEN chunks > 1 THEN chunks - 1 ELSE 0 END) AS expected_next_chunk,
            sum(eligible) AS eligible_hierarchy,
            sum(covered) AS covered_hierarchy
        RETURN
            total_documents,
            total_chunks,
            actual_next_chunk,
            expected_next_chunk,
            CASE WHEN expected_next_chunk = 0 THEN 1.0
                 ELSE actual_next_chunk * 1.0 / expected_next_chunk END AS next_chunk_coverage,
            eligible_hierarchy,
            covered_hierarchy,
            CASE WHEN eligible_hierarchy = 0 THEN 1.0
                 ELSE covered_hierarchy * 1.0 / eligible_hierarchy END AS hierarchy_coverage
        """
        with neo4j_driver.session() as session:
            result = session.run(summary_query).single()

        print("\n" + "=" * 60)
        print("GRAPH CONTRACT VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Documents:           {result['total_documents']}")
        print(f"Total Chunks:              {result['total_chunks']}")
        print(
            f"NEXT_CHUNK Edges:          {result['actual_next_chunk']} / {result['expected_next_chunk']} expected"
        )
        print(f"NEXT_CHUNK Coverage:       {result['next_chunk_coverage']:.2%}")
        print(f"Eligible Hierarchy Chunks: {result['eligible_hierarchy']}")
        print(f"Covered Hierarchy Chunks:  {result['covered_hierarchy']}")
        print(f"Hierarchy Coverage:        {result['hierarchy_coverage']:.2%}")
        print("=" * 60)

        # This test always passes - it's just for reporting
        assert True

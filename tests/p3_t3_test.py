# Phase 3, Task 3.3 Tests - Graph Construction
# NO MOCKS - Tests against live Neo4j and Qdrant

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.ingestion.build_graph import GraphBuilder
from src.ingestion.extract import extract_entities
from src.ingestion.parsers.markdown import parse_markdown
from src.shared.config import load_config
from src.shared.connections import get_connection_manager


class TestGraphConstruction:
    """Tests for graph construction with real database."""

    @pytest.fixture
    def config(self):
        config, _ = load_config()
        return config

    @pytest.fixture
    def neo4j_driver(self, config):
        manager = get_connection_manager()
        driver = manager.get_neo4j_driver()
        yield driver
        # Don't close - shared

    @pytest.fixture
    def qdrant_client(self, config):
        if config.search.vector.primary == "qdrant" or config.search.vector.dual_write:
            return QdrantClient(
                host="localhost",
                port=6333,
            )
        return None

    @pytest.fixture
    def graph_builder(self, neo4j_driver, config, qdrant_client):
        return GraphBuilder(neo4j_driver, config, qdrant_client)

    @pytest.fixture
    def sample_document(self):
        """Parse a sample document."""
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "getting_started.md"

        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        entities, mentions = extract_entities(result["Sections"])

        return {
            "document": result["Document"],
            "sections": result["Sections"],
            "entities": entities,
            "mentions": mentions,
        }

    def test_upsert_document_creates_nodes(
        self, graph_builder, neo4j_driver, sample_document
    ):
        """Test that upserting creates document and section nodes."""
        stats = graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        assert stats["sections_upserted"] > 0
        assert stats["entities_upserted"] > 0
        assert stats["mentions_created"] > 0

        # Verify document exists in graph
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (d:Document {id: $id}) RETURN count(d) as count",
                id=sample_document["document"]["id"],
            )
            count = result.single()["count"]
            assert count == 1

    def test_idempotent_upsert(self, graph_builder, neo4j_driver, sample_document):
        """Test that re-upserting same document produces same result (DoD)."""
        # First upsert
        graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        # Get counts after first upsert
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
                RETURN count(s) as section_count
                """,
                doc_id=sample_document["document"]["id"],
            )
            first_section_count = result.single()["section_count"]

            result = session.run(
                """
                MATCH ()-[m:MENTIONS]->()
                WHERE m.source_section_id IN [s IN $section_ids | s]
                RETURN count(m) as mention_count
                """,
                section_ids=[s["id"] for s in sample_document["sections"]],
            )
            first_mention_count = result.single()["mention_count"]

        # Second upsert (should be idempotent)
        graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        # Get counts after second upsert
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
                RETURN count(s) as section_count
                """,
                doc_id=sample_document["document"]["id"],
            )
            second_section_count = result.single()["section_count"]

            result = session.run(
                """
                MATCH ()-[m:MENTIONS]->()
                WHERE m.source_section_id IN [s IN $section_ids | s]
                RETURN count(m) as mention_count
                """,
                section_ids=[s["id"] for s in sample_document["sections"]],
            )
            second_mention_count = result.single()["mention_count"]

        # Counts should be identical (idempotent)
        assert first_section_count == second_section_count
        assert first_mention_count == second_mention_count

    def test_mentions_have_provenance(
        self, graph_builder, neo4j_driver, sample_document
    ):
        """Test that MENTIONS edges have required provenance fields."""
        graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        # Query MENTIONS relationships
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Section)-[m:MENTIONS]->(e)
                WHERE s.document_id = $doc_id
                RETURN m.confidence as confidence,
                       m.start as start,
                       m.end as end,
                       m.source_section_id as source_section_id
                LIMIT 10
                """,
                doc_id=sample_document["document"]["id"],
            )

            mentions = [dict(record) for record in result]
            assert len(mentions) > 0

            # Verify provenance fields
            for mention in mentions:
                assert mention["confidence"] is not None
                assert 0.0 <= mention["confidence"] <= 1.0
                assert mention["start"] is not None
                assert mention["end"] is not None
                assert mention["source_section_id"] is not None

    def test_embeddings_computed(self, graph_builder, sample_document):
        """Test that embeddings are computed for sections."""
        stats = graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        assert stats["embeddings_computed"] > 0
        assert stats["vectors_upserted"] > 0
        assert stats["embeddings_computed"] == len(sample_document["sections"])

    def test_vector_parity_with_graph(
        self, graph_builder, neo4j_driver, qdrant_client, config, sample_document
    ):
        """Test that vector store has same count as graph sections (DoD: parity)."""
        graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        # Count sections in graph
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.document_id = $doc_id
                RETURN count(s) as count
                """,
                doc_id=sample_document["document"]["id"],
            )
            graph_count = result.single()["count"]

        # Count vectors in primary store
        if config.search.vector.primary == "qdrant" and qdrant_client:
            # Query Qdrant for vectors with this document_id
            collection_name = config.search.vector.qdrant.collection_name
            result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"value": sample_document["document"]["id"]},
                        }
                    ]
                },
                limit=1000,
                with_vectors=False,
            )
            points, _ = result
            vector_count = len(points)

        else:  # Neo4j vectors
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Section)
                    WHERE s.document_id = $doc_id
                      AND s.vector_embedding IS NOT NULL
                    RETURN count(s) as count
                    """,
                    doc_id=sample_document["document"]["id"],
                )
                vector_count = result.single()["count"]

        # Verify parity
        assert (
            graph_count == vector_count
        ), f"Graph: {graph_count}, Vectors: {vector_count}"

    def test_embedding_version_set(
        self, graph_builder, neo4j_driver, config, sample_document
    ):
        """Test that embedding_version is set on sections."""
        graph_builder.upsert_document(
            sample_document["document"],
            sample_document["sections"],
            sample_document["entities"],
            sample_document["mentions"],
        )

        # Check if neo4j is primary or dual-write enabled
        if config.search.vector.primary == "neo4j" or config.search.vector.dual_write:
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Section)
                    WHERE s.document_id = $doc_id
                    RETURN s.embedding_version as version
                    LIMIT 1
                    """,
                    doc_id=sample_document["document"]["id"],
                )
                record = result.single()
                if record:
                    assert record["version"] == config.embedding.version


class TestBatchProcessing:
    """Tests for batch operations."""

    @pytest.fixture
    def config(self):
        config, _ = load_config()
        return config

    @pytest.fixture
    def neo4j_driver(self, config):
        manager = get_connection_manager()
        driver = manager.get_neo4j_driver()
        yield driver
        # Don't close - shared

    @pytest.fixture
    def graph_builder(self, neo4j_driver, config):
        return GraphBuilder(neo4j_driver, config, qdrant_client=None)

    def test_batch_size_respected(self, graph_builder, config, neo4j_driver):
        """Test that batch size configuration is respected."""
        # The batch size is in config
        assert config.ingestion.batch_size > 0

        # Create a large number of sections to test batching
        sections = []
        for i in range(config.ingestion.batch_size + 10):
            sections.append(
                {
                    "id": f"test-section-{i}",
                    "document_id": "test-doc-batch",
                    "title": f"Section {i}",
                    "text": f"Content for section {i}",
                    "tokens": 10,
                    "checksum": f"checksum-{i}",
                    "anchor": f"section-{i}",
                    "level": 1,
                    "order": i,
                }
            )

        document = {
            "id": "test-doc-batch",
            "source_uri": "test://batch",
            "source_type": "test",
            "title": "Batch Test",
            "version": "1.0",
            "checksum": "batch-checksum",
            "last_edited": None,
        }

        stats = graph_builder.upsert_document(document, sections, {}, [])

        # Verify all sections were created
        assert stats["sections_upserted"] == len(sections)

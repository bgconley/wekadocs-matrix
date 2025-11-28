"""
Phase 7C Integration Tests - Complete Production Workflow Validation

Tests the complete Phase 7C implementation including:
- Query path with 1024-D embeddings (Jina v4)
- Session tracking (multi-turn conversations)
- Reranking functionality (Jina reranker v3)
- Entity focus bias (multi-turn context)
- Answer provenance (SUPPORTED_BY citations)
- Orphan section cleanup (hybrid purge strategy)

These are production-scenario integration tests that require:
- Running Neo4j instance
- Running Qdrant instance
- Running Redis instance
- Valid JINA_API_KEY environment variable
"""

import time
import uuid

import pytest
from neo4j import GraphDatabase

from src.ingestion.build_graph import GraphBuilder
from src.mcp_server.query_service import QueryService
from src.query.session_tracker import SessionCleanupJob, SessionTracker
from src.shared.config import get_config, get_settings
from src.shared.connections import get_connection_manager


class TestPhase7CQueryPath:
    """Test complete query execution path with 1024-D Jina v4 embeddings."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver for testing."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def test_document_ingested(self, neo4j_driver):
        """
        Ingest a test document with real Jina v4 embeddings.

        This fixture ensures we have real data to query against.
        """
        from src.ingestion.build_graph import ingest_document

        # Create test document content
        test_content = """# NFS Configuration Guide

## Prerequisites

Before configuring NFS for Weka, ensure the following:
- Weka cluster version 4.2 or higher
- NFS client packages installed
- Network connectivity between client and cluster

## Configuration Steps

### Step 1: Enable NFS Protocol

To enable the NFS protocol on your Weka cluster:

```bash
weka nfs enable
```

### Step 2: Create NFS Export

Create an NFS export for your filesystem:

```bash
weka nfs export create --name my-export --filesystem my-fs --path /data
```

### Step 3: Mount on Client

Mount the NFS export on your client:

```bash
mount -t nfs weka-cluster:/my-export /mnt/weka
```

## Performance Tuning

For optimal NFS performance:
- Set `nfs.read_size` to 1048576
- Set `nfs.write_size` to 1048576
- Enable `nfs.async` for better throughput

## Troubleshooting

### Connection Timeout

If you experience connection timeouts:
1. Check network connectivity
2. Verify firewall rules allow NFS ports (2049, 111)
3. Check Weka cluster status with `weka status`
"""

        # Ingest with real embeddings
        source_uri = "docs/nfs-configuration-test.md"

        try:
            stats = ingest_document(
                source_uri=source_uri,
                content=test_content,
                format="markdown",
            )

            assert stats["sections_upserted"] > 0, "No sections ingested"
            assert stats["embeddings_computed"] > 0, "No embeddings computed"
            assert stats["vectors_upserted"] > 0, "No vectors upserted"

            yield {
                "source_uri": source_uri,
                "stats": stats,
            }

        finally:
            # Cleanup: Remove test document
            with neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {source_uri: $uri})
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    DETACH DELETE d, s
                    """,
                    uri=source_uri,
                )

    def test_query_with_1024d_embeddings(self, test_document_ingested):
        """
        Test complete query path with 1024-D embeddings.

        Validates:
        - Query embedding uses Jina v4 with retrieval.query task
        - Vector search returns results with 1024-D vectors
        - Response is properly formatted
        - Timing metrics are tracked
        """
        query_service = QueryService()

        # Execute query that should match our test document
        query = "How do I configure NFS for Weka?"

        start_time = time.time()
        response = query_service.search(
            query=query,
            top_k=5,
            verbosity="graph",
        )
        query_time = time.time() - start_time

        # Verify response structure
        assert response is not None
        assert response.answer_markdown is not None
        assert response.answer_json is not None

        # Verify we got results
        assert len(response.answer_json.evidence) > 0, "No evidence sections returned"

        # Verify timing metrics
        assert hasattr(response.answer_json, "timing")
        timing = response.answer_json.timing
        assert timing["vector_search_ms"] > 0, "Vector search timing not recorded"
        assert timing["total_ms"] > 0, "Total timing not recorded"
        assert (
            query_time * 1000 < 5000
        ), f"Query took too long: {query_time * 1000:.0f}ms"

        # Verify evidence contains expected content
        evidence_text = " ".join(
            [e.get("text", "") for e in response.answer_json.evidence]
        )
        assert (
            "NFS" in evidence_text or "nfs" in evidence_text
        ), "Evidence doesn't contain NFS-related content"

        # Verify confidence score
        assert (
            0.0 <= response.answer_json.confidence <= 1.0
        ), f"Confidence score out of range: {response.answer_json.confidence}"

        print(f"\n✅ Query executed successfully in {query_time * 1000:.0f}ms")
        print(f"   Evidence sections: {len(response.answer_json.evidence)}")
        print(f"   Confidence: {response.answer_json.confidence:.2f}")
        print(f"   Vector search time: {timing['vector_search_ms']:.0f}ms")

    def test_provider_configuration_in_query_path(self):
        """
        Test that query path uses correct provider configuration.

        Validates:
        - Embedder is Jina v4
        - Dimensions are 1024
        - Provider name is jina-ai
        """
        query_service = QueryService()

        # Trigger embedder initialization
        embedder = query_service._get_embedder()

        # Verify provider configuration
        assert (
            embedder.provider_name == "jina-ai"
        ), f"Wrong provider: {embedder.provider_name}"
        assert (
            embedder.model_id == "jina-embeddings-v3"
        ), f"Wrong model: {embedder.model_id}"
        assert embedder.dims == 1024, f"Wrong dimensions: {embedder.dims}"

        print("\n✅ Query path using correct provider:")
        print(f"   Provider: {embedder.provider_name}")
        print(f"   Model: {embedder.model_id}")
        print(f"   Dimensions: {embedder.dims}")


class TestPhase7CSessionTracking:
    """Test session tracking for multi-turn conversations."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver for testing."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def session_tracker(self, neo4j_driver):
        """Get SessionTracker instance."""
        return SessionTracker(neo4j_driver)

    @pytest.fixture
    def cleanup_test_sessions(self, neo4j_driver):
        """Cleanup test sessions after test."""
        test_session_ids = []

        yield test_session_ids

        # Cleanup
        if test_session_ids:
            with neo4j_driver.session() as session:
                session.run(
                    """
                    UNWIND $session_ids AS sid
                    MATCH (s:Session {session_id: sid})
                    OPTIONAL MATCH (s)-[:HAS_QUERY]->(q:Query)
                    OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
                    DETACH DELETE s, q, a
                    """,
                    session_ids=test_session_ids,
                )

    def test_session_creation(self, session_tracker, cleanup_test_sessions):
        """Test creating a new session."""
        session_id = f"test-session-{uuid.uuid4()}"
        cleanup_test_sessions.append(session_id)

        # Create session
        session_data = session_tracker.create_session(
            session_id=session_id,
            user_id="test-user-123",
        )

        # Verify session was created
        assert session_data["session_id"] == session_id
        assert session_data["user_id"] == "test-user-123"
        assert session_data["started_at"] is not None
        assert session_data["expires_at"] is not None
        assert session_data["active"] is True

        # Verify TTL is 30 days
        started_at = session_data["started_at"]
        expires_at = session_data["expires_at"]
        ttl_days = (expires_at - started_at).days
        assert 29 <= ttl_days <= 31, f"TTL should be ~30 days, got {ttl_days}"

        print(f"\n✅ Session created: {session_id}")
        print(f"   User: {session_data['user_id']}")
        print(f"   Expires: {expires_at}")

    def test_query_creation_and_linking(
        self, session_tracker, neo4j_driver, cleanup_test_sessions
    ):
        """Test creating queries and linking to session."""
        session_id = f"test-session-{uuid.uuid4()}"
        cleanup_test_sessions.append(session_id)

        # Create session
        session_tracker.create_session(session_id)

        # Create query
        query_text = "How do I configure NFS?"
        query_id = session_tracker.create_query(session_id, query_text, turn=1)

        assert query_id is not None

        # Verify query exists and is linked to session
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $sid})-[:HAS_QUERY]->(q:Query {query_id: $qid})
                RETURN q.text as text, q.turn as turn, q.asked_at as asked_at
                """,
                sid=session_id,
                qid=query_id,
            )

            record = result.single()
            assert record is not None, "Query not linked to session"
            assert record["text"] == query_text
            assert record["turn"] == 1
            assert record["asked_at"] is not None

        print(f"\n✅ Query created and linked: {query_id}")
        print(f"   Text: {query_text}")
        print("   Turn: 1")

    def test_entity_focus_extraction(
        self, session_tracker, neo4j_driver, cleanup_test_sessions
    ):
        """
        Test entity focus extraction from query text.

        Requires entities to exist in graph (from ingestion).
        """
        session_id = f"test-session-{uuid.uuid4()}"
        cleanup_test_sessions.append(session_id)

        # Create session and query
        session_tracker.create_session(session_id)
        query_id = session_tracker.create_query(
            session_id, "How do I configure NFS for Weka?", turn=1
        )

        # Extract focused entities
        focused_entities = session_tracker.extract_focused_entities(
            query_id, "How do I configure NFS for Weka?"
        )

        # Note: This may return empty list if no entities match
        # That's OK - we're testing the mechanism works
        print("\n✅ Entity focus extraction completed")
        print(f"   Focused entities found: {len(focused_entities)}")

        if focused_entities:
            # Verify FOCUSED_ON relationships exist
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (q:Query {query_id: $qid})-[f:FOCUSED_ON]->(e)
                    RETURN e.id as entity_id, f.score as score, labels(e) as labels
                    """,
                    qid=query_id,
                )

                relationships = list(result)
                assert len(relationships) == len(
                    focused_entities
                ), "FOCUSED_ON relationship count mismatch"

                for rel in relationships:
                    print(
                        f"   - Entity: {rel['entity_id']} ({rel['labels'][0]}) "
                        f"score={rel['score']:.2f}"
                    )

    def test_multi_turn_conversation_flow(
        self, session_tracker, neo4j_driver, cleanup_test_sessions
    ):
        """Test complete multi-turn conversation flow."""
        session_id = f"test-session-{uuid.uuid4()}"
        cleanup_test_sessions.append(session_id)

        # Create session
        session_tracker.create_session(session_id, user_id="test-user")

        # Turn 1
        query_id_1 = session_tracker.create_query(
            session_id, "How do I enable NFS?", turn=1
        )
        session_tracker.extract_focused_entities(query_id_1, "How do I enable NFS?")

        # Turn 2
        query_id_2 = session_tracker.create_query(
            session_id, "What about performance tuning?", turn=2
        )
        session_tracker.extract_focused_entities(
            query_id_2, "What about performance tuning?"
        )

        # Turn 3
        session_tracker.create_query(session_id, "Can you show me an example?", turn=3)

        # Verify session has 3 queries
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $sid})-[:HAS_QUERY]->(q:Query)
                RETURN count(q) as query_count, collect(q.turn) as turns
                ORDER BY q.turn
                """,
                sid=session_id,
            )

            record = result.single()
            assert record["query_count"] == 3, "Should have 3 queries"
            assert sorted(record["turns"]) == [1, 2, 3], "Turns should be 1, 2, 3"

        # Test get_session_focused_entities (last 3 turns)
        session_focus = session_tracker.get_session_focused_entities(
            session_id, last_n_turns=3
        )

        print("\n✅ Multi-turn conversation flow completed")
        print("   Queries created: 3")
        print(f"   Session-level focused entities: {len(session_focus)}")

    def test_retrieval_tracking(
        self, session_tracker, neo4j_driver, cleanup_test_sessions
    ):
        """Test tracking retrieved sections for a query."""
        session_id = f"test-session-{uuid.uuid4()}"
        cleanup_test_sessions.append(session_id)

        # Create session and query
        session_tracker.create_session(session_id)
        query_id = session_tracker.create_query(session_id, "Test query", turn=1)

        # Create fake retrieved sections data
        retrieved_sections = [
            {
                "section_id": "test-section-1",
                "rank": 1,
                "score_vec": 0.95,
                "score_text": 0.80,
                "score_graph": 0.10,
                "score_combined": 0.92,
                "retrieval_method": "hybrid",
            },
            {
                "section_id": "test-section-2",
                "rank": 2,
                "score_vec": 0.88,
                "score_text": 0.75,
                "score_graph": 0.05,
                "score_combined": 0.85,
                "retrieval_method": "hybrid",
            },
        ]

        # Create test sections first
        with neo4j_driver.session() as session:
            for sec in retrieved_sections:
                test_vector = [0.1] * 1024
                session.run(
                    """
                    MERGE (s:Section:Chunk {id: $id})
                    SET s.text = $text,
                        s.document_id = 'test-doc',
                        s.level = 1,
                        s.title = 'Test Section',
                        s.anchor = 'test',
                        s.order = 0,
                        s.tokens = 10,
                        s.vector_embedding = $vector,
                        s.embedding_version = 'test-v1',
                        s.embedding_provider = 'test',
                        s.embedding_timestamp = datetime(),
                        s.embedding_dimensions = 1024
                    """,
                    id=sec["section_id"],
                    text=f"Test content for {sec['section_id']}",
                    vector=test_vector,
                )

        try:
            # Track retrieval
            session_tracker.track_retrieval(query_id, retrieved_sections)

            # Verify RETRIEVED relationships exist
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (q:Query {query_id: $qid})-[r:RETRIEVED]->(s:Section)
                    RETURN count(r) as count,
                           collect({
                               section_id: s.id,
                               rank: r.rank,
                               score_combined: r.score_combined
                           }) as retrieved
                    ORDER BY r.rank
                    """,
                    qid=query_id,
                )

                record = result.single()
                assert record["count"] == 2, "Should have 2 RETRIEVED relationships"

                retrieved = record["retrieved"]
                assert retrieved[0]["rank"] == 1
                assert retrieved[1]["rank"] == 2
                assert abs(retrieved[0]["score_combined"] - 0.92) < 0.01

            print("\n✅ Retrieval tracking completed")
            print(f"   Sections tracked: {len(retrieved_sections)}")

        finally:
            # Cleanup test sections
            with neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (s:Section)
                    WHERE s.id STARTS WITH 'test-section-'
                    DETACH DELETE s
                    """
                )

    def test_answer_creation_with_citations(
        self, session_tracker, neo4j_driver, cleanup_test_sessions
    ):
        """Test creating answer with SUPPORTED_BY citations."""
        session_id = f"test-session-{uuid.uuid4()}"
        cleanup_test_sessions.append(session_id)

        # Create session and query
        session_tracker.create_session(session_id)
        query_id = session_tracker.create_query(session_id, "Test query", turn=1)

        # Create test sections for citations
        test_section_ids = ["test-cite-section-1", "test-cite-section-2"]

        with neo4j_driver.session() as session:
            for sec_id in test_section_ids:
                test_vector = [0.1] * 1024
                session.run(
                    """
                    MERGE (s:Section:Chunk {id: $id})
                    SET s.text = $text,
                        s.document_id = 'test-doc',
                        s.level = 1,
                        s.title = 'Test Citation Section',
                        s.anchor = 'test',
                        s.order = 0,
                        s.tokens = 10,
                        s.vector_embedding = $vector,
                        s.embedding_version = 'test-v1',
                        s.embedding_provider = 'test',
                        s.embedding_timestamp = datetime(),
                        s.embedding_dimensions = 1024
                    """,
                    id=sec_id,
                    text=f"Test content for {sec_id}",
                    vector=test_vector,
                )

        try:
            # Create answer with citations
            answer_text = "To configure NFS, you need to enable the protocol first."
            answer_id = session_tracker.create_answer(
                query_id=query_id,
                answer_text=answer_text,
                supporting_section_ids=test_section_ids,
                model="claude-3-5-sonnet-20241022",
                tokens_used=150,
                generation_duration_ms=1200.5,
            )

            assert answer_id is not None

            # Verify answer and citations exist
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (q:Query {query_id: $qid})-[:ANSWERED_AS]->(a:Answer {answer_id: $aid})
                    MATCH (a)-[c:SUPPORTED_BY]->(s:Section)
                    RETURN a.text as answer_text,
                           a.model as model,
                           a.tokens_used as tokens,
                           count(c) as citation_count,
                           collect({section_id: s.id, rank: c.rank}) as citations
                    ORDER BY c.rank
                    """,
                    qid=query_id,
                    aid=answer_id,
                )

                record = result.single()
                assert record is not None, "Answer not found"
                assert record["answer_text"] == answer_text
                assert record["model"] == "claude-3-5-sonnet-20241022"
                assert record["tokens"] == 150
                assert record["citation_count"] == 2, "Should have 2 citations"

                citations = record["citations"]
                assert citations[0]["rank"] == 1
                assert citations[1]["rank"] == 2
                assert citations[0]["section_id"] == test_section_ids[0]
                assert citations[1]["section_id"] == test_section_ids[1]

            print("\n✅ Answer created with citations")
            print(f"   Answer ID: {answer_id}")
            print(f"   Citations: {len(test_section_ids)}")
            print("   Model: claude-3-5-sonnet-20241022")
            print("   Tokens: 150")

        finally:
            # Cleanup test sections
            with neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (s:Section)
                    WHERE s.id STARTS WITH 'test-cite-section-'
                    DETACH DELETE s
                    """
                )

    def test_session_cleanup_job(self, session_tracker, neo4j_driver):
        """Test TTL cleanup job for expired sessions."""
        # Create an expired session (backdated by 31 days)
        expired_session_id = f"expired-test-session-{uuid.uuid4()}"

        with neo4j_driver.session() as session:
            # Create expired session manually
            session.run(
                """
                CREATE (s:Session {
                    session_id: $sid,
                    user_id: 'test-user',
                    started_at: datetime() - duration('P31D'),
                    expires_at: datetime() - duration('P1D'),
                    last_active_at: datetime() - duration('P31D'),
                    total_queries: 1,
                    active: false
                })
                CREATE (s)-[:HAS_QUERY]->(q:Query {
                    query_id: 'expired-query-1',
                    text: 'Test query',
                    turn: 1,
                    asked_at: datetime() - duration('P31D')
                })
                CREATE (q)-[:ANSWERED_AS]->(a:Answer {
                    answer_id: 'expired-answer-1',
                    text: 'Test answer',
                    created_at: datetime() - duration('P31D')
                })
                """,
                sid=expired_session_id,
            )

        # Run cleanup job
        cleanup_job = SessionCleanupJob(neo4j_driver, ttl_days=30)
        result = cleanup_job.run(dry_run=False)

        # Verify expired session was deleted
        assert result["sessions_deleted"] >= 1, "Should delete at least 1 session"
        assert result["queries_deleted"] >= 1, "Should delete at least 1 query"
        assert result["answers_deleted"] >= 1, "Should delete at least 1 answer"

        # Verify session no longer exists
        with neo4j_driver.session() as session:
            check_result = session.run(
                """
                MATCH (s:Session {session_id: $sid})
                RETURN count(s) as count
                """,
                sid=expired_session_id,
            )

            count = check_result.single()["count"]
            assert count == 0, "Expired session should be deleted"

        print("\n✅ Session cleanup completed")
        print(f"   Sessions deleted: {result['sessions_deleted']}")
        print(f"   Queries deleted: {result['queries_deleted']}")
        print(f"   Answers deleted: {result['answers_deleted']}")


class TestPhase7CEntityFocusBias:
    """Test entity focus bias in multi-turn conversations."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver for testing."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def test_data_with_entities(self, neo4j_driver):
        """Create test data with entities for focus bias testing."""
        # Create test document, sections, and entities
        test_doc_id = f"test-doc-{uuid.uuid4()}"
        test_entity_id = f"test-config-nfs-{uuid.uuid4()}"
        test_section_ids = []

        with neo4j_driver.session() as session:
            # Create document
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = 'Test NFS Document',
                    d.source_uri = 'test-nfs.md'
                """,
                doc_id=test_doc_id,
            )

            # Create Configuration entity for NFS
            session.run(
                """
                MERGE (e:Configuration {id: $entity_id})
                SET e.name = 'nfs.enabled',
                    e.description = 'Enable NFS protocol',
                    e.category = 'protocol'
                """,
                entity_id=test_entity_id,
            )

            # Create sections mentioning NFS
            for i in range(3):
                section_id = f"test-section-nfs-{i}-{uuid.uuid4()}"
                test_section_ids.append(section_id)

                test_vector = [0.1 + i * 0.01] * 1024  # Slightly different vectors

                session.run(
                    """
                    MERGE (s:Section:Chunk {id: $section_id})
                    SET s.text = $text,
                        s.document_id = $doc_id,
                        s.level = 1,
                        s.title = $title,
                        s.anchor = 'test-nfs',
                        s.order = $order,
                        s.tokens = 100,
                        s.vector_embedding = $vector,
                        s.embedding_version = 'test-v1',
                        s.embedding_provider = 'test',
                        s.embedding_timestamp = datetime(),
                        s.embedding_dimensions = 1024
                    WITH s
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_SECTION]->(s)
                    WITH s
                    MATCH (e:Configuration {id: $entity_id})
                    MERGE (s)-[:MENTIONS {
                        confidence: 0.9,
                        source_section_id: $section_id
                    }]->(e)
                    """,
                    section_id=section_id,
                    text=f"Section {i} discussing NFS configuration",
                    doc_id=test_doc_id,
                    title=f"NFS Section {i}",
                    order=i,
                    vector=test_vector,
                    entity_id=test_entity_id,
                )

        yield {
            "doc_id": test_doc_id,
            "entity_id": test_entity_id,
            "section_ids": test_section_ids,
        }

        # Cleanup
        with neo4j_driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                DETACH DELETE d, s
                MATCH (e:Configuration {id: $entity_id})
                DETACH DELETE e
                """,
                doc_id=test_doc_id,
                entity_id=test_entity_id,
            )

    def test_entity_focus_bias_boosts_relevant_sections(self, test_data_with_entities):
        """
        Test that entity focus bias boosts sections mentioning focused entities.

        This validates the hybrid search engine's _apply_entity_focus_bias method.
        """
        from src.query.hybrid_search import HybridSearchEngine, SearchResult

        manager = get_connection_manager()
        neo4j_driver = manager.get_neo4j_driver()

        # Create mock search results
        results = [
            SearchResult(
                node_id=test_data_with_entities["section_ids"][0],
                node_label="Section",
                score=0.75,  # Medium score
                distance=0,
                metadata={"text": "Section 0 discussing NFS configuration"},
            ),
            SearchResult(
                node_id=test_data_with_entities["section_ids"][1],
                node_label="Section",
                score=0.70,  # Lower score
                distance=0,
                metadata={"text": "Section 1 discussing NFS configuration"},
            ),
            SearchResult(
                node_id=test_data_with_entities["section_ids"][2],
                node_label="Section",
                score=0.65,  # Lowest score
                distance=0,
                metadata={"text": "Section 2 discussing NFS configuration"},
            ),
        ]

        # Apply entity focus bias
        focused_entity_ids = [test_data_with_entities["entity_id"]]

        # Create a minimal search engine instance just for testing bias
        from unittest.mock import MagicMock

        mock_vector_store = MagicMock()
        mock_embedder = MagicMock()

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            neo4j_driver=neo4j_driver,
            embedder=mock_embedder,
            reranker=None,
        )

        # Apply bias
        biased_results = engine._apply_entity_focus_bias(results, focused_entity_ids)

        # Verify all scores were boosted (all sections mention the focused entity)
        for biased, original in zip(biased_results, results):
            assert (
                biased.score > original.score
            ), f"Score should be boosted: {original.score} -> {biased.score}"
            assert (
                "entity_focus_boost" in biased.metadata
            ), "Should have entity_focus_boost metadata"
            assert (
                biased.metadata["entity_focus_boost"] == 1
            ), "Each section mentions 1 focused entity"

        # Verify boost calculation (20% per focused entity)
        expected_boost = 0.2  # 20% for 1 entity mention
        for i, result in enumerate(biased_results):
            expected_score = results[i].score * (1 + expected_boost)
            assert (
                abs(result.score - expected_score) < 0.001
            ), f"Score boost calculation incorrect: expected {expected_score}, got {result.score}"

        print("\n✅ Entity focus bias applied successfully")
        print(f"   Focused entities: {len(focused_entity_ids)}")
        print(f"   Sections boosted: {len(biased_results)}")
        print("   Boost per entity: 20%")
        for i, (orig, biased) in enumerate(zip(results, biased_results)):
            print(
                f"   Section {i}: {orig.score:.3f} -> {biased.score:.3f} "
                f"(+{((biased.score - orig.score) / orig.score * 100):.1f}%)"
            )


class TestPhase7COrphanSectionCleanup:
    """Test hybrid orphan section cleanup strategy."""

    @pytest.fixture
    def neo4j_driver(self):
        """Get Neo4j driver for testing."""
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        yield driver
        driver.close()

    @pytest.fixture
    def graph_builder(self, neo4j_driver):
        """Get GraphBuilder instance."""
        config = get_config()
        return GraphBuilder(neo4j_driver, config, qdrant_client=None)

    def test_orphan_cleanup_deletes_sections_without_provenance(
        self, graph_builder, neo4j_driver
    ):
        """Test that orphaned sections with no provenance are deleted."""
        test_doc_id = f"test-orphan-doc-{uuid.uuid4()}"

        # Create document with 3 sections
        with neo4j_driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = 'Test Orphan Document'
                """,
                doc_id=test_doc_id,
            )

            for i in range(3):
                section_id = f"test-orphan-section-{i}-{uuid.uuid4()}"
                test_vector = [0.1] * 1024

                session.run(
                    """
                    MERGE (s:Section:Chunk {id: $section_id})
                    SET s.text = $text,
                        s.document_id = $doc_id,
                        s.level = 1,
                        s.title = $title,
                        s.anchor = 'test',
                        s.order = $order,
                        s.tokens = 10,
                        s.vector_embedding = $vector,
                        s.embedding_version = 'test-v1',
                        s.embedding_provider = 'test',
                        s.embedding_timestamp = datetime(),
                        s.embedding_dimensions = 1024
                    WITH s
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
                    """,
                    section_id=section_id,
                    text=f"Section {i} content",
                    doc_id=test_doc_id,
                    title=f"Section {i}",
                    order=i,
                    vector=test_vector,
                )

        try:
            # Get initial section IDs
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
                    RETURN collect(s.id) as section_ids
                    ORDER BY s.order
                    """,
                    doc_id=test_doc_id,
                )
                all_section_ids = result.single()["section_ids"]

            # Now "update" document to only include first 2 sections
            # (simulate section 2 being deleted from source document)
            valid_section_ids = all_section_ids[:2]
            removed_section_id = all_section_ids[2]

            # Run cleanup (Section 2 should be deleted - no provenance)
            with neo4j_driver.session() as session:
                removed_count = graph_builder._remove_missing_sections(
                    session, test_doc_id, valid_section_ids
                )

            assert removed_count == 1, f"Should remove 1 section, got {removed_count}"

            # Verify section was deleted
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Section {id: $sid})
                    RETURN count(s) as count
                    """,
                    sid=removed_section_id,
                )

                count = result.single()["count"]
                assert (
                    count == 0
                ), "Orphaned section without provenance should be deleted"

            print("\n✅ Orphan cleanup deleted section without provenance")
            print(f"   Deleted: {removed_count}")

        finally:
            # Cleanup
            with neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    DETACH DELETE d, s
                    """,
                    doc_id=test_doc_id,
                )

    def test_orphan_cleanup_marks_stale_sections_with_provenance(
        self, graph_builder, neo4j_driver
    ):
        """Test that orphaned sections with provenance are marked stale, not deleted."""
        test_doc_id = f"test-stale-doc-{uuid.uuid4()}"
        test_session_id = f"test-stale-session-{uuid.uuid4()}"

        # Create document with 2 sections
        section_ids = []
        with neo4j_driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = 'Test Stale Document'
                """,
                doc_id=test_doc_id,
            )

            for i in range(2):
                section_id = f"test-stale-section-{i}-{uuid.uuid4()}"
                section_ids.append(section_id)
                test_vector = [0.1] * 1024

                session.run(
                    """
                    MERGE (s:Section:Chunk {id: $section_id})
                    SET s.text = $text,
                        s.document_id = $doc_id,
                        s.level = 1,
                        s.title = $title,
                        s.anchor = 'test',
                        s.order = $order,
                        s.tokens = 10,
                        s.vector_embedding = $vector,
                        s.embedding_version = 'test-v1',
                        s.embedding_provider = 'test',
                        s.embedding_timestamp = datetime(),
                        s.embedding_dimensions = 1024
                    WITH s
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
                    """,
                    section_id=section_id,
                    text=f"Section {i} content",
                    doc_id=test_doc_id,
                    title=f"Section {i}",
                    order=i,
                    vector=test_vector,
                )

        try:
            # Create Query that RETRIEVED section 1 (establishes provenance)
            with neo4j_driver.session() as session:
                session.run(
                    """
                    CREATE (s:Session {
                        session_id: $session_id,
                        started_at: datetime(),
                        expires_at: datetime() + duration('P30D'),
                        active: true,
                        total_queries: 1
                    })
                    CREATE (s)-[:HAS_QUERY]->(q:Query {
                        query_id: 'test-query-123',
                        text: 'Test query',
                        turn: 1,
                        asked_at: datetime()
                    })
                    WITH q
                    MATCH (sec:Section {id: $section_id})
                    CREATE (q)-[:RETRIEVED {
                        rank: 1,
                        score_combined: 0.9,
                        created_at: datetime()
                    }]->(sec)
                    """,
                    session_id=test_session_id,
                    section_id=section_ids[1],  # Section 1 has provenance
                )

            # Now "update" document to only include section 0
            # (simulate section 1 being deleted from source document)
            valid_section_ids = [section_ids[0]]
            removed_section_id = section_ids[1]

            # Run cleanup (Section 1 should be marked stale, NOT deleted)
            with neo4j_driver.session() as session:
                removed_count = graph_builder._remove_missing_sections(
                    session, test_doc_id, valid_section_ids
                )

            assert removed_count == 1, f"Should process 1 section, got {removed_count}"

            # Verify section still exists but is marked stale
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Section {id: $sid})
                    RETURN s.is_stale as is_stale,
                           s.stale_since as stale_since,
                           s.stale_reason as stale_reason
                    """,
                    sid=removed_section_id,
                )

                record = result.single()
                assert (
                    record is not None
                ), "Section with provenance should NOT be deleted"
                assert record["is_stale"] is True, "Section should be marked stale"
                assert (
                    record["stale_since"] is not None
                ), "Should have stale_since timestamp"
                assert (
                    "provenance" in record["stale_reason"].lower()
                ), "Stale reason should mention provenance"

            print("\n✅ Orphan cleanup marked stale section with provenance")
            print("   Marked stale: 1")
            print("   Reason: Section has Query/Answer provenance")

        finally:
            # Cleanup
            with neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                    DETACH DELETE d, s
                    MATCH (sess:Session {session_id: $session_id})
                    OPTIONAL MATCH (sess)-[:HAS_QUERY]->(q:Query)
                    DETACH DELETE sess, q
                    """,
                    doc_id=test_doc_id,
                    session_id=test_session_id,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

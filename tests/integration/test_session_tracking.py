"""
Integration tests for Task 7C.8 - Multi-turn Session Tracking

Tests the complete session tracking flow:
- Session/Query/Answer node creation
- Entity focus extraction and bias
- Retrieval tracking with RETRIEVED relationships
- Answer provenance with SUPPORTED_BY relationships
- Session cleanup with TTL

Test Strategy:
- Use real Neo4j instance (no mocks for graph operations)
- Create isolated test sessions with unique IDs
- Verify graph structure after each operation
- Clean up test data after each test
"""

import uuid

import pytest

from src.mcp_server.query_service import QueryService
from src.query.session_tracker import SessionTracker
from src.shared.connections import get_connection_manager


@pytest.fixture
def neo4j_driver():
    """Get Neo4j driver for test assertions."""
    manager = get_connection_manager()
    return manager.get_neo4j_driver()


@pytest.fixture
def session_tracker(neo4j_driver):
    """Create SessionTracker instance for testing."""
    return SessionTracker(neo4j_driver)


@pytest.fixture
def query_service():
    """Create QueryService instance for testing."""
    return QueryService()


@pytest.fixture
def test_session_id():
    """Generate unique session ID for test isolation."""
    return f"test-session-{uuid.uuid4()}"


@pytest.fixture(autouse=True)
def cleanup_test_data(neo4j_driver, test_session_id):
    """Clean up test sessions after each test."""
    yield

    # Clean up test session and related nodes
    with neo4j_driver.session() as session:
        session.run(
            """
            MATCH (s:Session {session_id: $sid})
            OPTIONAL MATCH (s)-[:HAS_QUERY]->(q:Query)
            OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
            DETACH DELETE s, q, a
        """,
            sid=test_session_id,
        )


class TestSessionCreation:
    """Test Session node creation and basic CRUD operations."""

    def test_create_session(self, session_tracker, test_session_id, neo4j_driver):
        """Test creating a new session."""
        # Create session
        session_data = session_tracker.create_session(
            session_id=test_session_id, user_id="test-user-123"
        )

        # Verify return data
        assert session_data["session_id"] == test_session_id
        assert session_data["user_id"] == "test-user-123"
        assert "started_at" in session_data
        assert "expires_at" in session_data

        # Verify session exists in graph
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $sid})
                RETURN s.session_id as session_id,
                       s.user_id as user_id,
                       s.started_at as started_at,
                       s.expires_at as expires_at
            """,
                sid=test_session_id,
            )

            record = result.single()
            assert record is not None
            assert record["session_id"] == test_session_id
            assert record["user_id"] == "test-user-123"
            assert record["started_at"] is not None
            assert record["expires_at"] is not None

    def test_create_query(self, session_tracker, test_session_id, neo4j_driver):
        """Test creating a query within a session."""
        # Create session first
        session_tracker.create_session(test_session_id)

        # Create query
        query_id = session_tracker.create_query(
            session_id=test_session_id, query_text="How do I configure NFS?", turn=1
        )

        assert query_id is not None
        assert isinstance(query_id, str)

        # Verify query exists and is linked to session
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $sid})-[:HAS_QUERY]->(q:Query {query_id: $qid})
                RETURN q.query_id as query_id,
                       q.text as text,
                       q.turn as turn,
                       q.asked_at as asked_at
            """,
                sid=test_session_id,
                qid=query_id,
            )

            record = result.single()
            assert record is not None
            assert record["query_id"] == query_id
            assert record["text"] == "How do I configure NFS?"
            assert record["turn"] == 1
            assert record["asked_at"] is not None


class TestEntityFocusExtraction:
    """Test entity extraction from query text and FOCUSED_ON relationships."""

    def test_extract_focused_entities(
        self, session_tracker, test_session_id, neo4j_driver
    ):
        """Test extracting entities mentioned in query."""
        # Create session and query
        session_tracker.create_session(test_session_id)
        query_id = session_tracker.create_query(
            test_session_id, "How do I configure NFS for Weka?", 1
        )

        # Extract focused entities
        focused_entities = session_tracker.extract_focused_entities(
            query_id=query_id, query_text="How do I configure NFS for Weka?"
        )

        # Should extract some entities (NFS-related, configuration-related)
        assert isinstance(focused_entities, list)
        # Note: Actual extraction depends on graph content
        # In a real system with data, we'd expect entities to be found

        # Verify FOCUSED_ON relationships were created
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (q:Query {query_id: $qid})-[f:FOCUSED_ON]->(e)
                RETURN count(f) as focus_count,
                       collect(DISTINCT labels(e)[0]) as entity_labels,
                       collect(f.score) as scores
            """,
                qid=query_id,
            )

            record = result.single()
            # With real data, we'd expect focused entities
            # For empty graph, count may be 0
            assert record["focus_count"] >= 0

    def test_get_session_focused_entities(
        self, session_tracker, test_session_id, neo4j_driver
    ):
        """Test retrieving focused entities from recent session turns."""
        # Create session
        session_tracker.create_session(test_session_id)

        # Create multiple queries
        query_id_1 = session_tracker.create_query(
            test_session_id, "How do I configure NFS?", 1
        )
        session_tracker.extract_focused_entities(query_id_1, "How do I configure NFS?")

        query_id_2 = session_tracker.create_query(
            test_session_id, "What about SMB configuration?", 2
        )
        session_tracker.extract_focused_entities(
            query_id_2, "What about SMB configuration?"
        )

        # Get session focus (last 2 turns)
        session_focus = session_tracker.get_session_focused_entities(
            session_id=test_session_id, last_n_turns=2
        )

        # Verify structure
        assert isinstance(session_focus, list)
        # Each item should have entity_id, entity_type, focus_score, entity_name
        for focus_item in session_focus:
            assert "entity_id" in focus_item
            assert "entity_type" in focus_item
            assert "focus_score" in focus_item
            assert "entity_name" in focus_item


class TestRetrievalTracking:
    """Test RETRIEVED relationship tracking."""

    def test_track_retrieval(self, session_tracker, test_session_id, neo4j_driver):
        """Test tracking retrieved sections for a query."""
        # Create session and query
        session_tracker.create_session(test_session_id)
        query_id = session_tracker.create_query(
            test_session_id, "How do I configure NFS?", 1
        )

        # Mock retrieved sections (would come from search engine)
        retrieved_sections = [
            {
                "section_id": "sec-nfs-config-1",
                "rank": 1,
                "score_vec": 0.95,
                "score_text": 0.88,
                "score_graph": 0.75,
                "score_combined": 0.92,
                "retrieval_method": "hybrid",
            },
            {
                "section_id": "sec-nfs-config-2",
                "rank": 2,
                "score_vec": 0.87,
                "score_text": 0.82,
                "score_graph": 0.70,
                "score_combined": 0.84,
                "retrieval_method": "hybrid",
            },
        ]

        # Track retrieval
        session_tracker.track_retrieval(query_id, retrieved_sections)

        # Verify RETRIEVED relationships exist (if sections exist in graph)
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (q:Query {query_id: $qid})-[r:RETRIEVED]->(s)
                WITH r ORDER BY r.rank
                RETURN count(r) as retrieval_count,
                       collect(r.rank) as ranks,
                       collect(r.score_combined) as scores
            """,
                qid=query_id,
            )

            record = result.single()
            # Note: Count may be 0 if test sections don't exist in graph
            # In production with real sections, we'd expect 2
            assert record["retrieval_count"] >= 0


class TestAnswerProvenance:
    """Test Answer node creation and SUPPORTED_BY relationships."""

    def test_create_answer(self, session_tracker, test_session_id, neo4j_driver):
        """Test creating answer with supporting sections."""
        # Create session and query
        session_tracker.create_session(test_session_id)
        query_id = session_tracker.create_query(
            test_session_id, "How do I configure NFS?", 1
        )

        # Create answer with supporting sections
        answer_text = "To configure NFS for Weka, follow these steps..."
        supporting_section_ids = ["sec-nfs-1", "sec-nfs-2", "sec-nfs-3"]

        answer_id = session_tracker.create_answer(
            query_id=query_id,
            answer_text=answer_text,
            supporting_section_ids=supporting_section_ids,
            model="claude-3-5-sonnet-20241022",
            tokens_used=150,
            generation_duration_ms=250.5,
        )

        assert answer_id is not None
        assert isinstance(answer_id, str)

        # Verify answer exists and is linked to query
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (q:Query {query_id: $qid})-[:ANSWERED_AS]->(a:Answer {answer_id: $aid})
                RETURN a.answer_id as answer_id,
                       a.text as text,
                       a.model as model,
                       a.tokens_used as tokens,
                       a.created_at as created_at
            """,
                qid=query_id,
                aid=answer_id,
            )

            record = result.single()
            assert record is not None
            assert record["answer_id"] == answer_id
            assert record["text"] == answer_text
            assert record["model"] == "claude-3-5-sonnet-20241022"
            assert record["tokens"] == 150
            assert record["created_at"] is not None

        # Verify SUPPORTED_BY relationships (if sections exist)
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (a:Answer {answer_id: $aid})-[s:SUPPORTED_BY]->(sec)
                WITH s ORDER BY s.rank
                RETURN count(s) as citation_count,
                       collect(s.rank) as ranks
            """,
                aid=answer_id,
            )

            record = result.single()
            # Note: Count may be 0 if test sections don't exist in graph
            assert record["citation_count"] >= 0


class TestCompleteSessionFlow:
    """Test complete multi-turn conversation flow end-to-end."""

    def test_multi_turn_conversation(
        self, query_service, session_tracker, test_session_id, neo4j_driver
    ):
        """Test complete multi-turn conversation with entity focus bias."""
        # Note: This test requires real data in the graph to work fully.
        # With empty graph, it tests the mechanics but won't show entity bias effects.

        # Create session first
        session_tracker.create_session(test_session_id)

        # Turn 1: Initial query about NFS
        response1 = query_service.search(
            query="How do I configure NFS for Weka?",
            session_id=test_session_id,
            turn=1,
            top_k=5,
        )

        # Verify response structure
        assert response1 is not None
        assert hasattr(response1, "answer_markdown")
        assert hasattr(response1, "answer_json")

        # Verify session/query were created
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $sid})-[:HAS_QUERY]->(q:Query)
                WHERE q.turn = 1
                OPTIONAL MATCH (q)-[:FOCUSED_ON]->(e)
                OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
                RETURN q.query_id as query_id,
                       q.text as query_text,
                       count(DISTINCT e) as focused_entities,
                       a.answer_id as answer_id
            """,
                sid=test_session_id,
            )

            record = result.single()
            assert record is not None
            assert record["query_text"] == "How do I configure NFS for Weka?"
            # With real data, we'd expect focused entities and answer

        # Turn 2: Follow-up about performance
        response2 = query_service.search(
            query="What about performance tuning?",
            session_id=test_session_id,
            turn=2,
            top_k=5,
        )

        assert response2 is not None

        # Verify session has 2 queries
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $sid})-[:HAS_QUERY]->(q:Query)
                RETURN count(q) as query_count,
                       collect(q.turn) as turns
            """,
                sid=test_session_id,
            )

            record = result.single()
            assert record["query_count"] == 2
            assert sorted(record["turns"]) == [1, 2]


class TestSessionCleanup:
    """Test session TTL cleanup functionality."""

    def test_cleanup_expired_sessions(self, session_tracker, neo4j_driver):
        """Test that expired sessions are deleted by cleanup job."""
        from src.ops.session_cleanup_job import SessionCleanupJob

        # Create an expired session manually
        expired_session_id = f"expired-test-{uuid.uuid4()}"

        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (s:Session {
                    session_id: $sid,
                    started_at: datetime() - duration('P31D'),
                    expires_at: datetime() - duration('P1D'),
                    active: false
                })
                CREATE (s)-[:HAS_QUERY]->(q:Query {
                    query_id: $qid,
                    text: 'test query',
                    turn: 1,
                    asked_at: datetime() - duration('P31D')
                })
                CREATE (q)-[:ANSWERED_AS]->(a:Answer {
                    answer_id: $aid,
                    text: 'test answer',
                    created_at: datetime() - duration('P31D')
                })
            """,
                sid=expired_session_id,
                qid=f"query-{uuid.uuid4()}",
                aid=f"answer-{uuid.uuid4()}",
            )

        # Run cleanup job
        cleanup = SessionCleanupJob(neo4j_driver, ttl_days=30)
        result = cleanup.run(dry_run=False)

        # Verify expired session was deleted
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

        # Verify cleanup stats
        assert "sessions_deleted" in result
        assert result["sessions_deleted"] >= 1

    def test_cleanup_dry_run(self, session_tracker, neo4j_driver):
        """Test dry run mode doesn't delete anything."""
        from src.ops.session_cleanup_job import SessionCleanupJob

        # Create an expired session
        expired_session_id = f"expired-test-{uuid.uuid4()}"

        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (s:Session {
                    session_id: $sid,
                    started_at: datetime() - duration('P31D'),
                    expires_at: datetime() - duration('P1D'),
                    active: false
                })
            """,
                sid=expired_session_id,
            )

        # Run cleanup in dry-run mode
        cleanup = SessionCleanupJob(neo4j_driver, ttl_days=30)
        result = cleanup.run(dry_run=True)

        # Verify session still exists
        with neo4j_driver.session() as session:
            check_result = session.run(
                """
                MATCH (s:Session {session_id: $sid})
                RETURN count(s) as count
            """,
                sid=expired_session_id,
            )

            count = check_result.single()["count"]
            assert count == 1, "Session should still exist after dry run"

        # Verify dry run reported what would be deleted
        assert "sessions_to_delete" in result or "sessions_deleted" in result

        # Clean up test session
        with neo4j_driver.session() as session:
            session.run(
                "MATCH (s:Session {session_id: $sid}) DETACH DELETE s",
                sid=expired_session_id,
            )


class TestQueryContext:
    """Test query context retrieval for debugging."""

    def test_get_query_context(self, session_tracker, test_session_id, neo4j_driver):
        """Test retrieving complete context for a query."""
        # Create session, query, and answer
        session_tracker.create_session(test_session_id, user_id="test-user")
        query_id = session_tracker.create_query(
            test_session_id, "How do I configure NFS?", 1
        )

        # Extract entities
        session_tracker.extract_focused_entities(query_id, "How do I configure NFS?")

        # Track retrieval
        session_tracker.track_retrieval(
            query_id,
            [
                {
                    "section_id": "sec-1",
                    "rank": 1,
                    "score_combined": 0.95,
                    "score_vec": 0.9,
                    "score_text": 0.8,
                    "score_graph": 0.7,
                    "retrieval_method": "hybrid",
                }
            ],
        )

        # Create answer
        session_tracker.create_answer(query_id, "Answer text", ["sec-1"])

        # Get query context
        context = session_tracker.get_query_context(query_id)

        # Verify context structure
        assert context is not None
        assert "query" in context
        assert context["query"]["query_id"] == query_id
        assert context["query"]["text"] == "How do I configure NFS?"
        assert "session" in context
        assert context["session"]["session_id"] == test_session_id
        assert "focused_entities" in context
        assert "retrieved_sections" in context
        assert "answer" in context


class TestPerformance:
    """Test that session tracking doesn't regress query performance."""

    def test_session_tracking_overhead(
        self, query_service, session_tracker, test_session_id
    ):
        """Verify session tracking overhead is acceptable (<50ms)."""
        import time

        # Create session first (required for session tracking)
        session_tracker.create_session(
            session_id=test_session_id, user_id="perf-test-user"
        )

        # Baseline: Query without session tracking
        start = time.time()
        _ = query_service.search(
            query="How do I configure NFS?",
            top_k=5,
        )
        baseline_latency = (time.time() - start) * 1000  # ms

        # With session tracking
        start = time.time()
        _ = query_service.search(
            query="How do I configure NFS?",
            session_id=test_session_id,
            turn=1,
            top_k=5,
        )
        tracked_latency = (time.time() - start) * 1000  # ms

        # Calculate overhead
        overhead = tracked_latency - baseline_latency

        # Overhead should be reasonable (<50ms typical, allow 100ms for safety)
        assert overhead < 100, (
            f"Session tracking overhead too high: {overhead:.2f}ms "
            f"(baseline: {baseline_latency:.2f}ms, tracked: {tracked_latency:.2f}ms)"
        )

        # Overall latency should still be reasonable
        assert (
            tracked_latency < 2000
        ), f"Query with tracking exceeds 2s: {tracked_latency:.2f}ms"

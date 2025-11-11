"""
Session tracking service for multi-turn conversation support.

Implements Session/Query/Answer graph tracking with entity focus extraction
and retrieval provenance for conversational RAG.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from neo4j import Driver

logger = logging.getLogger(__name__)


class SessionTracker:
    """
    Tracks user sessions, queries, and answer provenance for multi-turn conversations.

    Creates and manages Session/Query/Answer nodes in the knowledge graph,
    tracks entity focus across turns, and maintains retrieval provenance chains.
    """

    def __init__(self, neo4j_driver: Driver, ttl_days: int = 30):
        """
        Initialize session tracker.

        Args:
            neo4j_driver: Neo4j driver instance
            ttl_days: Session TTL in days (default: 30)
        """
        self.driver = neo4j_driver
        self.ttl_days = ttl_days

    def create_session(self, session_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Create a new Session node with TTL expiration.

        Args:
            session_id: Unique session identifier (UUID recommended)
            user_id: Optional user identifier for attribution

        Returns:
            Dictionary with session metadata:
                {session_id, user_id, started_at, expires_at, active}
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(days=self.ttl_days)

        query = """
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET
            s.user_id = $user_id,
            s.started_at = $started_at,
            s.last_active_at = $started_at,
            s.expires_at = $expires_at,
            s.total_queries = 0,
            s.active = true
        ON MATCH SET
            s.last_active_at = $started_at
        RETURN s.session_id as session_id,
               s.user_id as user_id,
               s.started_at as started_at,
               s.expires_at as expires_at,
               s.active as active
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                session_id=session_id,
                user_id=user_id,
                started_at=now,
                expires_at=expires_at,
            )
            record = result.single()

            if not record:
                raise RuntimeError(f"Failed to create session {session_id}")

            return {
                "session_id": record["session_id"],
                "user_id": record["user_id"],
                "started_at": record["started_at"],
                "expires_at": record["expires_at"],
                "active": record["active"],
            }

    def create_query(self, session_id: str, query_text: str, turn: int) -> str:
        """
        Create Query node and link to Session.

        Args:
            session_id: Parent session ID
            query_text: User's query text
            turn: Turn number within session (1-indexed)

        Returns:
            query_id: Generated query ID (UUID)
        """
        query_id = str(uuid.uuid4())
        now = datetime.utcnow()

        query = """
        MATCH (s:Session {session_id: $session_id})
        CREATE (q:Query {
            query_id: $query_id,
            text: $text,
            turn: $turn,
            asked_at: $asked_at
        })
        CREATE (s)-[:HAS_QUERY {created_at: $asked_at}]->(q)
        SET s.total_queries = s.total_queries + 1,
            s.last_active_at = $asked_at
        RETURN q.query_id as query_id
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                session_id=session_id,
                query_id=query_id,
                text=query_text,
                turn=turn,
                asked_at=now,
            )
            record = result.single()

            if not record:
                raise RuntimeError(f"Failed to create query for session {session_id}")

            return record["query_id"]

    def extract_focused_entities(self, query_id: str, query_text: str) -> List[str]:
        """
        Extract entities from query text and create FOCUSED_ON relationships.

        Uses keyword matching against entity names in the graph.
        Supports all 12 domain entity types.

        Args:
            query_id: Query ID to link entities to
            query_text: Query text to analyze

        Returns:
            List of entity IDs that were focused on
        """
        # Normalize query text for matching
        query_lower = query_text.lower()

        # Extract entity mentions by matching against graph entity names
        # Query all entity types and find mentions in query text
        # Use CALL with UNION ALL and WHERE inside each branch for valid Cypher
        query = """
        CALL {
            MATCH (e:Command)
            WHERE toLower(e.name) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.name)
            RETURN e.id as id, e.name as name, 'Command' as type
            UNION ALL
            MATCH (e:Configuration)
            WHERE toLower(e.name) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.name)
            RETURN e.id as id, e.name as name, 'Configuration' as type
            UNION ALL
            MATCH (e:Procedure)
            WHERE toLower(e.title) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.title)
            RETURN e.id as id, e.title as name, 'Procedure' as type
            UNION ALL
            MATCH (e:Error)
            WHERE toLower(e.message) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.message)
            RETURN e.id as id, e.message as name, 'Error' as type
            UNION ALL
            MATCH (e:Concept)
            WHERE toLower(e.term) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.term)
            RETURN e.id as id, e.term as name, 'Concept' as type
            UNION ALL
            MATCH (e:Example)
            WHERE toLower(e.title) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.title)
            RETURN e.id as id, e.title as name, 'Example' as type
            UNION ALL
            MATCH (e:Parameter)
            WHERE toLower(e.name) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.name)
            RETURN e.id as id, e.name as name, 'Parameter' as type
            UNION ALL
            MATCH (e:Component)
            WHERE toLower(e.name) CONTAINS $query_lower
               OR $query_lower CONTAINS toLower(e.name)
            RETURN e.id as id, e.name as name, 'Component' as type
        }
        RETURN id, name, type
        LIMIT 20
        """

        focused_entity_ids = []

        with self.driver.session() as session:
            result = session.run(query, query_lower=query_lower)
            matches = list(result)

            if not matches:
                logger.debug(f"No entity matches found for query: {query_text}")
                return []

            # Create FOCUSED_ON relationships for matched entities
            for record in matches:
                entity_id = record["id"]
                entity_name = record["name"]
                entity_type = record["type"]

                # Calculate focus score based on match quality
                # Exact match = 1.0, partial match = 0.7
                if query_lower == entity_name.lower():
                    score = 1.0
                elif entity_name.lower() in query_lower:
                    score = 0.9
                elif query_lower in entity_name.lower():
                    score = 0.8
                else:
                    score = 0.7

                # Create FOCUSED_ON relationship
                focus_query = """
                MATCH (q:Query {query_id: $query_id})
                MATCH (e {id: $entity_id})
                WHERE $entity_type IN labels(e)
                MERGE (q)-[f:FOCUSED_ON]->(e)
                ON CREATE SET
                    f.score = $score,
                    f.extraction_method = 'keyword',
                    f.created_at = datetime()
                RETURN e.id as entity_id
                """

                focus_result = session.run(
                    focus_query,
                    query_id=query_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    score=score,
                )

                if focus_result.single():
                    focused_entity_ids.append(entity_id)
                    logger.debug(
                        f"Focused on {entity_type} '{entity_name}' "
                        f"(score={score:.2f})"
                    )

        logger.info(
            f"Extracted {len(focused_entity_ids)} focused entities for query {query_id}"
        )
        return focused_entity_ids

    def get_session_focused_entities(
        self, session_id: str, last_n_turns: int = 3
    ) -> List[Dict]:
        """
        Get entities focused on in recent turns of session.

        Used to bias retrieval in follow-up queries (entity focus bias).

        Args:
            session_id: Session ID
            last_n_turns: Number of recent turns to consider (default: 3)

        Returns:
            List of dicts with entity metadata:
                [{entity_id, entity_type, focus_score, entity_name, turn}]
            Sorted by recency (most recent first)
        """
        query = """
        MATCH (s:Session {session_id: $session_id})-[:HAS_QUERY]->(q:Query)
        WHERE q.turn > (s.total_queries - $last_n_turns)
        MATCH (q)-[f:FOCUSED_ON]->(e)
        RETURN DISTINCT
            e.id as entity_id,
            head(labels(e)) as entity_type,
            f.score as focus_score,
            COALESCE(e.name, e.title, e.term, e.message, 'Unknown') as entity_name,
            q.turn as turn
        ORDER BY q.turn DESC, f.score DESC
        """

        with self.driver.session() as session:
            result = session.run(
                query, session_id=session_id, last_n_turns=last_n_turns
            )

            focused = [
                {
                    "entity_id": record["entity_id"],
                    "entity_type": record["entity_type"],
                    "focus_score": record["focus_score"],
                    "entity_name": record["entity_name"],
                    "turn": record["turn"],
                }
                for record in result
            ]

            logger.debug(
                f"Found {len(focused)} focused entities from last {last_n_turns} turns"
            )
            return focused

    def track_retrieval(self, query_id: str, retrieved_sections: List[Dict]) -> None:
        """
        Create RETRIEVED relationships from Query to Sections.

        Tracks which sections were retrieved for provenance and debugging.

        Args:
            query_id: Query ID
            retrieved_sections: List of retrieved sections with scores
                Format: [{
                    section_id: str,
                    rank: int,
                    score_vec: float,
                    score_text: float,
                    score_graph: float,
                    score_combined: float,
                    retrieval_method: str (optional)
                }]
        """
        if not retrieved_sections:
            logger.warning(f"No sections to track for query {query_id}")
            return

        # Batch create RETRIEVED relationships
        query = """
        MATCH (q:Query {query_id: $query_id})
        UNWIND $sections as section
        MATCH (s:Section {id: section.section_id})
        MERGE (q)-[r:RETRIEVED]->(s)
        ON CREATE SET
            r.rank = section.rank,
            r.score_vec = section.score_vec,
            r.score_text = section.score_text,
            r.score_graph = section.score_graph,
            r.score_combined = section.score_combined,
            r.retrieval_method = section.retrieval_method,
            r.created_at = datetime()
        RETURN count(r) as tracked_count
        """

        with self.driver.session() as session:
            result = session.run(query, query_id=query_id, sections=retrieved_sections)
            record = result.single()

            if record:
                tracked_count = record["tracked_count"]
                logger.info(
                    f"Tracked {tracked_count} retrieved sections for query {query_id}"
                )
            else:
                logger.warning(f"Failed to track retrieval for query {query_id}")

    def create_answer(
        self,
        query_id: str,
        answer_text: str,
        supporting_section_ids: List[str],
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        generation_duration_ms: Optional[float] = None,
    ) -> str:
        """
        Create Answer node and link to Query and supporting Sections.

        Establishes answer provenance via SUPPORTED_BY citations.

        Args:
            query_id: Parent query ID
            answer_text: Generated answer text (Markdown)
            supporting_section_ids: Section IDs cited in answer (in citation order)
            model: Optional LLM model identifier
            tokens_used: Optional token count
            generation_duration_ms: Optional generation latency

        Returns:
            answer_id: Generated answer ID (UUID)
        """
        answer_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Create Answer node
        create_answer_query = """
        MATCH (q:Query {query_id: $query_id})
        CREATE (a:Answer {
            answer_id: $answer_id,
            text: $text,
            model: $model,
            tokens_used: $tokens_used,
            generation_duration_ms: $generation_duration_ms,
            created_at: $created_at
        })
        CREATE (q)-[:ANSWERED_AS {created_at: $created_at}]->(a)
        RETURN a.answer_id as answer_id
        """

        with self.driver.session() as session:
            result = session.run(
                create_answer_query,
                query_id=query_id,
                answer_id=answer_id,
                text=answer_text,
                model=model,
                tokens_used=tokens_used,
                generation_duration_ms=generation_duration_ms,
                created_at=now,
            )
            record = result.single()

            if not record:
                raise RuntimeError(f"Failed to create answer for query {query_id}")

            # Create SUPPORTED_BY relationships to cited sections
            if supporting_section_ids:
                citation_query = """
                MATCH (a:Answer {answer_id: $answer_id})
                UNWIND $citations as citation
                MATCH (s:Section {id: citation.section_id})
                CREATE (a)-[r:SUPPORTED_BY {
                    rank: citation.rank,
                    created_at: datetime()
                }]->(s)
                RETURN count(r) as citation_count
                """

                citations = [
                    {"section_id": sec_id, "rank": idx + 1}
                    for idx, sec_id in enumerate(supporting_section_ids)
                ]

                citation_result = session.run(
                    citation_query, answer_id=answer_id, citations=citations
                )
                citation_record = citation_result.single()

                if citation_record:
                    logger.info(
                        f"Created answer {answer_id} with "
                        f"{citation_record['citation_count']} citations"
                    )
                else:
                    logger.warning(f"Answer {answer_id} created but citations failed")
            else:
                logger.warning(f"Answer {answer_id} created with no citations")

            return answer_id

    def get_query_context(self, query_id: str) -> Dict:
        """
        Get complete context for a query (for debugging/analysis).

        Args:
            query_id: Query ID

        Returns:
            Dictionary with complete query context:
                {
                    query_id, text, turn, asked_at,
                    session_id, session_started_at,
                    focused_entities: [{id, type, name, score}],
                    retrieved_sections: [{id, title, rank, score}],
                    answer: {id, text, model, citations: [{id, title, rank}]}
                }
        """
        query = """
        MATCH (s:Session)-[:HAS_QUERY]->(q:Query {query_id: $query_id})

        // Collect focused entities
        OPTIONAL MATCH (q)-[f:FOCUSED_ON]->(e)
        WITH q, s, collect(DISTINCT {
            id: e.id,
            type: head(labels(e)),
            name: COALESCE(e.name, e.title, e.term, e.message),
            score: f.score
        }) as focused_entities

        // Collect retrieved sections
        OPTIONAL MATCH (q)-[r:RETRIEVED]->(sec:Section)
        WITH q, s, focused_entities, collect(DISTINCT {
            id: sec.id,
            title: sec.title,
            rank: r.rank,
            score: r.score_combined
        }) as retrieved_sections

        // Collect answer and citations
        OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
        OPTIONAL MATCH (a)-[c:SUPPORTED_BY]->(cited:Section)
        WITH q, s, focused_entities, retrieved_sections,
             a.answer_id as answer_id,
             a.text as answer_text,
             a.model as answer_model,
             collect(DISTINCT {
                id: cited.id,
                title: cited.title,
                rank: c.rank
             }) as citations

        RETURN
            q.query_id as query_id,
            q.text as text,
            q.turn as turn,
            q.asked_at as asked_at,
            s.session_id as session_id,
            s.started_at as session_started_at,
            focused_entities,
            retrieved_sections,
            CASE WHEN answer_id IS NOT NULL THEN {
                id: answer_id,
                text: answer_text,
                model: answer_model,
                citations: citations
            } ELSE null END as answer
        """

        with self.driver.session() as session:
            result = session.run(query, query_id=query_id)
            record = result.single()

            if not record:
                raise ValueError(f"Query {query_id} not found")

            # Convert to expected nested structure
            return {
                "query": {
                    "query_id": record["query_id"],
                    "text": record["text"],
                    "turn": record["turn"],
                    "asked_at": record["asked_at"],
                },
                "session": {
                    "session_id": record["session_id"],
                    "started_at": record["session_started_at"],
                },
                "focused_entities": record["focused_entities"],
                "retrieved_sections": record["retrieved_sections"],
                "answer": record["answer"],
            }


class SessionCleanupJob:
    """
    TTL cleanup job for expired sessions.

    Runs periodically to delete sessions older than TTL threshold.
    Cascade deletes related Query and Answer nodes.
    """

    def __init__(self, neo4j_driver: Driver, ttl_days: int = 30):
        """
        Initialize cleanup job.

        Args:
            neo4j_driver: Neo4j driver instance
            ttl_days: Session TTL in days (default: 30)
        """
        self.driver = neo4j_driver
        self.ttl_days = ttl_days

    def run(self, dry_run: bool = False) -> Dict:
        """
        Delete expired sessions and cascade to Queries/Answers.

        Args:
            dry_run: If True, only count expired sessions without deleting

        Returns:
            Statistics: {
                sessions_deleted: int,
                queries_deleted: int,
                answers_deleted: int,
                cutoff_time: datetime
            }
        """
        cutoff_time = datetime.utcnow()

        if dry_run:
            # Count what would be deleted
            # Use Neo4j's datetime() function for proper comparison
            count_query = """
            MATCH (s:Session)
            WHERE s.expires_at < datetime()
            OPTIONAL MATCH (s)-[:HAS_QUERY]->(q:Query)
            OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
            RETURN
                count(DISTINCT s) as sessions,
                count(DISTINCT q) as queries,
                count(DISTINCT a) as answers
            """

            with self.driver.session() as session:
                result = session.run(count_query)
                record = result.single()

                logger.info(
                    f"[DRY RUN] Would delete {record['sessions']} sessions, "
                    f"{record['queries']} queries, {record['answers']} answers"
                )

                return {
                    "sessions_deleted": record["sessions"],
                    "queries_deleted": record["queries"],
                    "answers_deleted": record["answers"],
                    "cutoff_time": cutoff_time,
                    "dry_run": True,
                }

        # Delete expired sessions (cascade deletes handled by DETACH DELETE)
        # Use Neo4j's datetime() function for proper datetime comparison
        # First count what we'll delete
        count_query = """
        MATCH (s:Session)
        WHERE s.expires_at < datetime()
        OPTIONAL MATCH (s)-[:HAS_QUERY]->(q:Query)
        OPTIONAL MATCH (q)-[:ANSWERED_AS]->(a:Answer)
        RETURN
            count(DISTINCT s) as session_count,
            count(DISTINCT q) as query_count,
            count(DISTINCT a) as answer_count
        """

        # Then delete them
        delete_query = """
        MATCH (s:Session)
        WHERE s.expires_at < datetime()
        DETACH DELETE s
        """

        with self.driver.session() as session:
            # Get counts first
            count_result = session.run(count_query)
            counts = count_result.single()

            sessions_to_delete = counts["session_count"] or 0
            queries_to_delete = counts["query_count"] or 0
            answers_to_delete = counts["answer_count"] or 0

            # Perform deletion
            session.run(delete_query)

            sessions_deleted = sessions_to_delete
            queries_deleted = queries_to_delete
            answers_deleted = answers_to_delete

            logger.info(
                f"Deleted {sessions_deleted} expired sessions, "
                f"{queries_deleted} queries, {answers_deleted} answers "
                f"(cutoff: {cutoff_time})"
            )

            return {
                "sessions_deleted": sessions_deleted,
                "queries_deleted": queries_deleted,
                "answers_deleted": answers_deleted,
                "cutoff_time": cutoff_time,
                "dry_run": False,
            }

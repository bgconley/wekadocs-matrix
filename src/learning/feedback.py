"""Feedback collection system for query results.

Implements Phase 4 Task 4.4 - Learning & adaptation.
Logs query→result→feedback for ranking weight tuning and template mining.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import Driver


@dataclass
class QueryFeedback:
    """Structured feedback for a query execution."""

    query_id: str
    query_text: str
    intent: str
    cypher_query: str
    result_ids: List[str]
    rating: Optional[float] = None  # 0.0-1.0 or None if not rated
    notes: Optional[str] = None
    missed_entities: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ranking_features: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "intent": self.intent,
            "cypher_query": self.cypher_query,
            "result_ids": self.result_ids,
            "rating": self.rating,
            "notes": self.notes,
            "missed_entities": self.missed_entities,
            "timestamp": self.timestamp.isoformat(),
            "ranking_features": self.ranking_features,
            "execution_time_ms": self.execution_time_ms,
        }


class FeedbackCollector:
    """Collects and persists query feedback for learning."""

    def __init__(self, driver: Driver):
        """Initialize feedback collector.

        Args:
            driver: Neo4j driver for persistence
        """
        self.driver = driver
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure feedback schema exists in Neo4j."""
        with self.driver.session() as session:
            # Create QueryFeedback nodes with constraints
            session.run(
                """
                CREATE CONSTRAINT query_feedback_id IF NOT EXISTS
                FOR (qf:QueryFeedback) REQUIRE qf.query_id IS UNIQUE
            """
            )

            # Index for timestamp-based queries
            session.run(
                """
                CREATE INDEX query_feedback_timestamp IF NOT EXISTS
                FOR (qf:QueryFeedback) ON (qf.timestamp)
            """
            )

            # Index for intent-based analysis
            session.run(
                """
                CREATE INDEX query_feedback_intent IF NOT EXISTS
                FOR (qf:QueryFeedback) ON (qf.intent)
            """
            )

    def log_query(
        self,
        query_text: str,
        intent: str,
        cypher_query: str,
        result_ids: List[str],
        ranking_features: Dict[str, float],
        execution_time_ms: float,
        query_id: Optional[str] = None,
    ) -> str:
        """Log a query execution for later feedback.

        Args:
            query_text: Natural language query
            intent: Classified intent
            cypher_query: Generated Cypher
            result_ids: IDs of returned results
            ranking_features: Feature scores used in ranking
            execution_time_ms: Query execution time
            query_id: Optional pre-assigned query ID

        Returns:
            Query ID for later feedback attachment
        """
        if query_id is None:
            query_id = str(uuid.uuid4())

        feedback = QueryFeedback(
            query_id=query_id,
            query_text=query_text,
            intent=intent,
            cypher_query=cypher_query,
            result_ids=result_ids,
            ranking_features=ranking_features,
            execution_time_ms=execution_time_ms,
        )

        self._persist_feedback(feedback)
        return query_id

    def add_feedback(
        self,
        query_id: str,
        rating: float,
        notes: Optional[str] = None,
        missed_entities: Optional[List[str]] = None,
    ) -> None:
        """Add user feedback to a logged query.

        Args:
            query_id: Query ID from log_query
            rating: Rating score 0.0-1.0 (0=bad, 1=perfect)
            notes: Optional textual feedback
            missed_entities: Optional list of entities that should have been returned
        """
        if not 0.0 <= rating <= 1.0:
            raise ValueError(f"Rating must be in [0.0, 1.0], got {rating}")

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (qf:QueryFeedback {query_id: $query_id})
                SET qf.rating = $rating,
                    qf.notes = $notes,
                    qf.missed_entities = $missed_entities,
                    qf.feedback_timestamp = datetime()
                RETURN qf.query_id as query_id
            """,
                {
                    "query_id": query_id,
                    "rating": rating,
                    "notes": notes,
                    "missed_entities": missed_entities or [],
                },
            )

            if not result.single():
                raise ValueError(f"Query ID {query_id} not found")

    def _persist_feedback(self, feedback: QueryFeedback) -> None:
        """Persist feedback to Neo4j."""
        with self.driver.session() as session:
            # Serialize ranking_features to JSON string (Neo4j doesn't support nested dicts)
            ranking_features_json = (
                json.dumps(feedback.ranking_features)
                if feedback.ranking_features
                else "{}"
            )

            session.run(
                """
                MERGE (qf:QueryFeedback {query_id: $query_id})
                SET qf.query_text = $query_text,
                    qf.intent = $intent,
                    qf.cypher_query = $cypher_query,
                    qf.result_ids = $result_ids,
                    qf.rating = $rating,
                    qf.notes = $notes,
                    qf.missed_entities = $missed_entities,
                    qf.timestamp = datetime($timestamp),
                    qf.ranking_features_json = $ranking_features_json,
                    qf.execution_time_ms = $execution_time_ms
            """,
                {
                    "query_id": feedback.query_id,
                    "query_text": feedback.query_text,
                    "intent": feedback.intent,
                    "cypher_query": feedback.cypher_query,
                    "result_ids": feedback.result_ids,
                    "rating": feedback.rating,
                    "notes": feedback.notes,
                    "missed_entities": feedback.missed_entities,
                    "timestamp": feedback.timestamp.isoformat(),
                    "ranking_features_json": ranking_features_json,
                    "execution_time_ms": feedback.execution_time_ms,
                },
            )

    def get_feedback(
        self,
        intent: Optional[str] = None,
        rated_only: bool = False,
        limit: int = 1000,
    ) -> List[QueryFeedback]:
        """Retrieve stored feedback for analysis.

        Args:
            intent: Filter by intent (optional)
            rated_only: Only return feedback with ratings
            limit: Maximum number of results

        Returns:
            List of QueryFeedback objects
        """
        with self.driver.session() as session:
            query = "MATCH (qf:QueryFeedback)"

            where_clauses = []
            if intent:
                where_clauses.append("qf.intent = $intent")
            if rated_only:
                where_clauses.append("qf.rating IS NOT NULL")

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += """
                RETURN qf.query_id as query_id,
                       qf.query_text as query_text,
                       qf.intent as intent,
                       qf.cypher_query as cypher_query,
                       qf.result_ids as result_ids,
                       qf.rating as rating,
                       qf.notes as notes,
                       qf.missed_entities as missed_entities,
                       qf.timestamp as timestamp,
                       qf.ranking_features_json as ranking_features_json,
                       qf.execution_time_ms as execution_time_ms
                ORDER BY qf.timestamp DESC
                LIMIT $limit
            """

            result = session.run(query, {"intent": intent, "limit": limit})

            feedbacks = []
            for record in result:
                # Deserialize ranking_features from JSON
                ranking_features = {}
                if record["ranking_features_json"]:
                    try:
                        ranking_features = json.loads(record["ranking_features_json"])
                    except json.JSONDecodeError:
                        ranking_features = {}

                # Handle timestamp (may be datetime object or string)
                ts = record["timestamp"]
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts)
                else:
                    # Neo4j datetime object - convert to Python datetime
                    timestamp = datetime.fromisoformat(ts.iso_format())

                feedbacks.append(
                    QueryFeedback(
                        query_id=record["query_id"],
                        query_text=record["query_text"],
                        intent=record["intent"],
                        cypher_query=record["cypher_query"],
                        result_ids=record["result_ids"] or [],
                        rating=record["rating"],
                        notes=record["notes"],
                        missed_entities=record["missed_entities"] or [],
                        timestamp=timestamp,
                        ranking_features=ranking_features,
                        execution_time_ms=record["execution_time_ms"],
                    )
                )

            return feedbacks

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback collection statistics.

        Returns:
            Dict with counts, ratings, etc.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (qf:QueryFeedback)
                RETURN count(qf) as total,
                       count(qf.rating) as rated,
                       avg(qf.rating) as avg_rating,
                       collect(DISTINCT qf.intent) as intents
            """
            )

            record = result.single()
            return {
                "total_queries": record["total"],
                "rated_queries": record["rated"],
                "avg_rating": record["avg_rating"],
                "intents": record["intents"] or [],
            }

from __future__ import annotations

import uuid
from typing import List, Optional


class SessionTracker:
    def __init__(self, driver):
        self.driver = driver

    def create_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())
        with self.driver.session() as s:
            s.run(
                """
                MERGE (ss:Session {session_id: $sid})
                ON CREATE SET ss.started_at = datetime(),
                              ss.expires_at = datetime() + duration('P30D'),
                              ss.active = true
                SET ss.last_active_at = datetime()
            """,
                sid=sid,
            )
        return sid

    def create_query(self, session_id: str, text: str, turn: int) -> str:
        qid = str(uuid.uuid4())
        with self.driver.session() as s:
            s.run(
                """
                MATCH (ss:Session {session_id: $sid})
                CREATE (q:Query {query_id: $qid, text: $text, turn: $turn, asked_at: datetime()})
                CREATE (ss)-[:HAS_QUERY {created_at: datetime()}]->(q)
            """,
                sid=session_id,
                qid=qid,
                text=text,
                turn=turn,
            )
        return qid

    def track_retrieval(self, query_id: str, ranked):
        with self.driver.session() as s:
            s.run(
                """
                UNWIND $rows AS r
                MATCH (q:Query {query_id: $qid})
                MATCH (sec:Section {id: r.section_id})
                MERGE (q)-[rel:RETRIEVED]->(sec)
                SET rel.rank = r.rank, rel.score_combined = r.score
            """,
                qid=query_id,
                rows=[
                    {"section_id": r.result.node_id, "rank": i + 1, "score": r.score}
                    for i, r in enumerate(ranked)
                ],
            )

    def create_answer(
        self, query_id: str, answer_text: str, section_ids: List[str]
    ) -> str:
        aid = str(uuid.uuid4())
        with self.driver.session() as s:
            s.run(
                """
                MATCH (q:Query {query_id: $qid})
                CREATE (a:Answer {answer_id: $aid, text: $txt, created_at: datetime()})
                CREATE (q)-[:ANSWERED_AS {created_at: datetime()}]->(a)
            """,
                qid=query_id,
                aid=aid,
                txt=answer_text,
            )
            s.run(
                """
                UNWIND $ids AS sid
                MATCH (a:Answer {answer_id: $aid})
                MATCH (sec:Section {id: sid})
                MERGE (a)-[:SUPPORTED_BY {created_at: datetime()}]->(sec)
            """,
                aid=aid,
                ids=section_ids[:5],
            )
        return aid

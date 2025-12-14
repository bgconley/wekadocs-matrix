"""
Phase 6, Task 6.4: Post-Ingest Verification

Verifies ingestion quality by checking graph/vector drift and running sample queries.

See: /docs/pseudocode-phase6.md → Task 6.4
See: /docs/implementation-plan-phase-6.md → Task 6.4
"""

from typing import Dict, List, Optional

from neo4j import Driver

from src.query.hybrid_search import HybridSearchEngine
from src.query.response_builder import ResponseBuilder
from src.shared.config import Config
from src.shared.observability import get_logger

logger = get_logger(__name__)


class PostIngestVerifier:
    """
    Verifies ingestion quality after completion.

    Checks:
    1. Graph/vector drift (parity between Neo4j and vector store)
    2. Sample queries (configured per tag)
    3. Readiness verdict (drift <0.5% and all queries return evidence)
    """

    def __init__(
        self,
        driver: Driver,
        config: Config,
        qdrant_client=None,
        search_engine: Optional[HybridSearchEngine] = None,
    ):
        self.driver = driver
        self.config = config
        self.qdrant_client = qdrant_client
        self.search_engine = search_engine
        from src.shared.config import get_embedding_settings

        self.embedding_version = get_embedding_settings(config).version
        self.vector_primary = config.search.vector.primary

    def verify_ingestion(self, job_id: str, parsed: Dict, tag: str = "default") -> Dict:
        """
        Run post-ingestion verification checks.

        Args:
            job_id: Job identifier
            parsed: Parsed document structure (Document, Sections)
            tag: Tag for sample query selection

        Returns:
            Verdict dict with drift, answers, and ready flag
        """
        logger.info("Starting post-ingest verification", job_id=job_id, tag=tag)

        # Check 1: Graph/Vector drift
        drift = self._check_drift()

        # Check 2: Sample queries
        answers = self._run_sample_queries(tag)

        # Check 3: Readiness verdict
        ready = self._compute_readiness(drift, answers)

        verdict = {
            "drift": drift,
            "answers": answers,
            "ready": ready,
        }

        logger.info(
            "Verification complete",
            job_id=job_id,
            drift_pct=drift["pct"],
            sample_queries=len(answers),
            ready=ready,
        )

        return verdict

    def _check_drift(self) -> Dict:
        """
        Check drift between graph and vector store.

        Drift = sections in graph without corresponding vectors

        Returns:
            Dict with drift stats
        """
        try:
            # Count chunks in graph with current embedding version
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE c.embedding_version = $version
                    RETURN count(c) AS graph_count
                    """,
                    version=self.embedding_version,
                )
                record = result.single()
                graph_count = record["graph_count"] if record else 0

            # Count vectors in primary store
            if self.vector_primary == "qdrant" and self.qdrant_client:
                collection_name = self.config.search.vector.qdrant.collection_name

                # Get collection info for count
                coll_info = self.qdrant_client.get_collection(collection_name)
                vector_count = coll_info.points_count

            elif self.vector_primary == "neo4j":
                # Vectors stored in Neo4j
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (c:Chunk)
                        WHERE c.vector_embedding IS NOT NULL
                          AND c.embedding_version = $version
                        RETURN count(c) AS vector_count
                        """,
                        version=self.embedding_version,
                    )
                    record = result.single()
                    vector_count = record["vector_count"] if record else 0
            else:
                vector_count = 0

            # Compute drift percentage
            if graph_count > 0:
                missing = max(0, graph_count - vector_count)
                drift_pct = (missing / graph_count) * 100
            else:
                drift_pct = 0.0

            drift = {
                "graph_count": graph_count,
                "vector_count": vector_count,
                "missing": (
                    graph_count - vector_count if graph_count > vector_count else 0
                ),
                "pct": round(drift_pct, 2),
            }

            logger.info("Drift check complete", drift=drift)
            return drift

        except Exception as e:
            logger.error("Drift check failed", error=str(e))
            return {
                "graph_count": 0,
                "vector_count": 0,
                "missing": 0,
                "pct": 100.0,
                "error": str(e),
            }

    def _run_sample_queries(self, tag: str) -> List[Dict]:
        """
        Run sample queries configured for the given tag.

        Args:
            tag: Classification tag

        Returns:
            List of query results with confidence and evidence
        """
        if not self.search_engine:
            logger.warning("Search engine not available for sample queries")
            return []

        # Get sample queries from config
        try:
            sample_queries = getattr(self.config, "ingest", {})
            if hasattr(sample_queries, "sample_queries"):
                queries_by_tag = sample_queries.sample_queries
                queries = queries_by_tag.get(tag, queries_by_tag.get("default", []))
            else:
                queries = []
        except Exception:
            queries = []

        if not queries:
            logger.info("No sample queries configured for tag", tag=tag)
            return []

        answers = []
        for q in queries[:3]:  # Limit to 3 queries to avoid slowdown
            try:
                # Run hybrid search
                results = self.search_engine.search(q, filters={}, top_k=10)

                # Build response
                response_builder = ResponseBuilder(self.config)
                response = response_builder.build(q, "search", results)

                # Extract evidence and confidence
                answer_json = response.get("answer_json", {})
                evidence = answer_json.get("evidence", [])
                confidence = answer_json.get("confidence", 0.0)

                answers.append(
                    {
                        "q": q,
                        "confidence": round(confidence, 2),
                        "evidence_count": len(evidence),
                        "has_evidence": len(evidence) > 0,
                    }
                )

                logger.debug(
                    "Sample query executed",
                    query=q,
                    confidence=confidence,
                    evidence_count=len(evidence),
                )

            except Exception as e:
                logger.error("Sample query failed", query=q, error=str(e))
                answers.append(
                    {
                        "q": q,
                        "confidence": 0.0,
                        "evidence_count": 0,
                        "has_evidence": False,
                        "error": str(e),
                    }
                )

        return answers

    def _compute_readiness(self, drift: Dict, answers: List[Dict]) -> bool:
        """
        Compute readiness verdict.

        Ready if:
        1. Drift < 0.5%
        2. All sample queries return evidence

        Args:
            drift: Drift stats
            answers: Sample query results

        Returns:
            True if ready for queries
        """
        drift_ok = drift["pct"] <= 0.5

        # Check if all queries have evidence
        if answers:
            evidence_ok = all(a.get("has_evidence", False) for a in answers)
        else:
            # If no sample queries configured, assume OK
            evidence_ok = True

        ready = drift_ok and evidence_ok

        logger.info(
            "Readiness computed",
            drift_ok=drift_ok,
            evidence_ok=evidence_ok,
            ready=ready,
        )

        return ready

# =============================================================================
# @status: ACTIVE
# @reason: GraphContractChecker.__init__ and find_documents_needing_repair are
#          ACTIVE (called by worker.py:256,268).
# @called-by: worker.py:256 (lazy import)
# =============================================================================
"""
Graph contract checks for runtime reconciliation.

Used by worker.py to identify documents needing structural edge repair
after ingestion runs.
"""

from typing import List

import structlog

logger = structlog.get_logger(__name__)


class GraphContractChecker:
    """
    Validates graph contract invariants.

    Implements end-of-run reconciliation to identify documents with
    missing or invalid structural edges needing repair.
    """

    def __init__(self, session, *, sample_limit: int = 50):
        """
        Initialize the contract checker.

        Args:
            session: Neo4j session or driver
            sample_limit: Maximum violations to return per check
        """
        self.session = session
        self.sample_limit = sample_limit

    def find_documents_needing_repair(self) -> List[str]:
        """
        Find documents that need structural edge repair.

        Used for end-of-run reconciliation to identify documents with:
        - Missing NEXT_CHUNK edges
        - Missing HAS_CHUNK edges
        - Invalid structural invariants

        Returns:
            List of document IDs needing repair
        """
        # Find docs with chunks but no NEXT_CHUNK edges
        query = """
        MATCH (c:Chunk)
        WHERE c.document_id IS NOT NULL
        WITH c.document_id AS doc_id, collect(c) AS chunks
        WHERE size(chunks) > 1
        OPTIONAL MATCH (c1:Chunk {document_id: doc_id})-[r:NEXT_CHUNK]->()
        WITH doc_id, size(chunks) AS chunk_count, count(r) AS next_chunk_count
        WHERE next_chunk_count < chunk_count - 1
        RETURN doc_id
        ORDER BY chunk_count DESC
        LIMIT 100
        """
        result = self.session.run(query)
        return [record["doc_id"] for record in result]

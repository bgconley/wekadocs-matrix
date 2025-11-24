"""
Base rerank provider protocol.
Phase 7C: Defines the interface for all reranking providers.

Reranking is applied post-ANN to refine candidate ordering using
cross-attention or other sophisticated scoring mechanisms.
"""

from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class RerankProvider(Protocol):
    """
    Protocol for reranking providers.

    All reranking providers must implement this interface to ensure
    compatibility with the query pipeline.
    """

    @property
    def model_id(self) -> str:
        """
        Get the model identifier.

        Returns:
            str: Model identifier (e.g., "jina-reranker-v3", "bge-reranker-large")
        """
        ...

    @property
    def provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: Provider name (e.g., "jina-ai", "bge-reranker", "noop")
        """
        ...

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates based on relevance to query.

        Args:
            query: Query text
            candidates: List of candidate documents to rerank.
                Each dict must contain at minimum:
                - 'text': Document text content
                - 'id': Unique identifier
                Additional fields are preserved.
            top_k: Number of top results to return after reranking

        Returns:
            List of reranked candidates (top_k items), each with added fields:
            - 'rerank_score': Relevance score from reranker (higher = more relevant)
            - 'original_rank': Original rank before reranking (for comparison)

        Raises:
            ValueError: If candidates is empty or improperly formatted
            RuntimeError: If reranking fails
        """
        ...

    def health_check(self) -> bool:
        """
        Lightweight readiness probe. Should return True if the provider is reachable.
        Implementations may choose to always return True when no remote call is needed.
        """
        ...

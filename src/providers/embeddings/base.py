"""
Base embedding provider protocol.
Pre-Phase 7: Defines the interface for all embedding providers.

This abstraction enables:
1. Clean swapping between providers (SentenceTransformers -> Jina)
2. Consistent API across different embedding models
3. Type safety through Protocol typing
"""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    All embedding providers must implement this interface to ensure
    compatibility with the ingestion and query pipelines.

    The provider returns List[List[float]] for documents and List[float]
    for queries to ensure JSON serialization compatibility (no numpy arrays).
    """

    @property
    def dims(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.

        Returns:
            int: Number of dimensions in the embedding vector
        """
        ...

    @property
    def model_id(self) -> str:
        """
        Get the model identifier.

        Returns:
            str: Model identifier (e.g., "all-MiniLM-L6-v2", "jina-embeddings-v3")
        """
        ...

    @property
    def provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: Provider name (e.g., "sentence-transformers", "jina", "openai")
        """
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Used during ingestion to embed document sections for storage.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors, each as a list of floats

        Raises:
            ValueError: If texts is empty or contains invalid content
            RuntimeError: If embedding generation fails
        """
        ...

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Used during search to embed the user's query for vector similarity.
        Some models (like Jina) use different prompts for queries vs documents.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
        """
        ...

    def validate_dimensions(self, expected_dims: int) -> bool:
        """
        Validate that the provider produces embeddings of expected dimensions.

        Args:
            expected_dims: Expected number of dimensions

        Returns:
            bool: True if dimensions match, False otherwise
        """
        return self.dims == expected_dims

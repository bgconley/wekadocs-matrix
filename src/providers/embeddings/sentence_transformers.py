"""
SentenceTransformers embedding provider implementation.
Pre-Phase 7: Concrete implementation using sentence-transformers library.

This provider will be the default until Phase 7 introduces Jina.
"""

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.shared.config import get_config

logger = logging.getLogger(__name__)


class SentenceTransformersProvider:
    """
    Embedding provider using sentence-transformers library.

    This provider validates dimensions on initialization to ensure
    the model produces vectors of the expected size.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        expected_dims: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the SentenceTransformers provider.

        Args:
            model_name: Model identifier (defaults to config value)
            expected_dims: Expected embedding dimensions (defaults to config value)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)

        Raises:
            ValueError: If actual dimensions don't match expected dimensions
            RuntimeError: If model loading fails
        """
        config = get_config()

        # Use provided values or fall back to config
        self._model_name = model_name or config.embedding.embedding_model
        self._expected_dims = expected_dims or config.embedding.dims
        self._provider_name = "sentence-transformers"

        # Load the model
        try:
            logger.info(f"Loading SentenceTransformers model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device=device)

            # Get actual model config
            self._max_seq_length = self._model.max_seq_length

        except Exception as e:
            logger.error(f"Failed to load model {self._model_name}: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

        # Validate dimensions with a test embedding
        self._validate_dimensions()

        logger.info(
            f"SentenceTransformers provider initialized: "
            f"model={self._model_name}, dims={self._dims}, "
            f"max_seq_length={self._max_seq_length}"
        )

    def _validate_dimensions(self) -> None:
        """
        Validate that the model produces embeddings of expected dimensions.

        Raises:
            ValueError: If dimensions don't match
        """
        # Generate a test embedding
        test_text = "dimension validation test"
        test_embedding = self._model.encode([test_text], convert_to_numpy=True)

        actual_dims = test_embedding.shape[1]

        if actual_dims != self._expected_dims:
            error_msg = (
                f"Dimension mismatch for model {self._model_name}: "
                f"expected {self._expected_dims}, got {actual_dims}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Store validated dimensions
        self._dims = actual_dims
        logger.debug(f"Dimension validation successful: {self._dims} dimensions")

    @property
    def dims(self) -> int:
        """Get embedding dimensions."""
        return self._dims

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors as lists of floats

        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        try:
            # Generate embeddings as numpy array
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,  # Show progress for large batches
                batch_size=get_config().embedding.batch_size,
            )

            # Convert to list of lists for JSON compatibility
            # Using tolist() is more efficient than list comprehension
            embeddings_list = embeddings.tolist()

            # Validate all embeddings have correct dimensions
            for i, emb in enumerate(embeddings_list):
                if len(emb) != self._dims:
                    raise RuntimeError(
                        f"Embedding {i} has wrong dimensions: "
                        f"expected {self._dims}, got {len(emb)}"
                    )

            return embeddings_list

        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise RuntimeError(f"Document embedding failed: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Note: SentenceTransformers uses the same encoding for queries and documents,
        unlike some providers (e.g., Jina) that use different prompts.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query text")

        try:
            # Generate single embedding
            embedding = self._model.encode(
                [text],  # Model expects a list
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Extract first (and only) embedding and convert to list
            embedding_list = embedding[0].tolist()

            # Validate dimensions
            if len(embedding_list) != self._dims:
                raise RuntimeError(
                    f"Query embedding has wrong dimensions: "
                    f"expected {self._dims}, got {len(embedding_list)}"
                )

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")

    def validate_dimensions(self, expected_dims: int) -> bool:
        """
        Validate that the provider produces embeddings of expected dimensions.

        Args:
            expected_dims: Expected number of dimensions

        Returns:
            bool: True if dimensions match, False otherwise
        """
        return self.dims == expected_dims

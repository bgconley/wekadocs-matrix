"""
Provider factory for ENV-selectable embedding and rerank providers.
Phase 7C, Task 7C.1: Factory pattern for docker-compose friendly configuration.

Supported providers:
- Embedding: jina-ai, ollama, sentence-transformers
- Rerank: jina-ai, noop

Configuration via environment variables:
- EMBEDDINGS_PROVIDER: Provider name
- EMBEDDINGS_MODEL: Model identifier
- EMBEDDINGS_DIM: Embedding dimensions
- EMBEDDINGS_TASK: Task type (Jina-specific)
- RERANK_PROVIDER: Reranker provider
- RERANK_MODEL: Reranker model

See .env.example for full configuration options.
"""

import logging
import os
from typing import Optional

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.rerank.base import RerankProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating embedding and rerank providers from environment config.

    This enables docker-compose friendly provider selection without code changes.
    """

    @staticmethod
    def create_embedding_provider(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dims: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingProvider:
        """
        Create embedding provider from ENV config or parameters.

        Priority: explicit parameters > ENV vars > config defaults

        Args:
            provider: Provider name (jina-ai, ollama, sentence-transformers)
            model: Model identifier
            dims: Embedding dimensions
            task: Task type for Jina (retrieval.passage or retrieval.query)
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingProvider instance

        Raises:
            ValueError: If provider unknown or configuration invalid
            RuntimeError: If provider initialization fails

        Environment variables:
            EMBEDDINGS_PROVIDER: Provider name
            EMBEDDINGS_MODEL: Model identifier
            EMBEDDINGS_DIM: Embedding dimensions (integer)
            EMBEDDINGS_TASK: Task type (Jina-specific)
            JINA_API_KEY: Jina API key (required for jina-ai)
            OLLAMA_BASE_URL: Ollama base URL (optional)
        """
        # Get config from parameters or ENV
        from src.shared.config import get_config

        config = get_config()

        provider = provider or os.getenv(
            "EMBEDDINGS_PROVIDER", config.embedding.provider
        )
        model = model or os.getenv("EMBEDDINGS_MODEL", config.embedding.embedding_model)
        dims = dims or int(os.getenv("EMBEDDINGS_DIM", str(config.embedding.dims)))
        task = task or os.getenv("EMBEDDINGS_TASK", config.embedding.task)

        logger.info(
            f"Creating embedding provider: provider={provider}, model={model}, "
            f"dims={dims}, task={task}"
        )

        # Create provider based on type
        if provider == "jina-ai":
            from src.providers.embeddings.jina import JinaEmbeddingProvider

            api_key = kwargs.get("api_key") or os.getenv("JINA_API_KEY")
            return JinaEmbeddingProvider(
                model=model, dims=dims, api_key=api_key, task=task, **kwargs
            )

        elif provider == "ollama":
            from src.providers.embeddings.ollama import OllamaEmbeddingProvider

            # Use pop to remove base_url from kwargs to avoid duplicate argument
            base_url = kwargs.pop("base_url", None) or os.getenv("OLLAMA_BASE_URL")
            return OllamaEmbeddingProvider(
                model=model, dims=dims, base_url=base_url, **kwargs
            )

        elif provider == "sentence-transformers":
            from src.providers.embeddings.sentence_transformers import (
                SentenceTransformersProvider,
            )

            return SentenceTransformersProvider(
                model_name=model, expected_dims=dims, **kwargs
            )

        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                f"Supported: jina-ai, ollama, sentence-transformers"
            )

    @staticmethod
    def create_rerank_provider(
        provider: Optional[str] = None, model: Optional[str] = None, **kwargs
    ) -> RerankProvider:
        """
        Create rerank provider from ENV config or parameters.

        Args:
            provider: Provider name (jina-ai, noop)
            model: Model identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            RerankProvider instance

        Raises:
            ValueError: If provider unknown or configuration invalid
            RuntimeError: If provider initialization fails

        Environment variables:
            RERANK_PROVIDER: Provider name
            RERANK_MODEL: Model identifier
            JINA_API_KEY: Jina API key (required for jina-ai)
        """
        # Get config from parameters or ENV
        provider = provider or os.getenv("RERANK_PROVIDER", "jina-ai")
        model = model or os.getenv("RERANK_MODEL", "jina-reranker-v2-base-multilingual")

        logger.info(f"Creating rerank provider: provider={provider}, model={model}")

        # Create provider based on type
        if provider == "jina-ai":
            from src.providers.rerank.jina import JinaRerankProvider

            api_key = kwargs.get("api_key") or os.getenv("JINA_API_KEY")
            return JinaRerankProvider(model=model, api_key=api_key, **kwargs)

        elif provider == "noop" or provider == "none":
            from src.providers.rerank.noop import NoopReranker

            return NoopReranker()

        else:
            raise ValueError(
                f"Unknown rerank provider: {provider}. Supported: jina-ai, noop"
            )

    @staticmethod
    def get_provider_info(provider: EmbeddingProvider) -> dict:
        """
        Get metadata about a provider instance.

        Args:
            provider: Provider instance

        Returns:
            Dict with provider metadata
        """
        return {
            "provider": provider.provider_name,
            "model": provider.model_id,
            "dims": provider.dims,
        }

    @staticmethod
    def log_provider_config():
        """
        Log current provider configuration at startup.

        This provides visibility into which providers are active.
        """
        from src.shared.config import get_config

        config = get_config()

        # Get actual ENV values (may override config)
        embedding_provider = os.getenv("EMBEDDINGS_PROVIDER", config.embedding.provider)
        embedding_model = os.getenv(
            "EMBEDDINGS_MODEL", config.embedding.embedding_model
        )
        embedding_dims = int(os.getenv("EMBEDDINGS_DIM", str(config.embedding.dims)))

        rerank_provider = os.getenv("RERANK_PROVIDER", "jina-ai")
        rerank_model = os.getenv("RERANK_MODEL", "jina-reranker-v2-base-multilingual")

        logger.info(
            "=" * 60 + "\n"
            "Provider Configuration (Phase 7C)\n"
            "=" * 60 + "\n"
            f"Embedding Provider: {embedding_provider}\n"
            f"  Model: {embedding_model}\n"
            f"  Dimensions: {embedding_dims}\n"
            f"  Task: {config.embedding.task}\n"
            f"\n"
            f"Rerank Provider: {rerank_provider}\n"
            f"  Model: {rerank_model}\n"
            "=" * 60
        )


def create_default_providers() -> tuple[EmbeddingProvider, RerankProvider]:
    """
    Create default embedding and rerank providers from environment.

    This is the primary entry point for initializing providers in
    the application. It reads configuration from environment variables
    and returns ready-to-use provider instances.

    Returns:
        Tuple of (embedding_provider, rerank_provider)

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If provider initialization fails

    Example:
        >>> embedding_provider, rerank_provider = create_default_providers()
        >>> vectors = embedding_provider.embed_documents(["hello world"])
        >>> reranked = rerank_provider.rerank("query", candidates)
    """
    factory = ProviderFactory()

    # Log configuration
    factory.log_provider_config()

    # Create providers
    embedding_provider = factory.create_embedding_provider()
    rerank_provider = factory.create_rerank_provider()

    logger.info(
        f"Providers initialized successfully: "
        f"embedding={embedding_provider.provider_name}/{embedding_provider.model_id}, "
        f"rerank={rerank_provider.provider_name}/{rerank_provider.model_id}"
    )

    return embedding_provider, rerank_provider

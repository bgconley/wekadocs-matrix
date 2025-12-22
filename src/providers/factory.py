"""
Provider factory for ENV-selectable embedding and rerank providers.
Phase 7C, Task 7C.1: Factory pattern for docker-compose friendly configuration.

Supported providers:
- Embedding: jina-ai, sentence-transformers, bge-m3-service
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
from dataclasses import replace
from typing import Callable, Dict, Optional

from src.providers.embeddings.base import EmbeddingProvider
from src.providers.rerank.base import RerankProvider
from src.providers.settings import EmbeddingSettings as ProviderEmbeddingSettings
from src.providers.settings import (
    build_embedding_telemetry,
)
from src.shared.observability.metrics import embedding_provider_info

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating embedding and rerank providers from environment config.

    This enables docker-compose friendly provider selection without code changes.
    """

    _EMBEDDING_PROVIDER_CREATORS: Dict[
        str, Callable[[ProviderEmbeddingSettings], EmbeddingProvider]
    ] = {}
    _EMBEDDING_PROVIDER_ALIASES = {
        "bge-m3": "bge-m3-service",
        "bge_m3": "bge-m3-service",
        "bge_m3_service": "bge-m3-service",
        "st_minilm": "sentence-transformers",
        "st-minilm": "sentence-transformers",
        "sentence_transformers": "sentence-transformers",
        "huggingface": "sentence-transformers",
        "hf": "sentence-transformers",
    }

    @classmethod
    def create_embedding_provider(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dims: Optional[int] = None,
        task: Optional[str] = None,
        settings: Optional[ProviderEmbeddingSettings] = None,
        **kwargs,
    ) -> EmbeddingProvider:
        """
        Create embedding provider from profile-driven settings, allowing legacy overrides.
        """
        from src.shared.config import get_embedding_settings

        settings = settings or get_embedding_settings()
        if any([provider, model, dims, task]):
            settings = cls._apply_legacy_overrides(
                settings, provider, model, dims, task
            )

        provider_key = cls._normalize_provider(settings.provider)
        creator = cls._EMBEDDING_PROVIDER_CREATORS.get(provider_key)
        if not creator:
            raise ValueError(
                f"Unknown embedding provider: {settings.provider}. "
                f"Supported: {sorted(cls._EMBEDDING_PROVIDER_CREATORS.keys())}"
            )

        telemetry = build_embedding_telemetry(settings)
        logger.info(
            "Creating embedding provider",
            extra={**telemetry, "task": settings.task},
        )
        embedding_provider_info.info(telemetry)

        return creator(settings, **kwargs)

    @classmethod
    def _normalize_provider(cls, provider: Optional[str]) -> str:
        base = (provider or "").strip().lower()
        return cls._EMBEDDING_PROVIDER_ALIASES.get(base, base)

    @staticmethod
    def _apply_legacy_overrides(
        settings: ProviderEmbeddingSettings,
        provider_override: Optional[str],
        model_override: Optional[str],
        dims_override: Optional[int],
        task_override: Optional[str],
    ) -> ProviderEmbeddingSettings:
        overrides = {}
        if provider_override and provider_override != settings.provider:
            logger.warning(
                "create_embedding_provider(provider=...) overrides profile '%s'",
                settings.profile,
            )
            overrides["provider"] = provider_override
        if model_override and model_override != settings.model_id:
            logger.warning(
                "create_embedding_provider(model=...) overrides profile '%s'",
                settings.profile,
            )
            overrides["model_id"] = model_override
        if dims_override and dims_override != settings.dims:
            logger.warning(
                "create_embedding_provider(dims=...) overrides profile '%s'",
                settings.profile,
            )
            overrides["dims"] = dims_override
        if task_override and task_override != settings.task:
            logger.warning(
                "create_embedding_provider(task=...) overrides profile '%s'",
                settings.profile,
            )
            overrides["task"] = task_override
        if not overrides:
            return settings
        return replace(settings, **overrides)

    @staticmethod
    def _create_jina_embedding_provider(
        settings: ProviderEmbeddingSettings, **kwargs
    ) -> EmbeddingProvider:
        from src.providers.embeddings.jina import JinaEmbeddingProvider

        api_key = kwargs.get("api_key") or os.getenv("JINA_API_KEY")
        return JinaEmbeddingProvider(
            model=settings.model_id,
            dims=settings.dims,
            api_key=api_key,
            task=settings.task,
            **kwargs,
        )

    @staticmethod
    def _create_sentence_transformers_provider(
        settings: ProviderEmbeddingSettings, **kwargs
    ) -> EmbeddingProvider:
        from src.providers.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        return SentenceTransformersProvider(
            model_name=settings.model_id,
            expected_dims=settings.dims,
            **kwargs,
        )

    @staticmethod
    def _create_bge_m3_service_provider(
        settings: ProviderEmbeddingSettings, **kwargs
    ) -> EmbeddingProvider:
        from src.providers.embeddings.bge_m3_service import BGEM3ServiceProvider

        return BGEM3ServiceProvider(settings=settings, **kwargs)

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
        from src.shared.config import get_config

        def _normalize_provider(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            normalized = value.strip().lower()
            if not normalized:
                return None
            if normalized in {"jina", "jina-ai"}:
                return "jina-ai"
            if normalized in {"none", "disabled"}:
                return "noop"
            return normalized

        def _normalize_model(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            normalized = value.strip()
            return normalized or None

        config = get_config()
        reranker_cfg = getattr(
            getattr(getattr(config, "search", None), "hybrid", None), "reranker", None
        )
        config_provider = (
            _normalize_provider(getattr(reranker_cfg, "provider", None))
            if reranker_cfg
            else None
        )
        config_model = (
            _normalize_model(getattr(reranker_cfg, "model", None))
            if reranker_cfg
            else None
        )

        env_provider = _normalize_provider(os.getenv("RERANK_PROVIDER"))
        env_model = _normalize_model(os.getenv("RERANK_MODEL"))

        provider = (
            _normalize_provider(provider)
            or env_provider
            or config_provider
            or "jina-ai"
        )
        model = (
            _normalize_model(model) or env_model or config_model or "jina-reranker-v3"
        )

        logger.info(f"Creating rerank provider: provider={provider}, model={model}")

        # Create provider based on type
        if provider == "jina-ai":
            from src.providers.rerank.jina import JinaRerankProvider

            api_key = kwargs.get("api_key") or os.getenv("JINA_API_KEY")
            return JinaRerankProvider(model=model, api_key=api_key, **kwargs)

        elif provider in {"bge-reranker-service", "bge-reranker"}:
            from src.providers.rerank.local_bge_service import (
                BGERerankerServiceProvider,
            )

            base_url = kwargs.get("base_url") or os.getenv(
                "RERANKER_BASE_URL", "http://qwen3-reranker-lambda:9003"
            )
            timeout = kwargs.get("timeout") or float(
                os.getenv("RERANKER_TIMEOUT_SECONDS", "60")
            )
            return BGERerankerServiceProvider(
                model=model, base_url=base_url, timeout=timeout
            )

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
        # Get actual ENV values (may override config)
        from src.shared.config import get_embedding_settings, get_settings

        embedding_settings = get_embedding_settings()
        runtime_settings = get_settings()

        rerank_provider = os.getenv("RERANK_PROVIDER", "jina-ai")
        rerank_model = os.getenv("RERANK_MODEL", "jina-reranker-v3")

        telemetry = build_embedding_telemetry(embedding_settings)
        telemetry["embedding_namespace_mode"] = (
            runtime_settings.embedding_namespace_mode or "none"
        )
        telemetry["embedding_strict_mode"] = (
            "true" if runtime_settings.embedding_strict_mode else "false"
        )
        telemetry["embedding_profile_swappable"] = (
            "true" if runtime_settings.embedding_profile_swappable else "false"
        )
        telemetry["embedding_profile_experiment"] = (
            runtime_settings.embedding_profile_experiment or ""
        )
        embedding_provider_info.info(telemetry)

        logger.info(
            "=" * 60 + "\n"
            "Provider Configuration (Phase 7C)\n"
            "=" * 60 + "\n"
            f"Embedding Profile: {embedding_settings.profile}\n"
            f"  Provider: {embedding_settings.provider}\n"
            f"  Model: {embedding_settings.model_id}\n"
            f"  Dimensions: {embedding_settings.dims}\n"
            f"  Task: {embedding_settings.task}\n"
            f"  Strict Mode: {runtime_settings.embedding_strict_mode}\n"
            f"  Namespace Mode: {runtime_settings.embedding_namespace_mode}\n"
            f"  Profile Swappable: {runtime_settings.embedding_profile_swappable}\n"
            f"  Profile Experiment: {runtime_settings.embedding_profile_experiment}\n"
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


def create_embedding_provider(*args, **kwargs) -> EmbeddingProvider:
    """
    Module-level convenience wrapper for ProviderFactory.create_embedding_provider.
    """
    return ProviderFactory.create_embedding_provider(*args, **kwargs)


def create_rerank_provider(*args, **kwargs) -> RerankProvider:
    """
    Module-level convenience wrapper for ProviderFactory.create_rerank_provider.
    """
    return ProviderFactory.create_rerank_provider(*args, **kwargs)


ProviderFactory._EMBEDDING_PROVIDER_CREATORS = {
    "jina-ai": ProviderFactory._create_jina_embedding_provider,
    "sentence-transformers": ProviderFactory._create_sentence_transformers_provider,
    "bge-m3-service": ProviderFactory._create_bge_m3_service_provider,
}

# Mapping is appended later for rerank providers.

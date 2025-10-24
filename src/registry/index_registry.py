"""
Index registry with dimension enforcement.
Phase 7C, Task 7C.2: Prevent cross-dimension writes/queries.

The IndexRegistry maintains a catalog of vector indices and enforces
dimension compatibility between providers and indices. This prevents
384-D vectors from being written to 1024-D indices and vice versa.
"""

import logging
from typing import Dict, Optional

from src.providers.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class IndexRegistry:
    """
    Registry of vector indices with dimension enforcement.

    Maintains metadata about all vector indices in the system and
    validates that providers produce vectors compatible with target indices.

    During Phase 7C migration, we maintain two indices:
    - weka_sections: 384-D (legacy)
    - weka_sections_v2: 1024-D (new, default after cutover)
    """

    def __init__(self):
        """Initialize the index registry."""
        self._indices: Dict[str, Dict] = {}
        self._active_index_name: Optional[str] = None

        logger.info("IndexRegistry initialized")

    def register_index(
        self,
        name: str,
        dims: int,
        provider: str,
        model: str,
        version: str,
        collection_name: Optional[str] = None,
        is_active: bool = False,
    ) -> None:
        """
        Register a vector index.

        Args:
            name: Index identifier (e.g., "weka_sections_v2")
            dims: Vector dimensions
            provider: Provider name (e.g., "jina-ai", "ollama")
            model: Model identifier (e.g., "jina-embeddings-v4")
            version: Model version for provenance
            collection_name: Qdrant collection name (defaults to name)
            is_active: Whether this is the active index for queries

        Raises:
            ValueError: If index already registered or dims invalid
        """
        if name in self._indices:
            raise ValueError(f"Index '{name}' already registered")

        if dims <= 0:
            raise ValueError(f"Dimensions must be positive, got {dims}")

        self._indices[name] = {
            "name": name,
            "dims": dims,
            "provider": provider,
            "model": model,
            "version": version,
            "collection_name": collection_name or name,
            "is_active": is_active,
        }

        if is_active:
            self._active_index_name = name

        logger.info(
            f"Registered index: {name} ({dims}-D, {provider}/{model}, "
            f"active={is_active})"
        )

    def get_index(self, name: str) -> Dict:
        """
        Get index metadata by name.

        Args:
            name: Index name

        Returns:
            Index metadata dict

        Raises:
            KeyError: If index not found
        """
        if name not in self._indices:
            raise KeyError(
                f"Index '{name}' not found. Registered indices: {list(self._indices.keys())}"
            )

        return self._indices[name].copy()

    def get_active_index(self) -> Dict:
        """
        Get the currently active index for queries.

        Returns:
            Active index metadata

        Raises:
            RuntimeError: If no active index set
        """
        if not self._active_index_name:
            raise RuntimeError(
                "No active index set. Call register_index() with is_active=True "
                "or set_active_index()."
            )

        return self.get_index(self._active_index_name)

    def set_active_index(self, name: str) -> None:
        """
        Set the active index for queries.

        This is used during cutover to switch from legacy to new index.

        Args:
            name: Index name to activate

        Raises:
            KeyError: If index not found
        """
        if name not in self._indices:
            raise KeyError(f"Cannot activate unknown index: {name}")

        # Deactivate current active index
        if self._active_index_name:
            self._indices[self._active_index_name]["is_active"] = False

        # Activate new index
        self._indices[name]["is_active"] = True
        self._active_index_name = name

        logger.info(f"Active index changed to: {name}")

    def enforce_compatibility(
        self, index_name: str, provider: EmbeddingProvider
    ) -> None:
        """
        Enforce provider/dimension compatibility with index.

        This is the critical safety check that prevents dimension mismatches.

        Args:
            index_name: Target index name
            provider: Embedding provider to validate

        Raises:
            ValueError: If dimensions don't match
            KeyError: If index not found
        """
        index = self.get_index(index_name)

        # Check dimensions
        if index["dims"] != provider.dims:
            raise ValueError(
                f"Dimension mismatch for index '{index_name}': "
                f"index expects {index['dims']}-D vectors, "
                f"provider '{provider.provider_name}' generates {provider.dims}-D vectors. "
                f"This write would corrupt the index!"
            )

        # Warn if model mismatch (not a hard error, but suspicious)
        if index["model"] != provider.model_id:
            logger.warning(
                f"Model mismatch for index '{index_name}': "
                f"index uses {index['model']}, "
                f"provider uses {provider.model_id}. "
                f"This may indicate configuration drift."
            )

    def list_indices(self) -> list[Dict]:
        """
        List all registered indices.

        Returns:
            List of index metadata dicts
        """
        return [idx.copy() for idx in self._indices.values()]

    def get_index_for_provider(self, provider: EmbeddingProvider) -> Optional[Dict]:
        """
        Find compatible index for a provider based on dimensions.

        Args:
            provider: Embedding provider

        Returns:
            Compatible index metadata, or None if no match
        """
        for index in self._indices.values():
            if index["dims"] == provider.dims:
                return index.copy()

        return None


def create_default_registry() -> IndexRegistry:
    """
    Create and populate default index registry.

    This registers the standard indices used in the system:
    - weka_sections_v2: 1024-D (Jina v4, active by default)
    - weka_sections: 384-D (legacy, for rollback)

    Returns:
        Initialized IndexRegistry

    Example:
        >>> registry = create_default_registry()
        >>> active = registry.get_active_index()
        >>> print(active['dims'])  # 1024
    """
    import os

    from src.shared.config import get_config

    config = get_config()
    registry = IndexRegistry()

    # Get current provider config
    provider_name = os.getenv("EMBEDDINGS_PROVIDER", config.embedding.provider)
    model_name = os.getenv("EMBEDDINGS_MODEL", config.embedding.embedding_model)
    dims = int(os.getenv("EMBEDDINGS_DIM", str(config.embedding.dims)))
    version = config.embedding.version

    # Register new 1024-D index (active)
    registry.register_index(
        name="weka_sections_v2",
        dims=1024,
        provider=provider_name,
        model=model_name,
        version=version,
        collection_name="weka_sections_v2",
        is_active=True,  # Default to new index
    )

    # Register legacy 384-D index (for rollback)
    registry.register_index(
        name="weka_sections",
        dims=384,
        provider="sentence-transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
        version="miniLM-L6-v2-2024-01-01",
        collection_name="weka_sections",
        is_active=False,
    )

    logger.info(f"Default registry created: active index = weka_sections_v2 ({dims}-D)")

    return registry

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.providers.settings import EmbeddingSettings


@dataclass(frozen=True)
class QdrantSchemaPlan:
    """Represents the desired Qdrant schema for a given embedding profile."""

    vectors_config: Dict[str, VectorParams]
    sparse_vectors_config: Dict[str, SparseVectorParams] = field(default_factory=dict)
    payload_indexes: List[Tuple[str, PayloadSchemaType]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def build_qdrant_schema(
    settings: EmbeddingSettings,
    *,
    include_entity: bool = False,
    enable_sparse: Optional[bool] = None,
    enable_colbert: Optional[bool] = None,
) -> QdrantSchemaPlan:
    """Return the desired Qdrant schema for the embedding profile."""

    dims = settings.dims or 0
    vectors_config: Dict[str, VectorParams] = {
        "content": VectorParams(size=dims, distance=Distance.COSINE),
        "title": VectorParams(size=dims, distance=Distance.COSINE),
    }
    if include_entity:
        vectors_config["entity"] = VectorParams(size=dims, distance=Distance.COSINE)

    capabilities = settings.capabilities
    use_sparse = (
        enable_sparse if enable_sparse is not None else capabilities.supports_sparse
    )
    use_colbert = (
        enable_colbert if enable_colbert is not None else capabilities.supports_colbert
    )

    sparse_vectors_config: Dict[str, SparseVectorParams] = {}
    notes: List[str] = []

    if use_sparse:
        sparse_vectors_config["text-sparse"] = SparseVectorParams(
            index=SparseIndexParams(on_disk=True)
        )
    else:
        notes.append("Sparse vector support disabled for current profile or config.")

    if use_colbert:
        vectors_config["late-interaction"] = VectorParams(
            size=dims,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=HnswConfigDiff(m=0),
        )
    else:
        notes.append(
            "ColBERT multivector support disabled for current profile or config."
        )

    payload_indexes: List[Tuple[str, PayloadSchemaType]] = [
        ("id", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("parent_section_id", PayloadSchemaType.KEYWORD),
        ("parent_section_original_id", PayloadSchemaType.KEYWORD),
        ("order", PayloadSchemaType.INTEGER),
        ("heading", PayloadSchemaType.TEXT),
        ("updated_at", PayloadSchemaType.INTEGER),
        ("doc_tag", PayloadSchemaType.KEYWORD),
        ("snapshot_scope", PayloadSchemaType.KEYWORD),
        ("is_microdoc", PayloadSchemaType.BOOL),
        ("token_count", PayloadSchemaType.INTEGER),
        ("tenant", PayloadSchemaType.KEYWORD),
        ("lang", PayloadSchemaType.KEYWORD),
        ("version", PayloadSchemaType.KEYWORD),
        ("source_path", PayloadSchemaType.KEYWORD),
        ("embedding_version", PayloadSchemaType.KEYWORD),
        ("embedding_provider", PayloadSchemaType.KEYWORD),
        ("embedding_dimensions", PayloadSchemaType.INTEGER),
        ("text_hash", PayloadSchemaType.KEYWORD),
        ("shingle_hash", PayloadSchemaType.KEYWORD),
    ]

    return QdrantSchemaPlan(
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
        payload_indexes=payload_indexes,
        notes=notes,
    )


def validate_qdrant_schema(
    client,
    collection_name: str,
    settings: EmbeddingSettings,
    *,
    require_sparse: Optional[bool] = None,
    require_colbert: Optional[bool] = None,
) -> None:
    """
    Validate Qdrant collection against embedding settings and expected capabilities.

    Raises RuntimeError on mismatch.
    """
    require_sparse = (
        settings.capabilities.supports_sparse
        if require_sparse is None
        else require_sparse
    )
    require_colbert = (
        settings.capabilities.supports_colbert
        if require_colbert is None
        else require_colbert
    )

    try:
        info = client.get_collection(collection_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' not found or unreachable: {exc}"
        ) from exc

    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors

    # Validate dense vector dims for primary content vector
    primary_vec = None
    if isinstance(vectors, dict):
        primary_vec = vectors.get("content")
    else:
        primary_vec = vectors
    if primary_vec is None or getattr(primary_vec, "size", None) != settings.dims:
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' vector dims mismatch: "
            f"expected content size {settings.dims}, got {getattr(primary_vec, 'size', None)}"
        )

    if require_sparse:
        if not sparse or "text-sparse" not in sparse:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' missing required sparse vector 'text-sparse'"
            )

    if require_colbert:
        if isinstance(vectors, dict):
            colbert_vec = vectors.get("late-interaction")
            if not colbert_vec or getattr(colbert_vec, "size", None) != settings.dims:
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' missing ColBERT multivector "
                    f"or size mismatch (expected {settings.dims})"
                )
        else:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' missing ColBERT multivector"
            )

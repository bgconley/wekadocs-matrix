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

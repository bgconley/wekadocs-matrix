from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

logger = logging.getLogger(__name__)


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
    enable_doc_title_sparse: Optional[bool] = None,
) -> QdrantSchemaPlan:
    """Return the desired Qdrant schema for the embedding profile."""

    dims = settings.dims or 0
    vectors_config: Dict[str, VectorParams] = {
        "content": VectorParams(size=dims, distance=Distance.COSINE),
        "title": VectorParams(size=dims, distance=Distance.COSINE),
        "doc_title": VectorParams(size=dims, distance=Distance.COSINE),
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
        # doc_title-sparse: BM25-style lexical matching for document titles
        # Enables exact term matching for title-based queries (o3 recommendation)
        # Only add if enable_doc_title_sparse is True (default) or not explicitly disabled
        use_doc_title_sparse = (
            enable_doc_title_sparse if enable_doc_title_sparse is not None else True
        )
        if use_doc_title_sparse:
            sparse_vectors_config["doc_title-sparse"] = SparseVectorParams(
                index=SparseIndexParams(on_disk=True)
            )
        else:
            notes.append(
                "doc_title-sparse vector disabled via enable_doc_title_sparse=False."
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
        ("node_id", PayloadSchemaType.KEYWORD),
        ("kg_id", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("doc_title", PayloadSchemaType.TEXT),
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
    include_entity: Optional[bool] = None,
    require_doc_title_sparse: Optional[bool] = None,
    require_payload_fields: Optional[Sequence[str]] = None,
    strict: bool = False,
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
    include_entity = bool(include_entity)
    required_payload_fields = list(require_payload_fields or [])
    # Default require_doc_title_sparse to True when sparse is required (backward compat)
    effective_require_doc_title_sparse = (
        require_doc_title_sparse if require_doc_title_sparse is not None else True
    )
    expected_plan = build_qdrant_schema(
        settings,
        include_entity=include_entity,
        enable_sparse=require_sparse,
        enable_colbert=require_colbert,
        enable_doc_title_sparse=effective_require_doc_title_sparse,
    )

    try:
        info = client.get_collection(collection_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' not found or unreachable: {exc}"
        ) from exc

    params = getattr(info, "config", None)
    params = getattr(params, "params", params)
    vectors = getattr(params, "vectors", None)
    sparse = getattr(params, "sparse_vectors", None)
    payload_schema = (
        getattr(params, "payload_schema", None)
        or getattr(info, "payload_schema", None)
        or {}
    )

    # Validate dense vector dims for primary content vector and companions
    if isinstance(vectors, dict):
        for vec_name, vec_params in expected_plan.vectors_config.items():
            observed = vectors.get(vec_name)
            if observed is None:
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' missing required vector '{vec_name}'"
                )
            if getattr(observed, "size", None) != getattr(vec_params, "size", None):
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' vector dims mismatch for '{vec_name}': "
                    f"expected {getattr(vec_params, 'size', None)}, "
                    f"got {getattr(observed, 'size', None)}"
                )
    else:
        primary_vec = vectors
        if primary_vec is None or getattr(primary_vec, "size", None) != settings.dims:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' vector dims mismatch: "
                f"expected content size {settings.dims}, got {getattr(primary_vec, 'size', None)}"
            )
        if strict and len(expected_plan.vectors_config) > 1:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' missing named vector map for expected vectors"
            )

    if require_sparse:
        if not sparse or "text-sparse" not in sparse:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' missing required sparse vector 'text-sparse'"
            )
        # Validate doc_title-sparse presence when explicitly required
        if effective_require_doc_title_sparse:
            if not sparse or "doc_title-sparse" not in sparse:
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' missing required sparse vector "
                    f"'doc_title-sparse'. Set require_doc_title_sparse=False to skip this check "
                    f"or recreate the collection with doc_title-sparse support."
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

    if required_payload_fields:
        missing_payload = [
            f for f in required_payload_fields if f not in payload_schema
        ]
        if missing_payload and strict:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' missing required payload fields: {missing_payload}"
            )
        if missing_payload and not strict:
            logger.warning(
                "Qdrant collection missing optional payload indexes",
                extra={
                    "collection": collection_name,
                    "missing_payload_fields": missing_payload,
                },
            )

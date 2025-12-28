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
from src.shared.config import EmbeddingPlan

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
    embedding_plan: Optional[EmbeddingPlan] = None,
    include_entity: bool = False,  # DEPRECATED: Dense entity vector removed (was broken)
    enable_sparse: Optional[bool] = None,
    enable_colbert: Optional[bool] = None,
    enable_doc_title_sparse: Optional[bool] = None,
    enable_title_sparse: Optional[bool] = None,  # NEW: Lexical heading matching
    enable_entity_sparse: Optional[bool] = None,  # NEW: Lexical entity name matching
) -> QdrantSchemaPlan:
    """Return the desired Qdrant schema for the embedding profile."""

    dims = settings.dims or 0
    colbert_dims = dims
    if embedding_plan and embedding_plan.colbert:
        colbert_dims = embedding_plan.colbert.profile.dims
    vectors_config: Dict[str, VectorParams] = {
        "content": VectorParams(size=dims, distance=Distance.COSINE),
        "title": VectorParams(size=dims, distance=Distance.COSINE),
        "doc_title": VectorParams(size=dims, distance=Distance.COSINE),
    }
    # REMOVED: Dense entity vector - it was broken (duplicated content embedding)
    # and has been replaced by entity-sparse for lexical entity name matching.
    # The include_entity parameter is now a no-op for backward compatibility.
    if include_entity:
        logger.warning(
            "include_entity=True is deprecated. Dense entity vector has been removed "
            "and replaced by entity-sparse. This parameter will be ignored."
        )

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

        # title-sparse: Lexical matching for section headings
        # Enables exact term matching for heading-based queries
        # Example: "NFS mount" matches section titled "Mounting NFS Filesystems"
        use_title_sparse = (
            enable_title_sparse if enable_title_sparse is not None else True
        )
        if use_title_sparse:
            sparse_vectors_config["title-sparse"] = SparseVectorParams(
                index=SparseIndexParams(on_disk=True)
            )
        else:
            notes.append("title-sparse vector disabled via enable_title_sparse=False.")

        # entity-sparse: Lexical matching for entity names mentioned in chunks
        # Enables exact term matching for entity-based queries
        # Example: "WEKA" or "NFS" matches chunks mentioning those entities
        use_entity_sparse = (
            enable_entity_sparse if enable_entity_sparse is not None else True
        )
        if use_entity_sparse:
            sparse_vectors_config["entity-sparse"] = SparseVectorParams(
                index=SparseIndexParams(on_disk=True)
            )
        else:
            notes.append(
                "entity-sparse vector disabled via enable_entity_sparse=False."
            )
    else:
        notes.append("Sparse vector support disabled for current profile or config.")

    if use_colbert:
        vectors_config["late-interaction"] = VectorParams(
            size=colbert_dims,
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
        # === GLiNER Entity Metadata Indexes (Phase 3) ===
        # Enable efficient filtering and post-retrieval boosting for entity-enriched chunks
        # Added as part of GLiNER NER integration for entity-aware hybrid search
        (
            "entity_metadata.entity_types",
            PayloadSchemaType.KEYWORD,
        ),  # Entity type labels
        (
            "entity_metadata.entity_values",
            PayloadSchemaType.KEYWORD,
        ),  # Raw entity texts
        (
            "entity_metadata.entity_values_normalized",
            PayloadSchemaType.KEYWORD,
        ),  # Lowercased
        (
            "entity_metadata.entity_count",
            PayloadSchemaType.INTEGER,
        ),  # Entity count per chunk
        # === Phase 2: markdown-it-py Structural Metadata Indexes ===
        # Enable query-time filtering by structural characteristics
        # Added as part of markdown-it-py integration for query-type adaptive retrieval
        (
            "has_code",
            PayloadSchemaType.BOOL,
        ),  # Filter for code-heavy chunks (CLI queries)
        (
            "has_table",
            PayloadSchemaType.BOOL,
        ),  # Filter for table-heavy chunks (reference)
        ("code_ratio", PayloadSchemaType.FLOAT),  # Range filter on code density
        ("parent_path", PayloadSchemaType.TEXT),  # Text search on heading hierarchy
        (
            "line_start",
            PayloadSchemaType.INTEGER,
        ),  # Source line for citation correlation
        # === Phase 5: Enhanced Structural Retrieval Indexes ===
        # Enable query-type adaptive filtering and structural boosting
        (
            "line_end",
            PayloadSchemaType.INTEGER,
        ),  # Source line end for complete range
        (
            "parent_path_depth",
            PayloadSchemaType.INTEGER,
        ),  # Nesting level (0=root) for depth filtering
        (
            "block_type",
            PayloadSchemaType.KEYWORD,
        ),  # Dominant block type: "paragraph", "code", "table", "mixed"
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
    embedding_plan: Optional[EmbeddingPlan] = None,
    require_sparse: Optional[bool] = None,
    require_colbert: Optional[bool] = None,
    include_entity: Optional[bool] = None,  # DEPRECATED: No longer validated
    require_doc_title_sparse: Optional[bool] = None,
    require_title_sparse: Optional[bool] = None,  # NEW: Validate title-sparse
    require_entity_sparse: Optional[bool] = None,  # NEW: Validate entity-sparse
    require_entity_metadata_indexes: Optional[bool] = None,  # Phase 3: GLiNER indexes
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
    include_entity = bool(include_entity)  # Deprecated, ignored
    required_payload_fields = list(require_payload_fields or [])
    # Default require_doc_title_sparse to True when sparse is required (backward compat)
    effective_require_doc_title_sparse = (
        require_doc_title_sparse if require_doc_title_sparse is not None else True
    )
    expected_plan = build_qdrant_schema(
        settings,
        embedding_plan=embedding_plan,
        include_entity=include_entity,  # Deprecated, passed for compatibility
        enable_sparse=require_sparse,
        enable_colbert=require_colbert,
        enable_doc_title_sparse=effective_require_doc_title_sparse,
        enable_title_sparse=require_title_sparse,  # NEW
        enable_entity_sparse=require_entity_sparse,  # NEW
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
        # Validate title-sparse presence when required
        effective_require_title_sparse = (
            require_title_sparse if require_title_sparse is not None else require_sparse
        )
        if effective_require_title_sparse:
            if not sparse or "title-sparse" not in sparse:
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' missing required sparse vector "
                    f"'title-sparse'. Set require_title_sparse=False to skip this check "
                    f"or recreate the collection with title-sparse support."
                )
        # Validate entity-sparse presence when required
        effective_require_entity_sparse = (
            require_entity_sparse
            if require_entity_sparse is not None
            else require_sparse
        )
        if effective_require_entity_sparse:
            if not sparse or "entity-sparse" not in sparse:
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' missing required sparse vector "
                    f"'entity-sparse'. Set require_entity_sparse=False to skip this check "
                    f"or recreate the collection with entity-sparse support."
                )

    # Validate entity metadata payload indexes (Phase 3: GLiNER)
    if require_entity_metadata_indexes:
        entity_index_fields = [
            "entity_metadata.entity_types",
            "entity_metadata.entity_values",
            "entity_metadata.entity_values_normalized",
            "entity_metadata.entity_count",
        ]
        missing_entity_indexes = [
            f for f in entity_index_fields if f not in payload_schema
        ]
        if missing_entity_indexes:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' missing required entity metadata "
                f"payload indexes: {missing_entity_indexes}. These indexes are required for "
                f"GLiNER entity-aware hybrid search. Run the Phase 3 index creation script "
                f"or set require_entity_metadata_indexes=False to skip this check."
            )

    if require_colbert:
        if isinstance(vectors, dict):
            colbert_vec = vectors.get("late-interaction")
            expected_colbert_dims = getattr(
                expected_plan.vectors_config.get("late-interaction"), "size", None
            )
            if (
                not colbert_vec
                or getattr(colbert_vec, "size", None) != expected_colbert_dims
            ):
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' missing ColBERT multivector "
                    f"or size mismatch (expected {expected_colbert_dims})"
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

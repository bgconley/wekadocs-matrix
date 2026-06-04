# =============================================================================
# @status: CLEANED (DEAD methods removed)
# @reason: GraphBuilder.__init__, ensure_embedder, _build_section_text_for_embedding,
#          and _build_title_text_for_embedding are ACTIVE (used by atomic.py as a
#          settings container and text utility). All other methods (39 total) were
#          DEAD — their logic lives in atomic.py's saga-coordinated write path.
# @called-by: atomic.py:1055 (GraphBuilder instantiation)
# =============================================================================
# Implements Phase 3, Task 3.3 (Graph construction with embeddings)
# See: /docs/spec.md §3 (Data model, IDs, vectors)
# See: /docs/implementation-plan.md → Task 3.3
# See: /docs/pseudocode-reference.md → Task 3.3
# Pre-Phase 7 B3: Modified to use embedding provider abstraction

import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import redis
from neo4j import Driver
from qdrant_client.http.models import PayloadSchemaType, SparseVector
from qdrant_client.models import (
    HnswConfigDiff,
    OptimizersConfigDiff,
    SparseVectorParams,
    VectorParams,
)

from src.ingestion.chunk_assembler import get_chunk_assembler

# Phase 7E-4: Monitoring imports
from src.monitoring.metrics import MetricsCollector
from src.monitoring.slos import check_slos_and_log
from src.providers.factory import ProviderFactory
from src.shared.chunk_utils import (
    create_chunk_metadata,
    validate_chunk_schema,
)
from src.shared.config import (
    Config,
    get_embedding_plan,
    get_embedding_settings,
    get_expected_namespace_suffix,
    get_settings,
)
from src.shared.embedding_fields import (
    canonicalize_embedding_metadata,
    ensure_no_embedding_model_in_payload,
    validate_embedding_metadata,
)
from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    chunk_token_distribution,
    chunks_created_total,
    chunks_oversized_total,
    embedding_profile_guard_events_total,
    ingestion_duration_seconds,
)
from src.shared.qdrant_schema import build_qdrant_schema
from src.shared.schema import ensure_schema_version

logger = get_logger(__name__)
GRAPH_BUILDER_INIT_LOGGED = False

# C.1.1: Generic headings that should NOT become concept entities
GENERIC_HEADING_BLACKLIST = frozenset(
    {
        "overview",
        "introduction",
        "summary",
        "description",
        "details",
        "notes",
        "note",
        "example",
        "examples",
        "usage",
        "syntax",
        "parameters",
        "options",
        "arguments",
        "returns",
        "return value",
        "see also",
        "related",
        "prerequisites",
        "requirements",
        "warning",
        "warnings",
        "caution",
        "important",
        "tip",
        "tips",
        "troubleshooting",
        "faq",
        "appendix",
        "reference",
        "references",
        "contents",
        "table of contents",
        "index",
        "glossary",
        "about",
        "getting started",
        "quick start",
        "installation",
        "setup",
        "configuration",
        "conclusion",
        "next steps",
    }
)


class GraphBuilder:
    """Builds graph from parsed documents, sections, and entities."""

    def __init__(
        self,
        driver: Driver,
        config: Config,
        qdrant_client=None,
        strict_mode: Optional[bool] = None,
    ):
        self.driver = driver
        self.config = config

        # Wrap qdrant_client with CompatQdrantClient if it's not already wrapped
        if qdrant_client is not None:
            from src.shared.connections import CompatQdrantClient

            if not isinstance(qdrant_client, CompatQdrantClient):
                from qdrant_client import QdrantClient

                if isinstance(qdrant_client, QdrantClient):
                    self.qdrant_client = CompatQdrantClient(qdrant_client)
                else:
                    self.qdrant_client = qdrant_client
            else:
                self.qdrant_client = qdrant_client
        else:
            self.qdrant_client = None

        self.embedder = None
        self.sparse_embedder = None
        self.colbert_embedder = None
        self.embedding_plan = get_embedding_plan(config)
        self.embedding_settings = get_embedding_settings(config)
        self.embedding_version = self.embedding_settings.version
        self.embedding_dims = self.embedding_settings.dims or 0
        runtime_settings = get_settings()
        self.namespace_mode = getattr(
            runtime_settings, "embedding_namespace_mode", "none"
        )
        self.vector_primary = config.search.vector.primary
        self.dual_write = config.search.vector.dual_write
        self.expected_schema_version = (
            getattr(config.graph_schema, "version", None) if config else None
        )
        self.include_entity_vector = (
            os.getenv("QDRANT_INCLUDE_ENTITY_VECTOR", "true").lower() == "true"
        )
        self.manage_qdrant_on_init = (
            os.getenv("MANAGE_QDRANT_SCHEMA_ON_INIT", "false").lower() == "true"
        )
        global GRAPH_BUILDER_INIT_LOGGED
        if not GRAPH_BUILDER_INIT_LOGGED:
            search_cfg = getattr(config, "search", None)
            logger.info(
                "GraphBuilder initialized",
                extra={
                    "embedding_profile": getattr(
                        self.embedding_settings, "profile", None
                    ),
                    "embedding_provider": getattr(
                        self.embedding_settings, "provider", None
                    ),
                    "embedding_model": getattr(
                        self.embedding_settings, "model_id", None
                    ),
                    "embedding_version": getattr(
                        self.embedding_settings, "version", None
                    ),
                    "embedding_namespace_mode": self.namespace_mode,
                    "bm25_index_name": getattr(
                        getattr(search_cfg, "bm25", None), "index_name", None
                    ),
                    "qdrant_collection_name": getattr(
                        getattr(getattr(search_cfg, "vector", None), "qdrant", None),
                        "collection_name",
                        None,
                    ),
                },
            )
            GRAPH_BUILDER_INIT_LOGGED = True

        # Phase 7C.7: Fresh start with 1024-D (Session 06-08)
        # No dual-write complexity - starting fresh with Jina v4 @ 1024-D

        if strict_mode is None:
            strict_mode = runtime_settings.embedding_strict_mode
        self._strict_mode_enabled = bool(strict_mode)

    def ensure_embedder(self) -> None:
        """
        Initialize the embedding provider if it has not been created.

        This mirrors the initialization logic used in upsert_document and is a
        no-op when the embedder is already available.
        """
        if self.embedder:
            return

        logger.info(
            "Initializing embedding provider",
            provider=self.embedding_settings.provider,
            model=self.embedding_settings.model_id,
            dims=self.embedding_settings.dims,
            profile=self.embedding_settings.profile,
        )

        self.embedder = ProviderFactory.create_embedding_provider_for_role(
            self.embedding_plan.dense
        )
        self.sparse_embedder = self.embedder
        self.colbert_embedder = self.embedder
        if self.embedding_plan.sparse and (
            self.embedding_plan.sparse.profile_name
            != self.embedding_plan.dense.profile_name
        ):
            self.sparse_embedder = ProviderFactory.create_embedding_provider_for_role(
                self.embedding_plan.sparse
            )
        if self.embedding_plan.colbert:
            if (
                self.embedding_plan.sparse
                and self.embedding_plan.colbert.profile_name
                == self.embedding_plan.sparse.profile_name
            ):
                self.colbert_embedder = self.sparse_embedder
            elif (
                self.embedding_plan.colbert.profile_name
                != self.embedding_plan.dense.profile_name
            ):
                self.colbert_embedder = (
                    ProviderFactory.create_embedding_provider_for_role(
                        self.embedding_plan.colbert
                    )
                )

        if self.embedder.dims != self.embedding_dims:
            logger.warning(
                "Embedding dims mismatch; aligning local settings to provider",
                configured_dims=self.embedding_dims,
                provider_dims=self.embedder.dims,
                model=self.embedder.model_id,
            )
            self.embedding_dims = self.embedder.dims

        logger.info(
            "Embedding provider initialized",
            provider_name=self.embedder.provider_name,
            model_id=self.embedder.model_id,
            actual_dims=self.embedder.dims,
        )

    def _build_section_text_for_embedding(self, section: Dict) -> str:
        """Build text for embedding from section with title trail."""
        # Include title for better context
        title = section.get("title", "")
        text = section.get("text", "")

        if title:
            return f"{title}\n\n{text}"
        return text

    def _build_title_text_for_embedding(self, section: Dict) -> str:
        """Build a compact title/heading string for auxiliary vectors."""
        heading = section.get("title") or section.get("heading")
        if heading:
            return heading
        text = (section.get("text") or "").strip()
        return text[:256]

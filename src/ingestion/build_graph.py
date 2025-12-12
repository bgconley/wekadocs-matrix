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

# C.1.3: Entity validity criteria - entity_types that qualify for :Entity label
# Updated for GLiNER v2 labels (2024-12)
VALID_ENTITY_TYPES = frozenset(
    {
        # Legacy structural entity types
        "heading_concept",
        "cli_command",
        "config_param",
        "concept",
        "api_endpoint",
        "parameter",
        "option",
        "feature",
        "service",
        "module",
        # GLiNER v2 entity types (refined for retrieval)
        "COMMAND",
        "PARAMETER",
        "COMPONENT",
        "PROTOCOL",
        "CLOUD_PROVIDER",
        "STORAGE_CONCEPT",
        "VERSION",
        "PROCEDURE_STEP",
        "ERROR",
        "CAPACITY_METRIC",
        # Lowercase variants (for case-insensitive matching)
        "command",
        "component",
        "protocol",
        "cloud_provider",
        "storage_concept",
        "version",
        "procedure_step",
        "error",
        "capacity_metric",
    }
)


class GraphBuilder:
    """Builds graph from parsed documents, sections, and entities."""

    # Class-level caches to track what has been ensured in this process.
    # This prevents redundant DB calls when GraphBuilder is instantiated per-document.
    _payload_indexes_ensured: set = set()  # Qdrant collections with payload indexes
    _neo4j_indexes_ensured: set = set()  # Neo4j index names that have been ensured
    _schema_metadata_ensured: bool = False  # SchemaVersion metadata reconciled

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
        self._schema_version_verified = False
        self._profile_alignment_verified = False
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

        # Ensure Qdrant collections exist if using Qdrant
        if self.qdrant_client and (self.vector_primary == "qdrant" or self.dual_write):
            self._ensure_qdrant_collection()

        # Ensure Neo4j vector index and schema metadata align with active profile
        self._ensure_neo4j_vector_index()
        self._reconcile_schema_version_embedding_metadata()

        if strict_mode is None:
            strict_mode = runtime_settings.embedding_strict_mode
        self._strict_mode_enabled = bool(strict_mode)

    def _validate_schema_version(self, session) -> None:
        """
        Ensure the Neo4j SchemaVersion marker matches the configured expectation.

        Raises RuntimeError if the SchemaVersion node is missing or mismatched.
        """
        if self._schema_version_verified:
            return

        if not self.expected_schema_version:
            self._schema_version_verified = True
            return

        ensure_schema_version(self.driver, self.expected_schema_version)
        self._schema_version_verified = True

    def _enforce_profile_guard(self) -> None:
        """Detect and optionally block ingestion if legacy embeddings persist."""
        if self._profile_alignment_verified:
            return
        # When namespace isolation is enabled, we don’t enforce cross-profile drift
        # because each profile is intended to have its own storage “home”.
        if str(self.namespace_mode).lower() not in {"", "none", "disabled"}:
            self._profile_alignment_verified = True
            embedding_profile_guard_events_total.labels(
                profile=self.embedding_settings.profile or "legacy",
                outcome="namespaced",
            ).inc()
            return
        mismatched_ids: List[str] = []
        query = """
        MATCH (s:Section)
        WHERE s.embedding_version IS NOT NULL AND s.embedding_version <> $version
        RETURN s.id AS id
        LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query, version=self.embedding_settings.version)
            mismatched_ids = [record["id"] for record in result]

        if mismatched_ids:
            message = (
                "Embedding metadata drift detected for current profile "
                f"{self.embedding_settings.profile}: found Sections with older "
                f"embedding_version values (examples: {', '.join(mismatched_ids)}). "
                "Re-ingest or reconcile existing data before proceeding."
            )
            outcome = "blocked" if self._strict_mode_enabled else "warned"
            embedding_profile_guard_events_total.labels(
                profile=self.embedding_settings.profile or "legacy",
                outcome=outcome,
            ).inc()
            if self._strict_mode_enabled:
                raise RuntimeError(
                    message + " EMBEDDING_STRICT_MODE blocked ingestion."
                )
            logger.warning(message)
        else:
            embedding_profile_guard_events_total.labels(
                profile=self.embedding_settings.profile or "legacy",
                outcome="clean",
            ).inc()
        self._profile_alignment_verified = True

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

        self.embedder = ProviderFactory.create_embedding_provider(
            settings=self.embedding_settings
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

    def upsert_document(
        self,
        document: Dict,
        sections: List[Dict],
        entities: Dict[str, Dict],
        mentions: List[Dict],
    ) -> Dict[str, any]:
        """
        Upsert document, sections, entities, and mentions to graph.
        Idempotent - can be run multiple times safely.

        Phase 7E-4: Now collects comprehensive metrics and monitors SLOs.

        Args:
            document: Document metadata
            sections: List of sections
            entities: Dict of entities keyed by ID
            mentions: List of MENTIONS relationships

        Returns:
            Stats dict with metrics and SLO violation information
        """
        start_time = time.time()
        stats = {
            "document_id": document["id"],
            "sections_upserted": 0,
            "entities_upserted": 0,
            "mentions_created": 0,
            "embeddings_computed": 0,
            "vectors_upserted": 0,
            "duration_ms": 0,
        }

        logger.info("Starting graph upsert", document_id=document["id"])

        # Phase 7E-2: Combine small sections, then split if still too large
        assembler = get_chunk_assembler(
            getattr(self.config.ingestion, "chunk_assembly", None)
        )
        sections = assembler.assemble(document["id"], sections)

        doc_total_tokens = sum(int(s.get("token_count", 0)) for s in sections)
        document["total_tokens"] = doc_total_tokens
        document.setdefault("doc_id", document.get("id"))
        doc_fallback_threshold = int(
            os.getenv(
                "COMBINE_DOC_FALLBACK_DOC_TOKEN_MAX",
                str(max(int(os.getenv("COMBINE_TARGET_TOKENS", "1300")) * 2, 2600)),
            )
        )
        is_microdoc = doc_total_tokens <= doc_fallback_threshold
        for section in sections:
            section.setdefault("document_id", document["id"])
            section.setdefault("doc_id", document.get("doc_id"))
            section["document_total_tokens"] = doc_total_tokens
            section["is_microdoc"] = is_microdoc
            section.setdefault("tenant", document.get("tenant"))
            section.setdefault("lang", document.get("lang"))
            section.setdefault("version", document.get("version"))

        with self.driver.session() as session:
            # Fail fast if schema version drifts from configuration
            self._validate_schema_version(session)
            self._enforce_profile_guard()

            # Step 1: Upsert Document node
            self._upsert_document_node(session, document)

            # Step 2a: Phase 7E-1 Replace-by-set GC - Delete stale chunks BEFORE upsert
            # This ensures idempotency: only current chunks remain after ingestion
            self._delete_stale_chunks_neo4j(session, document["id"], sections)

            # Step 2b: Upsert Sections in batches (with chunk schema)
            stats["sections_upserted"] = self._upsert_sections(
                session, document["id"], sections
            )

            citation_units_upserted = self._upsert_citation_units(
                session, document["id"], sections
            )
            stats["citation_units_upserted"] = citation_units_upserted

            # remove transient citation payloads before further processing
            for section in sections:
                section.pop("_citation_units", None)

            # Step 2c: Create NEXT_CHUNK relationships for adjacency
            self._create_next_chunk_relationships(session, document["id"], sections)
            rel_counts = self._build_typed_relationships(session, document["id"])
            stats["relationship_builders"] = rel_counts

            # Optional Phase 7E repair: correct legacy combined flags
            repaired = self._repair_incorrect_combined_flags(session, document["id"])
            if repaired:
                logger.info(
                    "Repaired incorrect combined flag assignments",
                    document_id=document["id"],
                    repaired_count=repaired,
                )

            # Step 2d: C.1.1 - Create heading concept entities from section headings
            heading_concept_stats = self._create_heading_concept_entities(
                session, document["id"], sections
            )
            stats["heading_concepts"] = heading_concept_stats

            # Step 3: Upsert Entities in batches
            stats["entities_upserted"] = self._upsert_entities(session, entities)

            # Step 4: Create MENTIONS edges in batches
            stats["mentions_created"] = self._create_mentions(session, mentions)

        # Step 4.5: Attach mentions to sections for entity sparse embedding
        # This enables _process_embeddings to build entity text from mentions
        # NOTE: Combined chunks have NEW IDs but mentions are keyed by ORIGINAL section IDs.
        # We must check both current ID and original_section_ids to attach all mentions.
        from collections import defaultdict

        mentions_by_section = defaultdict(list)
        for m in mentions:
            sid = m.get("section_id")
            if sid:
                mentions_by_section[sid].append(m)

        for section in sections:
            section_mentions = []
            # Check current section ID
            section_id = section.get("id")
            if section_id and section_id in mentions_by_section:
                section_mentions.extend(mentions_by_section[section_id])
            # Check original section IDs (from chunk assembly - combined chunks)
            original_ids = section.get("original_section_ids", [])
            for orig_id in original_ids:
                if orig_id in mentions_by_section:
                    section_mentions.extend(mentions_by_section[orig_id])
            # Deduplicate by entity_id to avoid double-counting
            seen_entity_ids = set()
            unique_mentions = []
            for m in section_mentions:
                eid = m.get("entity_id")
                if eid and eid not in seen_entity_ids:
                    seen_entity_ids.add(eid)
                    unique_mentions.append(m)
            section["_mentions"] = unique_mentions

        # Step 5: Compute embeddings and upsert to vector store
        embedding_stats = self._process_embeddings(document, sections, entities)
        stats["embeddings_computed"] = embedding_stats["computed"]
        stats["vectors_upserted"] = embedding_stats["upserted"]

        # Step 6: Phase 7E-3 - Invalidate caches AFTER Neo4j and Qdrant commits
        # Reference: Canonical Spec L3313-3336 (invalidate post-commit)
        chunk_ids = [s["id"] for s in sections if "id" in s]
        invalidation_stats = self._invalidate_caches_post_ingest(
            document["id"], chunk_ids
        )
        stats["cache_invalidation"] = invalidation_stats

        # Optional reconciliation step to repair drift for legacy data
        if (
            self.config.ingestion.reconciliation.enabled
            and self.qdrant_client
            and self.vector_primary == "qdrant"
        ):
            from src.ingestion.reconcile import Reconciler

            try:
                reconciler = Reconciler(self.driver, self.config, self.qdrant_client)
                reconciler.reconcile()
            except Exception as exc:
                logger.warning("Reconciliation after upsert failed", error=str(exc))

        # Phase 7E-4: Collect metrics and monitor SLOs
        duration_seconds = time.time() - start_time
        stats["duration_ms"] = int(duration_seconds * 1000)

        if self.config.monitoring.metrics_enabled:
            # Collect chunk metrics with canonical spec defaults
            # Reference: Canonical Spec - TARGET_MIN=200, ABSOLUTE_MAX=7900
            collector = MetricsCollector(
                target_min=200,
                absolute_max=7900,
            )
            chunk_metrics = collector.collect_chunk_metrics(sections)
            ingestion_metrics = collector.collect_ingestion_metrics(
                document_id=document["id"],
                chunks=sections,
                duration_seconds=duration_seconds,
            )

            # Emit Prometheus metrics for each chunk
            for section in sections:
                token_count = section.get("token_count", 0)
                chunk_token_distribution.observe(token_count)
                chunks_created_total.labels(document_id=document["id"]).inc()

                # Track oversized chunks (ZERO tolerance SLO)
                # Reference: Canonical Spec ABSOLUTE_MAX=7900
                if token_count > 7900:
                    chunks_oversized_total.labels(document_id=document["id"]).inc()

            # Record ingestion duration
            ingestion_duration_seconds.observe(duration_seconds)

            # Store metrics in stats for response
            stats["chunk_metrics"] = chunk_metrics.to_dict()
            stats["ingestion_metrics"] = ingestion_metrics.to_dict()

            # Phase 7E-4: Check SLOs if monitoring enabled
            if self.config.monitoring.slo_monitoring_enabled:
                slo_metrics = collector.compute_slo_metrics(chunk_metrics)
                # Graph Channel Rehabilitation: Add sparse coverage to SLO metrics
                slo_metrics["sparse_content_missing"] = embedding_stats.get(
                    "sparse_content_missing", 0
                )
                violations = check_slos_and_log(slo_metrics, logger)

                stats["slo_metrics"] = slo_metrics
                stats["slo_violations"] = [
                    {
                        "slo": v.slo_name,
                        "level": v.level.value,
                        "message": v.message,
                        "value": v.value,
                        "threshold": v.threshold,
                    }
                    for v in violations
                ]

                # Log critical violations
                for v in violations:
                    if v.level.value == "page":
                        logger.error(
                            "CRITICAL SLO violation",
                            slo=v.slo_name,
                            value=v.value,
                            threshold=v.threshold,
                            document_id=document["id"],
                        )

        logger.info("Graph upsert complete", stats=stats)

        return stats

    def _upsert_document_node(self, session, document: Dict):
        """Upsert Document node."""
        query = """
        MERGE (d:Document {id: $id})
        SET d.source_uri = $source_uri,
            d.source_type = $source_type,
            d.title = $title,
            d.version = $version,
            d.checksum = $checksum,
            d.last_edited = $last_edited,
            d.doc_tag = $doc_tag,
            d.snapshot_scope = $snapshot_scope,
            d.total_tokens = $total_tokens,
            d.updated_at = datetime()
        RETURN d.id as id
        """
        params = dict(document)
        params.setdefault("doc_tag", None)
        params.setdefault("snapshot_scope", None)
        params.setdefault("total_tokens", 0)
        session.run(query, **params)
        logger.debug("Document upserted", document_id=document["id"])

    def _upsert_sections(self, session, document_id: str, sections: List[Dict]) -> int:
        """
        Upsert Section nodes with dual-labeling and HAS_SECTION relationships in batches.

        Phase 7E-1: Creates :Section:Chunk nodes with full canonical chunk schema.
        Each section becomes a single-section chunk with deterministic ID generation.
        Embedding metadata will be set later in _process_embeddings after vectors are generated.
        """
        batch_size = self.config.ingestion.batch_size
        total_sections = 0

        # Phase 7E-1: Enrich sections with chunk metadata IN-PLACE
        # This ensures chunk metadata is available to _process_embeddings later
        for section in sections:
            # CRITICAL: Preserve original section ID before generating chunk ID
            # Store original ID in original_section_ids if not already set
            if "original_section_ids" not in section:
                original_section_id = section["id"]

                # Create chunk metadata (single-section chunks for Phase 7E-1)
                chunk_meta = create_chunk_metadata(
                    section_id=original_section_id,  # Use preserved original ID
                    document_id=document_id,
                    level=section.get("level", 3),
                    order=section.get("order", 0),
                    heading=section.get("title"),
                    parent_section_id=None,  # Could derive from hierarchy if needed
                    is_combined=False,  # Single-section chunks
                    is_split=False,
                    boundaries_json="{}",
                    token_count=section.get("tokens", 0),
                )

                # Update section in-place with chunk metadata
                # This overwrites section["id"] with chunk ID, but we've saved original
                section.update(chunk_meta)

                # Validate chunk schema
                if not validate_chunk_schema(section):
                    raise ValueError(
                        f"Invalid chunk schema for section {section['id']}. "
                        "Ingestion blocked - schema validation failed."
                    )

        # Use enriched sections for batch processing
        chunks = sections

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Phase 7E-1: MERGE with dual-label :Section:Chunk + canonical chunk fields
            # Phase 3 (markdown-it-py): Added enhanced metadata fields for structural queries
            query = """
            UNWIND $chunks as chunk
            MERGE (s:Section:Chunk {id: chunk.id})
            SET s.document_id = chunk.document_id,
                s.level = chunk.level,
                s.title = chunk.heading,
                s.anchor = chunk.anchor,
                s.order = chunk.order,
                s.heading = chunk.heading,
                s.parent_section_id = chunk.parent_section_id,
                s.parent_section_original_id = chunk.parent_section_original_id,
                s.text = chunk.text,
                s.tokens = chunk.tokens,
                s.token_count = chunk.token_count,
                s.checksum = chunk.checksum,
                s.original_section_ids = chunk.original_section_ids,
                s.is_combined = chunk.is_combined,
                s.is_split = chunk.is_split,
                s.boundaries_json = chunk.boundaries_json,
                s.document_total_tokens = chunk.document_total_tokens,
                s.is_microdoc = chunk.is_microdoc,
                s.doc_is_microdoc = coalesce(chunk.doc_is_microdoc, chunk.is_microdoc),
                // Phase 3: Enhanced structural metadata from markdown-it-py
                s.line_start = chunk.line_start,
                s.line_end = chunk.line_end,
                s.parent_path = coalesce(chunk.parent_path, ''),
                s.block_types = coalesce(chunk.block_types, []),
                s.code_ratio = coalesce(chunk.code_ratio, 0.0),
                s.has_code = coalesce(chunk.has_code, false),
                s.has_table = coalesce(chunk.has_table, false),
                s.updated_at = datetime()
            WITH s, chunk
            MATCH (d:Document {id: $document_id})
            SET s.doc_tag = d.doc_tag,
                s.snapshot_scope = d.snapshot_scope
            MERGE (d)-[r:HAS_SECTION]->(s)
            SET r.order = chunk.order,
                r.updated_at = datetime()
            RETURN count(s) as count
            """

            result = session.run(query, chunks=batch, document_id=document_id)
            count = result.single()["count"]
            total_sections += count

            logger.debug(
                "Chunk batch upserted (dual-labeled with canonical schema)",
                batch_num=i // batch_size + 1,
                count=count,
            )

        return total_sections

    def _create_heading_concept_entities(
        self, session, document_id: str, sections: List[Dict]
    ) -> Dict[str, int]:
        """
        C.1.1: Create concept entities from qualifying section headings.

        For each section heading that meets validity criteria:
        - Creates an :Entity:Concept node with entity_type="heading_concept"
        - Creates DEFINES relationship from entity to chunk
        - Creates MENTIONS relationship (Chunk->Entity) per Neo4j best practice

        Criteria for qualifying headings:
        - Non-empty and >= 5 characters
        - Not in GENERIC_HEADING_BLACKLIST
        - Not purely numeric or bullet markers

        Returns:
            Dict with counts: entities_created, defines_created, mentions_created
        """
        stats = {"entities_created": 0, "defines_created": 0, "mentions_created": 0}

        # Extract qualifying headings from sections
        heading_entities = []
        for section in sections:
            heading = section.get("heading") or section.get("title") or ""
            heading = heading.strip()

            # Skip empty or too short
            if not heading or len(heading) < 5:
                continue

            # Normalize for comparison
            canonical = re.sub(r"\s+", " ", heading.lower().strip())
            # Remove leading articles
            canonical = re.sub(r"^(the|a|an)\s+", "", canonical)

            # Skip generic headings
            if canonical in GENERIC_HEADING_BLACKLIST:
                continue

            # Skip purely numeric or bullet markers (e.g., "1.", "1.2.3", "•", "-")
            if re.match(r"^[\d\.\-\•\*\#]+$", heading.strip()):
                continue

            # Generate deterministic entity ID
            entity_id = hashlib.sha256(
                f"heading_concept:{document_id}:{canonical}".encode()
            ).hexdigest()[:24]

            heading_entities.append(
                {
                    "id": entity_id,
                    "name": heading,
                    "canonical_name": canonical,
                    "entity_type": "heading_concept",
                    "source_section_id": section.get("id"),
                    "document_id": document_id,
                    "doc_tag": section.get("doc_tag"),
                }
            )

        if not heading_entities:
            logger.debug(
                "No qualifying heading concepts found",
                document_id=document_id,
                sections_checked=len(sections),
            )
            return stats

        # Batch upsert heading concept entities
        batch_size = self.config.ingestion.batch_size
        for i in range(0, len(heading_entities), batch_size):
            batch = heading_entities[i : i + batch_size]

            # Create :Entity:Concept nodes
            entity_query = """
            UNWIND $entities AS ent
            MERGE (e:Entity:Concept {id: ent.id})
            SET e.name = ent.name,
                e.canonical_name = ent.canonical_name,
                e.entity_type = ent.entity_type,
                e.source_section_id = ent.source_section_id,
                e.document_id = ent.document_id,
                e.doc_tag = ent.doc_tag,
                e.updated_at = datetime()
            RETURN count(e) AS count
            """
            result = session.run(entity_query, entities=batch)
            stats["entities_created"] += result.single()["count"]

            # Create DEFINES relationships (Entity -> Chunk)
            defines_query = """
            UNWIND $entities AS ent
            MATCH (e:Entity:Concept {id: ent.id})
            MATCH (c:Chunk {id: ent.source_section_id})
            MERGE (e)-[r:DEFINES]->(c)
            SET r.updated_at = datetime()
            RETURN count(r) AS count
            """
            result = session.run(defines_query, entities=batch)
            stats["defines_created"] += result.single()["count"]

            # Create MENTIONS relationships (Chunk -> Entity) - canonical direction per Neo4j best practice
            mentions_query = """
            UNWIND $entities AS ent
            MATCH (e:Entity:Concept {id: ent.id})
            MATCH (c:Chunk {id: ent.source_section_id})
            MERGE (c)-[r:MENTIONS]->(e)
            SET r.updated_at = datetime()
            RETURN count(r) AS count
            """
            result = session.run(mentions_query, entities=batch)
            stats["mentions_created"] += result.single()["count"]

        logger.info(
            "Created heading concept entities",
            document_id=document_id,
            entities=stats["entities_created"],
            defines=stats["defines_created"],
            mentions=stats["mentions_created"],
        )

        return stats

    def _upsert_citation_units(
        self, session, document_id: str, chunks: List[Dict]
    ) -> int:
        """Upsert lightweight CitationUnit nodes for subsection-level citations."""

        unit_map: Dict[str, Dict] = {}
        rels: List[Dict] = []

        for chunk in chunks:
            for unit in chunk.get("_citation_units", []) or []:
                if not unit.get("id"):
                    continue
                existing = unit_map.get(unit["id"])
                if existing is None or unit.get("order", 0) < existing.get("order", 0):
                    unit_map[unit["id"]] = unit
                rels.append(
                    {
                        "unit_id": unit["id"],
                        "chunk_id": unit["parent_chunk_id"],
                    }
                )

        units = list(unit_map.values())

        if not units:
            return 0

        session.run(
            """
            UNWIND $rows AS r
            MERGE (u:CitationUnit {id: r.id})
              ON CREATE SET u.created_at = timestamp()
            SET u.document_id     = r.document_id,
                u.heading         = r.heading,
                u.text            = r.text,
                u.level           = r.level,
                u.order           = r.order,
                u.token_count     = r.token_count,
                u.parent_chunk_id = r.parent_chunk_id,
                u.updated_at      = timestamp()
            WITH u, r
            MATCH (d:Document {id: r.document_id})
            SET u.doc_tag = d.doc_tag
            """,
            rows=units,
        )

        if rels:
            session.run(
                """
                UNWIND $pairs AS p
                MATCH (u:CitationUnit {id: p.unit_id})
                MATCH (c:Chunk {id: p.chunk_id})
                MERGE (u)-[:IN_CHUNK]->(c)
                """,
                pairs=rels,
            )

        session.run(
            """
            MATCH (d:Document {id: $doc_id})
            MATCH (u:CitationUnit {document_id: $doc_id})
            MERGE (d)-[:HAS_CITATION]->(u)
            """,
            doc_id=document_id,
        )

        return len(units)

    def _delete_stale_chunks_neo4j(
        self, session, document_id: str, current_sections: List[Dict]
    ):
        """
        Phase 7E-1: Replace-by-set GC for Neo4j chunks.

        Delete all chunks for this document that are NOT in the current set.
        This ensures idempotency - re-ingesting produces exactly the current chunks.

        Args:
            session: Neo4j session
            document_id: Document identifier
            current_sections: List of current sections (will become chunks)
        """
        # Current sections are already chunks (post-assembler). Use their IDs directly.
        current_chunk_ids = [s.get("id") for s in current_sections if s.get("id")]
        current_citation_ids: List[str] = []
        for section in current_sections:
            for unit in section.get("_citation_units", []) or []:
                if unit.get("id"):
                    current_citation_ids.append(unit["id"])
        current_citation_ids = list(dict.fromkeys(current_citation_ids))

        # Delete chunks not in current set
        delete_query = """
        MATCH (d:Document {id: $document_id})-[:HAS_SECTION]->(c:Chunk)
        WHERE NOT c.id IN $current_chunk_ids
        DETACH DELETE c
        RETURN count(c) as deleted
        """

        result = session.run(
            delete_query,
            document_id=document_id,
            current_chunk_ids=current_chunk_ids,
        )

        deleted = result.single()["deleted"] or 0

        if deleted > 0:
            logger.info(
                "Deleted stale chunks (replace-by-set GC)",
                document_id=document_id,
                deleted_count=deleted,
            )

        session.run(
            """
            MATCH (d:Document {id: $document_id})-[:HAS_CITATION]->(u:CitationUnit)
            WHERE NOT u.id IN $current_citation_ids
            DETACH DELETE u
            """,
            document_id=document_id,
            current_citation_ids=current_citation_ids,
        )

    def _create_next_chunk_relationships(
        self, session, document_id: str, sections: List[Dict]
    ):
        """
        Phase 7E-2: Create NEXT_CHUNK relationships for adjacency traversal.

        Links consecutive chunks by document order (not parent grouping).
        This enables bounded expansion (±1 neighbor) in retrieval and works
        correctly with combined chunks.

        Args:
            session: Neo4j session
            document_id: Document identifier
            sections: List of chunks (already ordered by assembler)
        """
        # Use Cypher to link chunks by order within document
        # This handles combined chunks correctly since they have proper IDs
        cypher = """
        MATCH (d:Document {id:$doc_id})-[:HAS_SECTION]->(c:Chunk)
        WITH c ORDER BY c.order, c.id
        WITH collect(c) AS cs
        UNWIND range(0, size(cs)-2) AS i
        WITH cs[i] AS c1, cs[i+1] AS c2
        MERGE (c1)-[r:NEXT_CHUNK {doc_id:$doc_id}]->(c2)
        SET r.updated_at = datetime()
        RETURN count(r) as count
        """

        result = session.run(cypher, doc_id=document_id)
        count = result.single()["count"] or 0

        logger.debug(
            "NEXT_CHUNK relationships created (document-ordered)",
            document_id=document_id,
            count=count,
        )

    def _build_typed_relationships(self, session, document_id: str) -> Dict[str, int]:
        """
        Materialize typed chunk/section relationships defined in the v2.2 schema.

        Mirrors the guarded Cypher snippets from PART 8 of
        create_graphrag_schema_v2_2_20251105_guard.cypher, ensuring we build:
            - CHILD_OF (Chunk -> Section)
            - PARENT_OF (Section -> Section)
            - NEXT (Chunk adjacency within parent/doc scope)

        Phase 2 Cleanup: Removed PREV (redundant - use <-[:NEXT]-) and
        SAME_HEADING (O(n²) fanout with zero query usage).
        """

        queries = {
            "child_of": """
            MATCH (c:Chunk)
            WHERE c.text IS NOT NULL
              AND coalesce(c.document_id, c.doc_id) = $doc_id
              AND c.parent_section_id IS NOT NULL
            MATCH (s:Section {id: c.parent_section_id})
            MERGE (c)-[:CHILD_OF]->(s)
            RETURN count(c) AS count
            """,
            "parent_of": """
            MATCH (child:Section)
            WHERE child.parent_section_id IS NOT NULL
              AND coalesce(child.document_id, child.doc_id) = $doc_id
            MATCH (parent:Section {id: child.parent_section_id})
            MERGE (parent)-[:PARENT_OF]->(child)
            RETURN count(child) AS count
            """,
            # Phase 2 Cleanup: Renamed from "next_prev", removed PREV edge creation
            # Reverse traversal uses <-[:NEXT]- pattern instead of [:PREV]
            # Also removed "same_heading" query - O(n²) fanout with zero query usage
            "next": """
            MATCH (c:Chunk)
            WHERE c.text IS NOT NULL
              AND coalesce(c.document_id, c.doc_id) = $doc_id
            WITH c.parent_section_id AS parent_id, c
            ORDER BY parent_id, c.order, c.id
            WITH parent_id, collect(c) AS chunks
            WHERE size(chunks) > 1
            UNWIND range(0, size(chunks)-2) AS idx
            WITH chunks[idx] AS a, chunks[idx+1] AS b
            MERGE (a)-[:NEXT]->(b)
            RETURN count(*) AS count
            """,
            # Phase 3 (markdown-it-py): PARENT_HEADING based on parent_path hierarchy
            # Creates heading-based parent-child edges distinct from PARENT_OF
            # Direction: child→parent (Neo4j hierarchy convention)
            "parent_heading": """
            MATCH (child:Section)
            WHERE child.document_id = $doc_id
              AND child.parent_path IS NOT NULL
              AND child.parent_path <> ''
              AND NOT EXISTS {
                MATCH (child)-[:PARENT_HEADING]->(:Section)
              }
            WITH child,
                 split(child.parent_path, ' > ') AS path_parts,
                 child.level AS child_level,
                 child.order AS child_order
            WHERE size(path_parts) > 0
            WITH child,
                 path_parts[size(path_parts) - 1] AS immediate_parent_title,
                 child_level,
                 child_order
            MATCH (parent:Section)
            WHERE parent.document_id = $doc_id
              AND parent.title = immediate_parent_title
              AND parent.level < child_level
              AND parent.order < child_order
            WITH child, parent
            ORDER BY parent.order DESC
            WITH child, collect(parent)[0] AS parent
            WHERE parent IS NOT NULL
            MERGE (child)-[r:PARENT_HEADING]->(parent)
            SET r.level_delta = child.level - parent.level,
                r.created_at = datetime()
            RETURN count(r) AS count
            """,
            # Phase 3 (markdown-it-py): Apply :CodeSection label for structural filtering
            "code_section_label": """
            MATCH (s:Section)
            WHERE s.document_id = $doc_id
              AND s.has_code = true
              AND NOT s:CodeSection
            SET s:CodeSection
            RETURN count(s) AS count
            """,
            # Phase 3 (markdown-it-py): Apply :TableSection label for structural filtering
            "table_section_label": """
            MATCH (s:Section)
            WHERE s.document_id = $doc_id
              AND s.has_table = true
              AND NOT s:TableSection
            SET s:TableSection
            RETURN count(s) AS count
            """,
        }

        log_rel_counts = (
            os.getenv(
                "LOG_RELATIONSHIP_COUNTS", os.getenv("INGEST_LOG_REL_COUNTS", "false")
            ).lower()
            == "true"
        )
        results: Dict[str, Dict[str, int]] = {}
        for name, query in queries.items():
            try:
                result = session.run(query, doc_id=document_id)
                record = None
                try:
                    record = result.single()
                except Exception:
                    record = None
                summary = result.consume()
                counters = summary.counters
                returned = int(record["count"]) if record and "count" in record else 0
                created = counters.relationships_created if counters else 0
                deleted = counters.relationships_deleted if counters else 0
                verification = None
                if log_rel_counts:
                    verification = self._verify_relationship_count(
                        session, name, document_id
                    )
                self._log_relationship_builder(
                    builder=name,
                    document_id=document_id,
                    returned=returned,
                    created=created,
                    deleted=deleted,
                    verification=verification,
                )
                results[name] = {
                    "returned": returned,
                    "created": created,
                    "deleted": deleted,
                    "verification": verification or 0,
                }
            except Exception as exc:
                logger.warning(
                    "Typed relationship builder failed",
                    document_id=document_id,
                    builder=name,
                    error=str(exc),
                )
                results[name] = {
                    "returned": 0,
                    "created": 0,
                    "deleted": 0,
                    "verification": 0,
                }

        logger.debug(
            "Typed relationship builders executed",
            document_id=document_id,
            counts={
                name: {
                    "returned": data["returned"],
                    "created": data["created"],
                    "deleted": data["deleted"],
                    "verification": data["verification"],
                }
                for name, data in results.items()
            },
        )

        return results

    def _log_relationship_builder(
        self,
        *,
        builder: str,
        document_id: str,
        returned: int,
        created: int,
        deleted: int,
        verification: Optional[int],
    ) -> None:
        logger.debug(
            "Relationship builder stats",
            builder=builder,
            document_id=document_id,
            returned=returned,
            created=created,
            deleted=deleted,
            verification=verification,
        )

    def _verify_relationship_count(
        self, session, builder: str, document_id: str
    ) -> int:
        if builder == "child_of":
            query = """
            MATCH (c:Chunk)-[:CHILD_OF]->(:Section)
            WHERE coalesce(c.document_id, c.doc_id) = $doc_id
            RETURN count(*) AS count
            """
        elif builder == "parent_of":
            query = """
            MATCH (:Section)-[:PARENT_OF]->(child:Section)
            WHERE coalesce(child.document_id, child.doc_id) = $doc_id
            RETURN count(*) AS count
            """
        # Phase 2 Cleanup: Renamed from "next_prev", removed PREV verification
        # Also removed "same_heading" verification (edge type no longer created)
        elif builder == "next":
            query = """
            MATCH (c:Chunk)-[:NEXT]->(:Chunk)
            WHERE coalesce(c.document_id, c.doc_id) = $doc_id
            RETURN count(*) AS count
            """
        # Phase 3 (markdown-it-py): PARENT_HEADING relationship verification
        elif builder == "parent_heading":
            query = """
            MATCH (child:Section)-[:PARENT_HEADING]->(:Section)
            WHERE child.document_id = $doc_id
            RETURN count(*) AS count
            """
        # Phase 3 (markdown-it-py): CodeSection label verification
        elif builder == "code_section_label":
            query = """
            MATCH (s:Section:CodeSection)
            WHERE s.document_id = $doc_id
            RETURN count(s) AS count
            """
        # Phase 3 (markdown-it-py): TableSection label verification
        elif builder == "table_section_label":
            query = """
            MATCH (s:Section:TableSection)
            WHERE s.document_id = $doc_id
            RETURN count(s) AS count
            """
        else:
            return 0
        record = session.run(query, doc_id=document_id).single()
        return int(record["count"]) if record and "count" in record else 0

    def _repair_incorrect_combined_flags(self, session, document_id: str) -> int:
        """
        Optional repair step: ensure chunks marked as combined truly reference >1 section.

        This implements the guidance to flip stray `is_combined` flags that lingered
        from earlier ingestion bugs while scoping the fix to the current document.
        """

        query = """
        MATCH (d:Document {id: $document_id})-[:HAS_SECTION]->(c:Chunk)
        WHERE c.is_combined = true AND (c.original_section_ids IS NULL OR size(c.original_section_ids) <= 1)
        SET c.is_combined = false
        RETURN count(c) as repaired
        """

        result = session.run(query, document_id=document_id)
        repaired = result.single()["repaired"] or 0
        return repaired

    def _remove_missing_sections(
        self, session, document_id: str, valid_section_ids: List[str]
    ) -> int:
        """
        Hybrid orphan section cleanup strategy.

        For sections no longer in the document:
        1. DELETE sections with NO Query/Answer provenance (truly orphaned)
        2. MARK sections with provenance as stale (preserve citation chains)

        This prevents breaking historical query/answer references while
        cleaning up sections that are genuinely no longer needed.

        Returns:
            Count of sections removed (deleted + marked stale)
        """
        # Step 1: Find orphaned sections (not in current document version)
        find_orphans_query = """
        MATCH (d:Document {id: $document_id})-[r:HAS_SECTION]->(s:Section)
        WHERE NOT s.id IN $section_ids

        // Check for provenance: RETRIEVED or SUPPORTED_BY relationships
        OPTIONAL MATCH (s)<-[:RETRIEVED]-(q:Query)
        OPTIONAL MATCH (s)<-[:SUPPORTED_BY]-(a:Answer)

        WITH s,
             count(DISTINCT q) + count(DISTINCT a) as provenance_count

        RETURN s.id as section_id,
               provenance_count,
               CASE
                   WHEN provenance_count = 0 THEN 'delete'
                   ELSE 'mark_stale'
               END as action
        """

        orphans_result = session.run(
            find_orphans_query,
            document_id=document_id,
            section_ids=valid_section_ids or [],
        )

        orphans = list(orphans_result)

        if not orphans:
            return 0

        # Step 2: Separate orphans by action
        to_delete = [o["section_id"] for o in orphans if o["action"] == "delete"]
        to_mark_stale = [
            o["section_id"] for o in orphans if o["action"] == "mark_stale"
        ]

        deleted_count = 0
        marked_stale_count = 0

        # Step 3: DELETE orphans with no provenance
        if to_delete:
            delete_query = """
            MATCH (s:Section)
            WHERE s.id IN $section_ids
            DETACH DELETE s
            RETURN count(s) as deleted
            """
            delete_result = session.run(delete_query, section_ids=to_delete)
            deleted_count = delete_result.single()["deleted"] or 0

            logger.debug(
                "Deleted orphaned sections with no provenance",
                document_id=document_id,
                deleted_count=deleted_count,
            )

        # Step 4: MARK orphans with provenance as stale
        if to_mark_stale:
            mark_stale_query = """
            MATCH (s:Section)
            WHERE s.id IN $section_ids
            SET s.is_stale = true,
                s.stale_since = datetime(),
                s.stale_reason = 'Section removed from document but has query/answer provenance'
            RETURN count(s) as marked
            """
            mark_result = session.run(mark_stale_query, section_ids=to_mark_stale)
            marked_stale_count = mark_result.single()["marked"] or 0

            logger.info(
                "Marked orphaned sections as stale (preserving provenance)",
                document_id=document_id,
                marked_stale_count=marked_stale_count,
                section_ids=to_mark_stale[:5],  # Log first 5 for debugging
            )

        total_removed = deleted_count + marked_stale_count

        if total_removed > 0:
            logger.info(
                "Orphan section cleanup complete",
                document_id=document_id,
                deleted=deleted_count,
                marked_stale=marked_stale_count,
                total=total_removed,
            )

        return total_removed

    def _upsert_entities(self, session, entities: Dict[str, Dict]) -> int:
        """
        Upsert Entity nodes in batches.

        C.1.3: Entity Label Hygiene
        - Only adds :Entity label to nodes that meet validity criteria
        - Filters out entities with NULL/empty names, short names, or invalid types
        - Logs filtered entities for debugging
        """
        batch_size = self.config.ingestion.batch_size
        entities_list = list(entities.values())
        total_entities = 0

        # C.1.3: Filter entities based on validity criteria
        valid_entities = []
        filtered_count = 0
        for entity in entities_list:
            name = entity.get("name") or ""
            entity_type = entity.get("category") or entity.get("entity_type") or ""

            # Skip entities with NULL or empty names
            if not name or not name.strip():
                filtered_count += 1
                continue

            # Skip entities with names too short (< 4 chars)
            if len(name.strip()) < 4:
                filtered_count += 1
                continue

            # Add canonical_name if not present
            if "canonical_name" not in entity:
                entity["canonical_name"] = re.sub(r"\s+", " ", name.lower().strip())

            # Add entity_type for tracking (use category as fallback)
            if "entity_type" not in entity:
                entity["entity_type"] = entity_type

            valid_entities.append(entity)

        if filtered_count > 0:
            logger.info(
                "Entity label hygiene: filtered invalid entities",
                filtered_count=filtered_count,
                valid_count=len(valid_entities),
            )

        # Group by label for efficient batching
        by_label = {}
        for entity in valid_entities:
            label = entity["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(entity)

        for label, entity_batch in by_label.items():
            for i in range(0, len(entity_batch), batch_size):
                batch = entity_batch[i : i + batch_size]

                # Dynamic label in query
                # C.1.3: Include canonical_name and entity_type for graph queries
                query = f"""
                UNWIND $entities as ent
                MERGE (e:Entity:{label} {{id: ent.id}})
                SET e.name = ent.name,
                    e.canonical_name = ent.canonical_name,
                    e.entity_type = ent.entity_type,
                    e.description = ent.description,
                    e.category = ent.category,
                    e.introduced_in = ent.introduced_in,
                    e.deprecated_in = ent.deprecated_in,
                    e.updated_at = datetime()
                RETURN count(e) as count
                """

                result = session.run(query, entities=batch)
                count = result.single()["count"]
                total_entities += count

                logger.debug(
                    "Entity batch upserted",
                    label=label,
                    batch_num=i // batch_size + 1,
                    count=count,
                )

        return total_entities

    def _create_mentions(self, session, mentions: List[Dict]) -> int:
        """
        Create relationship batches (both MENTIONS and other types like CONTAINS_STEP).
        Separates Section→Entity (MENTIONS) from Entity→Entity (typed) relationships.
        """
        batch_size = self.config.ingestion.batch_size

        # Separate Section→Entity from Entity→Entity relationships
        section_entity_rels = []
        entity_entity_rels = []

        for m in mentions:
            if "section_id" in m and "entity_id" in m:
                # Standard Section→Entity MENTIONS relationship
                section_entity_rels.append(m)
            elif "from_id" in m and "to_id" in m and "relationship" in m:
                # Entity→Entity typed relationship (e.g., CONTAINS_STEP)
                entity_entity_rels.append(m)
            else:
                logger.warning(f"Unknown mention structure, skipping: {m.keys()}")

        total_created = 0

        # Create Section→Entity MENTIONS relationships
        total_created += self._create_section_entity_mentions(
            session, section_entity_rels, batch_size
        )

        # Create Entity→Entity typed relationships
        total_created += self._create_entity_entity_relationships(
            session, entity_entity_rels, batch_size
        )

        return total_created

    def _create_section_entity_mentions(
        self, session, mentions: List[Dict], batch_size: int
    ) -> int:
        """Create Section→Entity MENTIONS relationships in batches."""
        total_mentions = 0

        for i in range(0, len(mentions), batch_size):
            batch = mentions[i : i + batch_size]

            # Phase 3.5: Single MENTIONS direction (Chunk->Entity) per Neo4j best practice
            # No bidirectional edges - use direction-agnostic queries instead
            query = """
            UNWIND $mentions as m
            MATCH (s:Section {id: m.section_id})
            MATCH (e {id: m.entity_id})
            MERGE (s)-[r:MENTIONS {entity_id: m.entity_id}]->(e)
            SET r.confidence = m.confidence,
                r.start = m.start,
                r.end = m.end,
                r.source_section_id = m.source_section_id,
                r.updated_at = datetime()
            RETURN count(r) as count
            """

            result = session.run(query, mentions=batch)
            count = result.single()["count"]
            total_mentions += count

            logger.debug(
                "Section→Entity MENTIONS batch created",
                batch_num=i // batch_size + 1,
                count=count,
            )

        return total_mentions

    def _create_entity_entity_relationships(
        self, session, relationships: List[Dict], batch_size: int
    ) -> int:
        """
        Create Entity→Entity typed relationships in batches.
        Supports dynamic relationship types (e.g., CONTAINS_STEP, CONFIGURES, RESOLVES).
        """
        if not relationships:
            return 0

        # Group by relationship type for efficient batch processing
        by_type = {}
        for rel in relationships:
            rel_type = rel.get("relationship", "UNKNOWN")
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        total_created = 0

        for rel_type, rels in by_type.items():
            for i in range(0, len(rels), batch_size):
                batch = rels[i : i + batch_size]

                # Build dynamic Cypher query with relationship type
                # Using CALL (vars) {} subquery to work around Cypher's limitation
                # on parameterized relationship types (Neo4j 5.x syntax)
                query = f"""
                UNWIND $rels as r
                MATCH (from {{id: r.from_id}})
                MATCH (to {{id: r.to_id}})
                CALL (from, to, r) {{
                    MERGE (from)-[rel:{rel_type}]->(to)
                    SET rel.confidence = r.confidence,
                        rel.source_section_id = r.source_section_id,
                        rel.updated_at = datetime()
                    SET rel = CASE
                        WHEN r.order IS NOT NULL THEN rel {{.*, order: r.order}}
                        ELSE rel
                    END
                    RETURN count(rel) as cnt
                }}
                RETURN sum(cnt) as count
                """

                result = session.run(query, rels=batch)
                count = result.single()["count"]
                total_created += count

                logger.debug(
                    f"Entity→Entity {rel_type} batch created",
                    batch_num=i // batch_size + 1,
                    count=count,
                    rel_type=rel_type,
                )

        return total_created

    def _process_embeddings(
        self, document: Dict, sections: List[Dict], entities: Dict[str, Dict]
    ) -> Dict:
        """
        Compute embeddings and upsert to vector store.

        Phase 7C.7: Fresh start with 1024-D from day one (Session 06-08).
        No dual-write complexity - uses configured provider (Jina v4 @ 1024-D by default).
        """
        stats = {
            "computed": 0,
            "upserted": 0,
            # Sparse coverage tracking (Graph Channel Rehabilitation)
            "sparse_eligible": 0,  # Non-stub chunks eligible for sparse
            "sparse_success": 0,  # Chunks that got sparse vectors
            "sparse_failures": 0,  # Batches where embed_sparse failed
            "sparse_content_missing": 0,  # Non-stub chunks without sparse (SLO metric)
        }

        # Phase 7C.7: Initialize embedding provider from factory
        self.ensure_embedder()

        # Phase 7E-1: Replace-by-set GC for Qdrant chunks
        # Delete all chunks for this document BEFORE upserting current set
        document_id = document.get("id") or (
            sections[0].get("document_id") if sections else None
        )

        if (
            sections
            and self.vector_primary == "qdrant"
            and self.qdrant_client
            and document_id
        ):
            collection_name = self.config.search.vector.qdrant.collection_name

            # Delete all chunks for this document (replace-by-set)
            filter_must = [
                {
                    "key": "node_label",
                    "match": {"value": "Section"},
                },  # Dual-labeled as Chunk
                {"key": "document_id", "match": {"value": document_id}},
            ]

            try:
                self.qdrant_client.delete_compat(
                    collection_name=collection_name,
                    points_selector={"filter": {"must": filter_must}},
                    wait=True,
                )
                logger.info(
                    "Deleted stale Qdrant chunks (replace-by-set GC)",
                    collection=collection_name,
                    document_id=document_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to delete stale Qdrant chunks",
                    error=str(exc),
                    collection=collection_name,
                    document_id=document_id,
                )

        # Phase 7C.7: Process sections with 1024-D embeddings (simplified, fresh start)
        # Batch process embeddings for efficiency
        # Extract document metadata for sections
        source_uri = document.get("source_uri", "")
        document_uri = document.get("source_uri", "")  # Use source_uri as document_uri

        sections_to_embed = []
        content_texts: List[str] = []
        title_texts: List[str] = []
        entity_texts: List[str] = []  # NEW: Entity names for sparse embedding
        for section in sections:
            section.setdefault("source_uri", source_uri)
            section.setdefault("document_uri", document_uri)
            content_texts.append(self._build_section_text_for_embedding(section))
            title_texts.append(self._build_title_text_for_embedding(section))
            entity_texts.append(
                self._build_entity_text_for_embedding(section, entities)
            )
            sections_to_embed.append(section)

        # Generate embeddings with token-budgeted micro-batches (do not split chunks)
        if sections_to_embed:
            batch_budget = int(os.getenv("EMBED_BATCH_MAX_TOKENS", "7000") or "7000")
            if batch_budget <= 0:
                batch_budget = 7000

            batches: List[List[int]] = []
            current: List[int] = []
            current_tokens = 0
            for idx, section in enumerate(sections_to_embed):
                tokens = (
                    section.get("token_count")
                    or section.get("tokens")
                    or len(content_texts[idx].split())
                )
                if current and current_tokens + tokens > batch_budget:
                    batches.append(current)
                    current = [idx]
                    current_tokens = tokens
                else:
                    current.append(idx)
                    current_tokens += tokens
            if current:
                batches.append(current)

            content_embeddings: List[List[float]] = []
            title_embeddings: List[List[float]] = []
            sparse_embeddings: Optional[List[dict]] = (
                []
                if getattr(
                    self.embedding_settings.capabilities, "supports_sparse", False
                )
                else None
            )
            # NEW: Title sparse embeddings for lexical heading matching
            title_sparse_embeddings: Optional[List[dict]] = (
                []
                if getattr(
                    self.embedding_settings.capabilities, "supports_sparse", False
                )
                else None
            )
            # NEW: Entity sparse embeddings for lexical entity name matching
            entity_sparse_embeddings: Optional[List[dict]] = (
                []
                if getattr(
                    self.embedding_settings.capabilities, "supports_sparse", False
                )
                else None
            )
            colbert_embeddings: Optional[List[List[List[float]]]] = (
                []
                if getattr(
                    self.embedding_settings.capabilities, "supports_colbert", False
                )
                else None
            )

            for batch in batches:
                batch_content = [content_texts[i] for i in batch]
                batch_title = [title_texts[i] for i in batch]
                content_embeddings.extend(self.embedder.embed_documents(batch_content))
                title_embeddings.extend(self.embedder.embed_documents(batch_title))

                # B.2: Per-batch error isolation for sparse embeddings
                # On failure, insert None placeholders to maintain index alignment
                # This prevents one failed batch from disabling sparse for ALL chunks
                # Graph Channel Rehabilitation: Added strict mode and failure tracking
                if sparse_embeddings is not None and hasattr(
                    self.embedder, "embed_sparse"
                ):
                    try:
                        sparse_embeddings.extend(
                            self.embedder.embed_sparse(batch_content)
                        )
                    except Exception as exc:
                        stats["sparse_failures"] += 1
                        # Check if strict mode is enabled
                        sparse_strict = getattr(
                            self.config.search.vector.qdrant,
                            "sparse_strict_mode",
                            False,
                        )
                        if sparse_strict:
                            logger.error(
                                "Sparse embedding generation failed (STRICT MODE); "
                                "failing ingestion as sparse_strict_mode=true",
                                error=str(exc),
                                batch_size=len(batch_content),
                                batch_indices=batch,
                            )
                            raise RuntimeError(
                                f"Sparse embedding failed in strict mode: {exc}"
                            ) from exc
                        else:
                            logger.warning(
                                "Sparse embedding generation failed for batch; "
                                "inserting None placeholders for affected chunks",
                                error=str(exc),
                                batch_size=len(batch_content),
                                batch_indices=batch,
                            )
                            # Insert None placeholders to maintain index alignment
                            # Only this batch's chunks lose sparse; others continue normally
                            sparse_embeddings.extend([None] * len(batch_content))

                # NEW: Title sparse embeddings for lexical heading matching
                if title_sparse_embeddings is not None and hasattr(
                    self.embedder, "embed_sparse"
                ):
                    try:
                        title_sparse_embeddings.extend(
                            self.embedder.embed_sparse(batch_title)
                        )
                    except Exception as exc:
                        logger.warning(
                            "Title sparse embedding generation failed for batch; "
                            "inserting None placeholders for affected chunks",
                            error=str(exc),
                            batch_size=len(batch_title),
                            batch_indices=batch,
                        )
                        title_sparse_embeddings.extend([None] * len(batch_title))

                # NEW: Entity sparse embeddings for lexical entity name matching
                if entity_sparse_embeddings is not None and hasattr(
                    self.embedder, "embed_sparse"
                ):
                    batch_entity = [entity_texts[i] for i in batch]
                    # Only embed non-empty entity texts; use None for empty
                    non_empty_indices = [
                        i for i, text in enumerate(batch_entity) if text.strip()
                    ]
                    if non_empty_indices:
                        try:
                            # Embed only non-empty texts
                            non_empty_texts = [
                                batch_entity[i] for i in non_empty_indices
                            ]
                            non_empty_results = self.embedder.embed_sparse(
                                non_empty_texts
                            )
                            # Build full result list with None for empty texts
                            result_map = dict(zip(non_empty_indices, non_empty_results))
                            batch_results = [
                                result_map.get(i) for i in range(len(batch_entity))
                            ]
                            entity_sparse_embeddings.extend(batch_results)
                        except Exception as exc:
                            logger.warning(
                                "Entity sparse embedding generation failed for batch; "
                                "inserting None placeholders for affected chunks",
                                error=str(exc),
                                batch_size=len(non_empty_texts),
                                batch_indices=batch,
                            )
                            entity_sparse_embeddings.extend([None] * len(batch_entity))
                    else:
                        # All entity texts empty - use None placeholders
                        entity_sparse_embeddings.extend([None] * len(batch_entity))

                # B.2: Per-batch error isolation for ColBERT embeddings
                if colbert_embeddings is not None and hasattr(
                    self.embedder, "embed_colbert"
                ):
                    try:
                        colbert_embeddings.extend(
                            self.embedder.embed_colbert(batch_content)
                        )
                    except Exception as exc:
                        logger.warning(
                            "ColBERT embedding generation failed for batch; "
                            "inserting None placeholders for affected chunks",
                            error=str(exc),
                            batch_size=len(batch_content),
                            batch_indices=batch,
                        )
                        # Insert None placeholders to maintain index alignment
                        colbert_embeddings.extend([None] * len(batch_content))

            # Process each section with its embeddings
            for idx, section in enumerate(sections_to_embed):
                embedding = content_embeddings[idx]
                title_embedding = title_embeddings[idx]
                sparse_vector = (
                    sparse_embeddings[idx]
                    if sparse_embeddings is not None and idx < len(sparse_embeddings)
                    else None
                )
                # NEW: Extract title sparse vector for this section
                title_sparse_vector = (
                    title_sparse_embeddings[idx]
                    if title_sparse_embeddings is not None
                    and idx < len(title_sparse_embeddings)
                    else None
                )
                # NEW: Extract entity sparse vector for this section
                entity_sparse_vector = (
                    entity_sparse_embeddings[idx]
                    if entity_sparse_embeddings is not None
                    and idx < len(entity_sparse_embeddings)
                    else None
                )
                colbert_vector = (
                    colbert_embeddings[idx]
                    if colbert_embeddings is not None and idx < len(colbert_embeddings)
                    else None
                )

                # Graph Channel Rehabilitation: Track sparse coverage for non-stub chunks
                # Stubs are structural placeholders with empty text - expected to lack sparse
                is_stub = section.get("is_microdoc_stub", False)
                if not is_stub and sparse_embeddings is not None:
                    stats["sparse_eligible"] += 1
                    # Check if sparse_vector is valid (has indices)
                    has_valid_sparse = (
                        sparse_vector is not None
                        and isinstance(sparse_vector, dict)
                        and sparse_vector.get("indices")
                    )
                    if has_valid_sparse:
                        stats["sparse_success"] += 1
                    else:
                        # Non-stub content chunk missing sparse - this is the SLO metric
                        stats["sparse_content_missing"] += 1
                        logger.warning(
                            "Non-stub content chunk missing sparse vector",
                            section_id=section.get("id"),
                            heading=section.get("heading"),
                            token_count=section.get("token_count", 0),
                        )

                # Phase 7E-1: CRITICAL - Comprehensive validation layer

                # Validate 1: Dimension check (1024-D for Jina v3)
                if len(embedding) != self.embedding_dims:
                    raise ValueError(
                        f"Embedding dimension mismatch for section {section['id']}: "
                        f"expected {self.embedding_dims}-D, got {len(embedding)}-D. "
                        "Ingestion blocked - dimension safety enforced."
                    )

                # Validate 2: Non-empty embedding
                if not embedding or len(embedding) == 0:
                    raise ValueError(
                        f"Section {section['id']} missing REQUIRED vector_embedding. "
                        "Ingestion blocked - embeddings are mandatory in hybrid system."
                    )

                # Validate 3: Chunk schema completeness
                if not validate_chunk_schema(section):
                    raise ValueError(
                        f"Section {section['id']} missing required chunk fields. "
                        "Ingestion blocked - chunk schema validation failed."
                    )

                # Validate 4: Embedding metadata completeness
                test_metadata = canonicalize_embedding_metadata(
                    embedding_model=self.embedding_settings.version,
                    dimensions=len(embedding),
                    provider=self.embedder.provider_name,
                    task=getattr(self.embedder, "task", self.embedding_settings.task),
                    profile=self.embedding_settings.profile,
                    timestamp=datetime.utcnow(),
                )

                if not validate_embedding_metadata(
                    test_metadata,
                    expected_dimensions=self.embedding_dims,
                    expected_provider=self.embedding_settings.provider,
                    expected_version=self.embedding_settings.version,
                ):
                    raise ValueError(
                        f"Section {section['id']} has invalid embedding metadata. "
                        "Ingestion blocked - metadata validation failed."
                    )

                stats["computed"] += 1

                section["vector_embedding"] = embedding
                section["title_vector_embedding"] = title_embedding
                # REMOVED: Dense entity vector was broken (duplicated content)
                # Now using entity-sparse for lexical entity name matching

                # Upsert to vector store
                if self.vector_primary == "qdrant":
                    vectors = {
                        "content": embedding,
                        "title": title_embedding,
                    }
                    # REMOVED: Dense entity vector - replaced by entity-sparse

                    self._upsert_to_qdrant(
                        section["id"],
                        vectors,
                        section,
                        document,
                        "Section",
                        sparse_vector=sparse_vector,
                        colbert_vectors=colbert_vector,
                        title_sparse_vector=title_sparse_vector,  # NEW
                        entity_sparse_vector=entity_sparse_vector,  # NEW
                    )
                    stats["upserted"] += 1

                    # Phase 7C.7: Update Neo4j with required embedding metadata
                    # Use canonicalization helper to ensure consistent field naming
                    embedding_metadata = canonicalize_embedding_metadata(
                        embedding_model=self.embedding_settings.version,
                        dimensions=len(embedding),
                        provider=self.embedder.provider_name,
                        task=getattr(
                            self.embedder, "task", self.embedding_settings.task
                        ),
                        profile=self.embedding_settings.profile,
                        timestamp=datetime.utcnow(),
                    )
                    self._upsert_section_embedding_metadata(
                        section["id"],
                        embedding,
                        embedding_metadata,
                    )

                else:  # neo4j primary
                    self._upsert_to_neo4j_vector(section["id"], embedding, "Section")
                    stats["upserted"] += 1

                    # Dual write to Qdrant if enabled
                    if self.dual_write and self.qdrant_client:
                        vectors = {
                            "content": embedding,
                            "title": title_embedding,
                        }
                        # REMOVED: Dense entity vector - replaced by entity-sparse
                        self._upsert_to_qdrant(
                            section["id"],
                            vectors,
                            section,
                            document,
                            "Section",
                            sparse_vector=sparse_vector,
                            colbert_vectors=colbert_vector,
                            title_sparse_vector=title_sparse_vector,  # NEW
                            entity_sparse_vector=entity_sparse_vector,  # NEW
                        )

        logger.info("Embeddings processed", stats=stats)
        return stats

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

    def _build_entity_text_for_embedding(
        self, section: Dict, entities: Dict[str, Dict]
    ) -> str:
        """Build concatenated entity names for sparse embedding.

        Creates a space-separated string of unique entity names mentioned in this section.
        Used for lexical entity matching via entity-sparse vector.

        Args:
            section: Section dict containing '_mentions' list (populated by upsert_document)
            entities: Dict mapping entity_id to entity data with 'name' field

        Returns:
            Space-separated string of entity names, or empty string if no entities

        Example:
            If section mentions entities "WEKA", "NFS", and "SMB":
            Returns: "WEKA NFS SMB"
        """
        # Get mentions from section - populated by upsert_document before _process_embeddings
        mentions = section.get("_mentions", []) or []
        if not mentions:
            return ""

        entity_names = []
        for mention in mentions:
            entity_id = mention.get("entity_id")
            if entity_id and entity_id in entities:
                name = entities[entity_id].get("name", "")
                if name:
                    entity_names.append(name)

        # Deduplicate while preserving order
        unique_names = list(dict.fromkeys(entity_names))
        return " ".join(unique_names)

    def _compute_text_hash(self, text: str) -> str:
        value = text or ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _compute_shingle_hash(self, text: str, n: int = 8) -> str:
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""
        shingles = []
        limit = 64
        for i in range(0, max(0, len(tokens) - n + 1)):
            shingles.append(" ".join(tokens[i : i + n]))
            if len(shingles) >= limit:
                break
        if not shingles:
            shingles = [" ".join(tokens)]
        joined = "||".join(shingles)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _extract_semantic_metadata(self, section: Dict) -> Dict[str, Any]:
        """
        Placeholder for semantic chunk metadata (NER, topics, etc.).

        Future semantic chunkers can populate section["semantic_metadata"] before
        reaching this point; for now we record empty placeholders to keep payload
        structure consistent.
        """
        metadata = section.get("semantic_metadata")
        if metadata:
            return metadata
        return {"entities": [], "topics": []}

    def _ensure_qdrant_collection(self):
        """
        Ensure Qdrant collection exists with multi-vector schema + payload indexes.
        """
        collection = self.config.search.vector.qdrant.collection_name
        expected_suffix = get_expected_namespace_suffix(
            self.embedding_settings, self.namespace_mode
        )
        if expected_suffix and isinstance(collection, str):
            if not collection.endswith(expected_suffix):
                raise RuntimeError(
                    f"Qdrant collection {collection!r} does not match expected namespace suffix {expected_suffix!r}"
                )

        try:
            schema_plan = build_qdrant_schema(
                self.embedding_settings,
                include_entity=self.include_entity_vector,
                enable_sparse=getattr(
                    self.config.search.vector.qdrant, "enable_sparse", False
                ),
                enable_colbert=getattr(
                    self.config.search.vector.qdrant, "enable_colbert", False
                ),
                # NEW: Sparse vectors for lexical matching on titles and entities
                enable_title_sparse=getattr(
                    self.config.search.vector.qdrant, "enable_title_sparse", True
                ),
                enable_entity_sparse=getattr(
                    self.config.search.vector.qdrant, "enable_entity_sparse", True
                ),
            )
            vectors_config = schema_plan.vectors_config
            sparse_vectors_config = schema_plan.sparse_vectors_config
            payload_fields = schema_plan.payload_indexes
            if schema_plan.notes:
                for note in schema_plan.notes:
                    logger.info("Qdrant schema note", note=note)

            collections = {
                c.name for c in self.qdrant_client.get_collections().collections
            }
            if collection not in collections:
                self._create_qdrant_collection(
                    collection, vectors_config, sparse_vectors_config
                )
            else:
                self._reconcile_qdrant_collection(
                    collection, vectors_config, sparse_vectors_config
                )

            self._ensure_qdrant_payload_indexes(collection, payload_fields)
        except Exception as e:
            logger.error(
                "Failed to ensure Qdrant collection",
                collection=collection,
                error=str(e),
            )
            raise

    def _ensure_neo4j_vector_index(self) -> None:
        """
        Auto-provision the namespaced Neo4j vector index for the active profile.

        Drops conflicting vector indexes on Section.vector_embedding to avoid dual index issues.
        Uses class-level caching to avoid redundant DB calls per document.
        """
        if not self.driver:
            return

        index_name = getattr(self.config.search.vector.neo4j, "index_name", None)
        dims = self.embedding_dims
        if not index_name or not dims:
            return

        # Skip if this index has already been ensured in this process
        if index_name in GraphBuilder._neo4j_indexes_ensured:
            return

        with self.driver.session() as session:
            try:
                # Drop conflicting vector indexes on Section.vector_embedding
                check_query = """
                SHOW INDEXES
                YIELD name, type, entityType, labelsOrTypes, properties
                WHERE type = 'VECTOR'
                  AND entityType = 'NODE'
                  AND 'Section' IN labelsOrTypes
                  AND 'vector_embedding' IN properties
                RETURN name
                """
                existing = [r["name"] for r in session.run(check_query)]
                for existing_name in existing:
                    if existing_name != index_name:
                        logger.info(
                            "Dropping conflicting Neo4j vector index",
                            extra={"index": existing_name},
                        )
                        session.run(f"DROP INDEX {existing_name} IF EXISTS")

                create_query = f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (s:Section) ON (s.vector_embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {dims},
                    `vector.similarity_function`: 'COSINE',
                    `vector.hnsw.m`: 16,
                    `vector.hnsw.ef_construction`: 100,
                    `vector.quantization.enabled`: true
                  }}
                }}
                """
                session.run(create_query)
                logger.info(
                    "Ensured Neo4j vector index exists",
                    extra={"index": index_name, "dimensions": dims},
                )
                # Mark index as ensured for this process
                GraphBuilder._neo4j_indexes_ensured.add(index_name)
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.warning(
                    "Failed to auto-create Neo4j vector index",
                    extra={"index": index_name, "error": str(exc)},
                )

    def _reconcile_schema_version_embedding_metadata(self) -> None:
        """
        Update SchemaVersion node to reflect the active embedding profile.

        Uses class-level caching since embedding settings don't change between documents.
        """
        if not self.driver:
            return

        # Skip if already reconciled in this process
        if GraphBuilder._schema_metadata_ensured:
            return

        settings = self.embedding_settings
        schema_version = (
            self.expected_schema_version
            or getattr(self.config.graph_schema, "version", None)
            or "v2.2"
        )

        query = """
        MERGE (s:SchemaVersion {id: 'singleton'})
        SET s.embedding_provider = $provider,
            s.embedding_model    = $model,
            s.version            = $schema_version,
            s.vector_dimensions  = $dims,
            s.updated_at         = datetime()
        """

        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    provider=getattr(settings, "provider", None),
                    model=getattr(settings, "model_id", None),
                    schema_version=schema_version,
                    dims=self.embedding_dims,
                )
                logger.info(
                    "Updated SchemaVersion embedding metadata",
                    extra={
                        "version": schema_version,
                        "dimensions": self.embedding_dims,
                    },
                )
                # Mark as ensured for this process
                GraphBuilder._schema_metadata_ensured = True
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.warning(
                "Failed to update SchemaVersion metadata",
                extra={"error": str(exc)},
            )

    def _ensure_qdrant_payload_indexes(
        self, collection: str, fields: Sequence[tuple[str, PayloadSchemaType]]
    ) -> None:
        """Ensure payload indexes exist for canonical hybrid fields.

        Uses class-level caching to avoid redundant HTTP calls when
        GraphBuilder is instantiated multiple times (e.g., per document).
        """
        # Skip if indexes have already been ensured for this collection in this process
        if collection in GraphBuilder._payload_indexes_ensured:
            return

        for field_name, schema in fields:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                continue

        # Mark collection as having indexes ensured
        GraphBuilder._payload_indexes_ensured.add(collection)

    def _create_qdrant_collection(
        self,
        collection: str,
        vectors_config: Dict[str, VectorParams],
        sparse_vectors_config: Optional[Dict[str, SparseVectorParams]] = None,
    ) -> None:
        kwargs = {
            "collection_name": collection,
            "vectors_config": vectors_config,
            "hnsw_config": HnswConfigDiff(
                m=48,
                ef_construct=256,
                full_scan_threshold=10000,
                max_indexing_threads=0,
                on_disk=False,
            ),
            "optimizer_config": OptimizersConfigDiff(
                default_segment_number=2,
                indexing_threshold=20000,
                deleted_threshold=0.2,
                vacuum_min_vector_number=2000,
                max_optimization_threads=1,
                flush_interval_sec=5,
            ),
            "shard_number": 1,
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "on_disk_payload": True,
        }
        if sparse_vectors_config:
            kwargs["sparse_vectors_config"] = sparse_vectors_config
        try:
            self.qdrant_client.create_collection(**kwargs)
        except AssertionError as exc:
            if "optimizer_config" in str(exc):
                kwargs.pop("optimizer_config", None)
                self.qdrant_client.create_collection(**kwargs)
            else:
                raise
        logger.info("Created Qdrant collection", collection=collection)

    def _reconcile_qdrant_collection(
        self,
        collection: str,
        vectors_config: Dict[str, VectorParams],
        sparse_vectors_config: Optional[Dict[str, SparseVectorParams]] = None,
    ) -> None:
        info_payload = self._get_qdrant_collection_payload(collection)
        vectors_meta = (
            info_payload.get("config", {}).get("params", {}).get("vectors", {})
        )
        existing_names, is_single = self._parse_qdrant_vectors_meta(vectors_meta)
        desired_names = set(vectors_config.keys())
        allow_recreate = getattr(
            self.config.search.vector.qdrant, "allow_recreate", False
        )

        if is_single:
            logger.warning(
                "Qdrant collection %s uses single-vector schema; attempting upgrade",
                collection,
            )
            if not self._update_qdrant_vectors_config(collection, vectors_config):
                self._maybe_recreate_qdrant_collection(
                    collection,
                    vectors_config,
                    allow_recreate,
                    sparse_vectors_config=sparse_vectors_config,
                )
            return

        missing_names = desired_names - existing_names
        if not missing_names:
            return

        missing_config = {name: vectors_config[name] for name in missing_names}
        logger.info(
            "Adding missing Qdrant named vectors",
            collection=collection,
            missing=sorted(missing_names),
        )
        if not self._update_qdrant_vectors_config(collection, missing_config):
            self._maybe_recreate_qdrant_collection(
                collection,
                vectors_config,
                allow_recreate,
                sparse_vectors_config=sparse_vectors_config,
            )

    def _get_qdrant_collection_payload(self, collection: str) -> Dict[str, Any]:
        try:
            info = self.qdrant_client.get_collection(collection_name=collection)
            if hasattr(info, "model_dump"):
                return info.model_dump()
            if hasattr(info, "dict"):
                return info.dict()
        except Exception as exc:
            logger.debug(
                "Unable to fetch Qdrant collection metadata",
                collection=collection,
                error=str(exc),
            )
        return {}

    @staticmethod
    def _parse_qdrant_vectors_meta(meta: Any) -> (set, bool):
        """
        Returns (vector_names, is_single_vector_schema)
        """
        if isinstance(meta, dict):
            # Single-vector schema encodes fields like {"size": 1024, "distance": "Cosine", ...}
            if "size" in meta:
                return set(), True
            return set(meta.keys()), False
        return set(), True

    def _update_qdrant_vectors_config(
        self, collection: str, vectors_config: Dict[str, VectorParams]
    ) -> bool:
        try:
            self.qdrant_client.update_collection(
                collection_name=collection,
                vectors_config=vectors_config,
            )
            logger.info(
                "Updated Qdrant collection vectors",
                collection=collection,
                vectors=list(vectors_config.keys()),
            )
            return True
        except Exception as exc:
            logger.warning(
                "Unable to update Qdrant vectors config",
                collection=collection,
                error=str(exc),
            )
            return False

    def _maybe_recreate_qdrant_collection(
        self,
        collection: str,
        vectors_config: Dict[str, VectorParams],
        allow_recreate: bool,
        sparse_vectors_config: Optional[Dict[str, SparseVectorParams]] = None,
    ) -> None:
        if not allow_recreate:
            raise RuntimeError(
                "Qdrant collection schema is incompatible with multi-vector layout. "
                "Set search.vector.qdrant.allow_recreate=true to allow automatic "
                "recreation or migrate the collection manually."
            )
        logger.warning(
            "Recreating Qdrant collection with multi-vector schema",
            collection=collection,
        )
        self.qdrant_client.delete_collection(collection_name=collection)
        self._create_qdrant_collection(
            collection, vectors_config, sparse_vectors_config
        )

    def _upsert_to_qdrant(
        self,
        node_id: str,
        vectors: Dict[str, List[float]],
        section: Dict,
        document: Dict,
        label: str,
        *,
        sparse_vector: Optional[Dict[str, List[float]]] = None,
        colbert_vectors: Optional[List[List[float]]] = None,
        title_sparse_vector: Optional[Dict[str, List[float]]] = None,  # NEW
        entity_sparse_vector: Optional[Dict[str, List[float]]] = None,  # NEW
    ):
        """
        Upsert embedding to Qdrant with deterministic UUID mapping.

        Phase 7E-1: Enhanced with chunk-specific payload fields from canonical schema.

        Args:
            node_id: Chunk identifier (deterministic)
            vectors: Mapping of named vectors (content/title)
            section: Section/Chunk data
            document: Document data
            label: Node label (Section/Chunk)
            sparse_vector: Sparse embedding for text-sparse (content)
            colbert_vectors: ColBERT multi-vector for late interaction
            title_sparse_vector: Sparse embedding for title-sparse (heading)
            entity_sparse_vector: Sparse embedding for entity-sparse (entity names)
        """
        collection = self.config.search.vector.qdrant.collection_name
        expected_suffix = get_expected_namespace_suffix(
            self.embedding_settings, self.namespace_mode
        )
        if expected_suffix and isinstance(collection, str):
            if not collection.endswith(expected_suffix):
                raise RuntimeError(
                    f"Qdrant collection {collection!r} does not match expected namespace suffix {expected_suffix!r}"
                )

        if not self.qdrant_client:
            logger.warning("Qdrant client not available")
            return

        import uuid

        from qdrant_client.models import PointStruct

        source_uri = document.get("source_uri", "")
        source_path = str(Path(source_uri).parent) if source_uri else ""
        document_uri = Path(source_uri).name if source_uri else source_uri
        document_id = document.get("id") or section.get("document_id")
        doc_alias = section.get("doc_id") or document.get("doc_id") or document_id
        tenant = section.get("tenant") or document.get("tenant")
        lang = section.get("lang") or document.get("lang")
        version = section.get("version") or document.get("version")

        # Convert chunk_id to UUID for Qdrant compatibility
        # Use UUID5 with a namespace to ensure deterministic mapping
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))

        # Phase 7E-1: Create canonical embedding metadata
        content_vector = vectors.get("content")
        if not content_vector:
            raise ValueError("Content vector missing for Qdrant upsert.")

        embedding_metadata = canonicalize_embedding_metadata(
            embedding_model=self.embedding_settings.version,
            dimensions=len(content_vector),
            provider=self.embedder.provider_name,
            task=getattr(self.embedder, "task", self.embedding_settings.task),
            profile=self.embedding_settings.profile,
            timestamp=datetime.utcnow(),
            namespace_mode=self.namespace_mode,
            namespace_suffix=expected_suffix,
            collection_name=collection,
        )

        # Phase 7E-1: Build payload with canonical chunk fields
        payload = {
            # Core identifiers
            "node_id": node_id,  # Chunk ID for matching
            "kg_id": node_id,  # Explicit graph id for cross-store joins
            "node_label": label,
            "document_id": document_id,
            "doc_id": doc_alias,
            "document_uri": document_uri,
            "source_uri": source_uri,
            "source_path": source_path,
            "doc_tag": document.get("doc_tag"),
            "snapshot_scope": document.get("snapshot_scope"),
            "document_total_tokens": document.get("total_tokens"),
            "is_microdoc": section.get("is_microdoc"),
            "doc_is_microdoc": section.get("doc_is_microdoc", False),
            "is_microdoc_stub": section.get("is_microdoc_stub", False),
            # Chunk-specific fields (Phase 7E-1)
            "id": node_id,
            "parent_section_id": section.get("parent_section_id"),
            "parent_section_original_id": section.get("parent_section_original_id"),
            "parent_chunk_id": section.get("parent_chunk_id"),
            "level": section.get("level", 3),
            "order": section.get("order", 0),
            "heading": section.get("title") or section.get("heading", ""),
            "text": section.get("text", ""),
            "token_count": section.get("token_count") or section.get("tokens", 0),
            "is_combined": section.get("is_combined", False),
            "is_split": section.get("is_split", False),
            "original_section_ids": section.get(
                "original_section_ids", [section.get("id")]
            ),
            "boundaries_json": section.get("boundaries_json", "{}"),
            "document_total_tokens_chunk": section.get("document_total_tokens"),
            "tenant": tenant,
            "lang": lang,
            "version": version,
            # Legacy fields (for compatibility)
            "title": section.get("title"),
            "anchor": section.get("anchor"),
            # Timestamps
            "updated_at": datetime.utcnow().isoformat() + "Z",
            # Canonical embedding fields
            **embedding_metadata,
        }

        vectors_payload = dict(vectors)

        if sparse_vector:
            indices = (
                sparse_vector.get("indices")
                if isinstance(sparse_vector, dict)
                else None
            )
            values = (
                sparse_vector.get("values") if isinstance(sparse_vector, dict) else None
            )
            if indices and values:
                vectors_payload["text-sparse"] = SparseVector(
                    indices=list(indices), values=list(values)
                )

        # NEW: title-sparse - lexical matching for section headings
        if title_sparse_vector:
            indices = (
                title_sparse_vector.get("indices")
                if isinstance(title_sparse_vector, dict)
                else None
            )
            values = (
                title_sparse_vector.get("values")
                if isinstance(title_sparse_vector, dict)
                else None
            )
            if indices and values:
                vectors_payload["title-sparse"] = SparseVector(
                    indices=list(indices), values=list(values)
                )

        # NEW: entity-sparse - lexical matching for entity names
        if entity_sparse_vector:
            indices = (
                entity_sparse_vector.get("indices")
                if isinstance(entity_sparse_vector, dict)
                else None
            )
            values = (
                entity_sparse_vector.get("values")
                if isinstance(entity_sparse_vector, dict)
                else None
            )
            if indices and values:
                vectors_payload["entity-sparse"] = SparseVector(
                    indices=list(indices), values=list(values)
                )

        if colbert_vectors:
            vectors_payload["late-interaction"] = [
                list(vector) for vector in colbert_vectors
            ]
            payload["colbert_vector_count"] = len(colbert_vectors)

        # Ensure no legacy embedding_model field in payload
        payload = ensure_no_embedding_model_in_payload(payload)

        # CRITICAL: Store original node_id in payload for reconciliation
        # Parity checks will use payload.node_id to match with Neo4j
        text_value = payload.get("text", "")
        existing_text_hash = section.get("text_hash")
        payload["text_hash"] = (
            existing_text_hash
            if existing_text_hash
            else self._compute_text_hash(text_value)
        )
        existing_shingle_hash = section.get("shingle_hash")
        payload["shingle_hash"] = (
            existing_shingle_hash
            if existing_shingle_hash
            else self._compute_shingle_hash(text_value)
        )
        payload["semantic_metadata"] = self._extract_semantic_metadata(section)

        # qdrant-client expects the keyword `vector` even when we supply a named
        # vector dictionary. Passing `vectors=` raises a validation error on older
        # client builds, so always map to `vector`.
        point = PointStruct(id=point_uuid, vector=vectors_payload, payload=payload)

        # Optional debug: dump the exact payload structure we're sending to Qdrant.
        # Disabled by default; enable by setting LOG_QDRANT_DEBUG_PAYLOAD=true (not recommended in production).
        if os.getenv("LOG_QDRANT_DEBUG_PAYLOAD", "false").lower() == "true":
            try:  # pragma: no cover - debug aid
                from qdrant_client.models import PointsList

                points_list = PointsList(points=[point])
                logger.error(
                    "Qdrant debug upsert payload",
                    collection=collection,
                    points_list=points_list.model_dump(),
                )
            except Exception as exc:  # pragma: no cover - best-effort logging
                logger.error("Qdrant debug payload logging failed", error=str(exc))

        expected_dims = {}
        for name, vec in vectors_payload.items():
            dim = self._vector_expected_dim(vec)
            if dim:
                expected_dims[name] = dim

        # Use validated upsert with dimension checking
        self.qdrant_client.upsert_validated(
            collection_name=collection,
            points=[point],
            expected_dim=expected_dims,
        )

        logger.debug(
            "Chunk vector upserted to Qdrant with canonical schema",
            node_id=node_id,
            point_uuid=point_uuid,
            collection=collection,
            provider=embedding_metadata["embedding_provider"],
            dimensions=expected_dims,
            is_combined=payload["is_combined"],
            original_section_count=len(payload["original_section_ids"]),
        )

    @staticmethod
    def _vector_expected_dim(vector: Any) -> Optional[int]:
        """Return the expected dimension for validation (if applicable)."""
        if isinstance(vector, list):
            if not vector:
                return None
            first = vector[0]
            if isinstance(first, list):
                return len(first)
            return len(vector)
        return None

    def _upsert_to_neo4j_vector(self, node_id: str, embedding: List[float], label: str):
        """Upsert embedding to Neo4j vector property."""
        query = f"""
        MATCH (n:{label} {{id: $node_id}})
        SET n.vector_embedding = $embedding,
            n.embedding_version = $version
        RETURN n.id as id
        """

        with self.driver.session() as session:
            session.run(
                query,
                node_id=node_id,
                embedding=embedding,
                version=self.embedding_version,
            )

        logger.debug(
            "Vector upserted to Neo4j",
            node_id=node_id,
            label=label,
        )

    def _set_embedding_version_in_neo4j(self, node_id: str, label: str):
        """Set embedding_version metadata in Neo4j without storing vector."""
        query = f"""
        MATCH (n:{label} {{id: $node_id}})
        SET n.embedding_version = $version
        RETURN n.id as id
        """

        with self.driver.session() as session:
            session.run(
                query,
                node_id=node_id,
                version=self.embedding_version,
            )

        logger.debug(
            "Embedding version set in Neo4j",
            node_id=node_id,
            label=label,
        )

    def _upsert_section_embedding_metadata(
        self, node_id: str, embedding: List[float], metadata: Dict
    ):
        """
        Update Section node with embedding vector and all required metadata.

        Phase 7C.7: Enforces schema v2.1 required embedding fields (Session 06-08).
        All fields are REQUIRED - ingestion fails if any are missing.

        Args:
            node_id: Section ID
            embedding: Embedding vector (stored in Neo4j for tracking)
            metadata: Dict with embedding metadata fields (all required)
        """
        # Validate metadata using canonicalization helper
        if not validate_embedding_metadata(
            metadata,
            expected_dimensions=self.embedding_dims,
            expected_provider=self.embedding_settings.provider,
            expected_version=self.embedding_settings.version,
        ):
            raise ValueError(
                f"Section {node_id} has invalid embedding metadata. "
                "Ingestion blocked - metadata validation failed."
            )

        # Ensure no legacy embedding_model field in metadata
        metadata = ensure_no_embedding_model_in_payload(metadata)

        query = """
        MATCH (s:Section {id: $node_id})
        SET s.vector_embedding = $vector_embedding,
            s.embedding_version = $embedding_version,
            s.embedding_provider = $embedding_provider,
            s.embedding_dimensions = $embedding_dimensions,
            s.embedding_timestamp = $embedding_timestamp,
            s.embedding_task = $embedding_task
        RETURN s.id as id
        """

        with self.driver.session() as session:
            session.run(
                query,
                node_id=node_id,
                vector_embedding=embedding,
                embedding_version=metadata["embedding_version"],
                embedding_provider=metadata["embedding_provider"],
                embedding_dimensions=metadata["embedding_dimensions"],
                embedding_timestamp=metadata["embedding_timestamp"],
                embedding_task=metadata.get("embedding_task", "retrieval.passage"),
            )

        logger.debug(
            "Section embedding and metadata updated in Neo4j",
            node_id=node_id,
            provider=metadata["embedding_provider"],
            dimensions=metadata["embedding_dimensions"],
            vector_stored=True,
        )

    def _invalidate_caches_post_ingest(
        self, document_id: str, chunk_ids: List[str]
    ) -> Dict[str, any]:
        """
        Invalidate caches after successful Neo4j + Qdrant commits.

        Phase 7E-3: Cache invalidation MUST happen AFTER both stores commit
        to avoid race conditions where stale cache is repopulated before
        epoch bump.

        Supports two modes:
        - epoch (preferred): O(1) invalidation by bumping epoch counters
        - scan (fallback): Pattern-scan deletion

        Args:
            document_id: Document identifier
            chunk_ids: List of chunk identifiers that were upserted

        Returns:
            Dict with invalidation stats

        Reference: Canonical Spec L3313-3336, L4506-4539
        """
        cache_mode = os.getenv("CACHE_MODE", "epoch")
        redis_url = (
            os.getenv("CACHE_REDIS_URI")
            or os.getenv("REDIS_URL")
            or os.getenv("REDIS_URI")
            or "redis://localhost:6379/0"
        )
        namespace = os.getenv("CACHE_NS", "rag:v1")

        stats = {
            "mode": cache_mode,
            "document_id": document_id,
            "chunk_count": len(chunk_ids),
            "success": False,
            "keys_deleted": 0,
            "epoch_bumped": False,
        }

        try:
            if cache_mode == "epoch":
                # Preferred: O(1) epoch bump
                from tools.redis_epoch_bump import (
                    bump_chunk_epochs,
                    bump_doc_epoch,
                )

                r = redis.Redis.from_url(redis_url, decode_responses=True)

                # Bump document epoch
                doc_epoch = bump_doc_epoch(r, namespace, document_id)
                stats["doc_epoch"] = doc_epoch

                # Bump chunk epochs
                if chunk_ids:
                    chunk_total = bump_chunk_epochs(r, namespace, chunk_ids)
                    stats["chunk_epoch_total"] = chunk_total

                stats["epoch_bumped"] = True
                stats["success"] = True

                logger.info(
                    "Cache invalidation (epoch) complete",
                    document_id=document_id,
                    doc_epoch=doc_epoch,
                    chunks=len(chunk_ids),
                )

            else:
                # Fallback: pattern-scan deletion
                from tools.redis_invalidation import invalidate

                deleted = invalidate(redis_url, namespace, document_id, chunk_ids)
                stats["keys_deleted"] = deleted
                stats["success"] = True

                logger.info(
                    "Cache invalidation (scan) complete",
                    document_id=document_id,
                    keys_deleted=deleted,
                    chunks=len(chunk_ids),
                )

        except Exception as e:
            logger.warning(
                "Cache invalidation failed (non-fatal)",
                document_id=document_id,
                mode=cache_mode,
                error=str(e),
            )
            stats["error"] = str(e)

        return stats


# Integration test wrapper
def ingest_document(
    source_uri: str,
    content: str,
    format: str = "markdown",
    *,
    embedding_model: Optional[str] = None,
    embedding_version: Optional[str] = None,
) -> Dict:
    """
    Top-level function for ingesting a document.

    Args:
        source_uri: URI of the document
        content: Document content
        format: Document format (markdown, html, notion)

    Returns:
        Ingestion stats
    """
    from neo4j import GraphDatabase

    from src.ingestion.extract import extract_entities
    from src.ingestion.parsers import parse_markdown  # Router selects engine
    from src.ingestion.parsers.html import parse_html
    from src.shared.config import get_config, get_settings
    from src.shared.connections import CompatQdrantClient

    config_global = get_config()
    settings = get_settings()

    # Work on a deep copy of the global config to avoid mutating
    # process-wide state when per-call overrides are supplied.
    try:
        config = config_global.model_copy(deep=True)  # pydantic v2
    except Exception:
        # Fallback if model_copy is unavailable (defensive; not expected)
        # Note: Our models are pydantic v2; this branch should not be hit.
        import copy as _copy

        config = _copy.deepcopy(config_global)

    # Apply optional per-call overrides on the local config copy only
    if embedding_model:
        try:
            # Correct attribute is `embedding_model` (alias `model_name` is for input)
            config.embedding.embedding_model = embedding_model
        except Exception as exc:
            logger.warning(
                "Failed to override embedding model via ingest_document",
                requested_model=embedding_model,
                error=str(exc),
            )
    if embedding_version:
        try:
            config.embedding.version = embedding_version
        except Exception as exc:
            logger.warning(
                "Failed to override embedding version via ingest_document",
                requested_version=embedding_version,
                error=str(exc),
            )

    # Initialize clients
    neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=3600,
    )

    qdrant_client = None
    if config.search.vector.primary == "qdrant" or config.search.vector.dual_write:
        qdrant_client = CompatQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=30
        )

    try:
        # Parse document
        if format == "markdown":
            result = parse_markdown(source_uri, content)
        elif format == "html":
            result = parse_html(source_uri, content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        document = result["Document"]
        sections = result["Sections"]

        # Extract per-document doc_tag and snapshot_scope
        # Rules:
        # 1) Prefer explicit "DocTag: <TAG>" in content for doc_tag.
        # 2) Else derive from filename stem. If stem is "<scope>__<slug>",
        #    treat <scope> as snapshot_scope and <slug> as doc_tag; otherwise
        #    use the full stem as doc_tag.
        doc_tag = None
        snapshot_scope = None

        m = re.search(r"DocTag:\s*([A-Za-z0-9_\-]+)", content or "", flags=re.I)
        if m:
            doc_tag = m.group(1)
        else:
            try:
                fname = Path(source_uri or "").name
                stem = Path(fname).stem
                if "__" in stem:
                    scope_part, slug_part = stem.split("__", 1)
                    snapshot_scope = scope_part
                    doc_tag = slug_part
                else:
                    doc_tag = stem
            except Exception:
                doc_tag = None
                snapshot_scope = None

        document["doc_tag"] = doc_tag
        document["snapshot_scope"] = snapshot_scope
        for section in sections:
            section["doc_tag"] = doc_tag
            section["snapshot_scope"] = snapshot_scope

        # Extract entities
        entities, mentions = extract_entities(sections)

        # Build graph
        builder = GraphBuilder(neo4j_driver, config, qdrant_client)
        stats = builder.upsert_document(document, sections, entities, mentions)

        return stats
    finally:
        neo4j_driver.close()

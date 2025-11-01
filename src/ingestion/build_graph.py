# Implements Phase 3, Task 3.3 (Graph construction with embeddings)
# See: /docs/spec.md §3 (Data model, IDs, vectors)
# See: /docs/implementation-plan.md → Task 3.3
# See: /docs/pseudocode-reference.md → Task 3.3
# Pre-Phase 7 B3: Modified to use embedding provider abstraction

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import redis
from neo4j import Driver

from src.ingestion.chunk_assembler import get_chunk_assembler

# Phase 7E-4: Monitoring imports
from src.monitoring.metrics import MetricsCollector
from src.monitoring.slos import check_slos_and_log
from src.shared.chunk_utils import (
    create_chunk_metadata,
    validate_chunk_schema,
)
from src.shared.config import Config
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
    ingestion_duration_seconds,
)

logger = get_logger(__name__)


class GraphBuilder:
    """Builds graph from parsed documents, sections, and entities."""

    def __init__(self, driver: Driver, config: Config, qdrant_client=None):
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
        self.embedding_version = config.embedding.version
        self.vector_primary = config.search.vector.primary
        self.dual_write = config.search.vector.dual_write

        # Phase 7C.7: Fresh start with 1024-D (Session 06-08)
        # No dual-write complexity - starting fresh with Jina v4 @ 1024-D

        # Ensure Qdrant collections exist if using Qdrant
        if self.qdrant_client and (self.vector_primary == "qdrant" or self.dual_write):
            self._ensure_qdrant_collection()

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
        assembler = get_chunk_assembler()
        sections = assembler.assemble(document["id"], sections)

        with self.driver.session() as session:
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

            # Optional Phase 7E repair: correct legacy combined flags
            repaired = self._repair_incorrect_combined_flags(session, document["id"])
            if repaired:
                logger.info(
                    "Repaired incorrect combined flag assignments",
                    document_id=document["id"],
                    repaired_count=repaired,
                )

            # Step 3: Upsert Entities in batches
            stats["entities_upserted"] = self._upsert_entities(session, entities)

            # Step 4: Create MENTIONS edges in batches
            stats["mentions_created"] = self._create_mentions(session, mentions)

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
            d.updated_at = datetime()
        RETURN d.id as id
        """
        session.run(query, **document)
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
                s.text = chunk.text,
                s.tokens = chunk.tokens,
                s.token_count = chunk.token_count,
                s.checksum = chunk.checksum,
                s.original_section_ids = chunk.original_section_ids,
                s.is_combined = chunk.is_combined,
                s.is_split = chunk.is_split,
                s.boundaries_json = chunk.boundaries_json,
                s.updated_at = datetime()
            WITH s, chunk
            MATCH (d:Document {id: $document_id})
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
                rels.append({
                    "unit_id": unit["id"],
                    "chunk_id": unit["parent_chunk_id"],
                })

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
        """Upsert Entity nodes in batches."""
        batch_size = self.config.ingestion.batch_size
        entities_list = list(entities.values())
        total_entities = 0

        # Group by label for efficient batching
        by_label = {}
        for entity in entities_list:
            label = entity["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(entity)

        for label, entity_batch in by_label.items():
            for i in range(0, len(entity_batch), batch_size):
                batch = entity_batch[i : i + batch_size]

                # Dynamic label in query
                query = f"""
                UNWIND $entities as ent
                MERGE (e:{label} {{id: ent.id}})
                SET e.name = ent.name,
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
        Supports dynamic relationship types (e.g., CONTAINS_STEP, REQUIRES, AFFECTS).
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
                # Using CALL {} subquery to work around Cypher's limitation
                # on parameterized relationship types
                query = f"""
                UNWIND $rels as r
                MATCH (from {{id: r.from_id}})
                MATCH (to {{id: r.to_id}})
                CALL {{
                    WITH from, to, r
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
        }

        # Phase 7C.7: Initialize embedding provider from factory
        if not self.embedder:
            from src.providers.factory import ProviderFactory

            logger.info(
                "Initializing embedding provider",
                provider=self.config.embedding.provider,
                model=self.config.embedding.embedding_model,
                dims=self.config.embedding.dims,
            )

            # Use ProviderFactory to create embedding provider
            self.embedder = ProviderFactory.create_embedding_provider()

            # Validate dimensions match configuration
            if self.embedder.dims != self.config.embedding.dims:
                raise ValueError(
                    f"Provider dimension mismatch: expected {self.config.embedding.dims}, "
                    f"got {self.embedder.dims}"
                )

            logger.info(
                "Embedding provider initialized",
                provider_name=self.embedder.provider_name,
                model_id=self.embedder.model_id,
                actual_dims=self.embedder.dims,
            )

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
        for section in sections:
            section.setdefault("source_uri", source_uri)
            section.setdefault("document_uri", document_uri)
            text_to_embed = self._build_section_text_for_embedding(section)
            sections_to_embed.append((section, text_to_embed))

        # Generate embeddings in batch
        if sections_to_embed:
            texts = [text for _, text in sections_to_embed]

            # Generate embeddings using configured provider (Jina v3 @ 1024-D)
            embeddings = self.embedder.embed_documents(texts)

            # Process each section with its embedding
            for (section, _), embedding in zip(sections_to_embed, embeddings):
                # Phase 7E-1: CRITICAL - Comprehensive validation layer

                # Validate 1: Dimension check (1024-D for Jina v3)
                if len(embedding) != self.config.embedding.dims:
                    raise ValueError(
                        f"Embedding dimension mismatch for section {section['id']}: "
                        f"expected {self.config.embedding.dims}-D, got {len(embedding)}-D. "
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
                    embedding_model=self.config.embedding.version,
                    dimensions=len(embedding),
                    provider=self.embedder.provider_name,
                    task=getattr(self.embedder, "task", "retrieval.passage"),
                    timestamp=datetime.utcnow(),
                )

                if not validate_embedding_metadata(test_metadata):
                    raise ValueError(
                        f"Section {section['id']} has invalid embedding metadata. "
                        "Ingestion blocked - metadata validation failed."
                    )

                stats["computed"] += 1

                # Upsert to vector store
                if self.vector_primary == "qdrant":
                    self._upsert_to_qdrant(
                        section["id"], embedding, section, document, "Section"
                    )
                    stats["upserted"] += 1

                    # Phase 7C.7: Update Neo4j with required embedding metadata
                    # Use canonicalization helper to ensure consistent field naming
                    embedding_metadata = canonicalize_embedding_metadata(
                        embedding_model=self.config.embedding.version,  # Maps to embedding_version
                        dimensions=len(embedding),
                        provider=self.embedder.provider_name,
                        task=getattr(self.embedder, "task", "retrieval.passage"),
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
                        self._upsert_to_qdrant(
                            section["id"], embedding, section, document, "Section"
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

    def _ensure_qdrant_collection(self):
        """
        Ensure Qdrant collection exists with correct schema.

        Phase 7C.7: Simplified for fresh start - creates single collection at configured dimensions.
        """
        from qdrant_client.models import Distance, VectorParams

        collection = self.config.search.vector.qdrant.collection_name
        dimensions = self.config.embedding.dims

        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            exists = any(c.name == collection for c in collections.collections)

            if not exists:
                # Create collection with vector config
                self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=dimensions,
                        distance=(
                            Distance.COSINE
                            if self.config.embedding.similarity == "cosine"
                            else Distance.EUCLID
                        ),
                    ),
                )
                logger.info(
                    "Created Qdrant collection",
                    collection=collection,
                    dims=dimensions,
                )
            else:
                logger.debug("Qdrant collection already exists", collection=collection)
        except Exception as e:
            logger.error(
                "Failed to ensure Qdrant collection",
                collection=collection,
                error=str(e),
            )
            raise

    def _upsert_to_qdrant(
        self,
        node_id: str,
        embedding: List[float],
        section: Dict,
        document: Dict,
        label: str,
    ):
        """
        Upsert embedding to Qdrant with deterministic UUID mapping.

        Phase 7E-1: Enhanced with chunk-specific payload fields from canonical schema.

        Args:
            node_id: Chunk identifier (deterministic)
            embedding: Embedding vector (1024-D)
            section: Section/Chunk data
            document: Document data
            label: Node label (Section/Chunk)
        """
        if not self.qdrant_client:
            logger.warning("Qdrant client not available")
            return

        import uuid

        from qdrant_client.models import PointStruct

        collection = self.config.search.vector.qdrant.collection_name

        source_uri = document.get("source_uri", "")
        document_uri = Path(source_uri).name if source_uri else source_uri
        document_id = document.get("id") or section.get("document_id")

        # Convert chunk_id to UUID for Qdrant compatibility
        # Use UUID5 with a namespace to ensure deterministic mapping
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))

        # Phase 7E-1: Create canonical embedding metadata
        embedding_metadata = canonicalize_embedding_metadata(
            embedding_model=self.config.embedding.version,  # Maps to embedding_version
            dimensions=len(embedding),
            provider=self.embedder.provider_name,
            task=getattr(self.embedder, "task", "retrieval.passage"),
            timestamp=datetime.utcnow(),
        )

        # Phase 7E-1: Build payload with canonical chunk fields
        payload = {
            # Core identifiers
            "node_id": node_id,  # Chunk ID for matching
            "node_label": label,
            "document_id": document_id,
            "document_uri": document_uri,
            "source_uri": source_uri,
            # Chunk-specific fields (Phase 7E-1)
            "id": node_id,
            "parent_section_id": section.get("parent_section_id"),
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
            # Legacy fields (for compatibility)
            "title": section.get("title"),
            "anchor": section.get("anchor"),
            # Timestamps
            "updated_at": datetime.utcnow().isoformat() + "Z",
            # Canonical embedding fields
            **embedding_metadata,
        }

        # Ensure no legacy embedding_model field in payload
        payload = ensure_no_embedding_model_in_payload(payload)

        # CRITICAL: Store original node_id in payload for reconciliation
        # Parity checks will use payload.node_id to match with Neo4j
        point = PointStruct(
            id=point_uuid,  # UUID compatible with Qdrant
            vector=embedding,
            payload=payload,
        )

        # Use validated upsert with dimension checking
        self.qdrant_client.upsert_validated(
            collection_name=collection,
            points=[point],
            expected_dim=embedding_metadata["embedding_dimensions"],
        )

        logger.debug(
            "Chunk vector upserted to Qdrant with canonical schema",
            node_id=node_id,
            point_uuid=point_uuid,
            collection=collection,
            provider=embedding_metadata["embedding_provider"],
            dimensions=embedding_metadata["embedding_dimensions"],
            is_combined=payload["is_combined"],
            original_section_count=len(payload["original_section_ids"]),
        )

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
        if not validate_embedding_metadata(metadata):
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
    from src.ingestion.parsers.markdown import parse_markdown
    from src.shared.config import get_config, get_settings
    from src.shared.connections import CompatQdrantClient

    config = get_config()
    settings = get_settings()

    # Allow optional overrides prior to ingestion
    if embedding_model:
        try:
            config.embedding.model_name = embedding_model
        except Exception:
            logger.warning(
                "Failed to override embedding model via ingest_document",
                requested_model=embedding_model,
            )
    if embedding_version:
        try:
            config.embedding.version = embedding_version
        except Exception:
            logger.warning(
                "Failed to override embedding version via ingest_document",
                requested_version=embedding_version,
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
        else:
            raise ValueError(f"Unsupported format: {format}")

        document = result["Document"]
        sections = result["Sections"]

        # Extract entities
        entities, mentions = extract_entities(sections)

        # Build graph
        builder = GraphBuilder(neo4j_driver, config, qdrant_client)
        stats = builder.upsert_document(document, sections, entities, mentions)

        return stats
    finally:
        neo4j_driver.close()

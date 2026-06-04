# =============================================================================
# @status: ACTIVE
# @called-by: atomic.py
# =============================================================================
"""
Neo4j write operations for atomic ingestion.

This module contains all Neo4j transaction-based write operations extracted from
atomic.py to reduce file size and improve maintainability. The AtomicIngestionCoordinator
uses a Neo4jWriter instance to perform all graph database writes.

Key responsibilities:
- Upsert Document, Chunk, and Entity nodes
- Create MENTIONS, REFERENCES, and entity relationship edges
- Handle ghost document resolution and pending reference conversion
- Sanitize data for Neo4j property storage
- Compute content hashes for drift detection
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List

from src.shared.observability import get_logger
from src.shared.observability.metrics import (
    entity_relationships_missing_total,
    entity_relationships_total,
)

logger = get_logger(__name__)

# Allowlist of valid Entity->Entity relationship types for Cypher injection defense
# Phase 1.2: Only these rel types are allowed to be interpolated into Cypher
# Phase 2 Cleanup: Removed DEPENDS_ON and REQUIRES (never materialized by ingestion)
ALLOWED_ENTITY_RELATIONSHIP_TYPES = frozenset(
    {
        "CONTAINS_STEP",  # Procedure->Step ordering
        "REFERENCES",  # Cross-document references (Phase 3)
        "CONFIGURES",  # Config->Component
        "RESOLVES",  # Error->Procedure
    }
)


class Neo4jWriter:
    """
    Handles all Neo4j write operations for atomic ingestion.

    This class provides transaction-based write methods that are called by
    the AtomicIngestionCoordinator during the Neo4j write phase of ingestion.
    """

    def __init__(self, config=None):
        """Initialize the Neo4j writer.

        Args:
            config: Application configuration (used by _neo4j_create_references_streaming
                    for batch size and fuzzy resolution settings).
        """
        self.config = config

    def _sanitize_for_neo4j(self, data: Dict) -> Dict:
        """
        Sanitize a dict for Neo4j property storage.

        Neo4j only accepts primitive types (str, int, float, bool, None)
        or arrays of primitives. Nested dicts/maps cause TypeError.

        Strategy:
        - Keep primitives and None
        - Keep lists of primitives
        - JSON-serialize nested dicts/lists with dicts
        - Skip fields that can't be safely serialized
        """
        sanitized = {}
        for key, value in data.items():
            if value is None:
                continue  # Skip None values entirely

            # Primitives are safe
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
                continue

            # Lists need inspection
            if isinstance(value, list):
                if not value:
                    # Empty list is safe
                    sanitized[key] = value
                elif all(isinstance(item, (str, int, float, bool)) for item in value):
                    # List of primitives is safe
                    sanitized[key] = value
                else:
                    # List contains complex types - JSON serialize
                    try:
                        sanitized[key] = json.dumps(value)
                    except (TypeError, ValueError):
                        logger.debug(
                            "neo4j_sanitize_skip_field",
                            field=key,
                            reason="list_not_serializable",
                        )
                continue

            # Dicts must be JSON serialized
            if isinstance(value, dict):
                try:
                    sanitized[key] = json.dumps(value)
                except (TypeError, ValueError):
                    logger.debug(
                        "neo4j_sanitize_skip_field",
                        field=key,
                        reason="dict_not_serializable",
                    )
                continue

            # Other types - try to convert to string
            try:
                sanitized[key] = str(value)
            except Exception:
                logger.debug(
                    "neo4j_sanitize_skip_field",
                    field=key,
                    reason="unconvertible_type",
                    value_type=type(value).__name__,
                )

        return sanitized

    def _neo4j_upsert_document(self, tx, document: Dict):
        """Upsert document node within a transaction.

        Phase 3 Enhancement: Also resolves Ghost Documents and PendingReferences
        that match the newly ingested document's title.
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d += $props
        """
        props = self._sanitize_for_neo4j(document)
        tx.run(query, id=document["id"], props=props)

        # Phase 3: Resolve Ghost Documents with matching title
        # When a real document is ingested, redirect REFERENCES edges from any
        # Ghost Document that was created as a forward reference placeholder
        title = document.get("title", "")
        if not title:
            # Fallback: derive title from document ID (usually contains filename)
            doc_id = document.get("id", "")
            if doc_id:
                from src.ingestion.extract.references import normalize_filename_to_title

                filename = doc_id.split("/")[-1]
                title = normalize_filename_to_title(filename)
                document["title"] = title
                logger.warning(
                    "empty_title_fallback",
                    document_id=document.get("id"),
                    derived_title=title,
                    reason="document_missing_title",
                )
        if title:
            title_cf = title.casefold()
            # Fix #3 (Phase 3): Atomic ghost resolution with explicit locking
            # The lock prevents race conditions when multiple processes try to:
            # 1. Resolve the same ghost simultaneously
            # 2. Create new edges to a ghost that's being deleted
            #
            # The lock pattern works by:
            # 1. SET ghost._resolve_lock = $doc_id (atomic claim)
            # 2. Only proceed if ghost._resolve_lock = $doc_id (verify we won)
            # 3. Delete ghost atomically with edge cleanup
            #
            # If another process wins the lock, we gracefully skip (the ghost
            # will either be resolved or still exist for our next attempt).
            resolve_ghost_query = """
            // Find Ghost Documents with matching title (case-insensitive)
            // Fix #3: Atomically claim lock before proceeding
            MATCH (ghost:GhostDocument)
            WHERE toLower(ghost.title) = toLower($title_cf)
              AND ghost._resolve_lock IS NULL
            // Atomic lock acquisition - only claim if currently unlocked
            SET ghost._resolve_lock = $doc_id
            WITH ghost
            // Verify we successfully acquired the lock (another tx may have won)
            WHERE ghost._resolve_lock = $doc_id
            // Find all REFERENCES edges pointing to the ghost
            OPTIONAL MATCH (src)-[old_r:REFERENCES]->(ghost)
            // Get the real document we just upserted
            MATCH (real:Document {id: $doc_id})
            WHERE NOT real:GhostDocument
            // Use FOREACH to conditionally create edges only if old edges exist
            // This avoids issues when there are no edges to redirect
            WITH ghost, real, collect({src: src, old_r: old_r}) AS edges
            UNWIND CASE WHEN size(edges) > 0 AND edges[0].old_r IS NOT NULL
                        THEN edges ELSE [] END AS edge
            WITH ghost, real, edge.src AS src, edge.old_r AS old_r
            // Create new edge to real document with same properties
            CREATE (src)-[new_r:REFERENCES]->(real)
            SET new_r = properties(old_r),
                new_r.is_ghost_target = null,
                new_r.resolved_from_ghost = ghost.id,
                new_r.resolved_at = datetime({timezone: 'UTC'})
            WITH ghost, old_r, count(new_r) AS redirected
            // Delete old edge
            DELETE old_r
            WITH ghost, sum(redirected) AS total_redirected
            // Atomically delete ghost - at this point we hold the lock
            // so no other process can add edges
            DETACH DELETE ghost
            RETURN total_redirected AS redirected
            """
            result = tx.run(
                resolve_ghost_query,
                title=title,
                title_cf=title_cf,
                doc_id=document["id"],
            )
            record = result.single()
            if record and record["redirected"] > 0:
                logger.info(
                    "ghost_references_resolved",
                    document_id=document["id"],
                    title=title,
                    redirected_count=record["redirected"],
                )

            # Phase 3: Resolve PendingReferences with matching hint
            # When a real document is ingested, convert pending references that
            # match the document's title into real REFERENCES edges
            # Fix #3: Apply same atomic locking pattern as ghost resolution
            resolve_pending_query = """
            // Find PendingReferences where hint matches document title
            // Fix #3: Atomically claim lock before proceeding
            MATCH (pending:PendingReference)
            WHERE (toLower($title_cf) CONTAINS toLower(pending.hint)
               OR toLower(pending.hint) CONTAINS toLower($title_cf))
              AND pending._resolve_lock IS NULL
            // Atomic lock acquisition
            SET pending._resolve_lock = $doc_id
            WITH pending
            // Verify we successfully acquired the lock
            WHERE pending._resolve_lock = $doc_id
            // Find all PENDING_REF edges
            OPTIONAL MATCH (src)-[old_r:PENDING_REF]->(pending)
            // Get the real document
            MATCH (real:Document {id: $doc_id})
            // Collect edges to handle empty case gracefully
            WITH pending, real, collect({src: src, old_r: old_r}) AS edges
            UNWIND CASE WHEN size(edges) > 0 AND edges[0].old_r IS NOT NULL
                        THEN edges ELSE [] END AS edge
            WITH pending, real, edge.src AS src, edge.old_r AS old_r
            // Create real REFERENCES edge
            CREATE (src)-[new_r:REFERENCES]->(real)
            SET new_r.type = old_r.type,
                new_r.reference_text = old_r.reference_text,
                new_r.confidence = old_r.confidence,
                new_r.source_type = old_r.source_type,
                new_r.target_hint = pending.hint,
                new_r.resolved_from_pending = true,
                new_r.created_at = old_r.created_at,
                new_r.resolved_at = datetime({timezone: 'UTC'})
            WITH pending, old_r, count(new_r) AS resolved
            // Delete old PENDING_REF edge
            DELETE old_r
            WITH pending, sum(resolved) AS total_resolved
            // Atomically delete pending - we hold the lock
            DELETE pending
            RETURN total_resolved AS resolved
            """
            result = tx.run(
                resolve_pending_query,
                title=title,
                title_cf=title_cf,
                doc_id=document["id"],
            )
            record = result.single()
            if record and record["resolved"] > 0:
                logger.info(
                    "pending_references_resolved",
                    document_id=document["id"],
                    title=title,
                    resolved_count=record["resolved"],
                )

    def _neo4j_upsert_sections(self, tx, document_id: str, sections: List[Dict]) -> int:
        """Upsert chunk nodes within a transaction."""
        # P0: HAS_SECTION deprecated - use only HAS_CHUNK for document->chunk membership
        query = """
        UNWIND $sections AS section
        MERGE (c:Chunk {id: section.id})
        SET c += section
        SET c.chunk_id = coalesce(c.chunk_id, section.chunk_id, section.id)
        WITH c, section
        MATCH (d:Document {id: $document_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """

        # Prepare sections for Cypher - sanitize to avoid Map{} errors
        section_data = []
        for s in sections:
            # Filter out internal fields
            data = {k: v for k, v in s.items() if k != "_citation_units"}
            # Ensure required fields
            data.setdefault("document_id", document_id)
            data.setdefault("chunk_id", data.get("id"))
            # Sanitize for Neo4j (convert dicts to JSON strings, filter non-primitives)
            sanitized = self._sanitize_for_neo4j(data)
            section_data.append(sanitized)

        tx.run(query, sections=section_data, document_id=document_id)
        return len(section_data)

    # Allowlist of valid entity labels to prevent Cypher injection
    # These map to the canonical plan's Entity subtypes
    # Phase 3.5: Expanded to include GLiNER v2 entity types for GraphRAG
    ENTITY_LABEL_ALLOWLIST = frozenset(
        {
            "Entity",  # Base/fallback label
            "Command",  # CLI commands (weka fs snapshot, etc.)
            "Configuration",  # Config parameters
            "Procedure",  # Multi-step procedures
            "Step",  # Individual steps within procedures
            "Error",  # Error codes/messages
            "Concept",  # Abstract concepts
            # Phase 3.5: GLiNER v2 entity types
            "Parameter",  # CLI parameters and flags
            "Component",  # System components (NFS, SMB, etc.)
            "Protocol",  # Network protocols
            "CloudProvider",  # AWS, Azure, GCP
            "StorageConcept",  # Storage concepts (filesystem, snapshot, etc.)
            "Version",  # Version numbers
            "ProcedureStep",  # Steps in procedures
            "CapacityMetric",  # Capacity/performance metrics
        }
    )

    # Map GLiNER label names to Neo4j-safe labels (PascalCase)
    GLINER_LABEL_MAP = {
        "COMMAND": "Command",
        "PARAMETER": "Parameter",
        "COMPONENT": "Component",
        "PROTOCOL": "Protocol",
        "CLOUD_PROVIDER": "CloudProvider",
        "STORAGE_CONCEPT": "StorageConcept",
        "VERSION": "Version",
        "PROCEDURE_STEP": "ProcedureStep",
        "ERROR": "Error",
        "CAPACITY_METRIC": "CapacityMetric",
    }

    def _neo4j_upsert_entities(self, tx, entities: Dict[str, Dict]) -> int:
        """Upsert entity nodes within a transaction.

        Phase 1.2 Enhancement: Entities now get proper subtype labels (Procedure, Step,
        Command, Configuration) in addition to the base Entity label. This enables
        type-specific queries like MATCH (p:Procedure)-[:CONTAINS_STEP]->(s:Step).

        Labels are validated against ENTITY_LABEL_ALLOWLIST to prevent Cypher injection.
        """
        if not entities:
            return 0

        # Group entities by their label for type-specific MERGE queries
        from collections import defaultdict

        by_label = defaultdict(list)

        for eid, edata in entities.items():
            data = dict(edata)  # Copy to avoid mutation
            data["id"] = eid

            # Extract and validate label (default to "Entity" if missing or invalid)
            label = data.get("label", "Entity")
            if label not in self.ENTITY_LABEL_ALLOWLIST:
                logger.warning(
                    "invalid_entity_label_rejected",
                    label=label,
                    entity_id=eid,
                    allowed=list(self.ENTITY_LABEL_ALLOWLIST),
                )
                label = "Entity"

            # Sanitize for Neo4j (convert dicts to JSON strings, filter non-primitives)
            sanitized = self._sanitize_for_neo4j(data)
            by_label[label].append(sanitized)

        # Run separate MERGE queries for each label type
        # This ensures proper Neo4j labels are applied, not just properties
        total = 0
        for label, entity_list in by_label.items():
            # Use f-string for label (safe due to allowlist validation above)
            query = f"""
            UNWIND $entities AS entity
            MERGE (e:Entity:{label} {{id: entity.id}})
            SET e += entity
            """
            tx.run(query, entities=entity_list)
            total += len(entity_list)

        return total

    def _neo4j_create_mentions(self, tx, mentions: List[Dict]):
        """Create MENTIONS relationships and Entity nodes within a transaction.

        Phase 3.5 Rewrite: Full GraphRAG MENTIONS support
        - Creates Entity nodes for GLiNER entities (those with source="gliner")
        - Creates (Chunk)-[:MENTIONS]->(Entity) relationships
        - Includes confidence score on relationship for query-time filtering
        - Routes Entity->Entity relationships to separate handler

        Direction convention (Neo4j best practice - single direction, no duplicates):
        - MENTIONS: Chunk -> Entity ("this chunk mentions this entity")
        - Queries needing Entity->Chunk traversal use direction-agnostic syntax
        - Example: (e:Entity)-[r:MENTIONS]-(c:Chunk) traverses either direction
        """
        if not mentions:
            return

        # Separate mentions by type based on key structure
        section_entity_mentions = []
        entity_entity_relationships = []
        gliner_entities_to_create = {}  # entity_id -> entity data

        for mention in mentions:
            if "from_id" in mention and "to_id" in mention:
                # Entity->Entity relationship (e.g., Procedure->Step via CONTAINS_STEP)
                entity_entity_relationships.append(mention)
            elif "section_id" in mention and "entity_id" in mention:
                # Section->Entity mention (MENTIONS)
                section_entity_mentions.append(mention)

                # Phase 3.5: Extract GLiNER entities that need Node creation
                if mention.get("source") == "gliner":
                    entity_id = mention["entity_id"]
                    if entity_id not in gliner_entities_to_create:
                        # Map GLiNER type to Neo4j label
                        entity_type = mention.get("type", "Entity")
                        neo4j_label = self.GLINER_LABEL_MAP.get(
                            entity_type.upper(), "Entity"
                        )
                        gliner_entities_to_create[entity_id] = {
                            "id": entity_id,
                            "name": mention.get("name", ""),
                            "entity_type": entity_type,
                            "label": neo4j_label,
                            "source": "gliner",
                        }
            else:
                # Log unexpected mention structure for debugging
                logger.warning(
                    "unroutable_mention_structure",
                    keys=list(mention.keys()),
                    mention_type=mention.get("relationship", "unknown"),
                )

        # Phase 3.5: Create Entity nodes for GLiNER entities BEFORE creating mentions
        if gliner_entities_to_create:
            gliner_count = self._neo4j_upsert_entities(tx, gliner_entities_to_create)
            logger.debug(
                "gliner_entities_created",
                count=gliner_count,
                entity_types=list(
                    set(e["entity_type"] for e in gliner_entities_to_create.values())
                ),
            )

        # Process Section->Entity mentions with correct direction (Chunk->Entity)
        if section_entity_mentions:
            # Phase 3.5: Fixed direction - (Chunk)-[:MENTIONS]->(Entity)
            # Also add confidence property for query-time filtering
            query = """
            UNWIND $mentions AS mention
            MATCH (c:Chunk {id: mention.section_id})
            MATCH (e:Entity {id: mention.entity_id})
            MERGE (c)-[r:MENTIONS]->(e)
            SET r.count = coalesce(r.count, 0) + 1,
                r.confidence = coalesce(mention.confidence, 0.5),
                r.source = coalesce(mention.source, 'structural')
            """
            tx.run(query, mentions=section_entity_mentions)

        # Process Entity->Entity relationships
        if entity_entity_relationships:
            self._neo4j_create_entity_relationships(tx, entity_entity_relationships)

    def _neo4j_create_references(self, tx, references: List[Dict]) -> int:
        """Create cross-document REFERENCES edges within a transaction.

        Phase 3: Implements (Chunk)-[:REFERENCES]->(Document) edge pattern.

        Fix #2 (Phase 3): Batched UNWIND implementation replaces O(N) per-reference
        queries with 4 batched operations:
        1. Batch resolve titles for hyperlinks
        2. Batch resolve fuzzy hints for non-hyperlinks
        3. Batch create ghost/pending/resolved references

        This method handles the consensus-approved hybrid edge pattern where:
        - Source is Chunk (preserves WHERE the reference occurred for RAG citations)
        - Target is Document (reliable resolution without brittle anchor matching)

        Reference dict structure (from extract/references.py):
        - source_chunk_id: The chunk where the reference was found
        - target_doc_id: Pre-resolved document ID (may be None)
        - target_hint: Original hint for Neo4j resolution (filename, title, phrase)
        - reference_type: Type of reference (hyperlink, see_also, related, refer_to)
        - reference_text: Display text of the reference
        - confidence: Extraction confidence score

        Returns:
            Number of REFERENCES edges created
        """
        # Delegate to streaming implementation (bug 11 hardening)
        return self._neo4j_create_references_streaming(tx, references)

    def _neo4j_create_references_streaming(self, tx, references: List[Dict]) -> int:
        """Streaming REFERENCES edge creation (rollback-safe, bug 11 fix)."""

        if not references:
            return 0

        from src.ingestion.extract.references import (
            normalize_filename_to_title,
            slugify_for_id,
        )

        refs_cfg = getattr(getattr(self, "config", None), "references", None)
        res_cfg = getattr(refs_cfg, "resolution", None) if refs_cfg else None
        batch_size = getattr(res_cfg, "batch_size", 100)
        min_hint_len = getattr(res_cfg, "min_hint_length", 3)
        penalty_cfg = getattr(
            getattr(res_cfg, "__dict__", res_cfg), "fuzzy_penalty", None
        )
        FUZZY_RESOLUTION_PENALTY = penalty_cfg if penalty_cfg is not None else 0.25

        created_count = 0
        unresolved_count = 0
        created_ghost_ids: List[str] = []

        resolved_buffer: List[Dict] = []
        ghost_buffer: List[Dict] = []
        pending_buffer: List[Dict] = []

        def flush_resolved(buf: List[Dict]):
            nonlocal created_count
            if not buf:
                return
            query = """
            UNWIND $refs AS ref
            MATCH (src:Chunk {id: ref.source_chunk_id})
            MATCH (d:Document {id: ref.target_doc_id})
            MERGE (src)-[r:REFERENCES]->(d)
            ON CREATE SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.final_confidence,
                r.target_hint = ref.target_hint,
                r.source_type = 'chunk',
                r.created_at = datetime({timezone: 'UTC'})
            ON MATCH SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.final_confidence,
                r.source_type = 'chunk',
                r.updated_at = datetime({timezone: 'UTC'})
            RETURN count(r) AS created
            """
            rec = tx.run(query, refs=buf).single()
            if rec:
                created_count += rec["created"]
            buf.clear()

        def flush_ghost(buf: List[Dict]):
            nonlocal created_count, created_ghost_ids
            if not buf:
                return
            query = """
            UNWIND $refs AS ref
            MERGE (ghost:GhostDocument {id: ref.ghost_id})
            ON CREATE SET
                ghost.title = ref.expected_title,
                ghost.stub = true,
                ghost.source_hint = ref.target_hint,
                ghost.created_at = datetime({timezone: 'UTC'})
            WITH ghost, ref
            MATCH (src:Chunk {id: ref.source_chunk_id})
            MERGE (src)-[r:REFERENCES]->(ghost)
            ON CREATE SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.target_hint = ref.target_hint,
                r.source_type = 'chunk',
                r.is_ghost_target = true,
                r.created_at = datetime({timezone: 'UTC'})
            ON MATCH SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.source_type = 'chunk',
                r.updated_at = datetime({timezone: 'UTC'})
            RETURN collect(DISTINCT ghost.id) AS ghost_ids, count(r) AS created
            """
            rec = tx.run(query, refs=buf).single()
            if rec:
                created_count += rec["created"]
                created_ghost_ids.extend(rec.get("ghost_ids", []))
            buf.clear()

        def flush_pending(buf: List[Dict]):
            nonlocal unresolved_count
            if not buf:
                return
            query = """
            UNWIND $refs AS ref
            MERGE (pending:PendingReference {hint: ref.target_hint})
            ON CREATE SET
                pending.created_at = datetime({timezone: 'UTC'}),
                pending.reference_count = 1
            ON MATCH SET
                pending.reference_count = coalesce(pending.reference_count, 0) + 1,
                pending.updated_at = datetime({timezone: 'UTC'})
            WITH pending, ref
            MATCH (src:Chunk {id: ref.source_chunk_id})
            MERGE (src)-[r:PENDING_REF]->(pending)
            ON CREATE SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.source_type = 'chunk',
                r.created_at = datetime({timezone: 'UTC'})
            ON MATCH SET
                r.type = ref.reference_type,
                r.reference_text = ref.reference_text,
                r.confidence = ref.confidence,
                r.source_type = 'chunk',
                r.updated_at = datetime({timezone: 'UTC'})
            RETURN count(r) AS created
            """
            rec = tx.run(query, refs=buf).single()
            if rec:
                unresolved_count += rec["created"]
            buf.clear()

        def stage_resolved(ref_obj: Dict):
            resolved_buffer.append(ref_obj)
            if len(resolved_buffer) >= batch_size:
                flush_resolved(resolved_buffer)

        def stage_ghost(ref_obj: Dict):
            ghost_buffer.append(ref_obj)
            if len(ghost_buffer) >= batch_size:
                flush_ghost(ghost_buffer)

        def stage_pending(ref_obj: Dict):
            pending_buffer.append(ref_obj)
            if len(pending_buffer) >= batch_size:
                flush_pending(pending_buffer)

        def process_title_batch(batch: List[Dict]):
            if not batch:
                return
            titles_to_resolve = list({r["possible_title"] for r in batch})
            title_to_doc = {}
            if titles_to_resolve:
                query = """
                UNWIND $titles AS title
                OPTIONAL MATCH (d:Document)
                WHERE toLower(d.title) = toLower(title)
                RETURN title, d.id AS doc_id
                """
                title_to_doc = {
                    rec["title"]: rec["doc_id"]
                    for rec in tx.run(query, titles=titles_to_resolve)
                    if rec["doc_id"]
                }

            for ref_data in batch:
                resolved_id = title_to_doc.get(ref_data["possible_title"])
                if resolved_id:
                    ref_data["target_doc_id"] = resolved_id
                    ref_data["final_confidence"] = ref_data["confidence"]
                    stage_resolved(ref_data)
                else:
                    expected_title = normalize_filename_to_title(
                        ref_data["target_hint"]
                    )
                    ref_data["ghost_id"] = f"ghost::{slugify_for_id(expected_title)}"
                    ref_data["expected_title"] = expected_title
                    stage_ghost(ref_data)
            batch.clear()

        def process_fuzzy_batch(batch: List[Dict]):
            if not batch:
                return

            # Phase 5.1 Fix: Use safe Lucene phrase queries to prevent ParseException
            # Build mapping: original_hint -> safe_phrase (or None if invalid)
            # This filters invalid hints BEFORE the query and escapes special chars
            from src.services.cross_doc_linking import prepare_lucene_phrase_query

            original_to_safe: Dict[str, str] = {}
            for r in batch:
                raw_hint = r.get("target_hint")
                if raw_hint:
                    safe_phrase = prepare_lucene_phrase_query(
                        raw_hint, min_length=min_hint_len
                    )
                    if safe_phrase:
                        original_to_safe[raw_hint] = safe_phrase

            hint_to_doc: Dict[str, str] = {}
            if original_to_safe:
                # Build reverse mapping: safe_phrase -> original_hint
                safe_to_original = {v: k for k, v in original_to_safe.items()}
                safe_phrases = list(original_to_safe.values())

                query = """
                UNWIND $hints AS hint
                CALL db.index.fulltext.queryNodes(
                    'document_title_ft', hint
                ) YIELD node AS d, score
                WHERE score > 0.5
                WITH hint, d, score
                ORDER BY score DESC, size(d.title) ASC
                WITH hint, collect(d.id)[0] AS doc_id
                RETURN hint, doc_id
                """
                # Map safe_phrase -> doc_id, then convert to original_hint -> doc_id
                for rec in tx.run(query, hints=safe_phrases):
                    if rec["doc_id"]:
                        safe_phrase = rec["hint"]
                        original_hint = safe_to_original.get(safe_phrase)
                        if original_hint:
                            hint_to_doc[original_hint] = rec["doc_id"]

            for ref_data in batch:
                target_hint = ref_data.get("target_hint", "")
                # Check if hint was valid (present in our mapping)
                if target_hint not in original_to_safe:
                    stage_pending(ref_data)
                    continue
                resolved_id = hint_to_doc.get(target_hint)
                if resolved_id:
                    final_conf = ref_data["confidence"]
                    final_conf = max(0.1, final_conf - FUZZY_RESOLUTION_PENALTY)
                    ref_data["target_doc_id"] = resolved_id
                    ref_data["final_confidence"] = final_conf
                    ref_data["is_fuzzy_match"] = True
                    stage_resolved(ref_data)
                else:
                    stage_pending(ref_data)
            batch.clear()

        title_batch: List[Dict] = []
        fuzzy_batch: List[Dict] = []

        try:
            for ref in references:
                source_chunk_id = ref.get("source_chunk_id")
                if not source_chunk_id:
                    logger.warning(
                        "reference_missing_source_chunk",
                        target_hint=ref.get("target_hint", ""),
                        reference_type=ref.get("reference_type", "unknown"),
                    )
                    continue

                ref_data = {
                    "source_chunk_id": source_chunk_id,
                    "target_doc_id": ref.get("target_doc_id"),
                    "target_hint": ref.get("target_hint", ""),
                    "reference_type": ref.get("reference_type", "unknown"),
                    "reference_text": ref.get("reference_text", ""),
                    "confidence": ref.get("confidence", 0.5),
                    "is_fuzzy_match": False,
                }

                if ref_data["target_doc_id"]:
                    ref_data["final_confidence"] = ref_data["confidence"]
                    stage_resolved(ref_data)
                    continue

                if ref_data["reference_type"] == "hyperlink" and ref_data[
                    "target_hint"
                ].endswith(".md"):
                    ref_data["possible_title"] = normalize_filename_to_title(
                        ref_data["target_hint"]
                    )
                    title_batch.append(ref_data)
                    if len(title_batch) >= batch_size * 2:
                        process_title_batch(title_batch)
                    continue

                fuzzy_batch.append(ref_data)
                if len(fuzzy_batch) >= batch_size * 2:
                    process_fuzzy_batch(fuzzy_batch)

            process_title_batch(title_batch)
            process_fuzzy_batch(fuzzy_batch)
            flush_resolved(resolved_buffer)
            flush_ghost(ghost_buffer)
            flush_pending(pending_buffer)

        except Exception as exc:
            if created_ghost_ids:
                tx.run(
                    """
                    UNWIND $ids AS gid
                    MATCH (g:GhostDocument {id: gid})
                    DETACH DELETE g
                    """,
                    ids=created_ghost_ids,
                )
            logger.warning(
                "references_rollback_ghosts",
                ghost_ids=created_ghost_ids,
                error=str(exc),
            )
            raise

        if created_count > 0 or unresolved_count > 0:

            def _safe_log_value(value: str, max_length: int = 200) -> str:
                if not value:
                    return ""
                sanitized = re.sub(r"[\\x00-\\x1f\\x7f-\\x9f]", "", value)
                return (
                    (sanitized[:max_length] + "...")
                    if len(sanitized) > max_length
                    else sanitized
                )

            logger.info(
                "references_created",
                created=created_count,
                unresolved=unresolved_count,
                total_attempted=len(references),
                sample_hint=_safe_log_value(
                    (resolved_buffer or ghost_buffer or pending_buffer or [{}])[0].get(
                        "target_hint", ""
                    )
                ),
            )

        return created_count

    def _neo4j_create_entity_relationships(self, tx, relationships: List[Dict]):
        """Create Entity->Entity relationships within a transaction.

        Phase 1.2: Handles relationships like CONTAINS_STEP (Procedure->Step)
        that were previously being dropped by _neo4j_create_mentions.

        Relationship dict structure:
        - from_id: Source entity ID
        - from_label: Source entity label (e.g., "Procedure")
        - to_id: Target entity ID
        - to_label: Target entity label (e.g., "Step")
        - relationship: Relationship type (e.g., "CONTAINS_STEP")
        - order: Optional ordering field for sequential relationships
        - confidence: Extraction confidence score
        - source_section_id: Section where relationship was extracted
        """
        if not relationships:
            return

        # Group relationships by type for efficient batch processing
        by_type: Dict[str, List[Dict]] = {}
        for rel in relationships:
            rel_type = rel.get("relationship")
            if not rel_type:
                logger.warning(
                    "entity_relationship_missing_type",
                    from_id=rel.get("from_id"),
                    to_id=rel.get("to_id"),
                    reason="relationship_type_required",
                )
                continue  # Skip relationships without explicit type
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        # Process each relationship type with a dedicated query
        rejected_count = 0
        for rel_type, rels in by_type.items():
            # Defense-in-depth: Validate relationship type against allowlist
            # This prevents Cypher injection via malformed/malicious extractor output
            if rel_type not in ALLOWED_ENTITY_RELATIONSHIP_TYPES:
                logger.warning(
                    "entity_relationship_type_rejected",
                    relationship_type=rel_type,
                    count=len(rels),
                    reason="not_in_allowlist",
                    allowed_types=list(ALLOWED_ENTITY_RELATIONSHIP_TYPES),
                )
                rejected_count += len(rels)
                continue

            # Relationship types must be known at query compile time in Cypher,
            # so we use separate queries per type (validated via allowlist)
            #
            # Issue #7 Fix: OPTIONAL MATCH + counts to detect missing entities
            # Issue #18 Fix: datetime({timezone: 'UTC'}) for explicit UTC
            #
            # Multi-model fix: Aggregate missing counts BEFORE filtering
            # collect/unwind pattern preserves missing counts through WHERE
            query = f"""
            UNWIND $rels AS rel
            OPTIONAL MATCH (from:Entity {{id: rel.from_id}})
            OPTIONAL MATCH (to:Entity {{id: rel.to_id}})
            WITH rel, from, to,
                 CASE WHEN from IS NULL THEN 1 ELSE 0 END AS mf,
                 CASE WHEN to IS NULL THEN 1 ELSE 0 END AS mt
            WITH collect({{r: rel, f: from, t: to}}) AS pairs,
                 sum(mf) AS total_missing_from,
                 sum(mt) AS total_missing_to
            UNWIND pairs AS p
            WITH p.r AS rel, p.f AS from, p.t AS to,
                 total_missing_from, total_missing_to
            WHERE from IS NOT NULL AND to IS NOT NULL
            MERGE (from)-[r:{rel_type}]->(to)
            SET r.order = rel.order,
                r.confidence = rel.confidence,
                r.source_section_id = rel.source_section_id,
                r.created_at = datetime({{timezone: 'UTC'}})
            // N1 Fix: COALESCE prevents NULL when all rows filtered by WHERE clause
            RETURN count(r) AS created_count,
                   COALESCE(max(total_missing_from), 0) AS missing_from_count,
                   COALESCE(max(total_missing_to), 0) AS missing_to_count
            """
            result = tx.run(query, rels=rels)
            record = result.single()

            # N1 Fix: Defensive None handling (belt-and-suspenders with Cypher COALESCE)
            created_count = record["created_count"] if record else 0
            missing_from = (record.get("missing_from_count") or 0) if record else 0
            missing_to = (record.get("missing_to_count") or 0) if record else 0

            # Issue #7: Log when entities are not found
            if missing_from > 0 or missing_to > 0:
                logger.warning(
                    "entity_relationships_missing_entities",
                    relationship_type=rel_type,
                    attempted=len(rels),
                    created=created_count,
                    missing_from_entities=missing_from,
                    missing_to_entities=missing_to,
                )
                # Issue #8: Prometheus metric for missing entities
                entity_relationships_missing_total.labels(
                    relationship_type=rel_type
                ).inc(missing_from + missing_to)

            # Issue #8: Prometheus metrics for created relationships
            if created_count > 0:
                entity_relationships_total.labels(
                    relationship_type=rel_type, status="created"
                ).inc(created_count)

            logger.debug(
                "entity_relationships_created",
                relationship_type=rel_type,
                attempted=len(rels),
                created=created_count,
            )

        # Issue #8: Prometheus metric for rejected relationships
        if rejected_count > 0:
            entity_relationships_total.labels(
                relationship_type="rejected", status="rejected"
            ).inc(rejected_count)
            logger.info(
                "entity_relationships_rejected_total",
                rejected_count=rejected_count,
            )

    def _neo4j_upsert_embedding_metadata(
        self,
        tx,
        sections: List[Dict],
        embeddings: Dict,
        builder,
    ) -> int:
        """
        Store embedding metadata on Chunk nodes in Neo4j.

        This ensures cross-store consistency between Neo4j and Qdrant by storing
        the same embedding metadata in both stores. Matches build_graph.py behavior.

        Args:
            tx: Neo4j transaction
            sections: List of section dicts
            embeddings: Pre-computed embeddings dict
            builder: GraphBuilder instance with embedding settings

        Returns:
            Number of chunks updated with embedding metadata
        """
        if not embeddings.get("sections"):
            return 0

        if not hasattr(builder, "embedding_settings") or not builder.embedding_settings:
            logger.warning(
                "neo4j_embedding_metadata_skipped",
                reason="No embedding_settings in builder",
            )
            return 0

        # Build batch update data
        updates = []
        for section in sections:
            section_id = section.get("id")
            if not section_id:
                continue

            section_embeddings = embeddings.get("sections", {}).get(section_id)
            if not section_embeddings:
                continue

            content_embedding = section_embeddings.get("content", [])

            updates.append(
                {
                    "id": section_id,
                    "embedding_version": builder.embedding_settings.version,
                    "embedding_provider": (
                        getattr(builder.embedder, "provider_name", None)
                        if hasattr(builder, "embedder")
                        else None
                    ),
                    "embedding_dimensions": (
                        len(content_embedding)
                        if content_embedding
                        else builder.embedding_settings.dims
                    ),
                    "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
                    "embedding_task": builder.embedding_settings.task,
                    # Note: We don't store vector_embedding in Neo4j for atomic path
                    # to reduce write amplification. Qdrant is the primary vector store.
                }
            )

        if not updates:
            return 0

        # Batch update embedding metadata on Chunk nodes
        query = """
        UNWIND $updates AS update
        MATCH (c:Chunk {id: update.id})
        SET c.embedding_version = update.embedding_version,
            c.embedding_provider = update.embedding_provider,
            c.embedding_dimensions = update.embedding_dimensions,
            c.embedding_timestamp = update.embedding_timestamp,
            c.embedding_task = update.embedding_task
        """

        tx.run(query, updates=updates)

        logger.debug(
            "neo4j_embedding_metadata_upserted",
            chunks_updated=len(updates),
            provider=updates[0]["embedding_provider"] if updates else None,
        )

        return len(updates)

    # =========================================================================
    # Helper methods for canonical payload fields (matching build_graph.py)
    # =========================================================================

    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA256 hash of text content for drift detection."""
        value = text or ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _compute_shingle_hash(self, text: str, n: int = 8) -> str:
        """Compute shingle hash for deduplication."""
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
            return ""
        combined = "|".join(sorted(set(shingles)))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _extract_semantic_metadata(self, section: Dict) -> Dict[str, Any]:
        """Extract semantic metadata from section (NER, topics, etc.)."""
        metadata = section.get("semantic_metadata")
        if metadata:
            return metadata
        return {"entities": [], "topics": []}

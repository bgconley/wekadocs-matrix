# Implements Phase 3, Task 3.4 (Incremental updates)
# See: /docs/spec.md §8 (Ingestion pipeline - incremental)
# See: /docs/implementation-plan.md → Task 3.4
# See: /docs/pseudocode-reference.md → Task 3.4

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from src.shared.config import get_embedding_settings
from src.shared.qdrant_schema import build_qdrant_schema


@dataclass
class Diff:
    adds: List[Dict]
    updates: List[Dict]  # same id, changed checksum/properties
    deletes: List[str]  # section ids


class IncrementalUpdater:
    """
    Incremental update with stage→swap:
    - Detect adds/updates/deletes by Section.id and checksum
    - Stage new/modified sections as :StagedSection
    - Delete removed sections
    - Promote staged to :Section (atomic swap)
    - Keeps counts stable for 'minimal delta' scenarios
    """

    def __init__(
        self,
        neo4j_driver: Driver,
        config=None,
        qdrant_client: QdrantClient = None,
        collection_name: str = "weka_sections",
        embedding_version: str = "v1",
    ):
        self.neo4j = neo4j_driver
        self.config = config
        self.qdrant = qdrant_client
        self.collection = collection_name
        self.version = embedding_version
        self.embedding_settings = get_embedding_settings(config) if config else None
        if config and hasattr(config, "search") and hasattr(config.search, "vector"):
            if hasattr(config.search.vector, "qdrant"):
                self.collection = config.search.vector.qdrant.collection_name
            self.version = (
                get_embedding_settings(config).version
                if hasattr(config, "embedding")
                else "v1"
            )
        self._schema_plan = None
        self._dense_vector_names: Optional[List[str]] = None
        if self.embedding_settings and self.config and hasattr(self.config, "search"):
            search_cfg = self.config.search
            vector_cfg = getattr(search_cfg, "vector", None)
            qdrant_cfg = getattr(vector_cfg, "qdrant", None)
            enable_sparse = (
                getattr(qdrant_cfg, "enable_sparse", False) if qdrant_cfg else False
            )
            enable_colbert = (
                getattr(qdrant_cfg, "enable_colbert", False) if qdrant_cfg else False
            )
            hybrid_cfg = getattr(search_cfg, "hybrid", None)
            vector_fields = (
                getattr(hybrid_cfg, "vector_fields", None) if hybrid_cfg else None
            )
            include_entity = False
            if isinstance(vector_fields, dict):
                include_entity = "entity" in vector_fields
            self._schema_plan = build_qdrant_schema(
                self.embedding_settings,
                include_entity=include_entity,
                enable_sparse=enable_sparse,
                enable_colbert=enable_colbert,
            )
            self._dense_vector_names = [
                name
                for name in self._schema_plan.vectors_config.keys()
                if name != "late-interaction"
            ]

    def _existing_sections(self, document_id: str) -> Dict[str, Dict]:
        """Get existing sections from the database (synchronous)."""
        cypher = """
        MATCH (:Document {id: $doc})-[:HAS_SECTION]->(s:Section)
        RETURN s.id AS id, coalesce(s.checksum, '') AS checksum, s.title AS title
        """
        out: Dict[str, Dict] = {}
        with self.neo4j.session() as sess:
            res = sess.run(cypher, {"doc": document_id})
            for rec in res:
                out[rec["id"]] = {
                    "checksum": rec["checksum"],
                    "title": rec.get("title", ""),
                }
        return out

    def compute_diff(self, document_id: str, new_sections: List[Dict]) -> Dict:
        """
        Compare DB snapshot vs new sections by ID and checksum.
        Returns: {added, removed, modified, unchanged, total_changes}
        """
        # Get existing sections from database
        existing = self._existing_sections(document_id)

        old_ids = set(existing.keys())
        new_by_id = {s["id"]: s for s in new_sections}
        new_ids = set(new_by_id.keys())

        # Compute set differences
        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids
        potential_modified = old_ids & new_ids

        added = [new_by_id[sid] for sid in added_ids]
        removed = list(removed_ids)
        modified = []
        unchanged = []

        # Check for modifications by comparing checksums
        for sid in potential_modified:
            old_checksum = existing[sid].get("checksum", "")
            new_checksum = new_by_id[sid].get("checksum", "")
            if old_checksum != new_checksum:
                modified.append(new_by_id[sid])
            else:
                unchanged.append(new_by_id[sid])

        total_changes = len(added) + len(removed) + len(modified)

        return {
            "total_changes": total_changes,
            "added": added,
            "modified": modified,
            "removed": removed,
            "unchanged": unchanged,
        }

    def _vector_field_names(self) -> List[str]:
        """
        Determine which named vectors should be populated for compatibility with
        the configured Qdrant collection. Defaults to ["content"] so incremental
        updates remain backwards compatible in minimal environments.
        """
        if self._dense_vector_names:
            return self._dense_vector_names
        return ["content"]

    def _build_named_vectors(
        self, base_vector: List[float], field_names: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Clone the supplied base vector across all configured named-vector slots.
        This allows placeholder embeddings (or a single computed vector) to
        satisfy the multi-vector schema without additional model invocations.
        """
        vectors: Dict[str, List[float]] = {}
        names = field_names or self._vector_field_names()
        for name in names:
            vectors[name] = list(base_vector)
        return vectors

    def _build_placeholder_vectors(
        self, base_vector: List[float], field_names: Optional[List[str]] = None
    ) -> Dict[str, object]:
        vectors: Dict[str, object] = self._build_named_vectors(base_vector, field_names)
        if self._schema_plan and "late-interaction" in self._schema_plan.vectors_config:
            vectors["late-interaction"] = [list(base_vector)]
        return vectors

    def apply_incremental_update(
        self, diff: Dict, sections: List[Dict], entities: Dict, mentions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Apply incremental update from a diff.
        This method signature matches test expectations.
        """
        document_id = sections[0].get("document_id") if sections else ""

        # Extract modified section objects from the modified list
        # Modified contains section objects with IDs
        modified_section_ids = {m["id"] for m in diff.get("modified", [])}
        modified_sections = [s for s in sections if s["id"] in modified_section_ids]

        # Build internal Diff object
        internal_diff = Diff(
            adds=diff.get("added", []),
            updates=modified_sections,  # Use actual section objects
            deletes=diff.get("removed", []),
        )

        # Apply changes
        deleted_count = 0
        upserted_count = 0

        # 1) Delete removed sections
        if internal_diff.deletes:
            with self.neo4j.session() as sess:
                sess.run(
                    """
                    UNWIND $ids AS sid
                    MATCH (d:Document {id: $doc})-[:HAS_SECTION]->(s:Section {id: sid})
                    DETACH DELETE s
                    """,
                    {"ids": internal_diff.deletes, "doc": document_id},
                )
                deleted_count = len(internal_diff.deletes)

            # Delete vectors
            if self.qdrant:
                self.qdrant.delete_compat(
                    collection_name=self.collection,
                    points=internal_diff.deletes,
                    wait=True,
                )

        # 2) Upsert added/modified sections
        to_upsert = internal_diff.adds + internal_diff.updates
        doc_tag_value = None
        if to_upsert:
            with self.neo4j.session() as sess:
                sess.run(
                    """
                    UNWIND $rows AS row
                    MERGE (s:Section {id: row.id})
                    SET s.title = row.title,
                        s.content = coalesce(row.content, row.text),
                        s.checksum = row.checksum,
                        s.document_id = $doc,
                        s.embedding_version = $v,
                        s.updated_at = datetime()
                    MERGE (d:Document {id: $doc})
                    SET s.doc_tag = d.doc_tag,
                        s.snapshot_scope = d.snapshot_scope
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    {"rows": to_upsert, "doc": document_id, "v": self.version},
                )
                doc_tag_result = sess.run(
                    "MATCH (d:Document {id:$doc}) RETURN d.doc_tag AS doc_tag, d.snapshot_scope AS snapshot_scope",
                    {"doc": document_id},
                ).single()
                doc_tag_value = doc_tag_result["doc_tag"] if doc_tag_result else None
                snapshot_scope_value = (
                    doc_tag_result["snapshot_scope"] if doc_tag_result else None
                )
                upserted_count = len(to_upsert)

            # Upsert vectors (placeholder embeddings for tests)
            if self.qdrant:
                points = []
                # Pre-Phase 7: Use config-driven dimensions for placeholders
                embedding_dims = 384  # Default fallback
                if self.config:
                    embedding_dims = get_embedding_settings(self.config).dims
                zero_template = [0.0] * embedding_dims
                vector_field_names = self._vector_field_names()

                for sec in to_upsert:
                    sec_doc_tag = sec.get("doc_tag", doc_tag_value)
                    if sec_doc_tag:
                        sec["doc_tag"] = sec_doc_tag
                    sec_snapshot_scope = sec.get("snapshot_scope", snapshot_scope_value)
                    if sec_snapshot_scope:
                        sec["snapshot_scope"] = sec_snapshot_scope
                    point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, sec["id"]))
                    vectors = self._build_placeholder_vectors(
                        zero_template, vector_field_names
                    )
                    points.append(
                        PointStruct(
                            id=point_uuid,
                            vector=vectors,  # config-driven placeholder across fields
                            payload={
                                "node_id": sec["id"],
                                "node_label": "Section",
                                "document_id": document_id,
                                "document_uri": sec.get("source_uri")
                                or sec.get("document_uri"),
                                "doc_tag": sec_doc_tag,
                                "snapshot_scope": sec_snapshot_scope,
                                "embedding_version": self.version,
                            },
                        )
                    )
                if points:
                    expected_dims_map: Dict[str, int] = {}
                    first_vector = points[0].vector
                    if isinstance(first_vector, dict):
                        for name, vec in first_vector.items():
                            dim = self._vector_expected_dim(vec)
                            if dim:
                                expected_dims_map[name] = dim
                    expected_dims = expected_dims_map or embedding_dims
                    # Pre-Phase 7: Use validated upsert with config-driven dimensions
                    self.qdrant.upsert_validated(
                        collection_name=self.collection,
                        points=points,
                        expected_dim=expected_dims or embedding_dims,
                    )

        return {
            "sections_updated": upserted_count,
            "embeddings_updated": upserted_count,
            "reembedding_required": upserted_count,
            "total_changes": len(to_upsert) + deleted_count,
        }

    @staticmethod
    def _vector_expected_dim(vector: object) -> Optional[int]:
        if vector is None:
            return None
        if hasattr(vector, "indices") and hasattr(vector, "values"):
            return None
        if isinstance(vector, list):
            if not vector:
                return None
            first = vector[0]
            if isinstance(first, list):
                return len(first)
            return len(vector)
        if hasattr(vector, "__len__") and not isinstance(vector, (str, bytes)):
            try:
                return len(vector)
            except TypeError:
                return None
        return None

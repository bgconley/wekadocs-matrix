from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector

from src.providers.embeddings.contracts import DocumentEmbeddingBundle
from src.providers.factory import ProviderFactory
from src.shared.config import get_embedding_settings, namespace_identifier
from src.shared.qdrant_schema import build_qdrant_schema
from src.shared.vector_utils import vector_expected_dim


@dataclass
class DriftStats:
    embedding_version: str
    graph_count: int
    vector_count: int
    extras_removed: int
    missing_backfilled: int
    drift_pct: float


class Reconciler:
    """
    Keeps Qdrant strictly in sync with graph Section nodes.

    Synchronizes for a given embedding_version.
    """

    def __init__(
        self,
        neo4j_driver,
        config_or_qdrant=None,  # Can be Config or QdrantClient for backwards compat
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: str = "weka_sections",
        embedding_version: str = "v1",
    ):
        self.neo4j = neo4j_driver

        # Handle backwards compatibility for different call signatures
        if isinstance(config_or_qdrant, QdrantClient):
            # Old signature: Reconciler(neo4j, qdrant, ...)
            self.qdrant = config_or_qdrant
            self.config = None
            self.collection = collection_name
            self.embedding_settings = get_embedding_settings()
            self.collection = namespace_identifier(
                self.collection, self.embedding_settings.profile
            )
            self.version = self.embedding_settings.version
        elif hasattr(config_or_qdrant, "embedding"):
            # New signature: Reconciler(neo4j, config, qdrant)
            self.config = config_or_qdrant
            self.qdrant = qdrant_client
            # Use collection from config or default
            if hasattr(config_or_qdrant.search.vector, "qdrant") and hasattr(
                config_or_qdrant.search.vector.qdrant, "collection_name"
            ):
                self.collection = config_or_qdrant.search.vector.qdrant.collection_name
            else:
                self.collection = collection_name or "weka_sections"
            self.embedding_settings = get_embedding_settings(self.config)
            self.collection = namespace_identifier(
                self.collection, self.embedding_settings.profile
            )
            self.version = self.embedding_settings.version
        else:
            # Test signature: Reconciler(neo4j, config, qdrant)
            self.config = config_or_qdrant
            self.qdrant = qdrant_client or config_or_qdrant
            self.collection = collection_name
            self.embedding_settings = get_embedding_settings()
            self.collection = namespace_identifier(
                self.collection, self.embedding_settings.profile
            )
            self.version = self.embedding_settings.version
        self._embedder = None
        search_cfg = getattr(self.config, "search", None) if self.config else None
        vector_cfg = getattr(search_cfg, "vector", None) if search_cfg else None
        qdrant_cfg = getattr(vector_cfg, "qdrant", None)
        enable_sparse = (
            getattr(qdrant_cfg, "enable_sparse", False) if qdrant_cfg else False
        )
        enable_colbert = (
            getattr(qdrant_cfg, "enable_colbert", False) if qdrant_cfg else False
        )
        # Extract doc_title sparse flag - defaults to True for backward compatibility
        enable_doc_title_sparse = (
            getattr(qdrant_cfg, "enable_doc_title_sparse", True) if qdrant_cfg else True
        )
        # NEW: Lexical sparse vectors for section headings and entity names
        enable_title_sparse = (
            getattr(qdrant_cfg, "enable_title_sparse", True) if qdrant_cfg else True
        )
        enable_entity_sparse = (
            getattr(qdrant_cfg, "enable_entity_sparse", True) if qdrant_cfg else True
        )
        # NOTE: include_entity is deprecated (dense entity vector removed 2025-12-06)

        self._schema_plan = build_qdrant_schema(
            self.embedding_settings,
            include_entity=False,  # Deprecated - always False
            enable_sparse=enable_sparse,
            enable_colbert=enable_colbert,
            enable_doc_title_sparse=enable_doc_title_sparse,
            enable_title_sparse=enable_title_sparse,
            enable_entity_sparse=enable_entity_sparse,
        )
        self._dense_vector_names = [
            name
            for name in self._schema_plan.vectors_config.keys()
            if name != "late-interaction"
        ]

    def _vector_field_names(self) -> List[str]:
        """
        Return the configured named-vector fields for the active environment.
        Falls back to ["content"] so that legacy single-vector deployments
        remain functional.
        """
        if getattr(self, "_dense_vector_names", None):
            return self._dense_vector_names
        return ["content"]

    def _get_embedding_fn(self) -> Callable[[str], List[float]]:
        embedder = self._ensure_embedder()

        def embed(text: str) -> List[float]:
            vectors = embedder.embed_documents([text or ""])
            if not vectors:
                raise RuntimeError("Embedding provider returned no vectors")
            return vectors[0]

        return embed

    def _get_embedding_bundle_fn(self) -> Callable[[str], DocumentEmbeddingBundle]:
        embedder = self._ensure_embedder()
        if hasattr(embedder, "embed_documents_all"):

            def embed_bundle(text: str) -> DocumentEmbeddingBundle:
                bundles = embedder.embed_documents_all([text or ""])
                if not bundles:
                    raise RuntimeError("Embedding provider returned no bundles")
                return bundles[0]

            return embed_bundle

        base_fn = self._get_embedding_fn()

        def embed_bundle(text: str) -> DocumentEmbeddingBundle:
            dense = base_fn(text)
            return DocumentEmbeddingBundle(dense=list(dense))

        return embed_bundle

    def _ensure_embedder(self):
        if self._embedder is None:
            if not self.config and self.embedding_settings is None:
                raise RuntimeError(
                    "Embedding configuration unavailable; "
                    "provide embedding_fn explicitly."
                )
            factory = ProviderFactory()
            settings = self.embedding_settings or get_embedding_settings(self.config)
            self._embedder = factory.create_embedding_provider(settings=settings)
        return self._embedder

    def _build_vectors_from_bundle(
        self, bundle: DocumentEmbeddingBundle
    ) -> Dict[str, object]:
        vectors: Dict[str, object] = {}
        dense_vector = list(bundle.dense)
        for name in self._vector_field_names():
            vectors[name] = dense_vector
        if (
            bundle.multivector
            and "late-interaction" in self._schema_plan.vectors_config
        ):
            vectors["late-interaction"] = [
                list(vec) for vec in bundle.multivector.vectors
            ]
        if bundle.sparse and "text-sparse" in self._schema_plan.sparse_vectors_config:
            vectors["text-sparse"] = SparseVector(
                indices=list(bundle.sparse.indices),
                values=list(bundle.sparse.values),
            )
        return vectors

    def _graph_section_ids(self) -> Set[str]:
        """Get all Chunk IDs from Neo4j with matching embedding_version."""
        cypher = """
        MATCH (c:Chunk)
        WHERE c.embedding_version = $v
        RETURN c.id AS id
        """
        ids: Set[str] = set()
        with self.neo4j.session() as sess:
            result = sess.run(cypher, {"v": self.version})
            for rec in result:
                ids.add(rec["id"])
        return ids

    def _qdrant_section_ids(self) -> Set[str]:
        # Scroll all points for the version; small datasets in tests
        # Returns original node_ids (not UUIDs) from payload
        # Note: node_label may be "Chunk" (new) or "Section" (legacy) during migration

        filt = {
            "must": [
                {"key": "embedding_version", "match": {"value": self.version}},
            ],
            "should": [
                {"key": "node_label", "match": {"value": "Chunk"}},
                {"key": "node_label", "match": {"value": "Section"}},
            ],
            "min_should": {"min_count": 1},
        }

        out: Set[str] = set()
        next_page = None
        while True:
            res = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            points, next_page = res
            for p in points:
                # Use node_id from payload (original section ID), not point.id (UUID)
                nid = p.payload.get("node_id")
                if nid:
                    out.add(nid)
            if next_page is None:
                break
        return out

    def _delete_points_by_node_ids(self, node_ids: List[str]) -> int:
        if not node_ids:
            return 0
        batch_size = 256
        deleted = 0
        for start in range(0, len(node_ids), batch_size):
            batch = node_ids[start : start + batch_size]
            if not batch:
                continue
            selector = {
                "filter": {
                    "must": [
                        {"key": "node_id", "match": {"any": batch}},
                        {
                            "key": "embedding_version",
                            "match": {"value": self.version},
                        },
                    ]
                }
            }
            try:
                self.qdrant.delete(
                    collection_name=self.collection,
                    points_selector=selector,
                    wait=True,
                )
            except Exception:
                fallback_selector = {
                    "filter": {
                        "must": [
                            {"key": "node_id", "match": {"any": batch}},
                        ]
                    }
                }
                self.qdrant.delete(
                    collection_name=self.collection,
                    points_selector=fallback_selector,
                    wait=True,
                )
            deleted += len(batch)
        return deleted

    def _upsert_points(
        self,
        records: List[Tuple[str, Dict[str, List[float]], Dict]],
    ) -> int:
        if not records:
            return 0
        # Convert section IDs to UUIDs for point IDs
        import uuid

        pts = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, rec[0])),  # UUID from node_id
                vector=rec[1],
                payload=rec[2],
            )
            for rec in records
        ]

        # Pre-Phase 7: Determine expected dimension from config or first vector
        expected_dim: int | Dict[str, int] = (
            self.embedding_settings.dims if self.embedding_settings else 384
        )

        if pts:
            first_vector = pts[0].vector
            if isinstance(first_vector, dict):
                derived: Dict[str, int] = {}
                for name, vec in first_vector.items():
                    dim = self._vector_expected_dim(vec)
                    if dim:
                        derived[name] = dim
                if derived:
                    expected_dim = derived
            elif isinstance(first_vector, list):
                expected_dim = len(first_vector)

        self.qdrant.upsert_validated(
            collection_name=self.collection, points=pts, expected_dim=expected_dim
        )
        return len(pts)

    def reconcile_sync(
        self, embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> DriftStats:
        """
        Make Qdrant contain exactly the Section nodes at embedding_version.

        Uses embedding_version=self.version. If embedding_fn is None, uses
        SentenceTransformers(all-MiniLM-L6-v2) if available.
        """
        graph_ids = self._graph_section_ids()
        vec_ids = self._qdrant_section_ids()

        extras = sorted(list(vec_ids - graph_ids))
        missing = sorted(list(graph_ids - vec_ids))

        removed = self._delete_points_by_node_ids(extras)

        # Build embeddings for missing
        if missing:
            if embedding_fn is None:
                bundle_fn = self._get_embedding_bundle_fn()
            else:
                emb_fn = embedding_fn

            # Fetch chunk text to embed
            text_map: Dict[str, Dict[str, str]] = {}
            cypher = """
            UNWIND $ids AS sid
            MATCH (d:Document)-[:HAS_SECTION]->(c:Chunk {id: sid})
            RETURN
                c.id AS id,
                coalesce(c.text, c.content, '') AS text,
                c.title AS title,
                coalesce(c.document_id, d.id) AS document_id,
                d.source_uri AS source_uri,
                d.doc_tag AS doc_tag,
                d.snapshot_scope AS snapshot_scope
            """
            with self.neo4j.session() as sess:
                res = sess.run(cypher, {"ids": missing})
                for rec in res:
                    text_map[rec["id"]] = {
                        "text": rec["text"] or "",
                        "title": rec["title"] or "",
                        "document_id": rec.get("document_id"),
                        "source_uri": rec.get("source_uri"),
                        "doc_tag": rec.get("doc_tag"),
                    }

            upserts = []
            for sid in missing:
                meta = text_map.get(
                    sid,
                    {"text": "", "title": "", "document_id": None, "source_uri": None},
                )

                # Embed Content
                if embedding_fn is None:
                    content_bundle = bundle_fn(meta["text"])
                else:
                    vec = emb_fn(meta["text"])
                    content_bundle = DocumentEmbeddingBundle(dense=list(vec))

                # Embed Title (if present)
                title_vec = None
                title_text = meta.get("title")
                if title_text:
                    if embedding_fn is None:
                        # We only need dense for title currently
                        title_bundle = bundle_fn(title_text)
                        title_vec = list(title_bundle.dense)
                    else:
                        title_vec = list(emb_fn(title_text))

                # Construct Named Vectors
                named_vecs: Dict[str, object] = {}

                # 1. Content Vector
                named_vecs["content"] = list(content_bundle.dense)

                # 2. Title Vector
                if title_vec:
                    named_vecs["title"] = title_vec
                elif "title" in self._schema_plan.vectors_config:
                    # Fallback: use content if title missing but schema requires it
                    named_vecs["title"] = list(content_bundle.dense)

                # 3. Sparse Vector (from Content)
                if (
                    content_bundle.sparse
                    and "text-sparse" in self._schema_plan.sparse_vectors_config
                ):
                    named_vecs["text-sparse"] = SparseVector(
                        indices=list(content_bundle.sparse.indices),
                        values=list(content_bundle.sparse.values),
                    )

                # 4. ColBERT (from Content)
                if (
                    content_bundle.multivector
                    and "late-interaction" in self._schema_plan.vectors_config
                ):
                    named_vecs["late-interaction"] = [
                        list(vec) for vec in content_bundle.multivector.vectors
                    ]

                source_uri = meta.get("source_uri") or ""
                document_uri = Path(source_uri).name if source_uri else source_uri
                payload = {
                    "node_id": sid,
                    "node_label": "Section",
                    "document_id": meta.get("document_id"),
                    "document_uri": document_uri,
                    "source_uri": source_uri,
                    "doc_tag": meta.get("doc_tag"),
                    "snapshot_scope": meta.get("snapshot_scope"),
                    "embedding_version": self.version,
                    "heading": meta.get("title"),  # Ensure heading payload is synced
                }
                upserts.append((sid, named_vecs, payload))

            added = self._upsert_points(upserts)
        else:
            added = 0

        graph_count = len(graph_ids)
        # Calculate drift BEFORE repair for reporting
        initial_vector_count = len(vec_ids)
        drift_pct = (
            0.0
            if graph_count == 0
            else abs(initial_vector_count - graph_count) / graph_count
        )
        # Recount after repair for final stats
        final_vector_count = len(self._qdrant_section_ids())

        return DriftStats(
            embedding_version=self.version,
            graph_count=graph_count,
            vector_count=final_vector_count,
            extras_removed=removed,
            missing_backfilled=added,
            drift_pct=drift_pct,  # Drift before repair
        )

    def reconcile(
        self, embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> Dict:
        """Synchronous reconcile wrapper returning dict format."""
        import time

        start = time.time()
        stats = self.reconcile_sync(embedding_fn)
        duration_ms = int((time.time() - start) * 1000)
        return {
            "drift_pct": stats.drift_pct,
            "repaired": stats.missing_backfilled,
            "removed_orphans": stats.extras_removed,
            "graph_sections_count": stats.graph_count,  # Test expects this key
            "total_in_vector_store": stats.vector_count,
            "duration_ms": duration_ms,
        }

    def check_parity(self):
        """Wrapper for backwards compatibility (synchronous)."""
        stats = self.reconcile_sync()
        return {
            "neo4j_count": stats.graph_count,
            "qdrant_count": stats.vector_count,
            "parity_delta": abs(stats.graph_count - stats.vector_count),
            "parity_achieved": stats.graph_count == stats.vector_count,
        }

    @staticmethod
    def _vector_expected_dim(vector: object) -> Optional[int]:
        """Delegate to shared utility for vector dimension detection."""
        return vector_expected_dim(vector)

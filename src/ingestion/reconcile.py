from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Optional: use the same embedder as build_graph uses
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


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
    Keeps Qdrant strictly in sync with graph Section nodes for a given embedding_version.
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
            self.version = embedding_version
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
            self.version = config_or_qdrant.embedding.version
        else:
            # Test signature: Reconciler(neo4j, config, qdrant)
            self.config = config_or_qdrant
            self.qdrant = qdrant_client or config_or_qdrant
            self.collection = collection_name
            self.version = embedding_version

    def _graph_section_ids(self) -> Set[str]:
        """Get all Section IDs from Neo4j with matching embedding_version (synchronous)."""
        cypher = """
        MATCH (s:Section)
        WHERE s.embedding_version = $v
        RETURN s.id AS id
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

        filt = {
            "must": [
                {"key": "node_label", "match": {"value": "Section"}},
                {"key": "embedding_version", "match": {"value": self.version}},
            ]
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
        # Convert section IDs to UUIDs before deleting
        for nid in node_ids:
            try:
                self.qdrant.delete(
                    collection_name=self.collection,
                    points_selector={
                        "filter": {
                            "must": [
                                {"key": "node_id", "match": {"value": nid}},
                                {
                                    "key": "embedding_version",
                                    "match": {"value": self.version},
                                },
                            ]
                        }
                    },
                    wait=True,
                )
            except Exception:
                # Fall back to best-effort deletion without embedding filter
                self.qdrant.delete(
                    collection_name=self.collection,
                    points_selector={
                        "filter": {
                            "must": [
                                {"key": "node_id", "match": {"value": nid}},
                            ]
                        }
                    },
                    wait=True,
                )
        return len(node_ids)

    def _upsert_points(
        self,
        records: List[Tuple[str, List[float], Dict]],
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
        expected_dim = 384  # Default fallback
        if self.config and hasattr(self.config, "embedding"):
            expected_dim = self.config.embedding.dims
        elif pts and len(pts[0].vector) > 0:
            expected_dim = len(pts[0].vector)

        self.qdrant.upsert_validated(
            collection_name=self.collection, points=pts, expected_dim=expected_dim
        )
        return len(pts)

    def reconcile_sync(
        self, embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> DriftStats:
        """
        Make Qdrant contain exactly the Section nodes at embedding_version=self.version.
        If embedding_fn is None, uses SentenceTransformers(all-MiniLM-L6-v2) if available.
        """
        graph_ids = self._graph_section_ids()
        vec_ids = self._qdrant_section_ids()

        extras = sorted(list(vec_ids - graph_ids))
        missing = sorted(list(graph_ids - vec_ids))

        removed = self._delete_points_by_node_ids(extras)

        # Build embeddings for missing
        if missing:
            if embedding_fn is None:
                if SentenceTransformer is None:
                    raise RuntimeError("No embedding function available")
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

                def emb_fn(t):
                    return model.encode(t).tolist()

            else:
                emb_fn = embedding_fn

            # Fetch section text to embed
            text_map: Dict[str, Dict[str, str]] = {}
            cypher = """
            UNWIND $ids AS sid
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section {id: sid})
            RETURN
                s.id AS id,
                coalesce(s.text, s.content, s.title, '') AS text,
                coalesce(s.document_id, d.id) AS document_id,
                d.source_uri AS source_uri,
                d.doc_tag AS doc_tag
            """
            with self.neo4j.session() as sess:
                res = sess.run(cypher, {"ids": missing})
                for rec in res:
                    text_map[rec["id"]] = {
                        "text": rec["text"] or "",
                        "document_id": rec.get("document_id"),
                        "source_uri": rec.get("source_uri"),
                        "doc_tag": rec.get("doc_tag"),
                    }

            upserts = []
            for sid in missing:
                meta = text_map.get(
                    sid, {"text": "", "document_id": None, "source_uri": None}
                )
                vec = emb_fn(meta["text"])
                source_uri = meta.get("source_uri") or ""
                document_uri = Path(source_uri).name if source_uri else source_uri
                payload = {
                    "node_id": sid,
                    "node_label": "Section",
                    "document_id": meta.get("document_id"),
                    "document_uri": document_uri,
                    "source_uri": source_uri,
                    "doc_tag": meta.get("doc_tag"),
                    "embedding_version": self.version,
                }
                upserts.append((sid, vec, payload))

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

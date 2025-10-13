from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (FieldCondition, Filter, MatchValue,
                                  PointIdsList, PointStruct)

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
            self.collection = "sections"  # Use config's default
            self.version = config_or_qdrant.embedding.version
        else:
            # Test signature: Reconciler(neo4j, config, qdrant)
            self.config = config_or_qdrant
            self.qdrant = qdrant_client or config_or_qdrant
            self.collection = collection_name
            self.version = embedding_version

    async def _graph_section_ids(self) -> Set[str]:
        cypher = """
        MATCH (s:Section)
        WHERE s.embedding_version = $v
        RETURN s.id AS id
        """
        ids: Set[str] = set()
        async with self.neo4j.session() as sess:
            result = await sess.run(cypher, {"v": self.version})
            async for rec in result:
                ids.add(rec["id"])
        return ids

    def _qdrant_section_ids(self) -> Set[str]:
        # Scroll all points for the version; small datasets in tests

        filt = Filter(
            must=[
                FieldCondition(key="label", match=MatchValue(value="Section")),
                FieldCondition(
                    key="embedding_version", match=MatchValue(value=self.version)
                ),
            ]
        )

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
                nid = p.payload.get("node_id")
                if nid:
                    out.add(nid)
            if next_page is None:
                break
        return out

    def _delete_points_by_node_ids(self, node_ids: List[str]) -> int:
        if not node_ids:
            return 0
        # Support both modern and legacy delete selectors via PointIdsList
        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=node_ids),
        )
        return len(node_ids)

    def _upsert_points(
        self,
        records: List[Tuple[str, List[float], Dict]],
    ) -> int:
        if not records:
            return 0
        pts = [
            PointStruct(
                id=rec[0],
                vector=rec[1],
                payload=rec[2],
            )
            for rec in records
        ]
        self.qdrant.upsert(collection_name=self.collection, points=pts)
        return len(pts)

    async def reconcile_async(
        self, embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> DriftStats:
        """
        Make Qdrant contain exactly the Section nodes at embedding_version=self.version.
        If embedding_fn is None, uses SentenceTransformers(all-MiniLM-L6-v2) if available.
        """
        graph_ids = await self._graph_section_ids()
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

                async def _embed_text(text: str) -> List[float]:
                    return model.encode(text).tolist()

                # use _embed_text below
                def emb_fn(t):
                    return model.encode(t).tolist()

            else:
                emb_fn = embedding_fn

            # Fetch section text to embed
            text_map: Dict[str, str] = {}
            cypher = """
            UNWIND $ids AS sid
            MATCH (s:Section {id: sid})
            RETURN s.id AS id, coalesce(s.content, s.title) AS text
            """
            async with self.neo4j.session() as sess:
                res = await sess.run(cypher, {"ids": missing})
                async for rec in res:
                    text_map[rec["id"]] = rec["text"] or ""

            upserts = []
            for sid in missing:
                vec = emb_fn(text_map.get(sid, ""))
                payload = {
                    "node_id": sid,
                    "label": "Section",
                    "embedding_version": self.version,
                }
                upserts.append((sid, vec, payload))

            added = self._upsert_points(upserts)
        else:
            added = 0

        graph_count = len(graph_ids)
        vector_count = len(self._qdrant_section_ids())  # recount after repair
        drift_pct = (
            0.0 if graph_count == 0 else abs(vector_count - graph_count) / graph_count
        )

        return DriftStats(
            embedding_version=self.version,
            graph_count=graph_count,
            vector_count=vector_count,
            extras_removed=removed,
            missing_backfilled=added,
            drift_pct=drift_pct,
        )

    def reconcile(
        self, embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> Dict:
        """Synchronous reconcile for backwards compatibility with tests."""
        import asyncio

        stats = asyncio.run(self.reconcile_async(embedding_fn))
        return {
            "drift_pct": stats.drift_pct,
            "repaired": stats.missing_backfilled,
            "removed_orphans": stats.extras_removed,
            "total_vectorized": stats.graph_count,
            "total_in_vector_store": stats.vector_count,
        }

    async def check_parity(self):
        """Wrapper for backwards compatibility."""
        stats = await self.reconcile_async()
        return {
            "neo4j_count": stats.graph_count,
            "qdrant_count": stats.vector_count,
            "parity_delta": abs(stats.graph_count - stats.vector_count),
            "parity_achieved": stats.graph_count == stats.vector_count,
        }

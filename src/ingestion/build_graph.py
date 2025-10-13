# Implements Phase 3, Task 3.3 (Graph construction with embeddings)
# See: /docs/spec.md §3 (Data model, IDs, vectors)
# See: /docs/implementation-plan.md → Task 3.3
# See: /docs/pseudocode-reference.md → Task 3.3

import time
from datetime import datetime
from typing import Dict, List

from neo4j import Driver
from sentence_transformers import SentenceTransformer

from src.shared.config import Config
from src.shared.observability import get_logger

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

        # Ensure Qdrant collection exists if using Qdrant
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

        Args:
            document: Document metadata
            sections: List of sections
            entities: Dict of entities keyed by ID
            mentions: List of MENTIONS relationships

        Returns:
            Stats dict
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

        with self.driver.session() as session:
            # Step 1: Upsert Document node
            self._upsert_document_node(session, document)

            # Step 2: Upsert Sections in batches
            stats["sections_upserted"] = self._upsert_sections(
                session, document["id"], sections
            )

            # Step 3: Upsert Entities in batches
            stats["entities_upserted"] = self._upsert_entities(session, entities)

            # Step 4: Create MENTIONS edges in batches
            stats["mentions_created"] = self._create_mentions(session, mentions)

        # Step 5: Compute embeddings and upsert to vector store
        embedding_stats = self._process_embeddings(sections, entities)
        stats["embeddings_computed"] = embedding_stats["computed"]
        stats["vectors_upserted"] = embedding_stats["upserted"]

        stats["duration_ms"] = int((time.time() - start_time) * 1000)
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
        """Upsert Section nodes and HAS_SECTION relationships in batches."""
        batch_size = self.config.ingestion.batch_size
        total_sections = 0

        for i in range(0, len(sections), batch_size):
            batch = sections[i : i + batch_size]

            query = """
            UNWIND $sections as sec
            MERGE (s:Section {id: sec.id})
            SET s.document_id = sec.document_id,
                s.level = sec.level,
                s.title = sec.title,
                s.anchor = sec.anchor,
                s.order = sec.order,
                s.text = sec.text,
                s.tokens = sec.tokens,
                s.checksum = sec.checksum,
                s.updated_at = datetime()
            WITH s, sec
            MATCH (d:Document {id: $document_id})
            MERGE (d)-[r:HAS_SECTION]->(s)
            SET r.order = sec.order,
                r.updated_at = datetime()
            RETURN count(s) as count
            """

            result = session.run(query, sections=batch, document_id=document_id)
            count = result.single()["count"]
            total_sections += count

            logger.debug(
                "Section batch upserted",
                batch_num=i // batch_size + 1,
                count=count,
            )

        return total_sections

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
        """Create MENTIONS relationships in batches."""
        batch_size = self.config.ingestion.batch_size
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
                "MENTIONS batch created",
                batch_num=i // batch_size + 1,
                count=count,
            )

        return total_mentions

    def _process_embeddings(
        self, sections: List[Dict], entities: Dict[str, Dict]
    ) -> Dict:
        """Compute embeddings and upsert to vector store."""
        stats = {"computed": 0, "upserted": 0}

        # Initialize embedder lazily
        if not self.embedder:
            logger.info(
                "Loading embedding model", model=self.config.embedding.model_name
            )
            self.embedder = SentenceTransformer(self.config.embedding.model_name)

        # Purge existing vectors for this document BEFORE upserting to prevent drift
        if sections and self.vector_primary == "qdrant" and self.qdrant_client:
            document_id = sections[0].get("document_id")
            if document_id:
                self.qdrant_client.purge_document(
                    self.config.search.vector.qdrant.collection_name, document_id
                )
                logger.debug("Purged vectors for document", document_id=document_id)

        # Process sections
        for section in sections:
            # Compute embedding
            text_to_embed = self._build_section_text_for_embedding(section)
            embedding = self.embedder.encode(text_to_embed).tolist()
            stats["computed"] += 1

            # Upsert to vector store
            if self.vector_primary == "qdrant":
                self._upsert_to_qdrant(section["id"], embedding, section, "Section")
                stats["upserted"] += 1

                # Dual write to Neo4j if enabled
                if self.dual_write:
                    self._upsert_to_neo4j_vector(section["id"], embedding, "Section")
                else:
                    # Always set embedding_version in Neo4j for reconciliation tracking
                    self._set_embedding_version_in_neo4j(section["id"], "Section")

            else:  # neo4j primary
                self._upsert_to_neo4j_vector(section["id"], embedding, "Section")
                stats["upserted"] += 1

                # Dual write to Qdrant if enabled
                if self.dual_write and self.qdrant_client:
                    self._upsert_to_qdrant(section["id"], embedding, section, "Section")

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
        """Ensure Qdrant collection exists with correct schema."""
        from qdrant_client.models import Distance, VectorParams

        collection_name = self.config.search.vector.qdrant.collection_name

        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)

            if not exists:
                # Create collection with vector config
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding.dims,
                        distance=(
                            Distance.COSINE
                            if self.config.embedding.similarity == "cosine"
                            else Distance.EUCLID
                        ),
                    ),
                )
                logger.info(
                    "Created Qdrant collection",
                    collection=collection_name,
                    dims=self.config.embedding.dims,
                )
            else:
                logger.debug(
                    "Qdrant collection already exists", collection=collection_name
                )
        except Exception as e:
            logger.error("Failed to ensure Qdrant collection", error=str(e))
            raise

    def _upsert_to_qdrant(
        self, node_id: str, embedding: List[float], metadata: Dict, label: str
    ):
        """Upsert embedding to Qdrant with exact section_id as point_id."""
        if not self.qdrant_client:
            logger.warning("Qdrant client not available")
            return

        import uuid

        from qdrant_client.models import PointStruct

        collection_name = self.config.search.vector.qdrant.collection_name

        # Convert section_id (SHA-256 hex string) to UUID for Qdrant compatibility
        # Use UUID5 with a namespace to ensure deterministic mapping
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))

        # CRITICAL: Store original node_id in payload for reconciliation
        # Parity checks will use payload.node_id to match with Neo4j
        point = PointStruct(
            id=point_uuid,  # UUID compatible with Qdrant
            vector=embedding,
            payload={
                "node_id": node_id,  # Original section_id for matching
                "label": label,
                "document_id": metadata.get("document_id"),
                "title": metadata.get("title"),
                "anchor": metadata.get("anchor"),
                "updated_at": datetime.utcnow().isoformat(),
                "embedding_version": self.embedding_version,
            },
        )

        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
        )

        logger.debug(
            "Vector upserted to Qdrant",
            node_id=node_id,
            point_uuid=point_uuid,
            collection=collection_name,
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


# Integration test wrapper
def ingest_document(source_uri: str, content: str, format: str = "markdown") -> Dict:
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
    from qdrant_client import QdrantClient

    from src.ingestion.extract import extract_entities
    from src.ingestion.parsers.markdown import parse_markdown
    from src.shared.config import get_config, get_settings

    config = get_config()
    settings = get_settings()

    # Initialize clients
    neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=3600,
    )

    qdrant_client = None
    if config.search.vector.primary == "qdrant" or config.search.vector.dual_write:
        qdrant_client = QdrantClient(
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

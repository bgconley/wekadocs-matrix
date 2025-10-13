#!/usr/bin/env python3
"""
Seed Minimal Graph for Phase 2 Testing
Creates deterministic test data in Neo4j and Qdrant.
NO MOCKS - uses real services.
"""

import hashlib
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.shared.config import get_config


def deterministic_id(namespace: str, *parts) -> str:
    """Generate deterministic ID from parts."""
    content = ":".join([namespace] + [str(p) for p in parts])
    return hashlib.sha256(content.encode()).hexdigest()


def seed_graph():
    """Seed minimal graph with ~10 sections + entities."""
    config = get_config()

    # Connect to Neo4j
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "testpassword123")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Connect to Qdrant if primary
    vector_primary = config.search.vector.primary
    qdrant_client = None
    if vector_primary == "qdrant":
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Create collection
        collection_name = "weka_sections"  # Hardcoded for now
        embedding_dims = config.embedding.dims

        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dims, distance=Distance.COSINE),
        )

    # Load embedder
    model_name = config.embedding.model_name
    embedder = SentenceTransformer(model_name)
    embedding_version = config.embedding.version

    print("🌱 Seeding minimal graph...")
    print(f"   Vector primary: {vector_primary}")
    print(f"   Embedder: {model_name}")

    # Define test data
    documents = [
        {
            "id": deterministic_id("doc", "install-guide"),
            "source_uri": "docs://weka/install-guide.md",
            "source_type": "markdown",
            "title": "WekaFS Installation Guide",
            "version": "4.2",
            "checksum": "abc123",
            "last_edited": "2024-01-15T00:00:00Z",
        }
    ]

    sections = [
        {
            "id": deterministic_id("section", "install-guide", "prerequisites"),
            "document_id": documents[0]["id"],
            "level": 2,
            "title": "Prerequisites",
            "anchor": "prerequisites",
            "order": 1,
            "text": "Before installing WekaFS, ensure your system meets minimum requirements: 8 CPUs, 64GB RAM, fast NVMe storage.",
            "tokens": 20,
        },
        {
            "id": deterministic_id("section", "install-guide", "installation"),
            "document_id": documents[0]["id"],
            "level": 2,
            "title": "Installation Steps",
            "anchor": "installation",
            "order": 2,
            "text": "Run weka cluster create to initialize a new cluster. Configure network settings and storage tiers.",
            "tokens": 18,
        },
        {
            "id": deterministic_id("section", "install-guide", "troubleshooting"),
            "document_id": documents[0]["id"],
            "level": 2,
            "title": "Troubleshooting",
            "anchor": "troubleshooting",
            "order": 3,
            "text": "If installation fails with E1001, check network connectivity. Use weka status to verify cluster health.",
            "tokens": 19,
        },
        {
            "id": deterministic_id("section", "install-guide", "configuration"),
            "document_id": documents[0]["id"],
            "level": 2,
            "title": "Post-Installation Configuration",
            "anchor": "configuration",
            "order": 4,
            "text": "Configure NFS_ENABLED and set mount points. Adjust CACHE_SIZE for optimal performance.",
            "tokens": 16,
        },
    ]

    commands = [
        {
            "id": deterministic_id("command", "weka cluster create"),
            "name": "weka cluster create",
            "description": "Create a new WekaFS cluster",
            "category": "cluster",
        },
        {
            "id": deterministic_id("command", "weka status"),
            "name": "weka status",
            "description": "Check cluster status and health",
            "category": "monitoring",
        },
    ]

    errors = [
        {
            "id": deterministic_id("error", "E1001"),
            "code": "E1001",
            "name": "Network Connectivity Error",
            "description": "Cannot reach cluster nodes",
        }
    ]

    configurations = [
        {
            "id": deterministic_id("config", "NFS_ENABLED"),
            "name": "NFS_ENABLED",
            "description": "Enable NFS protocol support",
            "category": "protocol",
        },
        {
            "id": deterministic_id("config", "CACHE_SIZE"),
            "name": "CACHE_SIZE",
            "description": "Size of the SSD cache tier",
            "category": "performance",
        },
    ]

    procedures = [
        {
            "id": deterministic_id("procedure", "resolve-e1001"),
            "name": "Resolve E1001 Network Error",
            "description": "Steps to fix network connectivity issues",
        }
    ]

    # Insert data into Neo4j
    with driver.session() as session:
        # Documents
        for doc in documents:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d += $props, d.updated_at = datetime()
            """,
                id=doc["id"],
                props=doc,
            )

        # Sections with embeddings
        section_vectors = {}
        for section in sections:
            # Generate embedding
            text = f"{section['title']}\n\n{section['text']}"
            vector = embedder.encode(text).tolist()
            section_vectors[section["id"]] = vector

            props = section.copy()
            if vector_primary == "neo4j":
                props["vector_embedding"] = vector
                props["embedding_version"] = embedding_version
            props["checksum"] = hashlib.md5(section["text"].encode()).hexdigest()

            session.run(
                """
                MERGE (s:Section {id: $id})
                SET s += $props, s.updated_at = datetime()
            """,
                id=section["id"],
                props=props,
            )

            # Link to document
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                MATCH (s:Section {id: $sec_id})
                MERGE (d)-[:HAS_SECTION {order: $order}]->(s)
            """,
                doc_id=section["document_id"],
                sec_id=section["id"],
                order=section["order"],
            )

        # Commands
        for cmd in commands:
            session.run(
                """
                MERGE (c:Command {id: $id})
                SET c += $props, c.updated_at = datetime()
            """,
                id=cmd["id"],
                props=cmd,
            )

        # Errors
        for error in errors:
            session.run(
                """
                MERGE (e:Error {id: $id})
                SET e += $props, e.updated_at = datetime()
            """,
                id=error["id"],
                props=error,
            )

        # Configurations
        for cfg in configurations:
            session.run(
                """
                MERGE (c:Configuration {id: $id})
                SET c += $props, c.updated_at = datetime()
            """,
                id=cfg["id"],
                props=cfg,
            )

        # Procedures
        for proc in procedures:
            session.run(
                """
                MERGE (p:Procedure {id: $id})
                SET p += $props, p.updated_at = datetime()
            """,
                id=proc["id"],
                props=proc,
            )

        # Create relationships (MENTIONS)
        session.run(
            """
            MATCH (s:Section {id: $sec_id})
            MATCH (c:Command {id: $cmd_id})
            MERGE (s)-[:MENTIONS {confidence: 0.95, source_section_id: $sec_id}]->(c)
        """,
            sec_id=sections[1]["id"],
            cmd_id=commands[0]["id"],
        )  # Installation → weka cluster create

        session.run(
            """
            MATCH (s:Section {id: $sec_id})
            MATCH (c:Command {id: $cmd_id})
            MERGE (s)-[:MENTIONS {confidence: 0.90, source_section_id: $sec_id}]->(c)
        """,
            sec_id=sections[2]["id"],
            cmd_id=commands[1]["id"],
        )  # Troubleshooting → weka status

        session.run(
            """
            MATCH (s:Section {id: $sec_id})
            MATCH (e:Error {id: $err_id})
            MERGE (s)-[:MENTIONS {confidence: 0.98, source_section_id: $sec_id}]->(e)
        """,
            sec_id=sections[2]["id"],
            err_id=errors[0]["id"],
        )  # Troubleshooting → E1001

        session.run(
            """
            MATCH (s:Section {id: $sec_id})
            MATCH (c:Configuration {id: $cfg_id})
            MERGE (s)-[:MENTIONS {confidence: 0.92, source_section_id: $sec_id}]->(c)
        """,
            sec_id=sections[3]["id"],
            cfg_id=configurations[0]["id"],
        )  # Configuration → NFS_ENABLED

        session.run(
            """
            MATCH (s:Section {id: $sec_id})
            MATCH (c:Configuration {id: $cfg_id})
            MERGE (s)-[:MENTIONS {confidence: 0.88, source_section_id: $sec_id}]->(c)
        """,
            sec_id=sections[3]["id"],
            cfg_id=configurations[1]["id"],
        )  # Configuration → CACHE_SIZE

        # Procedure resolves error
        session.run(
            """
            MATCH (e:Error {id: $err_id})
            MATCH (p:Procedure {id: $proc_id})
            MERGE (e)<-[:RESOLVES {confidence: 1.0}]-(p)
        """,
            err_id=errors[0]["id"],
            proc_id=procedures[0]["id"],
        )

        print(
            f"✅ Created {len(documents)} documents, {len(sections)} sections, {len(commands)} commands, {len(errors)} errors"
        )

    # Insert vectors into Qdrant if primary
    if qdrant_client:
        points = []
        for i, section in enumerate(sections):
            vector = section_vectors[section["id"]]
            # Qdrant expects string UUID or int, use hash as int
            point_id = abs(hash(section["id"])) % (10**16)  # Convert to positive int
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "node_id": section["id"],
                        "node_label": "Section",
                        "document_id": section["document_id"],
                        "title": section["title"],
                        "anchor": section["anchor"],
                        "updated_at": "2024-01-15T00:00:00Z",
                        "embedding_version": embedding_version,
                    },
                )
            )

        collection_name = "weka_sections"  # Hardcoded for now
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Inserted {len(points)} vectors into Qdrant")

    driver.close()
    print("🎉 Seed complete!")


if __name__ == "__main__":
    seed_graph()

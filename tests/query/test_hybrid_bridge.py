import random
import uuid

import pytest

from src.query.hybrid_retrieval import HybridRetriever
from src.shared.config import get_config, get_embedding_settings
from src.shared.connections import get_connection_manager

# Helper to generate unique IDs
TEST_ID = f"test_bridge_{uuid.uuid4().hex[:8]}"
TEST_DOC_ID = f"doc_{TEST_ID}"
TEST_CHUNK_ID = f"chunk_{TEST_ID}"
TEST_NEIGHBOR_ID = f"neighbor_{TEST_ID}"


class MockEmbeddingProvider:
    """Generates dummy vectors without network calls."""

    def __init__(self, dims=1024):
        self.dims = dims
        self.provider_name = "mock"
        self.model_id = "mock-model"

    def embed_query(self, text):
        # Return a consistent random vector
        rng = random.Random(text)
        return [rng.random() for _ in range(self.dims)]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    # Satisfy contract
    def embed_sparse(self, texts):
        return None

    def embed_colbert(self, texts):
        return None

    def embed_query_all(self, text):
        from src.providers.embeddings.contracts import QueryEmbeddingBundle

        return QueryEmbeddingBundle(dense=self.embed_query(text))


@pytest.fixture(scope="module")
def real_drivers():
    """Get real connections but MOCK the embedder."""
    manager = get_connection_manager()
    driver = manager.get_neo4j_driver()
    client = manager.get_qdrant_client()

    # Use our mock instead of the factory
    embedder = MockEmbeddingProvider(dims=1024)

    return driver, client, embedder


@pytest.fixture(scope="module")
def seed_data(real_drivers):
    """Insert test data into Neo4j and Qdrant."""
    driver, client, embedder = real_drivers

    print(f"\n--- Seeding Data: {TEST_CHUNK_ID} ---")

    # 1. Seed Neo4j (Chunk + Neighbor)
    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = \"This is the primary result found by vector search.\",
                c.document_id = $doc_id,
                c.heading = \"Vector Winner\",
                c.order = 0,
                c.level = 0,
                c.token_count = 0
            MERGE (n:Chunk {id: $neighbor_id})
            SET n.text = \"This is the neighbor found by graph expansion.\",
                n.document_id = $doc_id,
                n.heading = \"Graph Neighbor\",
                n.order = 1,
                n.level = 0,
                n.token_count = 0
            // Expansion logic uses NEXT_CHUNK edges (not NEXT)
            MERGE (c)-[:NEXT_CHUNK]->(n)
            MERGE (c)-[:IN_DOCUMENT]->(d)
            MERGE (n)-[:IN_DOCUMENT]->(d)
        """,
            doc_id=TEST_DOC_ID,
            chunk_id=TEST_CHUNK_ID,
            neighbor_id=TEST_NEIGHBOR_ID,
        )

    # 2. Seed Qdrant
    text = "This is the primary result found by vector search."
    vector = embedder.embed_query(text)

    config = get_config()
    embedding_settings = get_embedding_settings()
    collection_name = config.search.vector.qdrant.collection_name

    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, TEST_CHUNK_ID)),
                vector={"content": vector},
                payload={
                    "id": TEST_CHUNK_ID,
                    "text": text,
                    "document_id": TEST_DOC_ID,
                    "heading": "Vector Winner",
                    # Required for HybridRetriever filter alignment
                    "embedding_version": embedding_settings.version,
                },
            )
        ],
        wait=True,
    )

    yield

    # Cleanup
    print("\n--- Cleaning Data ---")
    with driver.session() as session:
        session.run(
            "MATCH (n) WHERE n.id IN [$c, $n, $d] DETACH DELETE n",
            c=TEST_CHUNK_ID,
            n=TEST_NEIGHBOR_ID,
            d=TEST_DOC_ID,
        )

    client.delete(
        collection_name=collection_name,
        points_selector=[str(uuid.uuid5(uuid.NAMESPACE_DNS, TEST_CHUNK_ID))],
        wait=True,
    )


@pytest.mark.asyncio
async def test_real_vector_to_graph_handoff(real_drivers, seed_data):
    """
    Integration Test:
    1. Search using the same text (which generates same random vector).
    2. Verify the result set includes the Seed Chunk.
    3. Verify the result set includes the Neighbor Chunk (via Graph Expansion).
    """
    driver, client, embedder = real_drivers

    # Initialize Retriever with real components + mock embedder
    retriever = HybridRetriever(driver, client, embedder)
    # Expansion is disabled in the dev config; enable it for this test.
    retriever.expansion_enabled = True

    # Query matches the seeded text (producing identical mock vector)
    query = "This is the primary result found by vector search."

    # Execute Retrieve (with expansion enabled)
    print(f"\n--- Executing Retrieve: '{query}' ---")
    results, metrics = retriever.retrieve(
        query, top_k=5, expand=True, expand_when="always"
    )

    print(f"Retrieved {len(results)} results.")

    found_seed = False
    found_neighbor = False

    for res in results:
        print(f" - [{res.chunk_id}] {res.heading}: {res.text[:50]}...")
        if res.chunk_id == TEST_CHUNK_ID:
            found_seed = True
        if res.chunk_id == TEST_NEIGHBOR_ID:
            found_neighbor = True

    assert found_seed, "Failed to retrieve the seed chunk from Qdrant!"
    assert found_neighbor, "Failed to retrieve the neighbor chunk via Graph Expansion!"

    print(
        "\nSUCCESS: Vector search found the seed, and Graph Expansion found the neighbor."
    )

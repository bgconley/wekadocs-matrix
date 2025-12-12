"""
Phase 7E-2: Hybrid Retrieval Integration Tests
Tests BM25+Vector fusion, bounded expansion, and context budget enforcement
Against REAL Neo4j and Qdrant - NO MOCKS

Reference: Phase 7E-2 Acceptance Criteria:
- API returns bm25_score, vec_score, fused_score + method per chunk
- Stitched context never exceeds budget (4500 tokens)
- Expansion gated by config
- A/B testing of RRF vs weighted fusion
"""

import hashlib
import os
import time
from datetime import datetime
from typing import Dict, List

import pytest
from neo4j import Driver
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.providers.embeddings.bge_m3_service import BGEM3ServiceProvider
from src.providers.settings import EmbeddingCapabilities, EmbeddingSettings
from src.providers.tokenizer_service import TokenizerService
from src.query.context_assembly import ContextAssembler
from src.query.hybrid_retrieval import (
    BM25Retriever,
    ChunkResult,
    FusionMethod,
    HybridRetriever,
    VectorRetriever,
)


class TestHybridRetrievalIntegration:
    """
    Integration tests for Phase 7E-2 hybrid retrieval.
    Tests against REAL databases, NO mocks.
    """

    @pytest.fixture
    def neo4j_driver(self, neo4j_driver) -> Driver:
        """Use the real Neo4j driver from conftest."""
        return neo4j_driver

    @pytest.fixture
    def qdrant_client(self, qdrant_client) -> QdrantClient:
        """Use the real Qdrant client from conftest."""
        return qdrant_client

    @pytest.fixture
    def embedder(self) -> BGEM3ServiceProvider:
        """Create BGEM3 embedder for tests (local service)."""
        settings = EmbeddingSettings(
            profile="bge_m3_test",
            provider="bge-m3-service",
            model_id="bge-m3",
            version="bge-m3",
            dims=1024,
            similarity="cosine",
            task="symmetric",
            tokenizer_backend="hf",
            tokenizer_model_id="BAAI/bge-m3",
            service_url=os.getenv("BGE_M3_API_URL", "http://127.0.0.1:9000"),
            capabilities=EmbeddingCapabilities(
                supports_dense=True,
                supports_sparse=True,
                supports_colbert=True,
                supports_long_sequences=True,
                normalized_output=True,
                multilingual=True,
            ),
        )
        return BGEM3ServiceProvider(settings=settings)

    @pytest.fixture
    def tokenizer(self) -> TokenizerService:
        """Create tokenizer for tests."""
        return TokenizerService()

    @pytest.fixture
    def hybrid_retriever(
        self, neo4j_driver, qdrant_client, embedder, tokenizer
    ) -> HybridRetriever:
        """Create hybrid retriever with real connections."""
        retriever = HybridRetriever(
            neo4j_driver=neo4j_driver,
            qdrant_client=qdrant_client,
            embedder=embedder,
            tokenizer=tokenizer,
        )
        retriever.vector_field_weights = {"content": 1.0}
        retriever.vector_retriever.field_weights = {"content": 1.0}
        retriever.vector_retriever.sparse_field_name = None
        retriever.vector_retriever.supports_sparse = False
        return retriever

    def _create_test_chunks(self, doc_id: str = "test_doc_hybrid") -> List[Dict]:
        """
        Create test chunks with varying characteristics for hybrid search.

        Creates chunks that will test:
        - BM25 scoring (text with keywords)
        - Vector similarity (semantic content)
        - Adjacency relationships (NEXT_CHUNK)
        - Token budget enforcement
        """
        chunks = []

        # Chunk 1: High BM25 relevance for "network configuration"
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_1".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_1",
                "order": 0,
                "level": 2,
                "heading": "Network Configuration Guide",
                "text": "Network configuration involves setting up IP addresses, subnets, and routing. "
                "Configure network interfaces using the network configuration tool. "
                "Network settings must be configured properly for optimal performance.",
                "token_count": 25,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_1"],
            }
        )

        # Chunk 2: Semantically similar to network but different keywords
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_2".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_1",
                "order": 1,
                "level": 2,
                "heading": "IP Setup Instructions",
                "text": "Setting up IP connectivity requires understanding of TCP/IP protocols. "
                "Establish connections between nodes using proper addressing schemes. "
                "Ensure all endpoints can communicate effectively.",
                "token_count": 20,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_2"],
            }
        )

        # Chunk 3: Adjacent chunk for expansion testing
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_3".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_1",
                "order": 2,
                "level": 2,
                "heading": "Troubleshooting Network Issues",
                "text": "Common problems include incorrect subnet masks and gateway addresses. "
                "Use ping and traceroute to diagnose connectivity issues. "
                "Check firewall rules if connections are blocked.",
                "token_count": 18,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_3"],
            }
        )

        # Chunk 4: Different topic for contrast
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_4".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_2",
                "order": 0,
                "level": 2,
                "heading": "Storage Management",
                "text": "Storage volumes can be managed through the admin interface. "
                "Allocate disk space according to workload requirements. "
                "Monitor storage usage regularly to prevent capacity issues.",
                "token_count": 22,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_4"],
            }
        )

        # Chunk 5: Large chunk for token budget testing (simulated)
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_5".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_3",
                "order": 0,
                "level": 2,
                "heading": "Complete System Reference",
                "text": "This is a very large reference section with extensive documentation. "
                * 50,
                "token_count": 500,  # Large token count
                "is_combined": True,
                "is_split": False,
                "original_section_ids": ["sec_5a", "sec_5b", "sec_5c"],
            }
        )

        # Chunk 6: Adjacent to chunk 3 for expansion testing
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_6".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_1",
                "order": 3,
                "level": 2,
                "heading": "Advanced Network Features",
                "text": "Advanced features include VLAN configuration and bonding. "
                "Quality of service settings ensure optimal traffic flow. "
                "Configure redundancy for high availability networks.",
                "token_count": 18,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_6"],
            }
        )

        # Chunk 7: Another unrelated chunk
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_7".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_2",
                "order": 1,
                "level": 2,
                "heading": "Backup Procedures",
                "text": "Regular backups protect against data loss. "
                "Schedule automated snapshots for critical volumes. "
                "Test restore procedures regularly.",
                "token_count": 15,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_7"],
            }
        )

        # Chunk 8: Another adjacent for parent_1
        chunks.append(
            {
                "id": hashlib.sha256(f"{doc_id}|sec_8".encode()).hexdigest(),
                "document_id": doc_id,
                "parent_section_id": "parent_1",
                "order": 4,
                "level": 2,
                "heading": "Network Monitoring",
                "text": "Monitor network performance using built-in tools. "
                "Track bandwidth utilization and packet loss. "
                "Set alerts for connectivity issues.",
                "token_count": 16,
                "is_combined": False,
                "is_split": False,
                "original_section_ids": ["sec_8"],
            }
        )

        return chunks

    def _ingest_test_chunks(
        self,
        neo4j_driver: Driver,
        qdrant_client: QdrantClient,
        chunks: List[Dict],
        embedder: BGEM3ServiceProvider,
    ):
        """Ingest test chunks into Neo4j and Qdrant."""
        # Clear any existing test data from Neo4j if driver provided
        if neo4j_driver:
            with neo4j_driver.session() as session:
                session.run(
                    "MATCH (c:Chunk {document_id: $doc_id}) DETACH DELETE c",
                    doc_id=chunks[0]["document_id"],
                )

        # Delete from Qdrant if client provided
        if qdrant_client:
            try:
                qdrant_client.delete(
                    collection_name="chunks",
                    points_selector={
                        "filter": {
                            "must": [
                                {
                                    "key": "document_id",
                                    "match": {"value": chunks[0]["document_id"]},
                                }
                            ]
                        }
                    },
                )
            except Exception:
                pass  # Collection might not exist yet

        # Add embedding fields to all chunks (needed for both Neo4j and Qdrant)
        for chunk in chunks:
            chunk["vector_embedding"] = embedder.embed_documents([chunk["text"]])[0]
            chunk["embedding_version"] = "bge-m3"
            chunk["embedding_provider"] = embedder.provider_name
            chunk["embedding_dimensions"] = embedder.dims
            chunk["embedding_timestamp"] = datetime.utcnow().isoformat()
            chunk["boundaries_json"] = "{}"
            chunk["updated_at"] = datetime.utcnow()

        # Ingest chunks into Neo4j if driver provided
        if neo4j_driver:
            with neo4j_driver.session() as session:
                for chunk in chunks:
                    # Create chunk node
                    session.run(
                        """
                        MERGE (c:Section:Chunk {id: $chunk_data.id})
                        SET c = $chunk_data
                    """,
                        chunk_data=chunk,
                    )

                # Create NEXT_CHUNK relationships
                session.run(
                    """
                    MATCH (c1:Chunk {document_id: $doc_id})
                    MATCH (c2:Chunk {document_id: $doc_id})
                    WHERE c1.parent_section_id = c2.parent_section_id
                      AND c1.order = c2.order - 1
                    MERGE (c1)-[:NEXT_CHUNK {parent_section_id: c1.parent_section_id}]->(c2)
                """,
                    doc_id=chunks[0]["document_id"],
                )

        # Ingest into Qdrant if client provided
        if qdrant_client:
            import uuid

            from qdrant_client.models import PointStruct

            qdrant_client.recreate_collection(
                collection_name="chunks",
                vectors_config={
                    "content": VectorParams(
                        size=embedder.dims, distance=Distance.COSINE
                    )
                },
            )

            points = []
            for chunk in chunks:
                # Create point for Qdrant - use UUID for ID
                point = PointStruct(
                    id=str(
                        uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"])
                    ),  # Deterministic UUID from chunk ID
                    vector={"content": chunk["vector_embedding"]},
                    payload={
                        k: v for k, v in chunk.items() if k not in ["vector_embedding"]
                    },
                )
                points.append(point)

            qdrant_client.upsert(collection_name="chunks", points=points, wait=True)

        time.sleep(1)  # Allow indexes to update

    @pytest.mark.integration
    def test_bm25_retriever(self, neo4j_driver, embedder):
        """Test BM25 retriever returns results with scores."""
        chunks = self._create_test_chunks()
        self._ingest_test_chunks(neo4j_driver, None, chunks, embedder)

        retriever = BM25Retriever(neo4j_driver)
        results = retriever.search("network configuration", top_k=5)

        # Verify results
        assert len(results) > 0, "BM25 should return results"

        # Check first result has high relevance to query
        first = results[0]
        assert first.bm25_score is not None, "BM25 score must be present"
        assert first.bm25_score > 0, "BM25 score must be positive"
        assert "network" in first.text.lower(), "Top result should contain query terms"

        # Verify all required fields
        for result in results:
            assert result.chunk_id
            assert result.document_id
            assert result.text
            assert result.token_count >= 0

    @pytest.mark.integration
    def test_vector_retriever(self, qdrant_client, embedder):
        """Test vector retriever returns results with scores."""
        chunks = self._create_test_chunks()
        self._ingest_test_chunks(None, qdrant_client, chunks, embedder)

        retriever = VectorRetriever(qdrant_client, embedder, "chunks")
        results = retriever.search("how to set up networking", top_k=5)

        # Verify results
        assert len(results) > 0, "Vector search should return results"

        # Check scores
        first = results[0]
        assert first.vector_score is not None, "Vector score must be present"
        assert 0 <= first.vector_score <= 1, "Cosine similarity should be in [0,1]"

        # Verify descending scores
        for i in range(len(results) - 1):
            assert (
                results[i].vector_score >= results[i + 1].vector_score
            ), "Results should be sorted by score"

    @pytest.mark.integration
    def test_rrf_fusion(self, hybrid_retriever):
        """Test RRF fusion combines BM25 and vector scores correctly."""
        chunks = self._create_test_chunks()
        embedder = hybrid_retriever.vector_retriever.embedder
        self._ingest_test_chunks(
            hybrid_retriever.neo4j_driver,
            hybrid_retriever.vector_retriever.client,
            chunks,
            embedder,
        )

        # Set fusion method to RRF
        hybrid_retriever.fusion_method = FusionMethod.RRF

        results, metrics = hybrid_retriever.retrieve(
            "network configuration setup",
            top_k=5,
            expand=False,  # Disable expansion for this test
        )

        # Verify fusion occurred
        assert len(results) > 0, "Hybrid search should return results"
        assert metrics["fusion_method"] == "rrf"

        # Check that results have all three scores
        for result in results:
            assert result.fused_score is not None, "Fused score must be present"
            assert result.fusion_method == "rrf", "Fusion method must be marked"

            # At least one of the component scores should be present
            has_score = (result.bm25_score is not None) or (
                result.vector_score is not None
            )
            assert has_score, "Result must have at least one component score"

        # Verify RRF score properties
        # RRF scores are in (0, 2/k) where k=60, so max ~0.033
        for result in results:
            assert (
                0 < result.fused_score <= 2 / 60 + 0.01
            ), f"RRF score {result.fused_score} out of expected range"

    @pytest.mark.integration
    def test_weighted_fusion(self, hybrid_retriever):
        """Test weighted fusion with configurable alpha."""
        chunks = self._create_test_chunks()
        embedder = hybrid_retriever.vector_retriever.embedder
        self._ingest_test_chunks(
            hybrid_retriever.neo4j_driver,
            hybrid_retriever.vector_retriever.client,
            chunks,
            embedder,
        )

        # Set fusion method to weighted
        hybrid_retriever.fusion_method = FusionMethod.WEIGHTED
        hybrid_retriever.fusion_alpha = 0.7  # Favor vector search

        results, metrics = hybrid_retriever.retrieve(
            "IP connectivity setup", top_k=5, expand=False
        )

        # Verify weighted fusion
        assert metrics["fusion_method"] == "weighted"

        for result in results:
            assert result.fusion_method == "weighted"
            assert (
                0 <= result.fused_score <= 1
            ), "Weighted fusion scores should be normalized to [0,1]"

    @pytest.mark.integration
    def test_bounded_expansion(self, hybrid_retriever):
        """Test adjacency expansion with proper bounds."""
        chunks = self._create_test_chunks()
        embedder = hybrid_retriever.vector_retriever.embedder
        self._ingest_test_chunks(
            hybrid_retriever.neo4j_driver,
            hybrid_retriever.vector_retriever.client,
            chunks,
            embedder,
        )

        # Enable expansion
        hybrid_retriever.expansion_enabled = True

        # Test with short query using QUERY_LENGTH_ONLY mode
        # This disables the "scores close" trigger to test length gating in isolation
        results_short, metrics_short = hybrid_retriever.retrieve(
            "network",  # Only 1 token
            top_k=3,  # Small top_k to leave room for expansion
            expand=True,
            expand_when="query_length_only",
        )
        assert (
            metrics_short.get("expansion_count", 0) == 0
        ), f"Short query should not expand under 'query_length_only' (reason={metrics_short.get('expansion_reason')})"

        # Test with long query in AUTO mode (spec-compliant)
        # Should expand via either trigger: query_long OR scores_close
        results_long, metrics_long = hybrid_retriever.retrieve(
            "how to configure network interfaces with proper IP addresses and subnet masks",  # >12 tokens
            top_k=3,  # Small top_k to leave room for expansion
            expand=True,
            expand_when="auto",
        )
        assert (
            metrics_long.get("expansion_count", 0) > 0
        ), "Long query should trigger expansion"
        assert metrics_long.get("expansion_reason") in {
            "query_long",
            "scores_close",
        }, "Expansion should be triggered by canonical conditions"

        # Check expanded chunks
        expanded = [r for r in results_long if r.is_expanded]
        for chunk in expanded:
            assert (
                chunk.expansion_source is not None
            ), "Expanded chunks must indicate their source"
            assert chunk.fused_score < max(
                r.fused_score for r in results_long if not r.is_expanded
            ), "Expanded chunks should have lower scores than primary results"

    @pytest.mark.integration
    def test_context_budget_enforcement(self, hybrid_retriever):
        """Test that context assembly respects token budget."""
        chunks = self._create_test_chunks()
        embedder = hybrid_retriever.vector_retriever.embedder
        self._ingest_test_chunks(
            hybrid_retriever.neo4j_driver,
            hybrid_retriever.vector_retriever.client,
            chunks,
            embedder,
        )

        # Set a small budget for testing
        hybrid_retriever.context_max_tokens = 100

        results, metrics = hybrid_retriever.retrieve(
            "system documentation",
            top_k=10,
            expand=False,  # Request many results
        )

        # Verify token budget enforced
        total_tokens = sum(r.token_count for r in results)
        assert total_tokens <= 100, f"Total tokens {total_tokens} exceeds budget 100"

        assert (
            metrics["total_tokens"] <= 100
        ), "Metrics should report correct token count"

    @pytest.mark.integration
    def test_context_assembly_ordering(self):
        """Test that ContextAssembler maintains proper chunk ordering."""
        assembler = ContextAssembler()

        # Create chunks with specific ordering
        chunks = [
            ChunkResult(
                chunk_id="c1",
                document_id="doc1",
                parent_section_id="parent1",
                order=1,
                level=2,
                heading="Section 1",
                text="Second in section.",
                token_count=4,
            ),
            ChunkResult(
                chunk_id="c2",
                document_id="doc1",
                parent_section_id="parent1",
                order=0,
                level=2,
                heading="Section 1",
                text="First in section.",
                token_count=4,
            ),
            ChunkResult(
                chunk_id="c3",
                document_id="doc1",
                parent_section_id="parent2",
                order=0,
                level=2,
                heading="Section 2",
                text="Different section.",
                token_count=3,
                fused_score=0.9,  # Higher score
            ),
        ]

        # Set scores for ordering test
        chunks[0].fused_score = 0.7
        chunks[1].fused_score = 0.6

        context = assembler.assemble(chunks)

        # Verify chunks are ordered by parent_section_id then order
        assert (
            "Section 2" in context.text.split("\n")[0]
        ), "Highest scoring section should come first"

        # Within parent1 section, order should be 0 then 1
        parent1_text = "First in section.\nSecond in section."
        assert (
            parent1_text in context.text
        ), "Chunks within same parent should maintain order"

    @pytest.mark.integration
    def test_idempotency(self, hybrid_retriever):
        """Test that repeated searches return consistent results."""
        chunks = self._create_test_chunks()
        embedder = hybrid_retriever.vector_retriever.embedder
        self._ingest_test_chunks(
            hybrid_retriever.neo4j_driver,
            hybrid_retriever.vector_retriever.client,
            chunks,
            embedder,
        )

        query = "network configuration"

        # Run search multiple times
        results1, _ = hybrid_retriever.retrieve(query, top_k=5, expand=False)
        results2, _ = hybrid_retriever.retrieve(query, top_k=5, expand=False)
        results3, _ = hybrid_retriever.retrieve(query, top_k=5, expand=False)

        # Extract chunk IDs
        ids1 = [r.chunk_id for r in results1]
        ids2 = [r.chunk_id for r in results2]
        ids3 = [r.chunk_id for r in results3]

        # Verify consistency
        assert (
            ids1 == ids2 == ids3
        ), "Repeated searches should return same results in same order"

        # Verify scores are consistent (within small epsilon for floating point)
        for i in range(len(results1)):
            assert (
                abs(results1[i].fused_score - results2[i].fused_score) < 0.001
            ), "Scores should be consistent across runs"

    @pytest.mark.integration
    def test_end_to_end_retrieval_pipeline(self, hybrid_retriever):
        """
        End-to-end test of complete retrieval pipeline:
        1. Hybrid search (BM25 + Vector)
        2. Fusion (RRF)
        3. Expansion (if conditions met)
        4. Context assembly with budget
        """
        chunks = self._create_test_chunks()
        embedder = hybrid_retriever.vector_retriever.embedder
        self._ingest_test_chunks(
            hybrid_retriever.neo4j_driver,
            hybrid_retriever.vector_retriever.client,
            chunks,
            embedder,
        )

        # Complex query to trigger all features
        query = "complete guide for configuring network interfaces with IP addresses"

        results, metrics = hybrid_retriever.retrieve(
            query,
            top_k=3,
            expand=True,  # Small top_k to leave room for expansion
        )

        # Assemble context
        assembler = ContextAssembler()
        context = assembler.assemble(results, query)

        # Comprehensive verification
        assert len(results) > 0, "Should return results"
        assert metrics["bm25_count"] > 0, "BM25 should find results"
        assert metrics["vec_count"] > 0, "Vector search should find results"
        assert metrics["fusion_method"] == "rrf", "Should use RRF fusion"
        assert "expansion_count" in metrics

        # Context verification
        assert (
            context.total_tokens <= assembler.max_tokens
        ), "Context should respect token budget"
        assert len(context.chunks) <= len(
            results
        ), "Context chunks should not exceed results"
        assert context.expansion_count >= 0, "Context should report expansion count"

        # Verify assembled text quality
        assert len(context.text) > 100, "Context should have substantial content"
        assert (
            "Network Configuration" in context.text or "network" in context.text.lower()
        ), "Context should be relevant to query"


class TestRRFvsWeightedComparison:
    """A/B testing for RRF vs Weighted fusion methods."""

    @pytest.mark.integration
    def test_fusion_method_comparison(self, neo4j_driver, qdrant_client):
        """Compare RRF and Weighted fusion on same queries."""
        # Create retriever with BGEM3 provider
        settings = EmbeddingSettings(
            profile="bge_m3_test",
            provider="bge-m3-service",
            model_id="bge-m3",
            version="bge-m3",
            dims=1024,
            similarity="cosine",
            task="symmetric",
            tokenizer_backend="hf",
            tokenizer_model_id="BAAI/bge-m3",
            service_url=os.getenv("BGE_M3_API_URL", "http://127.0.0.1:9000"),
            capabilities=EmbeddingCapabilities(
                supports_dense=True,
                supports_sparse=True,
                supports_colbert=True,
                supports_long_sequences=True,
                normalized_output=True,
                multilingual=True,
            ),
        )
        embedder = BGEM3ServiceProvider(settings=settings)
        tokenizer = TokenizerService()

        retriever = HybridRetriever(
            neo4j_driver=neo4j_driver,
            qdrant_client=qdrant_client,
            embedder=embedder,
            tokenizer=tokenizer,
        )
        retriever.vector_field_weights = {"content": 1.0}
        retriever.vector_retriever.field_weights = {"content": 1.0}
        retriever.vector_retriever.sparse_field_name = None
        retriever.vector_retriever.supports_sparse = False

        # Prepare test data
        test_helper = TestHybridRetrievalIntegration()
        chunks = test_helper._create_test_chunks("comparison_doc")
        test_helper._ingest_test_chunks(neo4j_driver, qdrant_client, chunks, embedder)

        # Test queries
        queries = [
            "network configuration",  # Keyword-heavy
            "how to set up IP connectivity",  # Semantic
            "troubleshooting connection issues",  # Mixed
            "storage management",  # Different topic
        ]

        comparison_results = []

        for query in queries:
            # Test RRF
            retriever.fusion_method = FusionMethod.RRF
            rrf_results, rrf_metrics = retriever.retrieve(query, top_k=5, expand=False)

            # Test Weighted (different alphas)
            weighted_results = {}
            for alpha in [0.3, 0.5, 0.7]:
                retriever.fusion_method = FusionMethod.WEIGHTED
                retriever.fusion_alpha = alpha
                w_results, w_metrics = retriever.retrieve(query, top_k=5, expand=False)
                weighted_results[alpha] = (w_results, w_metrics)

            # Compare top results
            comparison = {
                "query": query,
                "rrf_top": rrf_results[0].chunk_id if rrf_results else None,
                "rrf_time": rrf_metrics["fusion_time_ms"],
                "weighted_0.3_top": (
                    weighted_results[0.3][0][0].chunk_id
                    if weighted_results[0.3][0]
                    else None
                ),
                "weighted_0.5_top": (
                    weighted_results[0.5][0][0].chunk_id
                    if weighted_results[0.5][0]
                    else None
                ),
                "weighted_0.7_top": (
                    weighted_results[0.7][0][0].chunk_id
                    if weighted_results[0.7][0]
                    else None
                ),
            }
            comparison_results.append(comparison)

        # Log comparison results
        print("\n=== RRF vs Weighted Fusion Comparison ===")
        for comp in comparison_results:
            print(f"\nQuery: '{comp['query']}'")
            print(f"  RRF top result:        {comp['rrf_top']}")
            print(f"  Weighted(0.3) top:     {comp['weighted_0.3_top']}")
            print(f"  Weighted(0.5) top:     {comp['weighted_0.5_top']}")
            print(f"  Weighted(0.7) top:     {comp['weighted_0.7_top']}")

        # Verify both methods are implemented and produce valid results
        # Note: With homogeneous test data, both may produce same rankings
        # The key test is that both work and scores are normalized
        for comp in comparison_results:
            assert comp["rrf_top"] is not None, "RRF should return results"
            assert (
                comp["weighted_0.3_top"] is not None
            ), "Weighted fusion should return results"

        # Verify score normalization for all methods
        retriever.fusion_method = FusionMethod.RRF
        rrf_check, _ = retriever.retrieve(queries[0], top_k=5, expand=False)
        assert all(
            0.0 <= r.fused_score <= 1.0 for r in rrf_check if r.fused_score is not None
        ), "RRF scores should be normalized to [0,1]"

        retriever.fusion_method = FusionMethod.WEIGHTED
        weighted_check, _ = retriever.retrieve(queries[0], top_k=5, expand=False)
        assert all(
            0.0 <= r.fused_score <= 1.0
            for r in weighted_check
            if r.fused_score is not None
        ), "Weighted scores should be normalized to [0,1]"


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_test_data(neo4j_driver, qdrant_client):
    """Clean up test data after each test."""
    yield

    # Cleanup Neo4j
    with neo4j_driver.session() as session:
        session.run(
            """
            MATCH (c:Chunk)
            WHERE c.document_id STARTS WITH 'test_doc' OR c.document_id STARTS WITH 'comparison_doc'
            DETACH DELETE c
        """
        )

    # Cleanup Qdrant
    try:
        for doc_id in ["test_doc_hybrid", "comparison_doc"]:
            qdrant_client.delete(
                collection_name="chunks",
                points_selector={
                    "filter": {
                        "must": [{"key": "document_id", "match": {"value": doc_id}}]
                    }
                },
            )
    except Exception:
        pass  # Collection might not exist

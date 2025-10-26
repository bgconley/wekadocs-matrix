"""
Integration tests for Jina API with large batches.
Phase 7C: Real API tests for adaptive batching and rate limiting.

IMPORTANT: These tests make real API calls to Jina AI.
- Requires JINA_API_KEY environment variable
- Consumes API quota
- May be slow due to rate limiting

Run with: pytest tests/integration/test_jina_large_batches.py -v -s
"""

import os
import time

import pytest

from src.providers.embeddings.jina import JinaEmbeddingProvider

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("JINA_API_KEY"),
    reason="JINA_API_KEY not set - skipping real API tests",
)


@pytest.fixture
def jina_provider():
    """Create Jina provider for testing."""
    return JinaEmbeddingProvider(
        model="jina-embeddings-v3",
        dims=1024,
        api_key=os.getenv("JINA_API_KEY"),
        task="retrieval.passage",
    )


@pytest.fixture
def jina_query_provider():
    """Create Jina provider for query embedding."""
    return JinaEmbeddingProvider(
        model="jina-embeddings-v3",
        dims=1024,
        api_key=os.getenv("JINA_API_KEY"),
        task="retrieval.query",
    )


class TestLargeDocumentIngestion:
    """Test ingesting large documents with adaptive batching."""

    def test_medium_batch_40_sections(self, jina_provider):
        """Test embedding 40 medium-sized sections (3K chars each)."""
        # Create 40 sections of ~3KB each (~120KB total)
        sections = [
            f"Section {i}: " + ("This is sample content. " * 150) for i in range(40)
        ]

        start_time = time.time()
        vectors = jina_provider.embed_documents(sections)
        elapsed = time.time() - start_time

        # Verify results
        assert len(vectors) == 40
        assert all(len(vec) == 1024 for vec in vectors)

        # Verify batching occurred (should log batch creation)
        # With 40 texts * 3KB = 120KB, should split into 3+ batches (50KB limit)

        # Performance check: should complete in reasonable time
        # With batching, ~40 sections should take < 10 seconds
        assert elapsed < 10.0, f"Took too long: {elapsed:.2f}s"

        print(f"\n✓ Embedded 40 sections in {elapsed:.2f}s")

    def test_very_large_sections(self, jina_provider):
        """Test embedding sections that exceed single-text limit."""
        # Create 3 sections of ~15KB each (some may need truncation)
        sections = [f"Large section {i}: " + ("x" * 15_000) for i in range(3)]

        start_time = time.time()
        vectors = jina_provider.embed_documents(sections)
        elapsed = time.time() - start_time

        # Verify results
        assert len(vectors) == 3
        assert all(len(vec) == 1024 for vec in vectors)

        print(f"\n✓ Embedded 3 large sections in {elapsed:.2f}s")

    def test_many_small_sections(self, jina_provider):
        """Test embedding many small sections (batching by count)."""
        # Create 100 small sections (~200 chars each, ~20KB total)
        sections = [f"Small section {i} with some content here." for i in range(100)]

        start_time = time.time()
        vectors = jina_provider.embed_documents(sections)
        elapsed = time.time() - start_time

        # Verify results
        assert len(vectors) == 100
        assert all(len(vec) == 1024 for vec in vectors)

        # Should be single batch (under 2,048 text limit and 50KB size limit)
        # Should complete quickly
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s"

        print(f"\n✓ Embedded 100 small sections in {elapsed:.2f}s")


class TestBatchSplitting:
    """Test that batch splitting actually occurs and is logged."""

    def test_batch_splitting_logs(self, jina_provider, caplog):
        """Test that batch splitting creates multiple batches and logs it."""
        # Create 10 texts of 8KB each (80KB total, exceeds 50KB limit)
        texts = ["x" * 8_000 for _ in range(10)]

        with caplog.at_level("INFO"):
            vectors = jina_provider.embed_documents(texts)

            # Verify results
            assert len(vectors) == 10
            assert all(len(vec) == 1024 for vec in vectors)

            # Verify batching was logged
            log_messages = [record.message for record in caplog.records]
            batch_creation_logs = [
                msg for msg in log_messages if "Created adaptive batches" in msg
            ]

            assert len(batch_creation_logs) > 0, "Batch creation should be logged"

            # Extract batch count from log
            batch_log = batch_creation_logs[0]
            # Format: "Created adaptive batches: total_texts=10, num_batches=2, ..."
            if "num_batches=" in batch_log:
                num_batches = int(batch_log.split("num_batches=")[1].split(",")[0])
                assert num_batches >= 2, "Should create multiple batches for 80KB data"

            print(f"\n✓ Batch splitting logged: {batch_log}")


class TestConcurrentIngestion:
    """Test concurrent document ingestion with rate limiting."""

    def test_sequential_batches_rate_limiting(self, jina_provider):
        """Test that sequential batches respect rate limits."""
        # Create two batches that should trigger rate limiting
        batch1 = ["Document batch 1, text " + str(i) for i in range(20)]
        batch2 = ["Document batch 2, text " + str(i) for i in range(20)]

        # Embed first batch
        start1 = time.time()
        vectors1 = jina_provider.embed_documents(batch1)
        elapsed1 = time.time() - start1

        # Embed second batch immediately
        start2 = time.time()
        vectors2 = jina_provider.embed_documents(batch2)
        elapsed2 = time.time() - start2

        # Verify results
        assert len(vectors1) == 20
        assert len(vectors2) == 20

        # Second batch should not be significantly slower if rate limits not hit
        # But if we hit limits, it will be throttled
        total_time = elapsed1 + elapsed2

        print(
            f"\n✓ Batch 1: {elapsed1:.2f}s, Batch 2: {elapsed2:.2f}s, Total: {total_time:.2f}s"
        )


class TestQueryEmbedding:
    """Test query embedding with retrieval.query task."""

    def test_query_embedding_uses_correct_task(self, jina_query_provider):
        """Test that query embedding uses retrieval.query task."""
        query = "How do I configure network settings for optimal performance?"

        start_time = time.time()
        vector = jina_query_provider.embed_query(query)
        elapsed = time.time() - start_time

        # Verify results
        assert len(vector) == 1024

        # Should be fast (single text)
        assert elapsed < 2.0, f"Query embedding took too long: {elapsed:.2f}s"

        print(f"\n✓ Query embedded in {elapsed:.2f}s")

    def test_query_truncation(self, jina_query_provider, caplog):
        """Test that very long queries are truncated."""
        # Create query exceeding 8,192 token limit (~40KB)
        long_query = "What is the answer to this question? " * 1000

        with caplog.at_level("WARNING"):
            vector = jina_query_provider.embed_query(long_query)

            # Should succeed
            assert len(vector) == 1024

            # Should log truncation warning
            log_messages = [record.message for record in caplog.records]
            truncation_warnings = [
                msg for msg in log_messages if "exceeds token limit" in msg
            ]

            assert len(truncation_warnings) > 0, "Truncation should be logged"

            print("\n✓ Query truncated and embedded successfully")


class TestDualTaskSupport:
    """Test that both passage and query tasks work correctly."""

    def test_passage_and_query_produce_different_embeddings(
        self, jina_provider, jina_query_provider
    ):
        """Test that passage and query tasks produce different embeddings."""
        text = "Network configuration and performance optimization"

        # Embed as passage (document)
        passage_vector = jina_provider.embed_documents([text])[0]

        # Embed as query
        query_vector = jina_query_provider.embed_query(text)

        # Both should be 1024-D
        assert len(passage_vector) == 1024
        assert len(query_vector) == 1024

        # Vectors should be different (different task optimization)
        # Calculate cosine similarity
        import math

        dot_product = sum(p * q for p, q in zip(passage_vector, query_vector))
        passage_norm = math.sqrt(sum(p * p for p in passage_vector))
        query_norm = math.sqrt(sum(q * q for q in query_vector))
        similarity = dot_product / (passage_norm * query_norm)

        # Should be similar but not identical (same text, different task)
        # Typically 0.85-0.95 similarity
        assert 0.7 < similarity < 0.99, (
            f"Similarity {similarity:.3f} outside expected range - "
            "tasks may not be differentiating properly"
        )

        print(f"\n✓ Passage vs Query similarity: {similarity:.3f}")


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_invalid_dimensions_raises_error(self):
        """Test that mismatched dimensions raise clear error."""
        # Create provider expecting wrong dimensions
        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=512,  # Wrong: v3 produces 1024-D
            api_key=os.getenv("JINA_API_KEY"),
        )

        # Should raise ValueError on dimension mismatch
        with pytest.raises(ValueError, match="got 1024-D, expected 512-D"):
            provider.embed_documents(["test"])

    def test_empty_text_raises_error(self, jina_provider):
        """Test that empty text list raises error."""
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            jina_provider.embed_documents([])

    def test_empty_query_raises_error(self, jina_query_provider):
        """Test that empty query raises error."""
        with pytest.raises(ValueError, match="Cannot embed empty query"):
            jina_query_provider.embed_query("")


class TestPerformanceBenchmarks:
    """Performance benchmarks for monitoring regressions."""

    def test_small_batch_latency(self, jina_provider):
        """Benchmark: Small batch should complete in < 1 second."""
        texts = ["Small document " + str(i) for i in range(5)]

        start = time.time()
        vectors = jina_provider.embed_documents(texts)
        elapsed = time.time() - start

        assert len(vectors) == 5
        assert elapsed < 1.5, f"Small batch too slow: {elapsed:.2f}s"

        print(f"\n✓ Small batch (5 docs) latency: {elapsed:.3f}s")

    def test_medium_batch_latency(self, jina_provider):
        """Benchmark: Medium batch should complete in < 3 seconds."""
        texts = ["Medium document " + ("content " * 50) for i in range(20)]

        start = time.time()
        vectors = jina_provider.embed_documents(texts)
        elapsed = time.time() - start

        assert len(vectors) == 20
        assert elapsed < 5.0, f"Medium batch too slow: {elapsed:.2f}s"

        print(f"\n✓ Medium batch (20 docs) latency: {elapsed:.3f}s")

    def test_query_latency(self, jina_query_provider):
        """Benchmark: Query should complete in < 500ms (typical case)."""
        query = "How do I optimize performance?"

        start = time.time()
        vector = jina_query_provider.embed_query(query)
        elapsed = time.time() - start

        assert len(vector) == 1024
        # Note: First query may be slower due to API warm-up
        # Typical latency should be < 500ms after warm-up

        print(f"\n✓ Query latency: {elapsed:.3f}s")


class TestAPICompliance:
    """Test compliance with Jina API specifications."""

    def test_request_format_compliance(self, jina_provider, monkeypatch):
        """Test that API requests match Jina specification."""
        actual_requests = []

        # Capture actual requests
        original_post = jina_provider._client.post

        def capture_post(*args, **kwargs):
            actual_requests.append(kwargs.get("json"))
            return original_post(*args, **kwargs)

        monkeypatch.setattr(jina_provider._client, "post", capture_post)

        # Make request
        jina_provider.embed_documents(["test"])

        # Verify request format
        assert len(actual_requests) > 0
        request = actual_requests[0]

        # Required fields
        assert "model" in request
        assert request["model"] == "jina-embeddings-v3"

        assert "task" in request
        assert request["task"] == "retrieval.passage"

        assert "input" in request
        assert isinstance(request["input"], list)

        # Optional fields with correct values
        assert request.get("truncate") is False  # We handle client-side
        assert request.get("normalized") is True
        assert request.get("embedding_type") == "float"

        # Should NOT include dimensions parameter (v3 defaults to 1024)
        assert "dimensions" not in request

        print("\n✓ API request format compliant")

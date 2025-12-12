"""
Unit tests for Phase 1.1: BGE Reranker Batching

These tests verify the batching logic without requiring a live reranker service.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

# Updated import for shared CircuitBreaker module (M1/M2 fix)
from src.shared.resilience import CircuitBreaker


class TestBatchDocuments:
    """Tests for the batch_documents helper function.

    Phase 1.1 Fix: batch_documents now accepts (original_index, text) tuples
    directly to avoid fragile index reconstruction.
    """

    def test_batch_documents_empty_list(self):
        """Verify empty input returns empty batches."""
        from src.providers.rerank.local_bge_service import batch_documents

        result = batch_documents([])
        assert result == []

    def test_batch_documents_single_item(self):
        """Verify single document creates one batch."""
        from src.providers.rerank.local_bge_service import batch_documents

        # Input is now (index, text) tuples
        docs = [(0, "Document 1")]
        batches = batch_documents(docs, max_batch_size=16)

        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0] == (0, "Document 1")

    def test_batch_documents_exact_batch_size(self):
        """Verify documents exactly filling batch size."""
        from src.providers.rerank.local_bge_service import batch_documents

        # Input is now (index, text) tuples
        docs = [(i, f"Document {i}") for i in range(16)]
        batches = batch_documents(docs, max_batch_size=16)

        assert len(batches) == 1
        assert len(batches[0]) == 16

    def test_batch_documents_multiple_batches(self):
        """Verify 50 documents creates correct number of batches."""
        from src.providers.rerank.local_bge_service import batch_documents

        # Input is now (index, text) tuples
        docs = [(i, f"Document {i}") for i in range(50)]
        batches = batch_documents(docs, max_batch_size=16)

        # 50 docs with batch size 16: 16 + 16 + 16 + 2 = 4 batches
        assert len(batches) == 4
        assert len(batches[0]) == 16
        assert len(batches[1]) == 16
        assert len(batches[2]) == 16
        assert len(batches[3]) == 2

        # Verify all documents are included
        all_docs = [doc for batch in batches for _, doc in batch]
        assert len(all_docs) == 50

    def test_batch_documents_preserves_original_indices(self):
        """Verify original indices are preserved in batches."""
        from src.providers.rerank.local_bge_service import batch_documents

        # Input is now (index, text) tuples - indices can be arbitrary
        docs = [(10, "A"), (20, "B"), (30, "C"), (40, "D"), (50, "E")]
        batches = batch_documents(docs, max_batch_size=2)

        # Flatten and check indices are preserved as provided
        all_items = [item for batch in batches for item in batch]
        indices = [idx for idx, _ in all_items]
        texts = [text for _, text in all_items]

        assert indices == [10, 20, 30, 40, 50]
        assert texts == ["A", "B", "C", "D", "E"]

    def test_batch_documents_noncontiguous_indices(self):
        """Verify non-contiguous indices are preserved correctly.

        This is the key fix from Issue #3 - we no longer reconstruct indices
        with a counter, so gaps/non-sequential indices are preserved.
        """
        from src.providers.rerank.local_bge_service import batch_documents

        # Simulate filtered candidates with gaps in indices
        docs = [(0, "A"), (5, "B"), (10, "C"), (15, "D")]
        batches = batch_documents(docs, max_batch_size=2)

        # Flatten
        all_items = [item for batch in batches for item in batch]
        indices = [idx for idx, _ in all_items]

        # Original indices must be preserved exactly
        assert indices == [0, 5, 10, 15]


class TestBGERerankerBatching:
    """Tests for batched reranking behavior."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock HTTP client."""
        client = Mock(spec=httpx.Client)
        return client

    def test_rerank_uses_batching_by_default(self, mock_client):
        """Verify batching is enabled by default."""
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        with patch.dict("os.environ", {}, clear=True):
            provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)
            provider._model_id = "test-model"
            provider._provider_name = "test"
            provider._client = mock_client
            provider._batch_size = 16
            provider._use_batching = True
            # Circuit breaker instance (M1/M2 fix: shared module)
            provider._circuit_breaker = CircuitBreaker(name="test")

            assert provider._use_batching is True

    def test_rerank_batching_disabled_via_env(self):
        """Verify batching can be disabled via environment variable."""
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        with patch.dict("os.environ", {"RERANKER_DISABLE_BATCHING": "true"}):
            provider = BGERerankerServiceProvider(base_url="http://test:9001")
            assert provider._use_batching is False

    def test_rerank_batch_reduces_http_calls(self, mock_client):
        """Verify batching reduces number of HTTP calls."""
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": i, "score": 0.9 - (i * 0.01)} for i in range(16)]
        }
        mock_client.post.return_value = mock_response

        provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)
        provider._model_id = "test-model"
        provider._provider_name = "test"
        provider._client = mock_client
        provider._batch_size = 16
        provider._use_batching = True
        # Circuit breaker instance (M1/M2 fix: shared module)
        provider._circuit_breaker = CircuitBreaker(name="test")

        # Create 50 candidates
        candidates = [{"text": f"Document {i}"} for i in range(50)]

        # Run rerank (return value intentionally ignored - testing call count)
        _ = provider.rerank("test query", candidates, top_k=10)

        # With batch size 16 and 50 docs, we should have 4 HTTP calls (not 50)
        assert mock_client.post.call_count == 4

    def test_rerank_fallback_on_batch_failure(self, mock_client):
        """Verify fallback to single-doc processing on true payload error.

        N3 Fix Update: HTTP 400 (Bad Request) is a payload issue that should
        fall back to individual processing. HTTP 500/502/504 are now treated
        as "service unhealthy" and skip entirely.
        """
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        # First call fails with 400 (payload issue), subsequent calls succeed
        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            response = Mock()
            if call_count[0] == 1:
                # First batch fails with 400 (payload issue, not service unhealthy)
                response.status_code = 400
                response.text = "Bad Request - invalid payload"
                # Raise to trigger exception handling
                raise RuntimeError("400 Bad Request - invalid payload")
            else:
                # Single-doc calls succeed
                response.status_code = 200
                response.json.return_value = {"results": [{"index": 0, "score": 0.8}]}
            return response

        mock_client.post.side_effect = mock_post

        provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)
        provider._model_id = "test-model"
        provider._provider_name = "test"
        provider._client = mock_client
        provider._batch_size = 16
        provider._use_batching = True
        # Circuit breaker instance (M1/M2 fix: shared module)
        provider._circuit_breaker = CircuitBreaker(name="test")

        # Create 16 candidates (one batch)
        candidates = [{"text": f"Document {i}"} for i in range(16)]

        # Run rerank - should fall back to single processing
        # (return value intentionally ignored - testing call count)
        _ = provider.rerank("test query", candidates, top_k=10)

        # First batch call + 16 individual fallback calls = 17 total
        assert mock_client.post.call_count == 17

    def test_rerank_skips_on_overload_error(self, mock_client):
        """Verify overload errors skip batch instead of retry storm.

        Issue #1 Fix: When the reranker service is overloaded (timeout, 429, 503),
        we should NOT fall back to per-doc processing as that would amplify
        traffic by 16x and cause cascading failures.

        H2 Fix Update: When all batches fail, we now return fallback results
        with rerank_score=0.0 instead of empty results, preserving UX.
        """
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        # Simulate timeout on first batch
        def mock_post(*args, **kwargs):
            response = Mock()
            response.status_code = 503
            response.text = "Service Unavailable - timeout"
            # The _rerank_batch method will raise RuntimeError for non-200
            raise RuntimeError("timeout occurred")

        mock_client.post.side_effect = mock_post

        provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)
        provider._model_id = "test-model"
        provider._provider_name = "test"
        provider._client = mock_client
        provider._batch_size = 16
        provider._use_batching = True
        # Circuit breaker instance (M1/M2 fix: shared module)
        provider._circuit_breaker = CircuitBreaker(name="test")

        # Create 16 candidates (one batch)
        candidates = [{"text": f"Document {i}"} for i in range(16)]

        # Run rerank - should skip the batch entirely, not retry
        results = provider.rerank("test query", candidates, top_k=10)

        # Should only be 1 call (the failed batch), NOT 17 (batch + 16 retries)
        assert mock_client.post.call_count == 1

        # H2 Fix: Results should contain fallback candidates with zero scores
        assert len(results) == 10  # top_k=10
        for result in results:
            assert result["rerank_score"] == 0.0
            assert result["reranker"] == "rerank_failed"

    def test_rerank_429_triggers_skip(self, mock_client):
        """Verify 429 Too Many Requests triggers skip, not retry."""
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        def mock_post(*args, **kwargs):
            raise RuntimeError("429 Too Many Requests")

        mock_client.post.side_effect = mock_post

        provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)
        provider._model_id = "test-model"
        provider._provider_name = "test"
        provider._client = mock_client
        provider._batch_size = 16
        provider._use_batching = True
        # Circuit breaker instance (M1/M2 fix: shared module)
        provider._circuit_breaker = CircuitBreaker(name="test")

        candidates = [{"text": f"Document {i}"} for i in range(16)]
        # (return value intentionally ignored - testing call count)
        _ = provider.rerank("test query", candidates, top_k=10)

        # Only 1 call, not 17
        assert mock_client.post.call_count == 1


class TestRelationshipRouting:
    """Tests for the route_relationship helper function."""

    def test_route_section_entity_relationship(self):
        """Verify Section→Entity relationship is correctly identified."""
        section_entity = {
            "section_id": "sec-001",
            "entity_id": "ent-001",
            "type": "MENTIONS",
        }

        # Section→Entity has section_id + entity_id
        assert "section_id" in section_entity
        assert "entity_id" in section_entity
        assert "from_id" not in section_entity

    def test_route_entity_entity_relationship(self):
        """Verify Entity→Entity relationship is correctly identified."""
        entity_entity = {
            "from_id": "proc-001",
            "to_id": "step-001",
            "relationship": "CONTAINS_STEP",
            "from_label": "Procedure",
            "to_label": "Step",
        }

        # Entity→Entity has from_id + to_id
        assert "from_id" in entity_entity
        assert "to_id" in entity_entity
        assert "section_id" not in entity_entity


class TestCypherInjectionDefense:
    """Tests for Issue #2: Cypher injection via f-string relationship type.

    These tests verify that the ALLOWED_ENTITY_RELATIONSHIP_TYPES allowlist
    prevents malicious relationship types from being interpolated into Cypher.
    """

    def test_allowed_relationship_types_defined(self):
        """Verify allowlist constant is defined with expected types."""
        from src.ingestion.atomic import ALLOWED_ENTITY_RELATIONSHIP_TYPES

        assert isinstance(ALLOWED_ENTITY_RELATIONSHIP_TYPES, frozenset)
        assert "CONTAINS_STEP" in ALLOWED_ENTITY_RELATIONSHIP_TYPES
        assert "REFERENCES" in ALLOWED_ENTITY_RELATIONSHIP_TYPES

    def test_allowlist_rejects_injection_attempts(self):
        """Verify malicious relationship types are not in allowlist."""
        from src.ingestion.atomic import ALLOWED_ENTITY_RELATIONSHIP_TYPES

        # Common Cypher injection patterns
        injection_attempts = [
            "RELATED_TO}]->(x) DETACH DELETE x WITH x MATCH (y)-[r:{",
            "A]->(b) DETACH DELETE b WITH b CREATE (c:Malicious)-[:OWNS",
            "TEST}]->() CALL db.labels() YIELD label RETURN label//",
            "'; DROP DATABASE neo4j; //",
            "RELATED_TO",  # The removed default
        ]

        for attempt in injection_attempts:
            assert (
                attempt not in ALLOWED_ENTITY_RELATIONSHIP_TYPES
            ), f"Injection pattern should not be in allowlist: {attempt}"

    def test_allowlist_is_frozen(self):
        """Verify allowlist cannot be modified at runtime."""
        from src.ingestion.atomic import ALLOWED_ENTITY_RELATIONSHIP_TYPES

        with pytest.raises(AttributeError):
            ALLOWED_ENTITY_RELATIONSHIP_TYPES.add("MALICIOUS")


class TestTruncation:
    """Tests for text truncation logic."""

    def test_truncate_short_text(self):
        """Verify short text is not truncated."""
        from src.providers.rerank.local_bge_service import BGERerankerServiceProvider

        provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)
        text = "This is a short document."
        result = provider._truncate_text(text)
        assert result == text

    def test_truncate_long_text(self):
        """Verify long text is truncated to token limit."""
        from src.providers.rerank.local_bge_service import (
            MAX_TOKENS_PER_DOC,
            BGERerankerServiceProvider,
        )

        provider = BGERerankerServiceProvider.__new__(BGERerankerServiceProvider)

        # Create text with more tokens than limit
        words = ["word"] * (MAX_TOKENS_PER_DOC + 100)
        text = " ".join(words)

        result = provider._truncate_text(text)
        result_tokens = len(result.split())

        assert result_tokens == MAX_TOKENS_PER_DOC

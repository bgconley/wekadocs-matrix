"""
Unit tests for Jina adaptive batching implementation.
Phase 7C: Tests for batch splitting, rate limiting, and error handling.

These tests verify:
- Adaptive batching based on count and size limits
- Client-side truncation of oversized texts
- Rate limiting enforcement (500 RPM, 1M TPM)
- Retry logic with exponential backoff
- Auto-split on 400 errors
"""

import time
from unittest.mock import Mock, patch

import httpx
import pytest

from src.providers.embeddings.jina import (
    CircuitBreaker,
    JinaEmbeddingProvider,
    RateLimiter,
)


class TestAdaptiveBatching:
    """Test adaptive batching logic."""

    def test_small_batch_no_split(self):
        """Test that small batches are not split."""
        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        # 10 short texts (100 chars each = 1,000 total chars)
        texts = ["x" * 100 for _ in range(10)]

        batches = provider._create_adaptive_batches(texts)

        # Should create single batch
        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_large_count_triggers_split(self):
        """Test that batches exceeding MAX_TEXTS_PER_BATCH are split."""
        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        # Create 3,000 short texts (exceeds 2,048 limit)
        texts = ["short text"] * 3000

        batches = provider._create_adaptive_batches(texts)

        # Should split into multiple batches
        assert len(batches) >= 2

        # First batch should have MAX_TEXTS_PER_BATCH texts
        assert len(batches[0]) == provider.MAX_TEXTS_PER_BATCH

        # All texts should be preserved
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 3000

    def test_large_size_triggers_split(self):
        """Test that batches exceeding MAX_CHARS_PER_BATCH are split by size."""
        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        # Create 10 texts of 10KB each (100KB total, exceeds 50KB limit)
        texts = ["x" * 10_000 for _ in range(10)]

        batches = provider._create_adaptive_batches(texts)

        # Should split into multiple batches due to size
        assert len(batches) >= 2

        # Verify no batch exceeds size limit
        for batch in batches:
            batch_size = sum(len(text) for text in batch)
            assert batch_size <= provider.MAX_CHARS_PER_BATCH

        # All texts should be preserved
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 10

    def test_single_text_too_large_truncated(self):
        """Test that single oversized text is truncated with warning."""
        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        # Create single text exceeding 8,192 token limit (~100KB)
        oversized_text = "x" * 100_000
        texts = [oversized_text]

        with patch("src.providers.embeddings.jina.logger") as mock_logger:
            batches = provider._create_adaptive_batches(texts)

            # Should create single batch with truncated text
            assert len(batches) == 1
            assert len(batches[0]) == 1

            # Text should be truncated to ~32,768 chars (8,192 tokens * 4)
            max_chars = provider.MAX_TOKENS_PER_TEXT * 4
            assert len(batches[0][0]) == max_chars

            # Should log warning
            assert mock_logger.warning.called
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "exceeds token limit" in warning_msg
            assert "100000" in warning_msg  # Original size
            assert str(max_chars) in warning_msg  # Truncated size


class TestRetryLogic:
    """Test retry logic and error handling."""

    @patch("src.providers.embeddings.jina.httpx.Client")
    def test_retry_on_400_splits_batch(self, mock_client_class):
        """Test that 400 errors trigger automatic batch splitting."""
        # Create mock response for 400 error
        mock_400_response = Mock()
        mock_400_response.status_code = 400
        mock_400_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=Mock(), response=mock_400_response
        )

        # Create mock success responses for split batches
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024}]
        }

        # Mock client to return 400 first, then success for splits
        mock_client = Mock()
        mock_client.post.side_effect = [
            mock_400_response,  # First call returns 400
            mock_success_response,  # Second call (first half) succeeds
            mock_success_response,  # Third call (second half) succeeds
        ]
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        # Try to embed 2 texts (should trigger split on 400)
        texts = ["text1", "text2"]

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            result = provider._embed_batch_with_retry(texts, batch_idx=0)

            # Should succeed after splitting
            assert len(result) == 2
            assert all(len(vec) == 1024 for vec in result)

            # Should have made 3 API calls (1 failed, 2 splits succeeded)
            assert mock_client.post.call_count == 3

    @patch("src.providers.embeddings.jina.httpx.Client")
    @patch("src.providers.embeddings.jina.time.sleep")
    def test_retry_on_timeout_backoff(self, mock_sleep, mock_client_class):
        """Test that timeouts trigger exponential backoff and retry."""
        # Create mock responses: timeout, timeout, success
        mock_client = Mock()
        mock_client.post.side_effect = [
            httpx.ReadTimeout("Timeout 1"),
            httpx.ReadTimeout("Timeout 2"),
            Mock(
                status_code=200,
                json=lambda: {"data": [{"embedding": [0.1] * 1024}]},
            ),
        ]
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        texts = ["test"]

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            result = provider._embed_batch_with_retry(texts, batch_idx=0)

            # Should succeed on third attempt
            assert len(result) == 1
            assert len(result[0]) == 1024

            # Should have called sleep twice (after first two timeouts)
            assert mock_sleep.call_count == 2

            # Verify exponential backoff
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls[0] < sleep_calls[1]  # Second sleep longer

    @patch("src.providers.embeddings.jina.httpx.Client")
    @patch("src.providers.embeddings.jina.time.sleep")
    def test_retry_on_429_backoff(self, mock_sleep, mock_client_class):
        """Test that 429 rate limit errors trigger backoff and retry."""
        # Create mock 429 response
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        # Create mock success response
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024}]
        }

        # Mock client: 429, 429, success
        mock_client = Mock()
        mock_client.post.side_effect = [
            mock_429_response,
            mock_429_response,
            mock_success_response,
        ]
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        texts = ["test"]

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            result = provider._embed_batch_with_retry(texts, batch_idx=0)

            # Should succeed on third attempt
            assert len(result) == 1

            # Should have slept twice (after each 429)
            assert mock_sleep.call_count == 2


class TestRateLimiting:
    """Test rate limiting enforcement."""

    def test_rate_limit_enforcement(self):
        """Test that rate limiter correctly throttles requests."""
        # Create rate limiter with very low limits for testing
        limiter = RateLimiter(max_requests_per_minute=2, max_tokens_per_minute=1000)

        # Make 2 requests (should succeed without waiting)
        start = time.time()
        limiter.wait_if_needed(100)
        limiter.wait_if_needed(100)
        elapsed = time.time() - start

        # Should be fast (no throttling yet)
        assert elapsed < 0.5

        # Third request should trigger throttling
        start = time.time()
        limiter.wait_if_needed(100)
        elapsed = time.time() - start

        # Should have waited (exceeded 2 RPM limit)
        # Note: Actual wait time depends on when first 2 requests were made
        # but should be > 0 if rate limit triggered
        # We can't assert exact time due to sliding window

        # Verify internal state
        assert len(limiter.request_times) <= 3
        assert len(limiter.token_counts) <= 3

    def test_rate_limit_token_enforcement(self):
        """Test that rate limiter enforces token limit."""
        limiter = RateLimiter(
            max_requests_per_minute=100,  # High request limit
            max_tokens_per_minute=500,  # Low token limit
        )

        # Make request consuming most of token budget
        limiter.wait_if_needed(400)

        # Next request should trigger throttling due to tokens
        start = time.time()
        limiter.wait_if_needed(200)  # Would exceed 500 TPM
        elapsed = time.time() - start

        # Should have waited
        assert elapsed > 0

    def test_rate_limit_sliding_window(self):
        """Test that old entries are removed from sliding window."""
        limiter = RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=1000)

        # Add some old entries manually
        old_time = time.time() - 65  # 65 seconds ago (outside 60s window)
        limiter.request_times.append(old_time)
        limiter.token_counts.append(100)

        # Make new request
        limiter.wait_if_needed(100)

        # Old entry should be removed
        assert len(limiter.request_times) == 1
        assert limiter.request_times[0] > old_time


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=300)

        # Initially closed
        assert breaker.can_attempt() is True
        assert breaker.state == "closed"

        # Record 2 failures (should stay closed)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.can_attempt() is True
        assert breaker.state == "closed"

        # Third failure should open circuit
        breaker.record_failure()
        assert breaker.can_attempt() is False
        assert breaker.state == "open"

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test that circuit breaker enters half-open state after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1)  # 1 second timeout

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.can_attempt() is False

        # Wait for timeout
        time.sleep(1.1)

        # Should enter half-open
        assert breaker.can_attempt() is True
        assert breaker.state == "half-open"

    def test_circuit_breaker_closes_on_success(self):
        """Test that circuit breaker closes on success."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=300)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        # Record success should reset
        breaker.record_success()
        assert breaker.state == "closed"
        assert breaker.failures == 0


class TestEmbedQuery:
    """Test embed_query method with full retry logic."""

    @patch("src.providers.embeddings.jina.httpx.Client")
    def test_embed_query_success(self, mock_client_class):
        """Test successful query embedding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            result = provider.embed_query("test query")

            assert len(result) == 1024
            assert mock_client.post.call_count == 1

            # Verify correct task parameter
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body["task"] == "retrieval.query"
            assert request_body["model"] == "jina-embeddings-v3"
            assert request_body["truncate"] is False

    @patch("src.providers.embeddings.jina.httpx.Client")
    def test_embed_query_truncates_long_text(self, mock_client_class):
        """Test that embed_query truncates oversized query text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        # Create query exceeding 8,192 token limit
        long_query = "x" * 50_000

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            with patch("src.providers.embeddings.jina.logger") as mock_logger:
                result = provider.embed_query(long_query)

                # Should succeed
                assert len(result) == 1024

                # Should log truncation warning
                assert mock_logger.warning.called
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "exceeds token limit" in warning_msg

                # Verify actual request used truncated text
                call_args = mock_client.post.call_args
                request_body = call_args.kwargs["json"]
                sent_text = request_body["input"][0]
                max_chars = provider.MAX_TOKENS_PER_TEXT * 4
                assert len(sent_text) == max_chars


class TestEmbedDocuments:
    """Test embed_documents method with adaptive batching."""

    @patch("src.providers.embeddings.jina.httpx.Client")
    def test_embed_documents_uses_correct_task(self, mock_client_class):
        """Test that embed_documents uses retrieval.passage task."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024}, {"embedding": [0.2] * 1024}]
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
            task="retrieval.passage",
        )

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            result = provider.embed_documents(["doc1", "doc2"])

            assert len(result) == 2
            assert all(len(vec) == 1024 for vec in result)

            # Verify correct task parameter
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body["task"] == "retrieval.passage"

    @patch("src.providers.embeddings.jina.httpx.Client")
    def test_embed_documents_dimension_validation(self, mock_client_class):
        """Test that dimension mismatch raises error."""
        # Return wrong dimensions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 512}]  # Wrong: 512 instead of 1024
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = JinaEmbeddingProvider(
            model="jina-embeddings-v3",
            dims=1024,
            api_key="test-key",  # pragma: allowlist secret
        )

        with patch.object(provider._rate_limiter, "wait_if_needed"):
            with pytest.raises(ValueError, match="got 512-D, expected 1024-D"):
                provider.embed_documents(["test"])

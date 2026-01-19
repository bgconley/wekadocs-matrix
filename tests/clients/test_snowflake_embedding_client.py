"""
Unit tests for SnowflakeEmbeddingClient retry logic.

Tests the exponential backoff, retryable status codes, and error handling.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import httpx
import pytest

from src.clients.snowflake_embedding_client import SnowflakeEmbeddingClient


class TestSnowflakeEmbeddingClientInit:
    """Test client initialization."""

    def test_init_default_values(self):
        """Client initializes with default values."""
        client = SnowflakeEmbeddingClient()

        assert client.base_url == "http://127.0.0.1:9010/v1"
        assert client.api_key is None
        assert client._max_retries == 4
        assert client._min_backoff == 0.5
        assert client._max_backoff == 8.0

    def test_init_custom_values(self):
        """Client initializes with custom values."""
        client = SnowflakeEmbeddingClient(
            base_url="http://custom:8080/v1/",
            api_key="test-key",  # pragma: allowlist secret
            timeout=30.0,
            max_retries=2,
            min_backoff=1.0,
            max_backoff=4.0,
        )

        assert client.base_url == "http://custom:8080/v1"  # Trailing slash stripped
        assert client.api_key == "test-key"  # pragma: allowlist secret
        assert client._max_retries == 2
        assert client._min_backoff == 1.0
        assert client._max_backoff == 4.0

    def test_context_manager(self):
        """Client works as context manager."""
        with SnowflakeEmbeddingClient() as client:
            assert client is not None
        # After exit, client should be closed (no exception)


class TestHeaders:
    """Test header generation."""

    def test_headers_without_api_key(self):
        """Headers without API key only have Content-Type."""
        client = SnowflakeEmbeddingClient(api_key=None)

        headers = client._headers()

        assert headers == {"Content-Type": "application/json"}

    def test_headers_with_api_key(self):
        """Headers with API key include Authorization."""
        client = SnowflakeEmbeddingClient(api_key="test-key")

        headers = client._headers()

        assert headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key",
        }


class TestBackoffCalculation:
    """Test exponential backoff with jitter."""

    def test_backoff_increases_exponentially(self):
        """Backoff delay increases with each attempt."""
        client = SnowflakeEmbeddingClient(min_backoff=1.0, max_backoff=16.0)

        # Patch sleep to capture delay values
        delays = []
        with patch("time.sleep", side_effect=lambda d: delays.append(d)):
            with patch.object(client, "_client") as mock_http:
                # Simulate retryable failures - must also configure raise_for_status
                mock_response = Mock()
                mock_response.status_code = 503
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "503 Service Unavailable", request=Mock(), response=mock_response
                )
                mock_http.post.return_value = mock_response

                with pytest.raises(httpx.HTTPStatusError):
                    client._post_with_retry("/test", {})

        # Should have 3 backoff sleeps (max_retries - 1)
        assert len(delays) == 3
        # Base delays should roughly double: 1, 2, 4 (plus jitter)
        assert delays[0] >= 1.0 and delays[0] < 2.0  # ~1 + jitter
        assert delays[1] >= 2.0 and delays[1] < 3.0  # ~2 + jitter
        assert delays[2] >= 4.0 and delays[2] < 6.0  # ~4 + jitter

    def test_backoff_respects_max(self):
        """Backoff is capped at max_backoff."""
        client = SnowflakeEmbeddingClient(min_backoff=1.0, max_backoff=2.0)

        delays = []
        with patch("time.sleep", side_effect=lambda d: delays.append(d)):
            with patch.object(client, "_client") as mock_http:
                mock_response = Mock()
                mock_response.status_code = 503
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "503 Service Unavailable", request=Mock(), response=mock_response
                )
                mock_http.post.return_value = mock_response

                with pytest.raises(httpx.HTTPStatusError):
                    client._post_with_retry("/test", {})

        # All delays should be capped at max_backoff + jitter
        for delay in delays:
            assert delay <= 2.5  # max_backoff + max jitter (2.0/4)


class TestRetryLogic:
    """Test retry behavior on various error conditions."""

    def test_retry_on_429(self):
        """Client retries on 429 Too Many Requests."""
        client = SnowflakeEmbeddingClient(max_retries=3, min_backoff=0.01)

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            if call_count < 3:
                response.status_code = 429
            else:
                response.status_code = 200
                response.json.return_value = {"data": []}
                response.raise_for_status = Mock()
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            result = client._post_with_retry("/test", {})

        assert call_count == 3
        assert result == {"data": []}

    def test_retry_on_503(self):
        """Client retries on 503 Service Unavailable."""
        client = SnowflakeEmbeddingClient(max_retries=2, min_backoff=0.01)

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            if call_count < 2:
                response.status_code = 503
            else:
                response.status_code = 200
                response.json.return_value = {"result": "ok"}
                response.raise_for_status = Mock()
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            result = client._post_with_retry("/test", {})

        assert call_count == 2
        assert result == {"result": "ok"}

    def test_no_retry_on_400(self):
        """Client does not retry on 400 Bad Request."""
        client = SnowflakeEmbeddingClient(max_retries=3, min_backoff=0.01)

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            response.status_code = 400
            response.text = "Bad request"
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(),
                response=response,
            )
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            with pytest.raises(RuntimeError, match="rejected request"):
                client._post_with_retry("/test", {})

        # Should only call once - no retry
        assert call_count == 1

    def test_no_retry_on_401(self):
        """Client does not retry on 401 Unauthorized."""
        client = SnowflakeEmbeddingClient(max_retries=3, min_backoff=0.01)

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            response.status_code = 401
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=Mock(),
                response=response,
            )
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            with pytest.raises(RuntimeError, match="authentication failed"):
                client._post_with_retry("/test", {})

        assert call_count == 1

    def test_retry_on_timeout(self):
        """Client retries on timeout."""
        client = SnowflakeEmbeddingClient(max_retries=3, min_backoff=0.01)

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Connection timed out")
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"data": []}
            response.raise_for_status = Mock()
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            result = client._post_with_retry("/test", {})

        assert call_count == 3
        assert result == {"data": []}

    def test_max_retries_exceeded(self):
        """Client raises after exhausting retries."""
        client = SnowflakeEmbeddingClient(max_retries=2, min_backoff=0.01)

        def mock_post(*args, **kwargs):
            response = Mock()
            response.status_code = 503
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(),
                response=response,
            )
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            with pytest.raises(httpx.HTTPStatusError):
                client._post_with_retry("/test", {})


class TestEmbeddingsMethod:
    """Test the embeddings() convenience method."""

    def test_embeddings_builds_correct_payload(self):
        """embeddings() builds correct request payload."""
        client = SnowflakeEmbeddingClient()

        with patch.object(client, "_post_with_retry") as mock_post:
            mock_post.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}

            client.embeddings(
                model="test-model",
                input=["hello"],
                encoding_format="float",
                dimensions=1024,
                user="test-user",
            )

            mock_post.assert_called_once_with(
                "/embeddings",
                {
                    "model": "test-model",
                    "input": ["hello"],
                    "encoding_format": "float",
                    "dimensions": 1024,
                    "user": "test-user",
                },
            )

    def test_embeddings_omits_none_values(self):
        """embeddings() omits None optional parameters."""
        client = SnowflakeEmbeddingClient()

        with patch.object(client, "_post_with_retry") as mock_post:
            mock_post.return_value = {"data": []}

            client.embeddings(model="test-model", input=["hello"])

            call_args = mock_post.call_args
            payload = call_args[0][1]
            assert "dimensions" not in payload
            assert "user" not in payload


class TestListModelsMethod:
    """Test the list_models() method with retry logic."""

    def test_list_models_uses_retry(self):
        """list_models() uses retry logic."""
        client = SnowflakeEmbeddingClient(max_retries=3, min_backoff=0.01)

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            if call_count < 2:
                response.status_code = 503
            else:
                response.status_code = 200
                response.json.return_value = {"models": []}
                response.raise_for_status = Mock()
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.get.side_effect = mock_get

            result = client.list_models()

        assert call_count == 2
        assert result == {"models": []}


class TestRetryableStatusCodes:
    """Test that all retryable status codes are handled."""

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
    def test_retryable_status_codes(self, status_code):
        """All documented retryable status codes trigger retry."""
        client = SnowflakeEmbeddingClient(max_retries=2, min_backoff=0.01)

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            if call_count == 1:
                response.status_code = status_code
            else:
                response.status_code = 200
                response.json.return_value = {"ok": True}
                response.raise_for_status = Mock()
            return response

        with patch.object(client, "_client") as mock_http:
            mock_http.post.side_effect = mock_post

            result = client._post_with_retry("/test", {})

        assert call_count == 2
        assert result == {"ok": True}

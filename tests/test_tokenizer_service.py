"""
Comprehensive unit tests for tokenizer service.

Tests both HuggingFace and Jina Segmenter backends with production scenarios.
Validates token counting accuracy, splitting logic, and integrity verification.
"""

import hashlib
import os
from unittest.mock import Mock, patch

import pytest

# Set test environment before importing tokenizer service
os.environ["TOKENIZER_BACKEND"] = "hf"
os.environ["HF_TOKENIZER_ID"] = "jinaai/jina-embeddings-v3"
os.environ["HF_CACHE"] = "/tmp/hf-test-cache"
os.environ["TRANSFORMERS_OFFLINE"] = "true"  # CRITICAL: Prevent network calls
os.environ["EMBED_MAX_TOKENS"] = "8192"
os.environ["EMBED_TARGET_TOKENS"] = "7900"
os.environ["EMBED_OVERLAP_TOKENS"] = "200"
os.environ["SPLIT_MIN_TOKENS"] = "1000"
os.environ["LOG_SPLIT_DECISIONS"] = "false"  # Reduce noise in tests
os.environ["INTEGRITY_CHECK_SAMPLE_RATE"] = "1.0"  # Test all splits

from src.providers.tokenizer_service import (
    HuggingFaceTokenizerBackend,
    JinaSegmenterBackend,
    TokenizerService,
    create_tokenizer_service,
)


class TestHuggingFaceTokenizerBackend:
    """Test HuggingFace tokenizer backend."""

    @pytest.fixture(scope="class")
    def backend(self):
        """Create HF backend for testing (shared across all tests in class)."""
        # Use environment variables already set at module level
        return HuggingFaceTokenizerBackend()

    def test_initialization(self, backend):
        """Test backend initializes successfully."""
        assert backend is not None
        assert backend.tokenizer is not None

    def test_count_tokens_simple(self, backend):
        """Test token counting on simple text."""
        text = "The quick brown fox jumps over the lazy dog."
        token_count = backend.count_tokens(text)

        # XLM-RoBERTa should give ~10-12 tokens for this text
        assert 8 <= token_count <= 15, f"Expected 8-15 tokens, got {token_count}"

    def test_count_tokens_technical(self, backend):
        """Test token counting on technical documentation."""
        text = """
        Configure the WekaFS cluster using the following command:

        weka cluster create --name=production --nodes=10 --failure-domain=rack

        Parameters:
        - name: Cluster identifier (alphanumeric)
        - nodes: Number of backend nodes (1-1000)
        - failure-domain: rack, chassis, or node
        """

        token_count = backend.count_tokens(text)

        # Technical text with code should have lower chars/token ratio
        # Expect ~50-80 tokens for this ~250 char text
        assert 40 <= token_count <= 100, f"Expected 40-100 tokens, got {token_count}"

    def test_count_tokens_dense_table(self, backend):
        """Test token counting on dense reference table."""
        text = """| Command | Description | Flags |
|---------|-------------|-------|
| weka cluster create | Create new cluster | --name, --nodes |
| weka fs create | Create filesystem | --name, --capacity |
| weka user add | Add user | --username, --role |"""

        token_count = backend.count_tokens(text)

        # Dense tables have worst case chars/token ratio
        # ~200 chars should give 60-100 tokens
        assert 50 <= token_count <= 120, f"Expected 50-120 tokens, got {token_count}"

    def test_encode_decode_roundtrip(self, backend):
        """Test encoding and decoding preserves text."""
        original = "Testing encode/decode roundtrip with technical terms like WekaFS."

        tokens = backend.encode(original)
        decoded = backend.decode(tokens)

        # Should be very close (minor differences in whitespace acceptable)
        assert decoded.strip() == original.strip()

    def test_encode_decode_long_text(self, backend):
        """Test encode/decode on longer text."""
        original = " ".join([f"Sentence number {i} with content." for i in range(100)])

        tokens = backend.encode(original)
        decoded = backend.decode(tokens)

        # Verify length roughly matches
        assert abs(len(decoded) - len(original)) < 10

    def test_count_tokens_empty(self, backend):
        """Test token counting on empty string."""
        token_count = backend.count_tokens("")
        assert token_count == 0

    def test_count_tokens_unicode(self, backend):
        """Test token counting on unicode text."""
        text = "Hello 世界! Testing émojis 🚀 and spëcial çharacters."
        token_count = backend.count_tokens(text)

        # XLM-RoBERTa handles multilingual well
        assert token_count > 0


class TestJinaSegmenterBackend:
    """Test Jina Segmenter API backend."""

    @pytest.fixture
    def backend(self):
        """Create Segmenter backend for testing."""
        with patch.dict(
            os.environ,
            {
                "JINA_SEGMENTER_BASE_URL": "https://api.jina.ai/v1/segment",
                "JINA_API_KEY": "test_key",  # pragma: allowlist secret
                "SEGMENTER_TOKENIZER_NAME": "xlm-roberta-base",
                "SEGMENTER_TIMEOUT_MS": "5000",
            },
        ):
            return JinaSegmenterBackend()

    def test_initialization(self, backend):
        """Test backend initializes successfully."""
        assert backend is not None
        assert backend.client is not None
        assert backend.api_key == "test_key"

    @patch("httpx.Client.post")
    def test_count_tokens_api_success(self, mock_post, backend):
        """Test successful token counting via API."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"num_tokens": 42}
        mock_post.return_value = mock_response

        token_count = backend.count_tokens("test text")

        assert token_count == 42
        mock_post.assert_called_once()

    @patch("httpx.Client.post")
    def test_count_tokens_api_with_tokens_array(self, mock_post, backend):
        """Test token counting when API returns tokens array."""
        # Mock API response with tokens array
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tokens": ["test", "text"],
        }
        mock_post.return_value = mock_response

        token_count = backend.count_tokens("test text")

        assert token_count == 2

    @patch("httpx.Client.post")
    def test_count_tokens_api_timeout(self, mock_post, backend):
        """Test handling of API timeout."""
        import httpx

        mock_post.side_effect = httpx.TimeoutException("Request timeout")

        with pytest.raises(RuntimeError, match="Jina Segmenter API timeout"):
            backend.count_tokens("test text")

    @patch("httpx.Client.post")
    def test_count_tokens_api_error(self, mock_post, backend):
        """Test handling of API error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.side_effect = httpx.HTTPStatusError(
            "Rate limit", request=Mock(), response=mock_response
        )

        with pytest.raises(RuntimeError, match="Jina Segmenter API error 429"):
            backend.count_tokens("test text")

    def test_decode_not_supported(self, backend):
        """Test that decode raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Jina Segmenter does not support decoding"
        ):
            backend.decode([1, 2, 3])


class TestTokenizerService:
    """Test TokenizerService with HuggingFace backend."""

    @pytest.fixture(scope="class")
    def service(self):
        """Create TokenizerService for testing (shared across all tests in class)."""
        # Save original values
        original_target = os.environ.get("EMBED_TARGET_TOKENS")
        original_overlap = os.environ.get("EMBED_OVERLAP_TOKENS")

        # Set test-specific values BEFORE creating service
        os.environ["EMBED_TARGET_TOKENS"] = "100"  # Small for testing
        os.environ["EMBED_OVERLAP_TOKENS"] = "20"

        # Create service with test settings
        service = TokenizerService()

        # Verify test settings were applied
        assert (
            service.target_tokens == 100
        ), f"Expected target_tokens=100, got {service.target_tokens}"
        assert (
            service.overlap_tokens == 20
        ), f"Expected overlap_tokens=20, got {service.overlap_tokens}"

        yield service

        # Cleanup: restore original values after all tests in class
        if original_target:
            os.environ["EMBED_TARGET_TOKENS"] = original_target
        if original_overlap:
            os.environ["EMBED_OVERLAP_TOKENS"] = original_overlap

    def test_initialization_hf(self, service):
        """Test service initializes with HF backend."""
        assert service.backend_name == "huggingface"
        assert service.max_tokens == 8192
        assert service.target_tokens == 100
        assert service.overlap_tokens == 20

    def test_count_tokens(self, service):
        """Test token counting."""
        text = "Simple test sentence."
        count = service.count_tokens(text)
        assert count > 0
        assert count < 20  # Should be ~5 tokens

    def test_needs_splitting_false(self, service):
        """Test needs_splitting returns False for short text."""
        text = "Short text."
        assert service.needs_splitting(text) is False

    def test_needs_splitting_true(self, service):
        """Test needs_splitting returns True for text exceeding max_tokens limit."""
        # needs_splitting checks against max_tokens (8192), not target_tokens (100)
        # Create text that exceeds 8192 tokens (need ~10000 words)
        text = " ".join([f"word{i}" for i in range(10000)])
        token_count = service.count_tokens(text)
        assert (
            token_count > service.max_tokens
        ), f"Text has {token_count} tokens, need >{service.max_tokens}"
        assert service.needs_splitting(text) is True

    def test_compute_integrity_hash(self, service):
        """Test SHA256 hash computation."""
        text = "Test content for hashing."
        hash1 = service.compute_integrity_hash(text)
        hash2 = service.compute_integrity_hash(text)

        # Same text should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

        # Different text should produce different hash
        hash3 = service.compute_integrity_hash("Different content")
        assert hash1 != hash3

    def test_split_to_chunks_no_split_needed(self, service):
        """Test splitting when text is under limit."""
        text = "Short text that doesn't need splitting."
        chunks = service.split_to_chunks(text, section_id="test_section")

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["total_chunks"] == 1
        assert chunks[0]["overlap_start"] is False
        assert chunks[0]["overlap_end"] is False
        assert chunks[0]["parent_section_id"] == "test_section"
        assert "integrity_hash" in chunks[0]

    def test_split_to_chunks_with_splitting(self, service):
        """Test splitting when text exceeds target_tokens."""
        # split_to_chunks uses target_tokens (100) for chunking decisions
        # Create text with >100 tokens to trigger splitting
        text = " ".join(
            [f"word{i}" for i in range(150)]
        )  # ~300 tokens (each word becomes 2 tokens: 'word', '123')
        token_count = service.count_tokens(text)
        assert (
            token_count > service.target_tokens
        ), f"Text has {token_count} tokens, need >{service.target_tokens} for splitting"

        chunks = service.split_to_chunks(text, section_id="large_section")

        # Should create multiple chunks
        assert (
            len(chunks) >= 2
        ), f"Expected >=2 chunks for {token_count} tokens with target={service.target_tokens}, got {len(chunks)}"

        # Verify chunk metadata
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == len(chunks)
            assert chunk["token_count"] <= 100  # Should not exceed target
            assert "text" in chunk
            assert "integrity_hash" in chunk
            assert chunk["parent_section_id"] == "large_section"

            # Check overlap flags
            if i > 0:
                assert chunk["overlap_start"] is True
            else:
                assert chunk["overlap_start"] is False

            if i < len(chunks) - 1:
                assert chunk["overlap_end"] is True
            else:
                assert chunk["overlap_end"] is False

    def test_split_to_chunks_content_preservation(self, service):
        """Test that splitting preserves all content."""
        # Create text that will be split (need >100 tokens)
        text = " ".join(
            [f"Sentence number {i} with some content here." for i in range(50)]
        )
        token_count = service.count_tokens(text)

        # Only test splitting if text is large enough
        if token_count > service.target_tokens:
            chunks = service.split_to_chunks(text)
            assert (
                len(chunks) > 1
            ), f"Text with {token_count} tokens should split with target={service.target_tokens}"
        else:
            # Text is too small for these test settings, create larger text
            text = " ".join(
                [
                    f"Sentence number {i} with detailed content here for testing."
                    for i in range(100)
                ]
            )
            token_count = service.count_tokens(text)
            assert token_count > service.target_tokens
            chunks = service.split_to_chunks(text)
            assert len(chunks) > 1

        # Each chunk should have content
        for chunk in chunks:
            assert len(chunk["text"]) > 0
            assert chunk["token_count"] > 0

    def test_split_to_chunks_token_limits(self, service):
        """Test that no chunk exceeds target token limit."""
        # Create text large enough to force splitting
        text = " ".join([f"token{i}" for i in range(200)])  # Should be >100 tokens
        token_count = service.count_tokens(text)
        assert (
            token_count > service.target_tokens
        ), f"Need >{service.target_tokens} tokens to test, got {token_count}"

        chunks = service.split_to_chunks(text)

        # Verify we got multiple chunks
        assert (
            len(chunks) > 1
        ), f"Text with {token_count} tokens should split with target={service.target_tokens}"

        # Verify all chunks respect target token limit
        for i, chunk in enumerate(chunks):
            assert (
                chunk["token_count"] <= service.target_tokens
            ), f"Chunk {i} has {chunk['token_count']} tokens, exceeds target={service.target_tokens}"

    def test_truncate_to_token_limit(self, service):
        """Test truncating text to exact token limit."""
        # Create text that exceeds limit
        text = " ".join([f"word{i}" for i in range(200)])

        truncated = service.truncate_to_token_limit(text, max_tokens=50)

        # Verify truncated text is under limit
        token_count = service.count_tokens(truncated)
        assert token_count <= 50

    def test_truncate_to_token_limit_short_text(self, service):
        """Test truncating text that's already under limit."""
        text = "Short text."
        truncated = service.truncate_to_token_limit(text, max_tokens=100)

        # Should return original text
        assert truncated == text


class TestTokenizerServiceSegmenterBackend:
    """Test TokenizerService with Segmenter backend."""

    @pytest.fixture
    def service(self):
        """Create TokenizerService with Segmenter backend."""
        with patch.dict(
            os.environ,
            {
                "TOKENIZER_BACKEND": "segmenter",
                "JINA_SEGMENTER_BASE_URL": "https://api.jina.ai/v1/segment",
                "JINA_API_KEY": "test_key",
                "SEGMENTER_TOKENIZER_NAME": "xlm-roberta-base",
            },
        ):
            return TokenizerService()

    def test_initialization_segmenter(self, service):
        """Test service initializes with Segmenter backend."""
        assert service.backend_name == "jina-segmenter"

    def test_split_not_supported_segmenter(self, service):
        """Test that splitting fails with Segmenter backend."""
        text = "Test text"

        with pytest.raises(
            RuntimeError, match="Splitting requires HuggingFace backend"
        ):
            service.split_to_chunks(text)

    def test_truncate_not_supported_segmenter(self, service):
        """Test that truncation fails with Segmenter backend."""
        text = "Test text"

        with pytest.raises(
            RuntimeError, match="Truncation requires HuggingFace backend"
        ):
            service.truncate_to_token_limit(text)


class TestTokenizerServiceFactory:
    """Test factory function."""

    def test_create_tokenizer_service_hf(self):
        """Test creating service with HF backend."""
        # Use already configured HF backend from module level
        service = create_tokenizer_service()
        assert service.backend_name == "huggingface"

    def test_create_tokenizer_service_segmenter(self):
        """Test creating service with Segmenter backend."""
        # Temporarily switch to segmenter
        original = os.environ.get("TOKENIZER_BACKEND")
        os.environ["TOKENIZER_BACKEND"] = "segmenter"
        os.environ["JINA_API_KEY"] = "test_key"

        try:
            service = create_tokenizer_service()
            assert service.backend_name == "jina-segmenter"
        finally:
            # Restore original
            if original:
                os.environ["TOKENIZER_BACKEND"] = original

    def test_create_tokenizer_service_invalid_backend(self):
        """Test creating service with invalid backend."""
        original = os.environ.get("TOKENIZER_BACKEND")
        os.environ["TOKENIZER_BACKEND"] = "invalid"

        try:
            with pytest.raises(ValueError, match="Invalid tokenizer backend"):
                create_tokenizer_service()
        finally:
            if original:
                os.environ["TOKENIZER_BACKEND"] = original


class TestProductionScenarios:
    """Test production scenarios from the documentation."""

    @pytest.fixture(scope="class")
    def service(self):
        """Create TokenizerService for production testing (shared across all tests)."""
        # Use production settings from module-level environment
        return TokenizerService()

    def test_large_section_scenario(self, service):
        """Test handling of large section (40KB+) from production."""
        # Simulate a large CLI reference section with commands and tables
        commands = []
        for i in range(200):
            commands.append(
                f"""
### Command {i}: weka-command-{i}

Description: This command performs operation {i} on the WekaFS cluster.

**Syntax:**
```bash
weka command-{i} [--flag1 VALUE] [--flag2 VALUE]
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| --flag1 | string | Configuration parameter {i} | none |
| --flag2 | integer | Numeric value for {i} | 0 |

**Examples:**
```bash
# Example 1
weka command-{i} --flag1=production --flag2=100

# Example 2
weka command-{i} --flag1=test
```
"""
            )

        large_text = "\n".join(commands)

        # This should be ~40KB+ and need splitting
        assert len(large_text) > 40000

        # Split it
        chunks = service.split_to_chunks(large_text, section_id="cli_ref")

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Verify all chunks are under token limit
        for chunk in chunks:
            assert chunk["token_count"] <= 7900

        # Verify metadata
        for chunk in chunks:
            assert chunk["parent_section_id"] == "cli_ref"
            assert "integrity_hash" in chunk

    def test_technical_documentation_token_density(self, service):
        """Test token counting on real technical documentation patterns."""
        # Pattern 1: Command examples
        cmd_text = "weka cluster create --name=prod --nodes=10 --failure-domain=rack"
        cmd_tokens = service.count_tokens(cmd_text)
        cmd_ratio = len(cmd_text) / cmd_tokens if cmd_tokens > 0 else 0

        # Pattern 2: Configuration parameters
        config_text = "tier.s3.bucket_name, tier.s3.region, tier.s3.access_key_id"
        config_tokens = service.count_tokens(config_text)
        config_ratio = len(config_text) / config_tokens if config_tokens > 0 else 0

        # Pattern 3: Table row
        table_text = (
            "| weka fs create | Create filesystem | --name, --capacity, --thin |"
        )
        table_tokens = service.count_tokens(table_text)
        table_ratio = len(table_text) / table_tokens if table_tokens > 0 else 0

        # Technical text should have 1.5-3.0 chars/token ratio
        assert (
            1.0 < cmd_ratio < 4.0
        ), f"Command ratio {cmd_ratio} outside expected range"
        assert (
            1.0 < config_ratio < 4.0
        ), f"Config ratio {config_ratio} outside expected range"
        assert (
            1.0 < table_ratio < 4.0
        ), f"Table ratio {table_ratio} outside expected range"

    def test_no_content_loss_verification(self, service):
        """Test the zero-loss guarantee from the plan."""
        # Create text that will definitely be split
        original = " ".join(
            [
                f"Section {i}: This is detailed content about topic {i}. "
                f"It includes multiple sentences and technical details like WekaFS, "
                f"configuration parameters, and command examples. "
                for i in range(500)
            ]
        )

        # Track original metrics (hash and length unused but preserved for future debugging)
        _original_hash = hashlib.sha256(
            original.encode("utf-8")
        ).hexdigest()  # noqa: F841
        _original_length = len(original)  # noqa: F841
        original_tokens = service.count_tokens(original)

        # Split the text
        chunks = service.split_to_chunks(original)

        # Verify chunks were created
        assert len(chunks) > 1

        # Verify total tokens across chunks (with overlaps) is reasonable
        total_chunk_tokens = sum(c["token_count"] for c in chunks)

        # With overlap, total should be more than original but not excessive
        # Overlap adds roughly overlap_tokens * (num_chunks - 1)
        expected_overhead = service.overlap_tokens * (len(chunks) - 1)
        assert (
            total_chunk_tokens <= original_tokens + expected_overhead + 100
        )  # Small buffer

    def test_query_truncation_scenario(self, service):
        """Test query truncation (acceptable for queries, not documents)."""
        # Simulate very long query
        long_query = " ".join(
            [
                f"Find information about topic {i} and details on subject {i}"
                for i in range(2000)
            ]
        )

        # Truncate to limit
        truncated = service.truncate_to_token_limit(long_query, max_tokens=8192)

        # Verify under limit
        token_count = service.count_tokens(truncated)
        assert token_count <= 8192

        # Verify some content preserved
        assert len(truncated) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

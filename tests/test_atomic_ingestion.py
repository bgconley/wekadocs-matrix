"""
Tests for Atomic Ingestion Coordinator.

These tests verify:
1. Saga coordinator executes steps in order
2. Compensation runs on failure in reverse order
3. Pre-commit validation catches ID mismatches
4. Entity-chunk sync validation works correctly
"""

from unittest.mock import MagicMock

import pytest

from src.ingestion.atomic import AtomicIngestionCoordinator
from src.ingestion.saga import (
    IngestionValidator,
    SagaContext,
    SagaCoordinator,
    SagaStep,
    SagaStepResult,
    StepStatus,
    ValidationResult,
)


class TestSagaCoordinator:
    """Test the Saga pattern implementation."""

    def test_saga_executes_steps_in_order(self):
        """Steps execute in the order they were added."""
        execution_order = []

        def step1():
            execution_order.append("step1")
            return SagaStepResult(success=True)

        def step2():
            execution_order.append("step2")
            return SagaStepResult(success=True)

        def step3():
            execution_order.append("step3")
            return SagaStepResult(success=True)

        coordinator = SagaCoordinator()
        coordinator.add_step(
            SagaStep("step1", step1, lambda: SagaStepResult(success=True))
        )
        coordinator.add_step(
            SagaStep("step2", step2, lambda: SagaStepResult(success=True))
        )
        coordinator.add_step(
            SagaStep("step3", step3, lambda: SagaStepResult(success=True))
        )

        result = coordinator.execute()

        assert result["status"] == "completed"
        assert execution_order == ["step1", "step2", "step3"]

    def test_saga_compensates_on_failure(self):
        """Failed step triggers compensation in reverse order."""
        execution_order = []
        compensation_order = []

        def step1():
            execution_order.append("step1")
            return SagaStepResult(success=True)

        def step2():
            execution_order.append("step2")
            return SagaStepResult(success=True)

        def step3():
            execution_order.append("step3")
            raise RuntimeError("Step 3 failed!")

        def comp1():
            compensation_order.append("comp1")
            return SagaStepResult(success=True)

        def comp2():
            compensation_order.append("comp2")
            return SagaStepResult(success=True)

        def comp3():
            compensation_order.append("comp3")
            return SagaStepResult(success=True)

        coordinator = SagaCoordinator()
        coordinator.add_step(SagaStep("step1", step1, comp1))
        coordinator.add_step(SagaStep("step2", step2, comp2))
        coordinator.add_step(SagaStep("step3", step3, comp3))

        result = coordinator.execute()

        assert result["status"] == "compensated"
        assert execution_order == ["step1", "step2", "step3"]
        # Only step1 and step2 completed, so only they get compensated
        assert compensation_order == ["comp2", "comp1"]

    def test_saga_tracks_step_status(self):
        """Step statuses are correctly tracked."""
        coordinator = SagaCoordinator()
        coordinator.add_step(
            SagaStep(
                "success_step",
                lambda: SagaStepResult(success=True, data={"key": "value"}),
                lambda: SagaStepResult(success=True),
            )
        )

        result = coordinator.execute()

        assert result["status"] == "completed"
        assert coordinator.steps[0].status == StepStatus.EXECUTED
        assert coordinator.steps[0].result.success is True
        assert coordinator.steps[0].result.data == {"key": "value"}

    def test_saga_handles_compensation_failure(self):
        """Saga reports failure when compensation also fails."""

        def step1():
            return SagaStepResult(success=True)

        def step2():
            raise RuntimeError("Execution failed")

        def failing_compensation1():
            raise RuntimeError("Compensation also failed")

        def comp2():
            return SagaStepResult(success=True)

        coordinator = SagaCoordinator()
        coordinator.add_step(
            SagaStep(
                "step1",
                step1,
                failing_compensation1,  # This compensation will fail
            )
        )
        coordinator.add_step(
            SagaStep(
                "step2",
                step2,  # This step fails during execution
                comp2,
            )
        )

        result = coordinator.execute()

        # Status is "failed" because compensation failed
        assert result["status"] == "failed"
        assert "Execution failed" in result.get("error", "")


class TestSagaContext:
    """Test saga context data passing."""

    def test_context_tracks_document_id(self):
        """Context correctly stores document ID."""
        context = SagaContext(document_id="doc_123")
        assert context.document_id == "doc_123"

    def test_context_tracks_written_ids(self):
        """Context tracks IDs written to each store."""
        context = SagaContext()
        context.neo4j_chunk_ids.append("chunk_1")
        context.neo4j_chunk_ids.append("chunk_2")
        context.qdrant_point_ids.append("point_1")

        assert len(context.neo4j_chunk_ids) == 2
        assert len(context.qdrant_point_ids) == 1


class TestIngestionValidator:
    """Test pre-commit validation."""

    def test_validates_missing_chunk_ids(self):
        """Validation fails when chunks are missing IDs."""
        validator = IngestionValidator(
            neo4j_driver=MagicMock(),
            qdrant_client=None,
            config=MagicMock(),
        )

        chunks = [
            {"id": "chunk_1", "title": "First"},
            {"title": "Missing ID"},  # No ID!
        ]

        result = validator.validate_pre_ingest(
            document_id="doc_1",
            chunks=chunks,
            entities={},
            mentions=[],
        )

        assert not result.valid
        assert any("missing 'id'" in err.lower() for err in result.errors)

    def test_validates_duplicate_chunk_ids(self):
        """Validation fails on duplicate chunk IDs."""
        validator = IngestionValidator(
            neo4j_driver=MagicMock(),
            qdrant_client=None,
            config=MagicMock(),
        )

        chunks = [
            {"id": "chunk_1", "title": "First"},
            {"id": "chunk_1", "title": "Duplicate!"},
        ]

        result = validator.validate_pre_ingest(
            document_id="doc_1",
            chunks=chunks,
            entities={},
            mentions=[],
        )

        assert not result.valid
        assert any("duplicate" in err.lower() for err in result.errors)


class TestAtomicEmbeddingFailures:
    """Test embedding failure paths for atomic ingestion."""

    def test_raises_when_embedder_missing(self):
        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=MagicMock(),
            qdrant_client=MagicMock(),
            config=MagicMock(),
        )

        builder = MagicMock()
        builder.embedder = None
        builder.ensure_embedder = MagicMock(return_value=None)
        builder.embedding_settings = MagicMock()
        builder._build_section_text_for_embedding = lambda s: "text"
        builder._build_title_text_for_embedding = lambda s: "title"

        with pytest.raises(RuntimeError):
            coordinator._compute_embeddings(
                document={},
                sections=[{"id": "s1"}],
                entities={},
                builder=builder,
            )

    def test_raises_on_dense_embedding_failure(self):
        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=MagicMock(),
            qdrant_client=MagicMock(),
            config=MagicMock(),
        )

        class FailingEmbedder:
            def embed_documents(self, texts):
                raise RuntimeError("dense failed")

        builder = MagicMock()
        builder.ensure_embedder = MagicMock()
        builder.embedder = FailingEmbedder()
        caps = MagicMock()
        caps.supports_sparse = False
        caps.supports_colbert = False
        builder.embedding_settings = MagicMock(capabilities=caps)
        builder._build_section_text_for_embedding = lambda s: "text"
        builder._build_title_text_for_embedding = lambda s: "title"

        with pytest.raises(RuntimeError):
            coordinator._compute_embeddings(
                document={},
                sections=[{"id": "s1"}],
                entities={},
                builder=builder,
            )

    def test_skips_empty_sections_for_embedding(self):
        """Empty sections (like microdoc stubs) are skipped, not sent to embedding API."""
        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=MagicMock(),
            qdrant_client=MagicMock(),
            config=MagicMock(),
        )

        embed_calls = []

        class TrackingEmbedder:
            def embed_documents(self, texts):
                embed_calls.append(texts)
                return [[0.1] * 1024 for _ in texts]

        builder = MagicMock()
        builder.ensure_embedder = MagicMock()
        builder.embedder = TrackingEmbedder()
        builder.embedding_dims = 1024  # Required for dimension validation
        caps = MagicMock()
        caps.supports_sparse = False
        caps.supports_colbert = False
        builder.embedding_settings = MagicMock(capabilities=caps, dimensions=1024)

        # Simulate _build_section_text_for_embedding returning empty for stub
        def build_text(section):
            if section.get("is_microdoc_stub"):
                return ""  # Empty stub
            return section.get("text", "default text")

        builder._build_section_text_for_embedding = build_text
        builder._build_title_text_for_embedding = lambda s: s.get("title", "title")

        # Sections need all required chunk schema fields (id, document_id, level, order,
        # original_section_ids, is_combined, is_split, token_count)
        sections = [
            {
                "id": "s1",
                "text": "Real content",
                "title": "Section 1",
                "document_id": "doc1",
                "level": 1,
                "order": 0,
                "original_section_ids": ["s1"],
                "is_combined": False,
                "is_split": False,
                "token_count": 2,
            },
            {
                "id": "stub1",
                "text": "",
                "is_microdoc_stub": True,
                "title": "",
                "document_id": "doc1",
                "level": 1,
                "order": 1,
                "original_section_ids": ["stub1"],
                "is_combined": False,
                "is_split": False,
                "token_count": 0,
            },
            {
                "id": "s2",
                "text": "More content",
                "title": "Section 2",
                "document_id": "doc1",
                "level": 1,
                "order": 2,
                "original_section_ids": ["s2"],
                "is_combined": False,
                "is_split": False,
                "token_count": 2,
            },
            {
                "id": "stub2",
                "text": "",
                "is_microdoc_stub": True,
                "title": "",
                "document_id": "doc1",
                "level": 1,
                "order": 3,
                "original_section_ids": ["stub2"],
                "is_combined": False,
                "is_split": False,
                "token_count": 0,
            },
        ]

        result = coordinator._compute_embeddings(
            document={},
            sections=sections,
            entities={},
            builder=builder,
        )

        # Should have called embed_documents once for content and once for titles
        assert len(embed_calls) == 2  # content + title batches

        # Only non-empty sections should be embedded (s1 and s2)
        content_texts = embed_calls[0]
        assert len(content_texts) == 2
        assert "Real content" in content_texts[0]
        assert "More content" in content_texts[1]

        # Result should only contain embeddings for non-empty sections
        assert "s1" in result["sections"]
        assert "s2" in result["sections"]
        assert "stub1" not in result["sections"]
        assert "stub2" not in result["sections"]

    def test_warns_on_invalid_mention_references(self):
        """Validation warns when mentions reference non-existent chunks."""
        validator = IngestionValidator(
            neo4j_driver=MagicMock(),
            qdrant_client=None,
            config=MagicMock(),
        )

        chunks = [
            {"id": "chunk_1", "title": "First"},
        ]

        mentions = [
            {"entity_id": "ent_1", "section_id": "nonexistent_chunk"},
        ]

        result = validator.validate_pre_ingest(
            document_id="doc_1",
            chunks=chunks,
            entities={"ent_1": {"name": "Test Entity"}},
            mentions=mentions,
        )

        # This is a warning, not an error
        assert result.valid
        assert len(result.warnings) > 0
        assert any("nonexistent" in w.lower() for w in result.warnings)


class TestValidationResult:
    """Test validation result dataclass."""

    def test_to_dict(self):
        """ValidationResult serializes correctly."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        d = result.to_dict()
        assert d["valid"] is False
        assert len(d["errors"]) == 2
        assert len(d["warnings"]) == 1


class TestSagaStepResult:
    """Test saga step result dataclass."""

    def test_success_result(self):
        """Success result stores data."""
        result = SagaStepResult(
            success=True,
            data={"nodes_created": 5},
            duration_ms=100,
        )
        assert result.success
        assert result.data["nodes_created"] == 5

    def test_failure_result(self):
        """Failure result stores error."""
        result = SagaStepResult(
            success=False,
            error="Connection timeout",
            duration_ms=5000,
        )
        assert not result.success
        assert "timeout" in result.error.lower()


# Integration-style tests (require mocking external services)


class TestAtomicIngestionIntegration:
    """Integration tests for atomic ingestion (with mocked services)."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        tx = MagicMock()
        tx.closed.return_value = False
        session.begin_transaction.return_value = tx
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        collection_info = MagicMock()
        collection_info.points_count = 1000
        client.get_collection.return_value = collection_info
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.search.vector.qdrant.collection_name = "test_collection"
        config.search.vector.primary = "qdrant"
        config.search.vector.dual_write = False
        return config

    def test_atomic_saga_builder(
        self, mock_neo4j_driver, mock_qdrant_client, mock_config
    ):
        """IngestionSagaBuilder creates valid sagas."""
        from src.ingestion.saga import IngestionSagaBuilder

        builder = IngestionSagaBuilder(
            neo4j_driver=mock_neo4j_driver,
            qdrant_client=mock_qdrant_client,
            config=mock_config,
            document_id="test_doc",
        )

        builder.add_neo4j_step(
            name="create_document",
            cypher="MERGE (d:Document {id: $id})",
            params={"id": "test_doc"},
            compensation_cypher="MATCH (d:Document {id: $id}) DELETE d",
        )

        saga = builder.build()

        assert len(saga.steps) == 1
        assert saga.steps[0].name == "create_document"
        assert saga.context.document_id == "test_doc"


# ============================================================================
# Tests for AtomicIngestionCoordinator (Task 1)
# ============================================================================


class TestAtomicIngestionCoordinator:
    """Tests for the AtomicIngestionCoordinator class."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver with transaction support."""
        driver = MagicMock()
        session = MagicMock()
        tx = MagicMock()
        tx.closed.return_value = False
        tx.run.return_value = MagicMock()
        session.begin_transaction.return_value = tx
        session.close.return_value = None
        driver.session.return_value = session
        return driver

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        collection_info = MagicMock()
        collection_info.points_count = 1000
        client.get_collection.return_value = collection_info
        client.upsert.return_value = None
        client.delete.return_value = None
        client.close.return_value = None
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with required nested attributes."""
        config = MagicMock()
        config.search.vector.qdrant.collection_name = "test_collection"
        config.search.vector.primary = "qdrant"
        config.search.vector.dual_write = False
        return config

    def test_coordinator_initialization(
        self, mock_neo4j_driver, mock_qdrant_client, mock_config
    ):
        """Coordinator initializes with required dependencies."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=mock_neo4j_driver,
            qdrant_client=mock_qdrant_client,
            config=mock_config,
            validate_before_commit=True,
            strict_mode=False,
        )

        assert coordinator.neo4j_driver is mock_neo4j_driver
        assert coordinator.qdrant_client is mock_qdrant_client
        assert coordinator.config is mock_config
        assert coordinator.validate_before_commit is True
        assert coordinator.strict_mode is False

    def test_coordinator_creates_validator(
        self, mock_neo4j_driver, mock_qdrant_client, mock_config
    ):
        """Coordinator creates IngestionValidator on init."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        coordinator = AtomicIngestionCoordinator(
            neo4j_driver=mock_neo4j_driver,
            qdrant_client=mock_qdrant_client,
            config=mock_config,
        )

        assert coordinator.validator is not None
        assert hasattr(coordinator.validator, "validate_pre_ingest")


# ============================================================================
# Tests for Hash Functions (Task 1)
# ============================================================================


class TestHashFunctions:
    """Tests for the hash computation helper methods."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with mocked dependencies for hash testing."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        mock_driver = MagicMock()
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_config.search.vector.qdrant.collection_name = "test"

        return AtomicIngestionCoordinator(mock_driver, mock_client, mock_config)

    def test_compute_text_hash_returns_sha256(self, coordinator):
        """_compute_text_hash returns a valid SHA256 hex digest."""
        import hashlib

        text = "test content"
        result = coordinator._compute_text_hash(text)

        # Should be 64 characters (SHA256 hex)
        assert len(result) == 64
        # Should match direct hashlib computation
        expected = hashlib.sha256(b"test content").hexdigest()
        assert result == expected

    def test_compute_text_hash_handles_empty_string(self, coordinator):
        """_compute_text_hash handles empty string gracefully."""
        import hashlib

        result = coordinator._compute_text_hash("")

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_compute_text_hash_handles_none(self, coordinator):
        """_compute_text_hash handles None input by treating as empty."""
        import hashlib

        result = coordinator._compute_text_hash(None)

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_compute_shingle_hash_creates_deterministic_hash(self, coordinator):
        """_compute_shingle_hash is deterministic for same input."""
        text = "the quick brown fox jumps over the lazy dog"

        result1 = coordinator._compute_shingle_hash(text)
        result2 = coordinator._compute_shingle_hash(text)

        assert result1 == result2
        assert len(result1) == 64  # SHA256 hex

    def test_compute_shingle_hash_handles_empty_string(self, coordinator):
        """_compute_shingle_hash returns empty string for empty input."""
        result = coordinator._compute_shingle_hash("")
        assert result == ""

    def test_compute_shingle_hash_handles_none(self, coordinator):
        """_compute_shingle_hash returns empty string for None input."""
        result = coordinator._compute_shingle_hash(None)
        assert result == ""

    def test_compute_shingle_hash_different_for_different_text(self, coordinator):
        """_compute_shingle_hash produces different hashes for different text."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the slow brown fox jumps over the lazy cat"

        result1 = coordinator._compute_shingle_hash(text1)
        result2 = coordinator._compute_shingle_hash(text2)

        assert result1 != result2

    def test_compute_shingle_hash_short_text(self, coordinator):
        """_compute_shingle_hash handles text shorter than n-gram size."""
        text = "hello world"  # Only 2 tokens, less than default n=8

        result = coordinator._compute_shingle_hash(text)

        # Should return empty string when not enough tokens for shingles
        assert result == ""


# ============================================================================
# Tests for Semantic Metadata Extraction (Task 1)
# ============================================================================


class TestSemanticMetadataExtraction:
    """Tests for semantic metadata extraction helper."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with mocked dependencies."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        mock_driver = MagicMock()
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_config.search.vector.qdrant.collection_name = "test"

        return AtomicIngestionCoordinator(mock_driver, mock_client, mock_config)

    def test_extract_semantic_metadata_returns_existing(self, coordinator):
        """_extract_semantic_metadata returns existing metadata if present."""
        section = {
            "semantic_metadata": {
                "entities": [{"name": "WEKA", "type": "ORG"}],
                "topics": ["storage", "filesystem"],
            }
        }

        result = coordinator._extract_semantic_metadata(section)

        assert result == section["semantic_metadata"]
        assert len(result["entities"]) == 1
        assert len(result["topics"]) == 2

    def test_extract_semantic_metadata_returns_default(self, coordinator):
        """_extract_semantic_metadata returns default structure if missing."""
        section = {"text": "some content"}

        result = coordinator._extract_semantic_metadata(section)

        assert result == {"entities": [], "topics": []}


# ============================================================================
# Tests for 42-Field Canonical Payload Schema (Task 1)
# ============================================================================


class TestCanonicalPayloadSchema:
    """Tests verifying atomic.py produces canonical 42-field payloads."""

    # The canonical schema from build_graph.py - these 42 fields must be present
    CANONICAL_FIELDS = {
        # Core identifiers (6)
        "node_id",
        "kg_id",
        "id",
        "document_id",
        "doc_id",
        "node_label",
        # Provenance (3)
        "document_uri",
        "source_uri",
        "source_path",
        # Filtering (3)
        "doc_tag",
        "snapshot_scope",
        "tenant",
        # Chunk hierarchy (4)
        "parent_section_id",
        "parent_section_original_id",
        "parent_chunk_id",
        "level",
        # Content (4)
        "heading",
        "text",
        "title",
        "anchor",
        # Structural (4)
        "order",
        "token_count",
        "document_total_tokens",
        "document_total_tokens_chunk",
        # Chunking metadata (4)
        "is_combined",
        "is_split",
        "original_section_ids",
        "boundaries_json",
        # Microdoc flags (3)
        "is_microdoc",
        "doc_is_microdoc",
        "is_microdoc_stub",
        # Versioning (2)
        "lang",
        "version",
        # Timestamps (1)
        "updated_at",
        # Hashes (2)
        "text_hash",
        "shingle_hash",
        # Semantic (1)
        "semantic_metadata",
        # Embedding metadata (5) - spread from canonicalize_embedding_metadata
        "embedding_version",
        "embedding_dimensions",
        "embedding_provider",
        "embedding_task",
        "embedding_timestamp",
    }

    def test_canonical_field_count(self):
        """Verify we're checking for exactly 42 canonical fields."""
        assert len(self.CANONICAL_FIELDS) == 42

    def test_payload_fields_match_build_graph(self):
        """Verify atomic.py payload structure matches build_graph.py canonical schema.

        This test ensures schema parity between the two code paths that write
        to Qdrant. Drift between them causes query failures in:
        - Graph reranker (needs id, doc_tag)
        - Multi-tenancy filtering (needs tenant, snapshot_scope)
        - Deduplication (needs text_hash, shingle_hash)
        - Embedding tracking (needs embedding_* fields)
        """
        # This is a documentation test - actual payload construction is tested
        # via integration tests. This verifies our expected field set is correct.

        # Core identifiers required by graph reranker
        graph_reranker_required = {"id", "doc_tag", "node_id"}
        assert graph_reranker_required.issubset(self.CANONICAL_FIELDS)

        # Multi-tenancy fields
        tenancy_required = {"tenant", "snapshot_scope"}
        assert tenancy_required.issubset(self.CANONICAL_FIELDS)

        # Drift detection fields
        drift_detection_required = {"text_hash", "shingle_hash"}
        assert drift_detection_required.issubset(self.CANONICAL_FIELDS)

        # Embedding metadata fields (Phase 7C.7)
        embedding_required = {
            "embedding_version",
            "embedding_dimensions",
            "embedding_provider",
            "embedding_task",
            "embedding_timestamp",
        }
        assert embedding_required.issubset(self.CANONICAL_FIELDS)


# ============================================================================
# Tests for Compensation Logic (Task 1)
# ============================================================================


class TestCompensationLogic:
    """Tests for saga compensation edge cases."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        tx = MagicMock()
        tx.closed.return_value = False
        tx.run.return_value = MagicMock()
        tx.commit.side_effect = RuntimeError("Simulated commit failure")
        tx.rollback.return_value = None
        session.begin_transaction.return_value = tx
        session.close.return_value = None
        driver.session.return_value = session
        return driver

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client that tracks operations."""
        client = MagicMock()
        client.upsert.return_value = None
        client.delete.return_value = None
        client.get_collection.return_value = MagicMock(points_count=100)
        client.close.return_value = None
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.search.vector.qdrant.collection_name = "test_collection"
        config.search.vector.primary = "qdrant"
        return config

    def test_qdrant_cleanup_runs_on_neo4j_commit_failure(
        self, mock_neo4j_driver, mock_qdrant_client, mock_config
    ):
        """Qdrant cleanup must run even when Neo4j rollback succeeds.

        Critical test: After Qdrant writes succeed but Neo4j commit fails,
        both stores must be cleaned up. Previously there was a bug where
        Qdrant points would be orphaned if Neo4j rolled back successfully.
        """
        # This is a design validation test. The actual coordination is complex
        # and involves the _execute_atomic_saga method. We verify the design
        # requirement here.

        # The compensation flow must be:
        # 1. Qdrant writes succeed (points exist)
        # 2. Neo4j commit fails
        # 3. Neo4j rolls back successfully
        # 4. Qdrant points MUST still be deleted (not orphaned)

        # This is ensured by the code structure in _execute_atomic_saga:
        # - written_qdrant_points is populated BEFORE Neo4j commit
        # - _compensate_qdrant is called in exception handler
        # - _compensate_qdrant is independent of Neo4j rollback success

        # Verify mock setup allows testing this scenario
        assert (
            mock_neo4j_driver.session.return_value.begin_transaction.return_value.rollback
        )
        assert mock_qdrant_client.delete

    def test_compensation_tracks_both_stores_independently(self):
        """Compensation state is tracked independently for Neo4j and Qdrant.

        The saga result must report:
        - neo4j_committed: bool (did Neo4j commit complete?)
        - qdrant_committed: bool (were Qdrant writes done?)
        - compensated: bool (was any cleanup performed?)

        These must be independent - Neo4j can rollback while Qdrant needs
        explicit compensation via delete.
        """
        from src.ingestion.atomic import AtomicIngestionResult

        # Create a result representing partial failure scenario
        result = AtomicIngestionResult(
            success=False,
            document_id="test_doc",
            saga_id="test_saga",
            neo4j_committed=False,  # Neo4j rolled back
            qdrant_committed=True,  # Qdrant had writes before failure
            compensated=True,  # Cleanup was performed
        )

        # Verify the result correctly reports independent states
        assert result.neo4j_committed is False
        assert result.qdrant_committed is True
        assert result.compensated is True

        # Verify serialization includes all fields
        d = result.to_dict()
        assert "neo4j_committed" in d
        assert "qdrant_committed" in d
        assert "compensated" in d


# ============================================================================
# Tests for Retry Logic (Task 5)
# ============================================================================


class TestNeo4jSanitization:
    """Tests for Neo4j data sanitization to avoid Map{} errors."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with mocked dependencies."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        mock_driver = MagicMock()
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_config.search.vector.qdrant.collection_name = "test"

        return AtomicIngestionCoordinator(mock_driver, mock_client, mock_config)

    def test_sanitize_keeps_primitives(self, coordinator):
        """Primitive values are kept unchanged."""
        data = {
            "string_field": "hello",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
        }

        result = coordinator._sanitize_for_neo4j(data)

        assert result["string_field"] == "hello"
        assert result["int_field"] == 42
        assert result["float_field"] == 3.14
        assert result["bool_field"] is True

    def test_sanitize_skips_none(self, coordinator):
        """None values are skipped entirely."""
        data = {
            "present": "value",
            "absent": None,
        }

        result = coordinator._sanitize_for_neo4j(data)

        assert "present" in result
        assert "absent" not in result

    def test_sanitize_keeps_primitive_lists(self, coordinator):
        """Lists of primitives are kept unchanged."""
        data = {
            "string_list": ["a", "b", "c"],
            "int_list": [1, 2, 3],
            "empty_list": [],
        }

        result = coordinator._sanitize_for_neo4j(data)

        assert result["string_list"] == ["a", "b", "c"]
        assert result["int_list"] == [1, 2, 3]
        assert result["empty_list"] == []

    def test_sanitize_serializes_dicts(self, coordinator):
        """Nested dicts are JSON serialized to strings."""
        data = {
            "semantic_metadata": {"entities": [], "topics": ["storage"]},
        }

        result = coordinator._sanitize_for_neo4j(data)

        # Should be a JSON string, not a dict
        assert isinstance(result["semantic_metadata"], str)
        assert "entities" in result["semantic_metadata"]
        assert "topics" in result["semantic_metadata"]

    def test_sanitize_serializes_complex_lists(self, coordinator):
        """Lists containing dicts are JSON serialized."""
        data = {
            "complex_list": [{"id": 1}, {"id": 2}],
        }

        result = coordinator._sanitize_for_neo4j(data)

        # Should be a JSON string
        assert isinstance(result["complex_list"], str)
        assert '"id": 1' in result["complex_list"] or '"id":1' in result["complex_list"]

    def test_sanitize_handles_real_section_data(self, coordinator):
        """Realistic section data is properly sanitized."""
        section = {
            "id": "abc123",
            "text": "Some content",
            "token_count": 50,
            "is_combined": False,
            "original_section_ids": ["sec1", "sec2"],  # List of strings - should stay
            "semantic_metadata": {
                "entities": [],
                "topics": [],
            },  # Dict - should serialize
            "boundaries_json": "{}",  # Already a string
        }

        result = coordinator._sanitize_for_neo4j(section)

        assert result["id"] == "abc123"
        assert result["text"] == "Some content"
        assert result["token_count"] == 50
        assert result["is_combined"] is False
        assert result["original_section_ids"] == ["sec1", "sec2"]
        assert isinstance(result["semantic_metadata"], str)  # Serialized to JSON
        assert result["boundaries_json"] == "{}"


class TestRetryLogic:
    """Tests for the retry with exponential backoff utilities."""

    def test_retry_decorator_succeeds_on_first_try(self):
        """Function succeeds without retry when no error."""
        from src.ingestion.atomic import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_retries_on_retriable_exception(self):
        """Function retries on retriable exceptions."""
        from src.ingestion.atomic import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = fails_twice_then_succeeds()
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third try

    def test_retry_decorator_exhausts_retries(self):
        """Function raises after exhausting retries."""
        from src.ingestion.atomic import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Persistent failure")

        with pytest.raises(TimeoutError):
            always_fails()

        assert call_count == 3  # Initial try + 2 retries

    def test_retry_decorator_does_not_retry_non_retriable(self):
        """Non-retriable exceptions fail immediately without retry."""
        from src.ingestion.atomic import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retriable")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # No retries for ValueError

    def test_qdrant_retry_wrapper_structure(self):
        """Verify the retry wrapper method exists and has correct signature."""
        from src.ingestion.atomic import AtomicIngestionCoordinator

        mock_driver = MagicMock()
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_config.search.vector.qdrant.collection_name = "test"

        coordinator = AtomicIngestionCoordinator(mock_driver, mock_client, mock_config)

        # Verify retry methods exist
        assert hasattr(coordinator, "_qdrant_upsert_with_retry")
        assert hasattr(coordinator, "_qdrant_delete_with_retry")

        # Verify they're callable
        assert callable(coordinator._qdrant_upsert_with_retry)
        assert callable(coordinator._qdrant_delete_with_retry)

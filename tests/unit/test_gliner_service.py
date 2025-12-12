"""
Unit tests for GLiNER NER Service.

Phase: GLiNER Integration - Phase 1 Core Infrastructure

Tests verify:
- Singleton pattern behavior
- Device auto-detection
- Entity extraction (mocked)
- Batch extraction (mocked)
- Circuit breaker / graceful degradation
- Caching behavior
- Labels helper functions
"""

from unittest.mock import Mock, patch

import pytest


class TestEntity:
    """Tests for the Entity dataclass."""

    def test_entity_creation(self):
        """Verify Entity dataclass creates correctly."""
        from src.providers.ner.gliner_service import Entity

        entity = Entity(
            text="WEKA",
            label="weka_software_component",
            start=10,
            end=14,
            score=0.95,
        )

        assert entity.text == "WEKA"
        assert entity.label == "weka_software_component"
        assert entity.start == 10
        assert entity.end == 14
        assert entity.score == 0.95

    def test_entity_frozen(self):
        """Verify Entity is immutable (frozen dataclass)."""
        from src.providers.ner.gliner_service import Entity

        entity = Entity("NFS", "protocol", 0, 3, 0.9)

        with pytest.raises(AttributeError):
            entity.text = "SMB"

    def test_entity_to_dict(self):
        """Verify Entity serialization to dict."""
        from src.providers.ner.gliner_service import Entity

        entity = Entity("RHEL", "operating_system", 5, 9, 0.85)
        d = entity.to_dict()

        assert d == {
            "text": "RHEL",
            "label": "operating_system",
            "start": 5,
            "end": 9,
            "score": 0.85,
        }

    def test_entity_hashable(self):
        """Verify Entity can be used in sets (hashable)."""
        from src.providers.ner.gliner_service import Entity

        e1 = Entity("WEKA", "component", 0, 4, 0.9)
        e2 = Entity("WEKA", "component", 0, 4, 0.9)
        e3 = Entity("NFS", "protocol", 5, 8, 0.8)

        # Same content = same hash (frozen dataclass)
        assert e1 == e2
        assert hash(e1) == hash(e2)

        # Different content = different
        assert e1 != e3

        # Can add to set
        entity_set = {e1, e2, e3}
        assert len(entity_set) == 2  # e1 and e2 are duplicates


class TestGLiNERServiceSingleton:
    """Tests for GLiNERService singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Verify singleton pattern returns same instance."""
        from src.providers.ner.gliner_service import GLiNERService

        # Reset to ensure clean state
        GLiNERService._instance = None
        GLiNERService._initialized = False

        service1 = GLiNERService()
        service2 = GLiNERService()

        assert service1 is service2

    def test_singleton_initialization_once(self):
        """Verify initialization runs only once."""
        from src.providers.ner.gliner_service import GLiNERService

        # Reset
        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.5
            mock_ner_config.batch_size = 16
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            _service1 = GLiNERService()
            _service2 = GLiNERService()

            # Config should only be accessed once (singleton pattern)
            assert mock_config.call_count == 1
            # Verify both references point to singleton
            assert _service1 is _service2


class TestDeviceDetection:
    """Tests for auto device detection."""

    def test_explicit_device_config(self):
        """Verify explicit device config bypasses auto-detection."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.5
            mock_ner_config.batch_size = 16
            mock_ner_config.device = "cuda"  # Explicit device
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()
            assert service._device == "cuda"

    def test_auto_device_cpu_fallback(self):
        """Verify auto device falls back to CPU when no GPU available."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.5
            mock_ner_config.batch_size = 16
            mock_ner_config.device = "auto"
            mock_config.return_value.ner = mock_ner_config

            # Mock torch to return no GPU available
            with patch("torch.backends.mps.is_available", return_value=False):
                with patch("torch.cuda.is_available", return_value=False):
                    service = GLiNERService()
                    assert service._device == "cpu"


class TestEntityExtraction:
    """Tests for entity extraction with mocked GLiNER model."""

    @pytest.fixture
    def mock_service(self):
        """Create a service with mocked model."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()

            # Mock the model
            mock_model = Mock()
            mock_model.predict_entities.return_value = [
                {
                    "text": "WEKA",
                    "label": "component",
                    "start": 0,
                    "end": 4,
                    "score": 0.9,
                },
                {
                    "text": "NFS",
                    "label": "protocol",
                    "start": 10,
                    "end": 13,
                    "score": 0.85,
                },
            ]
            service._model = mock_model

            yield service

    def test_extract_entities_empty_text(self, mock_service):
        """Verify empty text returns empty list."""
        result = mock_service.extract_entities("", ["component"])
        assert result == []

    def test_extract_entities_empty_labels(self, mock_service):
        """Verify empty labels returns empty list."""
        result = mock_service.extract_entities("Some text", [])
        assert result == []

    def test_extract_entities_success(self, mock_service):
        """Verify successful extraction returns Entity objects."""
        labels = ["component", "protocol"]
        result = mock_service.extract_entities("WEKA uses NFS protocol", labels)

        assert len(result) == 2
        assert result[0].text == "WEKA"
        assert result[0].label == "component"
        assert result[1].text == "NFS"
        assert result[1].label == "protocol"

    def test_extract_entities_model_not_loaded(self):
        """Verify graceful degradation when model fails to load."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()
            service._model_load_failed = True  # Simulate failed load

            result = service.extract_entities("Some text", ["label"])
            assert result == []


class TestBatchExtraction:
    """Tests for batch entity extraction."""

    def test_batch_extract_empty_texts(self):
        """Verify empty texts returns list of empty lists."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()
            result = service.batch_extract_entities([], ["label"])
            assert result == []

    def test_batch_extract_success(self):
        """Verify batch extraction returns correct structure."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()

            # Mock batch_predict_entities
            mock_model = Mock()
            mock_model.batch_predict_entities.return_value = [
                [
                    {
                        "text": "WEKA",
                        "label": "component",
                        "start": 0,
                        "end": 4,
                        "score": 0.9,
                    }
                ],
                [
                    {
                        "text": "NFS",
                        "label": "protocol",
                        "start": 0,
                        "end": 3,
                        "score": 0.85,
                    }
                ],
            ]
            service._model = mock_model

            texts = ["WEKA documentation", "NFS mount guide"]
            result = service.batch_extract_entities(texts, ["component", "protocol"])

            assert len(result) == 2
            assert len(result[0]) == 1
            assert result[0][0].text == "WEKA"
            assert len(result[1]) == 1
            assert result[1][0].text == "NFS"


class TestLabelsHelper:
    """Tests for labels.py helper functions."""

    def test_extract_label_name_with_examples(self):
        """Verify label name extraction strips examples."""
        from src.providers.ner.labels import extract_label_name

        label = "weka_software_component (e.g. backend, frontend, agent)"
        result = extract_label_name(label)
        assert result == "weka_software_component"

    def test_extract_label_name_without_examples(self):
        """Verify plain labels pass through unchanged."""
        from src.providers.ner.labels import extract_label_name

        label = "operating_system"
        result = extract_label_name(label)
        assert result == "operating_system"

    def test_default_labels_not_empty(self):
        """Verify DEFAULT_LABELS has content."""
        from src.providers.ner.labels import DEFAULT_LABELS

        assert len(DEFAULT_LABELS) > 0
        assert "weka_software_component" in DEFAULT_LABELS[0]

    def test_get_label_names_returns_clean_names(self):
        """Verify get_label_names returns clean names without examples."""
        from src.providers.ner.labels import get_label_names

        with patch("src.providers.ner.labels.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.labels = [
                "component (e.g. backend)",
                "protocol (e.g. NFS)",
            ]
            mock_config.return_value.ner = mock_ner_config

            names = get_label_names()
            assert "component" in names
            assert "protocol" in names
            assert "(e.g." not in str(names)


class TestCircuitBreaker:
    """Tests for circuit breaker / graceful degradation."""

    def test_model_load_failure_sets_flag(self):
        """Verify failed model load sets circuit breaker flag."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "nonexistent-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()

            # Mock the gliner module import to raise ImportError
            # GLiNER is imported inside _load_model, so we patch the import itself
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "gliner":
                    raise ImportError("No module named 'gliner'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                # This should fail and set the flag
                result = service._load_model()
                assert result is False
                assert service._model_load_failed is True

    def test_circuit_breaker_prevents_retry(self):
        """Verify circuit breaker prevents repeated load attempts."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()
            service._model_load_failed = True

            # Should return False immediately without attempting load
            with patch("builtins.__import__") as mock_import:
                result = service._load_model()
                assert result is False
                mock_import.assert_not_called()


class TestServiceReset:
    """Tests for service reset functionality."""

    def test_reset_clears_state(self):
        """Verify reset clears model and flags."""
        from src.providers.ner.gliner_service import GLiNERService

        GLiNERService._instance = None
        GLiNERService._initialized = False

        with patch("src.providers.ner.gliner_service.get_config") as mock_config:
            mock_ner_config = Mock()
            mock_ner_config.model_name = "test-model"
            mock_ner_config.threshold = 0.45
            mock_ner_config.batch_size = 32
            mock_ner_config.device = "cpu"
            mock_config.return_value.ner = mock_ner_config

            service = GLiNERService()
            service._model = Mock()
            service._model_load_failed = True

            service.reset()

            assert service._model is None
            assert service._model_load_failed is False

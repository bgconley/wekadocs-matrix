"""
Integration tests for GLiNER configuration loading.

Phase: GLiNER Integration - Phase 1 Core Infrastructure

These tests verify that:
- NERConfig loads correctly from development.yaml
- Config defaults are applied when values missing
- Labels are properly loaded from YAML
- Config integrates with the rest of the system
"""


class TestNERConfigIntegration:
    """Integration tests for NER configuration."""

    def test_config_loads_ner_section(self):
        """Verify NER config section loads from development.yaml."""
        from src.shared.config import load_config

        # load_config returns (Config, Settings) tuple
        config, _settings = load_config()

        assert hasattr(config, "ner")
        assert config.ner is not None

    def test_config_ner_has_required_fields(self):
        """Verify NER config has all required fields."""
        from src.shared.config import load_config

        config, _ = load_config()
        ner = config.ner

        assert hasattr(ner, "enabled")
        assert hasattr(ner, "model_name")
        assert hasattr(ner, "threshold")
        assert hasattr(ner, "device")
        assert hasattr(ner, "batch_size")
        assert hasattr(ner, "labels")

    def test_config_ner_default_disabled(self):
        """Verify NER is disabled by default."""
        from src.shared.config import load_config

        config, _ = load_config()
        # Development config should have enabled: false
        assert config.ner.enabled is False

    def test_config_ner_model_name(self):
        """Verify model name is set correctly."""
        from src.shared.config import load_config

        config, _ = load_config()
        assert "gliner" in config.ner.model_name.lower()

    def test_config_ner_threshold_valid_range(self):
        """Verify threshold is in valid range (0.0-1.0)."""
        from src.shared.config import load_config

        config, _ = load_config()
        assert 0.0 <= config.ner.threshold <= 1.0

    def test_config_ner_device_auto_default(self):
        """Verify device defaults to auto-detection."""
        from src.shared.config import load_config

        config, _ = load_config()
        assert config.ner.device == "auto"

    def test_config_ner_batch_size_reasonable(self):
        """Verify batch size is reasonable for Apple Silicon."""
        from src.shared.config import load_config

        config, _ = load_config()
        # Should be in reasonable range for GPU batching
        assert 8 <= config.ner.batch_size <= 128

    def test_config_ner_labels_loaded(self):
        """Verify domain-specific labels are loaded from YAML."""
        from src.shared.config import load_config

        config, _ = load_config()
        labels = config.ner.labels

        assert isinstance(labels, list)
        assert len(labels) > 0
        # Should have WEKA-specific labels
        assert any("weka" in label.lower() for label in labels)

    def test_config_ner_labels_have_examples(self):
        """Verify labels include example hints for zero-shot."""
        from src.shared.config import load_config

        config, _ = load_config()
        labels = config.ner.labels

        # Most labels should have (e.g. ...) examples
        labels_with_examples = [label for label in labels if "(e.g." in label]
        assert len(labels_with_examples) >= len(labels) * 0.5  # At least half

    def test_config_ner_known_label_types(self):
        """Verify expected label types are present."""
        from src.shared.config import load_config

        config, _ = load_config()
        labels_str = " ".join(config.ner.labels).lower()

        # Check for expected domain-specific types
        expected_types = [
            "software_component",
            "operating_system",
            "hardware",
            "filesystem",
            "cloud",
            "cli_command",
            "protocol",
            "error",
            "performance",
            "path",
        ]

        found = [t for t in expected_types if t in labels_str]
        # Should find most expected types
        assert len(found) >= len(expected_types) * 0.7


class TestLabelsHelperIntegration:
    """Integration tests for labels helper with real config."""

    def test_get_default_labels_from_config(self):
        """Verify get_default_labels returns config values."""
        from src.providers.ner.labels import get_default_labels

        labels = get_default_labels()

        assert isinstance(labels, list)
        assert len(labels) > 0
        # Should match what's in config
        assert any("weka" in label.lower() for label in labels)

    def test_get_label_names_clean(self):
        """Verify get_label_names returns clean names."""
        from src.providers.ner.labels import get_label_names

        names = get_label_names()

        assert isinstance(names, list)
        assert len(names) > 0
        # Names should not contain example syntax
        for name in names:
            assert "(e.g." not in name
            assert ")" not in name


class TestNERConfigDefaults:
    """Test NERConfig default values."""

    def test_nerconfig_defaults_without_yaml(self):
        """Verify NERConfig has sensible defaults even without YAML."""
        from src.shared.config import NERConfig

        # Create with no values - should use defaults
        config = NERConfig()

        assert config.enabled is False
        assert config.model_name == "urchade/gliner_medium-v2.1"
        assert config.threshold == 0.45
        assert config.device == "auto"
        assert config.batch_size == 32
        assert config.labels == []  # Empty by default, populated from YAML

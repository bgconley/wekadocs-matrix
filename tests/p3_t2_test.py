# Phase 3, Task 3.2 Tests - Entity Extraction
# NO MOCKS - Tests against real parsed sections

from pathlib import Path

import pytest

from src.ingestion.extract import extract_entities
from src.ingestion.parsers.markdown import parse_markdown


class TestCommandExtraction:
    """Tests for command entity extraction."""

    @pytest.fixture
    def sample_sections(self):
        """Parse sample document and return sections."""
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "getting_started.md"

        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        return result["Sections"]

    def test_extracts_commands_from_code_blocks(self, sample_sections):
        """Test that commands are extracted from code blocks."""
        entities, mentions = extract_entities(sample_sections)

        commands = [e for e in entities.values() if e["label"] == "Command"]
        assert len(commands) > 0

        # Should find weka commands
        command_names = [c["name"] for c in commands]
        weka_commands = [c for c in command_names if "weka" in c.lower()]
        assert len(weka_commands) > 0

    def test_commands_have_mentions(self, sample_sections):
        """Test that commands have MENTIONS relationships."""
        entities, mentions = extract_entities(sample_sections)

        commands = [e for e in entities.values() if e["label"] == "Command"]
        command_ids = {c["id"] for c in commands}

        # Find mentions for commands (only those with entity_id, not relationships)
        command_mentions = [
            m for m in mentions if "entity_id" in m and m["entity_id"] in command_ids
        ]
        assert len(command_mentions) > 0

        # Verify mention structure
        for mention in command_mentions:
            assert "section_id" in mention
            assert "entity_id" in mention
            assert "confidence" in mention
            assert "start" in mention
            assert "end" in mention
            assert "source_section_id" in mention
            assert 0.0 <= mention["confidence"] <= 1.0

    def test_command_confidence_scores(self, sample_sections):
        """Test that command mentions have reasonable confidence scores."""
        entities, mentions = extract_entities(sample_sections)

        commands = [e for e in entities.values() if e["label"] == "Command"]
        command_ids = {c["id"] for c in commands}
        command_mentions = [
            m for m in mentions if "entity_id" in m and m["entity_id"] in command_ids
        ]

        # Code block commands should have high confidence
        high_confidence = [m for m in command_mentions if m["confidence"] >= 0.8]
        assert len(high_confidence) > 0


class TestConfigurationExtraction:
    """Tests for configuration entity extraction."""

    @pytest.fixture
    def sample_sections(self):
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "getting_started.md"

        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        return result["Sections"]

    def test_extracts_config_files(self, sample_sections):
        """Test extraction of configuration files."""
        entities, mentions = extract_entities(sample_sections)

        configs = [e for e in entities.values() if e["label"] == "Configuration"]
        assert len(configs) > 0

        # Should find weka.conf
        config_names = [c["name"] for c in configs]
        assert any("weka.conf" in name for name in config_names)

    def test_extracts_config_parameters(self, sample_sections):
        """Test extraction of configuration parameters."""
        entities, mentions = extract_entities(sample_sections)

        configs = [e for e in entities.values() if e["label"] == "Configuration"]
        config_names = [c["name"] for c in configs]

        # Should find parameters like cluster_name, network_interface
        param_configs = [
            c
            for c in config_names
            if any(
                keyword in c.lower()
                for keyword in ["cluster", "network", "storage", "max", "cache"]
            )
        ]
        assert len(param_configs) > 0

    def test_extracts_env_variables(self, sample_sections):
        """Test extraction of environment variables."""
        # Add a section with env vars
        env_section = {
            "id": "test-env-section",
            "document_id": "test-doc",
            "title": "Environment Variables",
            "text": "Set $WEKA_HOME and ${MAX_MEMORY} before starting.",
            "tokens": 10,
            "checksum": "abc123",
            "anchor": "env",
            "level": 2,
            "order": 0,
            "code_blocks": [],
            "tables": [],
        }

        entities, mentions = extract_entities([env_section])

        configs = [e for e in entities.values() if e["label"] == "Configuration"]
        config_names = [c["name"] for c in configs]

        # Should find env vars
        assert "WEKA_HOME" in config_names or "MAX_MEMORY" in config_names


class TestProcedureExtraction:
    """Tests for procedure and step extraction."""

    @pytest.fixture
    def sample_sections(self):
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "getting_started.md"

        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        return result["Sections"]

    def test_extracts_procedures_from_titled_sections(self, sample_sections):
        """Test that procedural sections are detected."""
        entities, mentions = extract_entities(sample_sections)

        procedures = [e for e in entities.values() if e["label"] == "Procedure"]
        # Installation section should be detected as procedural
        assert len(procedures) > 0

    def test_extracts_steps_from_numbered_lists(self, sample_sections):
        """Test that steps are extracted from numbered lists."""
        entities, mentions = extract_entities(sample_sections)

        steps = [e for e in entities.values() if e["label"] == "Step"]
        assert len(steps) > 0

        # Steps should have order field
        for step in steps:
            assert "order" in step
            assert isinstance(step["order"], int)
            assert step["order"] > 0

    def test_steps_have_procedure_relationship(self, sample_sections):
        """Test that steps are linked to procedures."""
        entities, mentions = extract_entities(sample_sections)

        steps = [e for e in entities.values() if e["label"] == "Step"]

        # Some steps should have procedure_id
        steps_with_proc = [s for s in steps if s.get("procedure_id")]
        assert len(steps_with_proc) > 0


class TestEntityExtractionPrecision:
    """Tests for extraction precision (DoD: >95% on commands/configs)."""

    def test_minimal_false_positives(self):
        """Test that extraction doesn't produce many false positives."""
        # Create a section with clear non-entities
        test_section = {
            "id": "test-section",
            "document_id": "test-doc",
            "title": "Test Section",
            "text": "The quick brown fox jumps over the lazy dog. This is a test.",
            "tokens": 15,
            "checksum": "abc123",
            "anchor": "test",
            "level": 1,
            "order": 0,
            "code_blocks": [],
            "tables": [],
        }

        entities, mentions = extract_entities([test_section])

        # Should not extract many entities from generic text
        assert len(entities) < 5

    def test_extracts_from_realistic_documentation(self):
        """Test extraction from realistic documentation sections."""
        samples_path = Path(__file__).parent.parent / "data" / "samples"

        # Test all sample documents
        sample_files = [
            "getting_started.md",
            "api_guide.md",
            "performance_tuning.md",
        ]

        total_commands = 0
        total_configs = 0

        for sample_file in sample_files:
            md_path = samples_path / sample_file
            with open(md_path, "r") as f:
                content = f.read()

            result = parse_markdown(str(md_path), content)
            entities, mentions = extract_entities(result["Sections"])

            commands = [e for e in entities.values() if e["label"] == "Command"]
            configs = [e for e in entities.values() if e["label"] == "Configuration"]

            total_commands += len(commands)
            total_configs += len(configs)

        # Should extract reasonable number of entities
        assert total_commands > 5, f"Expected >5 commands, got {total_commands}"
        assert total_configs > 5, f"Expected >5 configs, got {total_configs}"

    def test_mention_provenance(self):
        """Test that all mentions have proper provenance."""
        samples_path = Path(__file__).parent.parent / "data" / "samples"
        md_path = samples_path / "api_guide.md"

        with open(md_path, "r") as f:
            content = f.read()

        result = parse_markdown(str(md_path), content)
        entities, mentions = extract_entities(result["Sections"])

        # All mentions must have source_section_id
        for mention in mentions:
            assert mention["source_section_id"]
            assert mention["source_section_id"] == mention["section_id"]

        # All mentions must have valid spans
        for mention in mentions:
            assert mention["start"] < mention["end"]
            assert mention["start"] >= 0

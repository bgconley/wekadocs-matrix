"""
Phase 2 Schema Cleanup Tests

TDD tests that validate removal of dead relationship types from the schema.
These tests should FAIL initially, then PASS after implementation.

Dead types being removed:
- AFFECTS, CRITICAL_FOR, DEPENDS_ON, RELATED_TO, REQUIRES (never materialized)
- PREV (redundant - use <-[:NEXT]-)
- SAME_HEADING (O(n²) fanout with zero query usage)
"""

import re
import subprocess
from pathlib import Path

import pytest

# Frozen set of dead relationship types being removed in Phase 2
DEAD_TYPES = frozenset(
    [
        "AFFECTS",
        "CRITICAL_FOR",
        "DEPENDS_ON",
        "RELATED_TO",
        "REQUIRES",
        "PREV",
        "SAME_HEADING",
    ]
)

# Expected relationship types after cleanup
EXPECTED_ACTIVE_TYPES = frozenset(
    [
        "ANSWERED_AS",
        "CHILD_OF",
        "CONTAINS_STEP",
        "DEFINES",
        "EXECUTES",
        "FOCUSED_ON",
        "HAS_CITATION",
        "HAS_PARAMETER",
        "HAS_QUERY",
        "HAS_SECTION",
        "IN_CHUNK",
        "IN_SECTION",
        "MENTIONED_IN",
        "MENTIONS",
        "NEXT",
        "NEXT_CHUNK",
        "PARENT_OF",
        "RESOLVES",
        "RETRIEVED",
        "SUPPORTED_BY",
    ]
)


class TestSchemaRelationshipTypes:
    """Test that RELATIONSHIP_TYPES set excludes dead types."""

    def test_relationship_types_excludes_dead_types(self):
        """RELATIONSHIP_TYPES should not contain any dead types."""
        from src.neo.schema import RELATIONSHIP_TYPES

        intersection = RELATIONSHIP_TYPES & DEAD_TYPES
        assert not intersection, f"Dead types found in schema: {intersection}"

    def test_relationship_types_contains_expected_active_types(self):
        """RELATIONSHIP_TYPES should contain all expected active types."""
        from src.neo.schema import RELATIONSHIP_TYPES

        missing = EXPECTED_ACTIVE_TYPES - RELATIONSHIP_TYPES
        assert not missing, f"Missing expected active types: {missing}"

    def test_relationship_types_only_contains_expected_types(self):
        """RELATIONSHIP_TYPES should not contain unexpected types."""
        from src.neo.schema import RELATIONSHIP_TYPES

        unexpected = RELATIONSHIP_TYPES - EXPECTED_ACTIVE_TYPES
        assert not unexpected, f"Unexpected types in schema: {unexpected}"


class TestAtomicIngestionAllowlist:
    """Test that atomic ingestion allowlist excludes dead types."""

    def test_entity_relationship_allowlist_excludes_dead_types(self):
        """ALLOWED_ENTITY_RELATIONSHIP_TYPES should not contain dead types."""
        from src.ingestion.atomic import ALLOWED_ENTITY_RELATIONSHIP_TYPES

        intersection = ALLOWED_ENTITY_RELATIONSHIP_TYPES & DEAD_TYPES
        assert not intersection, f"Dead types in allowlist: {intersection}"


class TestCodebaseNoDeadReferences:
    """Grep codebase to ensure no dead type references remain in src/."""

    @pytest.fixture
    def src_path(self) -> Path:
        """Return path to src directory."""
        return Path(__file__).parent.parent.parent / "src"

    @pytest.mark.parametrize("dead_type", list(DEAD_TYPES))
    def test_no_dead_type_in_python_code(self, src_path: Path, dead_type: str):
        """
        No Python file in src/ should reference dead relationship types in active code.

        Excludes:
        - Comments (lines starting with # or containing cleanup documentation)
        - Docstrings
        - Cleanup documentation comments (Phase 2 Cleanup references)
        """
        if not src_path.exists():
            pytest.skip(f"src path not found: {src_path}")

        # Use grep to find references
        # Pattern: word boundary match for the dead type as a string literal
        pattern = rf'["\']?{dead_type}["\']?'

        result = subprocess.run(
            [
                "grep",
                "-rn",
                "--include=*.py",
                "-E",
                pattern,
                str(src_path),
            ],
            capture_output=True,
            text=True,
        )

        matches = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Extract the actual code content after the file:line: prefix
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            code_content = parts[2].strip()

            # Skip comment-only lines
            if code_content.startswith("#"):
                continue
            # Skip if it's in a docstring context (harder to detect, use heuristic)
            if '"""' in code_content or "'''" in code_content:
                continue
            # Skip Phase 2 Cleanup documentation comments
            if "Phase 2 Cleanup" in line:
                continue
            # Skip lines that are documenting removal (e.g., "Removed PREV")
            if "Removed" in line:
                continue
            # Skip lines mentioning removal context (dead types in parens after "Removed")
            if "fanout with zero query usage" in line:
                continue
            matches.append(line)

        assert (
            not matches
        ), f"Found references to dead type '{dead_type}' in src/:\n" + "\n".join(
            matches[:10]
        )  # Limit output

    def test_no_dead_types_in_cypher_templates(self, src_path: Path):
        """No Cypher template should reference dead relationship types in active queries."""
        if not src_path.exists():
            pytest.skip(f"src path not found: {src_path}")

        templates_path = src_path / "query" / "templates"
        if not templates_path.exists():
            pytest.skip("Templates path not found")

        issues = []
        for cypher_file in templates_path.rglob("*.cypher"):
            content = cypher_file.read_text()
            for dead_type in DEAD_TYPES:
                if dead_type in content:
                    # Find line numbers
                    for i, line in enumerate(content.splitlines(), 1):
                        if dead_type in line:
                            # Skip comment lines (Cypher comments start with --)
                            if line.strip().startswith("--"):
                                continue
                            # Skip Phase 2 Cleanup documentation
                            if "Phase 2 Cleanup" in line:
                                continue
                            issues.append(f"{cypher_file.name}:{i}: {dead_type}")

        assert not issues, "Dead types found in Cypher templates:\n" + "\n".join(
            issues[:20]
        )


class TestBuildGraphNoDeadEdges:
    """Test that build_graph.py doesn't create dead edge types."""

    def test_no_prev_edge_creation_query(self):
        """build_graph.py should not have a query creating PREV edges."""
        build_graph_path = (
            Path(__file__).parent.parent.parent / "src" / "ingestion" / "build_graph.py"
        )

        content = build_graph_path.read_text()

        # Check for MERGE patterns that create PREV edges
        prev_patterns = [
            r"MERGE\s*\([^)]*\)-\[:PREV\]->",
            r"CREATE\s*\([^)]*\)-\[:PREV\]->",
            r"-\[:PREV\]->\s*\(",
        ]

        for pattern in prev_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, f"Found PREV edge creation pattern: {matches}"

    def test_no_same_heading_edge_creation_query(self):
        """build_graph.py should not have a query creating SAME_HEADING edges."""
        build_graph_path = (
            Path(__file__).parent.parent.parent / "src" / "ingestion" / "build_graph.py"
        )

        content = build_graph_path.read_text()

        # Check for MERGE patterns that create SAME_HEADING edges
        same_heading_patterns = [
            r"MERGE\s*\([^)]*\)-\[:SAME_HEADING\]->",
            r"CREATE\s*\([^)]*\)-\[:SAME_HEADING\]->",
            r"-\[:SAME_HEADING\]->\s*\(",
            r'"same_heading":\s*"""',  # Query block key
        ]

        for pattern in same_heading_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, f"Found SAME_HEADING edge creation pattern: {matches}"

    def test_relationship_queries_dict_excludes_dead_keys(self):
        """The relationship queries dict should not have keys for dead types."""
        build_graph_path = (
            Path(__file__).parent.parent.parent / "src" / "ingestion" / "build_graph.py"
        )

        content = build_graph_path.read_text()

        # Check that "next_prev" is not used as a dict key (excluding comments)
        # We look for the pattern of it being a dict key: "next_prev":
        next_prev_as_key = re.search(r'"next_prev"\s*:', content)
        assert (
            next_prev_as_key is None
        ), 'Query key "next_prev" should be renamed to "next"'

        # Check that "same_heading" is not used as a dict key
        same_heading_as_key = re.search(r'"same_heading"\s*:', content)
        assert (
            same_heading_as_key is None
        ), 'Query key "same_heading" should be removed entirely'


class TestConfigNoDeadTypes:
    """Test that config defaults don't reference dead types."""

    def test_query_type_relationships_excludes_dead_types(self):
        """Config query_type_relationships should not contain dead types."""
        from src.shared.config import HybridSearchConfig

        config = HybridSearchConfig()

        for query_type, rel_types in config.query_type_relationships.items():
            dead_in_list = set(rel_types) & DEAD_TYPES
            assert not dead_in_list, (
                f"Dead types in config.query_type_relationships['{query_type}']: "
                f"{dead_in_list}"
            )

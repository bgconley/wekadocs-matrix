"""
Phase 4, Task 4.1 - Advanced Query Templates Tests (NO MOCKS)
"""

import time
from pathlib import Path

import pytest

from src.query.templates.advanced.schemas import ADVANCED_TEMPLATES, get_template


@pytest.fixture(scope="module")
def setup_test_data(neo4j_driver):
    """Create test graph data for advanced templates."""
    with neo4j_driver.session() as session:
        session.run(
            """
            MERGE (c1:Component {id: 'comp-test-1', name: 'WebAPI'})
            MERGE (c2:Component {id: 'comp-test-2', name: 'Database'})
            MERGE (c3:Component {id: 'comp-test-3', name: 'Cache'})
            MERGE (c1)-[:DEPENDS_ON]->(c2)
            MERGE (c1)-[:DEPENDS_ON]->(c3)

            MERGE (cfg1:Configuration {id: 'cfg-test-1', name: 'max_connections'})
            MERGE (cfg2:Configuration {id: 'cfg-test-2', name: 'timeout'})
            MERGE (svc:Component {id: 'svc-critical', name: 'AuthService'})
            MERGE (cfg1)-[:AFFECTS]->(c2)
            MERGE (c2)-[:CRITICAL_FOR]->(svc)

            MERGE (e1:Error {id: 'err-test-1', code: 'E404', name: 'Not Found'})
            MERGE (proc1:Procedure {id: 'proc-test-1', name: 'Reset Connection'})
            MERGE (step1:Step {id: 'step-test-1', text: 'Check config', order: 1})
            MERGE (cmd1:Command {id: 'cmd-test-1', name: 'netstat'})
            MERGE (e1)<-[:RESOLVES]-(proc1)
            MERGE (proc1)-[:CONTAINS_STEP]->(step1)
            MERGE (step1)-[:EXECUTES]->(cmd1)
        """
        )
    yield
    # Cleanup
    with neo4j_driver.session() as session:
        session.run(
            """
            MATCH (n) WHERE n.id STARTS WITH 'comp-test-' OR n.id STARTS WITH 'cfg-test-'
              OR n.id STARTS WITH 'err-test-' OR n.id STARTS WITH 'proc-test-'
              OR n.id STARTS WITH 'step-test-' OR n.id STARTS WITH 'cmd-test-'
              OR n.id = 'svc-critical'
            DETACH DELETE n
        """
        )


class TestTemplateRegistry:
    """Test template registry and metadata."""

    def test_list_templates(self):
        templates = list(ADVANCED_TEMPLATES.keys())
        assert "dependency_chain" in templates
        assert "impact_assessment" in templates
        assert "comparison" in templates
        assert "temporal" in templates
        assert "troubleshooting_path" in templates

    def test_get_template(self):
        template = get_template("dependency_chain")
        assert template.name == "dependency_chain"
        assert template.guardrails.max_depth > 0
        assert template.guardrails.timeout_ms > 0

    def test_get_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent")


class TestTemplateSchemas:
    """Test template schemas and guardrails."""

    def test_dependency_chain_schema(self):
        template = get_template("dependency_chain")
        assert "component_name" in template.input_schema["properties"]
        assert "component_name" in template.input_schema["required"]
        assert template.guardrails.max_depth == 5
        assert template.guardrails.max_results == 100
        assert template.guardrails.timeout_ms == 30000
        assert "DEPENDS_ON" in template.guardrails.allowed_rel_types

    def test_impact_assessment_schema(self):
        template = get_template("impact_assessment")
        assert "config_name" in template.input_schema["properties"]
        assert template.guardrails.max_depth == 3
        assert "AFFECTS" in template.guardrails.allowed_rel_types

    def test_comparison_schema(self):
        template = get_template("comparison")
        assert "entity_type" in template.input_schema["properties"]
        assert "entity_name_a" in template.input_schema["properties"]
        assert "entity_name_b" in template.input_schema["properties"]
        assert template.guardrails.max_depth == 2

    def test_temporal_schema(self):
        template = get_template("temporal")
        assert "version" in template.input_schema["properties"]
        assert "version" in template.input_schema["required"]
        assert template.guardrails.requires_indexes is not None
        assert "introduced_in" in template.guardrails.requires_indexes

    def test_troubleshooting_path_schema(self):
        template = get_template("troubleshooting_path")
        assert "error_code" in template.input_schema["properties"]
        assert template.guardrails.max_depth == 3
        assert "RESOLVES" in template.guardrails.allowed_rel_types


def extract_version_query(query_text: str, version: int = 1) -> str:
    """Extract a specific version of a query from template file, removing comments."""
    # Split by version markers
    parts = query_text.split(f"-- Version {version}:")
    if len(parts) < 2:
        raise ValueError(f"Version {version} not found in query")

    # Get the section for this version
    version_section = (
        parts[1].split(f"-- Version {version + 1}:")[0]
        if f"-- Version {version + 1}:" in parts[1]
        else parts[1]
    )

    # Remove comment lines and clean up
    lines = []
    cypher_keywords = [
        "MATCH",
        "WITH",
        "RETURN",
        "OPTIONAL",
        "MERGE",
        "CREATE",
        "CALL",
        "UNWIND",
    ]
    found_query_start = False

    for line in version_section.split("\n"):
        stripped = line.strip()

        # Check if we've reached actual Cypher code
        if not found_query_start:
            # Look for lines starting with Cypher keywords
            if any(stripped.startswith(kw) for kw in cypher_keywords):
                found_query_start = True

        # Once we find the query start, include non-comment lines
        if found_query_start and stripped and not stripped.startswith("--"):
            lines.append(line.rstrip())

    return "\n".join(lines)


class TestTemplateExecution:
    """Test template execution within budgets."""

    def test_dependency_chain_executes(self, neo4j_driver, setup_test_data):
        template = get_template("dependency_chain")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        start = time.time()
        with neo4j_driver.session() as session:
            result = session.run(
                query, {"component_name": "WebAPI", "max_depth": 5, "limit": 100}
            )
            records = list(result)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < template.guardrails.timeout_ms
        assert len(records) <= template.guardrails.max_results

    def test_impact_assessment_executes(self, neo4j_driver, setup_test_data):
        template = get_template("impact_assessment")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        start = time.time()
        with neo4j_driver.session() as session:
            result = session.run(
                query, {"config_name": "max_connections", "max_hops": 3}
            )
            records = list(result)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < template.guardrails.timeout_ms
        assert len(records) <= template.guardrails.max_results

    def test_troubleshooting_path_executes(self, neo4j_driver, setup_test_data):
        template = get_template("troubleshooting_path")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        start = time.time()
        with neo4j_driver.session() as session:
            result = session.run(query, {"error_code": "E404", "error_name": None})
            records = list(result)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < template.guardrails.timeout_ms
        assert len(records) <= template.guardrails.max_results


class TestTemplateOutputs:
    """Test template outputs match expected schema."""

    def test_dependency_chain_output_structure(self, neo4j_driver, setup_test_data):
        template = get_template("dependency_chain")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        with neo4j_driver.session() as session:
            result = session.run(
                query, {"component_name": "WebAPI", "max_depth": 5, "limit": 100}
            )
            records = list(result)

        if records:
            record = records[0]
            assert "dep" in record.keys() or "path" in record.keys()
            assert "depth" in record.keys()

    def test_impact_assessment_output_structure(self, neo4j_driver, setup_test_data):
        template = get_template("impact_assessment")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        with neo4j_driver.session() as session:
            result = session.run(
                query, {"config_name": "max_connections", "max_hops": 3}
            )
            records = list(result)

        if records:
            record = records[0]
            assert "cfg" in record.keys()
            assert "impact_level" in record.keys()


class TestTemplateEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_component(self, neo4j_driver, setup_test_data):
        template = get_template("dependency_chain")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        with neo4j_driver.session() as session:
            result = session.run(
                query,
                {
                    "component_name": "NonExistentComponent",
                    "max_depth": 5,
                    "limit": 100,
                },
            )
            records = list(result)

        assert records == []

    def test_nonexistent_config(self, neo4j_driver, setup_test_data):
        template = get_template("impact_assessment")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        with neo4j_driver.session() as session:
            result = session.run(
                query, {"config_name": "nonexistent_config", "max_hops": 3}
            )
            records = list(result)

        assert records == []

    def test_max_depth_respected(self, neo4j_driver, setup_test_data):
        template = get_template("dependency_chain")
        query_text = Path(template.file_path).read_text()
        query = extract_version_query(query_text, version=1)

        with neo4j_driver.session() as session:
            result = session.run(
                query, {"component_name": "WebAPI", "max_depth": 1, "limit": 100}
            )
            records = list(result)

        for record in records:
            assert record["depth"] <= 1


class TestTemplateGuardrails:
    """Test that guardrails are properly enforced."""

    def test_all_templates_have_guardrails(self):
        for name, template in ADVANCED_TEMPLATES.items():
            assert template.guardrails is not None
            assert template.guardrails.max_depth >= 1
            assert template.guardrails.max_depth <= 10
            assert template.guardrails.max_results >= 1
            assert template.guardrails.timeout_ms >= 1000

    def test_all_templates_have_file_paths(self):
        for name, template in ADVANCED_TEMPLATES.items():
            assert template.file_path is not None
            file_path = Path(template.file_path)
            assert file_path.exists(), f"Template file not found: {template.file_path}"

    def test_all_templates_have_schemas(self):
        for name, template in ADVANCED_TEMPLATES.items():
            assert template.input_schema is not None
            assert "type" in template.input_schema
            assert template.output_schema is not None
            assert "type" in template.output_schema

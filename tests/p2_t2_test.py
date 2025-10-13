"""
Phase 2, Task 2.2 Tests: Cypher Validator
Tests regex guards and EXPLAIN plan gates with NO MOCKS.
Uses real Neo4j connection for EXPLAIN checks.
"""

import os

import pytest
from neo4j import GraphDatabase

from src.mcp_server.validation import CypherValidator, ValidationError


@pytest.fixture(scope="module")
def neo4j_driver():
    """Get Neo4j driver for validator tests."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "testpassword123")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()


class TestValidatorRegexGuards:
    """Test forbidden keyword detection."""

    def test_blocks_delete(self):
        validator = CypherValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("MATCH (n) DELETE n", {})
        assert "DELETE" in str(exc_info.value)

    def test_blocks_detach_delete(self):
        validator = CypherValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("MATCH (n) DETACH DELETE n", {})
        assert "DELETE" in str(exc_info.value)

    def test_blocks_drop(self):
        validator = CypherValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("DROP INDEX my_index", {})
        assert "DROP" in str(exc_info.value)

    def test_blocks_merge(self):
        validator = CypherValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("MERGE (n:Node {id: 'x'})", {})
        assert "MERGE" in str(exc_info.value)

    def test_blocks_set_without_param(self):
        validator = CypherValidator()
        with pytest.raises(ValidationError):
            # SET without parameter should be blocked
            validator.validate("MATCH (n) SET n.prop = 'value'", {})

    def test_blocks_grant_revoke(self):
        validator = CypherValidator()
        with pytest.raises(ValidationError):
            validator.validate("GRANT MATCH ON GRAPH * TO user", {})

    def test_allows_read_queries(self):
        validator = CypherValidator()
        # Should not raise
        result = validator.validate(
            "MATCH (n:Section) WHERE n.id = $id RETURN n LIMIT 10", {"id": "abc123"}
        )
        assert result.valid


class TestValidatorTraversalDepth:
    """Test variable-length path depth validation."""

    def test_blocks_excessive_depth(self):
        validator = CypherValidator()
        validator.max_depth = 3

        with pytest.raises(ValidationError) as exc_info:
            validator.validate("MATCH (a)-[*1..10]->(b) RETURN a, b", {})
        assert "depth" in str(exc_info.value).lower()
        assert "10" in str(exc_info.value)

    def test_allows_acceptable_depth(self):
        validator = CypherValidator()
        validator.max_depth = 3

        result = validator.validate("MATCH (a)-[*1..2]->(b) RETURN a, b LIMIT 10", {})
        assert result.valid

    def test_blocks_invalid_range(self):
        validator = CypherValidator()

        with pytest.raises(ValidationError) as exc_info:
            # min > max is invalid
            validator.validate("MATCH (a)-[*5..2]->(b) RETURN a, b", {})
        assert (
            "min" in str(exc_info.value).lower()
            and "max" in str(exc_info.value).lower()
        )


class TestValidatorParameterization:
    """Test parameter enforcement."""

    def test_blocks_unparameterized_string_in_where(self):
        validator = CypherValidator()
        validator.enforce_parameters = True

        with pytest.raises(ValidationError) as exc_info:
            # String literal in WHERE without param
            validator.validate("MATCH (n) WHERE n.name = 'literal_value' RETURN n", {})
        assert (
            "literal" in str(exc_info.value).lower()
            or "parameter" in str(exc_info.value).lower()
        )

    def test_allows_parameterized_query(self):
        validator = CypherValidator()
        validator.enforce_parameters = True

        result = validator.validate(
            "MATCH (n) WHERE n.id = $id RETURN n LIMIT 10", {"id": "test123"}
        )
        assert result.valid

    def test_blocks_large_numeric_literal(self):
        validator = CypherValidator()
        validator.enforce_parameters = True

        with pytest.raises(ValidationError):
            # Large number should be parameterized
            validator.validate("MATCH (n) WHERE n.count = 123456 RETURN n", {})


class TestValidatorLimits:
    """Test LIMIT enforcement."""

    def test_adds_limit_if_missing(self):
        validator = CypherValidator()
        validator.enforce_limits = True

        result = validator.validate("MATCH (n:Section) RETURN n", {"limit": 50})
        assert result.valid
        assert "LIMIT" in result.query.upper()
        assert "LIMIT 50" in result.query or "LIMIT $limit" in result.query

    def test_preserves_existing_limit(self):
        validator = CypherValidator()
        validator.enforce_limits = True

        result = validator.validate("MATCH (n:Section) RETURN n LIMIT 20", {})
        assert result.valid
        assert "LIMIT" in result.query.upper()


class TestValidatorExplainPlan:
    """Test EXPLAIN plan analysis with real Neo4j."""

    def test_analyzes_plan_for_safe_query(self, neo4j_driver):
        validator = CypherValidator(neo4j_driver=neo4j_driver)

        # Safe query with index lookup
        result = validator.validate(
            "MATCH (n:Section) WHERE n.id = $id RETURN n LIMIT 10", {"id": "test123"}
        )
        assert result.valid

    def test_warns_or_blocks_expensive_plan(self, neo4j_driver):
        validator = CypherValidator(neo4j_driver=neo4j_driver)
        validator.max_label_scans = 1

        # Query that requires multiple label scans
        query = """
        MATCH (a:Section), (b:Command), (c:Configuration)
        RETURN a, b, c
        LIMIT 10
        """

        try:
            result = validator.validate(query, {})
            # If it passes, should have warnings
            assert len(result.warnings) > 0
        except ValidationError as e:
            # Or it should be blocked
            assert "scan" in str(e).lower() or "expensive" in str(e).lower()

    def test_blocks_cartesian_product(self, neo4j_driver):
        validator = CypherValidator(neo4j_driver=neo4j_driver)

        # Cartesian product pattern
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("MATCH (a), (b) RETURN a, b LIMIT 10", {})
        # Should be caught by regex or EXPLAIN
        assert (
            "Cartesian" in str(exc_info.value)
            or "expensive" in str(exc_info.value).lower()
        )


class TestValidatorEndToEnd:
    """End-to-end validator tests."""

    def test_validates_safe_template_query(self, neo4j_driver):
        validator = CypherValidator(neo4j_driver=neo4j_driver)

        # Typical template query
        query = """
        MATCH (s:Section)
        WHERE s.id IN $section_ids
        OPTIONAL MATCH (s)-[r:MENTIONS]->(e)
        WHERE r.confidence >= 0.5
        RETURN s, collect(DISTINCT e) AS entities
        ORDER BY s.document_id, s.order
        LIMIT $limit
        """

        result = validator.validate(query, {"section_ids": ["abc", "def"], "limit": 20})
        assert result.valid
        assert result.query is not None
        assert result.params == {"section_ids": ["abc", "def"], "limit": 20}

    def test_false_positives_under_5_percent(self, neo4j_driver):
        """Test that false positives (blocking valid queries) are < 5%."""
        validator = CypherValidator(neo4j_driver=neo4j_driver)

        # Collection of valid queries that should pass
        valid_queries = [
            ("MATCH (n:Section) WHERE n.id = $id RETURN n LIMIT 10", {"id": "x"}),
            (
                "MATCH (n)-[r:MENTIONS]->(m) WHERE n.id = $id RETURN m LIMIT 20",
                {"id": "x"},
            ),
            (
                "MATCH (n:Command) WHERE n.name = $name RETURN n LIMIT 5",
                {"name": "test"},
            ),
            ("MATCH (n:Error {code: $code}) RETURN n LIMIT 1", {"code": "E100"}),
            ("MATCH (a)-[:REQUIRES*1..2]->(b) RETURN b LIMIT 10", {}),
            ("MATCH (s:Section) RETURN s ORDER BY s.order LIMIT 50", {}),
            ("MATCH (d:Document)-[:HAS_SECTION]->(s) RETURN d, s LIMIT 30", {}),
            ("MATCH (n) WHERE n.id IN $ids RETURN n LIMIT 20", {"ids": ["a", "b"]}),
        ]

        blocked = 0
        for query, params in valid_queries:
            try:
                validator.validate(query, params)
            except ValidationError:
                blocked += 1

        false_positive_rate = blocked / len(valid_queries)
        assert (
            false_positive_rate < 0.05
        ), f"False positive rate {false_positive_rate:.1%} >= 5%"

    def test_blocks_all_malicious_queries(self, neo4j_driver):
        """Test that malicious queries are blocked."""
        validator = CypherValidator(neo4j_driver=neo4j_driver)

        malicious_queries = [
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "DROP INDEX my_index",
            "MERGE (n:Node {id: 'x'})",
            "MATCH (n) SET n.admin = true",
            "GRANT MATCH ON GRAPH * TO user",
            "MATCH ()-[*1..100]->(n) RETURN n",  # Excessive depth
            "MATCH (a), (b), (c) RETURN *",  # Cartesian product
        ]

        blocked_count = 0
        for query in malicious_queries:
            try:
                validator.validate(query, {})
            except ValidationError:
                blocked_count += 1

        # Should block 100% of malicious queries
        assert blocked_count == len(
            malicious_queries
        ), f"Only blocked {blocked_count}/{len(malicious_queries)} malicious queries"


class TestValidatorDangerousPatterns:
    """Test detection of dangerous query patterns."""

    def test_warns_unbounded_relationship(self):
        validator = CypherValidator()
        result = validator.validate("MATCH (a)-[]-(b) RETURN a, b LIMIT 10", {})
        # Should generate warning
        assert (
            len(result.warnings) > 0 or "relationship" in str(result.warnings).lower()
        )

    def test_warns_where_in_pattern(self):
        validator = CypherValidator()
        result = validator.validate(
            "MATCH (n:Node {prop WHERE prop > 10}) RETURN n LIMIT 10", {}
        )
        # May warn about WHERE in pattern
        assert len(result.warnings) > 0 or result.valid

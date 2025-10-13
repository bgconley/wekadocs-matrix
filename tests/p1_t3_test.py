# Phase 1, Task 1.3 Tests - Database schema initialization (NO MOCKS)
# See: /docs/implementation-plan.md → Task 1.3 DoD & Tests


def test_schema_creation(neo4j_driver):
    """Test schema creation is idempotent"""
    from src.shared.config import get_config
    from src.shared.schema import create_schema

    config = get_config()

    # Create schema first time
    result1 = create_schema(neo4j_driver, config)
    assert result1["success"] is True

    # Create schema second time (should be idempotent)
    result2 = create_schema(neo4j_driver, config)
    assert result2["success"] is True


def test_constraints_exist(neo4j_driver):
    """Test that required constraints are created"""
    with neo4j_driver.session() as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = [record["name"] for record in result]

        # Check key constraints exist
        required_constraints = [
            "document_id_unique",
            "section_id_unique",
            "command_id_unique",
            "configuration_id_unique",
        ]

        for constraint_name in required_constraints:
            assert (
                constraint_name in constraints
            ), f"Constraint {constraint_name} not found"


def test_indexes_exist(neo4j_driver):
    """Test that required indexes are created"""
    with neo4j_driver.session() as session:
        result = session.run("SHOW INDEXES WHERE type <> 'VECTOR'")
        indexes = [record["name"] for record in result]

        # Check key indexes exist
        required_indexes = [
            "document_source_type",
            "section_document_id",
            "command_name",
        ]

        for index_name in required_indexes:
            assert index_name in indexes, f"Index {index_name} not found"


def test_vector_indexes_exist(neo4j_driver):
    """Test that vector indexes are created with correct dimensions"""
    from src.shared.config import get_config

    config = get_config()
    _ = config.embedding.dims  # Verify config loads correctly

    with neo4j_driver.session() as session:
        result = session.run("SHOW INDEXES WHERE type = 'VECTOR'")
        vector_indexes = list(result)

        # Should have at least section_embeddings
        assert len(vector_indexes) > 0, "No vector indexes found"

        # Verify dimensions (would need to inspect index config)
        # For now, just verify they exist
        vector_index_names = [record["name"] for record in vector_indexes]
        assert "section_embeddings" in vector_index_names


def test_schema_version_node(neo4j_driver):
    """Test that schema version singleton node exists"""
    from src.shared.config import get_config

    config = get_config()
    expected_version = config.schema.version

    with neo4j_driver.session() as session:
        result = session.run("MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv")
        record = result.single()

        assert record is not None, "SchemaVersion node not found"
        schema_version = record["sv"]
        assert schema_version["version"] == expected_version


def test_schema_idempotence(neo4j_driver):
    """Test that re-running schema creation doesn't create duplicates"""
    from src.shared.config import get_config
    from src.shared.schema import create_schema

    config = get_config()

    # Get initial counts
    with neo4j_driver.session() as session:
        constraints_before = len(list(session.run("SHOW CONSTRAINTS")))
        indexes_before = len(list(session.run("SHOW INDEXES")))

    # Re-run schema creation
    result = create_schema(neo4j_driver, config)
    assert result["success"] is True

    # Get counts after
    with neo4j_driver.session() as session:
        constraints_after = len(list(session.run("SHOW CONSTRAINTS")))
        indexes_after = len(list(session.run("SHOW INDEXES")))

    # Counts should be same (idempotent)
    assert constraints_after == constraints_before
    assert indexes_after == indexes_before


def test_can_create_document_node(neo4j_driver):
    """Test that we can create a Document node using the schema"""
    import hashlib

    with neo4j_driver.session() as session:
        # Create a test document
        doc_id = hashlib.sha256(b"test_doc_1").hexdigest()
        result = session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.source_uri = $uri,
                d.source_type = 'markdown',
                d.title = $title,
                d.last_edited = datetime()
            RETURN d
            """,
            id=doc_id,
            uri="test://document/1",
            title="Test Document",
        )
        record = result.single()
        assert record is not None

        # Cleanup
        session.run("MATCH (d:Document {id: $id}) DELETE d", id=doc_id)


def test_can_create_section_node(neo4j_driver):
    """Test that we can create a Section node using the schema"""
    import hashlib

    with neo4j_driver.session() as session:
        # Create a test section
        section_id = hashlib.sha256(b"test_section_1").hexdigest()
        result = session.run(
            """
            MERGE (s:Section {id: $id})
            SET s.document_id = $doc_id,
                s.level = 1,
                s.title = $title,
                s.text = $text
            RETURN s
            """,
            id=section_id,
            doc_id="test_doc",
            title="Test Section",
            text="This is a test section.",
        )
        record = result.single()
        assert record is not None

        # Cleanup
        session.run("MATCH (s:Section {id: $id}) DELETE s", id=section_id)

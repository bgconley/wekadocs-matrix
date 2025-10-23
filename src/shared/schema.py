# Implements Phase 1, Task 1.3 (Database schema initialization)
# See: /docs/spec.md §3 (Data model), §3.3 (IDs, versions, consistency)
# See: /docs/implementation-plan.md → Task 1.3 DoD & Tests
# See: /docs/expert-coder-guidance.md → 1.3 (Schema)
# Schema creation with config-driven vector indexes

from pathlib import Path
from typing import Dict

from neo4j import Driver

from .config import Config
from .observability import get_logger

logger = get_logger(__name__)


def create_schema(driver: Driver, config: Config) -> Dict[str, any]:
    """
    Create Neo4j schema including constraints, indexes, and vector indexes.
    Idempotent - can be run multiple times safely.

    Args:
        driver: Neo4j driver instance
        config: Application configuration

    Returns:
        Dict with status and details
    """
    results = {
        "success": False,
        "constraints_created": 0,
        "indexes_created": 0,
        "vector_indexes_created": 0,
        "schema_version": config.graph_schema.version,
        "errors": [],
    }

    try:
        with driver.session() as session:
            # Step 1: Execute base schema (constraints and regular indexes)
            logger.info("Creating base schema (constraints and indexes)")
            cypher_script_path = (
                Path(__file__).parent.parent.parent
                / "scripts"
                / "neo4j"
                / "create_schema.cypher"
            )

            if not cypher_script_path.exists():
                raise FileNotFoundError(
                    f"Schema script not found: {cypher_script_path}"
                )

            with open(cypher_script_path, "r") as f:
                cypher_script = f.read()

            # Split script into individual statements (simple split on semicolon)
            statements = [
                stmt.strip()
                for stmt in cypher_script.split(";")
                if stmt.strip() and not stmt.strip().startswith("//")
            ]

            for stmt in statements:
                if not stmt:
                    continue
                try:
                    session.run(stmt)
                    if "CREATE CONSTRAINT" in stmt:
                        results["constraints_created"] += 1
                    elif "CREATE INDEX" in stmt and "VECTOR" not in stmt.upper():
                        results["indexes_created"] += 1
                except Exception as e:
                    # Ignore already exists errors
                    if (
                        "already exists" not in str(e).lower()
                        and "equivalent" not in str(e).lower()
                    ):
                        logger.warning(f"Error executing statement: {str(e)[:100]}")

            logger.info(
                "Base schema created",
                constraints=results["constraints_created"],
                indexes=results["indexes_created"],
            )

            # Step 2: Create vector indexes using config-driven dimensions
            logger.info("Creating vector indexes")
            vector_indexes = create_vector_indexes(session, config)
            results["vector_indexes_created"] = vector_indexes["created"]
            results["vector_indexes_details"] = vector_indexes["details"]

            # Step 2.5: Optionally run schema v2.1 (Pre-Phase 7, F2)
            # Only runs if file exists - safe to omit in production
            v2_1_result = apply_schema_v2_1(session)
            if v2_1_result["executed"]:
                results["schema_v2_1_applied"] = True
                results["schema_v2_1_details"] = v2_1_result
                logger.info("Schema v2.1 applied successfully", details=v2_1_result)
            else:
                results["schema_v2_1_applied"] = False
                logger.debug(
                    "Schema v2.1 not applied",
                    reason=v2_1_result.get("reason", "file not found"),
                )

            # Step 3: Verify schema
            logger.info("Verifying schema")
            verification = verify_schema(session)
            results["verification"] = verification
            results["success"] = True

            logger.info("Schema creation completed successfully", results=results)
            return results

    except Exception as e:
        logger.error("Schema creation failed", error=str(e))
        results["errors"].append(str(e))
        return results


def apply_schema_v2_1(session) -> Dict[str, any]:
    """
    Optionally apply schema v2.1 DDL (Pre-Phase 7, F2).

    This function:
    1. Checks if create_schema_v2_1.cypher exists
    2. If exists, executes it (idempotent - safe to re-run)
    3. If not, returns quietly (not an error)

    Purpose: Prepare for Phase 7 without disrupting current operations.
    Changes: Dual-label Sections, add Session/Query/Answer constraints.

    Args:
        session: Neo4j session

    Returns:
        Dict with execution status
    """
    result = {
        "executed": False,
        "dual_labeled_sections": 0,
        "constraints_created": 0,
        "schema_version": None,
    }

    try:
        # Check if v2.1 script exists
        v2_1_script_path = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "neo4j"
            / "create_schema_v2_1.cypher"
        )

        if not v2_1_script_path.exists():
            logger.debug("Schema v2.1 script not found, skipping")
            result["reason"] = "script not found"
            return result

        logger.info("Applying schema v2.1 (Pre-Phase 7 foundation)")

        # Read and execute script
        with open(v2_1_script_path, "r") as f:
            cypher_script = f.read()

        # Split into statements
        statements = [
            stmt.strip()
            for stmt in cypher_script.split(";")
            if stmt.strip()
            and not stmt.strip().startswith("//")
            and not stmt.strip().startswith("--")
        ]

        for stmt in statements:
            if not stmt:
                continue
            try:
                execution_result = session.run(stmt)

                # Track dual-labeling
                if "SET s:Chunk" in stmt:
                    # Get count from execution result
                    summary = execution_result.consume()
                    result["dual_labeled_sections"] = summary.counters.labels_added

                # Track constraint creation
                if "CREATE CONSTRAINT" in stmt:
                    result["constraints_created"] += 1

            except Exception as e:
                # Ignore already exists errors (idempotent)
                if (
                    "already exists" not in str(e).lower()
                    and "equivalent" not in str(e).lower()
                ):
                    logger.warning(f"Error executing v2.1 statement: {str(e)[:100]}")

        # Verify schema version was set
        version_result = session.run(
            "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv.version AS version"
        )
        version_record = version_result.single()
        if version_record:
            result["schema_version"] = version_record["version"]

        result["executed"] = True
        logger.info(
            "Schema v2.1 applied successfully",
            dual_labeled=result["dual_labeled_sections"],
            constraints=result["constraints_created"],
            version=result["schema_version"],
        )

    except Exception as e:
        logger.warning(f"Failed to apply schema v2.1: {str(e)}")
        result["error"] = str(e)

    return result


def create_vector_indexes(session, config: Config) -> Dict[str, any]:
    """
    Create vector indexes with config-driven dimensions and similarity.

    Args:
        session: Neo4j session
        config: Application configuration

    Returns:
        Dict with creation status
    """
    embedding_config = config.embedding
    dims = embedding_config.dims
    similarity = embedding_config.similarity.upper()

    # Map config similarity to Neo4j similarity function
    similarity_map = {
        "COSINE": "cosine",
        "EUCLIDEAN": "euclidean",
        "DOT": "dot",
    }
    neo4j_similarity = similarity_map.get(similarity, "cosine")

    # Define vector indexes to create
    vector_index_definitions = [
        ("section_embeddings", "Section", "vector_embedding"),
        ("command_embeddings", "Command", "vector_embedding"),
        ("configuration_embeddings", "Configuration", "vector_embedding"),
        ("procedure_embeddings", "Procedure", "vector_embedding"),
        ("error_embeddings", "Error", "vector_embedding"),
        ("concept_embeddings", "Concept", "vector_embedding"),
    ]

    results = {
        "created": 0,
        "details": [],
    }

    for index_name, label, property_name in vector_index_definitions:
        try:
            # Check if index already exists
            check_query = """
            SHOW INDEXES
            YIELD name, type
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN count(*) as count
            """
            result = session.run(check_query, index_name=index_name)
            exists = result.single()["count"] > 0

            if exists:
                logger.info(f"Vector index already exists: {index_name}")
                results["details"].append(
                    {
                        "index": index_name,
                        "status": "exists",
                        "label": label,
                        "dims": dims,
                        "similarity": neo4j_similarity,
                    }
                )
                continue

            # Create vector index
            create_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON n.{property_name}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dims},
                    `vector.similarity_function`: '{neo4j_similarity}'
                }}
            }}
            """

            session.run(create_query)
            results["created"] += 1
            results["details"].append(
                {
                    "index": index_name,
                    "status": "created",
                    "label": label,
                    "dims": dims,
                    "similarity": neo4j_similarity,
                }
            )
            logger.info(
                f"Created vector index: {index_name}",
                label=label,
                dims=dims,
                similarity=neo4j_similarity,
            )

        except Exception as e:
            logger.error(f"Failed to create vector index {index_name}", error=str(e))
            results["details"].append(
                {
                    "index": index_name,
                    "status": "error",
                    "error": str(e),
                }
            )

    return results


def verify_schema(session) -> Dict[str, any]:
    """
    Verify schema is correctly created.

    Args:
        session: Neo4j session

    Returns:
        Dict with verification results
    """
    verification = {
        "constraints": [],
        "indexes": [],
        "vector_indexes": [],
        "schema_version": None,
    }

    try:
        # Get constraints
        constraints_result = session.run("SHOW CONSTRAINTS")
        verification["constraints"] = [
            {"name": record["name"], "type": record["type"]}
            for record in constraints_result
        ]

        # Get indexes
        indexes_result = session.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE type <> 'VECTOR'"
        )
        verification["indexes"] = [
            {
                "name": record["name"],
                "type": record["type"],
                "labels": record["labelsOrTypes"],
                "properties": record["properties"],
            }
            for record in indexes_result
        ]

        # Get vector indexes
        vector_indexes_result = session.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE type = 'VECTOR'"
        )
        verification["vector_indexes"] = [
            {
                "name": record["name"],
                "type": record["type"],
                "labels": record["labelsOrTypes"],
                "properties": record["properties"],
            }
            for record in vector_indexes_result
        ]

        # Get schema version
        schema_version_result = session.run(
            "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv"
        )
        sv_record = schema_version_result.single()
        if sv_record:
            verification["schema_version"] = dict(sv_record["sv"])

    except Exception as e:
        logger.error("Schema verification failed", error=str(e))
        verification["error"] = str(e)

    return verification


def drop_schema(driver: Driver) -> Dict[str, any]:
    """
    Drop all schema elements (for testing/cleanup).
    WARNING: This is destructive!

    Args:
        driver: Neo4j driver instance

    Returns:
        Dict with status
    """
    results = {
        "success": False,
        "dropped": [],
    }

    try:
        with driver.session() as session:
            # Drop all constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            for record in constraints_result:
                constraint_name = record["name"]
                session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                results["dropped"].append(f"constraint:{constraint_name}")

            # Drop all indexes
            indexes_result = session.run("SHOW INDEXES")
            for record in indexes_result:
                index_name = record["name"]
                session.run(f"DROP INDEX {index_name} IF EXISTS")
                results["dropped"].append(f"index:{index_name}")

            results["success"] = True
            logger.info("Schema dropped successfully", dropped=results["dropped"])

    except Exception as e:
        logger.error("Failed to drop schema", error=str(e))
        results["error"] = str(e)

    return results

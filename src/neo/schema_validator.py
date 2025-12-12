"""Neo4j schema validation for MCP server startup.

This module validates the Neo4j graph schema at startup to detect
schema drift between ingestion and query layers. Validates:
- Required node labels exist (Document, Section/Chunk)
- Required relationship types exist (HAS_SECTION)
- Document->Section connectivity is established

Enhanced per multi-model Zen analysis (2025-12-03) to include
relationship type validation and connectivity checks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from neo4j import Driver

from src.shared.observability import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaValidationResult:
    """Result of Neo4j schema validation."""

    valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    node_counts: Dict[str, int] = field(default_factory=dict)
    relationship_counts: Dict[str, int] = field(default_factory=dict)
    missing_labels: Set[str] = field(default_factory=set)
    missing_relationships: Set[str] = field(default_factory=set)


# Content labels - at least one must exist with data
CONTENT_LABELS = {"Section", "Chunk"}

# Required labels - must exist with data
REQUIRED_LABELS = {"Document"}

# Required relationship types - must have instances
REQUIRED_RELATIONSHIPS = {"HAS_SECTION"}


def validate_neo4j_schema(
    driver: Driver, *, strict: bool = False
) -> SchemaValidationResult:
    """Validate Neo4j schema at startup.

    Args:
        driver: Neo4j driver instance
        strict: If True, treat warnings as errors

    Returns:
        SchemaValidationResult with validation status and details
    """
    result = SchemaValidationResult(valid=True)

    try:
        with driver.session() as session:
            # Get all label counts
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [r["label"] for r in labels_result]

            for label in labels:
                count_result = session.run(
                    f"MATCH (n:`{label}`) RETURN count(n) AS cnt"
                )
                result.node_counts[label] = count_result.single()["cnt"]

            # Check for content labels (Section or Chunk must exist)
            has_content = bool(CONTENT_LABELS & set(result.node_counts.keys()))
            if not has_content:
                result.errors.append(
                    f"No content labels found. Expected at least one of: {CONTENT_LABELS}"
                )
                result.valid = False
            else:
                content_total = sum(
                    result.node_counts.get(lbl, 0) for lbl in CONTENT_LABELS
                )
                if content_total == 0:
                    result.warnings.append(
                        "Content labels exist but have 0 nodes - database may be empty"
                    )

            # Check required labels
            for label in REQUIRED_LABELS:
                if label not in result.node_counts or result.node_counts[label] == 0:
                    result.errors.append(f"Required label '{label}' missing or empty")
                    result.missing_labels.add(label)
                    result.valid = False

            # Check required relationship types
            for rel_type in REQUIRED_RELATIONSHIPS:
                rel_count = session.run(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS cnt"
                ).single()["cnt"]
                result.relationship_counts[rel_type] = rel_count
                if rel_count == 0:
                    result.warnings.append(
                        f"Relationship type '{rel_type}' has 0 instances - "
                        "cross-document queries may fail"
                    )
                    result.missing_relationships.add(rel_type)

            # Check Document->Chunk connectivity
            connectivity_check = session.run(
                """
                MATCH (d:Document)-[:HAS_SECTION]->(c:Chunk)
                RETURN count(DISTINCT d) AS docs_with_sections
                """
            ).single()
            docs_with_sections = connectivity_check["docs_with_sections"]
            total_docs = result.node_counts.get("Document", 0)

            if docs_with_sections == 0 and total_docs > 0:
                result.warnings.append(
                    f"Found {total_docs} Document nodes but none have "
                    "HAS_SECTION relationships - text retrieval will fail"
                )
            elif docs_with_sections < total_docs and total_docs > 0:
                orphan_docs = total_docs - docs_with_sections
                result.warnings.append(
                    f"{orphan_docs} of {total_docs} Documents have no "
                    "HAS_SECTION relationships"
                )

    except Exception as e:
        result.errors.append(f"Schema validation failed with exception: {e}")
        result.valid = False

    # In strict mode, treat warnings as errors
    if strict and result.warnings:
        result.errors.extend(result.warnings)
        result.valid = False

    # Log results
    if result.valid:
        logger.info(
            "Neo4j schema validation passed",
            node_labels=len(result.node_counts),
            relationship_types=len(result.relationship_counts),
            content_nodes=sum(result.node_counts.get(lbl, 0) for lbl in CONTENT_LABELS),
        )
    else:
        logger.error(
            "Neo4j schema validation FAILED",
            errors=result.errors,
            node_counts=result.node_counts,
        )

    for warning in result.warnings:
        logger.warning("Schema validation warning", warning=warning)

    return result

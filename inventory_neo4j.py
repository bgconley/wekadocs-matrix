import os
from pathlib import Path

from neo4j import GraphDatabase

OUTPUT_FILE = Path("neo4j_schema_dump.cypher")


def get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is not set in the environment")
    return GraphDatabase.driver(uri, auth=(user, password))


def main():
    driver = get_driver()
    constraints: list[str] = []
    indexes: list[str] = []
    schema_version_stmts: list[str] = []

    with driver.session() as session:
        # Constraints
        result = session.run(
            "SHOW CONSTRAINTS YIELD name, createStatement RETURN name, createStatement ORDER BY name"
        )
        for record in result:
            stmt = record["createStatement"]
            if stmt:
                constraints.append(stmt.strip().rstrip(";") + ";")

        # Indexes
        result = session.run(
            "SHOW INDEXES YIELD name, type, createStatement "
            "RETURN name, type, createStatement ORDER BY name"
        )
        for record in result:
            stmt = record["createStatement"]
            if stmt:
                indexes.append(stmt.strip().rstrip(";") + ";")

        # SchemaVersion metadata node(s)
        result = session.run("MATCH (n:SchemaVersion) RETURN properties(n) AS props")
        for record in result:
            props = record["props"] or {}
            # Build a literal map for CREATE statement
            # Best-effort serialization; fall back to string for complex types
            assignments = []
            for key, value in props.items():
                if isinstance(value, str):
                    assignments.append(f"{key}: {value!r}")
                elif isinstance(value, (int, float, bool)):
                    assignments.append(f"{key}: {value!r}")
                else:
                    assignments.append(f"{key}: {str(value)!r}")
            props_literal = ", ".join(assignments)
            schema_version_stmts.append(f"CREATE (:SchemaVersion {{{props_literal}}});")

    # Write out to neo4j_schema_dump.cypher
    lines: list[str] = []
    lines.append("// Auto-generated Neo4j schema dump")
    lines.append("// Constraints")
    lines.extend(constraints or ["// (no constraints found)"])
    lines.append("")
    lines.append("// Indexes")
    lines.extend(indexes or ["// (no indexes found)"])
    lines.append("")
    lines.append("// SchemaVersion metadata")
    lines.extend(schema_version_stmts or ["// (no SchemaVersion nodes found)"])
    lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote schema dump to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repair chunk boundaries to ensure section headings are preserved.

This script backfills the `heading` field inside `Chunk.boundaries_json` by
looking up the corresponding `Section` nodes when headings are missing.

It is intended as a one-time maintenance task to run after deploying the
updated ingestion pipeline that preserves headings in new boundaries payloads.

Usage:
    python scripts/neo4j/fix_boundaries_headings.py

Environment variables:
    NEO4J_URI        (default: bolt://localhost:7687)
    NEO4J_USER       (default: neo4j)
    NEO4J_PASSWORD   (required if authentication enabled)

Requires APOC to be available in the Neo4j instance.
"""

import os
import sys

from neo4j import GraphDatabase

CYPHER = """
CALL {
    MATCH (c:Chunk)
    WITH c, apoc.convert.fromJsonMap(c.boundaries_json) AS bj
    WITH c, bj, CASE WHEN bj.sections IS NOT NULL THEN bj.sections ELSE [] END AS secs
    WITH c, bj, secs
    WHERE size(secs) > 0
    UNWIND range(0, size(secs) - 1) AS i
    WITH c, bj, secs, i,
         coalesce(
             secs[i].heading,
             head([(s:Section {id: secs[i].id}) | s.heading])
         ) AS heading
    WITH c, bj, secs, i, heading
    SET secs[i].heading = coalesce(heading, "")
    WITH c, bj, secs
    SET c.boundaries_json = apoc.convert.toJson({combined: bj.combined, sections: secs})
}
RETURN 'updated' AS status;
"""


def get_env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def main() -> int:
    uri = get_env("NEO4J_URI", "bolt://localhost:7687")
    user = get_env("NEO4J_USER", "neo4j")
    password = get_env("NEO4J_PASSWORD")

    if not password:
        print("NEO4J_PASSWORD must be set", file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            result = session.run(CYPHER)
            summary = result.consume()
            updated = summary.counters.properties_set
            print(f"Completed boundaries heading repair. Properties updated: {updated}")
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

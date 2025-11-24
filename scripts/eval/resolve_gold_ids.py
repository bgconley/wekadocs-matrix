"""
Utility: Resolve human-readable slugs in gold files to canonical document_ids.

Usage:
    python scripts/eval/resolve_gold_ids.py "<slug>"
    python scripts/eval/resolve_gold_ids.py --all docs/sample_ingest/gold_sparse_vs_bm25.yaml
"""

import argparse
from typing import Optional

from neo4j import GraphDatabase


def resolve_slug(slug: str, uri: str, user: str, password: str) -> Optional[str]:
    """Return the document_id whose source_uri contains the slug."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (d:Document)
    WHERE d.source_uri CONTAINS $slug OR d.doc_tag CONTAINS $slug
    RETURN d.id AS id
    LIMIT 1
    """
    with driver.session() as session:
        rec = session.run(query, slug=slug).single()
    driver.close()
    return rec["id"] if rec else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slug", nargs="?", help="Slug to resolve")
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="testpassword123")
    args = parser.parse_args()

    if not args.slug:
        parser.error("Provide a slug to resolve")

    doc_id = resolve_slug(args.slug, args.uri, args.user, args.password)
    if doc_id:
        print(doc_id)
    else:
        print("NOT_FOUND")


if __name__ == "__main__":
    main()

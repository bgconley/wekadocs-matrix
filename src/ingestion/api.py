import hashlib
from typing import Dict

from bs4 import BeautifulSoup
from markdown import markdown

# Minimal ingest: parse markdown->sections, then upsert via simple Cypher.
# Your full parser/extractor can be wired here if already implemented.


def _sections_from_markdown(content: str):
    html = markdown(content, extensions=["extra", "codehilite", "tables", "toc"])
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    current = None
    for el in soup.children:
        if getattr(el, "name", None) in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            if current:
                sections.append(current)
            title = el.get_text().strip()
            sid = hashlib.sha256(title.encode()).hexdigest()
            current = {
                "id": sid,
                "title": title,
                "content": "",
                "checksum": hashlib.sha256(title.encode()).hexdigest(),
            }
        elif current and isinstance(el, type(soup.new_string(""))) and el.string:
            current["content"] += el.string + "\n"
    if current:
        sections.append(current)
    return sections


async def ingest_document(source_uri: str, content: str, fmt: str = "markdown") -> Dict:
    # Choose parser (markdown only for tests here)
    if fmt != "markdown":
        raise ValueError("Only markdown supported in test ingest_document")

    # Deterministic doc id
    doc_id = hashlib.sha256(source_uri.encode()).hexdigest()

    sections = _sections_from_markdown(content)

    # Upsert doc + sections
    from src.shared.connections import get_connection_manager

    manager = get_connection_manager()
    neo4j = manager.get_neo4j_driver()
    qdrant = manager.get_qdrant_client()

    async with neo4j.session() as sess:
        await sess.run(
            """
            MERGE (d:Document {id: $doc})
            SET d.source_uri = $uri, d.updated_at = datetime()
            """,
            {"doc": doc_id, "uri": source_uri},
        )
        await sess.run(
            """
            UNWIND $rows AS row
            MERGE (s:Section {id: row.id})
            SET s.title = row.title,
                s.content = row.content,
                s.checksum = row.checksum,
                s.embedding_version = 'v1',
                s.updated_at = datetime()
            MERGE (d:Document {id: $doc})
            MERGE (d)-[:HAS_SECTION]->(s)
            """,
            {"rows": sections, "doc": doc_id},
        )

    # Minimal vectors (count parity only)
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=s["id"],
            vector=[0.0] * 384,
            payload={"node_id": s["id"], "label": "Section", "embedding_version": "v1"},
        )
        for s in sections
    ]
    qdrant.upsert(collection_name="weka_sections", points=points)

    return {"id": doc_id, "sections": len(sections)}

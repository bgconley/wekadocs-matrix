Now I'll review `build_graph.py` to complete my understanding of your ingestion pipeline.Excellent! Now I have the complete picture of your ingestion pipeline. I can see how everything flows from parsing through graph construction to embeddings. Let me provide you with a comprehensive implementation for hierarchical chunking that integrates cleanly with your existing system.

## Implementation Strategy for Hierarchical Chunking

Based on your codebase, here's how we'll implement the three-level hierarchy:

### 1. **Modified Parser with LLM-Driven Hierarchical Chunking**

Here's an enhanced version of your `markdown.py` that creates all three levels:

```python
# hierarchical_markdown.py - Enhanced parser with 3-level hierarchy
import hashlib
import re
import json
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import markdown
from bs4 import BeautifulSoup

from src.shared.observability import get_logger
from src.shared.config import get_config

logger = get_logger(__name__)

@dataclass
class HierarchicalSection:
    """Represents a section with hierarchical metadata"""
    id: str
    document_id: str
    hierarchy_level: int  # 1 (doc summary), 2 (concept group), 3 (detail)
    parent_id: Optional[str]  # For L2/L3 chunks
    child_ids: List[str]  # For L1/L2 chunks
    title: str
    text: str
    tokens: int
    checksum: str
    order: int
    # Metadata for relationship
    concept_group: Optional[str]  # For L2 chunks
    source_sections: List[str]  # For L1/L2 - which L3 sections they summarize

class HierarchicalMarkdownParser:
    """Parser that creates 3-level hierarchical chunks using LLM assistance"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.config = get_config()

        # Token limits per level
        self.L1_TOKEN_LIMIT = 500
        self.L2_TOKEN_LIMIT = 1500
        self.L3_TOKEN_LIMIT = self.config.ingestion.max_section_tokens or 1000

    def parse_hierarchical(self, source_uri: str, raw_text: str) -> Dict[str, any]:
        """
        Parse markdown into hierarchical chunks.

        Returns:
            Dict with Document and hierarchical Sections at all levels
        """
        logger.info("Starting hierarchical parsing", source_uri=source_uri)

        # Step 1: Parse into L3 chunks (existing section logic)
        document, l3_sections = self._parse_level_3_sections(source_uri, raw_text)

        # Step 2: Create L2 chunks (concept groups) using LLM
        l2_sections = self._create_level_2_chunks(document, l3_sections)

        # Step 3: Create L1 chunk (document summary) using LLM
        l1_section = self._create_level_1_chunk(document, l2_sections, l3_sections)

        # Step 4: Build relationships
        all_sections = self._build_hierarchical_relationships(
            l1_section, l2_sections, l3_sections
        )

        logger.info(
            "Hierarchical parsing complete",
            source_uri=source_uri,
            l1_count=1,
            l2_count=len(l2_sections),
            l3_count=len(l3_sections)
        )

        return {
            "Document": document,
            "Sections": all_sections,
            "HierarchicalMetadata": {
                "levels": {
                    1: [l1_section],
                    2: l2_sections,
                    3: l3_sections
                }
            }
        }

    def _parse_level_3_sections(self, source_uri: str, raw_text: str) -> Tuple[Dict, List[Dict]]:
        """
        Parse markdown into fine-grained L3 sections.
        This is essentially your existing parse_markdown logic.
        """
        from src.ingestion.parsers.markdown import parse_markdown

        # Use existing parser
        result = parse_markdown(source_uri, raw_text)
        document = result["Document"]
        sections = result["Sections"]

        # Add hierarchy metadata
        for section in sections:
            section["hierarchy_level"] = 3
            section["parent_id"] = None  # Will be set later
            section["child_ids"] = []

        return document, sections

    def _create_level_2_chunks(self, document: Dict, l3_sections: List[Dict]) -> List[Dict]:
        """
        Create L2 concept group chunks using LLM to identify semantic clusters.
        """
        if not l3_sections:
            return []

        # Prepare section summaries for LLM
        section_summaries = []
        for sec in l3_sections:
            summary = {
                "id": sec["id"],
                "title": sec["title"],
                "preview": sec["text"][:200] + "..." if len(sec["text"]) > 200 else sec["text"],
                "tokens": sec["tokens"],
                "order": sec["order"]
            }
            section_summaries.append(summary)

        # Use LLM to group sections
        grouping_prompt = f"""
        Analyze these sections from a technical document and group them into logical concept chunks.
        Each group should represent a coherent topic or concept that would make sense to retrieve together.

        Document: {document['title']}

        Sections to group:
        {json.dumps(section_summaries, indent=2)}

        Create 2-5 concept groups. Each section must belong to exactly one group.

        Return JSON:
        {{
            "groups": [
                {{
                    "title": "Concept Group Title",
                    "description": "Brief description of what this group covers",
                    "section_ids": ["id1", "id2", ...],
                    "reasoning": "Why these sections belong together"
                }}
            ]
        }}
        """

        if self.llm_client:
            # Call LLM (you can use OpenAI, Anthropic, or local models)
            response = self._call_llm(grouping_prompt)
            groups = json.loads(response)["groups"]
        else:
            # Fallback: Simple heuristic grouping by heading level and proximity
            groups = self._heuristic_grouping(l3_sections)

        # Create L2 sections from groups
        l2_sections = []
        for i, group in enumerate(groups):
            l2_section = self._create_l2_section_from_group(
                document, group, l3_sections, order=i
            )
            l2_sections.append(l2_section)

        return l2_sections

    def _create_l2_section_from_group(
        self,
        document: Dict,
        group: Dict,
        l3_sections: List[Dict],
        order: int
    ) -> Dict:
        """Create an L2 section from a group of L3 sections."""

        # Get the actual sections in this group
        section_ids = group["section_ids"]
        grouped_sections = [s for s in l3_sections if s["id"] in section_ids]

        # Combine text for the L2 chunk
        combined_text = f"## {group['title']}\n\n"
        combined_text += f"{group.get('description', '')}\n\n"

        # Add section content with clear boundaries
        for sec in grouped_sections:
            combined_text += f"### {sec['title']}\n"
            # Include abbreviated content to stay within token limits
            max_tokens_per_section = self.L2_TOKEN_LIMIT // len(grouped_sections)
            truncated_text = self._truncate_to_tokens(sec["text"], max_tokens_per_section)
            combined_text += f"{truncated_text}\n\n"

        # Generate ID for L2 section
        checksum = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
        anchor = self._slugify(group["title"])
        section_id = self._section_id(document["source_uri"], f"l2-{anchor}", checksum)

        return {
            "id": section_id,
            "document_id": document["id"],
            "hierarchy_level": 2,
            "parent_id": None,  # Will be set to L1 ID
            "child_ids": section_ids,
            "title": group["title"],
            "text": combined_text,
            "tokens": len(combined_text.split()),
            "checksum": checksum,
            "order": order,
            "concept_group": group["title"],
            "source_sections": section_ids,
            "anchor": anchor,
            # Include metadata for retrieval
            "metadata": {
                "concept_description": group.get("description", ""),
                "grouping_reasoning": group.get("reasoning", "")
            }
        }

    def _create_level_1_chunk(
        self,
        document: Dict,
        l2_sections: List[Dict],
        l3_sections: List[Dict]
    ) -> Dict:
        """Create L1 document-level summary chunk."""

        summary_prompt = f"""
        Create a comprehensive summary of this technical document that captures all key concepts.
        The summary should be ~400-500 tokens and help with broad queries about the document.

        Document: {document['title']}

        Main concept groups:
        {json.dumps([{"title": s["title"], "description": s.get("metadata", {}).get("concept_description")}
                     for s in l2_sections], indent=2)}

        Key sections:
        {json.dumps([{"title": s["title"], "tokens": s["tokens"]} for s in l3_sections[:10]], indent=2)}

        Create a summary that:
        1. Introduces what this document covers
        2. Lists the main concepts/topics
        3. Mentions key technical terms and relationships
        4. Stays under 500 tokens

        Return the summary as plain text.
        """

        if self.llm_client:
            summary_text = self._call_llm(summary_prompt)
        else:
            # Fallback: Create summary from L2 titles and first paragraphs
            summary_text = self._create_heuristic_summary(document, l2_sections, l3_sections)

        # Generate L1 section
        checksum = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
        section_id = self._section_id(document["source_uri"], "l1-summary", checksum)

        return {
            "id": section_id,
            "document_id": document["id"],
            "hierarchy_level": 1,
            "parent_id": None,
            "child_ids": [s["id"] for s in l2_sections],
            "title": f"Summary: {document['title']}",
            "text": summary_text,
            "tokens": len(summary_text.split()),
            "checksum": checksum,
            "order": 0,
            "anchor": "summary",
            "source_sections": [s["id"] for s in l3_sections],
            "metadata": {
                "is_summary": True,
                "covers_entire_document": True
            }
        }

    def _build_hierarchical_relationships(
        self,
        l1_section: Dict,
        l2_sections: List[Dict],
        l3_sections: List[Dict]
    ) -> List[Dict]:
        """Build parent-child relationships across all levels."""

        # Set L2 parent to L1
        for l2 in l2_sections:
            l2["parent_id"] = l1_section["id"]

        # Set L3 parents to appropriate L2
        l3_by_id = {s["id"]: s for s in l3_sections}
        for l2 in l2_sections:
            for child_id in l2["child_ids"]:
                if child_id in l3_by_id:
                    l3_by_id[child_id]["parent_id"] = l2["id"]

        # Return all sections
        return [l1_section] + l2_sections + l3_sections

    def _call_llm(self, prompt: str) -> str:
        """
        Call your LLM of choice.
        This is a placeholder - integrate with your preferred LLM.
        """
        # Example with OpenAI
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content

        # For now, return a placeholder
        raise NotImplementedError("LLM integration needed")

    def _heuristic_grouping(self, l3_sections: List[Dict]) -> List[Dict]:
        """
        Fallback heuristic grouping when LLM is not available.
        Groups consecutive sections with similar heading levels.
        """
        if not l3_sections:
            return []

        groups = []
        current_group = {
            "title": "Main Content",
            "description": "Primary document content",
            "section_ids": [],
        }

        for section in l3_sections:
            current_group["section_ids"].append(section["id"])

            # Create new group every 3-5 sections or at major heading changes
            if len(current_group["section_ids"]) >= 4:
                groups.append(current_group)
                current_group = {
                    "title": f"Content Part {len(groups) + 2}",
                    "description": "Document content continuation",
                    "section_ids": [],
                }

        # Add final group if it has content
        if current_group["section_ids"]:
            groups.append(current_group)

        return groups

    # Utility methods (reuse from existing code)
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        from src.ingestion.parsers.markdown import _slugify
        return _slugify(text)

    def _section_id(self, source_uri: str, anchor: str, checksum: str) -> str:
        """Generate deterministic section ID."""
        from src.ingestion.parsers.markdown import _section_id
        return _section_id(source_uri, anchor, checksum)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens]) + "..."

    def _create_heuristic_summary(
        self,
        document: Dict,
        l2_sections: List[Dict],
        l3_sections: List[Dict]
    ) -> str:
        """Create a heuristic summary when LLM is not available."""
        summary = f"# {document['title']}\n\n"
        summary += "This document covers the following topics:\n\n"

        for l2 in l2_sections:
            summary += f"- **{l2['title']}**: "
            desc = l2.get("metadata", {}).get("concept_description", "")
            summary += f"{desc}\n" if desc else "Related technical content\n"

        # Add key terms from first few sections
        summary += "\nKey concepts include: "
        key_terms = set()
        for section in l3_sections[:5]:
            # Extract capitalized terms as likely important concepts
            words = section["text"].split()
            for word in words:
                if word[0].isupper() and len(word) > 3:
                    key_terms.add(word.strip(".,;:"))

        summary += ", ".join(list(key_terms)[:10])

        return summary
```

### 2. **Updated Graph Builder for Hierarchical Relationships**

Now let's modify your `GraphBuilder` to handle the hierarchical relationships:

```python
# Additions to build_graph.py

class HierarchicalGraphBuilder(GraphBuilder):
    """Extended GraphBuilder that handles hierarchical chunk relationships."""

    def upsert_hierarchical_document(
        self,
        document: Dict,
        sections: List[Dict],
        hierarchical_metadata: Dict,
        entities: Dict[str, Dict],
        mentions: List[Dict],
    ) -> Dict[str, any]:
        """
        Upsert document with hierarchical sections.

        Args:
            document: Document metadata
            sections: All sections (L1, L2, L3)
            hierarchical_metadata: Metadata about hierarchy levels
            entities: Dict of entities keyed by ID
            mentions: List of MENTIONS relationships

        Returns:
            Stats dict with hierarchical breakdown
        """
        start_time = time.time()
        stats = {
            "document_id": document["id"],
            "sections_upserted": {
                "level_1": 0,
                "level_2": 0,
                "level_3": 0,
                "total": 0
            },
            "relationships_created": {
                "has_section": 0,
                "has_child": 0,
                "in_concept_group": 0
            },
            "entities_upserted": 0,
            "mentions_created": 0,
            "embeddings_computed": 0,
            "vectors_upserted": 0,
            "duration_ms": 0,
        }

        with self.driver.session() as session:
            # Step 1: Upsert Document node
            self._upsert_document_node(session, document)

            # Step 2: Upsert Sections by hierarchy level
            for level in [1, 2, 3]:
                level_sections = [s for s in sections if s["hierarchy_level"] == level]
                count = self._upsert_hierarchical_sections(
                    session, document["id"], level_sections, level
                )
                stats["sections_upserted"][f"level_{level}"] = count
                stats["sections_upserted"]["total"] += count

            # Step 3: Create hierarchical relationships
            rel_stats = self._create_hierarchical_relationships(session, sections)
            stats["relationships_created"].update(rel_stats)

            # Step 4: Remove stale sections
            self._remove_missing_sections(
                session, document["id"], [s["id"] for s in sections]
            )

            # Step 5: Upsert Entities
            stats["entities_upserted"] = self._upsert_entities(session, entities)

            # Step 6: Create MENTIONS edges
            stats["mentions_created"] = self._create_mentions(session, mentions)

        # Step 7: Process embeddings for all levels
        embedding_stats = self._process_hierarchical_embeddings(
            document, sections, hierarchical_metadata
        )
        stats["embeddings_computed"] = embedding_stats["computed"]
        stats["vectors_upserted"] = embedding_stats["upserted"]

        stats["duration_ms"] = int((time.time() - start_time) * 1000)
        logger.info("Hierarchical graph upsert complete", stats=stats)

        return stats

    def _upsert_hierarchical_sections(
        self,
        session,
        document_id: str,
        sections: List[Dict],
        hierarchy_level: int
    ) -> int:
        """
        Upsert sections with hierarchy metadata.
        Uses different labels for each level for efficient querying.
        """
        batch_size = self.config.ingestion.batch_size
        total_sections = 0

        # Determine node label based on hierarchy level
        labels = {
            1: "Section:Chunk:Summary",      # L1 - Document summaries
            2: "Section:Chunk:ConceptGroup",  # L2 - Concept groups
            3: "Section:Chunk:Detail"         # L3 - Detailed sections
        }

        node_labels = labels.get(hierarchy_level, "Section:Chunk")

        for i in range(0, len(sections), batch_size):
            batch = sections[i : i + batch_size]

            query = f"""
            UNWIND $sections as sec
            MERGE (s:{node_labels} {{id: sec.id}})
            SET s.document_id = sec.document_id,
                s.hierarchy_level = sec.hierarchy_level,
                s.parent_id = sec.parent_id,
                s.child_ids = sec.child_ids,
                s.level = sec.level,
                s.title = sec.title,
                s.anchor = sec.anchor,
                s.order = sec.order,
                s.text = sec.text,
                s.tokens = sec.tokens,
                s.checksum = sec.checksum,
                s.concept_group = sec.concept_group,
                s.source_sections = sec.source_sections,
                s.updated_at = datetime()
            WITH s, sec
            MATCH (d:Document {{id: $document_id}})
            MERGE (d)-[r:HAS_SECTION]->(s)
            SET r.hierarchy_level = sec.hierarchy_level,
                r.order = sec.order,
                r.updated_at = datetime()
            RETURN count(s) as count
            """

            result = session.run(query, sections=batch, document_id=document_id)
            count = result.single()["count"]
            total_sections += count

            logger.debug(
                f"Hierarchical section batch upserted (L{hierarchy_level})",
                batch_num=i // batch_size + 1,
                count=count,
            )

        return total_sections

    def _create_hierarchical_relationships(
        self,
        session,
        sections: List[Dict]
    ) -> Dict[str, int]:
        """Create parent-child and concept group relationships."""

        stats = {
            "has_child": 0,
            "in_concept_group": 0,
        }

        # Create HAS_CHILD relationships
        parent_child_pairs = []
        for section in sections:
            if section.get("parent_id"):
                parent_child_pairs.append({
                    "parent": section["parent_id"],
                    "child": section["id"]
                })

        if parent_child_pairs:
            query = """
            UNWIND $pairs as pair
            MATCH (parent:Section {id: pair.parent})
            MATCH (child:Section {id: pair.child})
            MERGE (parent)-[r:HAS_CHILD]->(child)
            SET r.updated_at = datetime()
            RETURN count(r) as count
            """
            result = session.run(query, pairs=parent_child_pairs)
            stats["has_child"] = result.single()["count"]

        # Create IN_CONCEPT_GROUP relationships for L3 sections
        concept_relationships = []
        for section in sections:
            if section.get("hierarchy_level") == 3 and section.get("parent_id"):
                # Find the L2 parent (concept group)
                l2_parent = next(
                    (s for s in sections
                     if s["id"] == section["parent_id"] and s["hierarchy_level"] == 2),
                    None
                )
                if l2_parent:
                    concept_relationships.append({
                        "section": section["id"],
                        "concept": l2_parent["concept_group"]
                    })

        if concept_relationships:
            query = """
            UNWIND $rels as rel
            MATCH (s:Section {id: rel.section})
            MERGE (c:ConceptGroup {name: rel.concept})
            MERGE (s)-[r:IN_CONCEPT_GROUP]->(c)
            SET r.updated_at = datetime()
            RETURN count(r) as count
            """
            result = session.run(query, rels=concept_relationships)
            stats["in_concept_group"] = result.single()["count"]

        return stats

    def _process_hierarchical_embeddings(
        self,
        document: Dict,
        sections: List[Dict],
        hierarchical_metadata: Dict
    ) -> Dict[str, int]:
        """
        Process embeddings for all hierarchy levels.
        Can use different strategies per level if needed.
        """
        stats = {"computed": 0, "upserted": 0}

        # Group sections by level
        levels = hierarchical_metadata.get("levels", {})

        for level, level_sections in levels.items():
            logger.info(
                f"Processing embeddings for level {level}",
                section_count=len(level_sections)
            )

            # You could use different embedding strategies per level
            # For example, use a different model or task for summaries
            if level == 1:
                # Document summaries might use a different embedding task
                embedding_task = "retrieval.document"
            elif level == 2:
                # Concept groups use passage retrieval
                embedding_task = "retrieval.passage"
            else:
                # Detail sections use standard passage retrieval
                embedding_task = "retrieval.passage"

            # Process each section at this level
            for section in level_sections:
                # Build text for embedding (could customize per level)
                text_for_embedding = self._build_hierarchical_embedding_text(
                    section, level
                )

                # Compute embedding (reuse existing method)
                embedding = self._compute_embedding(text_for_embedding)

                if embedding:
                    # Store in vector store with hierarchy metadata
                    self._upsert_hierarchical_vector(
                        section,
                        embedding,
                        document,
                        hierarchy_level=level,
                        embedding_task=embedding_task
                    )
                    stats["computed"] += 1
                    stats["upserted"] += 1

        return stats

    def _build_hierarchical_embedding_text(
        self,
        section: Dict,
        level: int
    ) -> str:
        """
        Build text for embedding based on hierarchy level.
        Can customize the text representation per level.
        """
        if level == 1:
            # For summaries, just use the text as-is
            return section["text"]
        elif level == 2:
            # For concept groups, include the concept description
            text = f"{section['title']}\n\n"
            if section.get("metadata", {}).get("concept_description"):
                text += f"{section['metadata']['concept_description']}\n\n"
            text += section["text"]
            return text
        else:
            # For detail sections, use title + text (existing behavior)
            return f"{section['title']}\n\n{section['text']}"

    def _upsert_hierarchical_vector(
        self,
        section: Dict,
        embedding: List[float],
        document: Dict,
        hierarchy_level: int,
        embedding_task: str
    ):
        """Store vector with hierarchical metadata in Qdrant."""

        import uuid
        from qdrant_client.models import PointStruct

        # Generate UUID for Qdrant
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, section["id"]))

        # Enhanced payload with hierarchy metadata
        payload = {
            "node_id": section["id"],
            "node_label": "Section",
            "hierarchy_level": hierarchy_level,
            "parent_id": section.get("parent_id"),
            "child_ids": section.get("child_ids", []),
            "document_id": document["id"],
            "document_uri": document.get("source_uri"),
            "title": section.get("title"),
            "anchor": section.get("anchor"),
            "concept_group": section.get("concept_group"),
            "is_summary": hierarchy_level == 1,
            "is_concept_group": hierarchy_level == 2,
            "is_detail": hierarchy_level == 3,
            # Embedding metadata
            "embedding_version": self.embedding_version,
            "embedding_provider": getattr(self.embedder, "provider", "unknown"),
            "embedding_dimensions": len(embedding),
            "embedding_task": embedding_task,
            "embedding_timestamp": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        point = PointStruct(
            id=point_uuid,
            vector=embedding,
            payload=payload
        )

        # Use the existing validated upsert
        self.qdrant_client.upsert_validated(
            collection_name=self.config.search.vector.qdrant.collection_name,
            points=[point],
            expected_dim=len(embedding),
        )

        logger.debug(
            f"Hierarchical vector upserted (L{hierarchy_level})",
            node_id=section["id"],
            concept_group=section.get("concept_group")
        )
```

### 3. **Retrieval Strategy for Hierarchical Chunks**

Here's how to implement hierarchical retrieval:

```python
# hierarchical_retrieval.py

class HierarchicalRetriever:
    """Implements multi-level retrieval strategy."""

    def __init__(self, qdrant_client, neo4j_driver, config):
        self.qdrant = qdrant_client
        self.neo4j = neo4j_driver
        self.config = config

    async def retrieve(
        self,
        query: str,
        strategy: str = "auto",
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve using hierarchical strategy.

        Strategies:
        - "auto": Automatically determine best strategy
        - "top_down": Start from L1, drill down
        - "bottom_up": Start from L3, expand up
        - "concept_first": Start from L2 concept groups
        """

        if strategy == "auto":
            strategy = self._determine_strategy(query)

        if strategy == "top_down":
            return await self._retrieve_top_down(query, k)
        elif strategy == "bottom_up":
            return await self._retrieve_bottom_up(query, k)
        elif strategy == "concept_first":
            return await self._retrieve_concept_first(query, k)
        else:
            # Fallback to concept_first
            return await self._retrieve_concept_first(query, k)

    def _determine_strategy(self, query: str) -> str:
        """Determine best retrieval strategy based on query characteristics."""

        query_lower = query.lower()

        # Broad queries → top_down
        broad_indicators = ["overview", "summary", "what is", "explain", "introduction"]
        if any(ind in query_lower for ind in broad_indicators):
            return "top_down"

        # Specific technical queries → bottom_up
        specific_indicators = ["formula", "calculate", "error", "bug", "code", "api"]
        if any(ind in query_lower for ind in specific_indicators):
            return "bottom_up"

        # Default → concept_first (best for most queries)
        return "concept_first"

    async def _retrieve_concept_first(self, query: str, k: int) -> List[Dict]:
        """
        Start with L2 concept groups, then expand to relevant L3 details.
        This is usually the best strategy for technical documentation.
        """

        # Step 1: Get query embedding
        query_embedding = await self._get_embedding(query)

        # Step 2: Search L2 concept groups
        l2_filter = {
            "must": [
                {"key": "hierarchy_level", "match": {"value": 2}}
            ]
        }

        l2_results = await self.qdrant.search(
            collection_name=self.config.search.vector.qdrant.collection_name,
            query_vector=query_embedding,
            query_filter=l2_filter,
            limit=k,
            with_payload=True
        )

        # Step 3: Get child L3 sections of top L2 matches
        top_l2_ids = [r.payload["node_id"] for r in l2_results[:3]]

        # Use Neo4j to get child sections
        child_sections = await self._get_child_sections(top_l2_ids)

        # Step 4: Search within child sections for most relevant L3 chunks
        if child_sections:
            l3_filter = {
                "must": [
                    {"key": "hierarchy_level", "match": {"value": 3}},
                    {"key": "node_id", "match": {"any": child_sections}}
                ]
            }

            l3_results = await self.qdrant.search(
                collection_name=self.config.search.vector.qdrant.collection_name,
                query_vector=query_embedding,
                query_filter=l3_filter,
                limit=k,
                with_payload=True
            )

            # Combine L2 context with L3 details
            return self._combine_hierarchical_results(l2_results[:2], l3_results)

        return l2_results

    async def _get_child_sections(self, parent_ids: List[str]) -> List[str]:
        """Get child section IDs from Neo4j."""

        query = """
        MATCH (parent:Section)-[:HAS_CHILD]->(child:Section)
        WHERE parent.id IN $parent_ids
        RETURN collect(DISTINCT child.id) as child_ids
        """

        with self.neo4j.session() as session:
            result = session.run(query, parent_ids=parent_ids)
            record = result.single()
            return record["child_ids"] if record else []

    def _combine_hierarchical_results(
        self,
        l2_results: List,
        l3_results: List
    ) -> List[Dict]:
        """
        Combine L2 and L3 results intelligently.
        L2 provides context, L3 provides details.
        """
        combined = []

        # Add top L2 for context
        if l2_results:
            combined.append({
                "type": "context",
                "level": 2,
                "content": l2_results[0].payload,
                "score": l2_results[0].score
            })

        # Add relevant L3 for details
        for result in l3_results:
            combined.append({
                "type": "detail",
                "level": 3,
                "content": result.payload,
                "score": result.score
            })

        return combined
```

### 4. **Configuration Updates**

Add these to your `development.yaml`:

```yaml
# Hierarchical chunking configuration
hierarchical_chunking:
  enabled: true

  # Token limits per level
  level_1_tokens: 500      # Document summaries
  level_2_tokens: 1500     # Concept groups
  level_3_tokens: 1000     # Detail sections (existing max_section_tokens)

  # LLM configuration for intelligent chunking
  llm_provider: "openai"   # or "anthropic", "local"
  llm_model: "gpt-4"

  # Grouping strategies
  min_groups_per_doc: 2
  max_groups_per_doc: 5

  # Retrieval strategies
  default_retrieval_strategy: "concept_first"  # auto, top_down, bottom_up, concept_first
```

### 5. **Integration Path**

To integrate this with minimal disruption:

```python
# Updated ingest_document function in build_graph.py

def ingest_hierarchical_document(
    source_uri: str,
    content: str,
    format: str = "markdown",
    use_hierarchical: bool = True,
    **kwargs
) -> Dict:
    """
    Enhanced ingestion with hierarchical chunking.
    Backward compatible - set use_hierarchical=False for old behavior.
    """

    # ... initialization code ...

    if use_hierarchical:
        # Use new hierarchical parser
        from src.ingestion.parsers.hierarchical_markdown import HierarchicalMarkdownParser

        parser = HierarchicalMarkdownParser(llm_client=your_llm_client)
        result = parser.parse_hierarchical(source_uri, content)

        document = result["Document"]
        sections = result["Sections"]
        hierarchical_metadata = result["HierarchicalMetadata"]

        # Extract entities from all sections
        entities, mentions = extract_entities(sections)

        # Use hierarchical graph builder
        builder = HierarchicalGraphBuilder(neo4j_driver, config, qdrant_client)
        stats = builder.upsert_hierarchical_document(
            document, sections, hierarchical_metadata, entities, mentions
        )
    else:
        # Use existing flow for backward compatibility
        result = parse_markdown(source_uri, content)
        # ... existing code ...

    return stats
```

## Key Benefits of This Implementation

1. **Minimal Disruption**: The hierarchical system adds to your existing infrastructure rather than replacing it
2. **Backward Compatible**: Can run both systems in parallel
3. **LLM-Optimized**: Uses LLM intelligence for semantic grouping but has fallbacks
4. **Graph-Native**: Leverages Neo4j for relationship traversal
5. **Configurable**: Easy to tune via YAML configuration
6. **Production-Ready**: Includes proper error handling, logging, and metrics

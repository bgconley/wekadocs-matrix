# RKG Content Handling Architecture
## Full Canonical Preservation - No Truncation

*This document corrects the architectural error of truncating content. The RKG system is designed to store and retrieve FULL canonical research and session context.*

---

## 1. Core Principle: Never Truncate on Storage

**The RKG system stores COMPLETE content. Truncation is ONLY acceptable for:**
- Display/UI rendering (user-configurable)
- API response pagination (with full content accessible)
- Model context windows (handled via chunking with full-document reference)

**Truncation is NEVER acceptable for:**
- Database storage (Qdrant payloads, Neo4j properties)
- Evidence/provenance tracking
- Session transcript ingestion
- Research content capture

---

## 2. Corrected Storage Architecture

### 2.1 Qdrant Storage - Full Content

```python
class ContentStorage:
    """Store full canonical content without truncation."""

    async def store_document(
        self,
        document_id: str,
        content: str,  # FULL content - never truncated
        metadata: dict
    ) -> list[str]:
        """
        Store document with full content preservation.

        For large documents:
        1. Store FULL content in payload
        2. Generate embeddings for chunks
        3. Link chunks to parent document

        Returns: List of chunk IDs (or single doc ID for small docs)
        """
        # Calculate content size
        content_bytes = len(content.encode('utf-8'))

        # Qdrant payload limit is ~1MB per point
        # For larger documents, use chunking strategy
        if content_bytes > 900_000:  # Leave headroom
            return await self._store_chunked(document_id, content, metadata)
        else:
            return await self._store_single(document_id, content, metadata)

    async def _store_single(
        self,
        document_id: str,
        content: str,
        metadata: dict
    ) -> list[str]:
        """Store document as single point with FULL content."""

        # Generate embedding for full content
        # (embedder handles internal chunking/pooling if needed)
        embedding = await self.embedder.embed(content)

        await self.qdrant.upsert(
            collection_name="research_documents",
            points=[{
                "id": document_id,
                "vector": {"dense": embedding},
                "payload": {
                    # FULL CONTENT - NO TRUNCATION
                    "content": content,
                    "content_length": len(content),
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),

                    # Metadata
                    **metadata,
                    "storage_type": "single",
                    "created_at": datetime.now().isoformat()
                }
            }]
        )

        return [document_id]

    async def _store_chunked(
        self,
        document_id: str,
        content: str,
        metadata: dict
    ) -> list[str]:
        """
        Store large document as chunks with FULL content preserved.

        Strategy:
        1. Create parent document node (stores full content reference)
        2. Create chunk points (each with chunk content + parent reference)
        3. Store full content in Neo4j (no size limit on properties)
        """
        # Semantic chunking (not arbitrary character splits)
        chunks = self.chunker.chunk(
            content,
            chunk_size=800,      # Target tokens per chunk
            chunk_overlap=100,   # Overlap for context continuity
            preserve_sentences=True
        )

        chunk_ids = []
        points = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}:chunk:{i}"

            # Generate embedding for this chunk
            embedding = await self.embedder.embed(chunk.text)

            points.append({
                "id": chunk_id,
                "vector": {"dense": embedding},
                "payload": {
                    # FULL chunk content
                    "content": chunk.text,
                    "content_length": len(chunk.text),

                    # Parent reference for full document retrieval
                    "parent_document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_start_char": chunk.start_char,
                    "chunk_end_char": chunk.end_char,

                    # Metadata inherited from parent
                    **metadata,
                    "storage_type": "chunk",
                    "created_at": datetime.now().isoformat()
                }
            })
            chunk_ids.append(chunk_id)

        # Batch upsert chunks
        await self.qdrant.upsert(
            collection_name="research_documents",
            points=points
        )

        # Store FULL content in Neo4j (no practical size limit)
        await self._store_full_content_neo4j(document_id, content, metadata, chunk_ids)

        return chunk_ids

    async def _store_full_content_neo4j(
        self,
        document_id: str,
        content: str,  # FULL content
        metadata: dict,
        chunk_ids: list[str]
    ):
        """Store full document content in Neo4j.

        Neo4j string properties have no practical size limit (tested to 100MB+).
        This ensures full content is always retrievable.
        """
        async with self.neo4j.session() as session:
            await session.run("""
                MERGE (d:Document {id: $id})
                SET d.content = $content,
                    d.content_length = $content_length,
                    d.content_hash = $content_hash,
                    d.chunk_ids = $chunk_ids,
                    d.source_type = $source_type,
                    d.source_url = $source_url,
                    d.title = $title,
                    d.project = $project,
                    d.created_at = datetime()
            """, {
                "id": document_id,
                "content": content,  # FULL CONTENT
                "content_length": len(content),
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "chunk_ids": chunk_ids,
                "source_type": metadata.get("source_type"),
                "source_url": metadata.get("source_url"),
                "title": metadata.get("title"),
                "project": metadata.get("project")
            })
```

### 2.2 Full Content Retrieval

```python
class ContentRetrieval:
    """Retrieve full canonical content."""

    async def get_document(
        self,
        document_id: str,
        include_full_content: bool = True  # Default to FULL content
    ) -> dict:
        """
        Retrieve document with full content.

        For chunked documents, reconstructs from Neo4j or chunks.
        """
        # Try Qdrant first (for single-point documents)
        points = await self.qdrant.retrieve(
            collection_name="research_documents",
            ids=[document_id],
            with_payload=True
        )

        if points:
            payload = points[0].payload

            if payload.get("storage_type") == "chunk":
                # This is a chunk - get parent document
                parent_id = payload.get("parent_document_id")
                return await self._get_full_from_neo4j(parent_id)
            else:
                # Single document - full content is in payload
                return {
                    "id": document_id,
                    "content": payload["content"],  # FULL content
                    "metadata": {
                        k: v for k, v in payload.items()
                        if k != "content"
                    }
                }

        # Fallback to Neo4j for chunked documents
        return await self._get_full_from_neo4j(document_id)

    async def _get_full_from_neo4j(self, document_id: str) -> dict:
        """Get full document content from Neo4j."""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (d:Document {id: $id})
                RETURN d.content as content,
                       d.content_length as content_length,
                       d.source_type as source_type,
                       d.source_url as source_url,
                       d.title as title,
                       d.project as project,
                       d.created_at as created_at
            """, {"id": document_id})

            record = await result.single()

            if record:
                return {
                    "id": document_id,
                    "content": record["content"],  # FULL content from Neo4j
                    "metadata": {
                        "content_length": record["content_length"],
                        "source_type": record["source_type"],
                        "source_url": record["source_url"],
                        "title": record["title"],
                        "project": record["project"],
                        "created_at": record["created_at"]
                    }
                }

        return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        return_full_content: bool = True  # Default to FULL
    ) -> list[dict]:
        """
        Search with full content retrieval.

        Search hits chunks, but returns full parent documents.
        """
        # Generate query embedding
        query_embedding = await self.embedder.embed(query, input_type="query")

        # Search (may hit chunks)
        results = await self.qdrant.search(
            collection_name="research_documents",
            query_vector=("dense", query_embedding),
            limit=limit * 2,  # Overfetch to handle chunk deduplication
            with_payload=True
        )

        # Deduplicate chunks to parent documents
        seen_parents = set()
        documents = []

        for result in results:
            payload = result.payload

            # Get parent ID (self for single docs, parent for chunks)
            if payload.get("storage_type") == "chunk":
                parent_id = payload.get("parent_document_id")
            else:
                parent_id = result.id

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            if return_full_content:
                # Fetch FULL content
                full_doc = await self.get_document(parent_id)
                documents.append({
                    "id": parent_id,
                    "score": result.score,
                    "content": full_doc["content"],  # FULL content
                    "metadata": full_doc["metadata"],
                    "matched_chunk": payload.get("content") if payload.get("storage_type") == "chunk" else None
                })
            else:
                # Return reference only (for listing)
                documents.append({
                    "id": parent_id,
                    "score": result.score,
                    "title": payload.get("title"),
                    "source_url": payload.get("source_url"),
                    "content_length": payload.get("content_length")
                })

            if len(documents) >= limit:
                break

        return documents
```

---

## 3. Corrected Session Transcript Storage

```python
class SessionTranscriptStorage:
    """Store COMPLETE session transcripts from Claude Code / OpenAI Codex."""

    async def ingest_session_file(
        self,
        filepath: str,
        interface: str  # "claude_code", "openai_codex"
    ) -> dict:
        """
        Ingest complete session file with NO truncation.

        Session files can be large (100MB+). We store everything.
        """
        # Read FULL file content
        with open(filepath, 'r') as f:
            content = f.read()  # COMPLETE file

        # Parse based on format
        if filepath.endswith('.jsonl'):
            entries = self._parse_jsonl(content)
        else:
            entries = self._parse_json(content)

        session_id = self._extract_session_id(filepath)

        # Store each entry with FULL content
        stored_count = 0
        for entry in entries:
            await self._store_session_entry(
                session_id=session_id,
                entry=entry,  # COMPLETE entry
                interface=interface
            )
            stored_count += 1

        # Store session metadata
        await self._store_session_metadata(
            session_id=session_id,
            filepath=filepath,
            interface=interface,
            entry_count=stored_count,
            total_size=len(content)
        )

        return {
            "session_id": session_id,
            "entries_stored": stored_count,
            "total_bytes": len(content)
        }

    async def _store_session_entry(
        self,
        session_id: str,
        entry: dict,  # COMPLETE entry - no truncation
        interface: str
    ):
        """Store a single session entry with full content."""

        # Extract content based on entry type
        if "content" in entry:
            content = entry["content"]  # FULL content
        elif "text" in entry:
            content = entry["text"]  # FULL text
        elif "message" in entry:
            content = entry["message"]  # FULL message
        else:
            content = json.dumps(entry)  # FULL JSON

        # Generate entry ID
        entry_id = hashlib.sha256(
            f"{session_id}:{entry.get('timestamp', '')}:{len(content)}".encode()
        ).hexdigest()[:24]

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Store in Qdrant with FULL content
        await self.qdrant.upsert(
            collection_name="research_documents",
            points=[{
                "id": entry_id,
                "vector": {"dense": embedding},
                "payload": {
                    "content": content,  # FULL CONTENT - NO TRUNCATION
                    "content_length": len(content),
                    "source_type": "session_transcript",
                    "session_id": session_id,
                    "interface": interface,
                    "entry_type": entry.get("type", "unknown"),
                    "role": entry.get("role"),
                    "timestamp": entry.get("timestamp"),
                    "raw_entry": json.dumps(entry),  # FULL original entry
                    "created_at": datetime.now().isoformat()
                }
            }]
        )

        # Store in Neo4j with FULL content
        async with self.neo4j.session() as session:
            await session.run("""
                MERGE (e:SessionEntry {id: $id})
                SET e.content = $content,
                    e.content_length = $content_length,
                    e.entry_type = $entry_type,
                    e.role = $role,
                    e.timestamp = $timestamp,
                    e.raw_entry = $raw_entry

                WITH e
                MERGE (s:Session {id: $session_id})
                MERGE (s)-[:CONTAINS]->(e)
            """, {
                "id": entry_id,
                "content": content,  # FULL CONTENT
                "content_length": len(content),
                "entry_type": entry.get("type", "unknown"),
                "role": entry.get("role"),
                "timestamp": entry.get("timestamp"),
                "raw_entry": json.dumps(entry),  # FULL ENTRY
                "session_id": session_id
            })
```

---

## 4. Corrected Evidence Storage

```python
class EvidenceStorage:
    """Store COMPLETE evidence for relationships and insights."""

    async def store_relationship(
        self,
        subject_id: str,
        object_id: str,
        relationship_type: str,
        evidence: str,  # FULL evidence text - never truncated
        source_document_id: str,
        confidence: float
    ):
        """Store relationship with COMPLETE evidence."""

        async with self.neo4j.session() as session:
            await session.run("""
                MATCH (s:Entity {canonical_id: $subj_id})
                MATCH (o:Entity {canonical_id: $obj_id})

                MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(o)

                // Store FULL evidence - never truncate
                ON CREATE SET
                    r.created_at = datetime(),
                    r.evidence = [$evidence],
                    r.evidence_sources = [$source_doc],
                    r.confidence = $confidence
                ON MATCH SET
                    r.evidence = r.evidence + $evidence,
                    r.evidence_sources = r.evidence_sources + $source_doc,
                    r.confidence = CASE
                        WHEN $confidence > r.confidence
                        THEN $confidence
                        ELSE r.confidence
                    END
            """, {
                "subj_id": subject_id,
                "obj_id": object_id,
                "rel_type": relationship_type,
                "evidence": evidence,  # FULL evidence - NO truncation
                "source_doc": source_document_id,
                "confidence": confidence
            })

    async def store_insight(
        self,
        insight_id: str,
        content: str,  # FULL insight content
        insight_type: str,
        source_evidence: list[dict],  # FULL evidence for each source
        confidence: float
    ):
        """Store insight with COMPLETE evidence trail."""

        # Generate embedding for FULL content
        embedding = await self.embedder.embed(content)

        # Store in Qdrant
        await self.qdrant.upsert(
            collection_name="research_documents",
            points=[{
                "id": insight_id,
                "vector": {"dense": embedding},
                "payload": {
                    "content": content,  # FULL content
                    "content_length": len(content),
                    "source_type": "insight",
                    "insight_type": insight_type,
                    "confidence": confidence,
                    # Store FULL evidence for each source
                    "source_evidence": [
                        {
                            "source_id": ev["source_id"],
                            "evidence_text": ev["text"],  # FULL text
                            "relevance": ev.get("relevance", 1.0)
                        }
                        for ev in source_evidence
                    ],
                    "created_at": datetime.now().isoformat()
                }
            }]
        )

        # Store in Neo4j with FULL evidence
        async with self.neo4j.session() as session:
            await session.run("""
                CREATE (i:Insight {id: $id})
                SET i.content = $content,
                    i.content_length = $content_length,
                    i.type = $type,
                    i.confidence = $confidence,
                    i.created_at = datetime()

                WITH i
                UNWIND $evidence as ev
                MATCH (d:Document {id: ev.source_id})
                CREATE (i)-[r:DERIVED_FROM]->(d)
                SET r.evidence_text = ev.text,
                    r.relevance = ev.relevance
            """, {
                "id": insight_id,
                "content": content,  # FULL content
                "content_length": len(content),
                "type": insight_type,
                "confidence": confidence,
                "evidence": [
                    {
                        "source_id": ev["source_id"],
                        "text": ev["text"],  # FULL evidence text
                        "relevance": ev.get("relevance", 1.0)
                    }
                    for ev in source_evidence
                ]
            })
```

---

## 5. Handling Model Context Windows

For operations that require LLM processing (reranking, contradiction detection), use proper strategies instead of arbitrary truncation:

```python
class ModelInputHandler:
    """Handle model context window limits without losing information."""

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def prepare_for_reranking(
        self,
        query: str,
        documents: list[dict]
    ) -> list[dict]:
        """
        Prepare documents for reranking without losing information.

        Strategy: If document exceeds token limit, use most relevant chunk
        but preserve reference to full document.
        """
        prepared = []

        for doc in documents:
            content = doc["content"]
            token_count = len(self.tokenizer.encode(content))

            if token_count <= self.max_tokens:
                # Full content fits
                prepared.append({
                    "id": doc["id"],
                    "text": content,  # FULL content
                    "is_truncated": False
                })
            else:
                # Need to select best segment for reranking
                # But STORE full content, just use segment for ranking
                best_segment = self._find_best_segment(query, content)
                prepared.append({
                    "id": doc["id"],
                    "text": best_segment,
                    "is_truncated": True,
                    "full_content_available": True,
                    "full_token_count": token_count
                })

        return prepared

    def _find_best_segment(self, query: str, content: str) -> str:
        """Find the most relevant segment of content for a query."""
        # Semantic chunking
        chunks = self._semantic_chunk(content, target_tokens=self.max_tokens)

        # Quick similarity scoring to find best chunk
        query_embedding = self._quick_embed(query)

        best_chunk = None
        best_score = -1

        for chunk in chunks:
            chunk_embedding = self._quick_embed(chunk)
            score = self._cosine_similarity(query_embedding, chunk_embedding)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk

    def prepare_for_contradiction_check(
        self,
        claim: str,
        reference_documents: list[dict]
    ) -> list[dict]:
        """
        Prepare documents for contradiction checking.

        Strategy: Prioritize sections most relevant to the claim,
        but preserve full document for detailed analysis if needed.
        """
        prepared = []

        for doc in reference_documents:
            content = doc["content"]

            # Find paragraphs most relevant to the claim
            relevant_sections = self._extract_relevant_sections(
                claim=claim,
                content=content,
                max_sections=5
            )

            prepared.append({
                "id": doc["id"],
                "relevant_sections": relevant_sections,
                "full_content_id": doc["id"],  # Reference for full retrieval
                "section_count": len(relevant_sections)
            })

        return prepared
```

---

## 6. API Response Handling

For API responses, support pagination rather than truncation:

```python
class APIResponseHandler:
    """Handle API responses with pagination, not truncation."""

    async def search_response(
        self,
        results: list[dict],
        include_full_content: bool = True,
        page: int = 1,
        page_size: int = 10
    ) -> dict:
        """
        Build search response with optional content inclusion.

        Never truncates stored content - client chooses what to receive.
        """
        # Paginate results
        start = (page - 1) * page_size
        end = start + page_size
        page_results = results[start:end]

        response_results = []
        for result in page_results:
            item = {
                "id": result["id"],
                "score": result["score"],
                "title": result.get("title"),
                "source_url": result.get("source_url"),
                "content_length": result.get("content_length"),
            }

            if include_full_content:
                # Return FULL content - never truncated
                item["content"] = result["content"]
            else:
                # Content available on request
                item["content_available"] = True
                item["content_endpoint"] = f"/documents/{result['id']}/content"

            response_results.append(item)

        return {
            "results": response_results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_results": len(results),
                "total_pages": (len(results) + page_size - 1) // page_size,
                "has_next": end < len(results),
                "has_prev": page > 1
            },
            "content_included": include_full_content
        }
```

---

## 7. Summary of Corrections

| Location | Wrong (Truncated) | Correct (Full) |
|----------|------------------|----------------|
| Qdrant payload | `content[:500]` | `content` (full) |
| Neo4j properties | `evidence[:300]` | `evidence` (full) |
| Search results | `content[:2000]` | Full + pagination |
| Session entries | `text[:1000]` | Full transcript |
| Relationship evidence | `source_sentence[:500]` | Full sentence/paragraph |
| Insight sources | Truncated quotes | Full source text |
| API responses | Arbitrary cuts | Pagination + full access |

**Core principle**: The database stores EVERYTHING. The API/UI decides how much to display, with full content always accessible.

---

## 8. Storage Capacity Planning

With full content storage, plan for:

| Content Type | Avg Size | Storage/1000 docs |
|--------------|----------|-------------------|
| Search results | 2KB | 2MB |
| Scraped pages | 50KB | 50MB |
| Session entries | 5KB | 5MB |
| Full sessions | 500KB | 500MB |
| Insights | 2KB | 2MB |

**Recommendations**:
- Qdrant: Use disk-backed collections for large deployments
- Neo4j: Enable page cache sizing based on content volume
- Consider blob storage (S3/MinIO) for very large documents with ID references

---

*This architecture ensures NO research or session context is ever lost to truncation.*

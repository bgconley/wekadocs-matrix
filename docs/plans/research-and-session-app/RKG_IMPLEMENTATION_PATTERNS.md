# RKG System - Implementation Patterns & Pseudocode

## 1. Hybrid Search Implementation

### 1.1 Search Pipeline Pseudocode

```python
async def hybrid_search(
    query: str,
    mode: SearchMode,
    filters: SearchFilters,
    limit: int,
    use_reranker: bool
) -> List[SearchResult]:
    """
    Execute hybrid search combining dense vectors, sparse vectors, and graph context.

    Pipeline:
    1. Generate query embedding (dense)
    2. Generate sparse query vector (BM25 weights)
    3. Execute parallel searches
    4. Fuse results using RRF
    5. Optionally rerank with Voyage
    6. Enrich with graph context
    """

    # Step 1: Generate embeddings
    dense_embedding = await embedding_provider.embed_text(query, input_type="query")
    sparse_vector = generate_bm25_sparse_vector(query)

    # Step 2: Build Qdrant filter
    qdrant_filter = build_qdrant_filter(filters)

    # Step 3: Execute searches based on mode
    if mode == SearchMode.HYBRID:
        # Parallel execution of both search types
        dense_results, sparse_results = await asyncio.gather(
            qdrant.search(
                collection="research_documents",
                query_vector=("dense", dense_embedding.vector),
                limit=limit * 2,  # Over-fetch for fusion
                query_filter=qdrant_filter
            ),
            qdrant.search(
                collection="research_documents",
                query_vector=("sparse", sparse_vector),
                limit=limit * 2,
                query_filter=qdrant_filter
            )
        )

        # Step 4: Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=60  # RRF constant
        )
        candidates = fused_results[:limit * 2]

    elif mode == SearchMode.SEMANTIC:
        candidates = await qdrant.search(
            collection="research_documents",
            query_vector=("dense", dense_embedding.vector),
            limit=limit * 2,
            query_filter=qdrant_filter
        )

    else:  # LEXICAL
        candidates = await qdrant.search(
            collection="research_documents",
            query_vector=("sparse", sparse_vector),
            limit=limit * 2,
            query_filter=qdrant_filter
        )

    # Step 5: Optional reranking
    if use_reranker and len(candidates) > 0:
        documents = [c.payload["content"] for c in candidates]
        rerank_results = await reranker_provider.rerank(
            query=query,
            documents=documents,
            top_k=limit
        )

        # Reorder candidates based on rerank scores
        reranked_candidates = []
        for r in rerank_results:
            candidate = candidates[r.index]
            candidate.score = r.relevance_score
            reranked_candidates.append(candidate)
        candidates = reranked_candidates
    else:
        candidates = candidates[:limit]

    # Step 6: Enrich with graph context
    enriched_results = await enrich_with_graph_context(candidates)

    return enriched_results


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each list where item appears

    Higher k values give more weight to lower-ranked items.
    """
    scores = {}  # doc_id -> cumulative RRF score
    docs = {}    # doc_id -> document

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            doc_id = result.id
            rrf_score = 1.0 / (k + rank)

            if doc_id in scores:
                scores[doc_id] += rrf_score
            else:
                scores[doc_id] = rrf_score
                docs[doc_id] = result

    # Sort by cumulative RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Return sorted results with updated scores
    return [
        SearchResult(
            id=doc_id,
            score=scores[doc_id],
            payload=docs[doc_id].payload
        )
        for doc_id in sorted_ids
    ]


async def enrich_with_graph_context(
    candidates: List[SearchResult]
) -> List[EnrichedSearchResult]:
    """
    Add graph context to search results.

    For each result:
    - Find related documents (2-hop neighbors)
    - Find associated entities
    - Find connected insights
    - Get session context
    """
    doc_ids = [c.payload["document_id"] for c in candidates]

    # Batch query Neo4j for efficiency
    graph_context = await neo4j.query("""
        UNWIND $doc_ids AS docId
        MATCH (d:Document {id: docId})
        OPTIONAL MATCH (d)-[:FROM_SOURCE]->(s:Source)
        OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (d)<-[:DERIVED_FROM]-(i:Insight)
        OPTIONAL MATCH (d)<-[:CAPTURED]-(sess:Session)
        OPTIONAL MATCH (d)-[:RELATED_TO]-(related:Document)
        RETURN docId,
               s.domain as source,
               collect(DISTINCT e.name) as entities,
               collect(DISTINCT {id: i.id, type: i.insight_type, content: i.content}) as insights,
               sess.id as session_id,
               collect(DISTINCT {id: related.id, title: related.title})[0..3] as related_docs
    """, {"doc_ids": doc_ids})

    # Merge graph context with search results
    context_map = {row["docId"]: row for row in graph_context}

    enriched = []
    for candidate in candidates:
        doc_id = candidate.payload["document_id"]
        ctx = context_map.get(doc_id, {})

        enriched.append(EnrichedSearchResult(
            id=candidate.id,
            score=candidate.score,
            payload=candidate.payload,
            source_domain=ctx.get("source"),
            entities=ctx.get("entities", []),
            insights=ctx.get("insights", []),
            session_id=ctx.get("session_id"),
            related_documents=ctx.get("related_docs", [])
        ))

    return enriched
```

### 1.2 BM25 Sparse Vector Generation

```python
import math
from collections import Counter
from typing import Dict, List, Tuple
import re

class BM25SparseVectorizer:
    """
    Generate BM25-weighted sparse vectors for hybrid search.

    This implementation pre-computes corpus statistics and generates
    sparse vectors compatible with Qdrant's sparse vector format.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        vocab: Optional[Dict[str, int]] = None
    ):
        self.k1 = k1
        self.b = b
        self.vocab = vocab or {}  # token -> index
        self.idf = {}  # token -> IDF score
        self.avg_doc_len = 0.0
        self.doc_count = 0

    def fit(self, documents: List[str]):
        """
        Compute corpus statistics for IDF calculation.
        """
        doc_freqs = Counter()  # How many docs contain each term
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            total_len += len(tokens)

            for token in unique_tokens:
                doc_freqs[token] += 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.doc_count = len(documents)
        self.avg_doc_len = total_len / max(self.doc_count, 1)

        # Compute IDF scores
        for token, df in doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[token] = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1
            )

    def transform(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Transform text to sparse vector (indices, values).

        Returns:
            Tuple of (indices, values) for Qdrant sparse vector format
        """
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)

        indices = []
        values = []

        for token, freq in tf.items():
            if token not in self.vocab:
                continue

            idx = self.vocab[token]
            idf = self.idf.get(token, 0.0)

            # BM25 term frequency normalization
            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1))
                )
            )

            weight = idf * tf_norm

            if weight > 0:
                indices.append(idx)
                values.append(weight)

        return indices, values

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - can be enhanced with proper NLP."""
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text)
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]

    def to_qdrant_sparse(self, text: str) -> Dict:
        """
        Generate Qdrant-compatible sparse vector.
        """
        indices, values = self.transform(text)
        return {
            "indices": indices,
            "values": values
        }
```

---

## 2. Document Ingestion Pipeline

### 2.1 Content Processing Flow

```python
async def ingest_document(
    content: str,
    metadata: DocumentMetadata,
    storage: HybridStorage,
    embedder: EmbeddingProvider,
    sparse_vectorizer: BM25SparseVectorizer
) -> str:
    """
    Full pipeline for ingesting a document into both Qdrant and Neo4j.

    Steps:
    1. Generate content hash for deduplication
    2. Check for existing document
    3. Chunk the content
    4. Generate embeddings for each chunk
    5. Generate sparse vectors
    6. Store in Qdrant
    7. Create/update Neo4j nodes and relationships
    """

    # Step 1: Content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Step 2: Check deduplication
    existing = await storage.neo4j.find_by_hash(content_hash)
    if existing:
        logger.info(f"Document already exists: {existing.id}")
        return existing.id

    # Step 3: Chunk content
    chunks = chunk_text(
        content,
        chunk_size=512,
        chunk_overlap=50,
        chunking_strategy="semantic"  # or "fixed", "sentence"
    )

    # Step 4: Generate dense embeddings
    embeddings = await embedder.embed_batch(
        [chunk.text for chunk in chunks],
        input_type="document"
    )

    # Step 5: Generate sparse vectors
    sparse_vectors = [
        sparse_vectorizer.to_qdrant_sparse(chunk.text)
        for chunk in chunks
    ]

    # Step 6: Store in Qdrant
    document_id = str(uuid.uuid4())
    qdrant_points = []

    for i, (chunk, embedding, sparse) in enumerate(
        zip(chunks, embeddings, sparse_vectors)
    ):
        point_id = str(uuid.uuid4())

        qdrant_points.append({
            "id": point_id,
            "vector": {
                "dense": embedding.vector,
                "sparse": sparse
            },
            "payload": {
                "document_id": document_id,
                "session_id": metadata.session_id,
                "project": metadata.project,
                "source_type": metadata.source_type,
                "source_url": metadata.source_url,
                "title": metadata.title,
                "content": chunk.text,
                "content_hash": content_hash,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": datetime.utcnow().isoformat(),
                "agentic_interface": metadata.agentic_interface,
                "tags": metadata.tags or [],
                "metadata": metadata.extra
            }
        })

    await storage.qdrant.upsert(
        collection_name="research_documents",
        points=qdrant_points
    )

    # Step 7: Create Neo4j nodes and relationships
    await storage.neo4j.execute("""
        // Create Document node
        CREATE (d:Document {
            id: $document_id,
            title: $title,
            url: $url,
            source_type: $source_type,
            content_preview: $preview,
            content_hash: $content_hash,
            word_count: $word_count,
            created_at: datetime(),
            qdrant_ids: $qdrant_ids
        })

        // Create or merge Source
        MERGE (s:Source {domain: $domain})
        ON CREATE SET s.name = $domain, s.reliability_score = 0.5
        ON MATCH SET s.last_accessed = datetime()
        CREATE (d)-[:FROM_SOURCE]->(s)

        // Create or merge Session
        MERGE (sess:Session {id: $session_id})
        CREATE (sess)-[:CAPTURED]->(d)

        // Create or merge Project
        MERGE (p:Project {name: $project})
        MERGE (sess)-[:BELONGS_TO]->(p)

        // Create Tags
        FOREACH (tag IN $tags |
            MERGE (t:Tag {name: tag})
            CREATE (d)-[:TAGGED_WITH]->(t)
        )

        RETURN d.id
    """, {
        "document_id": document_id,
        "title": metadata.title,
        "url": metadata.source_url,
        "source_type": metadata.source_type,
        "preview": content[:500],
        "content_hash": content_hash,
        "word_count": len(content.split()),
        "qdrant_ids": [p["id"] for p in qdrant_points],
        "domain": extract_domain(metadata.source_url),
        "session_id": metadata.session_id,
        "project": metadata.project,
        "tags": metadata.tags or []
    })

    # Extract and link entities (async background task)
    asyncio.create_task(
        extract_and_link_entities(document_id, content, storage)
    )

    return document_id


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: str = "semantic"
) -> List[Chunk]:
    """
    Split text into chunks for embedding.

    Strategies:
    - fixed: Fixed character-based chunks
    - sentence: Sentence boundary-aware
    - semantic: Paragraph/section aware
    """
    if chunking_strategy == "fixed":
        return _fixed_chunking(text, chunk_size, chunk_overlap)

    elif chunking_strategy == "sentence":
        return _sentence_chunking(text, chunk_size, chunk_overlap)

    else:  # semantic
        return _semantic_chunking(text, chunk_size, chunk_overlap)


def _semantic_chunking(
    text: str,
    target_size: int,
    overlap: int
) -> List[Chunk]:
    """
    Chunk by semantic boundaries (paragraphs, sections).
    """
    # Split by double newlines (paragraphs) or headers
    sections = re.split(r'\n\n+|(?=^#{1,3}\s)', text, flags=re.MULTILINE)

    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_size = len(section.split())

        if current_size + section_size > target_size and current_chunk:
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=len(chunks),
                metadata={}
            ))

            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > 1:
                # Keep last part for overlap
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text, section]
                current_size = len(overlap_text.split()) + section_size
            else:
                current_chunk = [section]
                current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size

    # Add final chunk
    if current_chunk:
        chunks.append(Chunk(
            text='\n\n'.join(current_chunk),
            start_idx=len(chunks),
            metadata={}
        ))

    return chunks
```

---

## 3. Session Transcript Ingestion

### 3.1 Batch Import Pipeline

```python
async def import_all_sessions(
    storage: HybridStorage,
    embedder: EmbeddingProvider,
    sparse_vectorizer: BM25SparseVectorizer,
    claude_only: bool = False,
    codex_only: bool = False,
    since: Optional[datetime] = None
):
    """
    Import all historical sessions from Claude Code and OpenAI Codex.
    """
    stats = {
        "sessions_processed": 0,
        "documents_imported": 0,
        "research_items_found": 0,
        "errors": []
    }

    # Import Claude Code sessions
    if not codex_only:
        claude_parser = ClaudeSessionParser()
        for session_path in claude_parser.find_sessions():
            try:
                session = claude_parser.parse_session(session_path)

                # Skip if before cutoff date
                if since and session.started_at and session.started_at < since:
                    continue

                result = await import_claude_session(
                    session, storage, embedder, sparse_vectorizer
                )

                stats["sessions_processed"] += 1
                stats["documents_imported"] += result["documents"]
                stats["research_items_found"] += result["research_items"]

            except Exception as e:
                stats["errors"].append({
                    "session": str(session_path),
                    "error": str(e)
                })

    # Import OpenAI Codex sessions
    if not claude_only:
        codex_parser = CodexSessionParser()
        for session_path in codex_parser.find_sessions():
            try:
                session = codex_parser.parse_session(session_path)

                if since and session.started_at and session.started_at < since:
                    continue

                result = await import_codex_session(
                    session, storage, embedder, sparse_vectorizer
                )

                stats["sessions_processed"] += 1
                stats["documents_imported"] += result["documents"]
                stats["research_items_found"] += result["research_items"]

            except Exception as e:
                stats["errors"].append({
                    "session": str(session_path),
                    "error": str(e)
                })

    return stats


async def import_claude_session(
    session: ClaudeSession,
    storage: HybridStorage,
    embedder: EmbeddingProvider,
    sparse_vectorizer: BM25SparseVectorizer
) -> Dict[str, int]:
    """
    Import a single Claude Code session.
    """
    # Create session node in Neo4j
    await storage.neo4j.execute("""
        MERGE (s:Session {id: $session_id})
        SET s.project = $project,
            s.agentic_interface = 'claude_code',
            s.started_at = datetime($started_at),
            s.ended_at = datetime($ended_at),
            s.transcript_path = $transcript_path

        MERGE (p:Project {name: $project})
        MERGE (s)-[:BELONGS_TO]->(p)
    """, {
        "session_id": session.session_id,
        "project": session.project_path,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "transcript_path": str(session.project_path)
    })

    # Extract research items
    parser = ClaudeSessionParser()
    research_items = parser.extract_research_content(session)

    documents_imported = 0

    for item in research_items:
        if item["result"]:
            # Parse tool result based on tool type
            if "brave" in item["tool"].lower():
                content = parse_brave_search_result(item["result"])
            elif "firecrawl" in item["tool"].lower():
                content = parse_firecrawl_result(item["result"])
            else:
                continue

            if content:
                metadata = DocumentMetadata(
                    session_id=session.session_id,
                    project=session.project_path,
                    source_type="brave_search" if "brave" in item["tool"].lower() else "firecrawl",
                    source_url=item["input"].get("url", item["input"].get("query", "")),
                    title=content.get("title", "Untitled"),
                    agentic_interface="claude_code",
                    tags=[]
                )

                await ingest_document(
                    content["text"],
                    metadata,
                    storage,
                    embedder,
                    sparse_vectorizer
                )
                documents_imported += 1

    return {
        "documents": documents_imported,
        "research_items": len(research_items)
    }
```

---

## 4. Graph Exploration Queries

### 4.1 Neo4j Cypher Patterns

```cypher
// Find documents related to a specific topic with graph context
MATCH (d:Document)
WHERE d.id IN $document_ids
OPTIONAL MATCH (d)-[:FROM_SOURCE]->(s:Source)
OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (d)<-[:DERIVED_FROM]-(i:Insight)
OPTIONAL MATCH (d)<-[:CAPTURED]-(sess:Session)-[:BELONGS_TO]->(p:Project)
OPTIONAL MATCH (d)-[rel:RELATED_TO]-(related:Document)
RETURN d, s, collect(DISTINCT e) as entities,
       collect(DISTINCT i) as insights,
       sess, p,
       collect(DISTINCT {doc: related, score: rel.similarity_score}) as related_docs;


// Explore knowledge graph from a starting point
MATCH path = (start:Document {id: $start_id})-[*1..2]-(connected)
WHERE connected:Document OR connected:Entity OR connected:Insight
RETURN path,
       [n IN nodes(path) | labels(n)[0]] as node_types,
       [r IN relationships(path) | type(r)] as rel_types
LIMIT 100;


// Find insights across sessions for a project
MATCH (p:Project {name: $project_name})<-[:BELONGS_TO]-(s:Session)-[:PRODUCED]->(i:Insight)
MATCH (i)-[:DERIVED_FROM]->(d:Document)
RETURN i.id, i.insight_type, i.content, i.confidence, i.created_at,
       s.id as session_id, s.started_at as session_date,
       collect({id: d.id, title: d.title}) as source_documents
ORDER BY i.created_at DESC;


// Find all documents mentioning a specific entity with context
MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(d:Document)
MATCH (d)<-[:CAPTURED]-(s:Session)-[:BELONGS_TO]->(p:Project)
OPTIONAL MATCH (d)-[:FROM_SOURCE]->(src:Source)
RETURN d.id, d.title, d.content_preview, d.created_at,
       p.name as project, s.id as session,
       src.domain as source_domain
ORDER BY d.created_at DESC
LIMIT 20;


// Build timeline of research activity
MATCH (s:Session)-[:CAPTURED]->(d:Document)
WHERE s.started_at >= datetime($from_date)
  AND s.started_at <= datetime($to_date)
WITH s, collect(d) as documents, count(d) as doc_count
MATCH (s)-[:BELONGS_TO]->(p:Project)
OPTIONAL MATCH (s)-[:PRODUCED]->(i:Insight)
RETURN s.id, s.started_at, s.agentic_interface,
       p.name as project,
       doc_count,
       count(i) as insight_count
ORDER BY s.started_at DESC;


// Find conflicting insights
MATCH (i1:Insight)-[:CONTRADICTS]->(i2:Insight)
MATCH (i1)-[:DERIVED_FROM]->(d1:Document)
MATCH (i2)-[:DERIVED_FROM]->(d2:Document)
RETURN i1.content as insight_1, i2.content as insight_2,
       collect(DISTINCT d1.title) as sources_1,
       collect(DISTINCT d2.title) as sources_2;


// Calculate document similarity and create relationships
CALL apoc.periodic.iterate(
    "MATCH (d:Document) WHERE NOT exists(d.similarity_processed) RETURN d",
    "
    MATCH (other:Document)
    WHERE other.id <> d.id
    WITH d, other,
         gds.similarity.cosine(d.embedding, other.embedding) as similarity
    WHERE similarity > 0.8
    MERGE (d)-[r:RELATED_TO]-(other)
    SET r.similarity_score = similarity,
        d.similarity_processed = true
    ",
    {batchSize: 100, parallel: true}
);
```

---

## 5. MCP Tool Implementation Patterns

### 5.1 Tool Handler Pattern

```python
from typing import Callable, Dict, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Tool registry
TOOL_HANDLERS: Dict[str, Callable] = {}

def tool_handler(name: str):
    """
    Decorator to register a function as an MCP tool handler.
    Provides automatic error handling, logging, and validation.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(arguments: Dict[str, Any], **deps) -> str:
            logger.info(f"Executing tool: {name}", extra={"arguments": arguments})

            try:
                result = await func(arguments, **deps)
                logger.info(f"Tool {name} completed successfully")
                return result

            except ValidationError as e:
                logger.warning(f"Validation error in {name}: {e}")
                return f"Validation error: {str(e)}"

            except Exception as e:
                logger.exception(f"Error in tool {name}")
                raise

        TOOL_HANDLERS[name] = wrapper
        return wrapper
    return decorator


@tool_handler("rkg_capture_search_result")
async def capture_search_result(
    arguments: Dict[str, Any],
    storage: HybridStorage,
    embedder: EmbeddingProvider
) -> str:
    """
    Capture a Brave Search result.
    """
    # Validate input
    input_data = CaptureSearchResultInput(**arguments)

    # Build metadata
    metadata = DocumentMetadata(
        session_id=input_data.session_id,
        project=input_data.project,
        source_type="brave_search",
        source_url=input_data.url,
        title=input_data.title,
        agentic_interface="current_session",  # Will be set by context
        tags=input_data.tags or []
    )

    # Use description if no full content
    content = input_data.content or input_data.description

    # Ingest document
    document_id = await ingest_document(
        content=content,
        metadata=metadata,
        storage=storage,
        embedder=embedder,
        sparse_vectorizer=get_sparse_vectorizer()  # Singleton
    )

    return json.dumps({
        "success": True,
        "document_id": document_id,
        "message": f"Captured search result: {input_data.title}"
    })


@tool_handler("rkg_semantic_search")
async def semantic_search(
    arguments: Dict[str, Any],
    storage: HybridStorage,
    embedder: EmbeddingProvider,
    reranker: RerankerProvider
) -> str:
    """
    Execute semantic search across the knowledge graph.
    """
    input_data = SemanticSearchInput(**arguments)

    # Map string mode to enum
    mode_map = {
        "semantic": SearchMode.SEMANTIC,
        "lexical": SearchMode.LEXICAL,
        "hybrid": SearchMode.HYBRID
    }
    mode = mode_map.get(input_data.search_mode, SearchMode.HYBRID)

    # Build filters
    filters = SearchFilters(
        project=input_data.project_filter,
        source_type=input_data.source_type_filter,
        date_from=parse_date(input_data.date_from),
        date_to=parse_date(input_data.date_to)
    )

    # Execute search
    results = await hybrid_search(
        query=input_data.query,
        mode=mode,
        filters=filters,
        limit=input_data.limit,
        use_reranker=input_data.use_reranker
    )

    # Format results
    formatted = []
    for r in results:
        formatted.append({
            "id": r.id,
            "title": r.payload.get("title", "Untitled"),
            "content_preview": r.payload.get("content", "")[:300],
            "source_url": r.payload.get("source_url"),
            "source_type": r.payload.get("source_type"),
            "project": r.payload.get("project"),
            "score": r.score,
            "created_at": r.payload.get("created_at"),
            "entities": r.entities if hasattr(r, "entities") else [],
            "related_docs": r.related_documents if hasattr(r, "related_documents") else []
        })

    return json.dumps({
        "query": input_data.query,
        "mode": input_data.search_mode,
        "count": len(formatted),
        "results": formatted
    })


@tool_handler("rkg_record_insight")
async def record_insight(
    arguments: Dict[str, Any],
    storage: HybridStorage,
    embedder: EmbeddingProvider
) -> str:
    """
    Record an insight derived from research.
    """
    input_data = RecordInsightInput(**arguments)

    # Generate insight ID
    insight_id = str(uuid.uuid4())

    # Embed the insight
    embedding = await embedder.embed_text(
        input_data.content,
        input_type="document"
    )

    # Generate sparse vector
    sparse_vector = get_sparse_vectorizer().to_qdrant_sparse(input_data.content)

    # Store in Qdrant
    await storage.qdrant.upsert(
        collection_name="session_insights",
        points=[{
            "id": insight_id,
            "vector": {
                "dense": embedding.vector,
                "sparse": sparse_vector
            },
            "payload": {
                "insight_id": insight_id,
                "session_id": input_data.session_id,
                "project": input_data.project,
                "insight_type": input_data.insight_type,
                "content": input_data.content,
                "source_document_ids": input_data.source_document_ids,
                "created_at": datetime.utcnow().isoformat(),
                "confidence": input_data.confidence
            }
        }]
    )

    # Create Neo4j relationships
    await storage.neo4j.execute("""
        // Create Insight node
        CREATE (i:Insight {
            id: $insight_id,
            insight_type: $insight_type,
            content: $content,
            confidence: $confidence,
            created_at: datetime(),
            qdrant_id: $insight_id
        })

        // Link to session
        MATCH (s:Session {id: $session_id})
        CREATE (s)-[:PRODUCED]->(i)

        // Link to source documents
        UNWIND $source_doc_ids AS docId
        MATCH (d:Document {id: docId})
        CREATE (i)-[:DERIVED_FROM]->(d)
    """, {
        "insight_id": insight_id,
        "session_id": input_data.session_id,
        "insight_type": input_data.insight_type,
        "content": input_data.content,
        "confidence": input_data.confidence,
        "source_doc_ids": input_data.source_document_ids
    })

    return json.dumps({
        "success": True,
        "insight_id": insight_id,
        "message": f"Recorded {input_data.insight_type}: {input_data.content[:50]}..."
    })
```

---

## 6. Reranker Integration

```python
# src/rkg_mcp/rerankers/voyage.py

from typing import List, Optional
from dataclasses import dataclass
import voyageai
from .base import RerankerProvider, RerankResult

@dataclass
class VoyageRerankResult:
    index: int
    relevance_score: float
    document: str

class VoyageRerankerProvider(RerankerProvider):
    """Voyage AI reranker using rerank-2.5 model."""

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-2.5"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Return only top K results (default: all)

        Returns:
            List of RerankResult with index, score, and document
        """
        if not documents:
            return []

        result = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k or len(documents)
        )

        return [
            RerankResult(
                index=r.index,
                relevance_score=r.relevance_score,
                document=documents[r.index]
            )
            for r in result.results
        ]
```

---

## 7. Error Handling & Resilience

```python
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from functools import wraps

class RetryableError(Exception):
    """Errors that should trigger retry."""
    pass

class FatalError(Exception):
    """Errors that should not retry."""
    pass


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 30
):
    """
    Circuit breaker decorator for external service calls.
    """
    class CircuitBreaker:
        def __init__(self):
            self.failures = 0
            self.last_failure_time = None
            self.state = "closed"  # closed, open, half-open

        def record_failure(self):
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= failure_threshold:
                self.state = "open"

        def record_success(self):
            self.failures = 0
            self.state = "closed"

        def can_execute(self) -> bool:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.last_failure_time > recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            return True  # half-open

    breaker = CircuitBreaker()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise FatalError(f"Circuit breaker open for {func.__name__}")

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper
    return decorator


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RetryableError)
)
@with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
async def call_voyage_api(client, texts, model, input_type):
    """
    Call Voyage API with retry and circuit breaker.
    """
    try:
        return client.embed(
            texts=texts,
            model=model,
            input_type=input_type
        )
    except voyageai.RateLimitError:
        raise RetryableError("Rate limited")
    except voyageai.APIConnectionError:
        raise RetryableError("Connection error")
    except voyageai.AuthenticationError:
        raise FatalError("Invalid API key")
```

---

## 8. Testing Patterns

```python
# tests/test_hybrid_search.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from rkg_mcp.storage.hybrid import hybrid_search, reciprocal_rank_fusion
from rkg_mcp.models import SearchResult

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    client = AsyncMock()
    return client

@pytest.fixture
def mock_embedder():
    """Mock embedding provider."""
    embedder = AsyncMock()
    embedder.embed_text.return_value = MagicMock(
        vector=[0.1] * 1024
    )
    return embedder

@pytest.fixture
def mock_reranker():
    """Mock reranker."""
    reranker = AsyncMock()
    return reranker


class TestReciprocalRankFusion:

    def test_single_list(self):
        """Single list should return items in order."""
        results = [
            SearchResult(id="1", score=0.9),
            SearchResult(id="2", score=0.8),
        ]

        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 2
        assert fused[0].id == "1"
        assert fused[1].id == "2"

    def test_two_lists_same_order(self):
        """Two lists with same order should boost top items."""
        list1 = [
            SearchResult(id="1", score=0.9),
            SearchResult(id="2", score=0.8),
        ]
        list2 = [
            SearchResult(id="1", score=0.95),
            SearchResult(id="2", score=0.85),
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        assert fused[0].id == "1"
        assert fused[0].score > fused[1].score

    def test_two_lists_different_order(self):
        """Items appearing in both lists get boosted."""
        list1 = [
            SearchResult(id="1", score=0.9),
            SearchResult(id="2", score=0.8),
        ]
        list2 = [
            SearchResult(id="3", score=0.95),
            SearchResult(id="1", score=0.85),
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        # "1" appears in both, should be ranked higher
        ids = [r.id for r in fused]
        assert ids.index("1") < ids.index("2")
        assert ids.index("1") < ids.index("3")


class TestHybridSearch:

    @pytest.mark.asyncio
    async def test_semantic_mode(self, mock_qdrant, mock_embedder, mock_reranker):
        """Semantic mode should only use dense vectors."""
        mock_qdrant.search.return_value = [
            SearchResult(id="1", score=0.9, payload={"content": "test"})
        ]

        results = await hybrid_search(
            query="test query",
            mode=SearchMode.SEMANTIC,
            filters=SearchFilters(),
            limit=10,
            use_reranker=False,
            qdrant=mock_qdrant,
            embedder=mock_embedder,
            reranker=mock_reranker
        )

        # Should call embed_text for query
        mock_embedder.embed_text.assert_called_once()

        # Should only call search once (not twice for hybrid)
        assert mock_qdrant.search.call_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_mode_fuses_results(
        self, mock_qdrant, mock_embedder, mock_reranker
    ):
        """Hybrid mode should fuse dense and sparse results."""
        # Return different results for each search
        mock_qdrant.search.side_effect = [
            [SearchResult(id="1", score=0.9, payload={"content": "dense"})],
            [SearchResult(id="2", score=0.95, payload={"content": "sparse"})]
        ]

        results = await hybrid_search(
            query="test query",
            mode=SearchMode.HYBRID,
            filters=SearchFilters(),
            limit=10,
            use_reranker=False,
            qdrant=mock_qdrant,
            embedder=mock_embedder,
            reranker=mock_reranker
        )

        # Should have both results
        ids = {r.id for r in results}
        assert "1" in ids
        assert "2" in ids
```

---

## 9. Performance Optimization

```python
# Batch processing for large imports
async def batch_embed_documents(
    documents: List[str],
    embedder: EmbeddingProvider,
    batch_size: int = 128,
    max_concurrent: int = 4
) -> List[List[float]]:
    """
    Embed documents in batches with controlled concurrency.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_batch(batch: List[str]) -> List[List[float]]:
        async with semaphore:
            results = await embedder.embed_batch(batch)
            return [r.vector for r in results]

    # Split into batches
    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    # Process all batches
    all_results = await asyncio.gather(*[
        embed_batch(batch) for batch in batches
    ])

    # Flatten results
    return [vec for batch_result in all_results for vec in batch_result]


# Caching for frequent queries
from functools import lru_cache
from cachetools import TTLCache
import hashlib

# In-memory cache for embeddings
embedding_cache = TTLCache(maxsize=10000, ttl=3600)

async def cached_embed(text: str, embedder: EmbeddingProvider) -> List[float]:
    """
    Cache embeddings to avoid redundant API calls.
    """
    cache_key = hashlib.md5(text.encode()).hexdigest()

    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    result = await embedder.embed_text(text)
    embedding_cache[cache_key] = result.vector

    return result.vector


# Connection pooling for databases
class ConnectionPool:
    """
    Async connection pool for database clients.
    """
    def __init__(self, factory, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self._pool = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        # Try to get existing connection
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Create new connection if under limit
        async with self._lock:
            if self._created < self.max_size:
                conn = await self.factory()
                self._created += 1
                return conn

        # Wait for available connection
        return await self._pool.get()

    async def release(self, conn):
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Pool full, close connection
            await conn.close()
            async with self._lock:
                self._created -= 1
```

This completes the detailed pseudocode and implementation patterns document.

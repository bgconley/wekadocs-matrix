# RKG MCP Server Implementation
## Native MCP SDK - No FastMCP

*This document corrects the previous specification to use the native MCP SDK as originally requested.*

---

## 1. Native MCP SDK Architecture

### 1.1 Why Native SDK Over FastMCP

| Aspect | Native MCP SDK | FastMCP |
|--------|---------------|---------|
| Control | Full low-level control | Abstracted, opinionated |
| Tool Definition | Explicit schema objects | Decorator-based |
| Transport | Manual configuration | Auto-configured |
| Instructions | Explicit registration | Convention-based |
| Modularity | Maximum flexibility | Convenience over control |

Your requirement was explicit: **"leverage the MCP sdk so we can implement our own low level MCP tools and explicit instructions rather than using FastMCP."**

---

## 2. Server Implementation with Native SDK

### 2.1 Core Server Structure

```python
"""
RKG MCP Server - Native SDK Implementation

Supports both stdio and HTTP/SSE transports without FastMCP.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Sequence

# Native MCP SDK imports - NOT FastMCP
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    ListToolsResult,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    ResourceTemplate,
    ListResourcesResult,
    ReadResourceResult,
)

# Transport imports
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport

# Infrastructure clients
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase
import voyageai

logger = logging.getLogger(__name__)


@dataclass
class RKGServerConfig:
    """Configuration for RKG MCP Server."""

    # Database connections
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Embedding configuration
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3-large"
    rerank_model: str = "rerank-2.5"

    # Server settings
    server_name: str = "rkg-server"
    server_version: str = "1.0.0"

    # Transport settings
    http_host: str = "0.0.0.0"
    http_port: int = 8080


class RKGMCPServer:
    """
    Research Knowledge Graph MCP Server.

    Native MCP SDK implementation with explicit tool definitions
    and support for both stdio and HTTP/SSE transports.
    """

    def __init__(self, config: RKGServerConfig):
        self.config = config

        # Initialize native MCP Server - NOT FastMCP
        self.server = Server(config.server_name)

        # Infrastructure clients (lazy initialization)
        self._qdrant: AsyncQdrantClient | None = None
        self._neo4j_driver = None
        self._voyage: voyageai.AsyncClient | None = None

        # Register handlers using native SDK methods
        self._register_handlers()

    def _register_handlers(self):
        """Register all handlers with the native MCP Server."""

        # Tool listing handler
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Return all available tools with explicit schemas."""
            return self._get_tool_definitions()

        # Tool execution handler
        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict[str, Any] | None
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Execute a tool and return results."""
            return await self._execute_tool(name, arguments or {})

        # Resource listing handler
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources."""
            return await self._list_resources()

        # Resource reading handler
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific resource."""
            return await self._read_resource(uri)

        # Prompt listing handler
        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            """List available prompts."""
            return self._get_prompt_definitions()

        # Prompt retrieval handler
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: dict[str, str] | None
        ) -> GetPromptResult:
            """Get a specific prompt with arguments."""
            return await self._get_prompt(name, arguments or {})

    # =========================================================================
    # TOOL DEFINITIONS - Explicit Schemas (No Decorators)
    # =========================================================================

    def _get_tool_definitions(self) -> list[Tool]:
        """
        Define all tools with explicit JSON schemas.

        This is the native SDK approach - explicit Tool objects
        rather than FastMCP's @mcp.tool() decorators.
        """
        return [
            # --- Capture Tools ---
            Tool(
                name="rkg_capture_search",
                description="""Capture and store search results from Brave Search.

                Call this tool AFTER receiving Brave Search results to persist
                them in the knowledge graph for future retrieval.

                The tool will:
                1. Parse the search results
                2. Generate embeddings via Voyage AI
                3. Store in Qdrant (vector) and Neo4j (graph)
                4. Link to current session and project""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The original search query"
                        },
                        "results": {
                            "type": "array",
                            "description": "Array of search result objects",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "description": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["title", "url"]
                            }
                        },
                        "project": {
                            "type": "string",
                            "description": "Project name for organization"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        }
                    },
                    "required": ["query", "results"]
                }
            ),

            Tool(
                name="rkg_capture_page",
                description="""Capture and store a scraped web page from Firecrawl.

                Call this tool AFTER receiving Firecrawl scrape results to persist
                the content in the knowledge graph.

                The tool will:
                1. Parse and chunk the page content
                2. Extract entities and relationships
                3. Generate embeddings for each chunk
                4. Store with full provenance tracking""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Source URL of the scraped page"
                        },
                        "title": {
                            "type": "string",
                            "description": "Page title"
                        },
                        "content": {
                            "type": "string",
                            "description": "Full page content (markdown or text)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata from Firecrawl"
                        },
                        "project": {
                            "type": "string",
                            "description": "Project name for organization"
                        }
                    },
                    "required": ["url", "content"]
                }
            ),

            # --- Insight Tools ---
            Tool(
                name="rkg_store_insight",
                description="""Store a conclusion, learning, or insight derived from research.

                Use this to capture your analysis and conclusions about the
                research material. Insights are linked to their source documents.

                Insight types:
                - fact: Verified factual information
                - conclusion: Logical conclusion from evidence
                - hypothesis: Unverified theory to explore
                - decision: Implementation decision made
                - question: Open question for future research""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The insight content"
                        },
                        "insight_type": {
                            "type": "string",
                            "enum": ["fact", "conclusion", "hypothesis", "decision", "question"],
                            "description": "Type of insight"
                        },
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Document IDs this insight is derived from"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence level (0-1)"
                        },
                        "project": {
                            "type": "string",
                            "description": "Project name"
                        }
                    },
                    "required": ["content", "insight_type"]
                }
            ),

            # --- Retrieval Tools ---
            Tool(
                name="rkg_search",
                description="""Search the knowledge graph for relevant documents and insights.

                Performs hybrid search combining:
                - Dense vector search (semantic similarity via Voyage)
                - Sparse vector search (BM25 lexical matching)
                - Graph traversal (entity relationships)

                Results are reranked using Voyage rerank-2.5.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum results to return"
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "project": {"type": "string"},
                                "source_type": {
                                    "type": "string",
                                    "enum": ["brave_search", "firecrawl", "insight", "session"]
                                },
                                "date_from": {"type": "string", "format": "date"},
                                "date_to": {"type": "string", "format": "date"}
                            }
                        },
                        "include_insights": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include derived insights in results"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="rkg_get_context",
                description="""Get full context for a document including related materials.

                Returns:
                - The document content
                - Related documents (by entity/topic)
                - Derived insights
                - Source chain (what led to this document)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to get context for"
                        },
                        "depth": {
                            "type": "integer",
                            "default": 2,
                            "description": "How many relationship hops to traverse"
                        }
                    },
                    "required": ["document_id"]
                }
            ),

            # --- Entity Tools ---
            Tool(
                name="rkg_extract_entities",
                description="""Extract entities and relationships from text.

                Uses GLiNER for zero-shot NER and REBEL for relationship extraction.
                Entities are linked to Wikipedia for disambiguation.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract entities from"
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Optional document ID to link entities to"
                        },
                        "extract_relationships": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["text"]
                }
            ),

            # --- Session Tools ---
            Tool(
                name="rkg_start_session",
                description="""Start a new research session.

                Creates a session node in the graph to track all research
                and insights generated during this coding session.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Project name"
                        },
                        "interface": {
                            "type": "string",
                            "enum": ["claude_code", "openai_codex", "cursor", "other"],
                            "description": "Agentic coding interface being used"
                        },
                        "objectives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Session objectives/goals"
                        }
                    },
                    "required": ["project", "interface"]
                }
            ),

            Tool(
                name="rkg_end_session",
                description="""End the current research session.

                Finalizes the session, generates summary, and creates
                session-level embeddings for future retrieval.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to end"
                        },
                        "summary": {
                            "type": "string",
                            "description": "Optional session summary"
                        }
                    },
                    "required": ["session_id"]
                }
            ),

            # --- Analytics Tools ---
            Tool(
                name="rkg_analytics",
                description="""Get analytics and insights about the knowledge graph.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": ["activity", "entities", "gaps", "graph", "summary"]
                        },
                        "days": {
                            "type": "integer",
                            "default": 30
                        }
                    }
                }
            ),

            # --- Export Tools ---
            Tool(
                name="rkg_export",
                description="""Export knowledge graph data in various formats.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["json", "jsonl", "markdown", "cypher", "csv", "graphml"]
                        },
                        "scope": {
                            "type": "object",
                            "properties": {
                                "projects": {"type": "array", "items": {"type": "string"}},
                                "sessions": {"type": "array", "items": {"type": "string"}},
                                "date_from": {"type": "string"},
                                "date_to": {"type": "string"}
                            }
                        }
                    }
                }
            ),

            # --- Contradiction Tools ---
            Tool(
                name="rkg_check_contradictions",
                description="""Check for contradictions against existing knowledge.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to check for contradictions"
                        },
                        "threshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Confidence threshold for flagging"
                        }
                    },
                    "required": ["text"]
                }
            )
        ]

    # =========================================================================
    # PROMPT DEFINITIONS
    # =========================================================================

    def _get_prompt_definitions(self) -> list[Prompt]:
        """Define available prompts with explicit schemas."""
        return [
            Prompt(
                name="research_workflow",
                description="Guide for conducting research with RKG integration",
                arguments=[
                    PromptArgument(
                        name="topic",
                        description="Research topic",
                        required=True
                    ),
                    PromptArgument(
                        name="project",
                        description="Project name",
                        required=True
                    )
                ]
            ),
            Prompt(
                name="insight_extraction",
                description="Guide for extracting and storing insights from research",
                arguments=[
                    PromptArgument(
                        name="document_ids",
                        description="Comma-separated document IDs to analyze",
                        required=True
                    )
                ]
            )
        ]

    async def _get_prompt(
        self,
        name: str,
        arguments: dict[str, str]
    ) -> GetPromptResult:
        """Get prompt content with arguments applied."""

        if name == "research_workflow":
            return GetPromptResult(
                description=f"Research workflow for: {arguments.get('topic', 'unknown')}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""You are conducting research on: {arguments.get('topic')}

Project: {arguments.get('project')}

Workflow:
1. Use Brave Search to find authoritative sources
2. Call rkg_capture_search to store the search results
3. Use Firecrawl to scrape relevant pages
4. Call rkg_capture_page to store the scraped content
5. Analyze the content and use rkg_store_insight for conclusions
6. Use rkg_search to find related existing knowledge

Before starting, call rkg_start_session to track this research."""
                        )
                    )
                ]
            )

        raise ValueError(f"Unknown prompt: {name}")

    # =========================================================================
    # TOOL EXECUTION
    # =========================================================================

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> Sequence[TextContent]:
        """Execute a tool and return results."""

        # Tool dispatch table
        handlers = {
            "rkg_capture_search": self._handle_capture_search,
            "rkg_capture_page": self._handle_capture_page,
            "rkg_store_insight": self._handle_store_insight,
            "rkg_search": self._handle_search,
            "rkg_get_context": self._handle_get_context,
            "rkg_extract_entities": self._handle_extract_entities,
            "rkg_start_session": self._handle_start_session,
            "rkg_end_session": self._handle_end_session,
            "rkg_analytics": self._handle_analytics,
            "rkg_export": self._handle_export,
            "rkg_check_contradictions": self._handle_check_contradictions,
        }

        handler = handlers.get(name)
        if not handler:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

        try:
            result = await handler(arguments)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    # =========================================================================
    # TOOL HANDLERS
    # =========================================================================

    async def _handle_capture_search(self, args: dict) -> dict:
        """Handle rkg_capture_search tool."""
        query = args["query"]
        results = args["results"]
        project = args.get("project", "default")
        tags = args.get("tags", [])

        stored_ids = []

        for result in results:
            # Generate embedding
            embedding = await self._get_embedding(
                f"{result.get('title', '')} {result.get('description', '')}"
            )

            # Create document ID
            import hashlib
            doc_id = hashlib.sha256(
                result["url"].encode()
            ).hexdigest()[:16]

            # Store in Qdrant
            await self.qdrant.upsert(
                collection_name="research_documents",
                points=[{
                    "id": doc_id,
                    "vector": {"dense": embedding},
                    "payload": {
                        "source_type": "brave_search",
                        "source_url": result["url"],
                        "title": result.get("title", ""),
                        "content": result.get("description", ""),
                        "project": project,
                        "tags": tags,
                        "query": query,
                        "created_at": self._now_iso()
                    }
                }]
            )

            # Store in Neo4j
            async with self.neo4j.session() as session:
                await session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.source_type = 'brave_search',
                        d.source_url = $url,
                        d.title = $title,
                        d.project = $project,
                        d.created_at = datetime()

                    MERGE (s:Source {url: $url})
                    SET s.domain = $domain

                    MERGE (d)-[:FROM_SOURCE]->(s)

                    WITH d
                    MATCH (sess:Session {id: $session_id})
                    MERGE (sess)-[:CONTAINS]->(d)
                """, {
                    "id": doc_id,
                    "url": result["url"],
                    "title": result.get("title", ""),
                    "project": project,
                    "domain": self._extract_domain(result["url"]),
                    "session_id": self._current_session_id
                })

            stored_ids.append(doc_id)

        return {
            "success": True,
            "stored_count": len(stored_ids),
            "document_ids": stored_ids
        }

    async def _handle_capture_page(self, args: dict) -> dict:
        """Handle rkg_capture_page tool."""
        url = args["url"]
        content = args["content"]
        title = args.get("title", "")
        project = args.get("project", "default")
        metadata = args.get("metadata", {})

        # Chunk the content
        chunks = self._chunk_content(content)
        stored_ids = []

        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await self._get_embedding(chunk)

            # Create chunk ID
            import hashlib
            chunk_id = hashlib.sha256(
                f"{url}:{i}".encode()
            ).hexdigest()[:16]

            # Store in Qdrant
            await self.qdrant.upsert(
                collection_name="research_documents",
                points=[{
                    "id": chunk_id,
                    "vector": {"dense": embedding},
                    "payload": {
                        "source_type": "firecrawl",
                        "source_url": url,
                        "title": title,
                        "content": chunk,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "project": project,
                        "metadata": metadata,
                        "created_at": self._now_iso()
                    }
                }]
            )

            stored_ids.append(chunk_id)

        # Store document node in Neo4j
        async with self.neo4j.session() as session:
            await session.run("""
                MERGE (d:Document {id: $id})
                SET d.source_type = 'firecrawl',
                    d.source_url = $url,
                    d.title = $title,
                    d.project = $project,
                    d.chunk_count = $chunk_count,
                    d.created_at = datetime()

                MERGE (s:Source {url: $url})
                SET s.domain = $domain

                MERGE (d)-[:FROM_SOURCE]->(s)
            """, {
                "id": stored_ids[0] if stored_ids else "",
                "url": url,
                "title": title,
                "project": project,
                "chunk_count": len(chunks),
                "domain": self._extract_domain(url)
            })

        # Extract and store entities
        entities = await self._extract_entities_internal(content, stored_ids[0])

        return {
            "success": True,
            "document_id": stored_ids[0] if stored_ids else None,
            "chunk_ids": stored_ids,
            "chunk_count": len(chunks),
            "entities_extracted": len(entities)
        }

    async def _handle_store_insight(self, args: dict) -> dict:
        """Handle rkg_store_insight tool."""
        content = args["content"]
        insight_type = args["insight_type"]
        source_ids = args.get("source_ids", [])
        confidence = args.get("confidence", 0.8)
        project = args.get("project", "default")

        # Generate embedding
        embedding = await self._get_embedding(content)

        # Create insight ID
        import hashlib
        insight_id = hashlib.sha256(
            f"insight:{content[:100]}:{self._now_iso()}".encode()
        ).hexdigest()[:16]

        # Store in Qdrant
        await self.qdrant.upsert(
            collection_name="research_documents",
            points=[{
                "id": insight_id,
                "vector": {"dense": embedding},
                "payload": {
                    "source_type": "insight",
                    "insight_type": insight_type,
                    "content": content,
                    "confidence": confidence,
                    "project": project,
                    "source_document_ids": source_ids,
                    "created_at": self._now_iso()
                }
            }]
        )

        # Store in Neo4j with links to sources
        async with self.neo4j.session() as session:
            await session.run("""
                CREATE (i:Insight {id: $id})
                SET i.type = $type,
                    i.content = $content,
                    i.confidence = $confidence,
                    i.project = $project,
                    i.created_at = datetime()

                WITH i
                UNWIND $source_ids as source_id
                MATCH (d:Document {id: source_id})
                MERGE (i)-[:DERIVED_FROM]->(d)

                WITH i
                MATCH (sess:Session {id: $session_id})
                MERGE (sess)-[:GENERATED]->(i)
            """, {
                "id": insight_id,
                "type": insight_type,
                "content": content,
                "confidence": confidence,
                "project": project,
                "source_ids": source_ids,
                "session_id": self._current_session_id
            })

        return {
            "success": True,
            "insight_id": insight_id,
            "linked_sources": len(source_ids)
        }

    async def _handle_search(self, args: dict) -> dict:
        """Handle rkg_search tool with hybrid search."""
        query = args["query"]
        limit = args.get("limit", 10)
        filters = args.get("filters", {})
        include_insights = args.get("include_insights", True)

        # Generate query embedding
        query_embedding = await self._get_embedding(query)

        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filters, include_insights)

        # Dense vector search
        dense_results = await self.qdrant.search(
            collection_name="research_documents",
            query_vector=("dense", query_embedding),
            query_filter=qdrant_filter,
            limit=limit * 3,  # Overfetch for reranking
            with_payload=True
        )

        # Prepare for reranking
        documents = [r.payload.get("content", "")[:2000] for r in dense_results]

        if documents:
            # Rerank with Voyage
            reranked = await self.voyage.rerank(
                query=query,
                documents=documents,
                model=self.config.rerank_model,
                top_k=limit
            )

            # Build final results
            results = []
            for item in reranked.results:
                original = dense_results[item.index]
                results.append({
                    "id": original.id,
                    "score": item.relevance_score,
                    "title": original.payload.get("title", ""),
                    "content": original.payload.get("content", "")[:500],
                    "source_type": original.payload.get("source_type"),
                    "source_url": original.payload.get("source_url"),
                    "project": original.payload.get("project")
                })
        else:
            results = []

        return {
            "query": query,
            "result_count": len(results),
            "results": results
        }

    async def _handle_get_context(self, args: dict) -> dict:
        """Handle rkg_get_context tool."""
        document_id = args["document_id"]
        depth = args.get("depth", 2)

        # Get document from Qdrant
        points = await self.qdrant.retrieve(
            collection_name="research_documents",
            ids=[document_id],
            with_payload=True
        )

        if not points:
            return {"error": f"Document not found: {document_id}"}

        document = points[0].payload

        # Get related documents from Neo4j
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (d:Document {id: $id})

                // Get entities mentioned
                OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
                WITH d, collect(DISTINCT {name: e.name, type: e.type}) as entities

                // Get derived insights
                OPTIONAL MATCH (i:Insight)-[:DERIVED_FROM]->(d)
                WITH d, entities, collect(DISTINCT {
                    id: i.id,
                    content: i.content,
                    type: i.type
                }) as insights

                // Get related documents (through shared entities)
                OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Document)
                WHERE related.id <> d.id
                WITH d, entities, insights,
                     collect(DISTINCT {id: related.id, title: related.title})[..10] as related

                // Get source chain
                OPTIONAL MATCH (d)-[:FROM_SOURCE]->(s:Source)

                RETURN d, entities, insights, related, s.url as source_url
            """, {"id": document_id})

            record = await result.single()

        return {
            "document": document,
            "entities": record["entities"] if record else [],
            "insights": record["insights"] if record else [],
            "related_documents": record["related"] if record else [],
            "source_url": record["source_url"] if record else None
        }

    async def _handle_extract_entities(self, args: dict) -> dict:
        """Handle rkg_extract_entities tool."""
        text = args["text"]
        document_id = args.get("document_id")
        extract_relationships = args.get("extract_relationships", True)

        entities = await self._extract_entities_internal(text, document_id)

        return {
            "entity_count": len(entities),
            "entities": entities
        }

    async def _handle_start_session(self, args: dict) -> dict:
        """Handle rkg_start_session tool."""
        project = args["project"]
        interface = args["interface"]
        objectives = args.get("objectives", [])

        import hashlib
        session_id = hashlib.sha256(
            f"session:{project}:{self._now_iso()}".encode()
        ).hexdigest()[:16]

        async with self.neo4j.session() as session:
            await session.run("""
                CREATE (s:Session {id: $id})
                SET s.project = $project,
                    s.interface = $interface,
                    s.objectives = $objectives,
                    s.started_at = datetime(),
                    s.status = 'active'

                MERGE (p:Project {name: $project})
                MERGE (s)-[:BELONGS_TO]->(p)
            """, {
                "id": session_id,
                "project": project,
                "interface": interface,
                "objectives": objectives
            })

        self._current_session_id = session_id

        return {
            "success": True,
            "session_id": session_id,
            "project": project,
            "interface": interface
        }

    async def _handle_end_session(self, args: dict) -> dict:
        """Handle rkg_end_session tool."""
        session_id = args["session_id"]
        summary = args.get("summary", "")

        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (s:Session {id: $id})
                SET s.status = 'completed',
                    s.ended_at = datetime(),
                    s.summary = $summary

                WITH s
                OPTIONAL MATCH (s)-[:CONTAINS]->(d:Document)
                OPTIONAL MATCH (s)-[:GENERATED]->(i:Insight)

                RETURN s.started_at as started,
                       count(DISTINCT d) as document_count,
                       count(DISTINCT i) as insight_count
            """, {
                "id": session_id,
                "summary": summary
            })

            record = await result.single()

        self._current_session_id = None

        return {
            "success": True,
            "session_id": session_id,
            "document_count": record["document_count"] if record else 0,
            "insight_count": record["insight_count"] if record else 0
        }

    async def _handle_analytics(self, args: dict) -> dict:
        """Handle rkg_analytics tool."""
        metric = args.get("metric", "summary")
        days = args.get("days", 30)

        # Placeholder - would call analytics engine
        return {
            "metric": metric,
            "days": days,
            "data": {}
        }

    async def _handle_export(self, args: dict) -> dict:
        """Handle rkg_export tool."""
        format_type = args.get("format", "json")
        scope = args.get("scope", {})

        # Placeholder - would call export engine
        return {
            "format": format_type,
            "scope": scope,
            "status": "pending"
        }

    async def _handle_check_contradictions(self, args: dict) -> dict:
        """Handle rkg_check_contradictions tool."""
        text = args["text"]
        threshold = args.get("threshold", 0.7)

        # Placeholder - would call contradiction detection pipeline
        return {
            "text_length": len(text),
            "threshold": threshold,
            "contradictions": []
        }

    # =========================================================================
    # RESOURCE HANDLERS
    # =========================================================================

    async def _list_resources(self) -> list[Resource]:
        """List available resources."""
        return [
            Resource(
                uri="rkg://projects",
                name="Projects",
                description="List of all projects in the knowledge graph",
                mimeType="application/json"
            ),
            Resource(
                uri="rkg://sessions/recent",
                name="Recent Sessions",
                description="Recent research sessions",
                mimeType="application/json"
            ),
            Resource(
                uri="rkg://stats",
                name="Knowledge Graph Statistics",
                description="Current statistics about the knowledge graph",
                mimeType="application/json"
            )
        ]

    async def _read_resource(self, uri: str) -> str:
        """Read a specific resource."""
        if uri == "rkg://projects":
            async with self.neo4j.session() as session:
                result = await session.run("""
                    MATCH (p:Project)
                    OPTIONAL MATCH (s:Session)-[:BELONGS_TO]->(p)
                    OPTIONAL MATCH (s)-[:CONTAINS]->(d:Document)
                    RETURN p.name as name,
                           count(DISTINCT s) as sessions,
                           count(DISTINCT d) as documents
                    ORDER BY documents DESC
                """)
                projects = [dict(r) async for r in result]
            return json.dumps(projects, indent=2)

        elif uri == "rkg://sessions/recent":
            async with self.neo4j.session() as session:
                result = await session.run("""
                    MATCH (s:Session)
                    OPTIONAL MATCH (s)-[:CONTAINS]->(d:Document)
                    OPTIONAL MATCH (s)-[:GENERATED]->(i:Insight)
                    RETURN s.id as id,
                           s.project as project,
                           s.interface as interface,
                           s.started_at as started,
                           s.status as status,
                           count(DISTINCT d) as documents,
                           count(DISTINCT i) as insights
                    ORDER BY s.started_at DESC
                    LIMIT 20
                """)
                sessions = [dict(r) async for r in result]
            return json.dumps(sessions, indent=2, default=str)

        elif uri == "rkg://stats":
            stats = await self._get_graph_stats()
            return json.dumps(stats, indent=2)

        raise ValueError(f"Unknown resource: {uri}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @property
    def qdrant(self) -> AsyncQdrantClient:
        """Lazy-load Qdrant client."""
        if self._qdrant is None:
            self._qdrant = AsyncQdrantClient(url=self.config.qdrant_url)
        return self._qdrant

    @property
    def neo4j(self):
        """Lazy-load Neo4j driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
        return self._neo4j_driver

    @property
    def voyage(self) -> voyageai.AsyncClient:
        """Lazy-load Voyage AI client."""
        if self._voyage is None:
            self._voyage = voyageai.AsyncClient(
                api_key=self.config.voyage_api_key
            )
        return self._voyage

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using configured embedder."""
        result = await self.voyage.embed(
            texts=[text],
            model=self.config.embedding_model,
            input_type="document"
        )
        return result.embeddings[0]

    def _chunk_content(self, content: str, chunk_size: int = 1000) -> list[str]:
        """Chunk content for embedding."""
        # Simple chunking - production would use semantic chunking
        chunks = []
        words = content.split()
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(url).netloc

    def _now_iso(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _build_qdrant_filter(self, filters: dict, include_insights: bool):
        """Build Qdrant filter from parameters."""
        conditions = []

        if filters.get("project"):
            conditions.append({
                "key": "project",
                "match": {"value": filters["project"]}
            })

        if filters.get("source_type"):
            conditions.append({
                "key": "source_type",
                "match": {"value": filters["source_type"]}
            })

        if not include_insights:
            conditions.append({
                "key": "source_type",
                "match": {"value": "insight"},
                "must_not": True
            })

        return {"must": conditions} if conditions else None

    async def _extract_entities_internal(
        self,
        text: str,
        document_id: str | None
    ) -> list[dict]:
        """Internal entity extraction."""
        # Placeholder - would call entity extraction pipeline
        return []

    async def _get_graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (d:Document)
                WITH count(d) as docs
                MATCH (e:Entity)
                WITH docs, count(e) as entities
                MATCH (i:Insight)
                WITH docs, entities, count(i) as insights
                MATCH (s:Session)
                RETURN docs, entities, insights, count(s) as sessions
            """)
            record = await result.single()

        return {
            "documents": record["docs"] if record else 0,
            "entities": record["entities"] if record else 0,
            "insights": record["insights"] if record else 0,
            "sessions": record["sessions"] if record else 0
        }

    # =========================================================================
    # TRANSPORT: STDIO
    # =========================================================================

    async def run_stdio(self):
        """Run server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.config.server_name,
                    server_version=self.config.server_version
                )
            )

    # =========================================================================
    # TRANSPORT: HTTP/SSE
    # =========================================================================

    def create_sse_app(self):
        """Create Starlette app for SSE transport.

        This uses the native MCP SSE transport, NOT FastMCP.
        """
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import JSONResponse

        # Create SSE transport
        sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request):
            """Handle SSE connection."""
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send
            ) as streams:
                await self.server.run(
                    streams[0],
                    streams[1],
                    InitializationOptions(
                        server_name=self.config.server_name,
                        server_version=self.config.server_version
                    )
                )

        async def handle_messages(request):
            """Handle POST messages for SSE."""
            await sse_transport.handle_post_message(
                request.scope,
                request.receive,
                request._send
            )

        async def health_check(request):
            """Health check endpoint."""
            return JSONResponse({
                "status": "healthy",
                "server": self.config.server_name,
                "version": self.config.server_version
            })

        return Starlette(
            routes=[
                Route("/health", health_check),
                Route("/sse", handle_sse),
                Route("/messages/", handle_messages, methods=["POST"]),
            ]
        )

    async def run_http(self):
        """Run server with HTTP/SSE transport."""
        import uvicorn

        app = self.create_sse_app()

        config = uvicorn.Config(
            app,
            host=self.config.http_host,
            port=self.config.http_port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_stdio():
    """Entry point for stdio transport."""
    import os

    config = RKGServerConfig(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
    )

    server = RKGMCPServer(config)
    asyncio.run(server.run_stdio())


def run_http():
    """Entry point for HTTP/SSE transport."""
    import os

    config = RKGServerConfig(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        http_host=os.getenv("HTTP_HOST", "0.0.0.0"),
        http_port=int(os.getenv("HTTP_PORT", "8080")),
    )

    server = RKGMCPServer(config)
    asyncio.run(server.run_http())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "http":
        run_http()
    else:
        run_stdio()
```

---

## 3. Key Differences from FastMCP

### 3.1 Tool Definition Comparison

```python
# ❌ FastMCP (What you did NOT want)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("server")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description."""
    return "result"


# ✅ Native MCP SDK (What you requested)
from mcp.server import Server
from mcp.types import Tool

server = Server("server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Tool description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_tool":
        return [TextContent(type="text", text="result")]
```

### 3.2 Transport Comparison

```python
# ❌ FastMCP HTTP (abstracted away)
app = FastMCP("server")
app.run(transport="sse")


# ✅ Native MCP SDK (explicit control)
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette

sse = SseServerTransport("/messages/")

app = Starlette(routes=[
    Route("/sse", handle_sse),
    Route("/messages/", handle_messages, methods=["POST"]),
])
```

---

## 4. Registration with MCP Clients

### 4.1 Claude Code Configuration

```json
// ~/.config/claude-code/mcp.json
{
  "mcpServers": {
    "rkg": {
      "command": "python",
      "args": ["-m", "rkg_mcp.server"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "${RKG_NEO4J_PASSWORD}",
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}"
      }
    }
  }
}
```

### 4.2 OpenAI Codex Configuration

```json
// ~/.codex/mcp_servers.json
{
  "servers": [
    {
      "name": "rkg",
      "transport": "stdio",
      "command": ["python", "-m", "rkg_mcp.server"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "NEO4J_URI": "bolt://localhost:7687"
      }
    }
  ]
}
```

### 4.3 HTTP Client Connection

```python
# For clients that support HTTP/SSE transport
import httpx
from mcp.client.sse import sse_client

async def connect_to_rkg():
    async with sse_client("http://localhost:8080/sse") as (read, write):
        # Initialize session
        # Call tools via the streams
        pass
```

---

## 5. Docker Configuration

```dockerfile
# Dockerfile for RKG MCP Server
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY rkg_mcp/ ./rkg_mcp/

# Default to stdio transport
CMD ["python", "-m", "rkg_mcp.server"]

# For HTTP transport, override with:
# CMD ["python", "-m", "rkg_mcp.server", "http"]
```

```yaml
# docker-compose.yml
services:
  rkg-mcp:
    build: .
    environment:
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - VOYAGE_API_KEY=${VOYAGE_API_KEY}
    ports:
      - "8080:8080"  # Only needed for HTTP transport
    depends_on:
      - qdrant
      - neo4j
    # For stdio, connect via docker exec
    # For HTTP, access via localhost:8080
```

---

*This implementation uses the native MCP SDK as requested, with explicit tool definitions, manual transport configuration, and no FastMCP abstraction.*

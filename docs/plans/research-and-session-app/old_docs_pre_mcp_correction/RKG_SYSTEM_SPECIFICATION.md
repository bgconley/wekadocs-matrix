# Research Knowledge Graph (RKG) System Specification

## Executive Summary

A modular system for capturing, embedding, storing, and retrieving research context from agentic coding sessions. The system integrates with Brave Search and Firecrawl MCP tools during live coding sessions, stores content in a hybrid Qdrant (vector) + Neo4j (graph) architecture, and provides a native macOS Swift application for exploration and retrieval.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC CODING INTERFACES                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   Claude Code   │    │  OpenAI Codex   │    │   Future IDEs   │          │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘          │
│           │                      │                      │                    │
│           └──────────────────────┼──────────────────────┘                    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        RKG MCP SERVER                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   Capture   │  │   Process   │  │    Store    │  │  Retrieve   │   │  │
│  │  │   Tools     │  │   Tools     │  │   Tools     │  │   Tools     │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │           │              │               │               │             │  │
│  │           │              ▼               │               │             │  │
│  │           │     ┌─────────────────┐      │               │             │  │
│  │           │     │ Embedding Layer │      │               │             │  │
│  │           │     │ ┌─────────────┐ │      │               │             │  │
│  │           │     │ │ Voyage-3    │ │      │               │             │  │
│  │           │     │ │ (Primary)   │ │      │               │             │  │
│  │           │     │ └─────────────┘ │      │               │             │  │
│  │           │     │ ┌─────────────┐ │      │               │             │  │
│  │           │     │ │ Local Model │ │      │               │             │  │
│  │           │     │ │ (Optional)  │ │      │               │             │  │
│  │           │     │ └─────────────┘ │      │               │             │  │
│  │           │     └────────┬────────┘      │               │             │  │
│  │           │              │               │               │             │  │
│  └───────────┼──────────────┼───────────────┼───────────────┼─────────────┘  │
└─────────────┼──────────────┼───────────────┼───────────────┼─────────────────┘
              │              │               │               │
              ▼              ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCKER INFRASTRUCTURE                               │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐         │
│  │         QDRANT              │    │          NEO4J              │         │
│  │  ┌─────────────────────┐    │    │  ┌─────────────────────┐    │         │
│  │  │   Dense Vectors     │    │    │  │    Document Nodes   │    │         │
│  │  │   (voyage-3-large)  │    │    │  │    Source Nodes     │    │         │
│  │  ├─────────────────────┤    │    │  │    Session Nodes    │    │         │
│  │  │   Sparse Vectors    │    │    │  │    Insight Nodes    │    │         │
│  │  │   (BM25/SPLADE)     │    │    │  │    Entity Nodes     │    │         │
│  │  └─────────────────────┘    │    │  └─────────────────────┘    │         │
│  └─────────────────────────────┘    └─────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
              │              │               │               │
              └──────────────┼───────────────┼───────────────┘
                             │               │
                             ▼               ▼
              ┌─────────────────────────────────────────────┐
              │           macOS SWIFT APPLICATION           │
              │  ┌───────────┐ ┌───────────┐ ┌───────────┐  │
              │  │  Search   │ │   Tree    │ │  Document │  │
              │  │   View    │ │   View    │ │   View    │  │
              │  └───────────┘ └───────────┘ └───────────┘  │
              └─────────────────────────────────────────────┘
```

---

## 2. Technology Stack Decisions

### 2.1 Language Selection: Python (with optional Rust acceleration)

**Primary Language: Python 3.11+**

Rationale:
- Official MCP Python SDK is mature (v1.24.0+) with full feature support
- Native async support aligns well with MCP's async architecture
- Excellent ecosystem for Qdrant (`qdrant-client`), Neo4j (`neo4j`), and Voyage AI (`voyageai`)
- Rapid prototyping and iteration capabilities
- Strong community support for all required libraries

**Optional Rust Acceleration via PyO3:**
Reserved for performance-critical components if profiling identifies bottlenecks:
- Custom text chunking/parsing at high throughput
- Batch embedding preprocessing
- Custom sparse vector generation (BM25/SPLADE)

**Implementation Strategy:**
```python
# Python module structure with optional Rust acceleration
rkg_mcp/
├── core/
│   ├── __init__.py
│   ├── embeddings.py      # Python implementation
│   └── _embeddings_rs/    # Optional Rust acceleration (PyO3)
│       ├── Cargo.toml
│       └── src/lib.rs
```

### 2.2 Infrastructure Components

| Component | Technology | Docker Image | Notes |
|-----------|------------|--------------|-------|
| Vector DB | Qdrant | `qdrant/qdrant:v1.12.0` | ARM64 & x86 support |
| Graph DB | Neo4j | `neo4j:5.25-community` | ARM64 & x86 support |
| MCP Server | Python | Custom Dockerfile | Multi-arch build |
| Embeddings | Voyage AI | API (voyage-3-large) | Remote API |
| Reranking | Voyage AI | API (rerank-2.5) | Remote API |
| Local Embeddings | Optional | sentence-transformers | For offline/cost reduction |

---

## 3. Schema Design

### 3.1 Qdrant Collection Schema

```python
# Collection: research_documents
{
    "name": "research_documents",
    "vectors": {
        "dense": {
            "size": 1024,  # voyage-3-large dimension
            "distance": "Cosine",
            "on_disk": False
        }
    },
    "sparse_vectors": {
        "sparse": {
            "modifier": "idf"  # BM25-style IDF weighting
        }
    },
    "payload_schema": {
        "document_id": "keyword",
        "session_id": "keyword",
        "project": "keyword",
        "source_type": "keyword",  # "brave_search", "firecrawl", "insight", "session_transcript"
        "source_url": "keyword",
        "title": "text",
        "content_hash": "keyword",
        "chunk_index": "integer",
        "total_chunks": "integer",
        "created_at": "datetime",
        "agentic_interface": "keyword",  # "claude_code", "openai_codex"
        "tags": "keyword[]",
        "metadata": "json"
    }
}

# Collection: session_insights
{
    "name": "session_insights",
    "vectors": {
        "dense": {
            "size": 1024,
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse": {
            "modifier": "idf"
        }
    },
    "payload_schema": {
        "insight_id": "keyword",
        "session_id": "keyword",
        "project": "keyword",
        "insight_type": "keyword",  # "conclusion", "decision", "learning", "question"
        "content": "text",
        "source_document_ids": "keyword[]",
        "created_at": "datetime",
        "confidence": "float"
    }
}
```

### 3.2 Neo4j Graph Schema

```cypher
// Node Types

// Document: Represents a captured research document
(:Document {
    id: STRING,              // UUID
    title: STRING,
    url: STRING,
    source_type: STRING,     // brave_search, firecrawl, manual
    content_preview: STRING, // First 500 chars
    content_hash: STRING,
    word_count: INTEGER,
    created_at: DATETIME,
    qdrant_ids: [STRING]     // References to Qdrant point IDs for chunks
})

// Session: Represents an agentic coding session
(:Session {
    id: STRING,              // UUID
    project: STRING,
    agentic_interface: STRING,  // claude_code, openai_codex
    started_at: DATETIME,
    ended_at: DATETIME,
    summary: STRING,
    transcript_path: STRING  // Path to original JSONL
})

// Project: Groups sessions and documents
(:Project {
    id: STRING,
    name: STRING,
    path: STRING,            // Filesystem path
    description: STRING,
    created_at: DATETIME
})

// Insight: Conclusions/learnings derived from research
(:Insight {
    id: STRING,
    insight_type: STRING,    // conclusion, decision, learning, question
    content: STRING,
    confidence: FLOAT,
    created_at: DATETIME,
    qdrant_id: STRING
})

// Entity: Named entities extracted from content
(:Entity {
    id: STRING,
    name: STRING,
    entity_type: STRING,     // person, organization, technology, concept
    description: STRING
})

// Tag: Organizational tags
(:Tag {
    name: STRING
})

// Source: External sources (websites, APIs)
(:Source {
    domain: STRING,
    name: STRING,
    reliability_score: FLOAT,
    last_accessed: DATETIME
})

// Relationship Types

// Session relationships
(:Session)-[:BELONGS_TO]->(:Project)
(:Session)-[:CAPTURED]->(:Document)
(:Session)-[:PRODUCED]->(:Insight)

// Document relationships
(:Document)-[:FROM_SOURCE]->(:Source)
(:Document)-[:MENTIONS]->(:Entity)
(:Document)-[:TAGGED_WITH]->(:Tag)
(:Document)-[:RELATED_TO]->(:Document)  // Semantic similarity above threshold

// Insight relationships
(:Insight)-[:DERIVED_FROM]->(:Document)
(:Insight)-[:SUPPORTS]->(:Insight)
(:Insight)-[:CONTRADICTS]->(:Insight)
(:Insight)-[:MENTIONS]->(:Entity)

// Entity relationships
(:Entity)-[:RELATED_TO]->(:Entity)

// Indices for performance
CREATE INDEX document_id_idx FOR (d:Document) ON (d.id);
CREATE INDEX session_id_idx FOR (s:Session) ON (s.id);
CREATE INDEX project_name_idx FOR (p:Project) ON (p.name);
CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name);
CREATE INDEX tag_name_idx FOR (t:Tag) ON (t.name);
CREATE FULLTEXT INDEX document_content_idx FOR (d:Document) ON EACH [d.title, d.content_preview];
```

---

## 4. MCP Server Design

### 4.1 Project Structure

```
rkg-mcp-server/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── README.md
├── CLAUDE.md                    # Instructions for Claude Code
├── src/
│   └── rkg_mcp/
│       ├── __init__.py
│       ├── server.py            # Main MCP server entry point
│       ├── config.py            # Configuration management
│       │
│       ├── transports/          # Transport implementations
│       │   ├── __init__.py
│       │   ├── stdio.py         # stdio transport
│       │   └── http.py          # Streamable HTTP transport
│       │
│       ├── tools/               # MCP Tool definitions
│       │   ├── __init__.py
│       │   ├── capture.py       # Content capture tools
│       │   ├── process.py       # Content processing tools
│       │   ├── store.py         # Storage tools
│       │   ├── retrieve.py      # Retrieval tools
│       │   └── session.py       # Session management tools
│       │
│       ├── embeddings/          # Embedding providers (modular)
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract base class
│       │   ├── voyage.py        # Voyage AI provider
│       │   ├── local.py         # Local model provider
│       │   └── factory.py       # Provider factory
│       │
│       ├── rerankers/           # Reranker providers (modular)
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract base class
│       │   ├── voyage.py        # Voyage rerank-2.5
│       │   └── factory.py       # Provider factory
│       │
│       ├── storage/             # Storage backends
│       │   ├── __init__.py
│       │   ├── qdrant.py        # Qdrant operations
│       │   ├── neo4j.py         # Neo4j operations
│       │   └── hybrid.py        # Hybrid query orchestration
│       │
│       ├── parsers/             # Content parsers
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract base class
│       │   ├── brave_search.py  # Brave search results parser
│       │   ├── firecrawl.py     # Firecrawl markdown parser
│       │   ├── claude_session.py  # Claude Code JSONL parser
│       │   ├── codex_session.py   # OpenAI Codex JSONL parser
│       │   └── chunker.py       # Text chunking utilities
│       │
│       └── models/              # Data models
│           ├── __init__.py
│           ├── document.py
│           ├── session.py
│           ├── insight.py
│           └── entities.py
│
├── tests/
│   ├── __init__.py
│   ├── test_tools/
│   ├── test_embeddings/
│   ├── test_storage/
│   └── test_parsers/
│
└── scripts/
    ├── setup_databases.py       # Initialize Qdrant/Neo4j schemas
    └── import_sessions.py       # Batch import historical sessions
```

### 4.2 MCP Tool Definitions

```python
# src/rkg_mcp/tools/capture.py

"""
RKG MCP Capture Tools

These tools intercept and store research content during agentic coding sessions.
They should be called by the agentic coder when processing Brave Search or
Firecrawl results to persist the research context.
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import hashlib

# Tool Input Schemas

class CaptureSearchResultInput(BaseModel):
    """Input for capturing a Brave Search result."""
    session_id: str = Field(description="Current session identifier")
    project: str = Field(description="Project name or path")
    query: str = Field(description="The search query that produced this result")
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    description: str = Field(description="Search result description/snippet")
    content: Optional[str] = Field(default=None, description="Full page content if scraped")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for organization")

class CaptureScrapedContentInput(BaseModel):
    """Input for capturing Firecrawl scraped content."""
    session_id: str = Field(description="Current session identifier")
    project: str = Field(description="Project name or path")
    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    markdown_content: str = Field(description="Scraped markdown content")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata from Firecrawl")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")

class RecordInsightInput(BaseModel):
    """Input for recording an insight derived from research."""
    session_id: str = Field(description="Current session identifier")
    project: str = Field(description="Project name or path")
    insight_type: str = Field(description="Type: conclusion, decision, learning, question")
    content: str = Field(description="The insight content")
    source_document_ids: List[str] = Field(description="IDs of documents this insight is derived from")
    confidence: Optional[float] = Field(default=0.8, description="Confidence level 0-1")

# Tool Implementations

CAPTURE_TOOLS = [
    Tool(
        name="rkg_capture_search_result",
        description="""
Capture a Brave Search result for persistent storage in the Research Knowledge Graph.

Call this tool when processing search results to preserve the research context.
The content will be embedded and stored for future semantic retrieval.

Example usage:
- After receiving Brave Search results, call this for each relevant result
- Include the full content if available from subsequent scraping
- Add tags to improve future discoverability
""",
        inputSchema=CaptureSearchResultInput.model_json_schema()
    ),
    Tool(
        name="rkg_capture_scraped_content",
        description="""
Capture scraped web content from Firecrawl for persistent storage.

Call this tool after scraping a page with Firecrawl to preserve the full content.
The markdown will be chunked, embedded, and stored with source attribution.

Example usage:
- After Firecrawl returns page content, call this to persist it
- The content will be linked to its source URL for provenance tracking
- Chunks will be created automatically for optimal retrieval
""",
        inputSchema=CaptureScrapedContentInput.model_json_schema()
    ),
    Tool(
        name="rkg_record_insight",
        description="""
Record a conclusion, decision, or learning derived from research.

Call this tool when you've synthesized information from captured documents
into an actionable insight. Insights are linked to their source documents
for full provenance tracking.

Insight types:
- conclusion: A finding based on the research
- decision: A choice made based on the research
- learning: New knowledge gained
- question: An open question for future research

Example usage:
- After analyzing multiple sources, record the key takeaways
- Link each insight to the documents that informed it
- Include confidence level based on source quality/agreement
""",
        inputSchema=RecordInsightInput.model_json_schema()
    )
]
```

```python
# src/rkg_mcp/tools/retrieve.py

"""
RKG MCP Retrieval Tools

These tools enable semantic and graph-based retrieval of stored research.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from mcp.types import Tool

class SemanticSearchInput(BaseModel):
    """Input for semantic search across stored documents."""
    query: str = Field(description="Natural language search query")
    limit: int = Field(default=10, description="Maximum results to return")
    project_filter: Optional[str] = Field(default=None, description="Filter to specific project")
    source_type_filter: Optional[str] = Field(default=None, description="Filter by source type")
    date_from: Optional[str] = Field(default=None, description="Filter by date (ISO format)")
    date_to: Optional[str] = Field(default=None, description="Filter by date (ISO format)")
    use_reranker: bool = Field(default=True, description="Apply Voyage reranker to results")
    search_mode: Literal["semantic", "lexical", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: semantic (dense), lexical (sparse/BM25), or hybrid"
    )

class GraphExploreInput(BaseModel):
    """Input for graph-based exploration."""
    start_node_id: str = Field(description="Starting node ID for exploration")
    relationship_types: Optional[List[str]] = Field(
        default=None,
        description="Types of relationships to traverse"
    )
    max_depth: int = Field(default=2, description="Maximum traversal depth")
    limit: int = Field(default=20, description="Maximum nodes to return")

class GetSessionContextInput(BaseModel):
    """Input for retrieving full session context."""
    session_id: str = Field(description="Session ID to retrieve")
    include_documents: bool = Field(default=True, description="Include captured documents")
    include_insights: bool = Field(default=True, description="Include recorded insights")
    include_transcript: bool = Field(default=False, description="Include session transcript")

class FindRelatedInput(BaseModel):
    """Input for finding related content."""
    document_id: str = Field(description="Document ID to find related content for")
    relationship_type: Literal["semantic", "citation", "entity", "all"] = Field(
        default="all",
        description="Type of relationship to find"
    )
    limit: int = Field(default=10, description="Maximum results")

RETRIEVE_TOOLS = [
    Tool(
        name="rkg_semantic_search",
        description="""
Search the Research Knowledge Graph using natural language.

Supports three search modes:
- semantic: Dense vector similarity (meaning-based)
- lexical: Sparse vector BM25 (keyword-based)
- hybrid: Combines both with Reciprocal Rank Fusion

Results are optionally reranked using Voyage rerank-2.5 for improved relevance.

Example queries:
- "How does parallel file system performance compare to NFS for AI workloads?"
- "Best practices for Neo4j schema design"
- "Embedding models suitable for code retrieval"
""",
        inputSchema=SemanticSearchInput.model_json_schema()
    ),
    Tool(
        name="rkg_graph_explore",
        description="""
Explore the knowledge graph starting from a specific node.

Traverses relationships to discover connected information:
- Documents related to an entity
- Insights derived from a document
- Sessions that captured specific content
- Entity relationships

Useful for understanding the provenance and context of information.
""",
        inputSchema=GraphExploreInput.model_json_schema()
    ),
    Tool(
        name="rkg_get_session_context",
        description="""
Retrieve the full context of a previous coding session.

Returns all documents captured, insights recorded, and optionally
the full session transcript. Useful for resuming work or reviewing
past research.
""",
        inputSchema=GetSessionContextInput.model_json_schema()
    ),
    Tool(
        name="rkg_find_related",
        description="""
Find content related to a specific document.

Relationship types:
- semantic: Similar content by meaning
- citation: Documents that reference each other
- entity: Documents mentioning the same entities
- all: All relationship types

Useful for discovering additional relevant research.
""",
        inputSchema=FindRelatedInput.model_json_schema()
    )
]
```

### 4.3 Server Implementation

```python
# src/rkg_mcp/server.py

"""
RKG MCP Server - Main Entry Point

A Model Context Protocol server for the Research Knowledge Graph system.
Supports both stdio and HTTP transports.
"""

import asyncio
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, CallToolResult,
    ListToolsResult, InitializationOptions
)

from .config import Settings
from .tools.capture import CAPTURE_TOOLS, handle_capture_tool
from .tools.retrieve import RETRIEVE_TOOLS, handle_retrieve_tool
from .tools.session import SESSION_TOOLS, handle_session_tool
from .storage.hybrid import HybridStorage
from .embeddings.factory import create_embedding_provider
from .rerankers.factory import create_reranker_provider

logger = logging.getLogger(__name__)

class RKGServer:
    """Research Knowledge Graph MCP Server."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.server = Server("rkg-mcp-server")
        self.storage: Optional[HybridStorage] = None
        self.embedding_provider = None
        self.reranker_provider = None

        self._register_handlers()

    def _register_handlers(self):
        """Register MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return all available tools."""
            return CAPTURE_TOOLS + RETRIEVE_TOOLS + SESSION_TOOLS

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Handle tool invocations."""
            try:
                # Route to appropriate handler
                if name.startswith("rkg_capture_"):
                    result = await handle_capture_tool(
                        name, arguments, self.storage, self.embedding_provider
                    )
                elif name.startswith("rkg_") and any(
                    name.endswith(suffix) for suffix in
                    ["_search", "_explore", "_context", "_related"]
                ):
                    result = await handle_retrieve_tool(
                        name, arguments, self.storage,
                        self.embedding_provider, self.reranker_provider
                    )
                elif name.startswith("rkg_session_"):
                    result = await handle_session_tool(name, arguments, self.storage)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                        isError=True
                    )

                return CallToolResult(
                    content=[TextContent(type="text", text=result)]
                )

            except Exception as e:
                logger.exception(f"Error handling tool {name}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True
                )

    @asynccontextmanager
    async def lifespan(self):
        """Manage server lifecycle."""
        logger.info("Initializing RKG MCP Server...")

        # Initialize storage
        self.storage = HybridStorage(
            qdrant_url=self.settings.qdrant_url,
            neo4j_uri=self.settings.neo4j_uri,
            neo4j_user=self.settings.neo4j_user,
            neo4j_password=self.settings.neo4j_password
        )
        await self.storage.connect()

        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider(
            self.settings.embedding_provider,
            api_key=self.settings.voyage_api_key
        )

        # Initialize reranker provider
        self.reranker_provider = create_reranker_provider(
            self.settings.reranker_provider,
            api_key=self.settings.voyage_api_key
        )

        logger.info("RKG MCP Server initialized")

        try:
            yield
        finally:
            logger.info("Shutting down RKG MCP Server...")
            await self.storage.disconnect()

    async def run_stdio(self):
        """Run server with stdio transport."""
        async with self.lifespan():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="rkg-mcp-server",
                        server_version="0.1.0",
                        capabilities=self.server.get_capabilities()
                    )
                )

    async def run_http(self, host: str = "0.0.0.0", port: int = 8080):
        """Run server with HTTP transport."""
        from mcp.server.fastmcp import FastMCP

        # Wrap in FastMCP for HTTP support
        app = FastMCP(
            "rkg-mcp-server",
            stateless_http=True,
            json_response=True
        )

        # Copy tool registrations
        for tool in CAPTURE_TOOLS + RETRIEVE_TOOLS + SESSION_TOOLS:
            # Register tools with FastMCP wrapper
            pass  # Implementation details

        async with self.lifespan():
            app.run(transport="streamable-http", host=host, port=port)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RKG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to use"
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")

    args = parser.parse_args()

    server = RKGServer()

    if args.transport == "stdio":
        asyncio.run(server.run_stdio())
    else:
        asyncio.run(server.run_http(args.host, args.port))


if __name__ == "__main__":
    main()
```

### 4.4 Embedding Provider Interface (Modular)

```python
# src/rkg_mcp/embeddings/base.py

"""
Abstract base class for embedding providers.
Enables swapping between Voyage AI, local models, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    vector: List[float]
    model: str
    usage_tokens: Optional[int] = None

class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    async def embed_text(self, text: str, input_type: str = "document") -> EmbeddingResult:
        """Embed a single text string."""
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        input_type: str = "document",
        batch_size: int = 128
    ) -> List[EmbeddingResult]:
        """Embed multiple texts efficiently."""
        pass


# src/rkg_mcp/embeddings/voyage.py

"""Voyage AI embedding provider."""

import voyageai
from typing import List
from .base import EmbeddingProvider, EmbeddingResult

class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embedding provider using voyage-3-large."""

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3-large"):
        self.client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimension = 1024  # voyage-3-large dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_text(self, text: str, input_type: str = "document") -> EmbeddingResult:
        result = self.client.embed(
            texts=[text],
            model=self._model,
            input_type=input_type
        )
        return EmbeddingResult(
            vector=result.embeddings[0],
            model=self._model,
            usage_tokens=result.total_tokens
        )

    async def embed_batch(
        self,
        texts: List[str],
        input_type: str = "document",
        batch_size: int = 128
    ) -> List[EmbeddingResult]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.client.embed(
                texts=batch,
                model=self._model,
                input_type=input_type
            )
            for embedding in result.embeddings:
                results.append(EmbeddingResult(
                    vector=embedding,
                    model=self._model
                ))
        return results


# src/rkg_mcp/embeddings/local.py

"""Local embedding provider using sentence-transformers."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider, EmbeddingResult

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding using sentence-transformers models."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_text(self, text: str, input_type: str = "document") -> EmbeddingResult:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return EmbeddingResult(
            vector=embedding.tolist(),
            model=self._model_name
        )

    async def embed_batch(
        self,
        texts: List[str],
        input_type: str = "document",
        batch_size: int = 32
    ) -> List[EmbeddingResult]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return [
            EmbeddingResult(vector=emb.tolist(), model=self._model_name)
            for emb in embeddings
        ]
```

---

## 5. Session Transcript Parsers

### 5.1 Claude Code Parser

```python
# src/rkg_mcp/parsers/claude_session.py

"""
Parser for Claude Code session transcripts.
Location: ~/.claude/projects/<project_path>/*.jsonl
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Iterator, Optional, List
from dataclasses import dataclass, field

@dataclass
class ClaudeMessage:
    """A single message in a Claude Code session."""
    role: str  # 'user', 'assistant', 'system', 'tool_use', 'tool_result'
    content: str
    timestamp: Optional[datetime] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[str] = None

@dataclass
class ClaudeSession:
    """A complete Claude Code session."""
    session_id: str
    project_path: str
    messages: List[ClaudeMessage] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @property
    def tool_uses(self) -> List[ClaudeMessage]:
        """Get all tool use messages."""
        return [m for m in self.messages if m.role == 'tool_use']

    @property
    def research_tools(self) -> List[ClaudeMessage]:
        """Get Brave Search and Firecrawl tool uses."""
        return [
            m for m in self.tool_uses
            if m.tool_name in ('brave_search', 'firecrawl_scrape', 'brave-search:brave_web_search')
        ]

class ClaudeSessionParser:
    """Parser for Claude Code JSONL session files."""

    CLAUDE_DIR = Path.home() / ".claude" / "projects"

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or self.CLAUDE_DIR

    def find_sessions(self, project_filter: Optional[str] = None) -> Iterator[Path]:
        """Find all session files, optionally filtered by project."""
        if not self.base_dir.exists():
            return

        for jsonl_file in self.base_dir.rglob("*.jsonl"):
            if project_filter is None or project_filter in str(jsonl_file):
                yield jsonl_file

    def parse_session(self, session_path: Path) -> ClaudeSession:
        """Parse a single session JSONL file."""
        messages = []
        session_id = session_path.stem
        project_path = str(session_path.parent.relative_to(self.base_dir))

        with open(session_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    message = self._parse_entry(entry)
                    if message:
                        messages.append(message)
                except json.JSONDecodeError:
                    continue

        session = ClaudeSession(
            session_id=session_id,
            project_path=project_path,
            messages=messages
        )

        # Set timestamps from first/last messages
        if messages:
            session.started_at = messages[0].timestamp
            session.ended_at = messages[-1].timestamp

        return session

    def _parse_entry(self, entry: dict) -> Optional[ClaudeMessage]:
        """Parse a single JSONL entry into a ClaudeMessage."""
        # Handle different entry formats
        if 'type' in entry:
            entry_type = entry['type']

            if entry_type == 'user':
                return ClaudeMessage(
                    role='user',
                    content=entry.get('message', ''),
                    timestamp=self._parse_timestamp(entry.get('timestamp'))
                )

            elif entry_type == 'assistant':
                return ClaudeMessage(
                    role='assistant',
                    content=entry.get('message', ''),
                    timestamp=self._parse_timestamp(entry.get('timestamp'))
                )

            elif entry_type == 'tool_use':
                return ClaudeMessage(
                    role='tool_use',
                    content='',
                    tool_name=entry.get('name'),
                    tool_input=entry.get('input', {}),
                    timestamp=self._parse_timestamp(entry.get('timestamp'))
                )

            elif entry_type == 'tool_result':
                return ClaudeMessage(
                    role='tool_result',
                    content='',
                    tool_name=entry.get('tool_use_id'),
                    tool_result=entry.get('content'),
                    timestamp=self._parse_timestamp(entry.get('timestamp'))
                )

        return None

    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    def extract_research_content(self, session: ClaudeSession) -> List[dict]:
        """Extract research-related content from a session for ingestion."""
        research_items = []

        for i, msg in enumerate(session.messages):
            if msg.role == 'tool_use' and msg.tool_name in (
                'brave_search',
                'brave-search:brave_web_search',
                'firecrawl_scrape',
                'firecrawl:firecrawl_scrape'
            ):
                # Find corresponding tool result
                tool_result = None
                for j in range(i + 1, len(session.messages)):
                    if session.messages[j].role == 'tool_result':
                        tool_result = session.messages[j].tool_result
                        break

                research_items.append({
                    'tool': msg.tool_name,
                    'input': msg.tool_input,
                    'result': tool_result,
                    'timestamp': msg.timestamp,
                    'session_id': session.session_id,
                    'project': session.project_path
                })

        return research_items
```

### 5.2 OpenAI Codex Parser

```python
# src/rkg_mcp/parsers/codex_session.py

"""
Parser for OpenAI Codex CLI session transcripts.
Location: ~/.codex/sessions/*.jsonl or history.jsonl
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Iterator, Optional, List
from dataclasses import dataclass, field

@dataclass
class CodexMessage:
    """A single message in an OpenAI Codex session."""
    role: str  # 'user', 'assistant', 'tool', 'system'
    content: str
    timestamp: Optional[datetime] = None
    function_call: Optional[dict] = None
    tool_calls: Optional[List[dict]] = None

@dataclass
class CodexSession:
    """A complete OpenAI Codex session."""
    session_id: str
    messages: List[CodexMessage] = field(default_factory=list)
    model: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @property
    def tool_calls(self) -> List[dict]:
        """Get all tool calls from messages."""
        calls = []
        for msg in self.messages:
            if msg.tool_calls:
                calls.extend(msg.tool_calls)
        return calls

class CodexSessionParser:
    """Parser for OpenAI Codex JSONL session files."""

    CODEX_DIR = Path.home() / ".codex"

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or self.CODEX_DIR

    def find_sessions(self) -> Iterator[Path]:
        """Find all session files."""
        sessions_dir = self.base_dir / "sessions"
        if sessions_dir.exists():
            yield from sessions_dir.glob("*.jsonl")

        # Also check history.jsonl
        history_file = self.base_dir / "history.jsonl"
        if history_file.exists():
            yield history_file

    def parse_session(self, session_path: Path) -> CodexSession:
        """Parse a single session JSONL file."""
        messages = []
        session_id = session_path.stem
        model = None

        with open(session_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)

                    # Extract model if present
                    if 'model' in entry and model is None:
                        model = entry['model']

                    message = self._parse_entry(entry)
                    if message:
                        messages.append(message)

                except json.JSONDecodeError:
                    continue

        session = CodexSession(
            session_id=session_id,
            messages=messages,
            model=model
        )

        if messages:
            session.started_at = messages[0].timestamp
            session.ended_at = messages[-1].timestamp

        return session

    def _parse_entry(self, entry: dict) -> Optional[CodexMessage]:
        """Parse a single JSONL entry."""
        role = entry.get('role')
        if not role:
            return None

        return CodexMessage(
            role=role,
            content=entry.get('content', ''),
            timestamp=self._parse_timestamp(entry.get('timestamp')),
            function_call=entry.get('function_call'),
            tool_calls=entry.get('tool_calls')
        )

    def _parse_timestamp(self, ts) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not ts:
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        try:
            return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
```

---

## 6. macOS Swift Application Design

### 6.1 Architecture

```
RKGExplorer/
├── RKGExplorer.xcodeproj
├── RKGExplorer/
│   ├── App/
│   │   ├── RKGExplorerApp.swift       # Main app entry
│   │   └── ContentView.swift          # Root view
│   │
│   ├── Views/
│   │   ├── Sidebar/
│   │   │   ├── SidebarView.swift      # Navigation sidebar
│   │   │   ├── ProjectTreeView.swift  # Project hierarchy
│   │   │   ├── SessionListView.swift  # Sessions list
│   │   │   └── TagCloudView.swift     # Tag navigation
│   │   │
│   │   ├── Search/
│   │   │   ├── SearchView.swift       # Search interface
│   │   │   ├── SearchResultsView.swift
│   │   │   ├── FilterPanel.swift
│   │   │   └── SearchHistoryView.swift
│   │   │
│   │   ├── Document/
│   │   │   ├── DocumentDetailView.swift
│   │   │   ├── DocumentPreviewView.swift
│   │   │   ├── MarkdownRenderer.swift
│   │   │   └── MetadataPanel.swift
│   │   │
│   │   ├── Graph/
│   │   │   ├── GraphExplorerView.swift  # Interactive graph viz
│   │   │   ├── GraphNodeView.swift
│   │   │   └── GraphLayoutEngine.swift
│   │   │
│   │   └── Insights/
│   │       ├── InsightListView.swift
│   │       └── InsightDetailView.swift
│   │
│   ├── ViewModels/
│   │   ├── SidebarViewModel.swift
│   │   ├── SearchViewModel.swift
│   │   ├── DocumentViewModel.swift
│   │   ├── GraphViewModel.swift
│   │   └── InsightViewModel.swift
│   │
│   ├── Models/
│   │   ├── Document.swift
│   │   ├── Session.swift
│   │   ├── Project.swift
│   │   ├── Insight.swift
│   │   ├── SearchResult.swift
│   │   └── GraphNode.swift
│   │
│   ├── Services/
│   │   ├── APIClient.swift            # HTTP client for MCP server
│   │   ├── QdrantClient.swift         # Direct Qdrant queries
│   │   ├── Neo4jClient.swift          # Direct Neo4j queries
│   │   └── SearchService.swift        # Search orchestration
│   │
│   ├── Utilities/
│   │   ├── DateFormatters.swift
│   │   ├── MarkdownParser.swift
│   │   └── ExportManager.swift
│   │
│   └── Resources/
│       ├── Assets.xcassets
│       └── Localizable.strings
│
└── RKGExplorerTests/
```

### 6.2 Key SwiftUI Views

```swift
// ContentView.swift - Main navigation structure

import SwiftUI

struct ContentView: View {
    @StateObject private var sidebarVM = SidebarViewModel()
    @StateObject private var searchVM = SearchViewModel()

    var body: some View {
        NavigationSplitView {
            SidebarView(viewModel: sidebarVM)
                .navigationSplitViewColumnWidth(min: 200, ideal: 250, max: 300)
        } content: {
            ContentListView(
                selection: sidebarVM.selectedCategory,
                searchVM: searchVM
            )
            .navigationSplitViewColumnWidth(min: 300, ideal: 400)
        } detail: {
            DetailView(selection: sidebarVM.selectedItem)
        }
        .searchable(
            text: $searchVM.searchQuery,
            placement: .toolbar,
            prompt: "Search documents, insights, sessions..."
        )
        .onSubmit(of: .search) {
            Task {
                await searchVM.performSearch()
            }
        }
    }
}

// SidebarView.swift - Tree navigation

struct SidebarView: View {
    @ObservedObject var viewModel: SidebarViewModel

    var body: some View {
        List(selection: $viewModel.selectedCategory) {
            // Quick Access
            Section("Quick Access") {
                NavigationLink(value: Category.recentSessions) {
                    Label("Recent Sessions", systemImage: "clock")
                }
                NavigationLink(value: Category.recentDocuments) {
                    Label("Recent Documents", systemImage: "doc.text")
                }
                NavigationLink(value: Category.insights) {
                    Label("Insights", systemImage: "lightbulb")
                }
            }

            // Projects Tree
            Section("Projects") {
                ForEach(viewModel.projects) { project in
                    DisclosureGroup {
                        ForEach(project.sessions) { session in
                            NavigationLink(value: Category.session(session.id)) {
                                Label(session.displayName, systemImage: "terminal")
                            }
                        }
                    } label: {
                        Label(project.name, systemImage: "folder")
                    }
                }
            }

            // By Date
            Section("By Date") {
                ForEach(viewModel.dateGroups) { group in
                    NavigationLink(value: Category.dateRange(group.range)) {
                        Label(group.title, systemImage: "calendar")
                    }
                }
            }

            // By Interface
            Section("By Interface") {
                NavigationLink(value: Category.interface(.claudeCode)) {
                    Label("Claude Code", systemImage: "cpu")
                }
                NavigationLink(value: Category.interface(.openAICodex)) {
                    Label("OpenAI Codex", systemImage: "brain")
                }
            }

            // Tags
            Section("Tags") {
                ForEach(viewModel.tags) { tag in
                    NavigationLink(value: Category.tag(tag.name)) {
                        Label(tag.name, systemImage: "tag")
                            .badge(tag.count)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("RKG Explorer")
    }
}

// SearchView.swift - Advanced search interface

struct SearchView: View {
    @ObservedObject var viewModel: SearchViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Search mode selector
            Picker("Search Mode", selection: $viewModel.searchMode) {
                Text("Hybrid").tag(SearchMode.hybrid)
                Text("Semantic").tag(SearchMode.semantic)
                Text("Lexical").tag(SearchMode.lexical)
            }
            .pickerStyle(.segmented)
            .padding()

            // Filters
            DisclosureGroup("Filters", isExpanded: $viewModel.filtersExpanded) {
                FilterPanel(viewModel: viewModel)
            }
            .padding(.horizontal)

            Divider()

            // Results
            if viewModel.isSearching {
                ProgressView("Searching...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.results.isEmpty {
                ContentUnavailableView(
                    "No Results",
                    systemImage: "magnifyingglass",
                    description: Text("Try different search terms or filters")
                )
            } else {
                List(viewModel.results) { result in
                    SearchResultRow(result: result)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            viewModel.selectResult(result)
                        }
                }
                .listStyle(.plain)
            }
        }
    }
}

// SearchResultRow.swift

struct SearchResultRow: View {
    let result: SearchResult

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: result.sourceType.iconName)
                    .foregroundColor(result.sourceType.color)
                Text(result.title)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                Text(result.score, format: .percent)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(result.snippet)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(2)

            HStack {
                Label(result.projectName, systemImage: "folder")
                Spacer()
                Text(result.date, style: .relative)
            }
            .font(.caption)
            .foregroundColor(.tertiary)
        }
        .padding(.vertical, 4)
    }
}
```

### 6.3 API Client

```swift
// Services/APIClient.swift

import Foundation

actor RKGAPIClient {
    private let baseURL: URL
    private let session: URLSession

    init(baseURL: URL = URL(string: "http://localhost:8080")!) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        self.session = URLSession(configuration: config)
    }

    // MARK: - Search

    func search(
        query: String,
        mode: SearchMode = .hybrid,
        limit: Int = 20,
        filters: SearchFilters? = nil
    ) async throws -> [SearchResult] {
        let request = SemanticSearchRequest(
            query: query,
            limit: limit,
            searchMode: mode.rawValue,
            projectFilter: filters?.project,
            sourceTypeFilter: filters?.sourceType,
            dateFrom: filters?.dateFrom?.ISO8601Format(),
            dateTo: filters?.dateTo?.ISO8601Format(),
            useReranker: true
        )

        return try await callTool("rkg_semantic_search", input: request)
    }

    // MARK: - Documents

    func getDocument(id: String) async throws -> Document {
        let request = GetDocumentRequest(documentId: id)
        return try await callTool("rkg_get_document", input: request)
    }

    func getRelatedDocuments(documentId: String, limit: Int = 10) async throws -> [Document] {
        let request = FindRelatedRequest(
            documentId: documentId,
            relationshipType: "all",
            limit: limit
        )
        return try await callTool("rkg_find_related", input: request)
    }

    // MARK: - Sessions

    func getSessions(projectFilter: String? = nil) async throws -> [Session] {
        let request = ListSessionsRequest(projectFilter: projectFilter)
        return try await callTool("rkg_list_sessions", input: request)
    }

    func getSessionContext(sessionId: String) async throws -> SessionContext {
        let request = GetSessionContextRequest(
            sessionId: sessionId,
            includeDocuments: true,
            includeInsights: true,
            includeTranscript: false
        )
        return try await callTool("rkg_get_session_context", input: request)
    }

    // MARK: - Graph Exploration

    func exploreGraph(
        startNodeId: String,
        maxDepth: Int = 2,
        relationshipTypes: [String]? = nil
    ) async throws -> GraphData {
        let request = GraphExploreRequest(
            startNodeId: startNodeId,
            maxDepth: maxDepth,
            relationshipTypes: relationshipTypes,
            limit: 50
        )
        return try await callTool("rkg_graph_explore", input: request)
    }

    // MARK: - Private

    private func callTool<T: Encodable, R: Decodable>(
        _ toolName: String,
        input: T
    ) async throws -> R {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("mcp"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let mcpRequest = MCPToolCallRequest(
            jsonrpc: "2.0",
            id: UUID().uuidString,
            method: "tools/call",
            params: MCPToolCallParams(
                name: toolName,
                arguments: input
            )
        )

        urlRequest.httpBody = try JSONEncoder().encode(mcpRequest)

        let (data, response) = try await session.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              200..<300 ~= httpResponse.statusCode else {
            throw APIError.invalidResponse
        }

        let mcpResponse = try JSONDecoder().decode(MCPToolCallResponse<R>.self, from: data)

        if let error = mcpResponse.error {
            throw APIError.mcpError(error.message)
        }

        guard let result = mcpResponse.result else {
            throw APIError.noResult
        }

        return result
    }
}

// MARK: - Request/Response Types

struct MCPToolCallRequest<T: Encodable>: Encodable {
    let jsonrpc: String
    let id: String
    let method: String
    let params: MCPToolCallParams<T>
}

struct MCPToolCallParams<T: Encodable>: Encodable {
    let name: String
    let arguments: T
}

struct MCPToolCallResponse<T: Decodable>: Decodable {
    let jsonrpc: String
    let id: String
    let result: T?
    let error: MCPError?
}

struct MCPError: Decodable {
    let code: Int
    let message: String
}

enum APIError: Error {
    case invalidResponse
    case mcpError(String)
    case noResult
}
```

---

## 7. Docker Infrastructure

### 7.1 docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    container_name: rkg-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5.25-community
    container_name: rkg-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    environment:
      - NEO4J_AUTH=neo4j/your_secure_password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

  mcp-server:
    build:
      context: ./rkg-mcp-server
      dockerfile: Dockerfile
    container_name: rkg-mcp-server
    ports:
      - "8080:8080"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_secure_password
      - VOYAGE_API_KEY=${VOYAGE_API_KEY}
      - EMBEDDING_PROVIDER=voyage
      - RERANKER_PROVIDER=voyage
      - LOG_LEVEL=INFO
    depends_on:
      qdrant:
        condition: service_healthy
      neo4j:
        condition: service_healthy
    volumes:
      - ${HOME}/.claude:/app/claude_sessions:ro
      - ${HOME}/.codex:/app/codex_sessions:ro

volumes:
  qdrant_data:
  neo4j_data:
  neo4j_logs:
```

### 7.2 MCP Server Dockerfile

```dockerfile
# rkg-mcp-server/Dockerfile

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim

WORKDIR /app

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Copy application code
COPY src/ ./src/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose HTTP port
EXPOSE 8080

# Default to HTTP transport
ENV TRANSPORT=http
ENV HOST=0.0.0.0
ENV PORT=8080

CMD ["python", "-m", "rkg_mcp.server", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Set up Docker infrastructure (Qdrant, Neo4j)
- [ ] Create MCP server skeleton with transport support
- [ ] Implement embedding provider abstraction (Voyage + local)
- [ ] Implement reranker provider abstraction
- [ ] Create schema initialization scripts

### Phase 2: Capture & Storage (Week 3-4)
- [ ] Implement capture tools (search results, scraped content)
- [ ] Implement storage layer (Qdrant + Neo4j writes)
- [ ] Implement chunking strategies
- [ ] Create insight recording tools
- [ ] Test with live agentic coding sessions

### Phase 3: Retrieval & Query (Week 5-6)
- [ ] Implement hybrid search (dense + sparse + reranking)
- [ ] Implement graph exploration tools
- [ ] Implement session context retrieval
- [ ] Implement related content discovery
- [ ] Performance tuning and optimization

### Phase 4: Session Ingestion (Week 7)
- [ ] Implement Claude Code session parser
- [ ] Implement OpenAI Codex session parser
- [ ] Create batch import scripts
- [ ] Implement incremental sync

### Phase 5: macOS Application (Week 8-10)
- [ ] Set up Xcode project with SwiftUI
- [ ] Implement navigation structure
- [ ] Implement search interface
- [ ] Implement document viewer
- [ ] Implement graph visualization
- [ ] Polish UI/UX

### Phase 6: Testing & Documentation (Week 11-12)
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] User documentation
- [ ] Developer documentation
- [ ] Deployment guides

---

## 9. Configuration

### 9.1 Environment Variables

```bash
# .env

# Voyage AI
VOYAGE_API_KEY=your_voyage_api_key

# Qdrant
QDRANT_URL=http://localhost:6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# Provider Selection
EMBEDDING_PROVIDER=voyage  # voyage, local
EMBEDDING_MODEL=voyage-3-large  # or BAAI/bge-large-en-v1.5 for local
RERANKER_PROVIDER=voyage
RERANKER_MODEL=rerank-2.5

# Server Configuration
LOG_LEVEL=INFO
TRANSPORT=stdio  # stdio, http
HTTP_HOST=0.0.0.0
HTTP_PORT=8080

# Session Paths (optional, defaults to ~/.claude and ~/.codex)
CLAUDE_SESSIONS_PATH=~/.claude/projects
CODEX_SESSIONS_PATH=~/.codex/sessions
```

### 9.2 MCP Client Configuration

For Claude Code (`~/.config/claude/mcp.json`):

```json
{
  "mcpServers": {
    "rkg": {
      "command": "docker",
      "args": [
        "exec", "-i", "rkg-mcp-server",
        "python", "-m", "rkg_mcp.server", "--transport", "stdio"
      ]
    }
  }
}
```

For HTTP transport:

```json
{
  "mcpServers": {
    "rkg": {
      "transport": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

---

## 10. Future Enhancements

1. **Additional Embedder Support**
   - OpenAI text-embedding-3-large
   - Cohere embed-v3
   - Custom fine-tuned models

2. **Enhanced Graph Features**
   - Automatic entity extraction (NER)
   - Citation graph construction
   - Concept clustering

3. **Collaboration Features**
   - Shared knowledge bases
   - Team annotations
   - Export/import capabilities

4. **Analytics Dashboard**
   - Research activity metrics
   - Knowledge coverage visualization
   - Session productivity insights

5. **iOS/iPadOS Companion App**
   - Read-only access to knowledge base
   - Quick capture of ideas/insights
   - Sync with macOS app

---

## 11. References

- [MCP Python SDK Documentation](https://modelcontextprotocol.github.io/python-sdk/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/sparse-vectors/)
- [Neo4j + Vectors](https://neo4j.com/blog/developer/vectors-graphs-better-together/)
- [Voyage AI API](https://docs.voyageai.com/)
- [SwiftUI Navigation](https://developer.apple.com/tutorials/swiftui)

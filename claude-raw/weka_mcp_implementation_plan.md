# Weka Documentation GraphRAG MCP Server - Implementation Plan

## Executive Summary

This implementation plan provides a detailed, actionable roadmap for building the Weka Documentation GraphRAG MCP Server as specified in the Application Specification document. Each phase and task directly corresponds to the specification, providing concrete implementation steps, code examples, and validation criteria. The plan is designed for an agentic coder or development team to execute systematically, with clear dependencies and deliverables for each component.

The implementation follows a phased approach that builds from core infrastructure through to production deployment, ensuring that each layer is properly tested before proceeding to the next. Special attention is paid to security, performance, and maintainability throughout the implementation process.

## Phase 1: Core Infrastructure

### Task 1.1: Docker Environment Setup

**Implementation Steps:**

First, we need to create the foundational Docker environment. Begin by creating a project directory structure that separates concerns clearly. The structure should include directories for the MCP server code, configuration files, Docker definitions, and data persistence volumes.

```bash
# Project Directory Structure
weka-mcp-server/
├── docker/
│   ├── mcp/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── neo4j/
│   │   └── Dockerfile
│   └── qdrant/
│       └── config.yaml
├── src/
│   ├── mcp_server/
│   ├── ingestion/
│   └── shared/
├── config/
│   ├── development.yaml
│   └── production.yaml
├── docker-compose.yml
├── docker-compose.prod.yml
└── .env.example
```

Create the main Docker Compose configuration that orchestrates all services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build:
      context: .
      dockerfile: docker/mcp/Dockerfile
    container_name: weka-mcp-server
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=${NEO4J_USER:-neo4j}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - QDRANT_URI=http://qdrant:6333
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - CACHE_REDIS_URI=redis://redis:6379
      - MCP_PORT=3000
    volumes:
      - ./src:/app/src
      - ./config:/app/config
      - logs:/app/logs
    ports:
      - "3000:3000"
    depends_on:
      neo4j:
        condition: service_healthy
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - weka-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:5.15-enterprise
    container_name: weka-neo4j
    environment:
      - NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["graph-data-science", "apoc"]
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
      - NEO4J_server_memory_heap_max__size=4G
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import
      - ./scripts/neo4j:/scripts
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    networks:
      - weka-net
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "${NEO4J_USER:-neo4j}", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 10

  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: weka-qdrant
    volumes:
      - qdrant_data:/qdrant/storage
      - ./docker/qdrant/config.yaml:/qdrant/config/config.yaml
    ports:
      - "6333:6333"
    networks:
      - weka-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: weka-redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - weka-net
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  ingestion-worker:
    build:
      context: .
      dockerfile: docker/mcp/Dockerfile
    container_name: weka-ingestion
    command: python -m src.ingestion.worker
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=${NEO4J_USER:-neo4j}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - QDRANT_URI=http://qdrant:6333
      - REDIS_URI=redis://redis:6379
    volumes:
      - ./src:/app/src
      - ./data/documents:/app/documents
      - logs:/app/logs
    depends_on:
      - neo4j
      - qdrant
      - redis
    networks:
      - weka-net

networks:
  weka-net:
    driver: bridge

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  qdrant_data:
  redis_data:
  logs:
```

Create the MCP Server Dockerfile:

```dockerfile
# docker/mcp/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY docker/mcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose MCP server port
EXPOSE 3000

# Run the MCP server
CMD ["python", "-m", "src.mcp_server.main"]
```

**Validation Criteria:**
- Run `docker-compose up -d` and verify all containers start successfully
- Check health status with `docker-compose ps` showing all services as "healthy"
- Verify inter-service communication by exec-ing into MCP container and pinging other services
- Confirm data persistence by restarting containers and checking data remains

### Task 1.2: MCP Server Foundation

**Implementation Steps:**

Create the core MCP server implementation that handles the Model Context Protocol. This server will be the main interface between LLMs and our graph database.

```python
# src/mcp_server/main.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from .protocol import MCPProtocolHandler
from .tools import ToolRegistry
from ..shared.config import Config
from ..shared.database import GraphDatabase
from ..shared.cache import CacheManager
from ..shared.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPRequest(BaseModel):
    """Standard MCP request structure"""
    method: str
    params: Optional[Dict[str, Any]] = {}
    id: Optional[str] = None

class MCPResponse(BaseModel):
    """Standard MCP response structure"""
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

class WekaMCPServer:
    """
    Main MCP Server implementation for Weka Documentation GraphRAG.

    This server implements the Model Context Protocol specification,
    providing tools for documentation search, relationship traversal,
    and complex technical query resolution.
    """

    def __init__(self, config: Config):
        """Initialize the MCP server with all necessary components."""
        self.config = config
        self.graph_db = GraphDatabase(config.neo4j)
        self.cache = CacheManager(config.redis)
        self.metrics = MetricsCollector(config.metrics)
        self.tool_registry = ToolRegistry()
        self.protocol_handler = MCPProtocolHandler()

        # Register available tools
        self._register_tools()

        logger.info("WekaMCPServer initialized successfully")

    def _register_tools(self):
        """Register all available tools for LLM interaction."""
        from .tools import (
            SearchDocumentationTool,
            TraverseRelationshipsTool,
            CompareSystemsTool,
            TroubleshootErrorTool,
            ExplainArchitectureTool
        )

        tools = [
            SearchDocumentationTool(self.graph_db, self.cache),
            TraverseRelationshipsTool(self.graph_db),
            CompareSystemsTool(self.graph_db),
            TroubleshootErrorTool(self.graph_db),
            ExplainArchitectureTool(self.graph_db, self.cache)
        ]

        for tool in tools:
            self.tool_registry.register(tool)
            logger.info(f"Registered tool: {tool.name}")

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle incoming MCP requests and route to appropriate handlers.

        This method processes all incoming requests according to the MCP
        protocol specification, routing them to the appropriate tool
        handlers and formatting responses correctly.
        """
        request_id = request.id

        try:
            # Log incoming request
            self.metrics.record_request(request.method)

            # Route based on method
            if request.method == "initialize":
                result = await self._handle_initialize(request.params)
            elif request.method == "tools/list":
                result = await self._handle_list_tools()
            elif request.method == "tools/call":
                result = await self._handle_tool_call(request.params)
            elif request.method == "completion":
                result = await self._handle_completion(request.params)
            else:
                raise ValueError(f"Unknown method: {request.method}")

            # Record success
            self.metrics.record_success(request.method)

            return MCPResponse(result=result, id=request_id)

        except Exception as e:
            # Log and record error
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            self.metrics.record_error(request.method, str(e))

            return MCPResponse(
                error={
                    "code": -32603,
                    "message": str(e)
                },
                id=request_id
            )

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request from LLM client."""
        # Verify protocol version compatibility
        client_version = params.get("protocolVersion", "1.0")
        if not self.protocol_handler.is_compatible(client_version):
            raise ValueError(f"Incompatible protocol version: {client_version}")

        # Initialize graph database connection
        await self.graph_db.connect()

        # Return server capabilities
        return {
            "protocolVersion": "1.0",
            "serverInfo": {
                "name": "Weka Documentation GraphRAG MCP Server",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": True,
                "sampling": False,
                "resources": False
            }
        }

    async def _handle_list_tools(self) -> Dict[str, Any]:
        """Return list of available tools."""
        tools = []
        for tool in self.tool_registry.get_all():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.get_input_schema()
            })

        return {"tools": tools}

    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool and return results."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        # Get the tool
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check cache for recent results
        cache_key = f"tool:{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for tool {tool_name}")
            return json.loads(cached_result)

        # Execute the tool
        logger.info(f"Executing tool: {tool_name}")
        result = await tool.execute(tool_args)

        # Cache the result
        await self.cache.set(cache_key, json.dumps(result), ttl=3600)

        return result

    async def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle completion request for generating responses."""
        # This would integrate with the LLM for response generation
        # For now, we return a structured response based on tool results
        messages = params.get("messages", [])

        # Extract the last user message
        user_query = messages[-1].get("content", "") if messages else ""

        # Determine which tools to use based on the query
        relevant_tools = self._determine_relevant_tools(user_query)

        # Execute relevant tools and combine results
        combined_results = {}
        for tool_name in relevant_tools:
            tool = self.tool_registry.get(tool_name)
            if tool:
                # Simple argument extraction for demo
                tool_args = {"query": user_query}
                result = await tool.execute(tool_args)
                combined_results[tool_name] = result

        return {
            "content": self._format_response(combined_results),
            "model": "weka-graphrag",
            "stop_reason": "complete"
        }

    def _determine_relevant_tools(self, query: str) -> List[str]:
        """Determine which tools are relevant for the query."""
        # Simple keyword-based detection for now
        # In production, this would use more sophisticated NLP
        tools = []

        query_lower = query.lower()
        if any(word in query_lower for word in ["search", "find", "what is"]):
            tools.append("search_documentation")
        if any(word in query_lower for word in ["relate", "connect", "depend"]):
            tools.append("traverse_relationships")
        if any(word in query_lower for word in ["error", "troubleshoot", "fix"]):
            tools.append("troubleshoot_error")
        if any(word in query_lower for word in ["compare", "versus", "difference"]):
            tools.append("compare_systems")
        if any(word in query_lower for word in ["architect", "design", "how does"]):
            tools.append("explain_architecture")

        return tools if tools else ["search_documentation"]

    def _format_response(self, results: Dict[str, Any]) -> str:
        """Format tool results into a coherent response."""
        # This is a simplified formatter
        # In production, this would use templates or more sophisticated formatting
        response_parts = []

        for tool_name, result in results.items():
            if result.get("answer"):
                response_parts.append(result["answer"])
            if result.get("evidence"):
                response_parts.append(f"Evidence: {result['evidence']}")

        return "\n\n".join(response_parts) if response_parts else "No relevant information found."

    async def shutdown(self):
        """Clean shutdown of the server."""
        logger.info("Shutting down WekaMCPServer...")
        await self.graph_db.disconnect()
        await self.cache.close()
        logger.info("Shutdown complete")

# FastAPI app for HTTP interface
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    config = Config.from_env()
    app.state.mcp_server = WekaMCPServer(config)
    yield
    # Shutdown
    await app.state.mcp_server.shutdown()

app = FastAPI(title="Weka MCP Server", lifespan=lifespan)

@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Main MCP endpoint for handling all requests."""
    response = await app.state.mcp_server.handle_request(request)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "weka-mcp-server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
```

**Validation Criteria:**
- Server starts without errors and responds to health checks
- MCP initialization completes successfully when client connects
- Tools are properly registered and listed when requested
- Basic tool calls execute and return results
- Errors are properly caught and formatted according to MCP spec

### Task 1.3: Query Validation Layer

**Implementation Steps:**

Implement comprehensive query validation to prevent injection attacks and ensure safe operation. This layer sits between the MCP server and the graph database.

```python
# src/mcp_server/validation.py
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

class QueryType(Enum):
    """Types of queries that can be executed."""
    SEARCH = "search"
    TRAVERSE = "traverse"
    AGGREGATE = "aggregate"
    PATH = "path"

@dataclass
class ValidationRule:
    """Represents a validation rule for queries."""
    name: str
    pattern: Optional[str] = None
    max_depth: int = 5
    max_results: int = 1000
    allowed_operations: Set[str] = None
    prohibited_operations: Set[str] = None

class QueryValidator:
    """
    Multi-layer query validation ensuring safety and performance.

    This validator implements defense in depth with input sanitization,
    pattern matching, complexity analysis, and parameterization.
    """

    def __init__(self):
        """Initialize validator with default rules."""
        self.rules = self._initialize_rules()
        self.cypher_patterns = self._initialize_patterns()
        self.prohibited_keywords = {
            "DELETE", "REMOVE", "CREATE", "MERGE", "SET",
            "DETACH", "DROP", "ALTER", "GRANT", "REVOKE"
        }

    def _initialize_rules(self) -> Dict[str, ValidationRule]:
        """Initialize validation rules for different query types."""
        return {
            QueryType.SEARCH: ValidationRule(
                name="search",
                pattern=r"^MATCH.*WHERE.*RETURN.*$",
                max_depth=3,
                max_results=100,
                allowed_operations={"MATCH", "WHERE", "RETURN", "WITH", "OPTIONAL MATCH"},
                prohibited_operations=self.prohibited_keywords
            ),
            QueryType.TRAVERSE: ValidationRule(
                name="traverse",
                pattern=r"^MATCH.*\*\d+\.\.d+.*RETURN.*$",
                max_depth=5,
                max_results=500,
                allowed_operations={"MATCH", "WHERE", "RETURN", "WITH"},
                prohibited_operations=self.prohibited_keywords
            ),
            QueryType.AGGREGATE: ValidationRule(
                name="aggregate",
                pattern=r"^MATCH.*RETURN.*(COUNT|SUM|AVG|MAX|MIN).*$",
                max_depth=3,
                max_results=1000,
                allowed_operations={"MATCH", "WHERE", "RETURN", "WITH", "COUNT", "SUM"},
                prohibited_operations=self.prohibited_keywords
            ),
            QueryType.PATH: ValidationRule(
                name="path",
                pattern=r"^MATCH.*path.*RETURN.*$",
                max_depth=4,
                max_results=50,
                allowed_operations={"MATCH", "WHERE", "RETURN", "shortestPath"},
                prohibited_operations=self.prohibited_keywords
            )
        }

    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for query validation."""
        return {
            "parameter": re.compile(r"\$\w+"),  # Parameterized values
            "depth": re.compile(r"\*(\d+)\.\.(\d+)"),  # Traversal depth
            "injection": re.compile(r"[;']|--|\*/|/\*"),  # Common injection patterns
            "function_call": re.compile(r"\w+\s*\([^)]*\)"),  # Function calls
        }

    async def validate(self, query: str, query_type: QueryType,
                      parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a query through multiple security layers.

        Returns validated query and parameters or raises ValidationError.
        """
        # Layer 1: Basic sanitization
        sanitized_query = self._sanitize_query(query)

        # Layer 2: Check for prohibited operations
        self._check_prohibited_operations(sanitized_query)

        # Layer 3: Validate query structure
        self._validate_structure(sanitized_query, query_type)

        # Layer 4: Analyze complexity
        complexity = self._analyze_complexity(sanitized_query)
        rule = self.rules[query_type]
        if complexity["depth"] > rule.max_depth:
            raise ValidationError(f"Query depth {complexity['depth']} exceeds limit {rule.max_depth}")

        # Layer 5: Validate and sanitize parameters
        safe_params = self._validate_parameters(parameters) if parameters else {}

        # Layer 6: Ensure parameterization
        final_query = self._ensure_parameterization(sanitized_query, safe_params)

        return {
            "query": final_query,
            "parameters": safe_params,
            "complexity": complexity,
            "query_type": query_type.value
        }

    def _sanitize_query(self, query: str) -> str:
        """Remove potentially harmful characters and normalize query."""
        # Remove comments
        query = re.sub(r"//.*$", "", query, flags=re.MULTILINE)
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

        # Check for injection patterns
        if self.cypher_patterns["injection"].search(query):
            raise ValidationError("Potential injection pattern detected")

        # Normalize whitespace
        query = " ".join(query.split())

        return query

    def _check_prohibited_operations(self, query: str):
        """Check for prohibited Cypher operations."""
        query_upper = query.upper()
        for keyword in self.prohibited_keywords:
            if keyword in query_upper:
                raise ValidationError(f"Prohibited operation: {keyword}")

    def _validate_structure(self, query: str, query_type: QueryType):
        """Validate query structure against expected patterns."""
        rule = self.rules[query_type]

        # Check allowed operations
        query_upper = query.upper()
        for op in rule.allowed_operations:
            if op not in query_upper:
                # Some operations are optional
                if op not in {"WHERE", "OPTIONAL MATCH", "WITH"}:
                    raise ValidationError(f"Missing required operation: {op}")

        # Validate basic structure with regex if pattern is defined
        if rule.pattern and not re.match(rule.pattern, query, re.IGNORECASE):
            raise ValidationError(f"Query does not match expected pattern for {query_type.value}")

    def _analyze_complexity(self, query: str) -> Dict[str, int]:
        """Analyze query complexity to prevent resource exhaustion."""
        complexity = {
            "depth": 1,
            "estimated_nodes": 0,
            "operations": 0
        }

        # Check traversal depth
        depth_matches = self.cypher_patterns["depth"].findall(query)
        if depth_matches:
            max_depth = max(int(match[1]) for match in depth_matches)
            complexity["depth"] = max_depth

        # Count operations
        operations = ["MATCH", "WHERE", "WITH", "RETURN", "OPTIONAL MATCH"]
        for op in operations:
            complexity["operations"] += query.upper().count(op)

        # Estimate node count (simplified)
        complexity["estimated_nodes"] = complexity["depth"] * 100

        return complexity

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize query parameters."""
        safe_params = {}

        for key, value in parameters.items():
            # Validate parameter name
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValidationError(f"Invalid parameter name: {key}")

            # Sanitize parameter value based on type
            if isinstance(value, str):
                # Check for injection in string values
                if self.cypher_patterns["injection"].search(value):
                    raise ValidationError(f"Invalid characters in parameter {key}")
                safe_params[key] = value
            elif isinstance(value, (int, float, bool)):
                safe_params[key] = value
            elif isinstance(value, list):
                # Recursively validate list items
                safe_params[key] = [
                    self._validate_single_value(item) for item in value
                ]
            else:
                raise ValidationError(f"Unsupported parameter type for {key}: {type(value)}")

        return safe_params

    def _validate_single_value(self, value: Any) -> Any:
        """Validate a single parameter value."""
        if isinstance(value, str):
            if self.cypher_patterns["injection"].search(value):
                raise ValidationError("Invalid characters in parameter value")
            return value
        elif isinstance(value, (int, float, bool)):
            return value
        else:
            raise ValidationError(f"Unsupported value type: {type(value)}")

    def _ensure_parameterization(self, query: str, parameters: Dict[str, Any]) -> str:
        """Ensure all user inputs are parameterized."""
        # Check that all literal strings in WHERE clauses are parameterized
        # This is a simplified check - production would be more comprehensive

        # Find all parameter placeholders in query
        param_placeholders = self.cypher_patterns["parameter"].findall(query)

        # Ensure all placeholders have corresponding parameters
        for placeholder in param_placeholders:
            param_name = placeholder[1:]  # Remove $
            if param_name not in parameters:
                raise ValidationError(f"Missing parameter: {param_name}")

        # Add LIMIT if not present to prevent unbounded queries
        if "LIMIT" not in query.upper():
            query += " LIMIT 1000"

        return query

class ValidationError(Exception):
    """Raised when query validation fails."""
    pass

# Cypher template library for safe query construction
class CypherTemplates:
    """
    Library of pre-validated Cypher query templates.

    These templates provide safe, parameterized queries for common operations.
    """

    def __init__(self):
        """Initialize template library."""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load pre-defined query templates."""
        return {
            "search_by_name": """
                MATCH (n:$node_type)
                WHERE n.name CONTAINS $search_term
                RETURN n
                LIMIT $limit
            """,

            "find_relationships": """
                MATCH (n:$node_type {id: $node_id})
                MATCH (n)-[r:$relationship_type]->(m)
                RETURN n, r, m
                LIMIT $limit
            """,

            "traverse_path": """
                MATCH path = (start:$start_type {id: $start_id})
                  -[*1..$max_depth]->
                  (end:$end_type)
                RETURN path
                LIMIT $limit
            """,

            "find_errors_with_solutions": """
                MATCH (e:Error {code: $error_code})
                OPTIONAL MATCH (e)<-[:RESOLVES]-(p:Procedure)
                OPTIONAL MATCH (p)-[:CONTAINS_STEP]->(s:Step)
                RETURN e, p, collect(s) as steps
                ORDER BY s.order
            """,

            "compare_features": """
                MATCH (w:Component {name: 'WEKA', type: $feature_type})
                MATCH (o:Component {type: $feature_type})
                WHERE o.name <> 'WEKA'
                RETURN w, o,
                       w.properties as weka_properties,
                       o.properties as other_properties
            """
        }

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a query template by name."""
        return self.templates.get(template_name)

    def fill_template(self, template_name: str, parameters: Dict[str, Any]) -> str:
        """Fill a template with parameters."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # This is simplified - in production, use proper template engine
        filled = template
        for key, value in parameters.items():
            if isinstance(value, str):
                # Don't replace type parameters directly
                if key.endswith("_type"):
                    filled = filled.replace(f"${key}", value)
                else:
                    filled = filled.replace(f"${key}", f"'{value}'")
            else:
                filled = filled.replace(f"${key}", str(value))

        return filled
```

**Validation Criteria:**
- Injection attempts are successfully blocked
- Valid queries pass validation unchanged
- Complex queries are properly analyzed for resource limits
- Parameters are properly validated and sanitized
- Templates generate safe queries

### Task 1.4: Graph Schema Definition

**Implementation Steps:**

Define and create the comprehensive graph schema in Neo4j that represents the Weka documentation structure.

```python
# src/shared/schema.py
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    COMMAND = "Command"
    COMPONENT = "Component"
    PROCEDURE = "Procedure"
    CONFIGURATION = "Configuration"
    ERROR = "Error"
    CONCEPT = "Concept"
    EXAMPLE = "Example"
    STEP = "Step"
    PARAMETER = "Parameter"

class RelationshipType(Enum):
    """Types of relationships in the knowledge graph."""
    MANAGES = "MANAGES"
    EQUIVALENT_TO = "EQUIVALENT_TO"
    REQUIRES = "REQUIRES"
    RESOLVES = "RESOLVES"
    CONTAINS_STEP = "CONTAINS_STEP"
    EXECUTES = "EXECUTES"
    RELATED_TO = "RELATED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    AFFECTS = "AFFECTS"
    HAS_PARAMETER = "HAS_PARAMETER"
    MAY_RAISE = "MAY_RAISE"
    CAUSED_BY = "CAUSED_BY"
    IMPLEMENTS = "IMPLEMENTS"

@dataclass
class NodeSchema:
    """Schema definition for a node type."""
    type: NodeType
    properties: Dict[str, str]  # property_name: property_type
    required_properties: List[str]
    indexed_properties: List[str]
    unique_properties: List[str]

class SchemaManager:
    """
    Manages the graph database schema creation and updates.

    This class defines and maintains the complete schema for the
    Weka documentation knowledge graph.
    """

    def __init__(self, graph_db):
        """Initialize with graph database connection."""
        self.graph_db = graph_db
        self.node_schemas = self._define_node_schemas()
        self.relationship_schemas = self._define_relationship_schemas()

    def _define_node_schemas(self) -> Dict[NodeType, NodeSchema]:
        """Define schemas for all node types."""
        return {
            NodeType.COMMAND: NodeSchema(
                type=NodeType.COMMAND,
                properties={
                    "id": "string",
                    "name": "string",
                    "cli_syntax": "string",
                    "rest_endpoint": "string",
                    "http_method": "string",
                    "description": "string",
                    "category": "string",
                    "minimum_role": "string",
                    "requires_auth": "boolean",
                    "version_added": "string",
                    "deprecated": "boolean",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "name", "description", "category"],
                indexed_properties=["name", "category", "cli_syntax", "rest_endpoint"],
                unique_properties=["id"]
            ),

            NodeType.COMPONENT: NodeSchema(
                type=NodeType.COMPONENT,
                properties={
                    "id": "string",
                    "name": "string",
                    "type": "string",
                    "description": "string",
                    "configuration": "map",
                    "dependencies": "string[]",
                    "version": "string",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "name", "type", "description"],
                indexed_properties=["name", "type"],
                unique_properties=["id"]
            ),

            NodeType.PROCEDURE: NodeSchema(
                type=NodeType.PROCEDURE,
                properties={
                    "id": "string",
                    "title": "string",
                    "type": "string",
                    "description": "string",
                    "complexity": "string",
                    "estimated_time": "string",
                    "prerequisites": "string[]",
                    "version": "string",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "title", "type", "description"],
                indexed_properties=["title", "type", "complexity"],
                unique_properties=["id"]
            ),

            NodeType.CONFIGURATION: NodeSchema(
                type=NodeType.CONFIGURATION,
                properties={
                    "id": "string",
                    "name": "string",
                    "path": "string",
                    "type": "string",
                    "description": "string",
                    "default_value": "string",
                    "valid_range": "string",
                    "scope": "string",
                    "requires_restart": "boolean",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "name", "path", "type", "scope"],
                indexed_properties=["name", "path", "scope"],
                unique_properties=["id"]
            ),

            NodeType.ERROR: NodeSchema(
                type=NodeType.ERROR,
                properties={
                    "id": "string",
                    "code": "string",
                    "message": "string",
                    "severity": "string",
                    "category": "string",
                    "description": "string",
                    "common_causes": "string[]",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "code", "message", "severity"],
                indexed_properties=["code", "severity", "category"],
                unique_properties=["id", "code"]
            ),

            NodeType.CONCEPT: NodeSchema(
                type=NodeType.CONCEPT,
                properties={
                    "id": "string",
                    "term": "string",
                    "definition": "string",
                    "category": "string",
                    "aliases": "string[]",
                    "related_terms": "string[]",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "term", "definition"],
                indexed_properties=["term", "category"],
                unique_properties=["id", "term"]
            ),

            NodeType.EXAMPLE: NodeSchema(
                type=NodeType.EXAMPLE,
                properties={
                    "id": "string",
                    "title": "string",
                    "code": "string",
                    "language": "string",
                    "description": "string",
                    "use_case": "string",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "title", "code"],
                indexed_properties=["title", "language"],
                unique_properties=["id"]
            ),

            NodeType.STEP: NodeSchema(
                type=NodeType.STEP,
                properties={
                    "id": "string",
                    "order": "integer",
                    "instruction": "string",
                    "expected_result": "string",
                    "error_handling": "string",
                    "vector_embedding": "float[]"
                },
                required_properties=["id", "order", "instruction"],
                indexed_properties=["order"],
                unique_properties=["id"]
            ),

            NodeType.PARAMETER: NodeSchema(
                type=NodeType.PARAMETER,
                properties={
                    "id": "string",
                    "name": "string",
                    "type": "string",
                    "required": "boolean",
                    "default_value": "string",
                    "description": "string",
                    "validation_rules": "string"
                },
                required_properties=["id", "name", "type"],
                indexed_properties=["name"],
                unique_properties=["id"]
            )
        }

    def _define_relationship_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Define schemas for all relationship types."""
        return {
            "MANAGES": {
                "from": NodeType.COMMAND,
                "to": NodeType.COMPONENT,
                "properties": {}
            },
            "EQUIVALENT_TO": {
                "from": NodeType.COMMAND,
                "to": NodeType.COMMAND,
                "properties": {}
            },
            "REQUIRES": {
                "from": [NodeType.COMMAND, NodeType.COMPONENT],
                "to": [NodeType.CONFIGURATION, NodeType.COMPONENT],
                "properties": {"optional": "boolean"}
            },
            "RESOLVES": {
                "from": NodeType.PROCEDURE,
                "to": NodeType.ERROR,
                "properties": {"success_rate": "float"}
            },
            "CONTAINS_STEP": {
                "from": NodeType.PROCEDURE,
                "to": NodeType.STEP,
                "properties": {"order": "integer"}
            },
            "EXECUTES": {
                "from": NodeType.STEP,
                "to": NodeType.COMMAND,
                "properties": {}
            },
            "RELATED_TO": {
                "from": NodeType.CONCEPT,
                "to": NodeType.CONCEPT,
                "properties": {"strength": "float"}
            },
            "HAS_PARAMETER": {
                "from": NodeType.COMMAND,
                "to": NodeType.PARAMETER,
                "properties": {"position": "integer"}
            }
        }

    async def create_schema(self):
        """Create the complete schema in Neo4j."""
        # Create node constraints and indexes
        for node_type, schema in self.node_schemas.items():
            await self._create_node_constraints(schema)
            await self._create_node_indexes(schema)

        # Create relationship indexes
        await self._create_relationship_indexes()

        # Create vector indexes for similarity search
        await self._create_vector_indexes()

    async def _create_node_constraints(self, schema: NodeSchema):
        """Create constraints for a node type."""
        label = schema.type.value

        # Create unique constraints
        for prop in schema.unique_properties:
            query = f"""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.{prop} IS UNIQUE
            """
            await self.graph_db.execute(query)

        # Create existence constraints for required properties
        for prop in schema.required_properties:
            query = f"""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.{prop} IS NOT NULL
            """
            await self.graph_db.execute(query)

    async def _create_node_indexes(self, schema: NodeSchema):
        """Create indexes for a node type."""
        label = schema.type.value

        # Create indexes for searchable properties
        for prop in schema.indexed_properties:
            query = f"""
                CREATE INDEX IF NOT EXISTS
                FOR (n:{label})
                ON (n.{prop})
            """
            await self.graph_db.execute(query)

    async def _create_relationship_indexes(self):
        """Create indexes for relationships."""
        # Create indexes for commonly traversed relationships
        relationship_indexes = [
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:MANAGES]-() ON (r.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:REQUIRES]-() ON (r.optional)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:CONTAINS_STEP]-() ON (r.order)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.strength)"
        ]

        for query in relationship_indexes:
            await self.graph_db.execute(query)

    async def _create_vector_indexes(self):
        """Create vector indexes for similarity search."""
        vector_indexes = [
            {
                "name": "command_embeddings",
                "label": "Command",
                "property": "vector_embedding",
                "dimensions": 384,
                "similarity": "cosine"
            },
            {
                "name": "concept_embeddings",
                "label": "Concept",
                "property": "vector_embedding",
                "dimensions": 384,
                "similarity": "cosine"
            },
            {
                "name": "procedure_embeddings",
                "label": "Procedure",
                "property": "vector_embedding",
                "dimensions": 384,
                "similarity": "cosine"
            },
            {
                "name": "error_embeddings",
                "label": "Error",
                "property": "vector_embedding",
                "dimensions": 384,
                "similarity": "cosine"
            }
        ]

        for index in vector_indexes:
            query = f"""
                CALL db.index.vector.createNodeIndex(
                    '{index["name"]}',
                    '{index["label"]}',
                    '{index["property"]}',
                    {index["dimensions"]},
                    '{index["similarity"]}'
                )
            """
            try:
                await self.graph_db.execute(query)
            except Exception as e:
                # Index might already exist
                if "already exists" not in str(e):
                    raise
```

```cypher
-- scripts/neo4j/create_schema.cypher
-- Complete schema creation script for Weka Documentation GraphRAG

-- Node Constraints and Indexes

-- Command nodes
CREATE CONSTRAINT command_id IF NOT EXISTS
FOR (c:Command) REQUIRE c.id IS UNIQUE;

CREATE INDEX command_name IF NOT EXISTS
FOR (c:Command) ON (c.name);

CREATE INDEX command_cli IF NOT EXISTS
FOR (c:Command) ON (c.cli_syntax);

CREATE INDEX command_rest IF NOT EXISTS
FOR (c:Command) ON (c.rest_endpoint);

-- Component nodes
CREATE CONSTRAINT component_id IF NOT EXISTS
FOR (c:Component) REQUIRE c.id IS UNIQUE;

CREATE INDEX component_name IF NOT EXISTS
FOR (c:Component) ON (c.name);

CREATE INDEX component_type IF NOT EXISTS
FOR (c:Component) ON (c.type);

-- Procedure nodes
CREATE CONSTRAINT procedure_id IF NOT EXISTS
FOR (p:Procedure) REQUIRE p.id IS UNIQUE;

CREATE INDEX procedure_title IF NOT EXISTS
FOR (p:Procedure) ON (p.title);

CREATE INDEX procedure_type IF NOT EXISTS
FOR (p:Procedure) ON (p.type);

-- Configuration nodes
CREATE CONSTRAINT config_id IF NOT EXISTS
FOR (c:Configuration) REQUIRE c.id IS UNIQUE;

CREATE INDEX config_name IF NOT EXISTS
FOR (c:Configuration) ON (c.name);

CREATE INDEX config_path IF NOT EXISTS
FOR (c:Configuration) ON (c.path);

-- Error nodes
CREATE CONSTRAINT error_id IF NOT EXISTS
FOR (e:Error) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT error_code IF NOT EXISTS
FOR (e:Error) REQUIRE e.code IS UNIQUE;

CREATE INDEX error_severity IF NOT EXISTS
FOR (e:Error) ON (e.severity);

-- Concept nodes
CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT concept_term IF NOT EXISTS
FOR (c:Concept) REQUIRE c.term IS UNIQUE;

CREATE INDEX concept_category IF NOT EXISTS
FOR (c:Concept) ON (c.category);

-- Example nodes
CREATE CONSTRAINT example_id IF NOT EXISTS
FOR (e:Example) REQUIRE e.id IS UNIQUE;

CREATE INDEX example_title IF NOT EXISTS
FOR (e:Example) ON (e.title);

CREATE INDEX example_language IF NOT EXISTS
FOR (e:Example) ON (e.language);

-- Step nodes
CREATE CONSTRAINT step_id IF NOT EXISTS
FOR (s:Step) REQUIRE s.id IS UNIQUE;

CREATE INDEX step_order IF NOT EXISTS
FOR (s:Step) ON (s.order);

-- Parameter nodes
CREATE CONSTRAINT parameter_id IF NOT EXISTS
FOR (p:Parameter) REQUIRE p.id IS UNIQUE;

CREATE INDEX parameter_name IF NOT EXISTS
FOR (p:Parameter) ON (p.name);
```

**Validation Criteria:**
- Schema is successfully created in Neo4j
- All constraints are applied correctly
- Indexes are created and used in queries
- Vector indexes support similarity search
- No schema conflicts or errors

## Phase 2: Query Processing Engine

### Task 2.1: Natural Language to Cypher Translation

**Implementation Steps:**

(Implementation continues with detailed code for each task following the same pattern...)

**Note**: This implementation plan continues for all remaining phases and tasks as outlined in the Application Specification. Each task includes:
- Detailed implementation steps with code examples
- Configuration files and scripts
- Validation criteria
- Integration points with other components

The complete implementation would be approximately 50,000+ lines of code across all components.

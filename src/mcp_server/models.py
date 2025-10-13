# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §9 (Interfaces)
# MCP protocol models

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# MCP Protocol Models (based on Model Context Protocol specification)


class MCPInitializeRequest(BaseModel):
    protocol_version: str = "1.0"
    client_info: Dict[str, Any] = Field(default_factory=dict)


class MCPInitializeResponse(BaseModel):
    protocol_version: str = "1.0"
    server_info: Dict[str, Any]
    capabilities: Dict[str, Any]


class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPToolsListResponse(BaseModel):
    tools: List[MCPTool]


class MCPToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    content: List[Dict[str, Any]]
    is_error: bool = False


class MCPCompletionRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None


class MCPCompletionResponse(BaseModel):
    completion: str
    metadata: Optional[Dict[str, Any]] = None


# Health and Metrics Models


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class ReadinessResponse(BaseModel):
    ready: bool
    services: Dict[str, bool]
    timestamp: str


class MetricsResponse(BaseModel):
    requests_total: int
    requests_by_endpoint: Dict[str, int]
    errors_total: int
    latency_p50: float
    latency_p95: float
    latency_p99: float

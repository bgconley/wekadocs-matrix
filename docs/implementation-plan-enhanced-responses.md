# Implementation Plan: Enhanced Response Verbosity & Graph Traversal

**Status:** Proposed
**Version:** 1.0
**Date:** 2025-10-18
**Related Spec:** `/docs/feature-spec-enhanced-responses.md`
**Phase Alignment:** Phase 2 (extension), Phase 4 (advanced features)

---

## Overview

This plan implements two features:
1. **Verbosity levels** for `search_documentation` (snippet | full | graph)
2. **New tool:** `traverse_relationships` for graph exploration

Both features are **backwards compatible** and follow existing architectural patterns.

---

## Task Breakdown

### **Task E1: Add Verbosity Parameter to Core Components**

**Owner:** Backend
**Depends:** None (extends existing Phase 2 code)
**Effort:** 4 hours

#### Steps

1. **Update data models** (`src/query/response_builder.py`)
   ```python
   # Line ~15: Extend Evidence dataclass
   @dataclass
   class Evidence:
       section_id: Optional[str] = None
       node_id: Optional[str] = None
       node_label: Optional[str] = None
       snippet: Optional[str] = None

       # NEW FIELDS for full/graph modes
       title: Optional[str] = None
       full_text: Optional[str] = None
       metadata: Optional[Dict[str, Any]] = None
       related_entities: Optional[List[Dict[str, Any]]] = None
       related_sections: Optional[List[Dict[str, Any]]] = None

       path: Optional[List[str]] = None
       confidence: float = 1.0
   ```

2. **Add Verbosity enum** (`src/query/response_builder.py`)
   ```python
   from enum import Enum

   class Verbosity(str, Enum):
       SNIPPET = "snippet"
       FULL = "full"
       GRAPH = "graph"
   ```

3. **Update ResponseBuilder** (`src/query/response_builder.py`)
   ```python
   def build_response(
       self,
       query: str,
       intent: str,
       ranked_results: List[RankedResult],
       timing: Dict[str, float],
       filters: Optional[Dict[str, Any]] = None,
       verbosity: Verbosity = Verbosity.SNIPPET,  # NEW
   ) -> Response:
   ```

4. **Implement mode-specific extraction** (`src/query/response_builder.py`)
   ```python
   def _extract_evidence(
       self,
       ranked_results: List[RankedResult],
       verbosity: Verbosity
   ) -> List[Evidence]:
       evidence_list = []

       for ranked in ranked_results:
           result = ranked.result

           if verbosity == Verbosity.SNIPPET:
               # Current behavior (lines 130-151)
               evidence_list.append(Evidence(
                   section_id=...,
                   snippet=self._extract_snippet(metadata, max_length=200),
                   confidence=...
               ))

           elif verbosity == Verbosity.FULL:
               # Get full text from metadata
               evidence_list.append(Evidence(
                   section_id=...,
                   title=metadata.get("title"),
                   full_text=metadata.get("text"),  # Full section text
                   metadata={
                       "document_id": metadata.get("document_id"),
                       "level": metadata.get("level"),
                       "anchor": metadata.get("anchor"),
                       "tokens": metadata.get("tokens"),
                   },
                   confidence=...
               ))

           elif verbosity == Verbosity.GRAPH:
               # Full text + related entities
               related = self._get_related_entities(result.node_id)
               evidence_list.append(Evidence(
                   section_id=...,
                   title=...,
                   full_text=...,
                   metadata=...,
                   related_entities=related["entities"],
                   related_sections=related["sections"],
                   confidence=...
               ))

       return evidence_list
   ```

5. **Implement `_get_related_entities()`** (new method)
   ```python
   def _get_related_entities(
       self,
       node_id: str,
       max_entities: int = 20
   ) -> Dict[str, List]:
       """
       Fetch related entities and sections from Neo4j.
       Query: 1-hop neighbors via MENTIONS, CONTAINS_STEP, REQUIRES, AFFECTS.
       """
       query = """
       MATCH (n {id: $node_id})-[r:MENTIONS|CONTAINS_STEP|REQUIRES|AFFECTS]->(e)
       WHERE labels(e)[0] IN ['Command', 'Configuration', 'Step', 'Error', 'Concept']
       RETURN DISTINCT
           e.id AS entity_id,
           labels(e)[0] AS label,
           e.name AS name,
           type(r) AS relationship,
           r.confidence AS confidence
       ORDER BY r.confidence DESC
       LIMIT $max_entities
       """
       # Execute via Neo4j driver (inject via constructor)
       # Return: {"entities": [...], "sections": [...]}
   ```

**Deliverables:**
- `src/query/response_builder.py` (updated)
- Enum + extended Evidence dataclass
- Mode-specific extraction logic

**DoD:**
- [ ] Verbosity enum defined with 3 values
- [ ] Evidence dataclass has all optional fields for graph mode
- [ ] `_get_related_entities()` queries Neo4j for 1-hop neighbors
- [ ] No breaking changes (default verbosity=SNIPPET)

**Tests:**
- `tests/test_response_builder.py::test_verbosity_snippet_unchanged()`
- `tests/test_response_builder.py::test_verbosity_full_includes_text()`
- `tests/test_response_builder.py::test_verbosity_graph_includes_relationships()`

---

### **Task E2: Wire Verbosity to QueryService and MCP Server**

**Owner:** Backend
**Depends:** E1
**Effort:** 2 hours

#### Steps

1. **Update QueryService.search()** (`src/mcp_server/query_service.py`)
   ```python
   # Line ~86
   from src.query.response_builder import Verbosity

   def search(
       self,
       query: str,
       top_k: int = 20,
       filters: Optional[Dict[str, Any]] = None,
       expand_graph: bool = True,
       find_paths: bool = False,
       verbosity: str = "snippet",  # NEW: Accept string for MCP JSON
   ) -> Response:
       # Convert string to enum
       verb_enum = Verbosity(verbosity)

       # ... existing search logic ...

       # Build response with verbosity
       response = build_response(
           query=query,
           intent=intent,
           ranked_results=ranked_results,
           timing=timing,
           filters=filters,
           verbosity=verb_enum,  # NEW
       )
   ```

2. **Update MCP tool schema** (`src/mcp_server/main.py`)
   ```python
   # Line ~180 (in initialize_tools())
   {
       "name": "search_documentation",
       "description": "Search documentation with configurable detail level",
       "inputSchema": {
           "type": "object",
           "properties": {
               "query": {
                   "type": "string",
                   "description": "Natural language query"
               },
               "top_k": {
                   "type": "integer",
                   "description": "Number of results",
                   "default": 20
               },
               "verbosity": {
                   "type": "string",
                   "enum": ["snippet", "full", "graph"],
                   "description": "Response detail level: snippet (200 chars), full (complete text), graph (text + relationships)",
                   "default": "snippet"
               }
           },
           "required": ["query"]
       }
   }
   ```

3. **Update tool handler** (`src/mcp_server/main.py`)
   ```python
   # Line ~320 (in mcp_tools_call)
   if request.name == "search_documentation":
       query = request.arguments.get("query", "")
       top_k = request.arguments.get("top_k", 20)
       verbosity = request.arguments.get("verbosity", "snippet")  # NEW

       if not query:
           result = MCPToolCallResponse(...)
       else:
           try:
               query_service = get_query_service()
               response = query_service.search(
                   query=query,
                   top_k=top_k,
                   expand_graph=True,
                   find_paths=False,
                   verbosity=verbosity,  # NEW
               )
   ```

**Deliverables:**
- `src/mcp_server/query_service.py` (updated)
- `src/mcp_server/main.py` (updated tool schema and handler)

**DoD:**
- [ ] QueryService accepts `verbosity` parameter
- [ ] MCP tool schema includes verbosity with enum validation
- [ ] Invalid verbosity values rejected with 400 error
- [ ] Default verbosity="snippet" maintains backwards compatibility

**Tests:**
- `tests/test_mcp_integration.py::test_search_with_verbosity_snippet()`
- `tests/test_mcp_integration.py::test_search_with_verbosity_full()`
- `tests/test_mcp_integration.py::test_search_with_verbosity_graph()`
- `tests/test_mcp_integration.py::test_invalid_verbosity_rejected()`

---

### **Task E3: Implement Neo4j Relationship Queries**

**Owner:** Graph Eng
**Depends:** E1
**Effort:** 3 hours

#### Steps

1. **Add Neo4j driver injection** to ResponseBuilder
   ```python
   # src/query/response_builder.py constructor
   class ResponseBuilder:
       def __init__(self, neo4j_driver=None):
           self.neo4j_driver = neo4j_driver or get_connection_manager().get_neo4j_driver()
   ```

2. **Implement `_get_related_entities()`** (from E1)
   - Query: 1-hop MENTIONS, CONTAINS_STEP, REQUIRES, AFFECTS
   - Filter: Only entity labels (Command, Configuration, Step, etc.)
   - Limit: 20 entities per section
   - Return format:
     ```python
     {
       "entities": [
         {
           "entity_id": "cmd_123",
           "label": "Command",
           "name": "weka cluster create",
           "relationship": "MENTIONS",
           "confidence": 0.8
         }
       ],
       "sections": [
         {
           "section_id": "abc...",
           "title": "Create cluster",
           "relationship_path": ["6f37b...", "abc..."],
           "distance": 1
         }
       ]
     }
     ```

3. **Add caching** for related entities (optional optimization)
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def _get_related_entities_cached(self, node_id: str, max_entities: int):
       # Cache key includes node_id + max_entities
       return self._get_related_entities(node_id, max_entities)
   ```

4. **Enforce size limits**
   ```python
   MAX_FULL_TEXT_BYTES = 32768  # 32KB per section
   MAX_RELATED_ENTITIES = 20    # Per section
   MAX_RESPONSE_BYTES = 65536   # 64KB total

   def _extract_evidence(...):
       full_text = metadata.get("text", "")
       if len(full_text) > MAX_FULL_TEXT_BYTES:
           full_text = full_text[:MAX_FULL_TEXT_BYTES] + "...[truncated]"
   ```

**Deliverables:**
- `src/query/response_builder.py` (Neo4j queries added)
- Size limit enforcement

**DoD:**
- [ ] `_get_related_entities()` executes parameterized Cypher
- [ ] Returns max 20 entities per section
- [ ] Full text capped at 32KB
- [ ] Total response capped at 64KB
- [ ] Query timeout: 5 seconds

**Tests:**
- `tests/test_response_builder.py::test_get_related_entities_finds_commands()`
- `tests/test_response_builder.py::test_related_entities_limit_enforced()`
- `tests/test_response_builder.py::test_full_text_size_limit()`

---

### **Task E4: Implement traverse_relationships Tool**

**Owner:** Backend + Graph Eng
**Depends:** E3
**Effort:** 4 hours

#### Steps

1. **Create TraversalService** (`src/query/traversal.py` - NEW FILE)
   ```python
   """
   Graph Traversal Service
   Implements traverse_relationships MCP tool.
   """

   from dataclasses import dataclass
   from typing import Any, Dict, List, Optional

   @dataclass
   class TraversalNode:
       id: str
       label: str
       properties: Dict[str, Any]
       distance: int  # Hops from start

   @dataclass
   class TraversalRelationship:
       from_id: str
       to_id: str
       type: str
       properties: Dict[str, Any]

   @dataclass
   class TraversalResult:
       nodes: List[TraversalNode]
       relationships: List[TraversalRelationship]
       paths: List[Dict[str, Any]]

   class TraversalService:
       # Whitelist of allowed relationship types
       ALLOWED_REL_TYPES = [
           "MENTIONS", "CONTAINS_STEP", "HAS_PARAMETER",
           "REQUIRES", "AFFECTS", "RESOLVES", "RELATED_TO",
           "HAS_SECTION", "EXECUTES"
       ]

       MAX_DEPTH = 3
       MAX_NODES = 100

       def __init__(self, neo4j_driver):
           self.driver = neo4j_driver

       def traverse(
           self,
           start_ids: List[str],
           rel_types: List[str] = None,
           max_depth: int = 2,
           include_text: bool = True
       ) -> TraversalResult:
           # Validate inputs
           if max_depth > self.MAX_DEPTH:
               raise ValueError(f"max_depth cannot exceed {self.MAX_DEPTH}")

           rel_types = rel_types or self.ALLOWED_REL_TYPES
           for rel_type in rel_types:
               if rel_type not in self.ALLOWED_REL_TYPES:
                   raise ValueError(f"Invalid relationship type: {rel_type}")

           # Build Cypher query
           rel_pattern = "|".join(rel_types)
           query = f"""
           UNWIND $start_ids AS start_id
           MATCH (start {{id: start_id}})
           OPTIONAL MATCH path=(start)-[r:{rel_pattern}*1..{max_depth}]->(target)
           WITH DISTINCT target, min(length(path)) AS dist,
                collect(DISTINCT path) AS paths
           WHERE dist <= {max_depth}
           RETURN target.id AS id,
                  labels(target)[0] AS label,
                  properties(target) AS props,
                  dist,
                  paths
           ORDER BY dist ASC
           LIMIT {self.MAX_NODES}
           """

           # Execute and parse results
           # Return TraversalResult with nodes + relationships + paths
   ```

2. **Add MCP tool definition** (`src/mcp_server/main.py`)
   ```python
   {
       "name": "traverse_relationships",
       "description": "Traverse graph relationships from given nodes",
       "inputSchema": {
           "type": "object",
           "properties": {
               "start_ids": {
                   "type": "array",
                   "items": {"type": "string"},
                   "description": "Starting section/entity IDs"
               },
               "rel_types": {
                   "type": "array",
                   "items": {"type": "string"},
                   "description": "Relationship types to follow (default: all)",
                   "default": ["MENTIONS", "CONTAINS_STEP", "REQUIRES", "AFFECTS"]
               },
               "max_depth": {
                   "type": "integer",
                   "description": "Maximum traversal depth (1-3)",
                   "default": 2,
                   "minimum": 1,
                   "maximum": 3
               },
               "include_text": {
                   "type": "boolean",
                   "description": "Include full text of nodes",
                   "default": true
               }
           },
           "required": ["start_ids"]
       }
   }
   ```

3. **Add tool handler** (`src/mcp_server/main.py`)
   ```python
   elif request.name == "traverse_relationships":
       from src.query.traversal import TraversalService

       start_ids = request.arguments.get("start_ids", [])
       rel_types = request.arguments.get("rel_types")
       max_depth = request.arguments.get("max_depth", 2)
       include_text = request.arguments.get("include_text", True)

       if not start_ids:
           result = MCPToolCallResponse(
               content=[{"type": "text", "text": "Error: start_ids required"}],
               is_error=True
           )
       else:
           try:
               manager = get_connection_manager()
               traversal_svc = TraversalService(manager.get_neo4j_driver())

               traversal_result = traversal_svc.traverse(
                   start_ids=start_ids,
                   rel_types=rel_types,
                   max_depth=max_depth,
                   include_text=include_text
               )

               # Build response
               result = MCPToolCallResponse(
                   content=[
                       {"type": "text", "text": f"Found {len(traversal_result.nodes)} nodes, {len(traversal_result.relationships)} relationships"},
                       {"type": "json", "json": {
                           "nodes": [asdict(n) for n in traversal_result.nodes],
                           "relationships": [asdict(r) for r in traversal_result.relationships],
                           "paths": traversal_result.paths
                       }}
                   ],
                   is_error=False
               )
           except ValueError as e:
               result = MCPToolCallResponse(
                   content=[{"type": "text", "text": f"Validation error: {str(e)}"}],
                   is_error=True
               )
   ```

**Deliverables:**
- `src/query/traversal.py` (NEW FILE - ~200 lines)
- `src/mcp_server/main.py` (tool definition + handler)

**DoD:**
- [ ] TraversalService implements depth-limited traversal
- [ ] Relationship type whitelist enforced
- [ ] Max 100 nodes returned
- [ ] Paths include full node sequences
- [ ] include_text parameter controls text inclusion

**Tests:**
- `tests/test_traversal.py::test_traverse_single_hop()`
- `tests/test_traversal.py::test_traverse_multi_hop()`
- `tests/test_traversal.py::test_max_depth_enforced()`
- `tests/test_traversal.py::test_invalid_rel_type_rejected()`
- `tests/test_traversal.py::test_max_nodes_limit()`

---

### **Task E5: Add Observability**

**Owner:** SRE
**Depends:** E2, E4
**Effort:** 1 hour

#### Steps

1. **Add metrics** (`src/shared/observability/metrics.py`)
   ```python
   from prometheus_client import Counter, Histogram

   mcp_search_verbosity_total = Counter(
       "mcp_search_verbosity_total",
       "Search requests by verbosity level",
       ["verbosity"]
   )

   mcp_search_response_size_bytes = Histogram(
       "mcp_search_response_size_bytes",
       "Response size distribution",
       ["verbosity"],
       buckets=[1024, 5120, 10240, 20480, 40960, 65536]  # 1KB to 64KB
   )

   mcp_traverse_depth_total = Counter(
       "mcp_traverse_depth_total",
       "Traversal requests by depth",
       ["depth"]
   )

   mcp_traverse_nodes_found = Histogram(
       "mcp_traverse_nodes_found",
       "Number of nodes found in traversal",
       buckets=[1, 5, 10, 20, 50, 100]
   )
   ```

2. **Instrument QueryService** (`src/mcp_server/query_service.py`)
   ```python
   from src.shared.observability.metrics import (
       mcp_search_verbosity_total,
       mcp_search_response_size_bytes
   )

   def search(..., verbosity: str = "snippet"):
       # Increment counter
       mcp_search_verbosity_total.labels(verbosity=verbosity).inc()

       # ... execute search ...

       # Measure response size
       response_json = json.dumps(response.to_dict())
       response_size = len(response_json.encode('utf-8'))
       mcp_search_response_size_bytes.labels(verbosity=verbosity).observe(response_size)
   ```

3. **Instrument TraversalService** (`src/query/traversal.py`)
   ```python
   from src.shared.observability.metrics import (
       mcp_traverse_depth_total,
       mcp_traverse_nodes_found
   )

   def traverse(..., max_depth: int = 2):
       mcp_traverse_depth_total.labels(depth=str(max_depth)).inc()

       # ... execute traversal ...

       mcp_traverse_nodes_found.observe(len(result.nodes))
   ```

4. **Add structured logging** (all modified files)
   ```python
   logger.info(
       f"Search completed: verbosity={verbosity}, "
       f"response_size_kb={response_size/1024:.1f}, "
       f"evidence_count={len(evidence)}"
   )

   logger.info(
       f"Traversal completed: depth={max_depth}, "
       f"nodes_found={len(nodes)}, "
       f"relationships={len(relationships)}"
   )
   ```

**Deliverables:**
- Prometheus metrics definitions
- Instrumentation in QueryService and TraversalService

**DoD:**
- [ ] All 4 metrics exposed via /metrics endpoint
- [ ] Metrics incremented on each request
- [ ] Structured logs include verbosity and traversal depth
- [ ] No PII in logs

**Tests:**
- `tests/test_metrics.py::test_verbosity_metrics_incremented()`
- `tests/test_metrics.py::test_response_size_histogram_buckets()`

---

### **Task E6: Performance Validation**

**Owner:** QA + Backend
**Depends:** E2, E4, E5
**Effort:** 2 hours

#### Steps

1. **Create load test script** (`scripts/perf/test_verbosity_latency.py`)
   ```python
   import requests
   import time

   QUERIES = [
       "How do I configure a cluster?",
       "What are the system requirements?",
       "Troubleshoot performance issues",
   ]

   for verbosity in ["snippet", "full", "graph"]:
       latencies = []
       for query in QUERIES * 10:  # 30 requests
           start = time.time()
           resp = requests.post(
               "http://localhost:8000/mcp/tools/call",
               json={
                   "name": "search_documentation",
                   "arguments": {"query": query, "verbosity": verbosity}
               }
           )
           latencies.append((time.time() - start) * 1000)

       latencies.sort()
       p50 = latencies[len(latencies)//2]
       p95 = latencies[int(len(latencies)*0.95)]

       print(f"{verbosity}: P50={p50:.1f}ms, P95={p95:.1f}ms")
       assert p95 < 500, f"{verbosity} P95 exceeded 500ms"
   ```

2. **Run performance tests**
   ```bash
   # Warm up embedder
   curl -X POST http://localhost:8000/mcp/tools/call \
     -d '{"name":"search_documentation","arguments":{"query":"test"}}'

   # Run load tests
   python scripts/perf/test_verbosity_latency.py
   python scripts/perf/test_traversal_latency.py
   ```

3. **Collect baselines**
   - snippet mode: P95 = 70ms (current)
   - full mode: P95 < 100ms (target)
   - graph mode: P95 < 150ms (target)
   - traverse (depth=2): P95 < 200ms (target)

**Deliverables:**
- Performance test scripts
- Baseline metrics documented

**DoD:**
- [ ] All verbosity modes meet P95 targets
- [ ] traverse_relationships P95 < 200ms
- [ ] No memory leaks observed over 1000 requests
- [ ] Response size limits enforced

**Tests:**
- `tests/perf/test_verbosity_latency.py` (automated)
- `tests/perf/test_traversal_latency.py` (automated)

---

### **Task E7: Documentation & Examples**

**Owner:** Docs
**Depends:** E2, E4
**Effort:** 2 hours

#### Steps

1. **Update MCP tool documentation** (inline in main.py descriptions)
   - Add usage examples for each verbosity level
   - Document when to use traverse_relationships
   - Add token cost estimates

2. **Create usage guide** (`docs/guides/verbosity-usage.md`)
   ```markdown
   # Verbosity Levels Guide

   ## When to use snippet mode
   - Quick lookups
   - Browsing topics
   - Cost-sensitive applications
   - Token budget: ~500 tokens

   ## When to use full mode
   - Complete answer generation
   - Step-by-step procedures
   - Token budget: ~10,000 tokens

   ## When to use graph mode
   - Understanding dependencies
   - Impact analysis
   - Troubleshooting with context
   - Token budget: ~15,000 tokens

   ## Chaining search + traverse
   ```python
   # Step 1: Find relevant sections
   search_resp = search_documentation("cluster config", verbosity="snippet")
   section_ids = [ev["section_id"] for ev in search_resp["evidence"]]

   # Step 2: Deep dive on top result
   traverse_resp = traverse_relationships(
       start_ids=[section_ids[0]],
       rel_types=["MENTIONS", "CONTAINS_STEP"],
       max_depth=2
   )
   ```

3. **Add examples** to tool schemas (main.py)
   ```python
   "examples": [
       {
           "query": "How do I configure a cluster?",
           "verbosity": "full",
           "description": "Get complete configuration guide"
       },
       {
           "query": "What commands are available?",
           "verbosity": "graph",
           "description": "Find commands and their related configurations"
       }
   ]
   ```

**Deliverables:**
- Usage guide document
- Inline examples in tool schemas

**DoD:**
- [ ] Guide covers all 3 verbosity modes
- [ ] Includes token cost estimates
- [ ] Documents chaining workflow
- [ ] Examples in tool schemas

---

## Testing Matrix

| Test Type | File | Coverage |
|-----------|------|----------|
| Unit | `tests/test_response_builder.py` | Verbosity modes, evidence extraction |
| Unit | `tests/test_traversal.py` | Depth limits, whitelist validation |
| Integration | `tests/test_mcp_integration.py` | E2E via /mcp/tools/call |
| Performance | `tests/perf/test_verbosity_latency.py` | P95 targets |
| Security | `tests/security/test_traversal_validation.py` | Injection prevention |

---

## Rollout Plan

### Phase 1: Internal Testing (Day 1)
- Deploy to dev environment
- Run full test suite
- Validate P95 latency metrics
- Test chaining workflow manually

### Phase 2: Staging Deployment (Day 2)
- Deploy to staging with feature flag
- Enable for internal users only
- Monitor metrics: verbosity usage, response sizes, latencies
- Collect feedback

### Phase 3: Production Rollout (Day 3)
- Gradual rollout: 10% → 50% → 100%
- Monitor error rates and P95 latency
- Alert if P95 > 400ms (safety margin)
- Full documentation published

---

## Metrics & Success Criteria

### Adoption Metrics (30 days)
- [ ] `verbosity=full` used in > 20% of queries
- [ ] `verbosity=graph` used in > 10% of queries
- [ ] `traverse_relationships` called in > 5% of sessions

### Quality Metrics
- [ ] No P95 latency regressions for snippet mode
- [ ] Error rate < 0.1% across all modes
- [ ] Response size limits never exceeded

### Performance Targets
- [x] snippet: P95 = 70ms (current baseline)
- [ ] full: P95 < 100ms
- [ ] graph: P95 < 150ms
- [ ] traverse (depth=2): P95 < 200ms

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Graph mode too slow | Cache related entities, limit to 20 per section |
| Large responses hit token limits | Enforce 64KB max, document limits |
| Traversal abused for scraping | Rate limits + audit logging |
| Neo4j load spike | Connection pooling + query timeouts |

---

## Definition of Done (Overall)

- [ ] All 7 tasks completed (E1-E7)
- [ ] Test coverage > 90% for new code
- [ ] P95 latency targets met for all modes
- [ ] Documentation published
- [ ] Deployed to production
- [ ] Metrics dashboard created
- [ ] No critical bugs in first 7 days

---

**Approval Required:** Yes
**Estimated Total Effort:** 18 hours (2.5 days)
**Risk Level:** Low (backwards compatible, incremental rollout)

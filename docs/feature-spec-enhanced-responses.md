# Feature Specification: Enhanced Response Verbosity & Graph Traversal

**Status:** Proposed
**Version:** 1.0
**Date:** 2025-10-18
**Related Phases:** Phase 2 (Query Processing), Phase 4 (Advanced Query Features)

---

## 1. Problem Statement

### Current Limitation

The MCP server's `search_documentation` tool currently returns:
- **200-character snippets** from matched sections
- **Top 5 results only** (even though hybrid search finds 19+ candidates)
- **No graph context** (related entities are found but not exposed)
- **No relationship paths** between results

### Impact on LLM Effectiveness

When an LLM queries "How do I configure a cluster?", it receives:
```
1. "Add clients that are always part of the clust..." (truncated)
2. "4. Create a cluster..." (no context)
3. "Workflow..." (generic title, no content)
```

**The LLM cannot:**
- See full section text to formulate complete answers
- Understand relationships between the 5 sections
- Access related entities (Commands, Configurations, Steps) found via graph expansion
- Determine if sections are part of the same procedure or separate topics

### Real-World Example

Query: "How do I configure a cluster?"

**Current response:** 5 section titles, 200-char snippets
**Ideal response:**
- Full text of 5 matched sections (2-3KB each)
- Related entities: `weka cluster create` command, `cluster.yaml` config
- Relationships: Section 2 CONTAINS_STEP → Step "Initialize cluster nodes"
- Path: Section "Plan a cluster" → Section "Create a cluster" → Section "Add clients"

---

## 2. Proposed Solution

### Feature 1: Verbosity Levels (Option 3)

Add `verbosity` parameter to `search_documentation` tool with three modes:

#### **Mode 1: `snippet` (default - current behavior)**
```json
{
  "evidence": [{
    "section_id": "6f37b...",
    "snippet": "Add clients that are always part of the clust...",  // 200 chars
    "confidence": 0.57
  }]
}
```

#### **Mode 2: `full`**
```json
{
  "evidence": [{
    "section_id": "6f37b...",
    "title": "Add clients that are always part of the cluster",
    "full_text": "To configure persistent clients in your Weka cluster, you must...",  // Complete section
    "metadata": {
      "document_id": "abc123...",
      "level": 2,
      "anchor": "#persistent-clients",
      "tokens": 450
    },
    "confidence": 0.57
  }]
}
```

#### **Mode 3: `graph`**
```json
{
  "evidence": [{
    "section_id": "6f37b...",
    "title": "Add clients that are always part of the cluster",
    "full_text": "...",
    "related_entities": [
      {
        "entity_id": "cmd_weka_cluster_add",
        "label": "Command",
        "name": "weka cluster add-client",
        "relationship": "MENTIONS",
        "confidence": 0.8
      },
      {
        "entity_id": "config_client_yaml",
        "label": "Configuration",
        "name": "client.yaml",
        "relationship": "REQUIRES",
        "confidence": 0.7
      }
    ],
    "related_sections": [
      {
        "section_id": "a9e5b...",
        "title": "4. Create a cluster",
        "relationship_path": ["6f37b...", "a9e5b..."],
        "distance": 1
      }
    ],
    "confidence": 0.57
  }]
}
```

### Feature 2: `traverse_relationships` Tool (Option 2)

New MCP tool for deep exploration:

```json
{
  "name": "traverse_relationships",
  "description": "Traverse graph relationships from given section/entity IDs",
  "inputSchema": {
    "type": "object",
    "properties": {
      "start_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Section or entity IDs to start from"
      },
      "rel_types": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Relationship types to follow (MENTIONS, CONTAINS_STEP, REQUIRES, etc.)",
        "default": ["MENTIONS", "CONTAINS_STEP", "REQUIRES", "AFFECTS"]
      },
      "max_depth": {
        "type": "integer",
        "description": "Maximum traversal depth",
        "default": 2,
        "minimum": 1,
        "maximum": 3
      },
      "include_text": {
        "type": "boolean",
        "description": "Include full text of traversed nodes",
        "default": true
      }
    },
    "required": ["start_ids"]
  }
}
```

**Response format:**
```json
{
  "nodes": [
    {
      "id": "6f37b...",
      "label": "Section",
      "title": "Add clients that are always part of the cluster",
      "full_text": "...",
      "distance": 0
    },
    {
      "id": "cmd_123",
      "label": "Command",
      "name": "weka cluster add-client",
      "description": "...",
      "distance": 1
    }
  ],
  "relationships": [
    {
      "from": "6f37b...",
      "to": "cmd_123",
      "type": "MENTIONS",
      "properties": {
        "confidence": 0.8,
        "source_section_id": "6f37b..."
      }
    }
  ],
  "paths": [
    {
      "nodes": ["6f37b...", "cmd_123", "config_456"],
      "length": 2,
      "relationships": ["MENTIONS", "REQUIRES"]
    }
  ]
}
```

---

## 3. Use Cases

### Use Case 1: Quick Summary (verbosity=snippet)
**LLM Goal:** Get quick overview of relevant topics
**Query:** "What monitoring tools are available?"
**Response:** 5 section titles + 200-char snippets
**Token cost:** ~500 tokens

### Use Case 2: Complete Answer (verbosity=full)
**LLM Goal:** Formulate comprehensive answer from full context
**Query:** "How do I configure a cluster?"
**Response:** 5 sections with complete text (2-3KB each)
**Token cost:** ~10,000 tokens
**LLM can:** Write step-by-step guide with all details

### Use Case 3: Deep Exploration (verbosity=graph)
**LLM Goal:** Understand relationships and dependencies
**Query:** "What configurations affect cluster performance?"
**Response:** Sections + related Configuration entities + AFFECTS relationships
**Token cost:** ~15,000 tokens
**LLM can:** Build dependency graph, explain cascading impacts

### Use Case 4: Targeted Traversal (traverse_relationships)
**LLM Workflow:**
1. `search_documentation("troubleshoot errors", verbosity="snippet")` → Get error section IDs
2. `traverse_relationships(start_ids=[...], rel_types=["RESOLVES"])` → Find solution procedures
3. Formulate troubleshooting guide from connected sections

---

## 4. Specification Details

### 4.1 Verbosity Parameter Schema

```typescript
type Verbosity = "snippet" | "full" | "graph";

interface SearchDocumentationArgs {
  query: string;
  top_k?: number;        // default: 20
  verbosity?: Verbosity; // default: "snippet"
  filters?: Record<string, any>;
}
```

### 4.2 Response Size Estimates

| Verbosity | Avg Response Size | Max Response Size | Use Case |
|-----------|------------------|-------------------|----------|
| `snippet` | 2 KB | 5 KB | Quick lookup |
| `full` | 12 KB | 30 KB | Complete answer |
| `graph` | 25 KB | 60 KB | Deep understanding |

### 4.3 Performance Targets

| Verbosity | P95 Latency | Additional Cost |
|-----------|-------------|-----------------|
| `snippet` | 70ms | Baseline |
| `full` | 100ms | +30ms (DB reads) |
| `graph` | 150ms | +80ms (graph queries) |

All modes must remain **< 500ms P95** per spec requirement.

### 4.4 Backwards Compatibility

- Default `verbosity="snippet"` maintains current behavior
- Existing MCP clients continue to work without changes
- New parameter is optional in schema

---

## 5. Security & Safety

### 5.1 Response Size Limits

```yaml
limits:
  max_response_tokens: 65536    # ~16K tokens
  max_full_text_bytes: 32768    # 32KB per section
  max_related_entities: 20      # Per section in graph mode
  max_graph_nodes: 100          # Total nodes in traverse_relationships
```

### 5.2 Query Validation

- Verbosity parameter: Enum validation (reject invalid values)
- traverse_relationships depth: Hard cap at 3 hops (prevent graph explosion)
- Relationship types: Whitelist only (no arbitrary Cypher injection)

### 5.3 Rate Limiting Considerations

`graph` mode and `traverse_relationships` consume more resources:
- Apply same rate limits (60 req/min)
- Monitor P95 latency per mode
- Alert if `graph` mode P95 > 200ms

---

## 6. Observability

### 6.1 Metrics to Add

```python
mcp_search_verbosity_total = Counter(
    "mcp_search_verbosity_total",
    "Search requests by verbosity level",
    ["verbosity"]
)

mcp_search_response_size_bytes = Histogram(
    "mcp_search_response_size_bytes",
    "Response size in bytes",
    ["verbosity"]
)

mcp_traverse_depth_histogram = Histogram(
    "mcp_traverse_depth",
    "Traversal depth distribution",
    buckets=[1, 2, 3]
)
```

### 6.2 Logging

```python
logger.info(
    f"Search completed: verbosity={verbosity}, "
    f"response_size_kb={response_size/1024:.1f}, "
    f"evidence_count={len(evidence)}, "
    f"related_entities={related_count}"
)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

- `test_verbosity_snippet_default()`: Verify default behavior unchanged
- `test_verbosity_full_includes_complete_text()`: Assert full text present
- `test_verbosity_graph_includes_relationships()`: Validate graph structure
- `test_traverse_relationships_depth_limit()`: Enforce max depth
- `test_response_size_limits()`: Verify size caps enforced

### 7.2 Integration Tests

- E2E: Query → `verbosity=full` → Validate response schema
- E2E: `search` + `traverse_relationships` chained workflow
- Performance: P95 latency for each verbosity mode under load

### 7.3 Acceptance Criteria

**Feature 1 (Verbosity):**
- [ ] `verbosity=snippet` returns current 200-char snippets
- [ ] `verbosity=full` returns complete section text (< 32KB each)
- [ ] `verbosity=graph` includes related entities and relationships
- [ ] P95 latency: snippet=70ms, full=100ms, graph=150ms
- [ ] Response size limits enforced (max 65KB)

**Feature 2 (Traverse):**
- [ ] `traverse_relationships` follows specified rel_types
- [ ] Max depth enforced (hard cap at 3)
- [ ] Returns nodes, relationships, and paths
- [ ] P95 latency < 200ms for depth=2
- [ ] Whitelist validation for relationship types

---

## 8. Success Metrics

### 8.1 Adoption Metrics

- `verbosity=full` usage > 20% of queries within 30 days
- `traverse_relationships` called > 5% of sessions
- `graph` mode used for troubleshooting queries > 50%

### 8.2 Quality Metrics

- LLM answer completeness score improves by 40%
- User feedback: "Answer included all necessary details" > 80%
- Follow-up query rate decreases by 25%

### 8.3 Performance Metrics

- P95 latency remains < 500ms across all modes
- Error rate < 0.1%
- Cache hit rate (for repeated traversals) > 60%

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large responses hit token limits | High | Enforce 65KB max, document in tool description |
| Graph mode too slow (>500ms) | High | Add caching for common traversals, limit depth |
| LLMs overuse `graph` mode | Medium | Document when to use each mode, add cost hints |
| traverse_relationships abused for graph scraping | Medium | Rate limits, max 100 nodes per call, audit logging |

---

## 10. Future Enhancements (Out of Scope)

- **Streaming responses:** Return sections incrementally for large `full` mode responses
- **Relevance filtering:** Auto-exclude low-confidence related entities in `graph` mode
- **Smart verbosity:** LLM-agent decides verbosity based on query complexity
- **Relationship scoring:** Rank related entities by relevance to original query
- **Compressed graph format:** Minimize token cost while preserving structure

---

## 11. Alignment with Existing Spec

### Spec.md §5 (Responses & Explainability)
✅ "Dual output: Markdown for humans, JSON with evidence"
✅ "Why these results? surface ranking features"
⚠️ **Extension:** Add verbosity to control detail level

### Spec.md §9 (Interfaces)
✅ `search_documentation` tool defined
✅ `traverse_relationships` tool mentioned
✅ **Implementation:** Both tools now fully specified

### Pseudocode Phase 2.4 (Response Generation)
✅ Lines 280-300: `build_response()` extracts evidence
✅ **Extension:** Add verbosity parameter to control evidence detail

---

## 12. Definition of Done

**Feature 1 (Verbosity):**
- [ ] Code: Add `verbosity` parameter to QueryService.search()
- [ ] Code: Modify ResponseBuilder to handle 3 verbosity modes
- [ ] Tests: Unit tests for each mode (pass rate > 95%)
- [ ] Tests: E2E tests via /mcp/tools/call
- [ ] Docs: Update MCP tool schema with verbosity parameter
- [ ] Metrics: Prometheus counters added and tested
- [ ] Performance: P95 < 150ms for graph mode validated

**Feature 2 (Traverse):**
- [ ] Code: Implement TraversalService with depth limiting
- [ ] Code: Add /mcp/tools/call handler for traverse_relationships
- [ ] Tests: Unit tests for depth limits and whitelist validation
- [ ] Tests: Integration test with chained search → traverse workflow
- [ ] Docs: MCP tool schema documented
- [ ] Security: Relationship type whitelist enforced
- [ ] Performance: P95 < 200ms for depth=2 validated

---

**Approval Required Before Implementation:** Yes
**Estimated Effort:** 2-3 days (1 day verbosity, 1 day traverse, 0.5 day testing)
**Risk Level:** Low (backwards compatible, follows existing patterns)

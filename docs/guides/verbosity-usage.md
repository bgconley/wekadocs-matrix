# Verbosity Levels Guide

**Enhanced Response Features (E1-E7)**

This guide explains when and how to use different verbosity levels and the `traverse_relationships` tool for multi-turn graph exploration.

---

## Overview

The `search_documentation` tool now supports three verbosity modes:

- **`snippet`** (default): Fast, concise 200-character snippets
- **`full`**: Complete section text with metadata
- **`graph`**: Full text + related entities and relationships

The `traverse_relationships` tool enables multi-turn exploration of the knowledge graph.

---

## Verbosity Modes

### `snippet` Mode (Default)

**When to use:**
- Quick lookups and browsing
- Getting an overview of available topics
- Cost-sensitive applications where token usage matters
- Initial exploration before diving deeper

**Characteristics:**
- Returns 200-character text snippets
- Fastest response time (P95 ~70ms)
- Lowest token cost (~500 tokens per response)
- Best for scanning multiple results quickly

**Example:**
```json
{
  "name": "search_documentation",
  "arguments": {
    "query": "How do I configure a cluster?",
    "verbosity": "snippet"
  }
}
```

**Response structure:**
```json
{
  "answer_json": {
    "evidence": [
      {
        "section_id": "abc123...",
        "snippet": "To configure a cluster, use the weka cluster create command with the following parameters...",
        "confidence": 0.89
      }
    ]
  }
}
```

---

### `full` Mode

**When to use:**
- Complete answer generation
- Step-by-step procedures and tutorials
- Detailed explanations
- When you need all the context without relationships

**Characteristics:**
- Returns complete section text (up to 32KB per section)
- Response time: P95 <100ms
- Token cost: ~10,000 tokens per response
- Includes metadata (document_id, level, anchor, tokens)

**Example:**
```json
{
  "name": "search_documentation",
  "arguments": {
    "query": "Explain WekaFS architecture",
    "verbosity": "full"
  }
}
```

**Response structure:**
```json
{
  "answer_json": {
    "evidence": [
      {
        "section_id": "abc123...",
        "title": "WekaFS Architecture Overview",
        "full_text": "[Complete section text...]",
        "metadata": {
          "document_id": "wekadocs",
          "level": 2,
          "anchor": "architecture-overview",
          "tokens": 1240
        },
        "confidence": 0.92
      }
    ]
  }
}
```

---

### `graph` Mode

**When to use:**
- Understanding dependencies and relationships
- Impact analysis ("what uses this?", "what does this affect?")
- Troubleshooting with full context
- Exploring interconnected concepts

**Characteristics:**
- Returns full text + related entities and sections
- Response time: P95 <150ms
- Token cost: ~15,000 tokens per response
- Includes up to 20 related entities per section

**Example:**
```json
{
  "name": "search_documentation",
  "arguments": {
    "query": "cluster configuration requirements",
    "verbosity": "graph"
  }
}
```

**Response structure:**
```json
{
  "answer_json": {
    "evidence": [
      {
        "section_id": "abc123...",
        "title": "Cluster Configuration",
        "full_text": "[Complete section text...]",
        "related_entities": [
          {
            "entity_id": "cmd_cluster_create",
            "label": "Command",
            "name": "weka cluster create",
            "relationship": "MENTIONS",
            "confidence": 0.9
          },
          {
            "entity_id": "cfg_min_nodes",
            "label": "Configuration",
            "name": "minimum_nodes",
            "relationship": "REQUIRES",
            "confidence": 0.85
          }
        ],
        "related_sections": []
      }
    ]
  }
}
```

---

## Multi-Turn Exploration: Chaining search + traverse

For complex queries requiring deep dives, use this two-step workflow:

### Step 1: Find Relevant Sections (Low Cost)

Use `snippet` mode to identify the most relevant sections:

```json
{
  "name": "search_documentation",
  "arguments": {
    "query": "cluster configuration",
    "verbosity": "snippet",
    "top_k": 5
  }
}
```

Extract `section_id` values from the top results.

### Step 2: Deep Dive with Traversal (High Detail)

Use `traverse_relationships` to explore the graph from the most relevant section:

```json
{
  "name": "traverse_relationships",
  "arguments": {
    "start_ids": ["abc123..."],  // Top section from Step 1
    "rel_types": ["MENTIONS", "CONTAINS_STEP", "REQUIRES", "AFFECTS"],
    "max_depth": 2,
    "include_text": true
  }
}
```

**Benefits of chaining:**
- Cost-effective: Pay for detailed data only when needed
- Iterative refinement: Adjust traversal depth based on results
- Complete context: Get full graph neighborhood, not just search matches

---

## `traverse_relationships` Tool

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_ids` | array | **required** | Starting section/entity IDs |
| `rel_types` | array | `["MENTIONS", "CONTAINS_STEP", "REQUIRES", "AFFECTS"]` | Relationship types to follow |
| `max_depth` | int | `2` | Maximum traversal depth (1-3 hops) |
| `include_text` | bool | `true` | Include full text in node properties |

### Allowed Relationship Types

- `MENTIONS`: Section mentions an entity (command, config, etc.)
- `CONTAINS_STEP`: Procedure contains a step
- `HAS_PARAMETER`: Command has parameters
- `REQUIRES`: Dependency relationships
- `AFFECTS`: Impact relationships
- `RESOLVES`: Error resolutions
- `RELATED_TO`: General relationships
- `HAS_SECTION`: Document structure
- `EXECUTES`: Execution relationships

### Example: Multi-Hop Dependency Analysis

```json
{
  "name": "traverse_relationships",
  "arguments": {
    "start_ids": ["section_cluster_config"],
    "rel_types": ["REQUIRES", "AFFECTS"],
    "max_depth": 3,
    "include_text": false  // Faster, just structure
  }
}
```

**Response:**
```json
{
  "nodes": [
    {"id": "cfg_min_nodes", "label": "Configuration", "distance": 1},
    {"id": "cfg_network_setup", "label": "Configuration", "distance": 1},
    {"id": "proc_network_init", "label": "Procedure", "distance": 2}
  ],
  "relationships": [
    {"from_id": "section_cluster_config", "to_id": "cfg_min_nodes", "type": "REQUIRES"},
    {"from_id": "cfg_min_nodes", "to_id": "proc_network_init", "type": "AFFECTS"}
  ],
  "paths": [
    {"nodes": ["section_cluster_config", "cfg_min_nodes", "proc_network_init"], "length": 2}
  ]
}
```

---

## Token Cost Comparison

| Verbosity | Avg Tokens | Best For |
|-----------|------------|----------|
| `snippet` | ~500 | Quick browsing, topic discovery |
| `full` | ~10,000 | Complete answers, procedures |
| `graph` | ~15,000 | Impact analysis, troubleshooting |

**Cost optimization tip:** Always start with `snippet` to identify the right sections, then use `full` or `graph` only on the top 1-2 results.

---

## Performance Characteristics

| Mode | P95 Latency | Use Case |
|------|-------------|----------|
| `snippet` | ~70ms | Real-time search |
| `full` | <100ms | Detailed answers |
| `graph` | <150ms | Relationship exploration |
| `traverse` (depth=2) | <200ms | Multi-hop traversal |

All measurements with warmed embedder cache.

---

## Best Practices

### 1. Start Small, Expand as Needed
```
snippet → identify sections → full on top result → traverse if relationships matter
```

### 2. Use Appropriate Depth for Traversal
- **depth=1**: Direct neighbors only (fastest)
- **depth=2**: Standard exploration (recommended)
- **depth=3**: Deep analysis (use sparingly)

### 3. Filter Relationship Types
Only include relationship types relevant to your query:
```json
{
  "rel_types": ["REQUIRES", "AFFECTS"]  // For dependency analysis
}
```

### 4. Toggle `include_text` Based on Need
- `include_text: true` → Full context for reading
- `include_text: false` → Structure only (2-3x faster)

---

## Common Patterns

### Pattern 1: Progressive Detail
```
1. search(query, verbosity="snippet") → find sections
2. search(query, verbosity="full") → deep dive on top result
```

### Pattern 2: Relationship Discovery
```
1. search(query, verbosity="snippet") → find entry point
2. traverse(start_ids=[top_result], depth=2) → explore neighborhood
```

### Pattern 3: Impact Analysis
```
1. search(target_entity, verbosity="snippet")
2. traverse(start_ids=[entity], rel_types=["AFFECTS", "REQUIRES"], depth=3)
```

---

## Error Handling

### Invalid Verbosity
```json
{
  "error": "Invalid verbosity 'detailed'. Must be one of: snippet, full, graph"
}
```

### Invalid Relationship Type
```json
{
  "error": "Invalid relationship type: CUSTOM_REL. Allowed types: MENTIONS, CONTAINS_STEP, ..."
}
```

### Max Depth Exceeded
```json
{
  "error": "max_depth cannot exceed 3"
}
```

---

## Summary

- Use **`snippet`** for fast exploration and topic discovery
- Use **`full`** when you need complete section text
- Use **`graph`** for understanding relationships and dependencies
- Use **`traverse_relationships`** for multi-turn deep dives into the knowledge graph
- Chain operations for optimal cost/performance balance

For implementation details, see:
- `/docs/feature-spec-enhanced-responses.md`
- `/docs/implementation-plan-enhanced-responses.md`

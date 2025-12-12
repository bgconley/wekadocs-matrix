# markdown-it-py Integration Plan for WekaDocs RAG Pipeline

**Created:** 2024-12-10
**Status:** Draft
**Branch:** `dense-graph-enhance`

---

## Executive Summary

This plan details the replacement of the current `markdown` + BeautifulSoup parser with `markdown-it-py` to unlock AST-based parsing, source position mapping, and improved integration with our semantic chunking (Chonkie), NER enrichment (GLiNER), and hybrid retrieval (BGE-M3 + Neo4j) pipeline.

### Why markdown-it-py?

| Capability | Current (`markdown` lib) | `markdown-it-py` |
|------------|-------------------------|------------------|
| Parsing approach | Markdown → HTML → BeautifulSoup | Direct token stream + AST |
| Source positions | Lost in HTML conversion | `Token.map` preserves line numbers |
| Structure access | Post-hoc HTML element scraping | Native `SyntaxTreeNode` traversal |
| Extension system | Limited Python extensions | Rich plugin ecosystem |
| CommonMark compliance | Partial | 100% spec compliant |
| Performance | Double conversion overhead | Single-pass parsing |

---

## Research Validation Summary

### Findings from External Sources

1. **markdown-it-py Official Docs** (readthedocs.io):
   - `SyntaxTreeNode` provides true AST with `.children`, `.next_sibling`, `.pretty()` traversal
   - Token stream is flat with `nesting` attributes (-1/0/1) for open/close
   - `Token.map = [start_line, end_line]` enables source correlation
   - Presets: `zero`, `commonmark`, `js-default`, `gfm-like`

2. **RAG Best Practices** (Weaviate, Databricks, Neo4j blogs):
   - Header-based section splitting preserves semantic boundaries
   - Context enrichment (parent headers) improves retrieval by 15-25%
   - Markdown structure maps directly to knowledge graph hierarchy
   - BGE-M3 benefits from clean structural metadata for multi-vector fusion

3. **Chonkie Integration** (docs.chonkie.ai, GitHub):
   - Works with any text input—agnostic to parser
   - `SemanticChunker` expects text strings, returns `Chunk` objects with `.text`, `.token_count`
   - Our existing `SemanticChunkerAssembler` already abstracts this

4. **md2chunks Pattern** (verloop/md2chunks):
   - Context-enriched chunking: prepends parent headers to each chunk
   - TextNode pattern from LlamaIndex for provenance tracking
   - Token-aware splitting with URL/decimal handling

### Validation Against Training Knowledge

**Confirmed patterns from my training:**
- markdown-it-py is the Python port of markdown-it.js (used by VS Code, Jupyter, etc.)
- AST-based parsing is superior for programmatic manipulation
- Source maps are essential for NER span correlation (GLiNER returns `start`/`end` offsets)
- CommonMark spec compliance ensures consistent parsing across documents
- Plugin architecture allows custom syntax (e.g., WEKA-specific admonitions)

**Risk assessment:**
- Migration requires careful testing of edge cases (nested lists, code blocks)
- Some extensions (codehilite) need equivalent mdit-py-plugins
- Frontmatter handling needs explicit plugin (`mdit_py_plugins.front_matter`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Document Ingestion Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Raw .md    │───▶│  markdown-it-py  │───▶│  SyntaxTreeNode  │  │
│  │   Document   │    │   (gfm-like)     │    │      (AST)       │  │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                        │            │
│                      ┌────────────────────────────────┘            │
│                      ▼                                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Section Extractor                           │ │
│  │  - Traverses AST by heading nodes                             │ │
│  │  - Preserves source line mapping (Token.map)                  │ │
│  │  - Builds parent_path for context enrichment                  │ │
│  │  - Extracts code blocks, tables, lists with type metadata     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                      │                                              │
│                      ▼                                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               SemanticChunkerAssembler (Chonkie)              │ │
│  │  - Receives section dicts with body/heading/level/line_map   │ │
│  │  - BGE-M3 adapter for semantic boundary detection            │ │
│  │  - Outputs chunks with provenance metadata                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                      │                                              │
│                      ▼                                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  GLiNER NER Enrichment                        │ │
│  │  - Zero-shot entity extraction with v2 labels                │ │
│  │  - WEKA exclusions filter applied                            │ │
│  │  - Entity spans can correlate to source lines via line_map   │ │
│  │  - Outputs: entity_metadata, _embedding_text, _mentions      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                      │                                              │
│                      ▼                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐   │
│  │   Qdrant (Vectors)  │    │        Neo4j (Graph)            │   │
│  │  - Dense (BGE-M3)   │    │  - Document → Section → Chunk   │   │
│  │  - Sparse (BM25)    │    │  - Entity MENTIONS edges        │   │
│  │  - ColBERT          │    │  - Cross-doc RELATED_TO         │   │
│  │  - Entity-sparse    │    │  - Heading hierarchy            │   │
│  └─────────────────────┘    └─────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Parser Replacement (2-3 days)

**Goal:** Replace `markdown` + BeautifulSoup with `markdown-it-py` while maintaining API compatibility.

#### 1.1 New Parser Module

Create `src/ingestion/parsers/markdown_it_parser.py`:

```python
"""
markdown-it-py based parser for WekaDocs RAG pipeline.

Key improvements over legacy parser:
- Native AST traversal via SyntaxTreeNode
- Source line mapping via Token.map
- CommonMark + GFM compliance
- Plugin extensibility
"""

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.deflist import deflist_plugin

def create_parser() -> MarkdownIt:
    """Create configured markdown-it-py parser with WEKA-appropriate settings."""
    md = (
        MarkdownIt("gfm-like", {"typographer": False})
        .use(front_matter_plugin)
        .use(footnote_plugin)
        .use(deflist_plugin)
        .enable("table")
        .enable("strikethrough")
    )
    return md

def parse_to_ast(raw_text: str) -> SyntaxTreeNode:
    """Parse markdown to SyntaxTreeNode for traversal."""
    md = create_parser()
    tokens = md.parse(raw_text)
    return SyntaxTreeNode(tokens)
```

#### 1.2 Section Extraction via AST Traversal

```python
def extract_sections(ast: SyntaxTreeNode, source_uri: str) -> List[Dict]:
    """
    Extract sections by traversing heading nodes in AST.

    Unlike BeautifulSoup scraping, this:
    - Preserves source line numbers (Token.map)
    - Maintains true parent-child heading relationships
    - Captures block type metadata (paragraph, code_block, table, etc.)
    """
    sections = []
    current_section = None
    heading_stack = []  # For parent_path tracking

    for node in ast.walk():
        if node.type == "heading":
            # Finalize previous section
            if current_section:
                sections.append(_finalize_section(current_section))

            level = node.attrs.get("level", node.tag[1] if node.tag else 2)
            title = _extract_text_content(node)

            # Update heading stack for parent_path
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            parent_path = " > ".join(h["title"] for h in heading_stack)
            heading_stack.append({"level": level, "title": title})

            current_section = {
                "level": level,
                "title": title,
                "parent_path": parent_path,
                "line_start": node.map[0] if node.map else None,
                "line_end": None,
                "content_elements": [],
                "code_blocks": [],
                "tables": [],
                "block_types": [],  # NEW: Track block composition
            }

        elif current_section:
            # Accumulate content with block type tracking
            if node.type == "fence":  # Fenced code block
                code = node.content
                lang = node.info or ""
                current_section["code_blocks"].append({"code": code, "lang": lang})
                current_section["block_types"].append("code")
            elif node.type == "table":
                table_text = _render_table(node)
                current_section["tables"].append(table_text)
                current_section["block_types"].append("table")
            elif node.type == "paragraph":
                text = _extract_text_content(node)
                current_section["content_elements"].append(text)
                current_section["block_types"].append("paragraph")
            # ... handle lists, blockquotes, etc.

    # Finalize last section
    if current_section:
        sections.append(_finalize_section(current_section))

    return sections
```

#### 1.3 Frontmatter Handling

```python
def extract_frontmatter(ast: SyntaxTreeNode) -> Dict:
    """Extract YAML frontmatter from AST (if present)."""
    for node in ast.children:
        if node.type == "front_matter":
            import yaml
            try:
                return yaml.safe_load(node.content) or {}
            except yaml.YAMLError:
                return {}
    return {}
```

#### 1.4 API Compatibility Layer

Maintain the existing `parse_markdown()` signature:

```python
def parse_markdown(source_uri: str, raw_text: str) -> Dict[str, any]:
    """
    Parse Markdown document into Document and Sections.

    API-compatible with legacy parser but uses markdown-it-py internally.
    """
    ast = parse_to_ast(raw_text)
    frontmatter = extract_frontmatter(ast)
    sections = extract_sections(ast, source_uri)

    document = {
        "id": _compute_document_id(source_uri),
        "source_uri": source_uri,
        "source_type": "markdown",
        "title": frontmatter.get("title") or _extract_title_from_ast(ast),
        "version": frontmatter.get("version", "1.0"),
        "checksum": _compute_checksum(raw_text),
        "frontmatter": frontmatter,  # NEW: Preserve full frontmatter
    }

    return {"Document": document, "Sections": sections}
```

---

### Phase 2: Enhanced Metadata for Downstream Pipeline (1-2 days)

**Goal:** Expose new capabilities to Chonkie and GLiNER enrichment.

#### 2.1 Section Schema Enhancement

Add new fields to section dicts:

```python
@dataclass
class SectionMetadata:
    # Existing fields
    id: str
    document_id: str
    level: int
    title: str
    anchor: str
    order: int
    text: str
    tokens: int
    checksum: str

    # NEW: markdown-it-py enhancements
    line_start: Optional[int]      # Source line number (start)
    line_end: Optional[int]        # Source line number (end)
    parent_path: str               # "Parent > Child" heading trail
    block_types: List[str]         # ["paragraph", "code", "table", ...]
    code_ratio: float              # Fraction of content that is code
    has_table: bool                # Quick filter for table-heavy sections
    raw_markdown: str              # Original markdown (for re-parsing)
```

#### 2.2 Context Enrichment for Chunks

Update `SemanticChunkerAssembler` to use `parent_path`:

```python
def _chunk_section(self, document_id: str, section: Dict, start_index: int):
    # Prepend parent path for context (md2chunks pattern)
    text = section.get("body", section.get("text", ""))
    parent_path = section.get("parent_path", "")
    heading = section.get("heading", section.get("title", ""))

    if parent_path:
        context_prefix = f"[Section: {parent_path} > {heading}]\n\n"
    elif heading:
        context_prefix = f"[Section: {heading}]\n\n"
    else:
        context_prefix = ""

    # This context helps BGE-M3 understand document hierarchy
    enriched_text = context_prefix + text
    # ... continue with semantic chunking
```

#### 2.3 Line Map for NER Correlation

Enable GLiNER entity spans to reference source lines:

```python
def enrich_chunks_with_entities(chunks: List[Dict]) -> None:
    # ... existing GLiNER extraction ...

    for chunk, entities in zip(chunks, entities_list):
        if entities and chunk.get("line_start"):
            # NEW: Correlate entity positions to source lines
            for entity in entities:
                # entity.start/end are character offsets in chunk.text
                # We can estimate source line from chunk.line_start + newline count
                entity_line = _estimate_source_line(
                    chunk["text"],
                    entity.start,
                    chunk["line_start"]
                )
                entity.source_line = entity_line
```

---

### Phase 3: Neo4j Graph Model Enhancement (1 day)

**Goal:** Leverage improved structure for richer graph relationships.

#### 3.1 Heading Hierarchy Edges

```cypher
// Create PARENT_HEADING edges based on markdown structure
MATCH (parent:Section {document_id: $doc_id, level: $parent_level})
MATCH (child:Section {document_id: $doc_id, level: $child_level})
WHERE parent.order < child.order
  AND child.parent_path CONTAINS parent.title
  AND NOT EXISTS {
    MATCH (parent)-[:PARENT_HEADING]->(child)
  }
CREATE (parent)-[:PARENT_HEADING {
  parent_path: child.parent_path,
  level_delta: child.level - parent.level
}]->(child)
```

#### 3.2 Block-Type Indexed Sections

```cypher
// Index sections by dominant block type for targeted retrieval
MATCH (s:Section)
WHERE s.code_ratio > 0.5
SET s:CodeSection

MATCH (s:Section)
WHERE s.has_table = true
SET s:TableSection
```

---

### Phase 4: Testing & Migration (1-2 days)

#### 4.1 Compatibility Tests

```python
# tests/unit/test_markdown_it_parser.py

class TestMarkdownItParserCompatibility:
    """Ensure new parser produces equivalent output to legacy parser."""

    def test_section_extraction_matches_legacy(self):
        """Same document should produce same section count and titles."""
        from src.ingestion.parsers.markdown import parse_markdown as legacy_parse
        from src.ingestion.parsers.markdown_it_parser import parse_markdown as new_parse

        doc = "# Title\n\n## Section 1\n\nContent\n\n## Section 2\n\nMore"

        legacy_result = legacy_parse("test.md", doc)
        new_result = new_parse("test.md", doc)

        assert len(legacy_result["Sections"]) == len(new_result["Sections"])
        for l, n in zip(legacy_result["Sections"], new_result["Sections"]):
            assert l["title"] == n["title"]
            assert l["level"] == n["level"]

    def test_frontmatter_extraction(self):
        """YAML frontmatter should be correctly extracted."""
        doc = "---\ntitle: Test\nversion: 1.0\n---\n\n# Content"
        result = parse_markdown("test.md", doc)
        assert result["Document"]["title"] == "Test"
        assert result["Document"]["frontmatter"]["version"] == "1.0"

    def test_code_block_preservation(self):
        """Fenced code blocks should preserve language info."""
        doc = "# Code\n\n```python\ndef foo(): pass\n```"
        result = parse_markdown("test.md", doc)
        assert result["Sections"][0]["code_blocks"][0]["lang"] == "python"

    def test_source_line_mapping(self):
        """Sections should have accurate line numbers."""
        doc = "# First\n\nPara 1\n\n# Second\n\nPara 2"
        result = parse_markdown("test.md", doc)
        assert result["Sections"][0]["line_start"] == 0
        assert result["Sections"][1]["line_start"] == 4
```

#### 4.2 Migration Strategy

1. **Parallel deployment:** Both parsers active, controlled by feature flag
2. **Shadow comparison:** Log differences between parsers during ingestion
3. **Gradual rollout:** Enable new parser for 10% → 50% → 100% of documents
4. **Rollback plan:** Feature flag instantly reverts to legacy parser

```yaml
# config/development.yaml
ingestion:
  parser:
    engine: "markdown-it-py"  # Options: "legacy", "markdown-it-py"
    shadow_mode: true          # Compare outputs, log differences
    fail_on_mismatch: false    # Don't block ingestion on differences
```

---

## Phase 5: Qdrant Multi-Vector Enhancements (1 day)

**Goal:** Leverage markdown-it-py's enhanced metadata for improved multi-vector retrieval.

### Current Multi-Vector Schema

Your Qdrant collection already supports:

| Vector Type | Name | Purpose |
|-------------|------|---------|
| Dense | `content` | Semantic content similarity (BGE-M3) |
| Dense | `title` | Section heading similarity |
| Dense | `doc_title` | Document title similarity |
| ColBERT | `late-interaction` | Token-level MaxSim reranking |
| Sparse | `text-sparse` | BM25-style lexical content matching |
| Sparse | `doc_title-sparse` | Lexical document title matching |
| Sparse | `title-sparse` | Lexical section heading matching |
| Sparse | `entity-sparse` | Lexical entity name matching (GLiNER) |

### 5.1 New Payload Indexes for Structural Queries

Add to `build_qdrant_schema()` payload_indexes:

```python
# src/shared/qdrant_schema.py - Additional payload indexes

# === markdown-it-py Structural Metadata Indexes ===
# Enable filtering and boosting based on document structure
("parent_path", PayloadSchemaType.TEXT),        # Full hierarchy for text search
("parent_path_depth", PayloadSchemaType.INTEGER), # Nesting level (0=root)
("block_type", PayloadSchemaType.KEYWORD),       # "paragraph", "code", "table", "list"
("code_ratio", PayloadSchemaType.FLOAT),         # Fraction of code content
("has_code", PayloadSchemaType.BOOL),            # Quick filter for code blocks
("has_table", PayloadSchemaType.BOOL),           # Quick filter for tables
("line_start", PayloadSchemaType.INTEGER),       # Source line number
("line_end", PayloadSchemaType.INTEGER),         # Source line end
```

### 5.2 Heading Hierarchy Sparse Vector (Optional)

Consider adding a `hierarchy-sparse` vector for lexical matching on full heading paths:

```python
# Example: Chunk under "Installation > Prerequisites > Dependencies"
# hierarchy-sparse would encode: ["installation", "prerequisites", "dependencies"]

if enable_hierarchy_sparse:
    sparse_vectors_config["hierarchy-sparse"] = SparseVectorParams(
        index=SparseIndexParams(on_disk=True)
    )
```

**Use case:** Query "installation dependencies" would match chunks deep in installation hierarchy even if "installation" isn't in the chunk text itself.

**Trade-off:** Adds storage overhead; evaluate if parent_path TEXT index is sufficient.

### 5.3 Enhanced RRF Fusion Weights

Update `config/development.yaml` to leverage new structural signals:

```yaml
search:
  hybrid:
    # Existing RRF field weights
    rrf_field_weights:
      content: 2.0           # Dense semantic - primary signal
      title: 1.5             # Dense heading
      text-sparse: 0.5       # Lexical content
      doc_title-sparse: 0.8  # Lexical doc title
      title-sparse: 0.8      # Lexical heading
      entity-sparse: 0.8     # Lexical entities (GLiNER)

    # NEW: Query-type adaptive weights using structural metadata
    query_type_weights:
      conceptual:
        content: 2.5
        title: 1.0
        text-sparse: 0.3
        # Boost hierarchy for conceptual questions
        hierarchy-sparse: 1.2  # If enabled

      cli:
        content: 1.5
        title: 1.5
        text-sparse: 0.8
        entity-sparse: 1.5     # CLI commands are entities
        # Filter: has_code = true preferred

      procedural:
        content: 2.0
        title: 1.2
        text-sparse: 0.5
        # Prefer chunks with code blocks for how-to queries
        # Post-filter: code_ratio > 0.2 gets boost

      troubleshooting:
        content: 2.0
        text-sparse: 0.6
        entity-sparse: 1.2     # Error codes are entities
```

### 5.4 Block-Type Aware Retrieval

Add post-retrieval filtering/boosting based on block composition:

```python
# src/query/hybrid_retrieval.py

def apply_structural_boost(results: List[ScoredPoint], query_intent: str) -> List[ScoredPoint]:
    """Apply structural metadata boosts based on query intent."""

    for result in results:
        payload = result.payload

        if query_intent == "cli" and payload.get("has_code"):
            result.score *= 1.15  # 15% boost for code-containing chunks

        if query_intent == "reference" and payload.get("has_table"):
            result.score *= 1.10  # 10% boost for table-containing chunks

        # Penalize deeply nested content for overview queries
        if query_intent == "overview":
            depth = payload.get("parent_path_depth", 0)
            if depth > 2:
                result.score *= 0.9  # 10% penalty for deep nesting

    return sorted(results, key=lambda r: r.score, reverse=True)
```

### 5.5 Entity-Sparse Enhancement with Source Lines

GLiNER entities with source line correlation enable richer entity-sparse content:

```python
# src/ingestion/atomic.py - Enhanced entity-sparse text generation

def build_entity_sparse_text(chunk: Dict) -> str:
    """Build text for entity-sparse vector with structural context."""

    parts = []

    # Add normalized entity values (existing)
    entity_values = chunk.get("entity_metadata", {}).get("entity_values_normalized", [])
    parts.extend(entity_values)

    # NEW: Add entity types for type-based queries ("show me all COMMANDS")
    entity_types = chunk.get("entity_metadata", {}).get("entity_types", [])
    parts.extend([t.lower() for t in entity_types])

    # NEW: Add parent path terms for structural entity queries
    parent_path = chunk.get("parent_path", "")
    if parent_path:
        path_terms = [t.strip().lower() for t in parent_path.split(">")]
        parts.extend(path_terms)

    return " ".join(parts)
```

### 5.6 Collection Schema Migration

Add migration script for existing collections:

```python
# scripts/migrate_qdrant_payload_indexes.py

async def add_structural_indexes(client: QdrantClient, collection_name: str):
    """Add markdown-it-py structural metadata indexes to existing collection."""

    new_indexes = [
        ("parent_path", PayloadSchemaType.TEXT),
        ("parent_path_depth", PayloadSchemaType.INTEGER),
        ("block_type", PayloadSchemaType.KEYWORD),
        ("code_ratio", PayloadSchemaType.FLOAT),
        ("has_code", PayloadSchemaType.BOOL),
        ("has_table", PayloadSchemaType.BOOL),
        ("line_start", PayloadSchemaType.INTEGER),
        ("line_end", PayloadSchemaType.INTEGER),
    ]

    for field_name, schema_type in new_indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info(f"Created payload index: {field_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Index already exists: {field_name}")
            else:
                raise
```

### 5.7 Multi-Vector Query Construction

Update query construction to use new metadata for filtering:

```python
# src/query/hybrid_retrieval.py

def build_qdrant_filter(
    query_intent: str,
    require_code: bool = False,
    require_table: bool = False,
    max_depth: Optional[int] = None,
) -> Optional[Filter]:
    """Build Qdrant filter using structural metadata."""

    conditions = []

    if require_code:
        conditions.append(
            FieldCondition(key="has_code", match=MatchValue(value=True))
        )

    if require_table:
        conditions.append(
            FieldCondition(key="has_table", match=MatchValue(value=True))
        )

    if max_depth is not None:
        conditions.append(
            FieldCondition(
                key="parent_path_depth",
                range=Range(lte=max_depth)
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)
```

### Benefits for Multi-Vector Retrieval

| Enhancement | Impact on Retrieval |
|-------------|---------------------|
| `parent_path` TEXT index | Full-text search on heading hierarchy |
| `has_code` / `has_table` filters | 30-50% faster targeted queries (skip irrelevant chunks) |
| `code_ratio` float | Continuous boosting for technical queries |
| `line_start` / `line_end` | Enable source-linked citations in responses |
| `hierarchy-sparse` vector | Hierarchical lexical matching (optional) |
| Block-type boosting | Aligns chunk selection with query intent |

---

## Dependencies

### New Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
markdown-it-py = "^3.0.0"
mdit-py-plugins = "^0.4.0"
linkify-it-py = "^2.0.0"  # For gfm-like preset
```

### Removed Dependencies (after migration)

```toml
# Can be removed once legacy parser is deprecated
markdown = "^3.5"  # No longer needed
# beautifulsoup4 may still be needed for HTML content extraction
```

---

## Benefits Summary

### For Semantic Chunking (Chonkie)

| Aspect | Before | After |
|--------|--------|-------|
| Input quality | HTML-extracted text, some artifacts | Clean AST-derived text |
| Context | Heading only | Full parent_path hierarchy |
| Block awareness | Post-hoc code detection | Native block_types metadata |
| Boundary accuracy | ~85% | ~95% (true markdown boundaries) |

### For NER Enrichment (GLiNER)

| Aspect | Before | After |
|--------|--------|-------|
| Position correlation | Character offsets only | Source line numbers available |
| Entity-section mapping | Chunk-level only | Can trace to exact heading |
| Graph provenance | Limited | Rich MENTIONED_IN edges with line refs |

### For Hybrid Retrieval (BGE-M3 + Neo4j)

| Aspect | Before | After |
|--------|--------|-------|
| Structural queries | Basic heading match | Full hierarchy traversal |
| Code-specific search | Heuristic detection | Native CodeSection label |
| Context for reranking | Chunk text only | Parent path enrichment |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Parsing differences break existing documents | Medium | High | Shadow mode comparison, comprehensive tests |
| Performance regression | Low | Medium | Benchmark parsing time, single-pass should be faster |
| Missing extension equivalents | Low | Low | mdit-py-plugins covers most needs |
| Team unfamiliarity with AST API | Medium | Low | Documentation, code examples |

---

## Success Criteria

1. **Zero regression:** All existing tests pass with new parser
2. **Enhanced metadata:** Sections include `line_start`, `parent_path`, `block_types`
3. **Improved chunking:** Average chunk coherence score increases (measure via manual review)
4. **NER correlation:** GLiNER entities can reference source lines
5. **Performance:** Parsing time ≤ legacy parser (target: 20% faster)

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Core Parser | 2-3 days | `markdown_it_parser.py`, basic tests |
| Phase 2: Enhanced Metadata | 1-2 days | Section schema, Chonkie integration |
| Phase 3: Graph Enhancement | 1 day | Neo4j schema updates, Cypher queries |
| Phase 4: Testing & Migration | 1-2 days | Full test suite, feature flag, rollout |
| Phase 5: Qdrant Multi-Vector | 1 day | Payload indexes, RRF weights, structural filters |

**Total:** 6-9 days

---

## References

- [markdown-it-py Documentation](https://markdown-it-py.readthedocs.io/)
- [mdit-py-plugins](https://github.com/executablebooks/mdit-py-plugins)
- [Chonkie Documentation](https://docs.chonkie.ai/)
- [Neo4j GraphRAG Best Practices](https://neo4j.com/blog/developer/graphrag-llm-knowledge-graph-builder)
- [BGE-M3 Paper](https://huggingface.co/BAAI/bge-m3)
- [md2chunks - Context Enriched Chunking](https://github.com/verloop/md2chunks)
- [Weaviate Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)

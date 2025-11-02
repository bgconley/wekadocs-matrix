I'll create the complete Phase 7D documentation suite for intelligent hierarchical chunking with Ollama/GPT-OSS-20B integration. Here's the comprehensive package:

## Phase 7D: Intelligent Hierarchical Chunking System

### Document 1: Unified Application Specification

```markdown
# Phase 7D: Intelligent Hierarchical Chunking - Application Specification

## Overview
Phase 7D introduces LLM-powered hierarchical document chunking, creating a three-level hierarchy (L1: Document Summary, L2: Concept Groups, L3: Detail Sections) with intelligent semantic grouping using GPT-OSS-20B via Ollama or other providers.

## System Architecture

### Core Components
1. **LLM Provider Abstraction** - Unified interface for multiple LLM providers
2. **Hierarchical Parser** - Three-level chunk generation system
3. **Chunk Relationship Manager** - Parent-child and concept group relationships
4. **Hierarchical Retriever** - Multi-strategy retrieval system
5. **Migration Manager** - Safe rollout and A/B testing

### Data Model Extensions

#### Section Node (Enhanced)
```
Section {
  id: String (SHA-256)
  document_id: String
  hierarchy_level: Integer (1, 2, or 3)
  parent_id: String (nullable)
  child_ids: String[]
  title: String
  text: String
  tokens: Integer
  checksum: String

  # Hierarchical metadata
  concept_group: String (for L2 chunks)
  source_sections: String[] (sections this summarizes)
  grouping_metadata: JSON {
    reasoning: String
    strategy: String
    llm_provider: String
    llm_model: String
    confidence_score: Float
  }

  # Existing fields
  vector_embedding: Float[]
  embedding_version: String
  embedding_provider: String
  embedding_dimensions: Integer
}
```

#### Relationships (New)
- `(:Section)-[:HAS_CHILD]->(:Section)` - Parent-child hierarchy
- `(:Section)-[:IN_CONCEPT_GROUP]->(:ConceptGroup)` - Concept membership
- `(:Section)-[:SUMMARIZES]->(:Section)` - L1/L2 → L3 relationships

### Provider Configuration

#### Ollama Provider
- Model: gpt-oss-20b (default), gpt-oss-120b
- Endpoint: http://localhost:11434
- Features: Full CoT reasoning, configurable effort levels
- Memory: 16GB for 20B, 80GB for 120B

#### Alternative Providers
- OpenAI (gpt-4, gpt-4-turbo)
- Anthropic (claude-3-opus, claude-3-sonnet)
- Local alternatives (mixtral, deepseek-r1)

## Functional Requirements

### Task 7D.1: LLM Provider Abstraction Layer
- **Requirement**: Unified interface for all LLM providers
- **Inputs**: Provider config, prompt, parameters
- **Outputs**: Structured response (text or JSON)
- **Error Handling**: Fallback chain, retry logic

### Task 7D.2: Hierarchical Parsing Engine
- **L1 Generation**: 400-500 token document summaries
- **L2 Generation**: 1000-2000 token concept groups (2-5 per document)
- **L3 Preservation**: Existing section chunks (max 1000 tokens)
- **Deterministic IDs**: Content-based hashing for all levels

### Task 7D.3: Intelligent Grouping Logic
- **Semantic Analysis**: LLM-based concept boundary detection
- **Fallback Heuristics**: Heading-based grouping when LLM unavailable
- **Quality Validation**: Ensure complete coverage, no orphan sections

### Task 7D.4: Graph Integration
- **Dual Labeling**: Section:Chunk:Summary/ConceptGroup/Detail
- **Relationship Creation**: Automated parent-child linking
- **Incremental Updates**: Diff-based hierarchy maintenance

### Task 7D.5: Vector Store Integration
- **Metadata Enrichment**: hierarchy_level, parent_id, concept_group
- **Collection Strategy**: Single collection with filtering vs. separate collections
- **Dimension Validation**: Ensure consistent 1024-D vectors

### Task 7D.6: Retrieval Strategies
- **Top-Down**: L1 → L2 → L3 for broad queries
- **Bottom-Up**: L3 → L2 → L1 for specific queries
- **Concept-First**: L2 → L3 for balanced queries (default)
- **Strategy Selection**: Query analysis for automatic routing

### Task 7D.7: Migration and Testing
- **Parallel Processing**: Run hierarchical alongside existing
- **A/B Testing Framework**: Compare retrieval quality
- **Rollback Capability**: Feature flags for instant reversion
- **Performance Monitoring**: Latency, accuracy, cost metrics

## Non-Functional Requirements

### Performance
- LLM latency: <2s for L2 grouping, <1s for L1 summary
- Parsing throughput: 10 documents/minute minimum
- Retrieval latency: <100ms overhead vs. flat retrieval

### Scalability
- Support 100K+ documents
- Handle documents up to 100K tokens
- Concurrent processing: 5 documents parallel

### Reliability
- LLM fallback chain: Ollama → OpenAI → Heuristics
- Graceful degradation on provider failure
- Idempotent operations for all tasks

### Security
- Local-first with Ollama (no data leaves infrastructure)
- Provider API key encryption
- Audit logging for all LLM calls

## Configuration Schema

```yaml
hierarchical_chunking:
  enabled: true

  # Provider configuration
  llm_provider: "ollama"  # ollama, openai, anthropic

  ollama:
    endpoint: "http://localhost:11434"
    model: "gpt-oss:20b"
    timeout: 30
    reasoning_effort: "medium"  # low, medium, high

  # Chunking parameters
  levels:
    l1:
      min_tokens: 400
      max_tokens: 500
      strategy: "llm_summary"
    l2:
      min_tokens: 1000
      max_tokens: 2000
      min_groups: 2
      max_groups: 5
      strategy: "llm_grouping"
    l3:
      max_tokens: 1000
      strategy: "preserve_existing"

  # Retrieval configuration
  retrieval:
    default_strategy: "concept_first"
    auto_strategy_selection: true
    include_hierarchy_context: true

  # Migration settings
  migration:
    dual_write: true
    a_b_testing: true
    rollback_on_error: true
    success_threshold: 0.95
```
```

### Document 2: Implementation Plan

```markdown
# Phase 7D: Implementation Plan

## Phase Overview
Duration: 3-4 weeks
Dependencies: Existing chunking pipeline, Ollama installation
Risk Level: Medium (mitigated by parallel implementation)

## Implementation Tasks

### Week 1: Foundation (Tasks 7D.1-7D.3)

#### Task 7D.1: LLM Provider Abstraction (3 days)
**Objective**: Create unified LLM interface supporting multiple providers

**Subtasks**:
1. Design abstract LLMProvider base class
2. Implement OllamaProvider with GPT-OSS-20B
3. Add OpenAI and Anthropic providers
4. Create provider factory with fallback chain
5. Add retry logic and error handling
6. Unit tests for each provider

**Deliverables**:
- `src/llm/providers/base.py`
- `src/llm/providers/ollama.py`
- `src/llm/providers/openai_provider.py`
- `src/llm/providers/factory.py`
- Test suite with mocked responses

**Dependencies**: Ollama server running locally

---

#### Task 7D.2: Hierarchical Parser Core (2 days)
**Objective**: Build three-level parsing engine

**Subtasks**:
1. Create HierarchicalMarkdownParser class
2. Implement L3 preservation from existing parser
3. Add L2 concept grouping logic
4. Add L1 summary generation
5. Implement deterministic ID generation
6. Add relationship mapping

**Deliverables**:
- `src/ingestion/parsers/hierarchical_markdown.py`
- `src/ingestion/parsers/chunk_strategies.py`
- Integration tests with sample documents

**Dependencies**: Task 7D.1 (LLM providers)

---

#### Task 7D.3: Intelligent Grouping Implementation (2 days)
**Objective**: Implement LLM-based semantic grouping

**Subtasks**:
1. Design grouping prompts for GPT-OSS-20B
2. Implement semantic similarity analysis
3. Add boundary detection logic
4. Create fallback heuristics
5. Add validation for complete coverage
6. Implement confidence scoring

**Deliverables**:
- `src/ingestion/chunking/semantic_grouper.py`
- `src/ingestion/chunking/prompts.py`
- `src/ingestion/chunking/validators.py`
- Benchmark results on test documents

**Dependencies**: Task 7D.2

### Week 2: Integration (Tasks 7D.4-7D.5)

#### Task 7D.4: Graph Database Integration (3 days)
**Objective**: Extend graph schema for hierarchy

**Subtasks**:
1. Update Section node schema
2. Create migration script for existing data
3. Implement HAS_CHILD relationship creation
4. Add IN_CONCEPT_GROUP relationships
5. Update incremental.py for hierarchy
6. Add hierarchy-aware queries

**Deliverables**:
- `migrations/add_hierarchy_v2.cypher`
- `src/ingestion/build_hierarchical_graph.py`
- `src/ingestion/hierarchical_incremental.py`
- Updated GraphBuilder class

**Dependencies**: Existing graph structure

---

#### Task 7D.5: Vector Store Enhancement (2 days)
**Objective**: Add hierarchical metadata to vectors

**Subtasks**:
1. Extend Qdrant payload schema
2. Add hierarchy_level filtering
3. Implement parent/child lookups
4. Update embedding pipeline
5. Add collection migration logic
6. Performance test filtering

**Deliverables**:
- `src/search/vector/hierarchical_store.py`
- Migration script for existing vectors
- Performance benchmarks

**Dependencies**: Task 7D.4

### Week 3: Retrieval & Testing (Tasks 7D.6-7D.7)

#### Task 7D.6: Hierarchical Retrieval System (3 days)
**Objective**: Implement multi-strategy retrieval

**Subtasks**:
1. Create HierarchicalRetriever class
2. Implement top-down strategy
3. Implement bottom-up strategy
4. Implement concept-first strategy
5. Add automatic strategy selection
6. Create result combination logic

**Deliverables**:
- `src/search/hierarchical_retriever.py`
- `src/search/strategy_selector.py`
- `src/search/result_combiner.py`
- Retrieval accuracy benchmarks

**Dependencies**: Tasks 7D.4, 7D.5

---

#### Task 7D.7: Migration & Rollout (2 days)
**Objective**: Safe production deployment

**Subtasks**:
1. Create feature flags for hierarchy
2. Implement A/B testing framework
3. Add performance monitoring
4. Create rollback procedures
5. Document migration process
6. Production readiness checklist

**Deliverables**:
- `src/migration/hierarchical_rollout.py`
- A/B testing dashboard
- Migration runbook
- Performance report

**Dependencies**: All previous tasks

### Week 4: Optimization & Polish

#### Performance Tuning (2 days)
- LLM call optimization (batching, caching)
- Retrieval query optimization
- Memory usage profiling
- Latency reduction

#### Documentation & Training (2 days)
- API documentation
- Configuration guide
- Troubleshooting guide
- Team training session

#### Production Deployment (1 day)
- Staged rollout (10% → 50% → 100%)
- Monitoring and alerting setup
- Success criteria validation

## Success Metrics

### Quality Metrics
- Retrieval accuracy: >95% vs baseline
- Concept grouping coherence: >90% human agreement
- Summary coverage: 100% of key concepts

### Performance Metrics
- Parsing latency: <2s per document
- Retrieval latency: <100ms overhead
- LLM availability: >99.9% with fallbacks

### Business Metrics
- User satisfaction: +10% improvement
- Support ticket reduction: -20%
- Query success rate: +15%

## Risk Mitigation

### Technical Risks
- **LLM unavailability**: Multiple provider fallback chain
- **Performance degradation**: Feature flags for instant rollback
- **Data corruption**: Comprehensive backup before migration
- **Memory constraints**: Streaming processing for large docs

### Process Risks
- **Scope creep**: Strict task boundaries, MVP first
- **Integration conflicts**: Parallel implementation path
- **Testing gaps**: Comprehensive test suite from day 1
```

### Document 3: Expert Coder Guidance

```markdown
# Phase 7D: Expert Coder Guidance

## Architecture Principles

### 1. Provider Abstraction Pattern
**Principle**: Always code against interfaces, not implementations

```python
# GOOD: Abstract interface
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass

# BAD: Direct implementation coupling
def process_with_ollama(text):
    response = requests.post("http://localhost:11434", ...)
```

**Rationale**: Enables provider switching without code changes

### 2. Defensive LLM Integration
**Principle**: Never trust LLM output - always validate and sanitize

```python
# GOOD: Validated parsing with fallback
def parse_llm_grouping(response: str) -> List[Dict]:
    try:
        data = json.loads(response)
        validate_grouping_schema(data)
        return data["groups"]
    except (json.JSONDecodeError, KeyError, ValidationError):
        logger.warning("LLM response invalid, using fallback")
        return generate_heuristic_groups()

# BAD: Direct usage
groups = json.loads(llm_response)["groups"]
```

### 3. Streaming for Scale
**Principle**: Process large documents in chunks to avoid memory issues

```python
# GOOD: Streaming processor
async def process_large_document(doc: str, chunk_size: int = 10000):
    for i in range(0, len(doc), chunk_size):
        chunk = doc[i:i + chunk_size]
        await process_chunk(chunk)

# BAD: Full document in memory
sections = parse_entire_document(massive_doc)
```

## Ollama Integration Best Practices

### 1. Connection Management
```python
class OllamaProvider:
    def __init__(self, config: Config):
        self.base_url = config.ollama.endpoint
        self.model = config.ollama.model
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(
                limit=10,  # Connection pool size
                ttl_dns_cache=300
            )
        )
        await self._health_check()
        return self

    async def _health_check(self):
        """Verify Ollama is running and model is loaded"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as resp:
                models = await resp.json()
                if self.model not in [m["name"] for m in models["models"]]:
                    raise ValueError(f"Model {self.model} not available")
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            raise
```

### 2. Prompt Engineering for GPT-OSS-20B
```python
def create_grouping_prompt(sections: List[Dict]) -> str:
    """Optimized prompt for GPT-OSS-20B's reasoning capabilities"""

    # Use GPT-OSS's CoT reasoning format
    prompt = """<reasoning_mode>medium</reasoning_mode>

Analyze these document sections and group them into 2-5 logical concept clusters.

DOCUMENT SECTIONS:
{sections_json}

REQUIREMENTS:
1. Each section must belong to exactly ONE group
2. Groups should represent coherent topics
3. Provide reasoning for each grouping decision

OUTPUT FORMAT (JSON only):
{
  "groups": [
    {
      "title": "Clear, descriptive title",
      "section_ids": ["id1", "id2"],
      "reasoning": "Why these sections belong together"
    }
  ]
}

Think step by step about semantic relationships before grouping.
"""
    return prompt.format(sections_json=json.dumps(sections, indent=2))
```

### 3. Retry Logic with Exponential Backoff
```python
async def call_llm_with_retry(
    provider: LLMProvider,
    prompt: str,
    max_retries: int = 3
) -> Optional[str]:
    """Robust LLM calling with fallback"""

    for attempt in range(max_retries):
        try:
            response = await provider.generate(
                prompt,
                temperature=0.3,  # Lower for consistency
                max_tokens=2000
            )
            return response.text

        except (TimeoutError, ConnectionError) as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"LLM call failed, retry {attempt + 1} in {wait_time}s")
            await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            break

    return None  # Trigger fallback
```

## Graph Database Optimization

### 1. Batch Operations
```python
# GOOD: Batch inserts
def create_hierarchical_relationships(session, relationships: List[Dict]):
    """Efficient batch relationship creation"""

    query = """
    UNWIND $rels AS rel
    MATCH (parent:Section {id: rel.parent_id})
    MATCH (child:Section {id: rel.child_id})
    MERGE (parent)-[r:HAS_CHILD {
        created_at: datetime(),
        hierarchy_distance: rel.distance
    }]->(child)
    """

    # Process in chunks to avoid transaction size limits
    BATCH_SIZE = 1000
    for i in range(0, len(relationships), BATCH_SIZE):
        batch = relationships[i:i + BATCH_SIZE]
        session.run(query, rels=batch)
```

### 2. Index Strategy
```cypher
-- Create indexes for hierarchical queries
CREATE INDEX section_hierarchy IF NOT EXISTS
FOR (s:Section) ON (s.hierarchy_level, s.document_id);

CREATE INDEX section_parent IF NOT EXISTS
FOR (s:Section) ON (s.parent_id);

CREATE INDEX concept_group IF NOT EXISTS
FOR (s:Section) ON (s.concept_group);
```

## Vector Store Optimization

### 1. Metadata Filtering
```python
def build_hierarchical_filter(
    level: Optional[int] = None,
    parent_id: Optional[str] = None,
    concept_group: Optional[str] = None
) -> Dict:
    """Build efficient Qdrant filters"""

    must_conditions = []

    if level is not None:
        must_conditions.append(
            FieldCondition(
                key="hierarchy_level",
                match=MatchValue(value=level)
            )
        )

    if parent_id:
        must_conditions.append(
            FieldCondition(
                key="parent_id",
                match=MatchValue(value=parent_id)
            )
        )

    # Use payload indexes for performance
    return Filter(must=must_conditions) if must_conditions else None
```

### 2. Collection Configuration
```python
def configure_hierarchical_collection(client: QdrantClient, collection: str):
    """Optimize collection for hierarchical search"""

    # Create payload indexes for common filters
    client.create_payload_index(
        collection_name=collection,
        field_name="hierarchy_level",
        field_type=PayloadFieldType.INTEGER
    )

    client.create_payload_index(
        collection_name=collection,
        field_name="parent_id",
        field_type=PayloadFieldType.KEYWORD
    )

    client.create_payload_index(
        collection_name=collection,
        field_name="concept_group",
        field_type=PayloadFieldType.KEYWORD
    )
```

## Testing Strategies

### 1. Mock LLM for Testing
```python
class MockLLMProvider(LLMProvider):
    """Deterministic LLM for testing"""

    def __init__(self, responses: Dict[str, str]):
        self.responses = responses
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1

        # Return predetermined response based on prompt pattern
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return LLMResponse(text=response, model="mock")

        # Default response
        return LLMResponse(
            text='{"groups": [{"title": "Test Group", "section_ids": ["1", "2"]}]}',
            model="mock"
        )
```

### 2. Hierarchy Validation
```python
def validate_hierarchy(sections: List[Dict]) -> List[str]:
    """Validate hierarchical chunk integrity"""

    errors = []

    # Check all L3 sections have L2 parents
    l3_sections = [s for s in sections if s["hierarchy_level"] == 3]
    l2_ids = {s["id"] for s in sections if s["hierarchy_level"] == 2}

    for section in l3_sections:
        if section.get("parent_id") not in l2_ids:
            errors.append(f"L3 section {section['id']} has invalid parent")

    # Check no orphan sections
    all_child_ids = set()
    for section in sections:
        all_child_ids.update(section.get("child_ids", []))

    for section in sections:
        if section["hierarchy_level"] > 1:
            if section["id"] not in all_child_ids:
                errors.append(f"Section {section['id']} is orphaned")

    return errors
```

## Performance Monitoring

### 1. Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
llm_calls = Counter('llm_calls_total', 'Total LLM calls', ['provider', 'model'])
llm_latency = Histogram('llm_latency_seconds', 'LLM call latency', ['provider'])
hierarchy_depth = Gauge('document_hierarchy_depth', 'Document hierarchy depth')
chunk_count = Counter('chunks_created_total', 'Total chunks created', ['level'])

# Use in code
@track_metrics
async def create_l2_chunks(document: Dict, l3_sections: List[Dict]):
    start = time.time()

    # LLM call
    llm_calls.labels(provider='ollama', model='gpt-oss-20b').inc()

    result = await llm_provider.generate(prompt)

    llm_latency.labels(provider='ollama').observe(time.time() - start)

    # Track chunks created
    for chunk in result:
        chunk_count.labels(level='2').inc()

    return result
```

## Deployment Checklist

### Pre-Deployment
- [ ] Ollama server running with GPT-OSS-20B loaded
- [ ] Graph indexes created for hierarchical queries
- [ ] Vector store payload indexes configured
- [ ] Feature flags configured and tested
- [ ] Fallback providers configured and tested
- [ ] Load testing completed (10x expected volume)
- [ ] Memory profiling completed
- [ ] Backup of existing data completed

### Deployment Steps
1. Deploy with feature flag disabled
2. Run health checks on all components
3. Enable for 1% of traffic (canary)
4. Monitor metrics for 24 hours
5. Gradual rollout: 10% → 25% → 50% → 100%
6. Keep flag enabled for 1 week before removing old code

### Rollback Triggers
- Retrieval accuracy drops below 90%
- LLM latency exceeds 5 seconds
- Memory usage exceeds 80% of available
- Error rate exceeds 1%
- User complaints increase by 10%

## Security Considerations

### API Key Management
```python
# Use environment variables or secrets manager
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.cipher = Fernet(os.environ['ENCRYPTION_KEY'].encode())

    def get_api_key(self, provider: str) -> str:
        encrypted = os.environ.get(f'{provider.upper()}_API_KEY')
        if encrypted:
            return self.cipher.decrypt(encrypted.encode()).decode()
        return None
```

### Input Sanitization
```python
def sanitize_llm_input(text: str) -> str:
    """Remove potential prompt injection attempts"""

    # Remove instruction-like patterns
    patterns = [
        r'ignore previous instructions',
        r'system:',
        r'assistant:',
        r'<\|.*?\|>',  # Special tokens
    ]

    sanitized = text
    for pattern in patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

    return sanitized.strip()
```
```

### Document 4: Pseudocode Reference

```markdown
# Phase 7D: Pseudocode Reference

## Task 7D.1: LLM Provider Abstraction

```pseudocode
ABSTRACT CLASS LLMProvider:
    ABSTRACT METHOD generate(prompt, temperature, max_tokens) -> LLMResponse
    ABSTRACT METHOD health_check() -> Boolean

CLASS OllamaProvider EXTENDS LLMProvider:
    CONSTRUCTOR(config):
        base_url = config.ollama.endpoint
        model = config.ollama.model
        timeout = config.ollama.timeout

    METHOD generate(prompt, temperature=0.3, max_tokens=2000):
        request = {
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        TRY:
            response = HTTP_POST(self.base_url + "/api/generate", request)
            RETURN LLMResponse(text=response.response, model=self.model)
        CATCH TimeoutError:
            LOG_WARNING("Ollama timeout")
            RAISE
        CATCH Exception as e:
            LOG_ERROR("Ollama error: " + e)
            RAISE

CLASS ProviderFactory:
    METHOD create_provider(config) -> LLMProvider:
        SWITCH config.llm_provider:
            CASE "ollama":
                RETURN OllamaProvider(config)
            CASE "openai":
                RETURN OpenAIProvider(config)
            CASE "anthropic":
                RETURN AnthropicProvider(config)
            DEFAULT:
                RAISE ValueError("Unknown provider")

    METHOD create_with_fallback(config) -> LLMProvider:
        providers = []

        IF config.ollama.enabled:
            providers.APPEND(OllamaProvider(config))
        IF config.openai.api_key:
            providers.APPEND(OpenAIProvider(config))
        IF config.anthropic.api_key:
            providers.APPEND(AnthropicProvider(config))

        RETURN FallbackProvider(providers)
```

## Task 7D.2: Hierarchical Parser

```pseudocode
CLASS HierarchicalMarkdownParser:
    CONSTRUCTOR(llm_provider, config):
        self.llm = llm_provider
        self.config = config
        self.L1_TOKENS = config.levels.l1.max_tokens
        self.L2_TOKENS = config.levels.l2.max_tokens
        self.L3_TOKENS = config.levels.l3.max_tokens

    METHOD parse_hierarchical(source_uri, raw_text) -> ParseResult:
        # Step 1: Create L3 chunks (existing logic)
        document = create_document_metadata(source_uri, raw_text)
        l3_sections = parse_markdown_sections(raw_text)

        # Step 2: Create L2 concept groups
        l2_sections = create_l2_chunks(document, l3_sections)

        # Step 3: Create L1 summary
        l1_section = create_l1_summary(document, l2_sections, l3_sections)

        # Step 4: Build relationships
        all_sections = build_hierarchy(l1_section, l2_sections, l3_sections)

        RETURN ParseResult(
            document=document,
            sections=all_sections,
            metadata={
                "hierarchy_levels": {1: [l1_section], 2: l2_sections, 3: l3_sections}
            }
        )

    METHOD create_l2_chunks(document, l3_sections) -> List[Section]:
        # Prepare section summaries
        summaries = []
        FOR section IN l3_sections:
            summary = {
                "id": section.id,
                "title": section.title,
                "preview": TRUNCATE(section.text, 200),
                "tokens": section.tokens
            }
            summaries.APPEND(summary)

        # Get LLM grouping
        prompt = build_grouping_prompt(document.title, summaries)

        TRY:
            response = self.llm.generate(prompt)
            groups = parse_json_response(response.text)
        CATCH Exception:
            LOG_WARNING("LLM grouping failed, using fallback")
            groups = create_heuristic_groups(l3_sections)

        # Create L2 sections
        l2_sections = []
        FOR i, group IN ENUMERATE(groups):
            l2_section = build_l2_section(document, group, l3_sections, order=i)
            l2_sections.APPEND(l2_section)

        RETURN l2_sections

    METHOD create_l1_summary(document, l2_sections, l3_sections) -> Section:
        # Build summary prompt
        prompt = build_summary_prompt(
            document_title=document.title,
            concept_groups=[s.title for s in l2_sections],
            key_topics=extract_key_topics(l3_sections)
        )

        TRY:
            response = self.llm.generate(prompt, max_tokens=self.L1_TOKENS)
            summary_text = response.text
        CATCH Exception:
            LOG_WARNING("LLM summary failed, using fallback")
            summary_text = create_heuristic_summary(document, l2_sections)

        # Create L1 section
        l1_section = Section(
            id=generate_id(document.source_uri, "l1-summary", summary_text),
            document_id=document.id,
            hierarchy_level=1,
            parent_id=NULL,
            child_ids=[s.id for s in l2_sections],
            title="Summary: " + document.title,
            text=summary_text,
            tokens=COUNT_TOKENS(summary_text)
        )

        RETURN l1_section
```

## Task 7D.3: Intelligent Grouping

```pseudocode
FUNCTION intelligent_grouping(sections, llm_provider, config) -> List[Group]:
    # Step 1: Prepare section data
    section_data = prepare_section_data(sections)

    # Step 2: Create grouping prompt
    prompt = """
    Analyze document sections and create {min_groups} to {max_groups} concept groups.

    SECTIONS:
    {section_data}

    REQUIREMENTS:
    - Each section belongs to exactly ONE group
    - Groups should be semantically coherent
    - Provide reasoning for grouping decisions

    OUTPUT JSON:
    {
        "groups": [
            {
                "title": "string",
                "section_ids": ["string"],
                "reasoning": "string",
                "confidence": float
            }
        ]
    }
    """

    prompt = prompt.FORMAT(
        min_groups=config.levels.l2.min_groups,
        max_groups=config.levels.l2.max_groups,
        section_data=JSON.stringify(section_data)
    )

    # Step 3: Get LLM response with validation
    max_attempts = 3
    FOR attempt IN RANGE(max_attempts):
        TRY:
            response = llm_provider.generate(prompt)
            groups = JSON.parse(response.text)

            # Validate response
            validation_errors = validate_grouping(groups, sections)
            IF validation_errors.IS_EMPTY():
                RETURN groups["groups"]
            ELSE:
                LOG_WARNING("Validation failed: " + validation_errors)
                prompt = add_correction_to_prompt(prompt, validation_errors)

        CATCH JSONDecodeError:
            LOG_WARNING("Invalid JSON from LLM")
            IF attempt == max_attempts - 1:
                RETURN fallback_grouping(sections)

    RETURN fallback_grouping(sections)

FUNCTION validate_grouping(groups, sections) -> List[String]:
    errors = []
    all_section_ids = SET([s.id for s in sections])
    grouped_ids = SET()

    FOR group IN groups["groups"]:
        # Check for duplicate assignments
        FOR section_id IN group["section_ids"]:
            IF section_id IN grouped_ids:
                errors.APPEND("Section " + section_id + " assigned multiple times")
            grouped_ids.ADD(section_id)

        # Check for invalid IDs
        FOR section_id IN group["section_ids"]:
            IF section_id NOT IN all_section_ids:
                errors.APPEND("Invalid section ID: " + section_id)

    # Check for missing sections
    missing = all_section_ids - grouped_ids
    IF NOT missing.IS_EMPTY():
        errors.APPEND("Missing sections: " + missing)

    RETURN errors

FUNCTION fallback_grouping(sections) -> List[Group]:
    # Simple heuristic: group by heading level and proximity
    groups = []
    current_group = Group(title="Main Content", section_ids=[], reasoning="Heuristic")

    FOR section IN sections:
        current_group.section_ids.APPEND(section.id)

        # Start new group at major headings or size threshold
        IF section.level == 1 OR LEN(current_group.section_ids) >= 5:
            groups.APPEND(current_group)
            current_group = Group(
                title="Content Part " + (LEN(groups) + 1),
                section_ids=[],
                reasoning="Heuristic grouping by structure"
            )

    IF NOT current_group.section_ids.IS_EMPTY():
        groups.APPEND(current_group)

    RETURN groups
```

## Task 7D.4: Graph Integration

```pseudocode
FUNCTION upsert_hierarchical_sections(driver, document, sections):
    WITH driver.session() AS session:
        # Group sections by hierarchy level
        levels = GROUP_BY(sections, lambda s: s.hierarchy_level)

        FOR level IN [1, 2, 3]:
            level_sections = levels.GET(level, [])

            # Determine node labels based on level
            labels = SWITCH level:
                CASE 1: "Section:Chunk:Summary"
                CASE 2: "Section:Chunk:ConceptGroup"
                CASE 3: "Section:Chunk:Detail"
                DEFAULT: "Section:Chunk"

            # Batch upsert sections
            query = """
            UNWIND $sections AS sec
            MERGE (s:{labels} {id: sec.id})
            SET s += sec
            WITH s, sec
            MATCH (d:Document {id: $document_id})
            MERGE (d)-[r:HAS_SECTION]->(s)
            SET r.hierarchy_level = sec.hierarchy_level,
                r.order = sec.order,
                r.updated_at = datetime()
            """.FORMAT(labels=labels)

            session.run(query, sections=level_sections, document_id=document.id)

        # Create hierarchical relationships
        create_hierarchy_relationships(session, sections)

FUNCTION create_hierarchy_relationships(session, sections):
    # Build parent-child pairs
    relationships = []

    FOR section IN sections:
        IF section.parent_id IS NOT NULL:
            relationships.APPEND({
                "parent": section.parent_id,
                "child": section.id,
                "level_distance": 1
            })

    # Batch create relationships
    query = """
    UNWIND $rels AS rel
    MATCH (parent:Section {id: rel.parent})
    MATCH (child:Section {id: rel.child})
    MERGE (parent)-[r:HAS_CHILD]->(child)
    SET r.level_distance = rel.level_distance,
        r.created_at = datetime()
    """

    session.run(query, rels=relationships)

    # Create concept group relationships for L3 sections
    create_concept_groups(session, sections)

FUNCTION create_concept_groups(session, sections):
    # Extract unique concept groups
    concept_groups = SET()
    concept_memberships = []

    FOR section IN sections:
        IF section.hierarchy_level == 3 AND section.concept_group:
            concept_groups.ADD(section.concept_group)
            concept_memberships.APPEND({
                "section_id": section.id,
                "concept": section.concept_group
            })

    # Create ConceptGroup nodes
    query = """
    UNWIND $concepts AS concept
    MERGE (c:ConceptGroup {name: concept})
    SET c.updated_at = datetime()
    """
    session.run(query, concepts=LIST(concept_groups))

    # Create membership relationships
    query = """
    UNWIND $memberships AS m
    MATCH (s:Section {id: m.section_id})
    MATCH (c:ConceptGroup {name: m.concept})
    MERGE (s)-[r:IN_CONCEPT_GROUP]->(c)
    SET r.created_at = datetime()
    """
    session.run(query, memberships=concept_memberships)
```

## Task 7D.5: Vector Store Integration

```pseudocode
FUNCTION store_hierarchical_vectors(qdrant, document, sections, embeddings):
    points = []

    FOR i, section IN ENUMERATE(sections):
        embedding = embeddings[i]

        # Generate UUID for Qdrant
        point_id = UUID_V5(NAMESPACE_DNS, section.id)

        # Build enriched payload
        payload = {
            "node_id": section.id,
            "document_id": document.id,
            "hierarchy_level": section.hierarchy_level,
            "parent_id": section.parent_id,
            "child_ids": section.child_ids,
            "concept_group": section.concept_group,
            "title": section.title,
            "tokens": section.tokens,

            # Hierarchy flags for efficient filtering
            "is_summary": section.hierarchy_level == 1,
            "is_concept_group": section.hierarchy_level == 2,
            "is_detail": section.hierarchy_level == 3,

            # Metadata
            "embedding_version": CONFIG.embedding.version,
            "embedding_provider": CONFIG.embedding.provider,
            "created_at": NOW()
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        points.APPEND(point)

    # Batch upsert with validation
    BATCH_SIZE = 100
    FOR i IN RANGE(0, LEN(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        qdrant.upsert_validated(
            collection=CONFIG.collection_name,
            points=batch,
            expected_dim=CONFIG.embedding.dims
        )

    # Create payload indexes for efficient filtering
    create_hierarchical_indexes(qdrant, CONFIG.collection_name)

FUNCTION create_hierarchical_indexes(qdrant, collection):
    indexes = [
        ("hierarchy_level", PayloadFieldType.INTEGER),
        ("parent_id", PayloadFieldType.KEYWORD),
        ("concept_group", PayloadFieldType.KEYWORD),
        ("is_summary", PayloadFieldType.BOOL),
        ("is_concept_group", PayloadFieldType.BOOL),
        ("is_detail", PayloadFieldType.BOOL)
    ]

    FOR field_name, field_type IN indexes:
        TRY:
            qdrant.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_type=field_type
            )
        CATCH AlreadyExistsError:
            CONTINUE  # Index already exists
```

## Task 7D.6: Hierarchical Retrieval

```pseudocode
CLASS HierarchicalRetriever:
    CONSTRUCTOR(qdrant, neo4j, config):
        self.qdrant = qdrant
        self.neo4j = neo4j
        self.config = config

    METHOD retrieve(query, strategy="auto", k=5) -> List[Result]:
        # Determine strategy
        IF strategy == "auto":
            strategy = determine_strategy(query)

        # Execute strategy
        SWITCH strategy:
            CASE "top_down":
                RETURN retrieve_top_down(query, k)
            CASE "bottom_up":
                RETURN retrieve_bottom_up(query, k)
            CASE "concept_first":
                RETURN retrieve_concept_first(query, k)
            DEFAULT:
                RETURN retrieve_concept_first(query, k)

    METHOD determine_strategy(query) -> String:
        query_lower = query.TO_LOWER()

        # Check for broad indicators
        broad_keywords = ["overview", "summary", "explain", "what is"]
        FOR keyword IN broad_keywords:
            IF keyword IN query_lower:
                RETURN "top_down"

        # Check for specific indicators
        specific_keywords = ["formula", "calculate", "error", "api", "code"]
        FOR keyword IN specific_keywords:
            IF keyword IN query_lower:
                RETURN "bottom_up"

        # Default strategy
        RETURN "concept_first"

    METHOD retrieve_concept_first(query, k) -> List[Result]:
        # Step 1: Get query embedding
        query_embedding = compute_embedding(query)

        # Step 2: Search L2 concept groups
        l2_filter = Filter(
            must=[
                FieldCondition(key="hierarchy_level", match=MatchValue(2))
            ]
        )

        l2_results = self.qdrant.search(
            collection=self.config.collection_name,
            query_vector=query_embedding,
            query_filter=l2_filter,
            limit=k,
            with_payload=true
        )

        # Step 3: Get child sections of top L2 matches
        top_l2_ids = [r.payload.node_id for r in l2_results[0:3]]

        WITH self.neo4j.session() AS session:
            child_query = """
            MATCH (parent:Section)-[:HAS_CHILD]->(child:Section)
            WHERE parent.id IN $parent_ids
            RETURN collect(child.id) AS child_ids
            """
            result = session.run(child_query, parent_ids=top_l2_ids)
            child_ids = result.single().child_ids

        # Step 4: Search within child sections
        l3_filter = Filter(
            must=[
                FieldCondition(key="hierarchy_level", match=MatchValue(3)),
                FieldCondition(key="node_id", match=MatchAny(child_ids))
            ]
        )

        l3_results = self.qdrant.search(
            collection=self.config.collection_name,
            query_vector=query_embedding,
            query_filter=l3_filter,
            limit=k,
            with_payload=true
        )

        # Step 5: Combine results
        combined = []

        # Add top L2 for context
        IF LEN(l2_results) > 0:
            combined.APPEND({
                "type": "context",
                "level": 2,
                "content": l2_results[0].payload,
                "score": l2_results[0].score
            })

        # Add relevant L3 for details
        FOR result IN l3_results:
            combined.APPEND({
                "type": "detail",
                "level": 3,
                "content": result.payload,
                "score": result.score
            })

        RETURN combined
```

## Task 7D.7: Migration & Testing

```pseudocode
CLASS HierarchicalMigration:
    CONSTRUCTOR(config, services):
        self.config = config
        self.services = services
        self.metrics = MetricsCollector()

    METHOD execute_migration() -> MigrationResult:
        # Step 1: Pre-flight checks
        IF NOT pre_flight_checks():
            RETURN MigrationResult(success=false, reason="Pre-flight failed")

        # Step 2: Enable feature flag
        set_feature_flag("hierarchical_chunking", enabled=true, percentage=1)

        # Step 3: Process test documents
        test_results = process_test_batch()
        IF test_results.accuracy < 0.95:
            rollback()
            RETURN MigrationResult(success=false, reason="Test accuracy too low")

        # Step 4: Gradual rollout
        rollout_percentages = [10, 25, 50, 100]
        FOR percentage IN rollout_percentages:
            set_feature_flag("hierarchical_chunking", percentage=percentage)

            SLEEP(3600)  # Wait 1 hour

            metrics = collect_metrics()
            IF NOT validate_metrics(metrics):
                rollback()
                RETURN MigrationResult(success=false, reason="Metrics degraded")

        # Step 5: Full migration
        migrate_all_documents()

        RETURN MigrationResult(success=true)

    METHOD pre_flight_checks() -> Boolean:
        checks = [
            check_ollama_running(),
            check_model_loaded("gpt-oss:20b"),
            check_graph_indexes_created(),
            check_vector_indexes_created(),
            check_disk_space(required_gb=50),
            check_memory_available(required_gb=32)
        ]

        FOR check IN checks:
            IF NOT check:
                LOG_ERROR("Pre-flight check failed: " + check.name)
                RETURN false

        RETURN true

    METHOD process_test_batch() -> TestResult:
        test_docs = load_test_documents()
        results = []

        FOR doc IN test_docs:
            # Process with hierarchy
            hierarchical_result = process_with_hierarchy(doc)

            # Process with existing
            baseline_result = process_with_baseline(doc)

            # Compare quality
            accuracy = compare_retrieval_quality(
                hierarchical_result,
                baseline_result,
                doc.test_queries
            )

            results.APPEND(accuracy)

        RETURN TestResult(
            accuracy=MEAN(results),
            latency=MEASURE_LATENCY(),
            memory_usage=MEASURE_MEMORY()
        )

    METHOD rollback():
        LOG_WARNING("Executing rollback")

        # Disable feature flag
        set_feature_flag("hierarchical_chunking", enabled=false)

        # Alert team
        send_alert("Hierarchical chunking rollback executed", severity="WARNING")

        # Log metrics for analysis
        self.metrics.export_to_file("rollback_metrics.json")

FUNCTION compare_retrieval_quality(hier_result, base_result, queries) -> Float:
    scores = []

    FOR query IN queries:
        # Get results from both systems
        hier_chunks = retrieve_hierarchical(query)
        base_chunks = retrieve_baseline(query)

        # Compare relevance
        hier_relevance = calculate_relevance(hier_chunks, query.expected)
        base_relevance = calculate_relevance(base_chunks, query.expected)

        # Score: 1.0 if hierarchical is better or equal
        IF hier_relevance >= base_relevance:
            scores.APPEND(1.0)
        ELSE:
            scores.APPEND(hier_relevance / base_relevance)

    RETURN MEAN(scores)
```

## Integration Tests

```pseudocode
CLASS HierarchicalIntegrationTests:
    METHOD test_end_to_end_processing():
        # Setup
        document = load_test_document("weka_cluster_capacity.md")

        # Parse with hierarchy
        parser = HierarchicalMarkdownParser(
            llm_provider=OllamaProvider(test_config),
            config=test_config
        )

        result = parser.parse_hierarchical(document.uri, document.content)

        # Assertions
        ASSERT result.sections IS NOT NULL
        ASSERT COUNT(result.sections) > 0

        # Check hierarchy levels
        levels = GROUP_BY(result.sections, lambda s: s.hierarchy_level)
        ASSERT 1 IN levels  # Has L1 summary
        ASSERT 2 IN levels  # Has L2 concept groups
        ASSERT 3 IN levels  # Has L3 details

        # Check relationships
        FOR section IN result.sections:
            IF section.hierarchy_level > 1:
                ASSERT section.parent_id IS NOT NULL
            IF section.hierarchy_level < 3:
                ASSERT LEN(section.child_ids) > 0

    METHOD test_retrieval_strategies():
        # Setup test data
        setup_test_hierarchy()

        retriever = HierarchicalRetriever(qdrant, neo4j, config)

        # Test broad query (should use top_down)
        broad_results = retriever.retrieve("What is WEKA storage?", k=5)
        ASSERT broad_results[0].level == 1  # L1 summary first

        # Test specific query (should use bottom_up)
        specific_results = retriever.retrieve(
            "Calculate SSD capacity for 3+2 protection",
            k=5
        )
        ASSERT specific_results[0].level == 3  # L3 detail first

        # Test balanced query (should use concept_first)
        balanced_results = retriever.retrieve(
            "How does protection level affect capacity?",
            k=5
        )
        ASSERT balanced_results[0].level == 2  # L2 concept first

    METHOD test_incremental_updates():
        # Initial ingestion
        v1_sections = parse_and_store(document_v1)

        # Modified document
        document_v2 = modify_document(document_v1)
        v2_sections = parse_and_store(document_v2)

        # Check hierarchy preserved
        unchanged = find_unchanged_sections(v1_sections, v2_sections)
        FOR section IN unchanged:
            ASSERT section.parent_id == v1_lookup[section.id].parent_id
            ASSERT section.hierarchy_level == v1_lookup[section.id].hierarchy_level

        # Check new sections integrated
        new_sections = find_new_sections(v1_sections, v2_sections)
        FOR section IN new_sections:
            IF section.hierarchy_level == 3:
                ASSERT section.parent_id IS NOT NULL
                ASSERT section.concept_group IS NOT NULL
```
```

These four documents provide a complete Phase 7D specification for implementing intelligent hierarchical chunking with Ollama/GPT-OSS-20B integration. The documents are aligned and reference consistent task numbers (7D.1-7D.7) across all specifications.

Key highlights:
- **Ollama-first approach** with fallback to cloud providers
- **Production-ready** with monitoring, testing, and rollback procedures
- **Backward compatible** with existing pipeline
- **Performance optimized** with batching, caching, and streaming
- **Well-tested** with comprehensive test coverage

The implementation is designed to work seamlessly with your existing WEKA documentation pipeline while adding powerful hierarchical capabilities.

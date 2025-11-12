# WekaDocs Matrix - Quick Start Commands

**Pre-Phase 7 Edition**
**Version:** 2.0 (Post Pre-Phase7 Foundation)
**Purpose:** Copy-paste ready commands for common operations

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Database Operations](#database-operations)
3. [Schema Operations](#schema-operations)
4. [Ingestion Operations](#ingestion-operations)
5. [Search Operations](#search-operations)
6. [Maintenance Operations](#maintenance-operations)
7. [Development Commands](#development-commands)
8. [Troubleshooting Commands](#troubleshooting-commands)

---

## Environment Setup

### Start Infrastructure (Docker Compose)

```bash
# Start all services (Neo4j, Qdrant, Redis)
docker-compose up -d

# Check service health
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f neo4j
docker-compose logs -f qdrant
docker-compose logs -f redis
```

### Environment Variables

```bash
# Export required environment variables (development)
export NEO4J_PASSWORD="testpassword123"  # pragma: allowlist secret
export REDIS_PASSWORD="testredis123"  # pragma: allowlist secret
export JWT_SECRET="dev-secret-key-change-in-production"  # pragma: allowlist secret

# Or load from .env file
set -a && source .env && set +a

# Verify environment
env | grep -E "(NEO4J|QDRANT|REDIS|JWT)"
```

### Verify Configuration

```bash
# Test configuration loading
python -c "
from src.shared.config import get_config
config = get_config()
print(f'Model: {config.embedding.embedding_model}')
print(f'Dims: {config.embedding.dims}')
print(f'Version: {config.embedding.version}')
print(f'Provider: {config.embedding.provider}')
"
```

---

## Database Operations

### Neo4j

```bash
# Connect to Neo4j (CLI)
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123

# Check database status
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "CALL dbms.components() YIELD name, versions, edition"

# Check node counts
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"

# Check database size
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "CALL apoc.meta.stats() YIELD nodeCount, relCount, labelCount, propertyKeyCount"
```

### Qdrant

```bash
# List collections
curl -s http://localhost:6333/collections | jq '.'

# Get collection info
curl -s http://localhost:6333/collections/weka_sections | jq '.'

# Count vectors in collection
curl -s http://localhost:6333/collections/weka_sections | jq '.result.points_count'

# Check collection configuration
curl -s http://localhost:6333/collections/weka_sections | jq '.result.config'
```

### Redis

```bash
# Connect to Redis
docker exec -it weka-redis redis-cli -a testredis123

# Check Redis stats (from shell)
docker exec -it weka-redis redis-cli -a testredis123 INFO stats

# Check cache keys
docker exec -it weka-redis redis-cli -a testredis123 KEYS "weka:cache:*" | head -20

# Get cache hit rate
docker exec -it weka-redis redis-cli -a testredis123 INFO stats | grep -E "keyspace_(hits|misses)"
```

---

## Schema Operations

### Initialize Schema

```bash
# Run schema initialization (idempotent)
python scripts/init_schema.py

# Or via Python module
python -m src.shared.schema
```

### Verify Schema

```bash
# Check constraints
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "SHOW CONSTRAINTS"

# Check indexes (regular)
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "SHOW INDEXES YIELD name, type WHERE type <> 'VECTOR'"

# Check vector indexes
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE type = 'VECTOR'"

# Verify vector index dimensions (Pre-Phase 7)
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "SHOW INDEXES YIELD name, options WHERE name = 'section_embeddings'"
```

### Schema v2.1 (Optional - Pre-Phase 7)

```bash
# Apply schema v2.1 (dual-label Sections as Chunks)
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  -f /scripts/create_schema_v2_1.cypher

# Verify dual-labeling
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "MATCH (s:Section) WITH count(s) as sections
   MATCH (c:Chunk) WITH sections, count(c) as chunks
   RETURN sections, chunks, sections = chunks as is_equal"
```

---

## Ingestion Operations

### Single Document Ingestion

```bash
# Ingest single document (via ingestctl)
./scripts/ingestctl ingest data/documents/my-doc.md

# Ingest with tag
./scripts/ingestctl ingest data/documents/my-doc.md --tag "version-5.0"

# Dry run (parse only, no database writes)
./scripts/ingestctl ingest data/documents/my-doc.md --dry-run

# JSON output (for scripting)
./scripts/ingestctl ingest data/documents/my-doc.md --json
```

### Batch Ingestion

```bash
# Ingest entire directory
./scripts/ingestctl ingest data/documents/

# Ingest directory with watch (monitor for changes)
./scripts/ingestctl ingest data/documents/ --watch

# Ingest from URL
./scripts/ingestctl ingest https://example.com/docs/page.html
```

### Ingestion Status

```bash
# Get ingestion job status
./scripts/ingestctl status JOB_ID

# Get status with JSON output
./scripts/ingestctl status JOB_ID --json

# Tail job logs
./scripts/ingestctl tail JOB_ID

# Cancel running job
./scripts/ingestctl cancel JOB_ID

# Get detailed report
./scripts/ingestctl report JOB_ID
```

### Direct Python Ingestion

```bash
# Ingest via Python API
python -c "
from pathlib import Path
from src.ingestion.build_graph import GraphBuilder
from src.ingestion.parsers.markdown import MarkdownParser
from src.shared.config import get_config
from src.shared.connections import get_connection_manager

# Load config and connections
config = get_config()
manager = get_connection_manager()
neo4j = manager.get_neo4j_driver()
qdrant = manager.get_qdrant_client()

# Parse document
parser = MarkdownParser()
doc_path = Path('data/documents/my-doc.md')
doc, sections = parser.parse_file(doc_path)

# Build graph
builder = GraphBuilder(neo4j, config, qdrant)
result = builder.upsert_document(doc, sections, {}, [])

print(f'Ingested: {result[\"sections_upserted\"]} sections')
print(f'Embeddings: {result[\"embeddings_computed\"]}')
"
```

---

## Search Operations

### Hybrid Search (via Python)

```bash
# Execute hybrid search
python -c "
from src.query.hybrid_search import HybridSearchEngine
from src.shared.config import get_config
from src.shared.connections import get_connection_manager

config = get_config()
manager = get_connection_manager()
neo4j = manager.get_neo4j_driver()
qdrant = manager.get_qdrant_client()

engine = HybridSearchEngine(neo4j, qdrant, config)
results = engine.search('How do I configure NFS?', top_k=5)

for i, result in enumerate(results.results[:5], 1):
    print(f'{i}. [{result.score:.3f}] {result.metadata.get(\"title\", \"Untitled\")}')
    print(f'   Section: {result.node_id[:16]}...')
    print(f'   Text: {result.text[:100]}...')
    print()
"
```

### Search via MCP Server

```bash
# Start MCP server
python -m src.mcp_server.main

# In another terminal, call search endpoint (HTTP)
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "name": "search_documentation",
    "arguments": {
      "query": "How do I configure NFS?",
      "top_k": 5,
      "verbosity": "graph"
    }
  }' | jq '.'
```

### Search with Filters

```bash
# Search with embedding version filter (Pre-Phase 7)
python -c "
from src.query.hybrid_search import HybridSearchEngine
from src.shared.config import get_config
from src.shared.connections import get_connection_manager

config = get_config()
manager = get_connection_manager()
engine = HybridSearchEngine(
    manager.get_neo4j_driver(),
    manager.get_qdrant_client(),
    config
)

# Filter by embedding version
results = engine.search(
    'How to upgrade Weka?',
    top_k=5,
    filters={'embedding_version': config.embedding.version}
)

print(f'Found {len(results.results)} results')
for r in results.results[:3]:
    print(f'  - {r.metadata.get(\"title\")}')
"
```

---

## Maintenance Operations

### Database Cleanup

```bash
# Clean up databases (removes all data)
python scripts/cleanup-databases.py --confirm

# Dry run (show what would be deleted)
python scripts/cleanup-databases.py --dry-run

# Clean specific database
python scripts/cleanup-databases.py --neo4j-only
python scripts/cleanup-databases.py --qdrant-only
python scripts/cleanup-databases.py --redis-only
```

### Reconciliation (Drift Detection)

```bash
# Run reconciliation check
python -c "
from src.ingestion.reconcile import ReconciliationService
from src.shared.config import get_config
from src.shared.connections import get_connection_manager

config = get_config()
manager = get_connection_manager()

service = ReconciliationService(
    manager.get_neo4j_driver(),
    manager.get_qdrant_client(),
    config
)

result = service.run_reconciliation()

print(f'Neo4j sections: {result[\"neo4j_count\"]}')
print(f'Qdrant vectors: {result[\"qdrant_count\"]}')
print(f'Drift: {result[\"drift_percentage\"]:.2f}%')
print(f'Status: {result[\"status\"]}')
"
```

### Metrics Check

```bash
# Get Prometheus metrics
curl -s http://localhost:8000/metrics

# Get specific metrics
curl -s http://localhost:8000/metrics | grep embedding

# Get query latency P95
curl -s http://localhost:8000/metrics | grep query_latency | grep quantile=\"0.95\"
```

---

## Development Commands

### Run Tests

```bash
# Run all pre-phase7 tests
for test in tests/test_phase*.py tests/test_integration_prephase7.py; do
  python "$test"
done

# Run specific phase tests
python tests/test_phase1_foundation.py
python tests/test_phase2_provider_wiring.py
python tests/test_phase3_qdrant_safety.py
python tests/test_phase4_ranking_coverage.py
python tests/test_phase5_response_schema.py

# Run integration tests
python tests/test_integration_prephase7.py
```

### Code Quality Checks

```bash
# Check for hardcoded embedding dimensions (should find none)
grep -rn "384" src/ --include="*.py" | grep -v "test" | grep -v "# OK:"

# Check for hardcoded model names
grep -rn "all-MiniLM-L6-v2" src/ --include="*.py" | grep -v "test"

# Verify provider usage (should not see direct SentenceTransformer)
grep -rn "SentenceTransformer(" src/ --include="*.py" | grep -v "provider" | grep -v "test"
```

### Performance Testing

```bash
# Test query latency
python scripts/perf/test_traversal_latency.py

# Test verbosity mode latency
python scripts/perf/test_verbosity_latency.py

# Stress test search
python -c "
import time
from src.query.hybrid_search import HybridSearchEngine
from src.shared.connections import get_connection_manager
from src.shared.config import get_config

config = get_config()
manager = get_connection_manager()
engine = HybridSearchEngine(
    manager.get_neo4j_driver(),
    manager.get_qdrant_client(),
    config
)

queries = ['NFS configuration', 'troubleshooting', 'installation', 'upgrade procedure']
latencies = []

for q in queries * 10:  # 40 queries
    start = time.time()
    results = engine.search(q, top_k=5)
    latency = (time.time() - start) * 1000
    latencies.append(latency)

print(f'Mean latency: {sum(latencies)/len(latencies):.2f}ms')
print(f'P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms')
"
```

---

## Troubleshooting Commands

### Check System Health

```bash
# Check all services are running
docker-compose ps | grep -E "(Up|healthy)"

# Check database connections
python -c "
from src.shared.connections import get_connection_manager

manager = get_connection_manager()

# Test Neo4j
try:
    neo4j = manager.get_neo4j_driver()
    with neo4j.session() as session:
        result = session.run('RETURN 1 as test')
        assert result.single()['test'] == 1
    print('✓ Neo4j connected')
except Exception as e:
    print(f'✗ Neo4j failed: {e}')

# Test Qdrant
try:
    qdrant = manager.get_qdrant_client()
    collections = qdrant.get_collections()
    print(f'✓ Qdrant connected ({len(collections.collections)} collections)')
except Exception as e:
    print(f'✗ Qdrant failed: {e}')

# Test Redis
try:
    redis = manager.get_redis_client()
    redis.ping()
    print('✓ Redis connected')
except Exception as e:
    print(f'✗ Redis failed: {e}')
"
```

### Check Embedding Configuration

```bash
# Verify embedding configuration
python -c "
from src.shared.config import get_config

config = get_config()
emb = config.embedding

print('Embedding Configuration:')
print(f'  Model: {emb.embedding_model}')
print(f'  Dimensions: {emb.dims}')
print(f'  Similarity: {emb.similarity}')
print(f'  Version: {emb.version}')
print(f'  Provider: {emb.provider}')

# Validate provider
from src.providers.embeddings import SentenceTransformersProvider

try:
    provider = SentenceTransformersProvider(
        model_name=emb.embedding_model,
        expected_dims=emb.dims
    )
    print(f'✓ Provider validated: {provider.dims} dims match config')
except Exception as e:
    print(f'✗ Provider validation failed: {e}')
"
```

### Check Vector Dimensions

```bash
# Verify all sections have correct embedding dimensions
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "MATCH (s:Section)
   WHERE s.embedding_dimensions IS NOT NULL
   RETURN s.embedding_dimensions as dims, count(*) as count
   ORDER BY dims"

# Check for sections with mismatched dimensions (should be empty)
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "MATCH (s:Section)
   WHERE s.embedding_dimensions IS NOT NULL
     AND s.embedding_dimensions <> 384
   RETURN count(*) as mismatch_count"

# Verify Qdrant collection dimensions
curl -s http://localhost:6333/collections/weka_sections | \
  jq '.result.config.params.vectors.size'
```

### Debugging Failed Ingestion

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run ingestion with verbose output
./scripts/ingestctl ingest data/documents/test.md --dry-run

# Check ingestion queue (Redis)
docker exec -it weka-redis redis-cli -a testredis123 \
  LLEN "weka:ingestion:queue"

# View failed jobs
docker exec -it weka-redis redis-cli -a testredis123 \
  LRANGE "weka:ingestion:failed" 0 -1
```

### Performance Debugging

```bash
# Check Neo4j query performance
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 \
  "CALL dbms.listQueries() YIELD query, elapsedTimeMillis
   WHERE elapsedTimeMillis > 100
   RETURN query, elapsedTimeMillis ORDER BY elapsedTimeMillis DESC LIMIT 10"

# Check Qdrant search latency
time curl -s -X POST http://localhost:6333/collections/weka_sections/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "limit": 10,
    "with_payload": true
  }' > /dev/null

# Check Redis latency
docker exec -it weka-redis redis-cli -a testredis123 --latency
```

---

## One-Liners (Quick Reference)

```bash
# Start everything
docker-compose up -d && sleep 10 && python scripts/init_schema.py

# Ingest a document
./scripts/ingestctl ingest path/to/doc.md

# Search
python -c "from src.query.hybrid_search import *; from src.shared.config import *; from src.shared.connections import *; c=get_config(); m=get_connection_manager(); e=HybridSearchEngine(m.get_neo4j_driver(),m.get_qdrant_client(),c); print(*[f\"{r.metadata.get('title')}\" for r in e.search('your query').results[:5]], sep='\n')"

# Check health
docker-compose ps && curl -s http://localhost:8000/health | jq '.'

# Run all tests
for t in tests/test_phase*.py tests/test_integration*.py; do python "$t"; done

# Clean and restart
docker-compose down -v && docker-compose up -d

# Check configuration
python -c "from src.shared.config import get_config; c=get_config(); print(f'Model: {c.embedding.embedding_model}, Dims: {c.embedding.dims}')"

# Relationship-builder logging (optional)
LOG_RELATIONSHIP_COUNTS=true docker compose up ingestion-worker
#   When enabled, ingestion logs include both Neo4j mutation counters and optional
#   verification MATCH counts for CHILD_OF / PARENT_OF / NEXT / SAME_HEADING builders.
#   Disable (default) for normal operation to avoid extra MATCH queries.

# Count everything
docker exec -it weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC"
```

---

## Additional Resources

- **Configuration Guide**: `docs/configuration.md` - Detailed configuration reference
- **API Contracts**: `docs/api-contracts.md` - MCP server API documentation
- **Phase Plans**: `docs/implementation-plan.md` - Implementation details
- **Runbooks**: `src/connectors/RUNBOOK.md` - Operational procedures

---

**Document Version:** 2.0
**Last Updated:** 2025-01-23
**Status:** Production Ready

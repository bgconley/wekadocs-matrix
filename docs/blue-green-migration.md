# Blue/Green Collection Migration Pattern

**Pre-Phase 7 Edition**
**Version:** 2.0 (Post Pre-Phase7 Foundation)
**Purpose:** Zero-downtime vector collection migrations
**Status:** Ready for Phase 7 Implementation

## Pre-Phase 7 Foundation

This document describes the blue/green migration pattern for safely transitioning vector collections when changing embedding dimensions (e.g., from 384-dim MiniLM to 768/1024-dim Jina in Phase 7).

**Pre-Phase 7 Preparations:**
- ✅ Configuration-driven dimensions (no hardcoded values)
- ✅ Dimension validation on upsert (`upsert_validated`)
- ✅ Collection creation helper (`create_collection_with_dims`)
- ✅ Embedding version tracking (provenance)
- ✅ Version-based filtering in queries

These foundation pieces make safe migrations possible.

## Overview

The blue/green pattern enables zero-downtime migrations by:
1. Creating a new collection with the target dimensions ("green")
2. Populating it while the old collection ("blue") serves traffic
3. Atomically switching traffic to the new collection via configuration
4. Decommissioning the old collection after verification

**Collection Naming Convention:**
- Blue (current): `weka_sections` (384 dims, MiniLM)
- Green (target): `weka_sections_v2` (768/1024 dims, Jina)
- Pattern: `{base_name}_v{version}`

**Key Principle:** Configuration is the single source of truth. Cutover is a config change + restart, not code changes.

## Migration Steps

### 1. Pre-Migration Validation

```python
# Verify current state
from src.shared.connections import get_connection_manager

manager = get_connection_manager()
qdrant = manager.get_qdrant_client()

# Check existing collection
collections = qdrant.get_collections()
current_collection = "weka_sections"  # Blue collection
print(f"Current collection: {current_collection}")
print(f"Current config dims: {config.embedding.dims}")
```

### 2. Create Green Collection

**Using the Pre-Phase 7 Helper (Recommended):**

```python
# Use the helper method added in Pre-Phase 7 (C4)
# Location: src/shared/connections.py (CompatQdrantClient)
from src.shared.connections import get_connection_manager

manager = get_connection_manager()
qdrant = manager.get_qdrant_client()

new_collection = "weka_sections_v2"  # Green collection
target_dims = 1024  # Jina embeddings-v3

# Safe creation with validation
qdrant.create_collection_with_dims(
    collection_name=new_collection,
    size=target_dims,
    distance="Cosine"
)

print(f"✓ Created green collection: {new_collection} ({target_dims} dims)")
```

**Verify Collection Creation:**

```bash
# Check collection exists and has correct dimensions
curl -s http://localhost:6333/collections/weka_sections_v2 | \
  jq '.result.config.params.vectors'

# Expected output:
# {
#   "size": 1024,
#   "distance": "Cosine"
# }
```

### 3. Populate Green Collection

**Step 3a: Update Configuration for Green Collection**

Create a new config file for the migration (don't modify production config yet):

```yaml
# config/migration-jina.yaml
embedding:
  model_name: "jinaai/jina-embeddings-v3"
  dims: 1024
  similarity: "cosine"
  version: "jina-v3-1024-2025-01-01"  # New version for tracking
  provider: "jina"
  task: "retrieval.passage"

search:
  vector:
    qdrant:
      collection_name: "weka_sections_v2"  # Point to green collection
```

**Step 3b: Run Re-Ingestion to Green Collection**

```bash
# Set environment to use migration config
export CONFIG_PATH=config/migration-jina.yaml
export ENV=migration

# Run full re-ingestion (this will take time for large corpus)
./scripts/ingestctl ingest data/documents/ --tag "jina-migration"

# Monitor progress
./scripts/ingestctl status LATEST --follow
```

**Step 3c: Verify Re-Ingestion Progress**

```bash
# Check green collection point count
curl -s http://localhost:6333/collections/weka_sections_v2 | \
  jq '.result.points_count'

# Compare to blue collection
curl -s http://localhost:6333/collections/weka_sections | \
  jq '.result.points_count'

# Should be equal when migration complete
```

**Pre-Phase 7 Safety:** The `upsert_validated` method (added in C3) ensures all vectors have exactly 1024 dimensions before insertion, preventing corruption.

### 4. Validate Green Collection

**Step 4a: Verify Data Completeness**

```python
# Comprehensive validation script
from src.shared.connections import get_connection_manager
from src.shared.config import get_config

# Load migration config
import os
os.environ['CONFIG_PATH'] = 'config/migration-jina.yaml'
config = get_config()

manager = get_connection_manager()
qdrant = manager.get_qdrant_client()
neo4j = manager.get_neo4j_driver()

# Check collection metadata
green_info = qdrant.get_collection("weka_sections_v2")
print(f"✓ Green collection exists")
print(f"  Points: {green_info.points_count}")
print(f"  Dimensions: {green_info.config.params.vectors.size}")
print(f"  Distance: {green_info.config.params.vectors.distance}")

# Check Neo4j section count
with neo4j.session() as session:
    result = session.run("""
        MATCH (s:Section)
        WHERE s.embedding_version = $version
        RETURN count(s) as count
    """, version=config.embedding.version)
    neo4j_count = result.single()['count']

print(f"\nData Parity Check:")
print(f"  Neo4j sections: {neo4j_count}")
print(f"  Qdrant vectors: {green_info.points_count}")
print(f"  Match: {neo4j_count == green_info.points_count}")

if neo4j_count != green_info.points_count:
    print(f"⚠️  WARNING: Data drift detected!")
    print(f"  Difference: {abs(neo4j_count - green_info.points_count)} sections")
```

**Step 4b: Test Search Quality**

```python
# Test searches with green collection
from src.query.hybrid_search import HybridSearchEngine
from src.providers.embeddings import JinaEmbeddingsProvider

# Initialize Jina provider
jina_provider = JinaEmbeddingsProvider(
    api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v3",
    expected_dims=1024
)

# Create search engine pointing to green collection
search_engine = HybridSearchEngine(
    neo4j_driver=neo4j,
    qdrant_client=qdrant,
    config=config,
    embedder=jina_provider
)

# Run test queries
test_queries = [
    "How do I configure NFS?",
    "What are the installation prerequisites?",
    "How to upgrade Weka cluster?",
    "Troubleshooting performance issues"
]

print("\nSearch Quality Tests:")
for query in test_queries:
    results = search_engine.search(query, top_k=5)
    print(f"✓ '{query}': {len(results.results)} results")
    if len(results.results) == 0:
        print(f"  ⚠️  WARNING: No results for query!")
```

**Step 4c: Automated Validation Script**

```bash
# Run comprehensive validation
python -c "
from src.ingestion.reconcile import ReconciliationService
from src.shared.config import get_config
from src.shared.connections import get_connection_manager
import os

os.environ['CONFIG_PATH'] = 'config/migration-jina.yaml'
config = get_config()
manager = get_connection_manager()

service = ReconciliationService(
    manager.get_neo4j_driver(),
    manager.get_qdrant_client(),
    config
)

result = service.run_reconciliation()

print(f'Reconciliation Status: {result[\"status\"]}')
print(f'Neo4j sections: {result[\"neo4j_count\"]}')
print(f'Qdrant vectors: {result[\"qdrant_count\"]}')
print(f'Drift: {result[\"drift_percentage\"]:.2f}%')

if result['status'] != 'OK':
    print('⚠️  Migration validation FAILED')
    exit(1)
else:
    print('✓ Migration validation PASSED')
"
```

### 5. Atomic Cutover

**The cutover is a configuration change only. No code changes required.**

**Step 5a: Update Production Configuration**

```yaml
# config/production.yaml (or development.yaml for testing)
embedding:
  model_name: "jinaai/jina-embeddings-v3"  # Changed from MiniLM
  dims: 1024  # Changed from 384
  similarity: "cosine"
  version: "jina-v3-1024-2025-01-01"  # Changed from miniLM-v2...
  provider: "jina"  # Changed from sentence-transformers

search:
  vector:
    qdrant:
      collection_name: "weka_sections_v2"  # Changed from weka_sections
```

**Step 5b: Perform Cutover**

```bash
# 1. Backup current config
cp config/production.yaml config/production.yaml.backup

# 2. Deploy new config
cp config/migration-jina.yaml config/production.yaml

# 3. Restart services (Docker Compose example)
docker-compose restart mcp-server

# 4. Verify services started successfully
docker-compose ps
docker-compose logs mcp-server | grep -E "(Embedding configuration|validation)"

# Expected log:
# INFO Embedding configuration loaded: model=jinaai/jina-embeddings-v3, dims=1024, version=jina-v3-1024-2025-01-01, provider=jina
# INFO Configuration validation successful
```

**Step 5c: Verify Cutover**

```bash
# Test that queries now use green collection
python -c "
from src.shared.config import get_config

config = get_config()
print(f'Active collection: {config.search.vector.qdrant.collection_name}')
print(f'Active dimensions: {config.embedding.dims}')
print(f'Active version: {config.embedding.version}')

# Should show:
#   Active collection: weka_sections_v2
#   Active dimensions: 1024
#   Active version: jina-v3-1024-2025-01-01
"

# Test a search query
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "name": "search_documentation",
    "arguments": {
      "query": "How do I configure NFS?",
      "top_k": 3
    }
  }' | jq '.results | length'

# Should return results (e.g., 3)
```

**Cutover Complete!** Traffic now flows to green collection with Jina embeddings.

### 6. Monitor & Verify

**Post-Cutover Monitoring (Critical Window: 24-48 hours)**

```bash
# Monitor Prometheus metrics
curl -s http://localhost:8000/metrics | grep -E "(query_latency|search_total|error)"

# Check query latency P95 (should be similar to pre-cutover)
curl -s http://localhost:8000/metrics | \
  grep query_latency | \
  grep 'quantile="0.95"'

# Check error rate (should be low)
curl -s http://localhost:8000/metrics | \
  grep -E "(error_total|failed)"

# Monitor Qdrant collection access patterns
curl -s http://localhost:6333/collections/weka_sections_v2 | \
  jq '.result.points_count, .result.segments_count'

# Verify OLD collection is not receiving new traffic
# Check metrics - searches should be against weka_sections_v2 only
docker-compose logs mcp-server | grep -i "qdrant" | tail -20
```

**Health Checks:**

```python
# Automated health monitoring script
import time
from src.query.hybrid_search import HybridSearchEngine
from src.shared.connections import get_connection_manager
from src.shared.config import get_config

config = get_config()
manager = get_connection_manager()

# Run test queries every 5 minutes
test_queries = [
    "NFS configuration",
    "troubleshooting",
    "installation prerequisites"
]

print("Starting post-cutover monitoring...")
for i in range(12):  # Monitor for 1 hour (12 * 5 min)
    print(f"\n--- Check {i+1}/12 ---")

    engine = HybridSearchEngine(
        manager.get_neo4j_driver(),
        manager.get_qdrant_client(),
        config
    )

    for query in test_queries:
        start = time.time()
        try:
            results = engine.search(query, top_k=5)
            latency = (time.time() - start) * 1000

            print(f"✓ '{query}': {len(results.results)} results, {latency:.0f}ms")

            if latency > 1000:  # Alert if > 1 second
                print(f"  ⚠️  HIGH LATENCY DETECTED")
        except Exception as e:
            print(f"✗ '{query}': ERROR - {e}")

    time.sleep(300)  # 5 minutes

print("\nMonitoring complete")
```

### 7. Decommission Blue Collection

**After 24-48 hour verification period with no issues:**

```bash
# Step 1: Create snapshot of blue collection (for safety)
curl -X POST http://localhost:6333/collections/weka_sections/snapshots

# Verify snapshot created
curl -s http://localhost:6333/collections/weka_sections/snapshots | jq '.'

# Step 2: Document collection metadata before deletion
curl -s http://localhost:6333/collections/weka_sections | \
  jq '{name: .result.name, points: .result.points_count, config: .result.config}' \
  > weka_sections_blue_metadata_$(date +%Y%m%d).json

echo "✓ Blue collection metadata archived"

# Step 3: Delete blue collection
curl -X DELETE http://localhost:6333/collections/weka_sections

# Step 4: Verify deletion
curl -s http://localhost:6333/collections | jq '.result.collections[].name'

# Should only show: "weka_sections_v2"
```

**Cleanup Checklist:**

```bash
# ✓ Verify green collection stable for 48+ hours
# ✓ Verify metrics (latency, error rate) are acceptable
# ✓ Verify user feedback is positive
# ✓ Snapshot created and verified
# ✓ Metadata archived
# ✓ Team notified of decommissioning
# ✓ Rollback plan documented (if needed in future)

# Then proceed with deletion
```

**Free up resources:**

```bash
# Check disk space reclaimed
docker exec -it weka-qdrant du -sh /qdrant/storage

# Verify only green collection exists
docker exec -it weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "MATCH (s:Section)
   RETURN s.embedding_version as version, count(*) as count
   ORDER BY count DESC"

# Should show only: jina-v3-1024-2025-01-01
```

## Rollback Procedure

**If issues detected during monitoring window:**

### Immediate Rollback (< 5 minutes)

```bash
# Step 1: Restore backup configuration
cp config/production.yaml.backup config/production.yaml

# Step 2: Restart services
docker-compose restart mcp-server

# Step 3: Verify rollback
python -c "
from src.shared.config import get_config
config = get_config()
print(f'Collection: {config.search.vector.qdrant.collection_name}')
print(f'Dimensions: {config.embedding.dims}')
print(f'Version: {config.embedding.version}')
"

# Should show:
#   Collection: weka_sections (blue)
#   Dimensions: 384
#   Version: miniLM-L6-v2-2024-01-01

# Step 4: Verify services healthy
curl -s http://localhost:8000/health | jq '.'

# Step 5: Test queries
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{"name": "search_documentation", "arguments": {"query": "test", "top_k": 3}}' \
  | jq '.results | length'
```

### Rollback Triggers

Consider rollback if:
- **Latency P95 > 2x baseline** for more than 5 minutes
- **Error rate > 5%** for more than 2 minutes
- **No results returned** for common queries
- **User complaints** about result quality
- **Infrastructure issues** (OOM, crashes)

### Post-Rollback Actions

```bash
# 1. Document the issue
echo "Migration rolled back at $(date)" >> migration_log.txt
echo "Reason: [DESCRIBE ISSUE]" >> migration_log.txt

# 2. Collect diagnostics
docker-compose logs mcp-server > rollback_logs_$(date +%Y%m%d_%H%M%S).txt

# 3. Analyze root cause
# - Check green collection data quality
# - Review embedding generation logs
# - Compare result quality between collections

# 4. Fix and re-attempt migration when ready
```

## Safety Considerations

### Pre-Phase 7 Preparations

The following safety mechanisms were added in Pre-Phase 7:

1. **Dimension Validation**: `upsert_validated()` prevents dimension mismatches
2. **Version Filtering**: Queries filter by `embedding_version` to prevent mixing
3. **Collection Helper**: `create_collection_with_dims()` ensures correct setup
4. **Config-Driven**: All dimensions come from configuration, no hardcoding

### Phase 7 Implementation

When implementing the actual migration in Phase 7:

1. **Never modify collections in-place** - Always create new
2. **Test thoroughly** in dev/staging before production
3. **Monitor closely** during and after cutover
4. **Keep blue collection** for fast rollback (24-48 hours minimum)
5. **Validate data parity** - Ensure all documents migrated

## Example Timeline

### Small Corpus (< 10,000 documents)

```
Hour 0:   Create green collection (5 min)
Hour 0-2: Populate green collection (2 hours)
Hour 2:   Validate green collection (30 min)
Hour 3:   Cutover during low-traffic window (5 min)
Hour 3-6: Monitor closely, ready to rollback (3 hours)
Hour 27:  Decommission blue collection if stable (24+ hours later)
```

### Large Corpus (> 100,000 documents)

```
Day 0:     Create green collection
Day 0-2:   Populate green collection (may take 24-48 hours)
Day 2:     Validate green collection (4-8 hours)
Day 3:     Cutover during weekend low-traffic window
Day 3-5:   Monitor closely, ready to rollback (48 hours)
Day 7:     Decommission blue collection if stable (96+ hours later)
```

### Critical Path Items

- **Pre-migration**: Test in dev/staging first (1-2 weeks)
- **Re-ingestion**: Parallelizable, but API rate limits may apply
- **Validation**: Cannot skip - data parity is critical
- **Cutover**: Choose lowest-traffic window (weekend, night)
- **Monitoring**: First 24 hours are most critical

## Code Integration Points

The following files interact with the collection name and must handle the migration:

- `src/ingestion/build_graph.py` - Writes vectors
- `src/query/hybrid_search.py` - Reads vectors
- `src/ingestion/reconcile.py` - Syncs vectors
- `src/ingestion/incremental.py` - Updates vectors
- `src/mcp_server/query_service.py` - Query endpoint

All use `config.search.vector.qdrant.collection_name` making the cutover centralized.

## Automation Script (Future)

```python
# Future: migration_orchestrator.py
class BlueGreenMigration:
    """Orchestrates blue/green migration for embedding changes"""

    def __init__(self, config, target_model, target_dims):
        self.config = config
        self.target_model = target_model
        self.target_dims = target_dims

    def execute(self):
        # 1. Create green collection
        # 2. Re-embed all documents
        # 3. Validate completeness
        # 4. Run comparison tests
        # 5. Perform cutover
        # 6. Monitor metrics
        # 7. Cleanup after grace period
        pass
```

## Pre-Phase 7 Foundation Checklist

Before executing migration, verify these Pre-Phase 7 enhancements are in place:

```bash
# 1. Configuration is single source of truth
grep -r "384" src/ --include="*.py" | grep -v "test" | grep -v "# OK:"
# Should find ZERO hardcoded dimensions

# 2. Dimension validation is active
python -c "
from src.shared.config import get_config
config = get_config()
print('Feature flags:')
print(f'  validate_dimensions: {config.feature_flags.validate_dimensions}')
print(f'  use_embedding_provider: {config.feature_flags.use_embedding_provider}')
"
# Both should be true

# 3. Provider abstraction is used
grep -rn "SentenceTransformer(" src/ --include="*.py" | \
  grep -v "provider" | grep -v "test" | wc -l
# Should be 0 (no direct usage)

# 4. Embedding version tracking is active
docker exec -it weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "MATCH (s:Section) WHERE s.embedding_version IS NOT NULL RETURN count(s)"
# Should return total section count (all have versions)

# 5. Collection helper exists
python -c "
from src.shared.connections import CompatQdrantClient
import inspect
assert 'create_collection_with_dims' in dir(CompatQdrantClient)
print('✓ Collection helper method exists')
"
```

**All checks must pass before proceeding with migration.**

## Advanced Topics

### Gradual Rollout (Future Enhancement)

```python
# Route percentage of traffic to green collection
# Useful for A/B testing or cautious rollout

class GradualRouter:
    def __init__(self, blue_collection, green_collection, green_percentage=10):
        self.blue = blue_collection
        self.green = green_collection
        self.green_pct = green_percentage

    def get_collection(self, query_id):
        # Use query_id hash to deterministically route
        if hash(query_id) % 100 < self.green_pct:
            return self.green
        return self.blue

# Week 1: 10% traffic to green
# Week 2: 50% traffic to green
# Week 3: 100% traffic to green
```

### Multi-Region Migration

For globally distributed deployments:
1. Migrate one region at a time
2. Monitor each region for 24 hours before proceeding
3. Keep regions at least 24 hours apart
4. Have region-specific rollback plans

### Capacity Planning

```bash
# Estimate storage requirements
# Blue collection size
du -sh /qdrant/storage/collections/weka_sections

# Green will be similar (vectors are similar size)
# Need 2x storage during migration
# Plan for 2.5x to be safe (overhead, snapshots)
```

## Troubleshooting

### Issue: Green collection point count != Blue

**Diagnosis:**
```bash
# Check for failed ingestion jobs
./scripts/ingestctl status --failed

# Check for dimension mismatches
docker-compose logs mcp-server | grep -i "dimension mismatch"
```

**Resolution:**
- Re-run ingestion for failed documents
- Verify provider outputs correct dimensions
- Check reconciliation service for drift details

### Issue: Search results quality degraded

**Diagnosis:**
```python
# Compare search results between collections
# Use same query on both collections and compare

from src.query.hybrid_search import HybridSearchEngine

# Blue collection results
blue_engine = HybridSearchEngine(..., collection="weka_sections")
blue_results = blue_engine.search("test query", top_k=10)

# Green collection results
green_engine = HybridSearchEngine(..., collection="weka_sections_v2")
green_results = green_engine.search("test query", top_k=10)

# Compare
print("Blue top results:", [r.node_id for r in blue_results.results[:5]])
print("Green top results:", [r.node_id for r in green_results.results[:5]])
```

**Resolution:**
- Different models may rank differently (expected)
- Evaluate with human judgments
- Consider model-specific tuning

### Issue: High latency after cutover

**Diagnosis:**
```bash
# Check Qdrant collection optimization
curl -s http://localhost:6333/collections/weka_sections_v2 | \
  jq '.result.optimizer_status'

# Check index status
curl -s http://localhost:6333/collections/weka_sections_v2 | \
  jq '.result.status'
```

**Resolution:**
```bash
# Trigger optimization if needed
curl -X POST http://localhost:6333/collections/weka_sections_v2/optimizer
```

## Notes

- **Universality**: This pattern supports ANY embedding model change, not just Jina
- **Distance metrics**: Can be used for distance changes (cosine → dot product)
- **Gradual rollout**: Supports feature-flag based percentage routing
- **Capacity planning**: Need 2x vector storage during migration (temporary)
- **Zero downtime**: Configuration change + restart = < 5 minute cutover
- **Rollback**: < 5 minute rollback window if issues detected

---

**Status**: Documentation complete for Pre-Phase 7
**Implementation**: Foundation ready, execution deferred to Phase 7
**Pre-requisites**: All Pre-Phase 7 workstreams (A-H) complete ✅
**Next**: Execute migration during Phase 7 Jina integration

**Document Version:** 2.0
**Last Updated:** 2025-01-23
**Status:** Production Ready

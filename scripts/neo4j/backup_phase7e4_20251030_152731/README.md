# Phase 7E-4 Complete Database Backup
**Generated:** 2025-10-30 15:27:31
**Schema Version:** v2.1
**Purpose:** Complete recovery point for Phase 7E-4 production-ready state

## 📁 Backup Contents

| File | Description | Usage |
|------|-------------|-------|
| `RECOVERY_MASTER.sh` | Master orchestration script | Run this for complete recovery |
| `neo4j_complete_ddl.cypher` | Neo4j full schema DDL | Contains all constraints, indexes, SchemaVersion |
| `qdrant_collections_setup.sh` | Qdrant collections configuration | Creates/verifies 1024-D collections |
| `redis_configuration.md` | Redis documentation & commands | Reference for Redis setup |
| `README.md` | This documentation | Backup overview and instructions |

## 🚀 Quick Recovery

### One-Command Recovery
```bash
# Full recovery with safety checks
bash RECOVERY_MASTER.sh

# Dry run to see what would happen
bash RECOVERY_MASTER.sh --dry-run

# Skip backup of current state
bash RECOVERY_MASTER.sh --skip-backup
```

### Manual Recovery Steps

1. **Neo4j Recovery**
```bash
docker cp neo4j_complete_ddl.cypher weka-neo4j:/tmp/
docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD -f /tmp/neo4j_complete_ddl.cypher
```

2. **Qdrant Recovery**
```bash
bash qdrant_collections_setup.sh
```

3. **Redis Recovery**
```bash
# Only if needed - Redis is ephemeral cache
docker exec weka-redis redis-cli -a $REDIS_PASSWORD -n 1 FLUSHDB ASYNC
```

## 📊 Database Specifications

### Neo4j (Enterprise 5.15)
- **Schema Version:** v2.1
- **Constraints:** 16 uniqueness constraints
- **B-tree Indexes:** 27 performance indexes
- **Vector Indexes:** 2 (1024-D, cosine distance)
- **Fulltext Indexes:** 3 for search functionality
- **Critical Indexes:** `section_document_id_idx`, `section_level_idx`, `section_order_idx`
- **Singleton Node:** SchemaVersion (required for health checks)

### Qdrant (v1.7.4)
- **Collections:** chunks, sections
- **Vector Dimensions:** 1024
- **Distance Metric:** Cosine
- **Quantization:** INT8 scalar
- **HNSW Config:** m=16, ef_construct=200

### Redis (7.2-alpine)
- **Production DB:** 0 (cache, locks, queues)
- **Test DB:** 1 (safe to flush)
- **Session DB:** 2 (auto-expiring)
- **Memory Limit:** 256MB
- **Eviction Policy:** allkeys-lru

## 🔍 Verification Commands

### Check Neo4j Schema
```bash
# Verify SchemaVersion
docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv.version"

# Count constraints and indexes
docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "SHOW CONSTRAINTS YIELD name RETURN count(*) as constraints"

docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "SHOW INDEXES YIELD name RETURN count(*) as indexes"
```

### Check Qdrant Collections
```bash
# List collections
curl -sS http://localhost:6333/collections | jq '.result.collections'

# Verify chunk collection config
curl -sS http://localhost:6333/collections/chunks | jq '.result.config.params.vectors'
```

### Check Redis
```bash
# Test connectivity
docker exec weka-redis redis-cli -a $REDIS_PASSWORD ping

# Check keyspace
docker exec weka-redis redis-cli -a $REDIS_PASSWORD INFO keyspace
```

### Check MCP Server Health
```bash
# Health endpoint
curl -sS http://localhost:8000/health | jq

# Metrics endpoint
curl -sS http://localhost:8000/metrics
```

## ⚠️ Important Notes

1. **Index Naming Convention**
   - Health checks expect indexes with `_idx` suffix
   - DDL includes both versions for compatibility
   - Example: `section_document_id` AND `section_document_id_idx`

2. **Service Dependencies**
   - Stop application services before recovery
   - Neo4j must be recovered before starting MCP server
   - Health checks will fail without SchemaVersion node

3. **Data Preservation**
   - This backup contains SCHEMA ONLY
   - No data (nodes, relationships, vectors) included
   - Suitable for clean environment setup

4. **Environment Variables Required**
   ```bash
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=testpassword123
   export REDIS_PASSWORD=testredis123
   export QDRANT=http://localhost:6333
   ```

## 🧪 Testing After Recovery

Run comprehensive smoke tests:
```bash
python3 tests/test_phase7e4_observability.py -v
```

Or use the smoke test script from recovery session:
```python
# Test all databases are correctly configured
python3 /tmp/smoke_tests.py
```

## 📝 Recovery Checklist

- [ ] Environment variables set
- [ ] Docker services running
- [ ] Application services stopped
- [ ] Neo4j DDL applied
- [ ] SchemaVersion node verified
- [ ] Qdrant collections created
- [ ] Redis test DB flushed
- [ ] Application services restarted
- [ ] Health checks passing
- [ ] Smoke tests passing

## 🔄 Updates to This Backup

If schema changes occur:
1. Export new DDL: `SHOW CONSTRAINTS; SHOW INDEXES;`
2. Update `neo4j_complete_ddl.cypher`
3. Test recovery in staging environment
4. Update this README with changes
5. Tag in git: `git tag backup-phase7e4-v2`

## 📞 Support

For issues with recovery:
1. Check Docker logs: `docker-compose logs [service]`
2. Verify network connectivity between services
3. Ensure correct passwords in environment
4. Check disk space for databases
5. Review health check logs in MCP server

---

**Backup Location:** `/Users/brennanconley/vibecode/wekadocs-matrix/scripts/neo4j/backup_phase7e4_20251030_152731/`
**Git Commit:** (add commit hash after committing)
**Created By:** Recovery process after Phase 7E-4 integration

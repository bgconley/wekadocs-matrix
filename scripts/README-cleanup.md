# Database Cleanup Script

**Location:** `scripts/cleanup-databases.py`

## Purpose

Performs surgical cleanup of test/development data from all databases while **preserving schemas**:

- **Neo4j:** Deletes all nodes and relationships, preserves constraints and indexes
- **Qdrant:** Deletes all vector collections, schemas auto-recreate on next use
- **Redis:** Flushes specified database (default: db=1 for tests)

## Safety Features

✅ **Schema Preservation**
- Neo4j constraints and indexes are NEVER deleted
- Qdrant collection schemas auto-recreate
- Only data is removed

✅ **Dry Run Mode**
- Preview changes before applying
- Zero risk testing

✅ **Comprehensive Reporting**
- Before/after state snapshots
- Timestamped JSON reports
- Schema verification checks

✅ **Selective Cleanup**
- Skip specific databases
- Choose Redis database
- Flexible execution

## Quick Start

### Basic Usage

```bash
# Standard cleanup (recommended for most cases)
python scripts/cleanup-databases.py

# Preview changes without applying
python scripts/cleanup-databases.py --dry-run

# Quiet mode (minimal output)
python scripts/cleanup-databases.py --quiet
```

### Advanced Usage

```bash
# Clean only Neo4j
python scripts/cleanup-databases.py --skip-qdrant --skip-redis

# Clean only Qdrant
python scripts/cleanup-databases.py --skip-neo4j --skip-redis

# Clean only Redis
python scripts/cleanup-databases.py --skip-neo4j --skip-qdrant

# Clean production Redis db (USE WITH CAUTION!)
python scripts/cleanup-databases.py --redis-db 0 --skip-neo4j --skip-qdrant

# Custom report directory
python scripts/cleanup-databases.py --report-dir /path/to/reports

# Dry run with custom Redis db
python scripts/cleanup-databases.py --dry-run --redis-db 2
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Show changes without applying | `false` |
| `--redis-db N` | Redis database number | `1` (test db) |
| `--skip-neo4j` | Skip Neo4j cleanup | `false` |
| `--skip-qdrant` | Skip Qdrant cleanup | `false` |
| `--skip-redis` | Skip Redis cleanup | `false` |
| `--report-dir PATH` | Report output directory | `reports/cleanup/` |
| `--quiet` | Minimal console output | `false` |
| `--help` | Show help message | - |

## Output Example

### Console Output

```
============================================================
DATABASE CLEANUP - SURGICAL DATA DELETION
============================================================
Timestamp: 2025-10-18T19:31:59.150715

Neo4j Data Cleanup
------------------------------------------------------------
  Before: 715 nodes, 81 relationships
  Schema: 13 constraints, 35 indexes
✅ Data deleted
  After: 0 nodes, 0 relationships
✅ Schema preserved: 13 constraints, 35 indexes

Qdrant Vector Cleanup
------------------------------------------------------------
  Collection: weka_sections
  Before: 560 vectors
✅ Deleted 560 vectors

Redis Test Database Cleanup
------------------------------------------------------------
  Database: db=1
  Before: 34 keys
✅ Deleted 34 keys
  After: 0 keys

============================================================
CLEANUP COMPLETE
============================================================
✅ Report saved: reports/cleanup/cleanup-report-20251018-193159.json
✅ All data deleted successfully
✅ All schemas preserved
✅ System ready for fresh test run
```

### Report File Structure

Reports are saved to `reports/cleanup/cleanup-report-YYYYMMDD-HHMMSS.json`:

```json
{
  "timestamp": "2025-10-18T19:31:59.150715",
  "dry_run": false,
  "options": {
    "dry_run": false,
    "redis_db": 1,
    "skip_neo4j": false,
    "skip_qdrant": false,
    "skip_redis": false,
    "report_dir": "reports/cleanup",
    "quiet": false
  },
  "before": {
    "neo4j": {
      "nodes": 715,
      "relationships": 81,
      "constraints": 13,
      "indexes": 35
    },
    "qdrant": {
      "weka_sections": {
        "vector_count": 560,
        "config": "..."
      }
    },
    "redis": {
      "db": 1,
      "key_count": 34,
      "sample_keys": ["ingest:state:...", "..."]
    }
  },
  "after": {
    "neo4j": {
      "nodes": 0,
      "relationships": 0,
      "constraints": 13,
      "indexes": 35
    },
    "qdrant": {
      "weka_sections": {
        "vector_count": 0,
        "deleted": true
      }
    },
    "redis": {
      "db": 1,
      "key_count": 0
    }
  },
  "actions": [
    {
      "timestamp": "2025-10-18T19:32:00.123456",
      "action": "neo4j_delete_all",
      "details": {
        "nodes_deleted": 715,
        "rels_deleted": 81
      }
    },
    {
      "timestamp": "2025-10-18T19:32:01.234567",
      "action": "qdrant_delete_collection",
      "details": {
        "collection": "weka_sections",
        "vectors_deleted": 560
      }
    },
    {
      "timestamp": "2025-10-18T19:32:02.345678",
      "action": "redis_flushdb",
      "details": {
        "db": 1,
        "keys_deleted": 34
      }
    }
  ],
  "errors": [],
  "summary": {
    "dry_run": false,
    "success": true,
    "databases_cleaned": [
      {
        "database": "neo4j",
        "nodes_deleted": 715,
        "relationships_deleted": 81,
        "schema_preserved": true
      },
      {
        "database": "qdrant",
        "collections_deleted": 1,
        "total_vectors_deleted": 560
      },
      {
        "database": "redis",
        "db": 1,
        "keys_deleted": 34
      }
    ]
  }
}
```

## Common Workflows

### Before Running Full Test Suite

```bash
# Clean all databases
python scripts/cleanup-databases.py

# Run tests
pytest tests/
```

### Before Development Session

```bash
# Preview what will be cleaned
python scripts/cleanup-databases.py --dry-run

# If acceptable, clean
python scripts/cleanup-databases.py
```

### Troubleshooting Failed Tests

```bash
# Clean everything for fresh start
python scripts/cleanup-databases.py

# Re-run failing test
pytest tests/failing_test.py -v
```

### Cleaning Specific Database Only

```bash
# Clean only Neo4j (preserve vectors)
python scripts/cleanup-databases.py --skip-qdrant --skip-redis

# Clean only Qdrant (preserve graph)
python scripts/cleanup-databases.py --skip-neo4j --skip-redis

# Clean only Redis (preserve graph + vectors)
python scripts/cleanup-databases.py --skip-neo4j --skip-qdrant
```

## Safety Considerations

### What Gets Deleted

✅ **Safe to Delete:**
- All Neo4j nodes and relationships (test data)
- All Qdrant vectors (test embeddings)
- All Redis keys in specified database (test state)

❌ **Never Deleted:**
- Neo4j constraints (schema)
- Neo4j indexes (schema)
- Qdrant collection configs (auto-recreate)
- Redis keys in other databases (isolated)

### When to Run

**Safe Times:**
- Before running test suite
- After completing development session
- When experiencing data consistency issues
- Before switching between feature branches

**Avoid Running:**
- During active test execution
- During production deployments
- On production Redis database (db=0)
- Without reviewing dry-run first (if unsure)

### Redis Database Numbers

| Database | Purpose | Safe to Clean? |
|----------|---------|----------------|
| `db=0` | Production/default | ⚠️ NO (unless you know what you're doing) |
| `db=1` | Tests (default for script) | ✅ YES |
| `db=2+` | Custom use | ⚠️ Check first |

## Environment Variables

Required environment variables:

```bash
export NEO4J_PASSWORD="your_neo4j_password"
export REDIS_PASSWORD="your_redis_password"  # if Redis has auth
```

The script uses `shared.config.init_config()` to load settings from:
- `.env` file (if present)
- Environment variables
- `config.yml`

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success - cleanup completed without errors |
| `1` | Failure - one or more cleanup steps failed |

## Troubleshooting

### "Import error" when running script

**Solution:** Run from project root:
```bash
cd /path/to/wekadocs-matrix
python scripts/cleanup-databases.py
```

### "Failed to load config"

**Solution:** Ensure environment variables are set:
```bash
export NEO4J_PASSWORD="testpassword123"
export REDIS_PASSWORD="testredis123"
python scripts/cleanup-databases.py
```

### "Schema changed!" error

**Issue:** Constraints or indexes were accidentally deleted

**Solution:**
1. Check Neo4j logs for errors
2. Re-run schema initialization: `python src/shared/schema.py`
3. Report issue if reproducible

### No collections found in Qdrant

**Status:** Normal - collections are created on first use

**Action:** None needed - tests will recreate collections

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Clean databases before tests
  run: |
    python scripts/cleanup-databases.py --quiet
  env:
    NEO4J_PASSWORD: ${{ secrets.NEO4J_PASSWORD }}
    REDIS_PASSWORD: ${{ secrets.REDIS_PASSWORD }}

- name: Run test suite
  run: pytest tests/ -v
```

### Makefile Integration

```makefile
.PHONY: clean-db
clean-db:
	@python scripts/cleanup-databases.py

.PHONY: clean-db-dry
clean-db-dry:
	@python scripts/cleanup-databases.py --dry-run

.PHONY: test-clean
test-clean: clean-db
	@pytest tests/ -v
```

## Comparison with Manual Cleanup

| Method | Pros | Cons |
|--------|------|------|
| **This Script** | Automated, verified, logged, safe | Requires Python |
| **Cypher + Redis CLI** | Direct control | Error-prone, no verification |
| **Docker restart** | Nukes everything | Loses schema, slow |

## Related Documentation

- [Phase 6 Test Suite Report](../reports/full-suite-20251018-191436/consolidated/FULL-SUITE-REPORT.md)
- [Database Schema](../src/shared/schema.py)
- [Config Documentation](../docs/configuration.md)

## Author & Maintenance

**Created:** 2025-10-18
**Author:** Claude Code
**Maintainer:** wekadocs-matrix team

**Last Updated:** 2025-10-18
**Version:** 1.0.0

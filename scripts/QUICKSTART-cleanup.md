# Database Cleanup - Quick Reference

## TL;DR

```bash
# Clean everything (most common use case)
python scripts/cleanup-databases.py

# Preview changes first (recommended for first-time users)
python scripts/cleanup-databases.py --dry-run
```

## Common Commands

| Task | Command |
|------|---------|
| **Standard cleanup** | `python scripts/cleanup-databases.py` |
| **Preview only** | `python scripts/cleanup-databases.py --dry-run` |
| **Clean Neo4j only** | `python scripts/cleanup-databases.py --skip-qdrant --skip-redis` |
| **Clean Qdrant only** | `python scripts/cleanup-databases.py --skip-neo4j --skip-redis` |
| **Clean Redis only** | `python scripts/cleanup-databases.py --skip-neo4j --skip-qdrant` |
| **Quiet mode** | `python scripts/cleanup-databases.py --quiet` |

## What Gets Cleaned

✅ **Deleted:**
- Neo4j: All nodes and relationships
- Qdrant: All vector collections
- Redis: All keys in db=1 (test database)

✅ **Preserved:**
- Neo4j: 13 constraints + 35 indexes
- Qdrant: Collection schemas (auto-recreate)
- Redis: Keys in other databases (db=0, db=2+)

## Output Location

Reports saved to: `reports/cleanup/cleanup-report-YYYYMMDD-HHMMSS.json`

## Pre-requisites

```bash
export NEO4J_PASSWORD="testpassword123"
export REDIS_PASSWORD="testredis123"
```

## Workflow Integration

### Before Test Suite
```bash
python scripts/cleanup-databases.py
pytest tests/
```

### Before Development
```bash
python scripts/cleanup-databases.py --dry-run  # Preview
python scripts/cleanup-databases.py            # Clean
```

### Troubleshooting
```bash
python scripts/cleanup-databases.py  # Fresh start
pytest tests/failing_test.py -v     # Re-run
```

## Safety Check

**Always safe to run?** YES ✅
- Only affects test data
- Preserves all schemas
- Generates audit reports
- Supports dry-run mode

**When to be careful?**
- `--redis-db 0` (production Redis)
- During active tests
- During deployments

## Help

```bash
python scripts/cleanup-databases.py --help
```

For full documentation, see: [README-cleanup.md](./README-cleanup.md)

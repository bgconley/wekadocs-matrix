# db-cleanup Agent Instructions

## Purpose
Clean up the Neo4j, Qdrant, and Redis databases in the wekadocs-matrix project. This is typically needed during development when testing fixes, new features, or when databases need a fresh state.

## Prerequisites
Before running cleanup, verify Docker containers are running:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep weka
```

Expected containers: `weka-neo4j`, `weka-qdrant`, `weka-redis`, `weka-ingestion-worker`

## Cleanup Approaches

### Approach 1: Run from Host Machine (PREFERRED)
Use this when running from the local development environment:

```bash
NEO4J_URI=bolt://localhost:7687 \
NEO4J_PASSWORD=testpassword123 \
QDRANT_HOST=localhost \
REDIS_HOST=localhost \
python scripts/cleanup-databases.py
```

### Approach 2: Run Inside Docker Container
Use this as a fallback if Python dependencies aren't available on the host:

```bash
docker exec weka-ingestion-worker python scripts/cleanup-databases.py
```

Note: This works because the container is on `weka-net` where Docker DNS resolves service names.

## Command Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview changes without deleting |
| `--skip-neo4j` | Skip Neo4j cleanup |
| `--skip-qdrant` | Skip Qdrant cleanup |
| `--skip-redis` | Skip Redis cleanup |
| `--restore-metadata` | Restore missing SchemaVersion nodes |
| `--quiet` | Minimal console output |

## What Gets Preserved

The cleanup script is **metadata-aware** and preserves:
- `SchemaVersion` nodes (required for health checks)
- `RelationshipTypesMarker` nodes (relationship type documentation)
- `SystemMetadata` and `MigrationHistory` nodes
- Neo4j constraints and indexes
- Qdrant collection schemas (just empties the vectors)

## Example Workflow

1. **Check current state** (optional):
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 \
     "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC LIMIT 10"
   ```

2. **Dry run first**:
   ```bash
   NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
   QDRANT_HOST=localhost REDIS_HOST=localhost \
   python scripts/cleanup-databases.py --dry-run
   ```

3. **Execute cleanup**:
   ```bash
   NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 \
   QDRANT_HOST=localhost REDIS_HOST=localhost \
   python scripts/cleanup-databases.py
   ```

4. **Verify cleanup**:
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 \
     "MATCH (n) RETURN count(n) as total_nodes"
   ```

## Troubleshooting

### Connection Refused
If you get connection errors from the host:
- Verify ports are mapped: `docker port weka-neo4j`
- Check containers are healthy: `docker ps`
- Try the Docker exec approach instead

### Missing Dependencies
If `ModuleNotFoundError` on the host:
- Activate venv: `source .venv/bin/activate`
- Or use Docker exec approach

### Metadata Missing After Cleanup
Run with `--restore-metadata` to recreate SchemaVersion nodes:
```bash
python scripts/cleanup-databases.py --restore-metadata
```

## Report Location
Cleanup reports are saved to: `reports/cleanup/cleanup-report-YYYYMMDD-HHMMSS.json`

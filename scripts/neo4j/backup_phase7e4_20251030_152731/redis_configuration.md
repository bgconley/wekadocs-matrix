# Redis Configuration - Phase 7E-4 Production Ready
**Generated:** 2025-10-30 15:27:31
**Redis Version:** 7.2-alpine

## Database Allocation

| DB | Purpose | Flush Policy |
|----|---------|-------------|
| 0 | Production cache & locks | NEVER flush without backup |
| 1 | Test data | Safe to flush |
| 2 | Session/temporary data | Expires automatically |
| 3-15 | Reserved | Not in use |

## Key Patterns and TTLs

### Cache Keys (DB 0)
```
# Document cache
doc:cache:{doc_id}                  TTL: 3600s (1 hour)
doc:embeddings:{doc_id}:{chunk_id}  TTL: 86400s (24 hours)

# Query cache
query:cache:{query_hash}             TTL: 1800s (30 minutes)
query:results:{query_hash}           TTL: 1800s (30 minutes)

# Fusion cache
fusion:rrf:{query_hash}              TTL: 900s (15 minutes)
fusion:weighted:{query_hash}         TTL: 900s (15 minutes)

# Expansion cache
expansion:candidates:{chunk_id}      TTL: 3600s (1 hour)
```

### Lock Keys (DB 0)
```
# Ingestion locks (ephemeral)
lock:ingest:{doc_id}                 TTL: 300s (5 minutes)
lock:processing:{doc_id}             TTL: 600s (10 minutes)

# Write locks
lock:write:neo4j:{entity_id}        TTL: 30s
lock:write:qdrant:{collection}      TTL: 30s
```

### Queue Keys (DB 0)
```
# Ingestion queue (Redis List)
queue:ingest:pending                 No TTL (persistent)
queue:ingest:processing              TTL: 3600s (1 hour)
queue:ingest:failed                  TTL: 86400s (24 hours)

# Retry queue
queue:retry:{timestamp}              TTL: 7200s (2 hours)
```

### Metrics Keys (DB 0)
```
# Counters (persistent)
metrics:ingestion:total              No TTL
metrics:queries:total                No TTL
metrics:cache:hits                   No TTL
metrics:cache:misses                 No TTL

# Rate limiting
ratelimit:{client_ip}:{endpoint}    TTL: 60s
ratelimit:global:{endpoint}          TTL: 60s
```

## Redis Configuration Parameters

```bash
# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru

# Persistence
appendonly yes
appendfsync everysec
save 900 1    # Save after 900 sec if at least 1 key changed
save 300 10   # Save after 300 sec if at least 10 keys changed
save 60 10000 # Save after 60 sec if at least 10000 keys changed

# Performance
tcp-keepalive 300
timeout 0
databases 16

# Security
requirepass ${REDIS_PASSWORD}
protected-mode yes
```

## Backup Commands

### Full Backup
```bash
# Create point-in-time backup
docker exec weka-redis redis-cli -a $REDIS_PASSWORD --rdb /data/backup_$(date +%Y%m%d_%H%M%S).rdb BGSAVE

# Export specific database
docker exec weka-redis redis-cli -a $REDIS_PASSWORD -n 0 --rdb /data/db0_export.rdb BGSAVE
```

### Restore from Backup
```bash
# Stop Redis first
docker-compose stop redis

# Copy backup to data directory
docker cp backup.rdb weka-redis:/data/dump.rdb

# Start Redis (will auto-load dump.rdb)
docker-compose start redis
```

## Flush Commands (Use with Caution)

```bash
# Flush test database only (SAFE)
docker exec weka-redis redis-cli -a $REDIS_PASSWORD -n 1 FLUSHDB ASYNC

# Flush specific pattern (CAREFUL)
docker exec weka-redis redis-cli -a $REDIS_PASSWORD --scan --pattern "cache:*" | xargs -L 1 docker exec weka-redis redis-cli -a $REDIS_PASSWORD DEL

# Flush all databases (DANGEROUS - requires confirmation)
# docker exec weka-redis redis-cli -a $REDIS_PASSWORD FLUSHALL
```

## Health Checks

```bash
# Basic connectivity
docker exec weka-redis redis-cli -a $REDIS_PASSWORD ping

# Memory usage
docker exec weka-redis redis-cli -a $REDIS_PASSWORD INFO memory

# Key statistics
docker exec weka-redis redis-cli -a $REDIS_PASSWORD INFO keyspace

# Check replication (if configured)
docker exec weka-redis redis-cli -a $REDIS_PASSWORD INFO replication
```

## Monitoring Queries

```bash
# Top keys by memory usage
docker exec weka-redis redis-cli -a $REDIS_PASSWORD --bigkeys

# Real-time command monitoring
docker exec weka-redis redis-cli -a $REDIS_PASSWORD MONITOR

# Slow queries log
docker exec weka-redis redis-cli -a $REDIS_PASSWORD SLOWLOG GET 10

# Client connections
docker exec weka-redis redis-cli -a $REDIS_PASSWORD CLIENT LIST
```

## Integration Notes

- **Cache Invalidation:** Handled by `tools/redis_invalidation.py`
- **Epoch Management:** Managed by `tools/redis_epoch_bump.py`
- **Connection Pooling:** Max 50 connections per service
- **Retry Policy:** Exponential backoff with max 3 retries
- **Circuit Breaker:** Opens after 5 consecutive failures

## Recovery Priority

1. Restore DB 0 first (production cache)
2. Let DB 1 remain empty (test data)
3. Session data (DB 2) will auto-populate
4. Warm up cache with common queries after restore

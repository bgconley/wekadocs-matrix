# WekaDocs GraphRAG MCP - Monitoring Runbook

**Phase 5, Task 5.2 — Monitoring & Observability**
**Version:** 1.0
**Last Updated:** 2025-10-14

## Overview

This runbook provides step-by-step procedures for diagnosing and responding to alerts from the WekaDocs GraphRAG MCP monitoring system.

## Quick Reference

| Alert | Severity | Response Time | Escalation |
|-------|----------|---------------|------------|
| HighP99Latency | Critical | 10 min | Page on-call |
| HighErrorRate | Critical | 10 min | Page on-call |
| ReconciliationDriftHigh | Critical | 15 min | Page on-call |
| ServiceDown | Critical | 5 min | Page on-call |
| HybridSearchSlowP95 | Warning | 30 min | Ticket |
| LowCacheHitRate | Warning | 1 hour | Ticket |
| IngestionQueueBacklog | Warning | 30 min | Ticket |

## SLO Targets

- **P50 Latency:** < 200ms
- **P95 Latency:** < 500ms
- **P99 Latency:** < 2s
- **Availability:** 99.9% (43m downtime/month)
- **Error Rate:** < 1%
- **Cache Hit Rate:** > 80%
- **Drift:** < 0.5%

---

## Alert Response Procedures

### 1. HighP99Latency

**Symptom:** HTTP request P99 latency > 2s for 5 minutes

**Impact:** Users experiencing slow responses; SLO breach

**Diagnosis:**

1. Check Grafana "Query Performance" dashboard
   ```bash
   # Identify slow endpoints
   curl http://localhost:9090/api/v1/query?query='topk(5, histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])))'
   ```

2. Review OpenTelemetry traces in your OTLP backend (New Relic by default)
   ```bash
   # Use New Relic Distributed tracing UI for service=weka-mcp-server
   ```

3. Check for slow Cypher queries
   ```bash
   curl http://localhost:9090/api/v1/query?query='histogram_quantile(0.99, rate(cypher_query_duration_seconds_bucket[5m])) > 1'
   ```

4. Verify Neo4j connection pool
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
     "CALL dbms.listConnections() YIELD connectionId, connectTime, connector"
   ```

**Mitigation:**

- **Immediate:** Scale up workers if CPU > 80%
  ```bash
  docker-compose up -d --scale mcp-server=2
  ```

- **Short-term:** Increase cache TTL to reduce query load
  ```yaml
  # config/development.yaml
  cache:
    l2:
      ttl_seconds: 7200  # Increase from 3600
  ```

- **Long-term:** Review query optimization (Phase 4, Task 4.2)

**Resolution Criteria:** P99 < 2s for 10 consecutive minutes

---

### 2. HighErrorRate

**Symptom:** HTTP 5xx error rate > 1% for 5 minutes

**Impact:** Service degradation; potential data loss

**Diagnosis:**

1. Check error breakdown by endpoint
   ```bash
   curl http://localhost:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m])'
   ```

2. Review application logs
   ```bash
   docker logs weka-mcp-server --since 10m | grep -i error
   ```

3. Check dependency health
   ```bash
   curl http://localhost:8000/ready
   ```

4. Verify database connectivity
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "RETURN 1"
   docker exec weka-redis redis-cli ping
   curl http://localhost:6333/health
   ```

**Mitigation:**

- **Neo4j down:** Restart container
  ```bash
  docker-compose restart neo4j
  ```

- **Qdrant down:** Check disk space; restart if needed
  ```bash
  df -h
  docker-compose restart qdrant
  ```

- **Memory exhaustion:** Increase container memory limits
  ```yaml
  # docker-compose.yml
  services:
    mcp-server:
      deploy:
        resources:
          limits:
            memory: 4G  # Increase from 2G
  ```

**Resolution Criteria:** Error rate < 0.5% for 10 consecutive minutes

---

### 3. ReconciliationDriftHigh

**Symptom:** Drift between graph and vector store > 0.5%

**Impact:** Inconsistent search results; data integrity issues

**Diagnosis:**

1. Check drift percentage
   ```bash
   curl http://localhost:9090/api/v1/query?query='reconciliation_drift_percentage'
   ```

2. Identify missing vectors
   ```bash
   docker exec weka-mcp-server python3 -c "
   from src.ingestion.reconcile import check_drift
   drift = check_drift()
   print(f'Missing in vector store: {len(drift.missing_in_vector)}')
   print(f'Missing in graph: {len(drift.missing_in_graph)}')
   "
   ```

3. Check reconciliation logs
   ```bash
   docker logs weka-ingestion-worker --since 24h | grep reconciliation
   ```

**Mitigation:**

- **Immediate:** Trigger manual reconciliation
  ```bash
  docker exec weka-ingestion-worker python3 -m src.ingestion.reconcile --repair
  ```

- **Short-term:** Verify embedding service is healthy
  ```bash
  docker exec weka-mcp-server python3 -c "
  from src.shared.embeddings import get_embedder
  embedder = get_embedder()
  vec = embedder.encode('test')
  print(f'Embedding dims: {len(vec)}')
  "
  ```

- **Long-term:** Investigate root cause (failed writes, timeout errors)

**Resolution Criteria:** Drift < 0.1% after reconciliation completes

---

### 4. HybridSearchSlowP95

**Symptom:** Hybrid search P95 latency > 500ms

**Impact:** Slow user queries; SLO warning

**Diagnosis:**

1. Check vector search latency
   ```bash
   curl http://localhost:9090/api/v1/query?query='histogram_quantile(0.95, rate(vector_search_duration_seconds_bucket[5m]))'
   ```

2. Check graph expansion latency
   ```bash
   curl http://localhost:9090/api/v1/query?query='histogram_quantile(0.95, rate(graph_expansion_duration_seconds_bucket[5m]))'
   ```

3. Review slow query traces in your OTLP backend (New Relic by default)
   ```bash
   # Look for traces with operation=hybrid.search and duration > 500ms in New Relic
   ```

4. Verify Neo4j indexes
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
     "CALL db.indexes() YIELD name, state, populationPercent WHERE state <> 'ONLINE'"
   ```

**Mitigation:**

- **Vector slow:** Check Qdrant collection size and consider sharding
  ```bash
  curl http://localhost:6333/collections/weka_sections
  ```

- **Graph slow:** Add missing indexes (see Phase 4, Task 4.2)
  ```cypher
  CREATE INDEX section_document_id IF NOT EXISTS FOR (s:Section) ON (s.document_id);
  ```

- **Both slow:** Reduce hybrid search `top_k` parameter
  ```yaml
  # config/development.yaml
  search:
    hybrid:
      top_k: 10  # Reduce from 20
  ```

**Resolution Criteria:** P95 < 400ms for 10 consecutive minutes

---

### 5. LowCacheHitRate

**Symptom:** Cache hit rate < 80% for 10 minutes

**Impact:** Increased query load; higher latency

**Diagnosis:**

1. Check cache hit/miss rates by layer
   ```bash
   curl http://localhost:9090/api/v1/query?query='cache_hit_rate'
   ```

2. Verify cache size
   ```bash
   docker exec weka-redis redis-cli INFO memory
   ```

3. Check for cache churn (rapid evictions)
   ```bash
   docker exec weka-redis redis-cli INFO stats | grep evicted_keys
   ```

**Mitigation:**

- **L1 cache too small:** Increase in-process cache size
  ```yaml
  # config/development.yaml
  cache:
    l1:
      max_size: 2000  # Increase from 1000
  ```

- **L2 cache evictions:** Increase Redis maxmemory
  ```yaml
  # docker-compose.yml
  services:
    redis:
      command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
  ```

- **Query patterns changed:** Run cache warmer
  ```bash
  docker exec weka-mcp-server python3 -m src.ops.warmers.query_warmer
  ```

**Resolution Criteria:** Hit rate > 85% sustained for 30 minutes

---

### 6. IngestionQueueBacklog

**Symptom:** Ingestion queue size > 1000 items for 10 minutes

**Impact:** Delayed document updates; stale search results

**Diagnosis:**

1. Check queue size and lag
   ```bash
   curl http://localhost:9090/api/v1/query?query='ingestion_queue_size'
   curl http://localhost:9090/api/v1/query?query='ingestion_queue_lag_seconds'
   ```

2. Check ingestion worker health
   ```bash
   docker logs weka-ingestion-worker --since 10m
   ```

3. Verify Neo4j write performance
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
     "CALL dbms.queryJmx('org.neo4j:*') YIELD attributes WHERE attributes.Name = 'TransactionsCommitted'"
   ```

**Mitigation:**

- **Worker overwhelmed:** Scale up ingestion workers
  ```bash
  docker-compose up -d --scale ingestion-worker=2
  ```

- **Neo4j backpressure:** Reduce batch size
  ```yaml
  # config/development.yaml
  ingestion:
    batch_size: 250  # Reduce from 500
  ```

- **Large documents:** Enable circuit breaker to pause ingestion
  ```yaml
  # config/development.yaml
  ingestion:
    circuit_breaker:
      enabled: true
      threshold: 0.8  # Pause at 80% queue capacity
  ```

**Resolution Criteria:** Queue size < 100 and lag < 60s

---

### 7. ServiceDown

**Symptom:** Service unreachable for 2 minutes

**Impact:** Complete service outage

**Diagnosis:**

1. Check container status
   ```bash
   docker ps --filter name=weka-mcp-server
   ```

2. Review recent logs
   ```bash
   docker logs weka-mcp-server --tail 100
   ```

3. Check resource constraints
   ```bash
   docker stats --no-stream weka-mcp-server
   ```

**Mitigation:**

- **Container stopped:** Restart service
  ```bash
  docker-compose up -d mcp-server
  ```

- **OOM killed:** Increase memory limit and restart
  ```yaml
  # docker-compose.yml
  services:
    mcp-server:
      deploy:
        resources:
          limits:
            memory: 4G
  ```

- **Crash loop:** Check startup dependencies; manually verify connections
  ```bash
  curl http://localhost:8000/ready
  ```

**Resolution Criteria:** Service healthy and responding to `/health` for 5 consecutive checks

---

### 8. Neo4jDown

**Symptom:** Prometheus alert `Neo4jDown` firing for 2+ minutes. Health checks failing on :7474/7687.

**Impact:** All graph operations fail; MCP tools depending on Cypher are degraded or return errors.

**Diagnosis Steps:**

1. **Container/Pod status**
   ```bash
   docker ps | grep neo4j
   docker logs weka-neo4j --tail=200
   ```

2. **Process & ports**
   ```bash
   docker exec weka-neo4j ss -lntp | grep -E '7474|7687'
   ```

3. **Disk / memory / heap**
   ```bash
   # Check host disk
   df -h

   # Check Neo4j logs for OOM or pagecache errors
   docker exec weka-neo4j cat /logs/neo4j.log | tail -100
   docker exec weka-neo4j cat /logs/debug.log | tail -100
   ```

4. **Bolt connectivity**
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "RETURN 1"
   ```

5. **Cluster (if applicable)**
   ```bash
   docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "CALL dbms.cluster.overview()"
   ```

**Immediate Mitigation:**

- **Single node:** Restart service
  ```bash
  docker restart weka-neo4j
  ```

- **Cluster:** Restart **followers** first; avoid leader restart until quorum confirmed

- **Reduce load:** Scale MCP replicas down temporarily or enable degraded mode (graph-only / no vector)
  ```bash
  docker-compose scale mcp-server=1
  ```

**Short-Term Fix:**

- **Clear stuck transactions**
  ```bash
  docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "CALL dbms.listQueries()"
  docker exec weka-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "CALL dbms.killQuery('<query-id>')"
  ```

- **Free disk:** Remove old logs and snapshots
  ```bash
  docker exec weka-neo4j find /logs -name "*.log.*" -mtime +7 -delete
  ```

- **Verify memory settings:** Ensure `NEO4J_dbms_memory_*` envs match capacity
  ```bash
  docker exec weka-neo4j env | grep NEO4J_dbms_memory
  ```

**Long-Term Fix:**

- Increase heap/pagecache to recommended ratios; add disk
- Add liveness/readiness probes and circuit breakers to ingestion
- Schedule compaction/maintenance during off-peak

**Resolution Criteria:** Health check passes, `RETURN 1` via cypher-shell succeeds, alerts clear.

---

### 9. QdrantDown

**Symptom:** Prometheus alert `QdrantDown` firing for 2+ minutes. Health checks failing on :6333/6334.

**Impact:** Vector search fails; hybrid search degraded to graph-only mode; embedding operations fail.

**Diagnosis Steps:**

1. **Container/Pod status**
   ```bash
   docker ps | grep qdrant
   docker logs weka-qdrant --tail=200
   ```

2. **Process & ports**
   ```bash
   docker exec weka-qdrant netstat -lntp | grep -E '6333|6334'
   ```

3. **Health endpoint**
   ```bash
   curl http://localhost:6333/health
   curl http://localhost:6333/collections/weka_sections
   ```

4. **Disk space & storage**
   ```bash
   df -h
   docker exec weka-qdrant du -sh /qdrant/storage/*
   ```

5. **Memory usage**
   ```bash
   docker stats --no-stream weka-qdrant
   ```

**Immediate Mitigation:**

- **Container stopped:** Restart service
  ```bash
  docker restart weka-qdrant
  ```

- **Disk full:** Free up space or increase volume size
  ```bash
  docker volume inspect wekadocs-matrix_qdrant_data
  ```

- **Reduce load:** Enable graph-only mode temporarily
  ```yaml
  # config/development.yaml
  search:
    vector:
      enabled: false  # Fallback to graph-only
  ```

**Short-Term Fix:**

- **Collection corrupted:** Re-create collection
  ```bash
  curl -X DELETE http://localhost:6333/collections/weka_sections
  # Then trigger re-indexing via ingestion
  ```

- **Memory exhaustion:** Increase container memory limit
  ```yaml
  # docker-compose.yml
  services:
    qdrant:
      deploy:
        resources:
          limits:
            memory: 4G
  ```

**Long-Term Fix:**

- Implement collection sharding for large datasets
- Set up Qdrant cluster for high availability
- Add disk monitoring and auto-cleanup of old vectors
- Configure vector compression to reduce storage

**Resolution Criteria:** Health endpoint returns 200, collections accessible, vector search operational.

---

### 10. RedisDown

**Symptom:** Prometheus alert `RedisDown` firing for 2+ minutes. Connection failures on :6379.

**Impact:** L2 cache unavailable; rate limiting disabled; increased load on Neo4j and Qdrant.

**Diagnosis Steps:**

1. **Container/Pod status**
   ```bash
   docker ps | grep redis
   docker logs weka-redis --tail=200
   ```

2. **Process & ports**
   ```bash
   docker exec weka-redis netstat -lntp | grep 6379
   ```

3. **Redis connectivity**
   ```bash
   docker exec weka-redis redis-cli ping
   docker exec weka-redis redis-cli INFO server
   ```

4. **Memory usage & evictions**
   ```bash
   docker exec weka-redis redis-cli INFO memory
   docker exec weka-redis redis-cli INFO stats | grep evicted
   ```

5. **Persistence (if enabled)**
   ```bash
   docker exec weka-redis redis-cli INFO persistence
   docker exec weka-redis ls -lh /data/
   ```

**Immediate Mitigation:**

- **Container stopped:** Restart service
  ```bash
  docker restart weka-redis
  ```

- **Memory maxed out:** Flush old keys or increase maxmemory
  ```bash
  # Temporary: flush least-recently-used keys
  docker exec weka-redis redis-cli --scan --pattern "weka:cache:*" | head -1000 | xargs docker exec -i weka-redis redis-cli DEL
  ```

- **Graceful degradation:** Application continues with L1 cache only (in-process)

**Short-Term Fix:**

- **Increase memory limit**
  ```yaml
  # docker-compose.yml
  services:
    redis:
      command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
  ```

- **Clear stale cache entries**
  ```bash
  docker exec weka-redis redis-cli FLUSHDB
  # Note: Will cause temporary cache miss spike
  ```

- **Check for memory leaks:** Monitor key count growth
  ```bash
  docker exec weka-redis redis-cli DBSIZE
  ```

**Long-Term Fix:**

- Set up Redis Sentinel or Cluster for high availability
- Implement cache key TTL policies to prevent unbounded growth
- Add monitoring for Redis slow log
  ```bash
  docker exec weka-redis redis-cli SLOWLOG GET 10
  ```
- Consider Redis persistence (RDB/AOF) for critical cache data

**Resolution Criteria:** Redis responding to PING, application reconnected, rate limiting and L2 cache operational.

---

## Monitoring Access

- **Grafana:** http://localhost:3000 (default: admin/admin)
- **Prometheus:** http://localhost:9090
- **Tracing UI:** New Relic Distributed tracing (or your configured OTLP backend)
- **Metrics Endpoint:** http://localhost:8000/metrics

## Escalation Contacts

- **On-call Engineer:** PagerDuty integration (TBD)
- **Platform Team:** Slack #wekadocs-platform
- **Data Team:** Slack #wekadocs-data

## Post-Incident Review

After resolving a critical incident:

1. Document timeline in incident tracker
2. Update runbook with lessons learned
3. Schedule post-mortem within 48 hours
4. Implement preventive measures (new alerts, automation, etc.)

---

## Appendix: Useful Queries

### Top 10 slowest endpoints (last hour)
```promql
topk(10, histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1h])))
```

### Error rate by endpoint
```promql
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

### Cache effectiveness
```promql
sum(rate(cache_operations_total{result="hit"}[5m])) / sum(rate(cache_operations_total[5m]))
```

### Ingestion throughput
```promql
rate(ingestion_documents_total{status="success"}[5m])
```

### Active vs idle connections
```promql
connection_pool_active / (connection_pool_active + connection_pool_idle)
```

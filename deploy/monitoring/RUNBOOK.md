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

2. Review OpenTelemetry traces in Jaeger
   ```bash
   open http://localhost:16686
   # Filter by service=wekadocs-mcp, minDuration=2s
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

3. Review slow query traces in Jaeger
   ```bash
   # Look for traces with operation=hybrid.search and duration > 500ms
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

## Monitoring Access

- **Grafana:** http://localhost:3000 (default: admin/admin)
- **Prometheus:** http://localhost:9090
- **Jaeger UI:** http://localhost:16686
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

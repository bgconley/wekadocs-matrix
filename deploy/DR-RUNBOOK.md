# Disaster Recovery Runbook

## Overview
Disaster recovery procedures for WekaDocs GraphRAG MCP production system.

**RTO Target:** 1 hour (from detection to full service restoration)
**RPO Target:** 15 minutes (maximum acceptable data loss)

## Recovery Scenarios

### Scenario 1: Complete Cluster Failure

**Symptoms:** All K8s nodes unresponsive, regional outage

**Recovery Steps:**

1. **Detect & Assess** (Target: 5 min)
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```

2. **Failover to DR Cluster** (Target: 10 min)
   ```bash
   kubectl config use-context dr-cluster
   kubectl get ns wekadocs
   ```

3. **Restore from Backups** (Target: 30 min)
   ```bash
   LATEST=$(ls -t /backups | head -1)
   ./deploy/scripts/restore-all.sh /backups/$LATEST
   ```

4. **Verify Services** (Target: 10 min)
   ```bash
   kubectl get pods -n wekadocs
   curl http://mcp-server:8000/health
   ```
5. **Verify BGE-M3 Embedding Service** (Target: 5 min)
   ```bash
   export BGE_M3_API_URL=${BGE_M3_API_URL:-http://127.0.0.1:9000}
   curl $BGE_M3_API_URL/healthz
   # Optional Prometheus scrape
   curl $BGE_M3_API_URL/metrics | head -20
   ```

5. **Update DNS** (Target: 5 min)
   - Point domain to DR cluster load balancer
   - Verify propagation

**Total RTO:** 60 minutes

### Scenario 2: Data Corruption

**Symptoms:** Invalid query results, integrity violations

**Recovery Steps:**

1. **Identify Corruption Scope** (5 min)
   ```bash
   # Check Neo4j constraints
   kubectl exec -n wekadocs neo4j-0 -- \
     cypher-shell "CALL db.constraints()"
   ```

2. **Stop Ingestion** (2 min)
   ```bash
   kubectl scale deploy ingestion-worker -n wekadocs --replicas=0
   ```

3. **Restore from Point-in-Time** (20 min)
   ```bash
   # Find backup before corruption
   ./deploy/scripts/restore-all.sh /backups/<timestamp>
   ```

4. **Verify Integrity** (10 min)
   - Run reconciliation
   - Check drift metrics

5. **Resume Operations** (3 min)
   ```bash
   kubectl scale deploy ingestion-worker -n wekadocs --replicas=2
   ```

**Total RTO:** 40 minutes

## Backup Schedule

- **Hourly:** Incremental snapshots (retention: 24h)
- **Daily:** Full backups (retention: 30 days)
- **Weekly:** Archive backups (retention: 1 year)

## DR Drill Procedure

Run quarterly DR drills using `./deploy/scripts/dr-drill.sh`

**Drill validates:**
- Backup integrity
- Restore procedures
- RTO/RPO measurements
- Team readiness

## Contacts

**On-Call Rotation:** [PagerDuty rotation]
**Escalation:** Platform Lead → VP Engineering
**Vendor Support:** Neo4j Enterprise Support (24/7)

## Post-Incident Review

After any DR event:
1. Document timeline and RTO/RPO actuals
2. Identify gaps in procedures
3. Update runbook
4. Schedule retro within 48h

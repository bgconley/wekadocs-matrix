#!/usr/bin/env bash
# DR Drill Script - Validates backup/restore and measures RTO/RPO
# Usage: ./dr-drill.sh

set -euo pipefail

NAMESPACE="wekadocs"
DRILL_START=$(date +%s)
BACKUP_DIR="/tmp/dr-drill-$(date +%Y%m%d-%H%M%S)"

echo "=== Disaster Recovery Drill ==="
echo "Start time: $(date)"
echo ""

# Step 1: Create test backup
echo "[1/5] Creating test backup..."
BACKUP_START=$(date +%s)
./deploy/scripts/backup-all.sh "$BACKUP_DIR"
BACKUP_END=$(date +%s)
BACKUP_DURATION=$((BACKUP_END - BACKUP_START))
echo "✓ Backup completed in ${BACKUP_DURATION}s"
echo ""

# Step 2: Record pre-restore state
echo "[2/5] Recording pre-restore state..."
PRE_RESTORE_DOCS=$(kubectl exec -n ${NAMESPACE} neo4j-0 -- \
  cypher-shell -u neo4j -p testpassword123 \
  "MATCH (d:Document) RETURN count(d) as count" | grep -oP '\d+' | head -1)
echo "✓ Document count: ${PRE_RESTORE_DOCS}"
echo ""

# Step 3: Simulate disaster (scale down services)
echo "[3/5] Simulating disaster (scaling down services)..."
kubectl scale deployment mcp-server-blue -n ${NAMESPACE} --replicas=0
kubectl scale deployment ingestion-worker -n ${NAMESPACE} --replicas=0
sleep 5
echo "✓ Services scaled down"
echo ""

# Step 4: Restore from backup
echo "[4/5] Restoring from backup..."
RESTORE_START=$(date +%s)
./deploy/scripts/restore-all.sh "$BACKUP_DIR"
RESTORE_END=$(date +%s)
RESTORE_DURATION=$((RESTORE_END - RESTORE_START))
echo "✓ Restore completed in ${RESTORE_DURATION}s"
echo ""

# Step 5: Verify restoration and scale up
echo "[5/5] Verifying restoration..."
kubectl scale deployment mcp-server-blue -n ${NAMESPACE} --replicas=3
kubectl scale deployment ingestion-worker -n ${NAMESPACE} --replicas=2
kubectl rollout status deployment/mcp-server-blue -n ${NAMESPACE} --timeout=5m

# Wait for health
sleep 10
HEALTH_CHECK=$(kubectl run -n ${NAMESPACE} --rm -i --restart=Never curl-test --image=curlimages/curl:latest -- \
  curl -s http://mcp-server:8000/health | grep -c "healthy" || echo "0")

if [[ "$HEALTH_CHECK" != "1" ]]; then
  echo "✗ Health check failed"
  exit 1
fi

POST_RESTORE_DOCS=$(kubectl exec -n ${NAMESPACE} neo4j-0 -- \
  cypher-shell -u neo4j -p testpassword123 \
  "MATCH (d:Document) RETURN count(d) as count" | grep -oP '\d+' | head -1)

DRILL_END=$(date +%s)
TOTAL_DURATION=$((DRILL_END - DRILL_START))

# Cleanup drill backup
rm -rf "$BACKUP_DIR"

echo ""
echo "=== DR Drill Results ==="
echo "Backup duration:  ${BACKUP_DURATION}s"
echo "Restore duration: ${RESTORE_DURATION}s"
echo "Total RTO:        ${TOTAL_DURATION}s (target: 3600s / 1h)"
echo "Data integrity:   ${PRE_RESTORE_DOCS} -> ${POST_RESTORE_DOCS} documents"
echo ""

if [[ $TOTAL_DURATION -lt 3600 ]]; then
  echo "✓ RTO Target MET (${TOTAL_DURATION}s < 3600s)"
else
  echo "✗ RTO Target MISSED (${TOTAL_DURATION}s > 3600s)"
fi

# Generate drill report
REPORT_FILE="reports/dr-drill-$(date +%Y%m%d-%H%M%S).json"
mkdir -p reports
cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "metrics": {
    "backup_duration_seconds": ${BACKUP_DURATION},
    "restore_duration_seconds": ${RESTORE_DURATION},
    "total_rto_seconds": ${TOTAL_DURATION},
    "rto_target_seconds": 3600,
    "rto_met": $([ $TOTAL_DURATION -lt 3600 ] && echo "true" || echo "false")
  },
  "data_integrity": {
    "pre_restore_documents": ${PRE_RESTORE_DOCS},
    "post_restore_documents": ${POST_RESTORE_DOCS},
    "data_loss": $((PRE_RESTORE_DOCS - POST_RESTORE_DOCS))
  }
}
EOF

echo ""
echo "✓ Drill report: $REPORT_FILE"

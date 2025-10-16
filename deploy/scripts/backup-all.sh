#!/usr/bin/env bash
# Backup all stateful services (Neo4j, Qdrant, Redis)
# Usage: ./backup-all.sh [backup-location]

set -euo pipefail

NAMESPACE="wekadocs"
BACKUP_DIR="${1:-/backups/$(date +%Y%m%d-%H%M%S)}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "=== WekaDocs Backup ==="
echo "Timestamp: $TIMESTAMP"
echo "Backup location: $BACKUP_DIR"
echo ""

mkdir -p "$BACKUP_DIR"

# Backup Neo4j
echo "Backing up Neo4j..."
NEO4J_POD=$(kubectl get pods -n ${NAMESPACE} -l app=neo4j -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${NEO4J_POD} -- \
  cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-testpassword123}" \
  "CALL apoc.export.cypher.all('/tmp/neo4j-backup-${TIMESTAMP}.cypher', {})" || true
kubectl cp ${NAMESPACE}/${NEO4J_POD}:/tmp/neo4j-backup-${TIMESTAMP}.cypher \
  ${BACKUP_DIR}/neo4j-backup-${TIMESTAMP}.cypher
echo "✓ Neo4j backup complete"

# Backup Qdrant
echo "Backing up Qdrant..."
QDRANT_POD=$(kubectl get pods -n ${NAMESPACE} -l app=qdrant -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${QDRANT_POD} -- \
  curl -X POST "http://localhost:6333/collections/weka_sections/snapshots" 2>/dev/null || true
sleep 5
SNAPSHOT_NAME=$(kubectl exec -n ${NAMESPACE} ${QDRANT_POD} -- \
  ls -t /qdrant/storage/snapshots/ 2>/dev/null | head -1)
kubectl cp ${NAMESPACE}/${QDRANT_POD}:/qdrant/storage/snapshots/${SNAPSHOT_NAME} \
  ${BACKUP_DIR}/qdrant-snapshot-${TIMESTAMP}.tar || true
echo "✓ Qdrant backup complete"

# Backup Redis
echo "Backing up Redis..."
REDIS_POD=$(kubectl get pods -n ${NAMESPACE} -l app=redis -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${REDIS_POD} -- redis-cli BGSAVE
sleep 5
kubectl cp ${NAMESPACE}/${REDIS_POD}:/data/dump.rdb \
  ${BACKUP_DIR}/redis-dump-${TIMESTAMP}.rdb
echo "✓ Redis backup complete"

# Create manifest
cat > ${BACKUP_DIR}/manifest.json <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "files": {
    "neo4j": "neo4j-backup-${TIMESTAMP}.cypher",
    "qdrant": "qdrant-snapshot-${TIMESTAMP}.tar",
    "redis": "redis-dump-${TIMESTAMP}.rdb"
  }
}
EOF

echo ""
echo "=== Backup Complete ==="
echo "Location: $BACKUP_DIR"
ls -lh "$BACKUP_DIR"

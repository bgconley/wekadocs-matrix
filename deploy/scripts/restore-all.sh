#!/usr/bin/env bash
# Restore all stateful services from backup
# Usage: ./restore-all.sh <backup-location>

set -euo pipefail

NAMESPACE="wekadocs"
BACKUP_DIR="${1:-}"

if [[ -z "$BACKUP_DIR" ]] || [[ ! -d "$BACKUP_DIR" ]]; then
  echo "Usage: $0 <backup-location>"
  echo "Error: Backup directory not found"
  exit 1
fi

echo "=== WekaDocs Restore ==="
echo "Backup location: $BACKUP_DIR"
echo ""

# Read manifest
if [[ ! -f "${BACKUP_DIR}/manifest.json" ]]; then
  echo "Error: manifest.json not found in backup directory"
  exit 1
fi

NEO4J_FILE=$(jq -r '.files.neo4j' < ${BACKUP_DIR}/manifest.json)
QDRANT_FILE=$(jq -r '.files.qdrant' < ${BACKUP_DIR}/manifest.json)
REDIS_FILE=$(jq -r '.files.redis' < ${BACKUP_DIR}/manifest.json)

echo "Manifest loaded:"
echo "  Neo4j: $NEO4J_FILE"
echo "  Qdrant: $QDRANT_FILE"
echo "  Redis: $REDIS_FILE"
echo ""

# Restore Neo4j
echo "Restoring Neo4j..."
NEO4J_POD=$(kubectl get pods -n ${NAMESPACE} -l app=neo4j -o jsonpath='{.items[0].metadata.name}')
kubectl cp ${BACKUP_DIR}/${NEO4J_FILE} ${NAMESPACE}/${NEO4J_POD}:/tmp/restore.cypher
kubectl exec -n ${NAMESPACE} ${NEO4J_POD} -- \
  cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-testpassword123}" \
  -f /tmp/restore.cypher
echo "✓ Neo4j restore complete"

# Restore Qdrant
echo "Restoring Qdrant..."
QDRANT_POD=$(kubectl get pods -n ${NAMESPACE} -l app=qdrant -o jsonpath='{.items[0].metadata.name}')
kubectl cp ${BACKUP_DIR}/${QDRANT_FILE} ${NAMESPACE}/${QDRANT_POD}:/tmp/snapshot.tar
kubectl exec -n ${NAMESPACE} ${QDRANT_POD} -- \
  curl -X PUT "http://localhost:6333/collections/weka_sections/snapshots/upload" \
  --data-binary @/tmp/snapshot.tar
echo "✓ Qdrant restore complete"

# Restore Redis
echo "Restoring Redis..."
REDIS_POD=$(kubectl get pods -n ${NAMESPACE} -l app=redis -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${REDIS_POD} -- redis-cli SHUTDOWN NOSAVE || true
sleep 5
kubectl cp ${BACKUP_DIR}/${REDIS_FILE} ${NAMESPACE}/${REDIS_POD}:/data/dump.rdb
kubectl delete pod -n ${NAMESPACE} ${REDIS_POD}
echo "✓ Redis restore complete (pod restarting)"

echo ""
echo "=== Restore Complete ==="

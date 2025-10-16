#!/usr/bin/env bash
# Blue/Green deployment traffic switch script
# Usage: ./blue-green-switch.sh <blue|green>

set -euo pipefail

TARGET_VERSION="${1:-}"
NAMESPACE="wekadocs"
SERVICE_NAME="mcp-server"

if [[ -z "$TARGET_VERSION" ]]; then
  echo "Usage: $0 <blue|green>"
  exit 1
fi

if [[ "$TARGET_VERSION" != "blue" && "$TARGET_VERSION" != "green" ]]; then
  echo "Error: Version must be 'blue' or 'green'"
  exit 1
fi

echo "=== Blue/Green Deployment Switch ==="
echo "Target version: $TARGET_VERSION"
echo "Namespace: $NAMESPACE"
echo ""

# Check if target deployment is ready
echo "Checking if ${TARGET_VERSION} deployment is ready..."
READY_REPLICAS=$(kubectl get deployment mcp-server-${TARGET_VERSION} -n ${NAMESPACE} \
  -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
DESIRED_REPLICAS=$(kubectl get deployment mcp-server-${TARGET_VERSION} -n ${NAMESPACE} \
  -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

if [[ "$READY_REPLICAS" != "$DESIRED_REPLICAS" ]] || [[ "$READY_REPLICAS" == "0" ]]; then
  echo "Error: ${TARGET_VERSION} deployment is not ready"
  echo "Ready replicas: ${READY_REPLICAS}/${DESIRED_REPLICAS}"
  exit 1
fi

echo "✓ ${TARGET_VERSION} deployment is ready (${READY_REPLICAS}/${DESIRED_REPLICAS} replicas)"
echo ""

# Perform health check on target version
echo "Performing health checks on ${TARGET_VERSION} pods..."
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=mcp-server,version=${TARGET_VERSION} \
  -o jsonpath='{.items[0].metadata.name}')

if ! kubectl exec -n ${NAMESPACE} ${POD_NAME} -- curl -sf http://localhost:8000/health > /dev/null; then
  echo "Error: Health check failed for ${TARGET_VERSION}"
  exit 1
fi

echo "✓ Health check passed"
echo ""

# Switch service selector to target version
echo "Switching traffic to ${TARGET_VERSION}..."
kubectl patch service ${SERVICE_NAME} -n ${NAMESPACE} -p \
  "{\"spec\":{\"selector\":{\"app\":\"mcp-server\",\"version\":\"${TARGET_VERSION}\"}}}"

echo "✓ Service selector updated"
echo ""

# Verify switch
CURRENT_VERSION=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} \
  -o jsonpath='{.spec.selector.version}')

if [[ "$CURRENT_VERSION" == "$TARGET_VERSION" ]]; then
  echo "✓ Traffic switch successful!"
  echo "Current version: $CURRENT_VERSION"
else
  echo "✗ Traffic switch failed!"
  echo "Expected: $TARGET_VERSION, Current: $CURRENT_VERSION"
  exit 1
fi

echo ""
echo "=== Switch Complete ==="

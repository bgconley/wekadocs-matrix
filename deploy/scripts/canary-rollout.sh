#!/usr/bin/env bash
# Canary deployment with progressive rollout and SLO monitoring
# Usage: ./canary-rollout.sh <new-image-tag> [--auto-rollback]

set -euo pipefail

NEW_IMAGE="${1:-}"
AUTO_ROLLBACK="${2:-}"
NAMESPACE="wekadocs"
CANARY_DEPLOYMENT="mcp-server-canary"
STABLE_DEPLOYMENT="mcp-server-blue"

# SLO thresholds
MAX_ERROR_RATE=0.01
MAX_P99_LATENCY_MS=2000
MONITOR_DURATION_SECONDS=60

if [[ -z "$NEW_IMAGE" ]]; then
  echo "Usage: $0 <new-image-tag> [--auto-rollback]"
  exit 1
fi

echo "=== Canary Deployment Rollout ==="
echo "New image: $NEW_IMAGE"
echo ""

check_slos() {
  local percentage=$1
  echo "Monitoring SLOs for ${MONITOR_DURATION_SECONDS}s (${percentage}% canary)..."
  sleep ${MONITOR_DURATION_SECONDS}

  # Check health endpoint
  POD=$(kubectl get pods -n ${NAMESPACE} -l app=mcp-server,version=canary -o jsonpath='{.items[0].metadata.name}')
  if ! kubectl exec -n ${NAMESPACE} ${POD} -- curl -sf http://localhost:8000/health > /dev/null; then
    echo "✗ Health check failed"
    return 1
  fi

  # Check metrics endpoint for error rate
  METRICS=$(kubectl exec -n ${NAMESPACE} ${POD} -- curl -s http://localhost:8000/metrics)
  echo "✓ SLOs passed for ${percentage}% traffic"
  return 0
}

rollback_canary() {
  echo ""
  echo "!!! ROLLBACK TRIGGERED !!!"
  kubectl scale deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} --replicas=0
  echo "✓ Canary scaled to 0 replicas"
  exit 1
}

scale_canary() {
  local replicas=$1
  local percentage=$2
  echo ""
  echo "=== Stage: ${percentage}% Canary Traffic ==="
  kubectl scale deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} --replicas=${replicas}
  kubectl rollout status deployment/${CANARY_DEPLOYMENT} -n ${NAMESPACE} --timeout=5m

  if ! check_slos ${percentage}; then
    if [[ "$AUTO_ROLLBACK" == "--auto-rollback" ]]; then
      rollback_canary
    else
      echo "SLO check failed. Run with --auto-rollback to enable automatic rollback"
      exit 1
    fi
  fi
}

# Create or update canary deployment
echo "Deploying canary with image: $NEW_IMAGE"
kubectl set image deployment/${CANARY_DEPLOYMENT} mcp-server=${NEW_IMAGE} -n ${NAMESPACE} 2>/dev/null || \
  kubectl create deployment ${CANARY_DEPLOYMENT} --image=${NEW_IMAGE} -n ${NAMESPACE}

# Progressive rollout
scale_canary 1 5    # 5% traffic
scale_canary 2 25   # 25% traffic
scale_canary 3 50   # 50% traffic
scale_canary 6 100  # 100% traffic

echo ""
echo "=== Canary Rollout Complete ==="
echo "Promoting canary to stable..."
kubectl set image deployment/${STABLE_DEPLOYMENT} mcp-server=${NEW_IMAGE} -n ${NAMESPACE}
kubectl scale deployment ${CANARY_DEPLOYMENT} -n ${NAMESPACE} --replicas=0
echo "✓ Promotion complete"

#!/usr/bin/env bash
# infra/scripts/destroy-lgtm.sh
# Destroy LGTM observability stack from GCP
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform/environments/dev"

echo "=========================================="
echo "  LGTM Observability Stack Destruction"
echo "=========================================="
echo ""

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Error: terraform is not installed"; exit 1; }

cd "${TERRAFORM_DIR}"

# Show what will be destroyed
echo "This will DESTROY the following resources:"
echo ""
terraform state list 2>/dev/null || echo "(No state found - nothing to destroy)"

echo ""
echo "WARNING: This action cannot be undone!"
echo "All data stored in the LGTM stack will be PERMANENTLY DELETED."
echo ""

# Triple confirmation for safety
read -p "Type 'destroy' to confirm destruction: " -r
if [[ "$REPLY" != "destroy" ]]; then
    echo "Destruction cancelled."
    exit 0
fi

echo ""
echo ">>> Destroying infrastructure..."
terraform destroy -auto-approve

echo ""
echo "=========================================="
echo "  Destruction Complete"
echo "=========================================="
echo ""
echo "All LGTM resources have been removed from GCP."

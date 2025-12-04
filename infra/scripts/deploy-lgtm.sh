#!/usr/bin/env bash
# infra/scripts/deploy-lgtm.sh
# Deploy LGTM observability stack to GCP
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform/environments/dev"

echo "=========================================="
echo "  LGTM Observability Stack Deployment"
echo "=========================================="
echo ""

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Error: terraform is not installed"; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo "Error: gcloud CLI is not installed"; exit 1; }

# Check for tfvars file
if [[ ! -f "${TERRAFORM_DIR}/terraform.tfvars" ]]; then
    echo "Error: terraform.tfvars not found!"
    echo "Please copy terraform.tfvars.example to terraform.tfvars and configure it."
    exit 1
fi

# Confirm deployment
echo "This will deploy the LGTM stack to GCP."
echo "Estimated cost: ~\$126/month"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

cd "${TERRAFORM_DIR}"

# Initialize Terraform
echo ""
echo ">>> Initializing Terraform..."
terraform init -upgrade

# Plan deployment
echo ""
echo ">>> Planning deployment..."
terraform plan -out=tfplan

# Show plan summary
echo ""
echo ">>> Plan created. Review above output carefully."
read -p "Apply this plan? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    rm -f tfplan
    exit 0
fi

# Apply deployment
echo ""
echo ">>> Applying Terraform plan..."
terraform apply tfplan
rm -f tfplan

# Show outputs
echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
terraform output

echo ""
echo "Next steps:"
echo "1. Wait 2-3 minutes for cloud-init to complete"
echo "2. Test connectivity via Tailscale:"
echo "   curl \$(terraform output -raw loki_url)/ready"
echo "   curl \$(terraform output -raw grafana_url)/api/health"
echo "3. Access Grafana at: \$(terraform output -raw grafana_url)"
echo ""

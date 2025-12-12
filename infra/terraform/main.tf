# infra/terraform/main.tf
# Root Terraform configuration for LGTM Observability Stack
# This file is typically not used directly - use environments/dev instead

terraform {
  required_version = ">= 1.5.0"
}

# Provider configuration should be done in environment-specific files

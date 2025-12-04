# infra/terraform/environments/dev/backend.tf
# Backend configuration for Terraform state storage

# Option 1: Local backend (for development only)
# Uncomment this for local state storage:
# terraform {
#   backend "local" {
#     path = "terraform.tfstate"
#   }
# }

# Option 2: GCS backend (recommended for team/production use)
# Uncomment and configure for GCS state storage:
# terraform {
#   backend "gcs" {
#     bucket = "your-terraform-state-bucket"
#     prefix = "lgtm/dev"
#   }
# }

# Default: No backend configured (will use local state)
# To initialize with a specific backend, run:
#   terraform init -backend-config="bucket=your-bucket"

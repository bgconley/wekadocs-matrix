# infra/terraform/environments/dev/main.tf
# Development environment LGTM Stack deployment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# LGTM VM Module
module "lgtm" {
  source = "../../modules/lgtm-vm"

  instance_name          = var.instance_name
  machine_type           = var.machine_type
  zone                   = var.zone
  network                = var.network
  subnetwork             = var.subnetwork
  gcp_subnet_cidr        = var.gcp_subnet_cidr
  disk_size_gb           = var.disk_size_gb
  data_disk_size_gb      = var.data_disk_size_gb
  loki_retention_days    = var.loki_retention_days
  tempo_retention_hours  = var.tempo_retention_hours
  grafana_admin_password = var.grafana_admin_password
  service_account_email  = var.service_account_email
  environment            = "dev"
}

# Variables for dev environment
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "instance_name" {
  description = "Name of the LGTM instance"
  type        = string
  default     = "lgtm-observability-dev"
}

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "e2-standard-4"
}

variable "network" {
  description = "VPC network name"
  type        = string
}

variable "subnetwork" {
  description = "Subnetwork for the instance"
  type        = string
}

variable "gcp_subnet_cidr" {
  description = "GCP subnet CIDR for firewall rules"
  type        = string
  default     = "10.10.0.0/22"
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 20
}

variable "data_disk_size_gb" {
  description = "Data disk size in GB"
  type        = number
  default     = 100
}

variable "loki_retention_days" {
  description = "Loki log retention in days"
  type        = number
  default     = 7
}

variable "tempo_retention_hours" {
  description = "Tempo trace retention in hours"
  type        = number
  default     = 72
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

variable "service_account_email" {
  description = "Service account email for the VM"
  type        = string
}

# Outputs
output "instance_name" {
  description = "Name of the LGTM instance"
  value       = module.lgtm.instance_name
}

output "instance_internal_ip" {
  description = "Internal IP address of the LGTM instance"
  value       = module.lgtm.instance_internal_ip
}

output "grafana_url" {
  description = "Grafana URL (via Tailscale)"
  value       = module.lgtm.grafana_url
}

output "loki_url" {
  description = "Loki URL for log ingestion"
  value       = module.lgtm.loki_url
}

output "tempo_otlp_grpc_url" {
  description = "Tempo OTLP gRPC endpoint"
  value       = module.lgtm.tempo_otlp_grpc_url
}

output "mimir_url" {
  description = "Mimir URL for metrics"
  value       = module.lgtm.mimir_url
}

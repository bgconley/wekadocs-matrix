# infra/terraform/modules/lgtm-vm/variables.tf
# Variable definitions for LGTM VM module

variable "instance_name" {
  description = "Name of the GCE instance"
  type        = string
  default     = "lgtm-observability"
}

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "e2-standard-4"  # 4 vCPU, 16 GB RAM
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
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

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

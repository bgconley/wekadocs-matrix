# infra/terraform/modules/lgtm-vm/main.tf
# LGTM Observability Stack VM Module

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    cloudinit = {
      source  = "hashicorp/cloudinit"
      version = "~> 2.3"
    }
  }
}

# Data source for Ubuntu 22.04 LTS image
# Switched from COS because COS doesn't have Docker Compose v2 pre-installed
# and its read-only filesystem makes installation complex
data "google_compute_image" "ubuntu" {
  family  = "ubuntu-2204-lts"
  project = "ubuntu-os-cloud"
}

# Cloud-init configuration
data "cloudinit_config" "lgtm" {
  gzip          = false
  base64_encode = false

  part {
    content_type = "text/cloud-config"
    content      = templatefile("${path.module}/cloud-init/cloud-config.yaml.tpl", {
      instance_name          = var.instance_name
      docker_compose_content = base64encode(templatefile("${path.module}/cloud-init/docker-compose.yaml.tpl", {
        loki_retention_days    = var.loki_retention_days
        tempo_retention_hours  = var.tempo_retention_hours
        grafana_admin_password = var.grafana_admin_password
      }))
      loki_config_content       = base64encode(file("${path.module}/cloud-init/loki-config.yaml"))
      tempo_config_content      = base64encode(file("${path.module}/cloud-init/tempo-config.yaml"))
      mimir_config_content      = base64encode(file("${path.module}/cloud-init/mimir-config.yaml"))
      grafana_datasources_content = base64encode(file("${path.module}/cloud-init/grafana-datasources.yaml"))
    })
  }
}

# Compute instance
resource "google_compute_instance" "lgtm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["lgtm-stack", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu.self_link
      size  = var.disk_size_gb
      type  = "pd-ssd"
    }
  }

  # Data disk attached at boot time (available for cloud-init)
  attached_disk {
    source      = google_compute_disk.lgtm_data.self_link
    device_name = "${var.instance_name}-data"
    mode        = "READ_WRITE"
  }

  network_interface {
    subnetwork = var.subnetwork

    # Internal IP only - accessed via Tailscale
    # No access_config = no external IP
  }

  metadata = {
    user-data = data.cloudinit_config.lgtm.rendered
  }

  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }

  labels = {
    environment = var.environment
    purpose     = "observability"
    stack       = "lgtm"
  }

  allow_stopping_for_update = true
}

# Persistent disk for data
resource "google_compute_disk" "lgtm_data" {
  name = "${var.instance_name}-data"
  type = "pd-ssd"
  zone = var.zone
  size = var.data_disk_size_gb

  labels = {
    environment = var.environment
    purpose     = "lgtm-storage"
  }
}

# Note: Disk is now attached via attached_disk block in the instance
# (removed google_compute_attached_disk resource - it caused race condition with cloud-init)

# Firewall rule for LGTM stack ingress from Tailscale
resource "google_compute_firewall" "lgtm_ingress" {
  name    = "${var.instance_name}-allow-tailscale"
  network = var.network

  allow {
    protocol = "tcp"
    ports    = ["3000", "3100", "3200", "4317", "4318", "9009", "9090"]
  }

  # Allow from Tailscale subnet (100.64.0.0/10) and local GCP subnet
  source_ranges = ["100.64.0.0/10", var.gcp_subnet_cidr]
  target_tags   = ["lgtm-stack"]

  description = "Allow LGTM observability traffic from Tailscale mesh"
}

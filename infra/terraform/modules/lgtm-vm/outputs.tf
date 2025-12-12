# infra/terraform/modules/lgtm-vm/outputs.tf
# Output values for LGTM VM module

output "instance_name" {
  description = "Name of the LGTM instance"
  value       = google_compute_instance.lgtm.name
}

output "instance_internal_ip" {
  description = "Internal IP address of the LGTM instance"
  value       = google_compute_instance.lgtm.network_interface[0].network_ip
}

output "instance_zone" {
  description = "Zone of the LGTM instance"
  value       = google_compute_instance.lgtm.zone
}

output "instance_self_link" {
  description = "Self-link of the LGTM instance"
  value       = google_compute_instance.lgtm.self_link
}

output "data_disk_name" {
  description = "Name of the data disk"
  value       = google_compute_disk.lgtm_data.name
}

output "grafana_url" {
  description = "Grafana URL (via Tailscale)"
  value       = "http://${google_compute_instance.lgtm.network_interface[0].network_ip}:3000"
}

output "loki_url" {
  description = "Loki URL for log ingestion"
  value       = "http://${google_compute_instance.lgtm.network_interface[0].network_ip}:3100"
}

output "tempo_otlp_grpc_url" {
  description = "Tempo OTLP gRPC endpoint"
  value       = "${google_compute_instance.lgtm.network_interface[0].network_ip}:4317"
}

output "tempo_otlp_http_url" {
  description = "Tempo OTLP HTTP endpoint"
  value       = "http://${google_compute_instance.lgtm.network_interface[0].network_ip}:4318"
}

output "mimir_url" {
  description = "Mimir URL for metrics"
  value       = "http://${google_compute_instance.lgtm.network_interface[0].network_ip}:9009"
}

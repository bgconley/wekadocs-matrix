# Docker Compose for LGTM Stack on GCP
# Deployed via cloud-init on Ubuntu 22.04 LTS
version: "3.8"

services:
  loki:
    image: grafana/loki:2.9.3
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - /etc/lgtm/loki-config.yaml:/etc/loki/config.yaml:ro
      - /mnt/lgtm-data/loki:/loki
    command: -config.file=/etc/loki/config.yaml
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:3100/ready || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  tempo:
    image: grafana/tempo:2.3.1
    container_name: tempo
    ports:
      - "3200:3200"   # Tempo API
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    volumes:
      - /etc/lgtm/tempo-config.yaml:/etc/tempo/config.yaml:ro
      - /mnt/lgtm-data/tempo:/var/tempo
    command: -config.file=/etc/tempo/config.yaml
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:3200/ready || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  mimir:
    image: grafana/mimir:2.11.0
    container_name: mimir
    ports:
      - "9009:9009"
    volumes:
      - /etc/lgtm/mimir-config.yaml:/etc/mimir/config.yaml:ro
      - /mnt/lgtm-data/mimir:/data
    command: -config.file=/etc/mimir/config.yaml
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:9009/ready || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  grafana:
    image: grafana/grafana:10.2.3
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${grafana_admin_password}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_AUTH_ANONYMOUS_ENABLED=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    volumes:
      - /mnt/lgtm-data/grafana:/var/lib/grafana
      - /etc/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - loki
      - tempo
      - mimir
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  default:
    name: lgtm-network

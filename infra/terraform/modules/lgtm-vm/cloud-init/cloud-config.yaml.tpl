#cloud-config
# Cloud-init configuration for LGTM Observability Stack
# Ubuntu 22.04 LTS

# Install Docker and Docker Compose via apt
package_update: true
package_upgrade: false

packages:
  - apt-transport-https
  - ca-certificates
  - curl
  - gnupg
  - lsb-release

write_files:
  - path: /etc/docker-compose/docker-compose.yaml
    permissions: '0644'
    encoding: b64
    content: ${docker_compose_content}

  - path: /etc/lgtm/loki-config.yaml
    permissions: '0644'
    encoding: b64
    content: ${loki_config_content}

  - path: /etc/lgtm/tempo-config.yaml
    permissions: '0644'
    encoding: b64
    content: ${tempo_config_content}

  - path: /etc/lgtm/mimir-config.yaml
    permissions: '0644'
    encoding: b64
    content: ${mimir_config_content}

  - path: /etc/grafana/provisioning/datasources/datasources.yaml
    permissions: '0644'
    encoding: b64
    content: ${grafana_datasources_content}

  - path: /etc/systemd/system/lgtm-stack.service
    permissions: '0644'
    content: |
      [Unit]
      Description=LGTM Observability Stack
      After=docker.service
      Requires=docker.service

      [Service]
      Type=oneshot
      RemainAfterExit=yes
      WorkingDirectory=/etc/docker-compose
      ExecStart=/usr/bin/docker compose up -d
      ExecStop=/usr/bin/docker compose down
      Restart=on-failure

      [Install]
      WantedBy=multi-user.target

runcmd:
  # Install Docker using official method
  - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  - echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
  - apt-get update
  - apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

  # Start Docker
  - systemctl enable docker
  - systemctl start docker

  # Create mount point for data disk
  - mkdir -p /mnt/lgtm-data

  # Wait for attached disk to appear (up to 60 seconds)
  - |
    for i in $(seq 1 60); do
      if [ -e /dev/disk/by-id/google-${instance_name}-data ]; then
        echo "Disk found after $i seconds"
        break
      fi
      echo "Waiting for disk... ($i/60)"
      sleep 1
    done

  # Format and mount data disk (only if not already formatted)
  - |
    if ! blkid /dev/disk/by-id/google-${instance_name}-data; then
      mkfs.ext4 -F /dev/disk/by-id/google-${instance_name}-data
    fi
  - mount /dev/disk/by-id/google-${instance_name}-data /mnt/lgtm-data
  - echo '/dev/disk/by-id/google-${instance_name}-data /mnt/lgtm-data ext4 defaults 0 2' >> /etc/fstab

  # Create data directories on the mounted disk
  - mkdir -p /mnt/lgtm-data/{loki,tempo,mimir,grafana}
  # Loki runs as user 10001:10001, Mimir runs as 10001:10001
  # Grafana runs as 472:0, Tempo runs as 10001:10001
  - chown -R 10001:10001 /mnt/lgtm-data/loki
  - chown -R 10001:10001 /mnt/lgtm-data/tempo
  - chown -R 10001:10001 /mnt/lgtm-data/mimir
  - chown -R 472:0 /mnt/lgtm-data/grafana
  - chmod -R 755 /mnt/lgtm-data

  # Enable and start the LGTM stack
  - systemctl daemon-reload
  - systemctl enable lgtm-stack.service
  - systemctl start lgtm-stack.service

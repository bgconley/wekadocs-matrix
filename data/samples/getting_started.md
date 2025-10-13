# Getting Started with WekaFS

WekaFS is a high-performance parallel file system designed for AI and HPC workloads.

## Installation

To install WekaFS, follow these steps:

1. Download the installation package from the Weka website
2. Extract the package: `tar -xzf weka-installer.tar.gz`
3. Run the installation script: `./install-weka.sh`
4. Verify the installation: `weka version`

## Configuration

### Basic Configuration

The main configuration file is located at `/etc/weka/weka.conf`. Key parameters include:

- `cluster_name`: Name of your WekaFS cluster
- `network_interface`: Network interface to use (e.g., eth0)
- `storage_devices`: List of storage devices to use

Example configuration:

```yaml
cluster_name: production-cluster
network_interface: eth0
storage_devices:
  - /dev/nvme0n1
  - /dev/nvme1n1
```

### Advanced Settings

For production deployments, configure these additional parameters:

- `MAX_MEMORY=32G`: Maximum memory for Weka processes
- `CACHE_SIZE=100G`: Size of the SSD cache
- `TIER_POLICY=hot_cold`: Data tiering policy

## Commands Reference

### Cluster Management

Use `weka cluster` commands to manage your cluster:

```bash
weka cluster info
weka cluster status
weka cluster create --name=prod-cluster
```

### Mount Operations

Mount a filesystem with:

```bash
weka mount /mnt/weka --filesystem=default
```

Unmount with:

```bash
weka umount /mnt/weka
```

## Troubleshooting Common Errors

### Error E001: Connection Timeout

If you see error E001, check your network connectivity:

1. Verify network interfaces are up
2. Check firewall rules
3. Ensure Weka services are running with `systemctl status weka`

### Error E002: Insufficient Resources

This error indicates resource constraints. To resolve:

1. Check available memory: `free -h`
2. Review storage capacity: `df -h`
3. Adjust resource limits in weka.conf

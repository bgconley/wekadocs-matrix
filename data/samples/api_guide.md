# WekaFS API Guide

This guide covers the WekaFS REST API and command-line interface.

## REST API Overview

The WekaFS REST API is accessible at `https://your-cluster:14000/api/v1/`.

### Authentication

All API requests require authentication using an API token:

```bash
curl -H "Authorization: Bearer $WEKA_TOKEN" https://cluster:14000/api/v1/status
```

### Common Endpoints

#### Cluster Status

```bash
GET /api/v1/cluster/status
```

Returns cluster health and statistics.

#### File System Operations

Create a filesystem:

```bash
POST /api/v1/filesystems
{
  "name": "fs1",
  "capacity": "10TB",
  "type": "standard"
}
```

#### Data Tiers

Configure tiering with the `tier` endpoint:

```bash
PUT /api/v1/filesystems/fs1/tier
{
  "tier_policy": "hot_cold",
  "cold_threshold_days": 30
}
```

## CLI Reference

The `weka` command provides comprehensive cluster management.

### System Commands

- `weka status`: Show system status
- `weka version`: Display version information
- `weka debug collect-logs`: Collect diagnostic logs

### Configuration Commands

Set configuration values:

```bash
weka config set memory.max 64G
weka config set network.mtu 9000
weka config apply
```

View configuration:

```bash
weka config show
```

## Python SDK

Install the Python SDK:

```bash
pip install weka-sdk
```

Example usage:

```python
from weka import WekaCluster

cluster = WekaCluster(host="cluster.example.com", token=WEKA_TOKEN)
status = cluster.get_status()
print(f"Cluster: {status.name}, Nodes: {status.node_count}")
```

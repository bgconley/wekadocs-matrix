# Performance Tuning Guide

Learn how to optimize WekaFS for maximum performance.

## Network Optimization

### MTU Configuration

Set the correct MTU size for your network:

```bash
weka config set network.mtu 9000
ip link set eth0 mtu 9000
```

### RDMA Setup

For InfiniBand networks, enable RDMA:

1. Install RDMA drivers
2. Configure `/etc/weka/rdma.conf`
3. Restart Weka services: `systemctl restart weka`

## Storage Configuration

### SSD Tier Optimization

Configure SSD caching for optimal performance:

- Use NVMe drives for best results
- Set cache size to 20-30% of active dataset
- Enable write-back caching

Example:

```yaml
cache:
  size: 200G
  policy: write-back
  devices:
    - /dev/nvme0n1
    - /dev/nvme1n1
```

### Stripe Width

Adjust stripe width based on workload:

- Sequential I/O: Larger stripes (1M-4M)
- Random I/O: Smaller stripes (64K-256K)

```bash
weka filesystem create fs1 --stripe-width=1M --capacity=10TB
```

## Memory Tuning

Set memory limits appropriately:

- Minimum: 8GB per server
- Recommended: 32GB per server
- Maximum: 256GB per server

Configure in `/etc/weka/weka.conf`:

```ini
[memory]
max_size = 64G
page_cache = 32G
metadata_cache = 8G
```

## Monitoring Performance

### Key Metrics

Monitor these metrics for optimal performance:

- **IOPS**: Target > 1M IOPS
- **Throughput**: Target > 100 GB/s
- **Latency**: Target < 100us

### Performance Commands

Check real-time performance:

```bash
weka stats
weka stats --interval=1 --count=60
weka performance report
```

## Troubleshooting Performance Issues

### Slow Write Performance

If writes are slow:

1. Check network bandwidth: `iperf3 -c server`
2. Verify SSD health: `smartctl -a /dev/nvme0n1`
3. Review cache hit rate: `weka stats cache`

### High Latency

To reduce latency:

1. Tune TCP/IP stack
2. Disable CPU frequency scaling
3. Use dedicated networks for Weka traffic

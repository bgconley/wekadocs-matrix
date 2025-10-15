# External Systems Integration Runbook

**Phase 5, Task 5.1 - Connector Operations Guide**

## Overview

The connector subsystem enables real-time documentation ingestion from external sources (GitHub, Notion, Confluence). It provides:

- **Polling**: Scheduled checks for changes
- **Webhooks**: Real-time event processing
- **Queue**: Redis-backed ingestion queue with backpressure handling
- **Circuit Breaker**: Automatic failure protection
- **Monitoring**: Stats and health endpoints

---

## Architecture

```
External System (GitHub/Notion)
    ↓ (polling/webhook)
Connector (with Circuit Breaker)
    ↓
Ingestion Queue (Redis)
    ↓
Ingestion Worker
    ↓
Neo4j + Qdrant
```

**Components:**
- `BaseConnector`: Abstract connector with polling & webhook support
- `CircuitBreaker`: Protects against cascading failures
- `IngestionQueue`: Redis-backed event queue
- `ConnectorManager`: Coordinates all connectors
- `webhooks.py`: FastAPI webhook endpoints

---

## Configuration

### GitHub Connector

```yaml
connectors:
  github:
    enabled: true
    poll_interval_seconds: 300  # 5 minutes
    batch_size: 50
    max_retries: 3
    backoff_base_seconds: 2.0
    circuit_breaker_enabled: true
    circuit_breaker_failure_threshold: 5
    circuit_breaker_timeout_seconds: 60
    webhook_secret: "${GITHUB_WEBHOOK_SECRET}"
    metadata:
      owner: "your-org"
      repo: "your-repo"
      docs_path: "docs"
```

### Environment Variables

```bash
# Required for GitHub API
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"

# Required for webhook verification
export GITHUB_WEBHOOK_SECRET="your-secret-key"

# Redis connection
export REDIS_PASSWORD="your-redis-password"
```

---

## Operations

### Starting Connectors

Connectors start automatically with the MCP server if configured. To manually start:

```python
from src.connectors import ConnectorManager

manager = ConnectorManager(redis_client, config)
manager.register_connector("github", "github", github_config)
await manager.start_polling()
```

### Monitoring

#### Health Check

```bash
curl http://localhost:8000/webhooks/health
```

Response:
```json
{
  "status": "healthy",
  "connectors": [
    {
      "name": "github",
      "status": "idle",
      "stats": {
        "events_received": 142,
        "events_queued": 140,
        "errors": 2,
        "last_sync": "2025-10-14T04:30:00Z"
      },
      "circuit_breaker": "closed"
    }
  ]
}
```

#### Queue Stats

```python
stats = manager.get_queue_stats()
# {
#   "queue_name": "ingestion:queue",
#   "size": 45,
#   "max_size": 10000,
#   "usage_pct": 0.45,
#   "backpressure": false
# }
```

### Webhook Setup

#### GitHub Webhook Configuration

1. Go to repository Settings → Webhooks → Add webhook
2. **Payload URL**: `https://your-domain.com/webhooks/github`
3. **Content type**: `application/json`
4. **Secret**: Your `GITHUB_WEBHOOK_SECRET`
5. **Events**: Select `push` events
6. Save

#### Testing Webhooks Locally

Use ngrok or similar:
```bash
ngrok http 8000
# Use https://xxx.ngrok.io/webhooks/github as webhook URL
```

---

## Troubleshooting

### Circuit Breaker Opened

**Symptom**: Connector status shows `degraded`, circuit breaker `open`

**Diagnosis**:
```python
stats = connector.circuit_breaker.get_stats()
# Check: failure_count, last_failure_time
```

**Resolution**:
1. Check logs for repeated errors
2. Verify API credentials (`GITHUB_TOKEN`)
3. Check rate limits
4. Wait for timeout (default 60s) for automatic retry
5. Manual reset (testing only):
   ```python
   connector.circuit_breaker.reset()
   ```

### Queue Backpressure

**Symptom**: `is_backpressure() == True`, polling paused

**Diagnosis**:
```python
stats = queue.get_stats()
# Check: size, usage_pct
```

**Resolution**:
1. Check ingestion worker is running
2. Scale ingestion workers if needed
3. Check Neo4j/Qdrant performance
4. Temporary: increase `max_queue_size` in config

### High Error Rate

**Symptom**: `stats["errors"]` increasing rapidly

**Common Causes**:
1. **Rate Limiting**: GitHub API has rate limits (5000/hour authenticated)
   - Solution: Increase `poll_interval_seconds`
2. **Invalid Token**: Token expired or lacks permissions
   - Solution: Regenerate token with `repo` scope
3. **Network Issues**: Timeouts, DNS failures
   - Solution: Check network, retry logic should handle transient errors
4. **Malformed Webhooks**: Invalid payloads
   - Solution: Check webhook secret, verify payload format

### Webhook Signature Failures

**Symptom**: 401 errors on `/webhooks/github`

**Resolution**:
1. Verify `GITHUB_WEBHOOK_SECRET` matches GitHub configuration
2. Check logs for signature comparison details
3. Ensure webhook sends `X-Hub-Signature-256` header
4. Test signature manually:
   ```python
   import hmac, hashlib
   secret = b"your-secret"
   payload = b'{"test": "data"}'
   sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
   # Should match GitHub's signature (without 'sha256=' prefix)
   ```

---

## Degraded Mode Operations

When external systems are unavailable, the system operates in degraded mode:

1. **Circuit Breaker Open**: Polling paused, queue still processes backlog
2. **Queue Full**: New events rejected, existing events processed
3. **Redis Down**: Webhooks fail gracefully, polling continues (events lost)

**Recovery**: System automatically recovers when conditions improve.

---

## Maintenance

### Scaling Ingestion

If queue consistently has backpressure:

1. **Horizontal**: Add more ingestion worker pods
2. **Vertical**: Increase worker resources (CPU/memory)
3. **Batch Size**: Tune `batch_size` in connector config
4. **Polling**: Adjust `poll_interval_seconds` to control load

### Rate Limit Management

GitHub API limits:
- Authenticated: 5000 requests/hour
- Unauthenticated: 60 requests/hour

**Best Practices**:
- Always use `GITHUB_TOKEN`
- Set `poll_interval_seconds` to avoid limit (300s = 5 min recommended)
- Monitor rate limit headers in logs
- Implement exponential backoff (already built-in)

### Queue Maintenance

Clear old events (testing only):
```python
queue.clear()  # Deletes all events
```

Monitor queue age:
```bash
redis-cli LLEN ingestion:queue  # Check size
redis-cli LRANGE ingestion:queue 0 0 # Peek first event
```

---

## Alerts

Recommended alerts:

1. **Circuit Breaker Open**
   - Alert if open > 5 minutes
   - Page: Yes (indicates persistent failure)

2. **Queue Backpressure**
   - Alert if usage > 90% for > 10 minutes
   - Page: No (warning only)

3. **High Error Rate**
   - Alert if error rate > 10% over 5 minutes
   - Page: Yes

4. **Webhook Failures**
   - Alert if webhook failures > 50% over 1 hour
   - Page: No (check secret configuration)

---

## Testing

Run connector tests:
```bash
# Unit + integration tests
pytest tests/p5_t1_test.py -v

# Requires: Redis running, GITHUB_TOKEN set (optional)
```

Simulate webhook:
```bash
curl -X POST http://localhost:8000/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature-256: sha256=abc123..." \
  -H "X-GitHub-Event: push" \
  -d @tests/fixtures/github_push_webhook.json
```

---

## References

- [GitHub Webhooks Documentation](https://docs.github.com/webhooks)
- [GitHub API Rate Limits](https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting)
- Circuit Breaker Pattern: `src/connectors/circuit_breaker.py`
- Implementation Plan: `/docs/implementation-plan.md` (Phase 5, Task 5.1)

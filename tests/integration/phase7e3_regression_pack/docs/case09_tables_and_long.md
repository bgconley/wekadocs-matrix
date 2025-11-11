# Configuration Matrix

DocTag: REGPACK-09

## Parameters

| Name | Default | Description |
|------|---------|-------------|
| threads | 4 | Worker threads |
| timeout | 30s | Request timeout |

### Step 1: Set Threads

ORDER_STEP_ONE
- Increase to 8 for high throughput.

### Step 2: Tune Timeout

ORDER_STEP_TWO
- Increase to 60s for long-running calls.

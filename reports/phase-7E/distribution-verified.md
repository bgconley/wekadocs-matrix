# Phase 7E.0 - Baseline Distribution Analysis

**Generated:** 2025-10-28 00:32:50

**Purpose:** Establish baseline metrics before chunking implementation

---

## Overall Statistics

- **Total Sections:** 371
- **Total Tokens:** 48,091
- **Average Tokens/Section:** 129.6
- **Median Tokens/Section:** 74.0
- **Min:** 9
- **Max:** 297

### Percentile Distribution

| Percentile | Token Count |
|------------|-------------|
| p50 | 74 |
| p75 | 262 |
| p90 | 262 |
| p95 | 262 |
| p99 | 296 |

### Token Range Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| < 200 | 260 | 70.1% |
| 200-800 | 111 | 29.9% |
| 800-1,500 | 0 | 0.0% |
| 1,500-7,900 | 0 | 0.0% |
| > 7,900 | 0 | 0.0% |

### Analysis

⚠️ **CRITICAL:** 70.1% of sections are under 200 tokens (severe fragmentation)

⚠️ **HIGH:** 100.0% of sections are under 800 tokens (needs combining)

⚠️ **TARGET:** Only 0.0% in optimal range (800-1,500 tokens)

## Per-Document Statistics

| Document | Sections | Total Tokens | Avg | p50 | p90 | p95 |
|----------|----------|--------------|-----|-----|-----|-----|
| ad685d7ad69fe8a20fbbdc34021987 | 262 | 40,194 | 153 | 74 | 262 | 262 |
| bddeb4f860b99dbaf37a0816789275 | 35 | 1,574 | 45 | 45 | 72 | 79 |
| cdf3dff253f62cf2dd1a3e62dbf440 | 34 | 3,031 | 89 | 88 | 161 | 178 |
| 4ba0ac8c8244f08f4de285bf33431f | 24 | 2,836 | 118 | 120 | 175 | 184 |
| 30a32ba39d73d63cf5af318b5d1ebc | 16 | 456 | 28 | 29 | 42 | 50 |

## H2 Grouping Analysis

- **Total H2 Groups:** 10
- **Avg Tokens/Group:** 4809.1
- **Avg Sections/Group:** 37.1
- **Min Tokens:** 23
- **Max Tokens:** 26,223

---

*This baseline will be compared with post-chunking metrics in Phase 2*
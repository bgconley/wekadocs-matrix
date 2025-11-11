# Golden Set Queries - Phase 7a Baseline

**Status:** Frozen as of 2025-10-20
**Purpose:** Baseline query set for measuring completeness, latency, and retrieval quality across verbosity modes

## Overview

This document defines 20 representative queries that cover common Weka documentation use cases. These queries serve as the **golden set** for all Phase 7-9 performance testing and quality measurement.

## Query Categories

- **Installation & Setup** (Queries 1-4)
- **Configuration & Management** (Queries 5-8)
- **Operations & Troubleshooting** (Queries 9-12)
- **Performance & Optimization** (Queries 13-16)
- **Advanced Features** (Queries 17-20)

---

## Golden Set Queries

### Installation & Setup (1-4)

#### Query 1: Basic Installation
**Query:** `How do I install Weka on Ubuntu?`

**Expected Coverage:**
- Installation prerequisites
- Package installation steps
- Initial configuration
- Verification steps

**Target Sections:** 3-5 sections from installation documentation

---

#### Query 2: Prerequisites and Compatibility
**Query:** `What are the hardware requirements for Weka?`

**Expected Coverage:**
- Minimum CPU/RAM/disk requirements
- Network requirements
- Supported operating systems
- Compatibility matrix

**Target Sections:** 2-4 sections from planning/compatibility docs

---

#### Query 3: Cluster Setup
**Query:** `How do I set up a Weka cluster?`

**Expected Coverage:**
- Cluster architecture overview
- Node preparation steps
- Cluster formation commands
- Initial validation

**Target Sections:** 4-6 sections from cluster setup guide

---

#### Query 4: License Configuration
**Query:** `How do I configure Weka licensing?`

**Expected Coverage:**
- License types
- License installation procedure
- License verification
- Troubleshooting license issues

**Target Sections:** 2-3 sections from licensing documentation

---

### Configuration & Management (5-8)

#### Query 5: Filesystem Creation
**Query:** `How do I create a filesystem in Weka?`

**Expected Coverage:**
- Filesystem creation command
- Configuration parameters
- Best practices for sizing
- Verification steps

**Target Sections:** 3-5 sections from filesystem management

---

#### Query 6: User Management
**Query:** `How do I manage users and permissions in Weka?`

**Expected Coverage:**
- User creation
- Role-based access control
- Permission management
- Authentication configuration

**Target Sections:** 3-4 sections from security/user management

---

#### Query 7: Network Configuration
**Query:** `How do I configure networking for Weka?`

**Expected Coverage:**
- Network interface configuration
- VLAN setup
- Network performance tuning
- Network troubleshooting

**Target Sections:** 4-6 sections from networking documentation

---

#### Query 8: Snapshot Management
**Query:** `How do I create and manage snapshots?`

**Expected Coverage:**
- Snapshot creation commands
- Snapshot scheduling
- Snapshot restoration
- Snapshot deletion and cleanup

**Target Sections:** 3-5 sections from data protection

---

### Operations & Troubleshooting (9-12)

#### Query 9: Monitoring and Alerts
**Query:** `How do I monitor Weka cluster health?`

**Expected Coverage:**
- Monitoring tools and dashboards
- Key metrics to watch
- Alert configuration
- Health check procedures

**Target Sections:** 4-6 sections from monitoring/operations

---

#### Query 10: Drive Failure Handling
**Query:** `What do I do if a drive fails?`

**Expected Coverage:**
- Drive failure detection
- Replacement procedure
- Rebuild process
- Data integrity verification

**Target Sections:** 3-4 sections from troubleshooting/operations

---

#### Query 11: Performance Diagnostics
**Query:** `How do I diagnose performance issues?`

**Expected Coverage:**
- Performance monitoring tools
- Common bottlenecks
- Diagnostic commands
- Performance tuning recommendations

**Target Sections:** 5-7 sections from troubleshooting/performance

---

#### Query 12: Log Analysis
**Query:** `How do I collect and analyze Weka logs?`

**Expected Coverage:**
- Log locations and types
- Log collection procedures
- Common error patterns
- Log analysis tools

**Target Sections:** 3-4 sections from troubleshooting/logging

---

### Performance & Optimization (13-16)

#### Query 13: SSD Optimization
**Query:** `How do I optimize SSD performance in Weka?`

**Expected Coverage:**
- SSD tiering configuration
- Cache optimization
- Write amplification reduction
- SSD health monitoring

**Target Sections:** 4-5 sections from performance optimization

---

#### Query 14: Network Performance Tuning
**Query:** `How do I tune network performance?`

**Expected Coverage:**
- Network parameters
- Jumbo frames configuration
- TCP/RDMA tuning
- Network performance testing

**Target Sections:** 3-5 sections from network optimization

---

#### Query 15: IO Performance Analysis
**Query:** `How do I analyze IO performance?`

**Expected Coverage:**
- IO statistics collection
- Performance metrics interpretation
- Bottleneck identification
- Optimization recommendations

**Target Sections:** 4-6 sections from performance analysis

---

#### Query 16: Capacity Planning
**Query:** `How do I plan capacity for growth?`

**Expected Coverage:**
- Capacity calculation methods
- Growth projections
- Scaling strategies
- Resource planning

**Target Sections:** 3-4 sections from planning/capacity

---

### Advanced Features (17-20)

#### Query 17: Tiering Configuration
**Query:** `How do I configure tiering to object storage?`

**Expected Coverage:**
- Tiering policy configuration
- Object storage backends
- Data migration process
- Performance considerations

**Target Sections:** 5-7 sections from tiering documentation

---

#### Query 18: Disaster Recovery
**Query:** `How do I set up disaster recovery?`

**Expected Coverage:**
- DR architecture
- Replication configuration
- Failover procedures
- Recovery testing

**Target Sections:** 5-7 sections from DR/replication

---

#### Query 19: API Integration
**Query:** `How do I use the Weka REST API?`

**Expected Coverage:**
- API authentication
- Common API endpoints
- API usage examples
- Rate limiting and best practices

**Target Sections:** 4-6 sections from API documentation

---

#### Query 20: Upgrade Procedures
**Query:** `How do I upgrade Weka to a new version?`

**Expected Coverage:**
- Pre-upgrade preparation
- Upgrade procedure
- Version compatibility
- Rollback procedures

**Target Sections:** 4-6 sections from upgrade/maintenance

---

## Success Criteria

### Completeness Metrics

For each query, measure:
- **Section Coverage:** Percentage of expected sections retrieved
- **Information Completeness:** Percentage of required information present in response
- **Relevance Score:** Average relevance of retrieved sections (0.0-1.0)

### Performance Targets (Phase 7a)

| Mode | P50 Latency | P95 Latency | Completeness Target |
|------|-------------|-------------|---------------------|
| `snippet` | <150ms | <200ms | ≥30% (baseline) |
| `full` | <250ms | <350ms | ≥60% improvement |
| `graph` | <350ms | <450ms | ≥80% improvement |

### Quality Targets

- **Precision@3:** ≥0.80 (top 3 results highly relevant)
- **NDCG@10:** ≥0.75 (overall ranking quality)
- **Provenance:** 100% of assertions traceable to section IDs
- **Zero Hallucinations:** All facts must be grounded in retrieved sections

---

## Testing Protocol

### Baseline Testing (Day 1)

1. Run all 20 queries in `snippet` mode
2. Collect P50/P95 latency metrics
3. Measure completeness (human evaluation on sample)
4. Document baseline in `/reports/phase-7/baseline.csv`

### Verbosity Mode Testing (Day 2)

1. Run all 20 queries × 3 modes (60 total queries)
2. Compare completeness across modes
3. Verify P95 latency targets
4. Document results in `/reports/phase-7/verbosity-comparison.csv`

### Graph Mode Testing (Day 2-4)

1. Run all 20 queries in `graph` mode
2. Analyze graph expansion patterns
3. Measure node/edge counts
4. Verify provenance chains
5. Document in `/reports/phase-7/graph-analysis.md`

---

## Maintenance

This golden set is **frozen** for Phase 7-9. No queries should be added, removed, or modified without:

1. Written justification
2. Approval from technical lead
3. Re-baselining all metrics
4. Version increment in this document

**Current Version:** 1.0
**Last Updated:** 2025-10-20
**Next Review:** After Phase 9 completion

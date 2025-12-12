---
title: "Disaster Recovery"
doc_id: "test-dr-001"
---

## Overview

Disaster recovery ensures business continuity when primary systems fail.
WEKA supports multiple DR strategies using snapshots and replication.

## DR Strategies

1. **Snapshot-based DR**: Regular snapshots replicated to remote site
2. **Continuous Replication**: Real-time data mirroring
3. **Hybrid Approach**: Combination of both methods

## Implementation

For snapshot-based DR, see: [Filesystem Snapshots](filesystem-snapshots.md)

Configure snapshot policies for automated protection: [Snapshot Policies](snapshot-policies.md)

## Related Topics

Related: Tiering Configuration - understand how tiered data is handled in DR

See also: Backup Procedures for comprehensive data protection

## See Also

- [Tiering Configuration](tiering-configuration.md)
- Object Store Integration

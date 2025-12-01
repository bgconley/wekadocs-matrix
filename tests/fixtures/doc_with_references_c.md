---
title: "Tiering Configuration"
doc_id: "test-tiering-001"
---

## Overview

WEKA tiering allows automatic data movement between hot and cold storage tiers.
This reduces costs while maintaining performance for frequently accessed data.

## Configuration Steps

Use the following command to configure tiering:

```bash
weka fs tier create --name archive-tier --backend s3://bucket
```

## Related Topics

Related: Snapshot Policies - snapshots interact with tiered data

Related: Filesystem Snapshots - understand how snapshots work with tiers

## See Also

- [Snapshot Policies](snapshot-policies.md)
- [Filesystem Snapshots](filesystem-snapshots.md)
- Object Store Integration

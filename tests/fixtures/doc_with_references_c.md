---
title: "Tiering Configuration"
doc_id: "test-tiering-001"
---

## Overview

Nutanix Objects lifecycle policies support data movement between active and object storage tiers.
This reduces costs while maintaining performance for frequently accessed data.

## Configuration Steps

Use the following command to configure tiering:

```bash
ncli objects lifecycle-policy create name=archive-tier target=s3://bucket
```

## Related Topics

Related: Snapshot Policies - snapshots interact with tiered data

Related: Filesystem Snapshots - understand how snapshots work with tiers

## See Also

- [Snapshot Policies](snapshot-policies.md)
- [Filesystem Snapshots](filesystem-snapshots.md)
- Object Store Integration

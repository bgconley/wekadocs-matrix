---
title: "Filesystem Snapshots"
doc_id: "test-snapshots-001"
---

## Overview

WEKA filesystems support point-in-time snapshots for data protection.
Snapshots capture the state of your filesystem at a specific moment.

## Creating Snapshots

Use the `weka fs snapshot create` command to create a snapshot:

```bash
weka fs snapshot create --filesystem my-fs --name daily-backup
```

This creates a snapshot named "daily-backup" of the filesystem.

## Snapshot Policies

For automated snapshot creation, see also: [Snapshot Policies](snapshot-policies.md)

You can configure policies to automatically create and retain snapshots based on schedules.
This reduces manual intervention and ensures consistent backup coverage.

## Related Topics

Related: Tiering Configuration - snapshots interact with tiered data in important ways

Related: Backup Procedures - snapshots are part of a comprehensive backup strategy

## Cross-References

For detailed policy configuration, refer to the Snapshot Policies Guide.

See also: Object Store Integration for tiered snapshot management.

## See Also

- Object Store Integration
- Data Protection Best Practices
- [Disaster Recovery](disaster-recovery.md)

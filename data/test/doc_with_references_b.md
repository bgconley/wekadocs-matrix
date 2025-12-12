---
title: "Snapshot Policies"
doc_id: "test-policies-001"
---

## Overview

Snapshot policies automate the creation and lifecycle management of filesystem snapshots.
They provide hands-off data protection with configurable retention.

## Configuring Policies

Create a policy with the following command:

```bash
weka fs snapshot policy create --name hourly --schedule "0 * * * *" --retain 24
```

This creates an hourly snapshot policy that retains the last 24 snapshots.

## Policy Types

- **Hourly**: Create snapshots every hour, retain last 24
- **Daily**: Create daily snapshots, retain last 7
- **Weekly**: Create weekly snapshots, retain last 4

## Manual Snapshots

See: [Filesystem Snapshots](filesystem-snapshots.md) for manual snapshot creation.

For one-off snapshots outside the policy schedule, use manual creation.
This is useful during maintenance windows or before major changes.

## Integration with Tiering

When snapshots are created, tiered data is handled specially.
Refer to the Tiering Guide for details on snapshot behavior with tiered filesystems.

Related: Tiering Configuration - understand how tiered data interacts with snapshots

## Advanced Configuration

For complex multi-tier setups, see also: [Advanced Tiering](advanced-tiering.md)

You can configure snapshot policies that account for data locality and tier placement.

## See Also

- [Filesystem Snapshots](filesystem-snapshots.md)
- Tiering Configuration
- Retention Management

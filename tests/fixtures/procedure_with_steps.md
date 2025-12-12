---
title: "Test Procedure Document"
doc_id: "test-procedure-001"
---

## Procedure: Configure Tiering

This procedure explains how to configure tiering for your WEKA filesystem.
Follow these steps:

1. Enable tiering on the filesystem using `weka fs tier enable --filesystem my-fs`. Verify the command completed successfully.

2. Configure the tier policy with an age threshold using `weka fs tier policy set --filesystem my-fs --age 30d`. This moves data older than 30 days to the object store tier.

3. Check that tiering is properly configured using `weka fs tier status --filesystem my-fs`. The output should show tiering enabled with your policy settings.

4. Monitor tiering operations using `weka fs tier progress --filesystem my-fs` to track progress.

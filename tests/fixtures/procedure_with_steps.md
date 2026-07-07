---
title: "Test Procedure Document"
doc_id: "test-procedure-001"
---

## Procedure: Configure Tiering

This procedure explains how to configure object lifecycle tiering for Nutanix Files and Nutanix Objects.
Follow these steps:

1. Enable object lifecycle tiering for the share using `ncli files lifecycle enable share=my-share`. Verify the command completed successfully.

2. Configure the lifecycle policy with an age threshold using `ncli objects lifecycle-policy set share=my-share age=30d`. This moves data older than 30 days to the object storage tier.

3. Check that tiering is properly configured using `ncli files lifecycle status share=my-share`. The output should show lifecycle tiering enabled with your policy settings.

4. Monitor tiering operations using `ncli files lifecycle progress share=my-share` to track progress.

# Open Questions

1. Where is the authoritative Nutanix corpus? I did not find one in the current tree.
2. Should the target product domain be all Nutanix documentation, or a subset such as Nutanix Unified Storage, AOS/AHV, NDB, Files, or Cloud Platform?
3. Is Docker compose the production path, or is STDIO MCP still used by a client in production?
4. Which model gateway at `10.25.0.50:8080` is authoritative today, and what model IDs does `/v1/models` actually serve?
5. Should WEKA remain supported as a legacy domain profile, or should this repo be fully converted to Nutanix and archived under a new name?
6. Are the current dirty worktree changes intentional cleanup work that should become the baseline before refactor?
7. Are deleted `reports/retrieval_diagnostics/2026-03-04` and `2026-03-05` artifacts intentional?
8. Should fail-open retrieval behavior remain acceptable for production if degradation is visible, or should critical channels fail closed for Nutanix launch?
9. Is cross-document linking required for Nutanix v1, or can it be deferred after basic retrieval/citation quality is proven?
10. What is the minimum install-from-scratch proof expected before declaring the repo converted?

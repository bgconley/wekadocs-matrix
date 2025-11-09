
# Phase 7E-3 Production Docs Pack

This pack ingests **two real production Markdown documents** and runs a regression-style
check to ensure your 7E doc-tag scoping and stitched-citation rendering behave on real content.

## Contents

```
prod_docs_pack/
  docs/
    prod01_s3_information_lifecycle_management.md
    prod02_attaching_detaching_object_stores.md
  tests/test_phase7e3_prod_docs_pack.py
  runner/prod_docs_runner.py
  artifacts/                   # JSON report will be written here
```

## How to run

From your repository root (where `src/` lives), copy `prod_docs_pack/` into your tree
(e.g., under `tests/integration/`), then:

```bash
# Ensure services and API keys are set like your other tests
export JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi

# Optional: control pack behavior
export PRODPACK_VERBOSITY=graph          # or full
export PRODPACK_TOP_K=5
export PRODPACK_EMBED_VERSION=jina-embeddings-v3
export PRODPACK_CLEAN=1                  # purge prior prod-pack docs from Neo4j before run (default 1)

# Option A: pytest
pytest -q prod_docs_pack/tests/test_phase7e3_prod_docs_pack.py

# Option B: helper script
python prod_docs_pack/runner/prod_docs_runner.py
```

## What the tests assert

- Ingestion succeeds for both docs (ingest via `ingest_document(..., format="markdown")`).
- Query includes a **doc tag** (`PRODPACK-01`/`PRODPACK-02`) derived from the test's `src_uri`, ensuring doc scoping.
- Stitched Markdown answer includes **sequentially numbered citations** `[1]`, `[2]`, …
- At least **two citations** are present.
- A soft relevance check on the **first citation**:
  - PRODPACK-01: contains one of `lifecycle / management / policy / s3`
  - PRODPACK-02: contains one of `attach / attaching / detach / detaching / object / store / filesystem`
- A machine-readable JSON report is written to `artifacts/phase7e3_prod_docs_report.json`.

## Notes

- Source URIs use `tests://prodpack/PRODPACK-XX/...` so your `doc_tag` extraction finds and
  enforces the tag during retrieval.
- The tests are intentionally **resilient** to small heading wording changes in production docs—
  we verify structure and relevance rather than exact strings.

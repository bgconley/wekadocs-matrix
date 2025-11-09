
# Phase 7E-3 Production Docs Pack (No Filename Prefixes)

**No filenames were changed.** This pack preserves your original Markdown filenames and only
uses a test-only tag in the **source URI** and **query** (`PRODPACK-01/02`) to scope retrieval
during the test. Your app does **not** rely on filename prefixes; tags are not derived from names.

## Layout

```
prod_docs_pack_noprefix/
  docs/
    additional-protocols_s3_s3-information-lifecycle-management_s3-information-lifecycle-management.md
    weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md
  tests/test_phase7e3_prod_docs_pack.py
  runner/prod_docs_runner.py
  artifacts/
```

## Run

```bash
# From your repo root (where src/ lives), after copying this folder into your tree:
export JINA_API_KEY=jina_35169a1e714a41aab7b4c37817b58910Z65UGWJRVNStkMbt12lxaWrmIsVi

# Optional knobs
export PRODPACK_VERBOSITY=graph
export PRODPACK_TOP_K=5
export PRODPACK_EMBEDDING_VERSION=jina-embeddings-v3
export PRODPACK_CLEAN=1

# Option A
pytest -q prod_docs_pack_noprefix/tests/test_phase7e3_prod_docs_pack.py

# Option B
python prod_docs_pack_noprefix/runner/prod_docs_runner.py
```

## Why a tag at all?

For test runs, we include a doc tag in the **source URI** (e.g., `tests://prodpack/PRODPACK-01/additional-protocols_s3_s3-information-lifecycle-management_s3-information-lifecycle-management.md`) and in the **query** so the pipeline
scopes retrieval to the intended document and avoids cross-document interference. In **production**, you do not need to supply a tag—your filters
apply **only** when the tag is present, so normal queries behave as usual.

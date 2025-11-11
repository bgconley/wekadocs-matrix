# Phase 7E-3 Regression Pack

This package contains a curated set of **normal** and **corner-case** technical documentation samples plus a pytest suite that verifies:
- dual‑granularity retrieval (Chunk + CitationUnit),
- stitched context order (Step 1 before Step 2),
- citation numbering and titles,
- robustness to break keywords (FAQ/Glossary),
- support for non‑ASCII and setext headings,
- and endpoint/reference style sections.

## Contents

```
regression_pack/
  docs/                       # input Markdown documents
  tests/test_phase7e3_regression_pack.py
  runner/regression_runner.py
  artifacts/                  # report will be written here after tests
```

## How to run

From the root of your repository (where your `src/` package lives):

```bash
# Ensure dependencies and services are up (Neo4j, Qdrant, Redis).
# Set embedding provider env if needed:
export JINA_API_KEY=...

# Copy this folder into your repo (e.g., under tests/integration/regression_pack)
# or unpack and run from anywhere inside your repo.

# Option A: With pytest
pytest -q tests/test_phase7e3_regression_pack.py

# Option B: With the helper runner
python runner/regression_runner.py
```

After completion, review:
```
artifacts/phase7e3_regression_report.json
```
You can upload this JSON back for verification.

## Notes

- Each document carries a `DocTag: REGPACK-XX` string. Queries include the same token to ensure BM25 retrieval focuses on the intended document, avoiding cross‑doc interference.
- The test suite attempts to call `QueryService.search()` with a flexible signature to tolerate minor API differences. It also extracts citations by scanning lines like `[1] ...` from the markdown answer.
- Some cases include `ORDER_STEP_ONE` / `ORDER_STEP_TWO` markers to validate stitched context order heuristically.
- If your environment uses a different embedding provider or version, set `EMBEDDING_VERSION` accordingly, or adjust the test filter logic.

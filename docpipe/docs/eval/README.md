# Nutanix Eval Specs

This directory contains the populated post-conversion evaluation inputs for the
local Nutanix corpus.

- `nutanix_gold.json` is the deterministic `docpipe eval` spec: 45
  source-grounded presence and reading-order checks across all nine PDFs.
- `citation_questions.json` is the downstream RAG/citation prompt set: 14
  source-grounded questions with expected source PDFs, slugs, and required answer
  terms.

Run the deterministic gold spec after a full corpus conversion:

```bash
docpipe eval --out ../ETL-for-corpus/transformed-corpus --spec docs/eval/nutanix_gold.json
```

The citation questions are intentionally not executed by `docpipe eval`; they are
the handoff fixture for the downstream RAG gateway citation check. Validate the
fixture without a gateway:

```bash
python -m docpipe.citation_eval --spec docs/eval/citation_questions.json --dry-run
```

Run the live citation check after the RAG gateway is serving the ingested corpus:

```bash
python -m docpipe.citation_eval \
  --spec docs/eval/citation_questions.json \
  --base-url http://localhost:8000 \
  --report ../ETL-for-corpus/docpipe-citation-eval.json
```

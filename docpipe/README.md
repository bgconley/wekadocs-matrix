# docpipe — Nutanix PDF → Markdown extraction pipeline

Converts Nutanix product-documentation **PDFs** into a clean, ingestion-ready
**Markdown** corpus by rendering each page to an image and transcribing it with a
**Qwen3.6-27B** vision-language model over an OpenAI-compatible API.

`docpipe` is a **standalone component**: it *produces* a corpus; the RAG ingestion
path *consumes* it. It never imports the RAG code, and the RAG code needs no change
to accept its output. The only coupling is the on-disk Markdown contract (below).

---

## Why visual transcription

These PDFs are multi-column-free but dense: spec tables, CLI/nCLI/aCLI blocks,
admonition callouts, and hundreds of Prism UI screenshots. PDF text-layer
extraction mangles tables and loses the visual structure, so the primary path
renders each page and lets the VLM read it. (The text layer is used only as an
advisory QA cross-check.)

## Pipeline stages

| Stage | Module | What it does |
|------:|--------|--------------|
| 0 | `manifest.py` | Discover PDFs, sha256, page counts, embedded metadata → JSONL manifest (resume backbone) |
| 1 | `rasterize.py` | Page → PNG via PyMuPDF at a target DPI, long-side clamped |
| 2 | `convert.py` | One page/request to the VLM, prev-page context, concurrent + resumable |
| 3 | `stitch.py` | Concatenate pages; repair split tables/code/paragraphs; strip headers/footers |
| 4 | `clean.py` | De-hyphenate, single-H1, collapse whitespace, prepend YAML front matter |
| 5 | `validate.py` | Per-page garbled/empty detection + advisory text-overlap; document coverage |
| 6 | `output.py` | Write one `.md` per document in the ingestion-ready shape |

## Install

Everything it needs (`PyMuPDF`, `httpx`, `pydantic`, `Pillow`, `markdown-it-py`) is
already importable in the repo venv. To get the `docpipe` console script:

```bash
pip install -e docpipe            # from the repo root
# or run without installing:
python -m docpipe <command>
```

## Quick start

```bash
# 1. Confirm both endpoints are live (probes /v1/models + a tiny image round-trip)
python -m docpipe doctor

# 2. Calibrate on a single page (renders a PNG + prints the Markdown + image_tokens)
python -m docpipe inspect ETL-for-corpus/5c-book-of-ahv-administration.pdf 1

# 3. Convert the whole corpus (resumable; safe to Ctrl-C)
python -m docpipe convert --in ETL-for-corpus --out ETL-for-corpus/transformed-corpus

# 4. Check coverage / re-attempt any failures
python -m docpipe status
python -m docpipe retry-failed
```

Defaults already target the lab environment, so `docpipe convert` with no flags
works. Pass `--config docpipe/docpipe.toml` to tweak endpoints/DPI/concurrency.

## Commands

- **`convert`** — full pipeline over `--in` → `--out`. Flags: `--dpi`, `--figures
  {enriched,minimal,skip}`, `--only <slug/sha>`, `--limit N`, `--allow-incomplete`,
  `--json`.
- **`status`** — per-document coverage (ok / failed / missing pages) read from the
  work dir. No network.
- **`retry-failed`** — re-attempt only failed/missing pages, then re-assemble.
- **`inspect <pdf> <page>`** — render + convert one page; writes the PNG and `.md`
  under `<work>/inspect/` and prints image-token/latency stats. Use it to tune DPI.
- **`doctor`** — probe both endpoints and confirm image input works.

## Resume & idempotency

Each page is cached under `<work_dir>/pages/<sha256>/d<dpi>_<model>/NNNNN.md`. A
re-run converts only pages that are missing or failed for that exact `(dpi, model)`
key — finished pages are instant cache hits. Interrupting mid-run loses nothing.
Bumping `--dpi` or changing the model is a natural cache miss (fresh conversion).

## Endpoint targeting

Two Qwen3.6-27B-FP8 endpoints are configured. **Oxcart (pinned-stable vLLM) is the
default target**; Blackbird (dev-nightly SGLang, lower latency but unpinned) is
configured but disabled by default. Select with `--endpoint`:

```bash
docpipe convert                       # Oxcart only (default)
docpipe convert --endpoint all        # both endpoints, ~2x throughput
docpipe convert --endpoint blackbird  # Blackbird only
```

`doctor` always health-checks **both** regardless of the default.

## Tuning concurrency

Each endpoint caps at `max_running_requests = 4`. `inflight` (per endpoint) is set
just above that — 6 — so the server's continuous-batch stays full without deep,
wasteful queuing. A single shared work queue drained by per-endpoint workers gives
least-outstanding-requests routing and cross-endpoint failover for free (relevant
under `--endpoint all`). If you add GPUs or raise `max_running_requests`, raise
`inflight` to match. Watch the per-endpoint `req/s` and `fail` counts in the report.

## The run report

Every run ends with pages (converted / cached / failed), per-endpoint requests,
failures, average latency, throughput, and token in/out, plus documents written,
any incomplete docs, and QA-suspect pages. `--json` emits the same as machine data.

## Corpus contract (what the output guarantees)

Verified against `src/ingestion/parsers/markdown_it_parser.py` and `atomic.py`:

- **One `.md` per document**, UTF-8, at `<out>/<slug>/<slug>.md` (`per-doc`) — the
  ingestion CLI globs `*.md`/`*.html` recursively.
- **YAML front matter at byte 0.** Parser-consumed keys: `title`, `version`,
  `last_edited`. Also emitted (provenance, inert to the parser): `product`,
  `source_pdf`, `sha256`, `page_count`, `dpi`, `model_id`, `endpoint`,
  `pipeline_version`, `extracted_at`.
- **Title** = the document's own first H1 (faithful) → embedded PDF title →
  humanized filename. The body starts with that single H1 so nothing is lost to the
  parser's "content-before-first-heading is dropped" rule.
- **Headings** are chunk boundaries → a real ATX tree. **Fenced code** (with a
  language hint) and **GFM tables** are kept **top-level** — never nested in a list
  item (which the parser drops) and **never HTML** (which the parser drops).

To hand the corpus to ingestion, point `--out` at `data/ingest/nutanix/` (all files
under it derive `doc_category="nutanix"`), then:

```bash
scripts/ingestctl ingest data/ingest/nutanix --tag nutanix   # NOT --watch (broken stub)
```

## Offline vs. live

Rendering, stitching, cleaning, validation, `status`, and the unit tests are
**offline**. Only `convert` / `retry-failed` / `inspect` / `doctor` — the VLM calls
— need the Qwen3.6-27B endpoints (`blackbird`/`oxcart:18002`). These are **separate**
from the RAG model gateway (`10.25.0.50:8080`), which is only needed later, at
ingestion/eval time.

## Tests

```bash
python -m pytest docpipe/tests -q                 # offline unit tests
DOCPIPE_LIVE=1 python -m pytest docpipe/tests -q  # + live smoke against Blackbird
```

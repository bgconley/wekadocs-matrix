# docpipe — Design Review & Reference Specification

_A retroactive "what good looks like" specification for the docpipe PDF→Markdown
extraction pipeline, plus an adversarial review of the current implementation
against it. Derived from the 2023–2026 document-AI literature, not from the
implementers' priors._

**Method.** A multi-agent workflow: 4 web/arXiv research agents (SOTA systems,
VLM transcription best-practices, RAG+evaluation methodology, Qwen-VL fitness) and
6 code-review dimensions ran in parallel; every review finding was then handed to
an independent skeptic agent instructed to *refute* it, and only survivors are
recorded. A synthesis pass produced the ideal spec, the artifact spec, and the gap
analysis. (68 agents; ~3.8M tokens. One transient API drop crashed the first run —
it was resumed from cache.)

---

## The documents

| Document | What it is |
|---|---|
| **[IDEAL_SPEC.md](IDEAL_SPEC.md)** | The prescriptive reference design — 9 dimensions, RFC-2119 requirements (`R<dim>.<n>`), each with a rationale grounded in cited research. "What a solid docpipe *should* be." |
| **[ARTIFACT_SPEC.md](ARTIFACT_SPEC.md)** | The expected output contract — exact frontmatter schema, heading/code/table/figure fidelity rules, a **golden annotated `.md` skeleton**, and a 4-tier per-artifact QA acceptance checklist with metrics (TEDS, text-layer recall, Presence/Absence tests). "What we expect out of the artifacts." |
| **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** | Dimension-by-dimension scorecard of the current code vs. the ideal spec, the consolidated **P0/P1/P2 roadmap**, and the overall verdict. |
| **[FINDINGS.md](FINDINGS.md)** | All **49 adversarially-verified** code-review findings (9 high / 23 medium / 17 low), with failure scenarios and fixes. |
| **[RESEARCH.md](RESEARCH.md)** | The four cited research reports (SOTA survey, best-practices, RAG/eval, Qwen fitness). The evidence base. |

---

## Resolution log

- **2026-07-07 — Cluster 1 (fence-awareness): RESOLVED.** Findings **#2, #3, #22,
  #31, #40** fixed via a shared `docpipe/fences.py` fence-state primitive
  (`iter_lines_with_fence_state` / `open_fence_at_end`). `demote_extra_h1s`,
  `first_h1_text`, and `balance_fences` are now fence-aware — `balance_fences`
  acts only on an odd/dangling fence count, so a `#`/`##` comment inside a
  balanced block is never mistaken for a heading; `stitch._merge_seam`
  newline-joins an open-fence seam and keeps real hyphens (`read-only`, not
  `readonly`). TDD: 5 failing tests written first → 9 tests added (**41 pass**),
  `cd docpipe && pyright .` clean, and finding #2's exact failure verified resolved
  through the real downstream parser (**4 corrupt sections → 1 intact section**,
  commands retained as `code_blocks`).
- **2026-07-07 — Cluster 2 (payload-validation gate): RESOLVED.** Findings
  **#1, #6, #9, #24, #27** fixed fail-closed: empty/still-truncated model
  responses retry/fail instead of caching `ok`, and `is_done` /
  `document_coverage` now require a valid non-empty `.md` whose length matches
  the sidecar. TDD: async converter harness + artifact/coverage tests.
- **2026-07-07 — Cheap fail-soft slice: RESOLVED.** Findings **#15, #26, #28,
  #29** fixed: `/v1/models` probing skips dead endpoints and resolves from any
  live endpoint, all-dead resolution raises `VLMError`, pending conversion with
  zero runnable endpoints fails fast, and one bad PDF no longer aborts manifest
  discovery.
- **2026-07-07 — Cluster 3 remaining fail-soft paths: RESOLVED.** Findings
  **#7, #25, #30** fixed: non-retryable HTTP errors fail fast, retryable HTTP
  errors honor `Retry-After`, worker crash handlers survive failure-sidecar I/O
  errors, advisory `page_text()` QA cannot abort assembly, and `assemble_all`
  isolates one document failure instead of aborting the whole output stage.
- **2026-07-07 — P0 furniture/table-row stripping: RESOLVED.** Finding **#4**
  fixed: multi-page footer-band removal now requires a recurring page-edge band
  key, and the single-page page-number fallback now preserves table-shaped rows.
  Real rows ending in numeric cells (for example, capacity limits) survive while
  recurring footer bands like `AHV | Networking | 11/12` still drop.
- **2026-07-07 — P0 furniture structural-safety: RESOLVED.** Finding **#18**
  fixed: recurring section labels such as `Syntax`/`Parameters`/`Example`, table
  headers, separators, and fence lines are no longer classified as page
  furniture; repeated furniture strings are removed only from page-edge
  positions, so body occurrences survive.
- **2026-07-07 — P0 stitch seam-repair cluster: RESOLVED.** Findings **#19,
  #20, #21** fixed: `_merge_seam` only deduplicates a repeated table header when
  it matches the current table header, joins rule-8-compliant table continuation
  rows with a single newline, and preserves closed fence pairs as separate code
  blocks so the second block's language is not lost.
- **2026-07-07 — P0 manifest stale-reuse: RESOLVED.** Finding **#32 / P0-5**
  fixed: manifest fast-path reuse now requires same `rel_path`, same byte size,
  and the same current `sha256`, so same-size content edits are rescanned and
  cannot serve stale page artifacts under the old cache key.
- **2026-07-07 — P0 manifest relocation path refresh: RESOLVED.** Finding
  **#5** now fully closed: when a record is safely reused by matching
  `rel_path`/size/current `sha256`, the cached metadata is kept but `pdf_path`,
  `rel_path`, and `size_bytes` are refreshed from the currently discovered file,
  so moved-but-identical corpus inputs do not retain stale absolute paths.
- **2026-07-07 — P0 slug collision overwrite: RESOLVED.** Finding **#8 / P0-6**
  fixed: `assemble_all` detects duplicate output slugs in a batch and writes
  colliding documents under stable short-SHA-suffixed slugs, preventing
  same-stem PDFs in different directories from overwriting each other.
- **2026-07-07 — P1-5/R3.9 bare-CLI fencing: RESOLVED.** `clean_document` now
  deterministically wraps unfenced `ncli`/`acli`/`ncli>`/`nutanix@`/
  `<acropolis>`/`$ ` command groups in top-level `bash` fences, while skipping
  existing fences. This closes the proven command-structure-loss fix; the
  structure-aware `prev_tail` half of P1-5/R3.8 was closed in the
  2026-07-08 engine-justification P1 cluster below.
- **2026-07-07 — Convert-loop reliability mediums: RESOLVED.** Findings **#13,
  #14, #16, #17** fixed: retry budget is tracked per endpoint so a dead oxcart
  worker cannot exhaust blackbird's opportunity to transcribe a page; workers
  defer retry jobs to still-untried endpoints; backoff sleeps happen in detached
  requeue tasks instead of occupying worker slots; truncation retries use a
  genuinely higher token ceiling and refuse shorter retry output; and
  `_prev_tail` sidecar reads are executor-backed, guarded, and fall back to
  native PDF text on read faults. TDD: 5 behavior tests written RED first, then
  green against the real converter loop. Follow-up validation found and fixed a
  `--endpoint all` livelock in the #13 anti-affinity predicate: once every active
  endpoint has tried a page, workers now stop deferring and allow the remaining
  per-endpoint retry budget to make progress.
- **2026-07-07 — Review correction pass: RESOLVED.** The external follow-up's
  three corrections are reflected in code/status: finding **#5** is fully closed
  by the relocation path refresh, the finding **#4** single-page fallback
  residual is covered by regression test, and the full package Pyright gate
  (`cd docpipe && pyright .`) is clean with tests included.
- **2026-07-08 — Config/status operator-safety cluster: RESOLVED.** Findings
  **#10, #11, #12, #34, #35, #36, #37, #38, #41** fixed: zero active endpoints
  fail fast before pool startup, malformed `--endpoint` selections and invalid
  `DOCPIPE_DPI` values raise clear errors, partial TOML endpoint overrides merge
  by endpoint name over curated defaults, unknown TOML keys are rejected,
  `--limit 0` selects zero records, and status/discovery preserve prompt-version
  keysets while showing the full model id. TDD: 9 behavior tests written RED
  first, then green; full gate is **85 passed / 1 skipped** and `pyright .` clean.
- **2026-07-08 — Engine-justification P1 cluster: RESOLVED.** **P1-5/R3.8,
  P1-1, and P1-2** fixed: `prev_tail` now expands to the last open fence/table/
  list boundary instead of blindly clipping to a fixed character count; current
  page text-layer anchors are injected into `build_messages` with strict
  glyph-disambiguation framing; low text-layer overlap on text-heavy pages retries
  instead of caching a hallucinated/omissive page; and sampler requests now use
  `temperature=0.1`, `top_p=0.9`, and `frequency_penalty=0.2`. `PROMPT_VERSION`
  bumped to **3** so anchored prompts are a natural cache miss. TDD: 6 behavior
  tests written RED first, then green; full gate is **91 passed / 1 skipped** and
  `pyright .` clean.
- **2026-07-08 — Fail-closed quality gates: RESOLVED.** **P1-3 and the
  deterministic P1-10 harness** fixed: final Markdown now passes a hard Tier-0
  contract gate before write, contract failures surface in text/JSON reports, and
  `docpipe eval` provides offline pass-fraction, Tier-0 pass rate, boilerplate
  leak count, table-validity checks, baseline loop/compression guards, and
  optional presence/absence/reading-order/table JSON cases. TDD: 8 behavior tests
  written RED first, then green; full gate is **101 passed / 1 skipped** and
  `pyright docpipe/.` clean.
- **Remaining gating work:** rasterization fidelity, populated golden/citation
  eval, and remaining robustness lows/mediums not covered here.

## Executive verdict

**The architecture is sound.** Single-VLM full-page transcription, one page per
request, Markdown out, over a resumable manifest/cache — this is *literally the
olmOCR reference architecture* (arXiv:2502.18443), and it is the correct choice
for this single-column, screenshot- and CLI-dense, contract-bound, **one-time**
corpus. The stage decomposition, offline-by-default posture, content-addressed
cache (with `prompt_version` in the key), probe-don't-hardcode endpoint handling,
and the contract-tuned prompt are all real and well-executed. **Every gap is
fixable within the existing stage boundaries — none requires re-architecting.**

**Qwen3.6-27B is the right engine — conditionally, and the core operating
conditions are now implemented.** On pure OCR-per-parameter, specialists beat it:
OmniDocBench places
MinerU2.5 (1.2B) at 90.67 and PaddleOCR-VL (0.9B) at 92.56, *above* Qwen3-VL-235B
(89.15). But a general VLM is justified here for four reasons a specialist cannot
satisfy: (i) it **describes** figures for retrieval (specialists only *box* them);
(ii) it is **steerable to the exact downstream contract** — ATX + top-level GFM +
fenced code + single H1 — whereas the leaderboard winners emit HTML/OTSL/DocTags
that our parser *silently drops*; (iii) it does CLI-vs-prose + cross-page seam
judgment; (iv) one-model operational simplicity. Throughput is irrelevant at 936
one-time pages. The two literature-imposed operating conditions are now present:
current-page **text-layer anchoring** is injected as a glyph-disambiguation
reference and text-heavy low-overlap pages retry, while the sampler no longer runs
bare `temperature=0` without penalties. Remaining proof shifts from engine
eligibility to corpus QA: Tier-0 contract gating, eval/golden checks, and the full
build/citation evaluation.

**The holdbacks are a well-bounded set of data-loss/liveness bugs and two missing
quality layers** — not architecture.

---

## Scorecard (see GAP_ANALYSIS.md for justifications)

| # | Dimension | Rating |
|---|-----------|--------|
| 1 | Architecture & stage decomposition | **PARTIAL** — anchoring present; Tier-0/eval layers still pending |
| 2 | Rasterization strategy | **PARTIAL** — DPI-not-patch-aligned; escalation dead/defective |
| 3 | Model & prompt strategy | **PARTIAL** — anchoring, structure-aware tail, and sampler hardening present; metadata-first/eval still pending |
| 4 | Model / endpoint choice | **PARTIAL** — engine sound; text-layer arbiter active; specialist oracle/A-B still pending |
| 5 | Concurrency & throughput | **FULLY MET** (design) — bugs are robustness-within-the-design |
| 6 | Resumability & caching | **PARTIAL** — page-level excellent; manifest reuse is size-based |
| 7 | Validation / QA gates | **PARTIAL** — fail-closed Tier-0 gate active; richer corpus QA still pending |
| 8 | Evaluation harness & metrics | **PARTIAL** — offline pass-fraction harness active; populated gold/TEDS eval still pending |
| 9 | Output artifact contract | **PARTIAL** — design met; runtime output violated by the P0 bugs |

**1 fully met · 8 partial · 0 missing.**

---

## The P0 roadmap (correctness / data-loss / liveness)

These matter most because this is a **one-time 936-page build** — a silent drop or
a hang is maximally costly. All are addressable without changing the design.

- **P0-1 — Truncated & empty output cached as `status=ok` forever.** Re-check
  `truncated` after the token-bump; treat still-truncated **or** empty content as a
  retryable failure, not `write_success`. *(convert.py — 3 findings converge here.)*
- **P0-2 — Fence-unaware clean passes shatter code blocks.** `demote_extra_h1s`,
  `balance_fences`, `first_h1_text` all treat `#`/`##` **inside** a ```` ``` ````
  block as headings — a bash comment becomes a false chunk boundary or the doc
  title. Add a shared open-fence line-state and skip fenced regions. *(clean.py — I
  flagged the seed of this earlier; the review found the full cluster.)*
- **P0-3 — Furniture stripping deletes real content.** `_FURNITURE_BAND_RE` matches
  a **table row whose last cell is a number** (RF values, ports, node maxima — the
  corpus's highest-value facts). Exclude structural lines (headings, table rows,
  fences) from furniture classification. *(stitch.py)*
- **P0-4 — Seam repair is structure-state-blind.** `_merge_seam` can graft a
  different table's rows under a header, break a split table with a blank line, and
  fuse two separate code blocks losing the language. Track open-fence/open-table
  state across the fold. *(stitch.py — 5 findings)*
- **P0-5 — Manifest stale-reuse serves old transcriptions.** Same-size in-place edit
  is skipped (size-only fast-path); rehash or also require matching mtime. *(manifest.py)*
- **P0-6 — Slug collision silently overwrites a whole doc.** Two same-named PDFs in
  different dirs clobber to one `.md`. Disambiguate `output_path`. *(output.py)*
- **P0-7 — "Complete" judged from the sidecar alone.** A torn/0-byte `.md` beside an
  intact `ok` sidecar ships as complete. Require `md_path.is_file()` + length check. *(validate.py/output.py)*
- **P0-8 — Worker death hangs the whole run.** If `write_failure` raises (disk full)
  inside the worker's `except`, `queue.join()` wedges forever. Guard the handler. *(convert.py)*
- **P0-9 — One dead endpoint / bad PDF / moved source aborts the build.** `probe_models`
  is all-or-nothing; `scan_pdf` and per-doc `page_text` are unguarded. Make failover
  actually engage. *(vlm_client.py, manifest.py, output.py)*

**P1** now centers on rasterization fidelity plus populated golden/citation eval.
**P2** is polish (dead config, DX, tests).
Full detail in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

---

## External review cross-check (2026-07-07)

A second independent review was run against the same code. It **corroborates** this
review (all its valid findings map to items already here — convergent validation)
and adds one empirically-measured data point worth elevating. Rolled in:

- **Elevated → top of P1: command-structure loss on cheat-sheet pages (verified).**
  Parsing the sample `5c` output through the repo's own `markdown_it_parser` yields
  **1 of 22 sections with `code_blocks`**; every `manage_ovs`/`ovs-vsctl` reference
  parsed as prose. Since `extract_commands()` Pattern 1 reads only
  `section["code_blocks"]`, those ~20 command groups never become `Command` graph
  nodes — and prompt-strengthening (v2) did **not** fix it. **P1-5 (deterministic
  Stage-4 bare-CLI fencing pass) is the model-independent fix and moves to the top
  of P1.**
- **Rolled in — QA flags must feed retry, not just telemetry.** `assess_page()`
  detects empty/garbled/short pages but assembly only *reports* them. Route Tier-1
  QA flags (empty/garbled) to retry/quarantine (pairs with P0-1 and IDEAL_SPEC R7.5).
- **Rolled in — strengthen the live smoke test** to a full-pipeline 2–3 page
  `convert` (manifest→…→output+coverage), not a single `VLMPool.chat()` call.
- **Already covered:** manifest size-only reuse = **P0-5**; previous-page continuity
  = **P1-5** (structure-aware tail) + **P0-4** (authoritative stitch). Note: per the
  anchoring literature (R3.3), the text-layer "fallback" the external review flags is
  actually the *better* input — the fix is P0-4 + first-class anchoring (P1-1), not
  forcing generated-markdown context.
- **Settled decision, not a defect:** Oxcart-only defaults / no `weight` routing. The
  Blackbird-primary, dual-saturation language in the original brief was superseded by
  the decision to target Oxcart. (`weight` is dead config regardless — delete it, P2-1.)

## Definition of "corpus-ready"

A build is corpus-ready when **(A)** Tier-0 contract-conformance pass rate = 100%
(fail-closed); **(B)** the olmOCR-bench-style unit-test pass-fraction meets target,
broken down by content type; **(C)** zero boilerplate leaks; **(D)** table-validity
and text-heavy-page text-layer recall clear their floors, quarantines triaged;
**(E)** every page is either transcribed or explicitly recorded failed — none
silently dropped.

---

## Recommended next step

Run the remaining proof layers before the full 936-page build: Tier-0 contract
gating, eval/golden checks, and rasterization-fidelity review. The engine-choice
prerequisites (**P1-5/R3.8**, **P1-1**, **P1-2**) are now implemented; the next risk
is proving the resulting corpus artifacts at build scale. P2 is optional polish.

# docpipe — Gap Analysis vs. the Ideal-Design Spec

## Credit first: what the implementation gets right

This is a **genuinely working, tested, resumable, offline-by-default pipeline**, and the bones match the reference design closely. Specific strengths, grounded in code:

- **Correct paradigm.** Single-VLM full-page transcription, one page/request, Markdown out (`prompts.py`, `convert.py`) — exactly the olmOCR architecture the spec prescribes, not a modular detector pipeline.
- **Clean stage decomposition, all resumable.** Manifest (`manifest.py`) → rasterize (`rasterize.py`) → transcribe (`convert.py`) → stitch (`stitch.py`) → clean (`clean.py`) → validate (`validate.py`) → output (`output.py`), each a pure-ish module.
- **Content-addressed page cache** keyed on `(sha, dpi, model, PROMPT_VERSION)` (`artifacts.py:45`) — a prompt bump is a natural cache miss, as the spec demands.
- **Probe-don't-hardcode** model id from `/v1/models` (`vlm_client.py:113-138`); dual endpoint with pinned-vLLM default + SGLang nightly (`config.py:91-112`).
- **A genuinely well-tuned prompt** written against the *verified* downstream parser (`prompts.py:1-19`): forbids raw HTML tables, forces top-level GFM + fenced code, single H1, furniture skipping.
- **Provenance-rich frontmatter** at byte 0 (`clean.py:124-147`) and a machine-readable `--json` run report (`report.py:90`).
- **161 passing offline unit tests** over stitch/clean/validate/manifest/config/rasterize/convert/output/eval/citation.

The remaining gaps below are **not architectural** — they are downstream
citation/RAG proof, optional scored metric proof layers, and lower-priority
polish after the full corpus build. The
original engine-justification gaps — anchoring, structure-aware continuity tail,
and sampler hardening — were closed in the 2026-07-08 Phase 6.0 remediation; the
fail-closed Tier-0 contract gate, deterministic eval harness, and populated
gold/citation fixtures were also added in Phase 6.0. The Oxcart full build now
has 936/936 ok pages, 9/9 written docs, default eval 34/34, and populated gold
eval 45/45. The source-aligned live citation fixture now reaches 14/14, but
downstream citation eval is provisional rather than resolved because the
retrieval changes that improved it remain uncommitted.

---

## Dimension-by-dimension ratings

| # | Dimension | Rating | One-line justification (code) |
|---|-----------|--------|-------------------------------|
| **1** | Architecture & Stage Decomposition | **PARTIAL** | Stages 0-6 all present, resumable, single-VLM (R1.1/R1.2/R1.5 met), and current-page text-layer anchoring now feeds `build_messages` with low-overlap retry. Remaining gaps are downstream citation/RAG proof and optional scored-metric layers, not architecture. |
| **2** | Rasterization Strategy | **FULLY MET** | Base render targets a patch-grid-aligned ~2400 px long side (`dpi=218`, `max_long_px=2408`), rendered PNG dimensions are rounded to Qwen's 28 px effective grid, QA escalation raises both DPI and clamp (`300` / `3304`), and chat requests set explicit `mm_processor_kwargs` pixel bounds. `image_tokens` logging remains in place. |
| **3** | Model & Prompt Strategy | **PARTIAL** | Strong on R3.1 (zero-shot), R3.3 anchoring, R3.5 (GFM+flatten, rule 2), R3.6/R3.9 command fencing, R3.7 enriched figures, R3.8 structure-aware tail, and R3.10 sampler hardening (`temperature=0.1`, `top_p=0.9`, `frequency_penalty=0.2`). Still partial only because metadata-first output and optional scored metrics remain out of scope. |
| **4** | Model / Endpoint Choice | **PARTIAL** | R4.1 engine choice is sound for this corpus; R4.3 text-layer arbitration is active for text-heavy low-overlap pages; R4.6 probe + dual-endpoint met (`vlm_client.py`, `config.py`). **Missing:** R4.4 specialist QA oracle and R4.7 8B A/B. |
| **5** | Concurrency & Throughput | **FULLY MET** (design) | Shared queue drained by per-endpoint worker pools (`convert.py:122-135`), `inflight=6` above `max_running=4` (`config.py:100`), page-level parallelism, bounded retries + fail-loud (`convert.py:198-221`), reasoning suppressed. The review findings here are *robustness bugs inside* the correct design, not missing capability. |
| **6** | Resumability & Caching | **PARTIAL** (page-level excellent) | R6.1 manifest, R6.2 per-page cache w/ prompt_version, R6.3 offline-by-default, R6.4 retry-failed, R6.5 input-keyed — all met at the **page** level. Undermined at the **doc** level: manifest reuse is **size-based, not sha** (`manifest.py:192`), so a same-size edit serves stale pages; retry-failed doesn't escalate the pixel budget (R6.4 partial). |
| **7** | Validation / QA Gates | **PARTIAL** | Tier-0 now fails closed after `clean_document`: frontmatter@0, non-empty title, exactly one H1, no orphan pre-heading content, monotone ATX headings, balanced/top-level fences and tables, no raw HTML table/code, and no indented code blocks. Page-level heuristics also gate retries for empty/truncated/low-overlap pages. Remaining gaps: richer AST round-trip/TEDS-style checks and corpus-run calibration. |
| **8** | Evaluation Harness & Metrics | **PARTIAL** | Offline eval harness now exists with pass-fraction headline, reference-free corpus checks, JSON spec cases for presence/absence/reading-order/table/contract/baseline, boilerplate leak count, Tier-0 pass rate, and table-validity rate. The populated Nutanix gold and citation fixtures now exist. Full-build deterministic eval is green; live downstream citation eval is source-aligned at 14/14 but provisional pending retrieval/RAG review. Remaining gaps are retrieval/RAG review plus optional NED/TEDS/CDM-style scored metrics. |
| **9** | Output Artifact Contract | **PARTIAL** (design FULLY MET; runtime conformance now gated) | Contract is correctly specified and constructively produced: per-doc path (`output.py:43`), frontmatter+provenance including parser-consumed `last_edited`, title precedence (`clean.py:53`), ATX tree, top-level fences/tables + balance, furniture clean, hard Tier-0 gating, and `--json` report. Remaining: R9.7 canonical terminology is only product-inference (`manifest.py:89`), no CLI-spelling normalization. |

**Scorecard:** 2 fully met (Dims 2 and 5), 7 partial (Dims 1, 3, 4, 6, 7, 8, 9 — with 9's design fully met), 0 missing.

---

## Historical remediation roadmap

Historical priority rule: **P0 = corpus corruption, silent data-loss, or a hung/aborted build** (this is a one-time 936-page build — a silent drop or a hang is maximally costly and unrecoverable-by-default). **P1 = quality/eval** (the spec dimensions the impl lacks + retry-quality bugs). **P2 = polish** (dead config, DX, tests). Related findings were consolidated by root cause. The bullets below retain the original fix intent; the README resolution log is the current completion record. Gating P0/P1 remediation for the extraction artifacts is closed; downstream citation/RAG proof remains provisional and should be reviewed separately from docpipe extraction.

### Historical P0 — correctness / data-loss / liveness

All P0 items below are resolved in the implementation and recorded in
`README.md`'s resolution log.

- **P0-1 — Truncated & empty output written as `status=ok`, cached forever.** *Change:* in `convert.py:181-194`, re-check `result.truncated` after the bump and treat still-truncated **or** empty/whitespace content (`vlm_client.py:197`) as a retryable failure, not `write_success`; add `truncated`/`finish_reason` to `PageMeta` (`artifacts.py:26`). *Benefit:* eliminates silently cut-off tables and blank pages that `is_done()` then skips forever — the single most dangerous data-loss class (three findings converge here).
- **P0-2 — Fence-unaware clean passes shatter code blocks and mis-title docs.** *Change:* add a shared open-fence line-state iterator and make `demote_extra_h1s` (`clean.py:57`), `balance_fences` (`clean.py:87`), and `first_h1_text` (`clean.py:41`) skip lines inside `` ``` `` fences. *Benefit:* stops `#`/`##` comments in bash/config blocks becoming **false ATX chunk boundaries** + un-fenced commands, and stops a code comment being chosen as the frontmatter title.
- **P0-3 — Furniture stripping deletes real content.** *Change:* in `stitch.py:29-66`, exclude structural lines (ATX headings, table rows/separators, fences) from furniture classification+removal, tighten `_FURNITURE_BAND_RE` (`stitch.py:19`) to true page-number tails on non-table lines, and only remove a match in its original edge position. *Benefit:* stops deletion of **numeric table rows** (RF values, ports, node maxima — the corpus's highest-value facts) and multi-page table headers/recurring real headings.
- **P0-4 — Seam repair is structure-state-blind.** *Change:* in `_merge_seam` (`stitch.py:81-120`) track open-fence/open-table state across the fold; dedup a table header only when headers are byte-equal; join a continued table/prose with a single `\n` (never `\n\n`); fuse fences only when acc's block is actually open; gate de-hyphenation. *Benefit:* fixes wrong-header table grafting, broken rule-8-compliant tables, fused code blocks with lost language, welded CLI commands (five findings).
- **P0-5 — Manifest stale-reuse serves old transcriptions.** *Change:* in `manifest.py:190-194`, drop the size-only shortcut (rehash — negligible vs a VLM pass) or also require matching mtime, and always refresh the reused record's absolute `pdf_path`. *Benefit:* an in-place same-size edit actually re-converts; a relocated corpus tree stops failing every page / crashing assembly.
- **P0-6 — Slug collision silently overwrites a whole doc.** *Change:* disambiguate `output_path` (`output.py:43`) by `rel_path`/sha suffix on collision, or detect duplicate slugs and error. *Benefit:* two same-named PDFs in different product dirs no longer clobber to one `.md`.
- **P0-7 — "Complete" is judged from the sidecar alone; a torn `.md` is baked in.** *Change:* make `document_coverage` (`validate.py:110`) require `md_path.is_file()` (mirror `is_done`), validate `len(md)` against `meta.char_len` on read, and stop `read_md(...) or ""` masking a missing file (`output.py:79`). *Benefit:* an rsync-torn/0-byte `.md` beside an intact `ok` sidecar is no longer shipped as complete.
- **P0-8 — Worker death hangs the whole run.** *Change:* wrap the `except` body (incl. `write_failure`) in `_worker` (`convert.py:150-158`) in its own try/except so a worker can never die; cancel the run on unrecoverable disk errors. *Benefit:* a disk-full/read-only FS fails loudly instead of wedging `queue.join()` forever.
- **P0-9 — One dead endpoint / bad PDF / moved source aborts or hangs the entire build.** *Change:* make `probe_models` (`vlm_client.py:117`) tolerate partial failures (resolve from any live endpoint, fail only if all dead); wrap per-page `page_text` (`output.py:84`) and per-doc assembly, and per-file `scan_pdf` (`manifest.py:195`), in try/except; raise a clear `SystemExit` when `active_endpoints` is empty (`convert.py:126`). *Benefit:* the cross-endpoint failover the design advertises actually engages; one corrupt input or one down host can't zero out a 936-page run.

### P1 — quality / eval (gating items closed)

- **P1-1 — Document anchoring (Dim 1.5/3.3/4.3): RESOLVED 2026-07-08.** Text-layer extraction is now first-class for conversion: this page's born-digital lines are capped and injected into `build_messages` as a glyph-disambiguation reference, and text-heavy low-overlap output retries before caching.
- **P1-2 — Sampler hardening (R3.10): RESOLVED 2026-07-08.** Chat requests now send `temperature=0.1`, `top_p=0.9`, and `frequency_penalty=0.2`, avoiding the prior bare τ=0/no-penalty configuration.
- **P1-3 — Tier-0 fail-closed contract gate (R7.0): RESOLVED 2026-07-08.** After `clean_document`, `validate_contract` asserts frontmatter@0 + non-empty title, exactly one title-matched H1, monotone ATX heading tree, no pre-heading content, balanced/top-level fences, top-level GFM tables, no raw HTML table/code, and no indented code blocks; `assemble_document` now refuses to write violating artifacts and surfaces violation codes in text/JSON reports.
- **P1-4 — Truncation-bump math (`convert.py:183`): RESOLVED 2026-07-07.** The bump now uses a genuinely higher token ceiling, refuses shorter retry output, and still-truncated/empty output is routed through the fail-closed retry path.
- **P1-5 — Structure-aware prev_tail + deterministic bare-CLI fencing (R3.8/R3.9): RESOLVED 2026-07-08.** `prev_tail` now expands to the last open fence/table/list boundary, and Stage 4 already fences unfenced command-looking lines.
- **P1-6 — HTTP error classification + Retry-After (`vlm_client.py:190`, `convert.py:198`): RESOLVED 2026-07-07.** Non-retryable HTTP errors fail fast, retryable HTTP errors honor `Retry-After`, and auth/client failures no longer burn the page retry budget.
- **P1-7 — Retry anti-affinity + fast-fail damping (`convert.py:200-212`): RESOLVED 2026-07-07.** Retry budget is tracked per endpoint, failed endpoints are avoided while alternatives remain, and backoff requeues happen off-worker; the follow-up `--endpoint all` livelock is also fixed.
- **P1-8 — Rasterization fidelity (`rasterize.py`, Dim 2): RESOLVED 2026-07-08.** Base long-side is now patch-grid-aligned at ~2400px, rendered PNG axes are multiples of 28, QA-flagged pages retry with both higher DPI and a higher long-side clamp, and each VLM request sets server `min_pixels`/`max_pixels`.
- **P1-9 — Rule-4 procedure ordinals lost (`prompts.py:92`): RESOLVED 2026-07-08.** Rule 4 now tells the model to preserve the visible ordinal in prose (for example, `Step 2: ...`) before moving command/code/table blocks to top-level form, avoiding downstream list-renumbering without nesting command blocks where the parser drops them.
- **P1-10 — Evaluation harness + richer detectors (Dim 8, R7.4): RESOLVED for the deterministic harness and populated fixtures 2026-07-08; downstream citation proof remains provisional.** `docpipe eval` now runs offline corpus checks with pass-fraction as the headline; default reference-free cases cover Tier-0 contract, baseline repetition/compression/long-line/intra-line loops, boilerplate absence, and table rectangularity, while optional JSON specs add presence, absence, reading-order, table, contract, and baseline cases. The populated Nutanix gold and citation fixtures now exist and the full-build deterministic runs are green (default 34/34, gold 45/45). Live downstream citation eval currently reaches 14/14 on the source-aligned fixture and depends on uncommitted retrieval/MCP changes, so it is not yet a resolved RAG gate.

### P2 — optional polish (dead config, DX, observability, tests)

- **P2-1 — Remove or wire dead config (optional):** endpoint `weight` (`config.py:30`) is read nowhere; Oxcart-only operation is a settled decision, so this is polish rather than a gating defect.
- **P2-2 — Config/CLI robustness: RESOLVED 2026-07-08.** `--limit 0`, invalid `DOCPIPE_DPI`, TOML endpoint merging, unknown TOML keys, and malformed/empty endpoint selections are covered by the config/status operator-safety remediation.
- **P2-3 — `status` truthfulness: RESOLVED 2026-07-08.** Prompt-version keysets and full model-id reporting are covered by the config/status operator-safety remediation.
- **P2-4 — Observability/DX: RESOLVED 2026-07-08.** kv logging now escapes newlines and quotes values containing `=`, `_outputs` is bounded to the latest page per document, zero-page/encrypted PDFs are rejected at scan, and frontmatter emits the parser-consumed `last_edited` key.
- **P2-5 — Tests: RESOLVED 2026-07-08.** Low-alnum QA tolerates ASCII tables/box diagrams while still flagging symbol noise; offline tests now cover `assemble_document` composition, incomplete-doc gating, output layouts, report text/JSON math, VLM HTTP/timeout/malformed/truncated paths, and manifest JSONL round trips.

---

## Overall verdict

**Is the architecture sound? Yes.** The single-VLM full-page-transcription design over a resumable manifest/cache, with deterministic post-passes as the joining authority and reference-free QA, is precisely the olmOCR reference architecture the spec prescribes — and it is the *correct* choice for this single-column, screenshot-/CLI-dense, contract-bound, **one-time** corpus. The stage boundaries, offline-by-default posture, content-addressed cache (with `prompt_version` in the key), probe-don't-hardcode endpoint handling, current-page anchoring, sampler hardening, Tier-0 contract gate, eval harness, populated eval fixtures, rasterization fidelity, and the contract-tuned prompt are all real and well-executed. **Every remaining gap is fixable within the existing stage boundaries** — none requires re-architecting. The remaining extraction-side work is optional scored metrics/polish; downstream citation proof belongs to the retrieval/RAG layer and is still provisional.

**Is Qwen3.6-27B the right engine? Yes — conditionally, and the core operating conditions are now met.** Per the spec's own R4.1 test, all five justifying conditions hold for this corpus (hundreds of pages so throughput is irrelevant; figure-dense so semantic caption enrichment is needed — a task specialists *box* but cannot *describe*; a bespoke contract that must be emitted directly, avoiding the HTML/OTSL/DocTags that leaderboard-winning specialists emit and the parser silently drops; CLI-vs-prose + seam judgment; one-model operational simplicity). You are correctly buying **figure understanding + steerability + simplicity, not OCR-per-dollar** (where 1-2B specialists beat it). Dim-3 anchoring, structure-aware continuity, repetition/determinism hardening, Tier-0 gating, and the offline eval harness are now implemented and proven on the full build. Downstream citation behavior is promising but not yet resolved.

**Bottom line:** sound architecture and a defensible engine choice, executed as a real working pipeline with green full-build deterministic corpus gates. What remains on the extraction side is optional polish/scored-metric work; downstream citation/RAG proof is source-aligned but provisional and must not be treated as resolved until the retrieval changes are reviewed and committed.

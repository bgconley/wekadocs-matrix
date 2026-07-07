# IDEAL-DESIGN SPECIFICATION — VLM PDF→Markdown Extraction Pipeline for a Nutanix-Docs GraphRAG Corpus

**Status:** Reference design (prescriptive). Describes what a best-in-class `docpipe` *should* be, derived from the 2023–2026 document-AI literature — not the current implementation.

**North-star workload (fixed design driver).** A *one-time* build over ~936 pages / 9 born-digital, WeasyPrint-generated Nutanix PDFs: single-column, screenshot-dense (hundreds of Prism UI captures), CLI/nCLI/aCLI- and spec-table-dense, feeding a **GraphRAG** corpus through an **idiosyncratic downstream Markdown contract** (ATX headings as hard chunk boundaries, single H1, frontmatter at byte 0, top-level GFM tables + fenced code only — raw-HTML and list-nested blocks are *silently dropped*). Two consequences dominate every dimension: (a) **steerability to the contract outweighs raw OCR-accuracy-per-dollar**, and (b) **throughput is a non-goal** (~936 one-time pages).

**Conventions.** MUST / SHOULD / MAY are normative (RFC-2119 sense). Requirements are ID'd `R<dim>.<n>`. Every requirement carries a rationale grounded in the cited research.

---

## Dimension 1 — Architecture & Stage Decomposition

**Requirement.**

- **R1.1** The pipeline MUST be a **single-VLM full-page-transcription pipeline** — render each page to one image, transcribe the whole page to Markdown in one request — *not* a modular detector→OCR→table→reading-order pipeline.
- **R1.2** It MUST decompose into **discrete, individually resumable, mostly-offline stages** over a manifest backbone:

```
 0 Manifest      discover PDFs → sha256, page count, embedded metadata → JSONL (resume backbone)
 1 Rasterize     page → PNG at an adaptive pixel budget, patch-grid-aligned (Dim 2)
 1.5 Anchor      extract born-digital text layer + block coords per page (PyMuPDF)   ← first-class stage
 2 Transcribe    one page/request: image + anchor + structure-aware prev-tail; concurrent + cached (Dim 3,5,6)
 3 Stitch        concatenate; AUTHORITATIVE seam-repair of split tables/code/paras; strip furniture; de-hyphenate
 4 Clean         single-H1, ATX normalize, balance fences, deterministic bare-CLI fencing, frontmatter@byte0
 5 Validate      tiered QA gates: contract / structural / text-layer+property / sampled image-judge (Dim 7)
 6 Output        one .md/doc in ingestion shape + machine-readable run report/manifest
```

- **R1.3** **Anchor extraction MUST be a first-class stage (1.5), not a Stage-5 QA afterthought** — its output feeds *both* the transcription prompt (Dim 3) and the QA gates (Dim 7).
- **R1.4** **Separation of authority.** The model transcribes *best-effort*; deterministic post-passes are the **authority** for cross-page joining, table-header re-attachment, fence balancing, and furniture stripping. The model MUST NOT be relied on to produce globally-valid structure across page seams.
- **R1.5** `docpipe` MUST remain a **standalone producer** decoupled from the RAG consumer; the *only* coupling is the on-disk contract (Dim 9). It never imports RAG code.

**Rationale.** "Render page → full-page VLM transcription on vLLM/SGLang, one page/request, Markdown out, resumable cache" is *literally the olmOCR reference architecture* (**olmOCR**, arXiv:2502.18443) — the field's validated design for born-digital PDF→Markdown; this spec adopts it deliberately. The modular detector-pipeline alternative (**Docling** RT-DETR+TableFormer, **Marker**+Surya) earns its keep chiefly through **reading-order recovery on multi-column pages**, and **OmniDocBench** (arXiv:2412.07626) shows reading order is precisely where the field's accuracy diverges — but the corpus is single-column, so that edge does not apply and the simpler single-model paradigm is correct. R1.4 is forced by cross-page continuity mechanics: a GFM table continued on page N *has no header row*, so it is invalid GFM in isolation — the model cannot simultaneously "continue" and "emit standalone-valid GFM," so the stitcher must own re-attachment (research: cross-page continuity; olmOCR and **Nougat**, arXiv:2308.13418, both process pages independently and defer joining to post-processing).

---

## Dimension 2 — Rasterization Strategy

**Requirement.**

- **R2.1** Render to a **pixel budget aligned to the model's patch grid, not a bare DPI.** For the Qwen VL family (14×14-px patches, 2×2 merge → 28×28-px effective cell, image-tokens ≈ pixels/784), each rendered axis MUST be **pre-rounded to a multiple of 28** to avoid a second server-side resample that blurs glyphs.
- **R2.2** **Base render: long side ≈ 2400 px** (≈218 DPI on US-Letter) → 6 pt cap-height ≈ 18 px, 8 pt ≈ 24 px, ~5.7 k image tokens. This clears the ~20 px cap-height glyph-reliability floor for body text. The base MUST NOT be 2000 px (≈182 DPI → 6 pt ≈ 15 px, marginal).
- **R2.3** **Escalation MUST raise the pixel clamp, not merely the DPI.** The garbled-page retry MUST raise the long-side clamp (e.g., to **3300 px ≈ 300 DPI**, 6 pt ≈ 25 px, ~10.7 k tokens). A design that escalates DPI while holding a fixed long-side clamp is **defective**: at a 2000-px clamp both 200 and 300 DPI bind the clamp and re-render a **byte-identical image**, so the retry gains *zero* resolution.
- **R2.4** **Server pixel bounds MUST be set explicitly** — launch vLLM/SGLang with `--mm-processor-kwargs '{"max_pixels": ~8.6e6, "min_pixels": ~200704}'` (or per-request `extra_body`). The assumption that "server pixel bounds are not exposed" is **false**.
- **R2.5** **Verify, don't assume.** Every page MUST log `image_tokens` from `usage.prompt_tokens_details`. A native 2400×1854 page MUST report ~5.7 k (a 2000×1545 page ~3.9 k); **~1,280 means the server silently downscaled to its ~1.0 MP default and small-table text is already lost.**
- **R2.6** **Per-page adaptivity:** default to the base budget; escalate the clamp only on QA-flagged (garbled / low-overlap) pages; cap below the 16,384-visual-token ceiling (≈4077 px ≈370 DPI) and well under context.

**Rationale.** Qwen tiling math and the 4–16,384 visual-token range are from the **Qwen2-VL technical report** (arXiv:2409.12191); the **HF Qwen2-VL processor** defaults `max_pixels = 28·28·1280 ≈ 1.0 MP`, the silent bottleneck R2.4/R2.5 defend against (a 1.0 MP clamp is ~104 DPI → 6 pt ≈ 8.6 px, unreadable). **OmniDocBench** names input resolution / token length as the headline VLM failure mode, producing "missing content in dense pages" and "hallucinations in hard-to-recognize pages." **olmOCR** budgets ~1024 px / ~1000 image tokens; exceeding it here is the *right* call because Nutanix pages are screenshot- and small-CLI-dense — *but only if the server actually receives the pixels* (R2.4/R2.5). Server-side `min_pixels`/`max_pixels` exposure is documented in **vLLM PR #9612 / issue #13099** (SGLang has the same flag).

---

## Dimension 3 — Model & Prompt Strategy

**Requirement.**

- **R3.1 Zero-shot instruction, no image exemplars.** The prompt MUST be zero-shot-instructed. If format drift appears, add a **text-only** micro-example (one 2×2 GFM table + one fenced command) — never an image exemplar (they cost ~1 k tokens and invite copying exemplar content).
- **R3.2 Metadata-first structure.** The model MUST emit page metadata first (rotation, language, table-presence, *is-this-a-continuation*), then the reading-order body.
- **R3.3 Document anchoring (headline requirement).** The prompt MUST inject **this page's** born-digital text-layer lines (capped ~1–2 k tokens, optionally with block bboxes) as a **reference**, framed: *"Transcribe the image. Use this text only to disambiguate glyphs; do not invent text not present here."* Anchor is reference, **not** ground truth (born-digital reading order can be imperfect).
- **R3.4 Uncertainty-aware abstention.** Illegible text MUST be omitted / rendered as a space, never guessed. The prompt MUST forbid "helpful completion" of truncated or ambiguous content.
- **R3.5 Tables → top-level GFM, merged cells flattened** by repeating the spanned value. Raw-HTML tables are forbidden. Complex spanning tables MUST be flagged for the Dim-7 overlap check.
- **R3.6 Code → fence everything.** Every command (including lone / reference-list commands) MUST be fenced, with a language hint, top-level (never nested under a list item), fences always closed. This is backed by a deterministic Stage-4 pass (R3.9).
- **R3.7 Figures → enriched captions** (default for this corpus). Each caption MUST (a) open with a stable subject noun-phrase ("Prism Element VM dashboard showing…"), (b) transcribe **verbatim legible on-screen text** (menu paths, field/column/button/tab/dialog labels, concrete values) and named diagram components/relationships, (c) invent nothing not legible, (d) emit as top-level `*[Figure: …]*` text (survives the parser). Screenshot text NOT present in the anchor MUST NOT be invented.
- **R3.8 Cross-page continuity — structure-aware tail + authoritative stitch.** The `prev_tail` passed to the model MUST be computed to the **last open block boundary** (carrying an unbalanced fence count, open table pipe-rows, or a dangling list) — not a fixed char count. An explicit CONTINUATION rule forbids re-emitting a table header or re-opening a fence. But the model's continuation is best-effort: **Stage 3 is the authority** for header re-attachment + fragment merge + seam dedup; **Stage 4 fence-balancing is the final guarantor**.
- **R3.9 Deterministic bare-CLI fencing pass (Stage 4).** A regex pass MUST fence unfenced command-looking lines (`^\s*(nutanix@|<acropolis>|ncli |acli |ncli>|\$ )…`) — a model-independent fix for cheat-sheet pages.
- **R3.10 Sampler.** MUST NOT run bare `temperature=0` with no penalties. MUST add mild `frequency_penalty ≈ 0.1–0.3`; SHOULD consider `temperature 0.1–0.2` + `top_p ≈ 0.9`. MUST NOT use `no_repeat_ngram_size` (or set ≥20). Reasoning tokens SHOULD be suppressed for the transcription pass.

**Rationale.** Anchoring (R3.3) is the single strongest anti-hallucination lever: **olmOCR** reports document anchoring yields "significantly fewer hallucinations," because image-only prompting "was prone to models completing unfinished sentences or inventing texts when the image was ambiguous" — and general VLMs are the *class most prone* to omit/rewrite/complete (olmOCR documents this for GPT-4o; **"Seeing is Believing?"** arXiv:2506.20168 shows MLLMs fall back on language priors under visual degradation). The corpus is born-digital → clean text layer → anchoring is nearly free and directly supplies exact command bytes (attacks bare-command emission) and reduces reliance on determinism. R3.4's abstention posture is the explicit recommendation of 2506.20168 (unrecognizable text "should not be included… to prevent any hallucination"). R3.5 diverges from benchmark convention deliberately: HTML tables score *higher* (OmniDocBench scores tables as HTML for TEDS; **LightOnOCR** reports HTML tables lift scores) *because HTML expresses rowspan/colspan and GFM structurally cannot* — but the downstream parser **drops raw HTML**, so GFM+flatten is the contract-dictated correct choice. R3.10 follows from **Holtzman** (arXiv:1904.09751): maximization (τ=0 is its extreme) provably degenerates into repetition because repeated-phrase probability rises with each repetition; olmOCR found raising τ 0.1→0.8 reduces repetitions; and since FP8+MTP already breaks bit-exact determinism at τ=0, the determinism argument for τ=0 is void. Mild penalties (not hard n-gram bans) preserve legitimately-repeating tokens in tables/CLI flags. R3.7 follows multimodal-RAG guidance: a screenshot caption's retrieval value is its verbatim on-screen text + a retrievable subject phrase, grounded strictly in what is legible. **Qwen3-VL** (arXiv:2511.21631) itself treats repetition as *suppressed-not-solved* (post-trained with high-frequency penalties), reinforcing R3.10.

---

## Dimension 4 — Model / Endpoint Choice

**Requirement.**

- **R4.1 A large general VLM (Qwen3.6-27B-FP8 class) is the correct PRIMARY transcriber IFF ALL hold:** (i) corpus is hundreds–low-thousands of pages (throughput irrelevant); (ii) screenshot/figure-dense, needing **semantic figure enrichment** (a VLM reasoning task specialists cannot do — they *box* figures, they don't *describe* them); (iii) a bespoke downstream contract that must be **emitted directly** (steerability — the only approach that emits exactly ATX + top-level GFM + fenced code + single H1, avoiding the HTML/OTSL/DocTags/layout-JSON that OmniDocBench-winning specialists emit and the parser drops); (iv) CLI-vs-prose judgment + cross-page seam reasoning; (v) one-model operational simplicity. **This corpus satisfies all five → the choice is sound, conditional on Dim-3 anchoring + repetition/determinism hardening.**
- **R4.2** The choice MUST NOT be justified on OCR-accuracy-per-dollar. You are buying figure understanding + steerability + simplicity, **not** better OCR.
- **R4.3 Text-layer arbitration MUST always augment the primary.** Promote born-digital text-overlap from *advisory* to an **active arbiter**: when VLM output diverges from the anchor beyond threshold, flag/re-run rather than merely advise (catches hallucination AND dropped/garbled commands).
- **R4.4 A specialist SHOULD augment as a targeted QA oracle / conditional fallback** — run a small vLLM-servable specialist (**PaddleOCR-VL** 0.9B or **MinerU2.5** 1.2B) on QA-flagged pages and **diff**; optionally route pure dense-table or command-cheat-sheet page *bodies* to it while the general VLM keeps figure/prose pages. This is a targeted two-model ensemble, not a wholesale switch. **GOT-OCR2.0** MAY give a verbatim second opinion on pure CLI strings.
- **R4.5** Any specialist added MUST be followed by an **HTML→GFM + lift-nested-tables normalization pass** (specialists emit HTML tables / layout-JSON the parser drops).
- **R4.6 Endpoint topology:** OpenAI-compatible endpoints; model id **auto-probed from `/v1/models`, never hardcoded**; dual-endpoint with a **pinned-stable vLLM default** (reproducible builds) + optional **dev-nightly SGLang** for ~2× throughput; both FP8, native dynamic resolution, ≥256 K context (headroom for anchor + prev-tail).
- **R4.7 Cost/size hygiene:** SHOULD A/B the 27B against an **8B-class** general-VLM sibling — most of the figure-reasoning + steering benefit is likely available far cheaper.
- **R4.8 Flip conditions (MUST be documented):** demote the 27B to a *figure-only* specialist (with PaddleOCR-VL / MinerU2.5 / LightOnOCR as text/table primary) if the corpus scales to 100 k+ pages, or pages become predominantly dense-table/plain-text with few figures, or cost/throughput start to dominate.

**Rationale.** On pure transcription, specialists win: OmniDocBench places **MinerU2.5** (1.2B) at 90.67 and **PaddleOCR-VL** (0.9B) at 92.56 — *above* **Qwen3-VL-235B** (89.15) and **dots.ocr** (88.41); a 27–32B general VLM lands below its own 235B. So a general VLM is strictly the wrong tool *for OCR-per-parameter* and is justified only by R4.1(ii)–(v). The specialists that "win" natively emit HTML/OTSL tables, layout-JSON (dots.ocr, MinerU2.5), or DocTags (**SmolDocling**) — precisely the forms the parser drops — so adopting one as primary adds a lossy converter and a new silent-drop class; a prompted general VLM is the *only* approach steerable to the exact flavor (R4.1(iii)). Hallucination on ambiguous CLI flags/values (2506.20168) is the #1 risk feeding GraphRAG, which R4.3's arbiter and R4.4's specialist diff retire where they bite hardest. **olmOCR-2** (Qwen2.5-VL-7B, arXiv:2510.19817) is the most drop-in fallback transcriber if determinism/robustness is ever needed (same stack, Markdown-native, anchoring built-in). R4.7 follows from the leaderboard: since a 235B flagship only reaches ~89, the figure/steering benefit likely survives heavy downsizing.

---

## Dimension 5 — Concurrency & Throughput

**Requirement.**

- **R5.1** A **single shared work-queue of page-tasks drained by per-endpoint worker pools** → least-outstanding-requests routing + free cross-endpoint failover.
- **R5.2** Per-endpoint `inflight` MUST be set a shade above the server's `max_running_requests` (e.g., `max_running=4` → `inflight=6`) to keep the continuous batch full without wasteful deep queuing; raise `inflight` only when GPUs / `max_running` rise.
- **R5.3 Page-level parallelism only.** Pages are independent requests; cross-page continuity is handled by prev-tail context + deterministic stitch (Dim 3), NOT by serializing transcription.
- **R5.4 Throughput is an explicit non-goal.** The design MUST NOT trade steerability, model size, or anchoring for pages/sec it does not need. Bounded retries with backoff (e.g., `max_retries≈4`, `timeout≈240 s` for long dense-table pages at `max_tokens≈6–8 k`); a page that exhausts retries is recorded **failed (never silently dropped)** and picked up by `retry-failed`.
- **R5.5** Reasoning-token suppression MUST be enabled for the transcription pass (latency/cost with no transcription benefit).

**Rationale.** olmOCR ships exactly this serving pattern (one page/request, batched on vLLM/SGLang; L40S ~906 tok/s, H100 ~3050 tok/s, ~$178/M pages) — throughput matters only at scale, and the 27B is 1–2 orders of magnitude slower per GPU-hour than a 1–3B specialist, which is *irrelevant for 936 one-time pages* (Qwen-fitness analysis). olmOCR and Nougat both process pages independently (R5.3). R5.4's fail-loud discipline matches olmOCR's ~12% production retry behavior — failures are re-enqueued, never dropped.

---

## Dimension 6 — Resumability & Caching

**Requirement.**

- **R6.1 Manifest-backed idempotency.** The Stage-0 JSONL manifest (per-PDF sha256, page count, embedded metadata) is the resume backbone.
- **R6.2 Content-addressed per-page cache** keyed by the **full determinant tuple**: `(pdf_sha256, page_no, pixel_budget/dpi, model_id, prompt_version, anchor_version)`, e.g. `<work>/pages/<sha256>/<render+model key>/NNNNN.md`. A re-run converts only missing/failed pages for that exact key; finished pages are instant cache hits; Ctrl-C loses nothing. Bumping the pixel budget, model, **or prompt/anchor version** MUST be a natural cache miss.
- **R6.3 Offline by default.** Rendering, anchor extraction, stitch, clean, validate, and `status` MUST run fully offline; only `convert`/`retry-failed`/`inspect`/`doctor` touch endpoints.
- **R6.4 `retry-failed`** re-attempts only failed/missing pages (optionally with an escalated pixel budget + nudged sampler per Dim 3), then re-assembles.
- **R6.5 Determinism caveat.** Because FP8+MTP is non-bit-exact at τ=0, the cache MUST be keyed on **inputs**, and re-run equivalence tracked as **tolerance-based semantic drift, never byte-identity.**

**Rationale.** This is the olmOCR/production resumability model. R6.2's inclusion of `prompt_version` and `anchor_version` in the key is essential once anchoring (Dim 3) and prompt iteration exist — otherwise a prompt change silently serves stale cache. R6.5 is forced by the known FP8+MTP nondeterminism (Qwen-fitness); the rag-and-eval research is explicit that a build must **never gate on byte-identical reproducibility**.

---

## Dimension 7 — Validation / QA Gates

**Requirement — tiered, mostly reference-free, fail-closed on the contract.** The rasterized PNG is the *true* ground truth (an image-grounded judge is therefore reference-based without gold Markdown); the born-digital text layer is a free noisy partial reference.

- **R7.0 Tier 0 — Contract conformance (HARD GATE; deterministic; blocks a doc).** MUST assert on the final `.md`: frontmatter parseable at byte 0 with non-empty `title`; **exactly one H1**; heading tree monotone (no level jump >1); **no signal-bearing content before the first heading**; **every fenced code block and GFM table top-level** (not list-nested, not in raw-HTML); balanced fences; no stray raw-HTML table/code. A violation = silent downstream data loss → **fail closed**.
- **R7.1 Tier 1 — Structural self-consistency (SOFT/quarantine; deterministic; no reference).** Every GFM table rectangular + round-trips (mini **TEDS-Struct against a re-parse**); Markdown AST round-trip stable (parse→serialize→parse invariant); repetition/degenerate-decode guard; command-density-without-fence advisory.
- **R7.2 Tier 2 — Text-layer + property tests (SOFT; deterministic; free partial reference).** `token_overlap` promoted to a tracked per-page metric with a **soft floor on text-heavy pages only** (guarded by text-layer length so screenshot pages don't trip it — "long text layer but very low overlap" = the missing-content/hallucination signature); an **output ≫ text-layer length-ratio** upper bound (fabrication/loop guard); a reading-order proxy (order-agreement of shared tokens); **olmOCR-style Absence tests** (the page-1 WeasyPrint disclaimer, running headers/footers, page numbers MUST be gone) and **Presence tests** (anchor phrases survived).
- **R7.3 Tier 3 — Sampled image-grounded LLM judge (ADVISORY; never a gate).** A stratified sample of table-/CLI-/screenshot-dense pages scored by a **multimodal judge against the page PNG** for faithfulness/completeness/structure + table round-trip, using **two differently-worded prompts**; report score + agreement.
- **R7.4 Layered repetition/hallucination detection** (beyond line-based): `finish_reason=="length"` / max_tokens hit; **Nougat logit-variance** (sliding window B=15, threshold ~6.75) when logprobs are available; cheap text heuristics — **zlib/gzip compression ratio** (degenerate text compresses ~5–10× more), char/token n-gram repetition ratio, max single-line length, and intra-line phrase loops (whole-line detectors miss `the the the…`).
- **R7.5 Gate policy.** Tier 0 hard-fail; Tiers 1–2 quarantine for review (summarized in the report); Tier 3 telemetry only. MUST NOT gate on byte-identical reproducibility; track cross-endpoint / re-run **semantic drift** as advisory.

**Rationale.** The tier design and philosophy are **olmOCR-bench** (arXiv:2510.19817): machine-checkable binary Presence/Absence/Reading-Order/Table/Math/Baseline-Robustness tests, chosen because edit distance and LLM-judges "reward/penalize in a manner that doesn't correlate with practical correctness" and treat equivalent representations differently. Fail-closed Tier 0 is dictated by the contract: raw-HTML and list-nested blocks are silently dropped downstream, so these are *data-loss* faults. R7.4's mechanisms trace to **Holtzman** (loop cause), **Nougat**/**LOCR** (arXiv:2403.02127 — repetition from lost visual grounding; logit-variance detector), and olmOCR (loops caught at `finish_reason`). The image-grounded judge (R7.3) is the strongest *semantic* signal but carries position/self-preference/prompt-sensitivity bias (**Wang** arXiv:2304.00723; **Wolfe**; **Margin-Adaptive Confidence Ranking** arXiv:2605.15416), hence two-prompt agreement and never-a-gate.

---

## Dimension 8 — Evaluation Harness & Metrics

**Requirement.** (Distinct from Dim 7: Dim 7 is inline gating; Dim 8 is corpus-level measurement + regression.)

- **R8.1 Adopt olmOCR-bench methodology as the headline metric:** a corpus-wide suite of machine-checkable binary unit-tests — Text-Presence (known commands/anchor phrases), Text-Absence (WeasyPrint disclaimer, headers/footers, page numbers), Natural-Reading-Order (span A before span B), Table-Accuracy (target cell value + relative position), Math (KaTeX-render equivalence), Baseline-Robustness (no long repeated n-grams / off-language runs). **Pass-fraction (0–1) is the headline number** and doubles as an RL reward if ever fine-tuning.
- **R8.2 Report by content type AND page attribute, never one global score:** text **NED/CER**; table **TEDS / TEDS-Struct** (and **GriTS** for multi-hop cell misalignment); reading-order NED over text blocks; formula **CDM** (robust vs BLEU/edit). Break down by table-dense / CLI-dense / screenshot-dense / small-text attributes.
- **R8.3 Reference-free where no gold Markdown exists** (this corpus): text-layer overlap (CER-adjacent + reading-order proxy), structural self-consistency, property unit-tests, and the sampled image-grounded judge (PNG = ground truth → reference-based without gold).
- **R8.4 Small anchored gold set:** hand-correct **~30–50 stratified pages** for reference-based NED/TEDS/CDM calibration of the reference-free signals — but the *corpus* gate stays reference-free.
- **R8.5 Regression discipline:** run the unit-test suite on every rebuild; track pass-fraction, Tier-0 pass rate, quarantine list + reasons, mean/min text-layer recall on text-heavy pages, table-validity rate, and boilerplate-leak count over time. Weight corpus-specific unit-tests **above** generic benchmark scores.
- **R8.6 Drift:** measure semantic (not byte) drift between endpoints and across rebuilds as advisory.

**Rationale.** olmOCR-bench's binary-property design lets "different-yet-equivalently-correct" outputs score equally and is verifiable — the single most transferable idea for a reference-free corpus. R8.2's per-type/attribute discipline and the NED/TEDS/CDM/mAP metric families are **OmniDocBench**'s method (with **TEDS**/PubTabNet arXiv:1911.10683, **GriTS**, and **CDM** as the robust formula metric). R8.5's "weight corpus tests above generic benchmarks" heeds the **LlamaIndex "OmniDocBench is saturated"** caution (top scores 90–96 are saturating; the field is moving to harder suites), so a bespoke unit-test suite over *this* corpus is more informative than a leaderboard number.

---

## Dimension 9 — Output Artifact Contract

**Requirement.** The on-disk contract is the **only** coupling to the GraphRAG consumer and is treated as sacred.

- **R9.1** One `.md` per document, UTF-8, at `<out>/<slug>/<slug>.md` (per-doc; consumer globs `*.md` recursively).
- **R9.2 YAML frontmatter at byte 0.** Parser-consumed: `title`, `version`, `last_edited`. Provenance (inert to parser, load-bearing for citations + GraphRAG entity disambiguation): `product`, `source_pdf`, `sha256`, `page_count`, `dpi`, `model_id`, `endpoint`, `pipeline_version`, `extracted_at`.
- **R9.3 Title precedence:** document's own first H1 (faithful) → embedded PDF title → humanized filename; the body starts with that **single H1** so nothing is lost to the parser's "content-before-first-heading is dropped" rule.
- **R9.4 Headings = hard chunk boundaries → a real ATX tree that lands on true semantic sections.**
- **R9.5 Fenced code (with language hint) + GFM tables kept top-level** — never nested in a list, never raw HTML (both silently dropped); tables rectangular/atomic (header kept with the table); fences balanced.
- **R9.6 Clean of furniture:** no running headers/footers, page numbers, WeasyPrint disclaimer, or hyphenation breaks.
- **R9.7 Canonical terminology:** consistent product names + CLI/nCLI/aCLI spellings so logically identical entities collapse to one node/edge.
- **R9.8 Self-contained sections:** running per-page prose with prev-page continuity → sections read standalone (minimal dangling anaphora).
- **R9.9 Machine-readable run report/manifest** alongside the corpus (pages converted/cached/failed, per-endpoint stats, QA-suspect pages, quarantine list + reasons; `--json`).

**Rationale.** The contract is verified against the downstream `markdown_it_parser.py` / `atomic.py`; R9.4 also *maximizes GraphRAG quality*: **GraphRAG "From Local to Global"** (arXiv:2404.16130) extracted ~**2× the entities from 600-token vs 2400-token chunks**, so structure-aligned (smaller, self-contained) sections yield a denser graph; heading paths double as free per-chunk context à la **Anthropic Contextual Retrieval** (50–100-token context cut failed retrievals up to 49%); structure-aware chunking beats fixed-size on formal docs (**ACM AI 2025**). R9.6's furniture-stripping is **GraphRAG-critical, not cosmetic** — boilerplate "distorts how entities and relationships are detected," fragmenting the graph (**memgraph**, **PremAI**). R9.2/R9.7 attack GraphRAG's core failure, entity linking: provenance enables citation-grounded RAG (**Citation-Enforced RAG** arXiv:2603.14170) and the `version` field distinguishes same-named Nutanix entities across releases (**ideasthesia** GraphRAG lessons). R9.8 follows from the rag-and-eval finding that self-contained sections yield cleaner extraction and higher-quality relationships.

---

### Cross-cutting acceptance criteria (definition of "good")

A build is corpus-ready when: **(A)** Tier-0 contract pass rate = 100% (fail-closed); **(B)** olmOCR-bench-style unit-test pass-fraction meets target, broken down by content type/attribute (Dim 8); **(C)** zero boilerplate leaks (WeasyPrint disclaimer / headers / footers / page numbers absent); **(D)** table-validity rate and text-heavy-page text-layer recall above their floors, with quarantines triaged; **(E)** every page either transcribed or explicitly recorded failed — none silently dropped. The through-line across all nine dimensions: **clean, top-level, heading-structured Markdown with provenance and canonical naming, produced by a steerable large general VLM that is anchored to the born-digital text layer and guarded by reference-free unit-test gates** — docpipe's downstream contract and GraphRAG's quality needs point the same direction.

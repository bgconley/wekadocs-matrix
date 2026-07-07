# docpipe Design Review — Research Appendix

_Web + arXiv research (2023–2026) that grounds the ideal spec. Four independent agents; citations inline._



---

## Appendix 1: SOTA VLM/OCR Document-Extraction Systems & Benchmarks

# State of the Art: PDF → Markdown via VLM / OCR-Free Models (2023–2026)

## 1. Framing — three paradigms

Document-to-Markdown systems in this era fall into three architectural families, and OmniDocBench (the field's reference benchmark) explicitly groups models the same way:

1. **Layout-detection + OCR pipelines** ("modular"): a detector segments the page into blocks (RT-DETR / YOLO / Surya), specialist models read each block (OCR engine, table model, formula model), and a reading-order model linearizes. Examples: **Docling**, **Marker**, classic **MinerU 1.x**.
2. **OCR-free full-page transformers / doc-specialized VLMs**: a single image-to-sequence model transcribes the whole page (or crops) directly to markup. Early: **Nougat**, **GOT-OCR2.0**. Current frontier: **olmOCR**, **MinerU2.5**, **dots.ocr**, **PaddleOCR-VL**, **DeepSeek-OCR**, **SmolDocling**.
3. **General-purpose VLMs** prompted for transcription: **Qwen2.5-VL / Qwen3-VL**, **InternVL3/3.5**, **DeepSeek-VL**, GPT-4o, Gemini. Not doc-specialized, but strong and maximally promptable.

**docpipe sits in paradigm 2/3**: it uses a *large general VLM* (paradigm 3) in *full-page transcription* mode (paradigm 2's operating pattern). Its single closest published analog is **olmOCR**, which runs a Qwen VLM on vLLM/SGLang, one page per request, emitting markdown — the same design as docpipe.

A caveat worth stating: OmniDocBench uses two score conventions. **v1.0** reports *normalized edit distance, lower = better* (~0.1–0.5). **v1.5/v1.6** report an *overall score /100, higher = better* (~80–96). olmOCR-bench reports *unit-test pass rate /100, higher = better*. I flag direction with ↑/↓ throughout and never mix scales in one column.

---

## 2. The systems

### Early OCR-free transformers (now superseded, but foundational)

**Nougat** — *Nougat: Neural Optical Understanding for Academic Documents*, arXiv:2308.13418 (Meta, Aug 2023). [abs](https://arxiv.org/abs/2308.13418) · [github](https://github.com/facebookresearch/nougat)
- **Method**: pure OCR-free. Swin-Transformer image encoder → mBART-style autoregressive decoder emits a Markdown/LaTeX-like markup. Image-only input; no PDF text layer, no layout model.
- **Strengths**: first credible end-to-end scientific-PDF → markup model; excellent inline/display math; simple single-model deployment.
- **Weaknesses**: trained almost exclusively on arXiv/PMC → poor domain transfer; **degenerate repetition loops** (autoregressive hallucination) on out-of-distribution pages; weak tables; essentially English/Latin only.
- **Accuracy**: OmniDocBench v1.0 overall edit **EN 0.464 / ZH 0.973** (worst of the field; ZH ≈ non-functional). Superseded.

**GOT-OCR2.0** — *General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model*, arXiv:2409.01704 (Sep 2024). [abs](https://arxiv.org/abs/2409.01704) · [github](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- **Method**: 580M-param unified encoder–decoder (high-compression vision encoder + long-context decoder). Treats text, math, tables, molecular/chemical formulas, sheet music, charts, geometry as "characters." Outputs plain text or markdown/TikZ/SMILES via a prompt. Slice + whole-page input.
- **Strengths**: tiny, elegant, genuinely verbatim; strong formula/table *content*; interactive/region OCR.
- **Weaknesses**: no explicit layout/reading-order module → struggles on complex multi-element or multi-column full pages; limited robustness on messy real-world scans.
- **Accuracy**: OmniDocBench v1.0 overall edit **EN 0.302 / ZH 0.429** (mid-tier for its era). DeepSeek-OCR later beats GOT with ~100 vision tokens vs GOT's 256. Superseded on rankings but still a reference for compact verbatim OCR.

### Current doc-specialized VLMs (the 2025 frontier)

**olmOCR / olmOCR 2** — *olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models*, arXiv:2502.18443 (AI2, Feb 2025); *olmOCR 2: Unit Test Rewards for Document OCR*, arXiv:2510.19817. [abs](https://arxiv.org/abs/2502.18443) · [olmOCR2 blog](https://allenai.org/blog/olmocr-2) · [github](https://github.com/allenai/olmocr)
- **Method**: fine-tuned **Qwen2-VL-7B** (olmOCR 2 = **Qwen2.5-VL-7B**). Full-page image → markdown. Signature trick = **"document anchoring"**: extracts the born-digital PDF's text blocks + element coordinates and injects them into the prompt alongside the image, anchoring the transcription to ground-truth text and cutting hallucination. Prompt asks for page metadata (rotation, language, table-presence) *before* the reading-order text. **olmOCR 2** adds RL (**GRPO**) with *unit-test rewards* — reward = fraction of programmatic checks passed (table structure, equation correctness), 28 candidates/doc.
- **Deployment (highly relevant to docpipe)**: ships an inference pipeline that **supports both vLLM and SGLang**, one page/request, batched. L40S ≈ 906 tok/s; H100 ≈ 3,050 tok/s; **~$178 / million pages** (vs GPT-4o batch ~$6,240/M). ~12% retry rate in production.
- **Strengths**: purpose-built for exactly docpipe's job; anchoring dramatically improves faithfulness on born-digital PDFs; open weights + data.
- **Weaknesses**: anchoring needs a text layer (image-only PDFs fall back to pure vision); 7B is less "smart" than a 27B general model on ambiguous layout/code-fence decisions.
- **Accuracy**: olmOCR-bench **82.4 ↑** (v0.4.0); OmniDocBench v1.5 overall **81.79 ↑** (7B, respectable but below the 0.9–3B Chinese specialists that over-fit OmniDocBench).

**MinerU2.5 / MinerU2.5-Pro** — *MinerU2.5: A Decoupled VLM for Efficient High-Resolution Document Parsing*, arXiv:2509.22186; classic *MinerU*, arXiv:2409.18839; *MinerU2.5-Pro*, arXiv:2604.04771. [abs](https://arxiv.org/abs/2509.22186) · [github](https://github.com/opendatalab/MinerU)
- **Method**: 1.2B **decoupled two-stage VLM**. Stage 1: layout analysis on a *downsampled* page (cheap). Stage 2: content recognition on *native-resolution crops* guided by that layout — preserving dense text/formula/table detail without paying full-res cost on the whole page. (Classic MinerU 1.x was a modular PDF-Extract-Kit pipeline; 2.5 is a single VLM.)
- **Strengths**: SOTA-tier accuracy at tiny size and low VRAM; excellent tables/formulas; vLLM-servable; MD + JSON output.
- **Weaknesses**: emits its own structured layout/format (crops, HTML tables) → needs a converter to hit an arbitrary downstream markdown flavor; two-stage adds orchestration.
- **Accuracy**: OmniDocBench v1.5 overall **90.67 ↑**, Formula-CDM 88.46, Table-TEDS 88.22; **MinerU2.5-Pro tops OmniDocBench v1.6 at 95.75 ↑**. olmOCR-bench 75.2–77.5 ↑.

**dots.ocr** — *Multilingual Document Layout Parsing in a Single VLM* (rednote-hilab, released 2025-07-30). [hf](https://huggingface.co/rednote-hilab/dots.ocr) · [github](https://github.com/rednote-hilab/dots.ocr)
- **Method**: single ~**1.7B-LLM** VLM (~3B total) that **unifies layout detection + content recognition + reading order** in one model; emits layout JSON with bboxes + text/markdown. Prompt-switchable tasks.
- **Strengths**: SOTA text + reading order on OmniDocBench; ~100-language coverage incl. low-resource; converts graphics to SVG; compact/fast.
- **Weaknesses**: **complex tables and formulas remain hard** given the compact backbone (their own README); SVG/figure parsing not yet robust; occasional edge-case failures; JSON-bbox output needs conversion to clean markdown.
- **Accuracy**: OmniDocBench v1.5 overall **88.41 ↑** (text-edit 0.048, reading-order 0.053); olmOCR-bench **79.1 ↑**. (Successor "dots.mocr" reports text-edit 0.031 / reading-order 0.029.)

**PaddleOCR-VL** — *PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact VLM*, arXiv:2510.14528 (Baidu, Oct 2025). [abs](https://arxiv.org/abs/2510.14528) · [ernie blog](https://ernie.baidu.com/blog/posts/paddleocr-vl/)
- **Method**: 0.9B VLM = **NaViT-style native-resolution dynamic encoder + ERNIE-4.5-0.3B** decoder. Native-res avoids resize distortion in text-dense regions. Two-stage (layout → element recognition). 109 languages.
- **Strengths**: **current best-in-class small model** — leads OmniDocBench v1.5 *and* olmOCR-bench simultaneously at 0.9B; strong formula/table; fast (1.22 pages/s on A100); vLLM-servable.
- **Weaknesses**: structured multi-stage output → conversion glue for bespoke markdown contracts; Chinese-lab tuning may over-index on OmniDocBench-style pages.
- **Accuracy**: OmniDocBench v1.5 overall **92.56 ↑** (#1 among sub-3B), text-edit 0.035, Formula-CDM 91.43, Table-TEDS 89.76, reading-order 0.043; olmOCR-bench **80.0 ↑**; in-house formula CDM 0.988.

**DeepSeek-OCR** — *DeepSeek-OCR: Contexts Optical Compression*, arXiv:2510.18234 (Oct 2025). [abs](https://arxiv.org/abs/2510.18234) · [github](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Method**: **DeepEncoder** (high-compression vision encoder, low activations at high res) + **DeepSeek3B-MoE (A570M active)** decoder. Framed as *optical context compression*: text rendered as image → few vision tokens. <10× compression → **~97% decode precision**; 20× → ~60%.
- **Strengths**: extreme token efficiency (beats GOT-OCR at 100 vision tokens; beats MinerU2.0 with <800 vs 6000+ tokens); huge throughput (200k+ pages/day on one A100-40G); MoE = cheap active compute.
- **Weaknesses**: research-forward; markdown fidelity secondary to the compression thesis; MoE serving is more finicky.
- **Accuracy**: olmOCR-bench **75.7 ↑**; competitive on OmniDocBench at a fraction of the tokens.

**SmolDocling** — *SmolDocling: An ultra-compact VLM for end-to-end multi-modal document conversion*, arXiv:2503.11576 (IBM + HF, Mar 2025). [abs](https://arxiv.org/abs/2503.11576) · [hf](https://huggingface.co/ds4sd/SmolDocling-256M-preview)
- **Method**: **256M** SmolVLM fine-tune emitting **DocTags** — a location-aware universal markup (not markdown) capturing element type + bbox + content, losslessly convertible to MD/HTML. Feeds the Docling ecosystem (now continued as **Granite-Docling-258M**).
- **Strengths**: astonishing size/quality ratio (0.35 s/page, **0.489 GB VRAM**); reproduces code, tables, equations, charts; competes with models up to 27× larger.
- **Weaknesses**: preview-grade robustness; DocTags → markdown conversion step; small model → more transcription errors on hard pages than 1–7B specialists.
- **Accuracy**: strong per-parameter but not at the 90+ OmniDocBench frontier; positioned as an efficiency artifact.

### Layout-detection + OCR pipelines

**Docling** — *Docling Technical Report*, arXiv:2408.09869; *Docling: An Efficient Open-Source Toolkit…*, arXiv:2501.17887 (IBM). [abs](https://arxiv.org/abs/2408.09869) · [github](https://github.com/docling-project/docling)
- **Method**: modular, MIT-licensed. **RT-DETR layout model (DocLayNet)** classifies blocks → **TableFormer** recovers table structure (borderless, spanning, hierarchical headers) → OCR/text extraction → reading order → a `DoclingDocument` exported to MD/JSON/HTML. Optionally uses SmolDocling/Granite-Docling VLM.
- **Strengths**: runs on commodity CPU/GPU in a small budget; **excellent structured table recovery**; deep ecosystem integration (LangChain, LlamaIndex); very controllable; local, no LLM required.
- **Weaknesses**: multi-model PyTorch pipeline — **not a single OpenAI-compatible endpoint**, doesn't fit a vLLM/SGLang serving model; **tables export as HTML/OTSL** (a problem for markdown-only downstreams); accuracy trails frontier VLMs on messy pages.
- **Accuracy**: mid-tier on OmniDocBench (strong tables, weaker on complex/dense mixed pages); its VLM path = SmolDocling-class.

**Marker** — VikParuchuri / datalab-to (no arXiv; active OSS). [github](https://github.com/datalab-to/marker)
- **Method**: pipeline of **Surya** models (detection, layout, reading-order, OCR) + **Texify** (math) + heuristics; **optional `--use_llm`** hybrid that calls an external LLM (Gemini/Claude/OpenAI/Ollama/**any OpenAI-compatible**) to merge cross-page tables, fix inline math, extract form fields. Outputs MD/JSON/HTML/RAG-chunks.
- **Strengths**: fast (0.18 s/page; ~122 pages/s projected on H100 at 3.17 GB), accurate, flexible outputs; `--use_llm` lifts hard cases; strong self-reported vs Llamaparse/Mathpix/Docling.
- **Weaknesses**: complex nested layouts/forms/tables can fail without `--use_llm`; local multi-model pipeline (not a single served VLM); its LLM mode is an *augmentation*, not the primary transcriber.
- **Accuracy**: olmOCR-bench **76.1 ↑**; OmniDocBench v1.6 overall **78.44 ↑** / text-edit 0.157; FinTabNet tables 0.816 → **0.907 with `--use_llm`**; v1.0-era overall edit EN 0.416 / ZH 0.560.

### General-purpose VLMs used for doc parsing

**Qwen2.5-VL** — arXiv:2502.13923; **Qwen3-VL** — arXiv:2511.21631 (Alibaba). [Qwen2.5-VL abs](https://arxiv.org/abs/2502.13923) · [Qwen3-VL abs](https://arxiv.org/abs/2511.21631) · [github](https://github.com/QwenLM/Qwen3-VL)
- **Method**: general multimodal LLMs (dense + MoE variants) with strong native document parsing, grounding, and long-context. **Qwen3-VL**: OCR in 39 languages (>70% acc on 32), robust to low-light/blur/tilt, improved long-document structure parsing, 30M in-house OCR training samples. No layout module — you *prompt* for markdown.
- **Strengths**: maximally **promptable/steerable** (can be told the exact markdown contract), best "reasoning" about ambiguous layout/code, huge context, first-class **vLLM/SGLang** support, FP8. This is docpipe's family.
- **Weaknesses**: general VLMs are the class most prone to **omitting / rewriting / "helpfully completing" / captioning** content unfaithfully (documented for GPT-4o in the olmOCR paper — the very motivation for document anchoring); heavier/slower/pricier than 0.9–3B specialists; non-determinism under sampling/FP8.
- **Accuracy**: Qwen2.5-VL-72B OmniDocBench v1.5 overall **87.02 ↑** (text-edit 0.094); Qwen2-VL v1.0 overall edit EN 0.392 / ZH 0.408. Qwen3-VL substantially stronger (SOTA long-doc understanding, e.g. 57% MMLongBench-Doc).

**InternVL3 / InternVL3.5** — arXiv:2504.10479 / arXiv:2508.18265 (Shanghai AI Lab / OpenGVLab). [InternVL3 abs](https://arxiv.org/abs/2504.10479) · [InternVL3.5 abs](https://arxiv.org/abs/2508.18265)
- **Method**: general MLLM family; native multimodal pretraining (v3), Cascade-RL + big efficiency gains (v3.5, +16% reasoning, 4.05× speedup). Strong OCR/chart/doc understanding (OCRBench).
- **Strengths**: open, scalable to 241B-A28B, competitive general doc understanding.
- **Weaknesses**: same general-VLM faithfulness caveats; not doc-transcription-tuned by default.
- **Accuracy**: InternVL2 v1.0-era overall edit EN 0.457 / ZH 0.464; **InternVL3.5-241B** OmniDocBench v1.6 overall **83.76 ↑** — i.e., even a 241B general VLM trails a 1.2B *specialist* (MinerU2.5 93.04) on pure transcription.

**DeepSeek-VL / DeepSeek-VL2** (general VLM lineage behind DeepSeek-OCR): competent doc understanding but, like all general VLMs, superseded on *transcription* metrics by specialists; DeepSeek's doc-specific effort is DeepSeek-OCR above.

---

## 3. The benchmarks

**OmniDocBench** — arXiv:2412.07626, CVPR 2025 (OpenDataLab). [abs](https://arxiv.org/abs/2412.07626) · [github](https://github.com/opendatalab/OmniDocBench)
- 981 real PDF pages, 9 doc types (papers, textbooks, exams, financial reports, newspapers, magazines, handwritten notes, slides…), 4 layout types, EN/ZH. 28 block-level + 4 span-level annotations; tables in both LaTeX+HTML; explicit **reading-order** labels.
- Metrics: per-module **text (normalized edit distance ↓)**, **formula (CDM ↑ / edit ↓)**, **table (TEDS ↑)**, **reading order (edit ↓)**. Splits models into pipeline tools / expert (specialized) VLMs / general VLMs.
- Versions matter: v1.0 = edit-distance (↓); v1.5/v1.6 = 0–100 overall (↑). The top of the board (MinerU2.5-Pro 95.75, GLM-OCR 95.22, Gemini 3 Pro 92.91) is now **saturating** — LlamaIndex publicly argues [OmniDocBench is saturated](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks), pushing the field toward harder suites.

**olmOCR-bench** — in *olmOCR 2* arXiv:2510.19817 (AI2). [hf dataset](https://huggingface.co/datasets/allenai/olmOCR-bench) · [github](https://github.com/allenai/olmocr/tree/main/olmocr/bench)
- 1,403 PDFs, **7,010 machine-checkable unit tests** ("facts"): text-presence, text-absence (headers/footers/page-numbers must NOT appear), natural reading order of span pairs, table cell relationships, math. Categories: ArXiv math, old-scan math, tables, old scans, headers/footers, multi-column, long-tiny-text, base.
- Philosophy: avoids fuzzy full-page similarity; each test is unambiguous → also usable as an **RL reward** (olmOCR 2's unit-test GRPO). Current leaders: Chandra 83.1, **olmOCR v0.4.0 82.4**, Infinity-Parser 82.5, PaddleOCR-VL 80.0, dots.ocr 79.1, MinerU2.5 ~77, Marker 76.1, DeepSeek-OCR 75.7, Mistral-OCR 72.0.

**Table & formula sub-benchmarks**
- **Tables**: **TEDS / TEDS-S** (Tree-Edit-Distance Similarity, structure-only variant) on **PubTabNet** and **FinTabNet** (financial). Docling's TableFormer, MinerU2.5, PaddleOCR-VL lead; Marker reports FinTabNet 0.816→0.907 with LLM.
- **Formulas**: **CDM** (Character Detection Matching) — robust alternative to BLEU/edit for LaTeX; used in OmniDocBench and by PaddleOCR-VL (0.988 in-house), MinerU2.5 (0.977).

---

## 4. Comparison on the requested axes

Ratings are relative within the doc-parsing field. "Deployability" = fit for a **single OpenAI-compatible vLLM/SGLang endpoint**. "Contract fit" = alignment to docpipe's downstream markdown rules (top-level GFM tables + fenced code, ATX headings, single H1, frontmatter@byte0).

| Approach | Faithfulness (verbatim, low-hallucination) | Table fidelity | Reading order | Robustness (real-world/screenshot) | Deployable on vLLM/SGLang | Fit: single-column CLI-heavy + contract |
|---|---|---|---|---|---|---|
| **Large general VLM, full-page (docpipe today: ~27B Qwen3-family)** | Med-High — strong, but general VLMs are the class most prone to omit/rewrite/complete (olmOCR's GPT-4o finding); mitigate via prompt/anchoring | Med — good GFM; loses complex spanning cells | **High** — reasons about order; trivial for single-column | High — big model handles screenshots/CLI, but may hallucinate UI text | **Native ✓** (already running FP8 on oxcart/blackbird) | **Best** — only approach you can *steer* to emit exactly ATX + top-level GFM + fenced code + single H1 |
| **olmOCR / olmOCR2 (Qwen2/2.5-VL-7B + doc anchoring)** | **High** — anchoring to born-digital text is the SOTA anti-hallucination trick | Med-High | High | High | **Native ✓** (ships vLLM+SGLang, 1 page/req) | **Very good** — same pattern as docpipe; markdown-native; anchoring ideal for born-digital Nutanix PDFs |
| **MinerU2.5 / PaddleOCR-VL / dots.ocr (0.9–3B specialists)** | High (verbatim-tuned) | **High** (TEDS ~88–90) | **High** | High | Servable ✓ but custom output | Med — SOTA accuracy, but emit layout-JSON / **HTML tables** → conversion glue + contract-violation risk (HTML/nested dropped downstream) |
| **DeepSeek-OCR** | Med-High | Med-High | High | High (token-efficient) | Servable (MoE, fiddlier) | Med — research-forward; markdown fidelity secondary |
| **Docling (RT-DETR + TableFormer pipeline)** | High (deterministic) | **High** (best structured tables) | High | Med-High | **✗** multi-model PyTorch, not one endpoint | Low-Med — **HTML/OTSL tables**, not a served VLM; strong tables wasted on a table-light corpus |
| **Marker (Surya pipeline + optional `--use_llm`)** | Med (High with LLM) | Med → High w/ LLM | High | Med (fails hard layouts w/o LLM) | Partial — LLM mode can call oxcart, but core is local | Med — decent MD; could point `--use_llm` at your stack, but it's an augmenter not a transcriber |
| **GOT-OCR2.0** | High (verbatim) | Med | **Low-Med** (no layout module) | Low-Med (weak on complex pages) | Servable (HF, not vLLM-optimized) | Low — no reading-order/layout for mixed screenshot pages |
| **SmolDocling / Granite-Docling (256M)** | Med | Med-High | Med | Med (preview) | Servable (tiny) but DocTags output | Low-Med — DocTags→MD conversion; small-model errors on hard CLI pages |
| **Nougat** | Low-Med (repetition loops) | Low | Med | **Low** (arXiv-domain only) | ✗ (HF, slow) | **Poor** — wrong domain, no tables, hallucination loops |

---

## 5. Verdict for docpipe (`/Users/brennanconley/vibecode/wekadocs-matrix/docpipe`)

**Keep the architecture. Add document anchoring. Don't switch to a pipeline tool or a small specialist as the primary.**

**1. The current design is the right paradigm — and is externally validated.** docpipe's "render page → full-page transcription by a Qwen VLM on vLLM/SGLang, one page/request, markdown out, resumable cache" is *literally the olmOCR architecture* (olmOCR ships exactly vLLM+SGLang backends, one page/request). You've independently arrived at the field's reference design for born-digital PDF→markdown. Your two deltas vs olmOCR — a **much larger general model (~27B vs 7B)** and **previous-page-tail continuity instead of anchoring** — are deliberate trades: you buy maximal prompt-steerability and cross-page reasoning at the cost of throughput. For a **936-page, one-time corpus**, throughput is irrelevant, so paying 27B for steerability is the correct trade. Do not downsize to a 0.9–3B specialist to chase pages/sec you don't need.

**2. Steerability is your decisive advantage for the ingestion contract — protect it.** Your downstream parser has an *idiosyncratic* contract: ATX-only headings as hard boundaries, content-before-first-heading dropped, **fenced code + GFM tables must be top-level**, raw-HTML and list-nested blocks silently dropped, frontmatter at byte 0, single H1. The specialists that "win" OmniDocBench (MinerU2.5, PaddleOCR-VL, dots.ocr, Docling) natively emit **HTML/OTSL tables, layout-JSON, DocTags, or setext/bold-as-heading** — precisely the forms your parser *drops*. Adopting one would add a lossy converter and a new class of silent-drop bugs. A prompted general VLM is the *only* approach you can instruct to emit your exact markdown flavor directly. This is the single strongest reason **not** to migrate.

**3. Highest-value upgrade — olmOCR-style document anchoring.** Your biggest live risk is the general-VLM faithfulness failure mode (omit/rewrite/normalize), which the olmOCR paper documents specifically for large general VLMs (GPT-4o) and *solves* by injecting the born-digital PDF text layer + element coordinates into the prompt. Your Nutanix PDFs are **WeasyPrint-generated → born-digital with a clean, reliable text layer**, so anchoring is high-signal and nearly free. Concretely:
   - You already extract a text layer for the Stage-5 "text-layer overlap" advisory. **Promote that same text from a QA check to a Stage-2 prompt input** (per-page anchor text, optionally with bboxes).
   - This directly attacks **known-issue (a)** — dense CLI/cheat-sheet pages emitting *bare, possibly mangled* commands: the anchor gives the model the exact command bytes to fence, rather than re-deriving glyphs from a 200-DPI raster. It also hardens verbatim fidelity on screenshot-dense pages (reduces hallucinated UI text) and reduces reliance on temperature-0 determinism (mitigating **known-issue (b)**, FP8+MTP nondeterminism, since the output is anchored to fixed text).
   - Borrow olmOCR's **"emit page metadata first, then reading-order body"** prompt structure — cheap, improves rotation/table handling.

**4. Cheap wins you can layer in:**
   - **Reading order is a non-problem for you** (single-column corpus), so the specialists' main edge doesn't apply — another reason the simple full-page approach fits.
   - **Screenshots**: decide policy explicitly (faithful transcription vs. `[figure]` placeholder). A general VLM will "helpfully" transcribe UI text and can hallucinate; anchoring + an explicit prompt rule ("do not invent text inside screenshots not present in the anchor") curbs this.
   - **Use a specialist as a QA oracle, not a replacement**: run **PaddleOCR-VL (0.9B)** or **MinerU2.5 (1.2B)** — both cheap and vLLM-servable — on pages your Stage-5 flags as garbled, and diff. Adopt **olmOCR-bench-style unit tests** (text-presence for known commands, text-absence for the page-1 WeasyPrint disclaimer — **known-issue (c)** — header/footer suppression) as a deterministic regression gate over the 936-page corpus.

**5. If you ever needed determinism or a fallback transcriber** (not recommended as primary): **olmOCR 2 (Qwen2.5-VL-7B)** is the most drop-in (same stack, markdown-native, anchoring built in); **PaddleOCR-VL** is the accuracy/efficiency leader among small models but needs output conversion. A verbatim second opinion on flagged CLI pages could use **GOT-OCR2.0** for pure command strings.

**Net**: docpipe is on the correct 2025-frontier architecture for its task. The gap to SOTA isn't the model choice — it's that you're running a general VLM *without anchoring*, which is exactly the configuration the olmOCR authors showed to be hallucination-prone. Add born-digital anchoring (you already have the text layer in hand) and a unit-test QA gate; keep the large steerable general VLM as the primary transcriber to satisfy your unusual downstream markdown contract.

---

### Sources (title — arXiv id / URL)
- Nougat: Neural Optical Understanding for Academic Documents — [2308.13418](https://arxiv.org/abs/2308.13418)
- General OCR Theory (GOT-OCR2.0) — [2409.01704](https://arxiv.org/abs/2409.01704)
- olmOCR: Unlocking Trillions of Tokens in PDFs with VLMs — [2502.18443](https://arxiv.org/abs/2502.18443); olmOCR 2: Unit Test Rewards — [2510.19817](https://arxiv.org/abs/2510.19817); [blog](https://allenai.org/blog/olmocr-2)
- MinerU — [2409.18839](https://arxiv.org/abs/2409.18839); MinerU2.5 — [2509.22186](https://arxiv.org/abs/2509.22186); MinerU2.5-Pro — [2604.04771](https://arxiv.org/pdf/2604.04771)
- Marker — [github.com/datalab-to/marker](https://github.com/datalab-to/marker)
- Docling Technical Report — [2408.09869](https://arxiv.org/abs/2408.09869); Docling toolkit — [2501.17887](https://arxiv.org/abs/2501.17887)
- SmolDocling — [2503.11576](https://arxiv.org/abs/2503.11576)
- dots.ocr — [github.com/rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr)
- PaddleOCR-VL — [2510.14528](https://arxiv.org/abs/2510.14528)
- DeepSeek-OCR: Contexts Optical Compression — [2510.18234](https://arxiv.org/abs/2510.18234)
- Qwen2.5-VL — [2502.13923](https://arxiv.org/abs/2502.13923); Qwen3-VL — [2511.21631](https://arxiv.org/abs/2511.21631)
- InternVL3 — [2504.10479](https://arxiv.org/abs/2504.10479); InternVL3.5 — [2508.18265](https://arxiv.org/abs/2508.18265)
- OmniDocBench — [2412.07626](https://arxiv.org/abs/2412.07626); [leaderboard](https://github.com/opendatalab/OmniDocBench); ["saturated" analysis](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks)
- olmOCR-bench — [dataset](https://huggingface.co/datasets/allenai/olmOCR-bench)



---

## Appendix 2: VLM Page-Transcription Best Practices

# Faithful VLM Page-Image → Markdown Transcription: Best Practices & Prescriptions for docpipe

## Top findings (highest leverage first)

1. **The 2000px long-side clamp, not DPI, is the binding resolution limit — and it lands docpipe at ~182 effective DPI, marginal for 6-8pt text.** Worse, `escalate_dpi=300` is almost certainly **inert**: at both 200 and 300 DPI the clamp binds, so the "high-fidelity garbled retry" re-renders a byte-identical image.
2. **docpipe's assumption that "server pixel bounds are not exposed" is false.** Both vLLM and SGLang expose Qwen's `min_pixels`/`max_pixels` via `--mm-processor-kwargs` (and per-request `extra_body`). If the server is still at Qwen's **default `max_pixels = 28*28*1280 ≈ 1.0 MP`, every 2000×1545 page is being silently downscaled to ~1.0 MP (~104 DPI) — which would gut small table text.** This is measurable *today* from the `image_tokens` docpipe already captures.
3. **Temperature 0 is the single worst setting for repetition loops**, and docpipe sends **no** frequency/presence/repetition penalty. Since MTP+FP8 already breaks determinism (known issue b), the determinism argument for τ=0 is already void.
4. The prompt is genuinely strong and correctly diverges from OCR-benchmark convention (forces GFM, not HTML) because of the downstream parser contract — keep that.

---

## 1. Rasterization: DPI, pixel budget, and Qwen dynamic-resolution tiling

**How Qwen tiles.** Qwen2/2.5/3-VL split images into **14×14 px patches**, then an MLP merges each **2×2 group into one visual token**, so the effective unit is a **28×28 px cell** ([Qwen2-VL report](https://arxiv.org/abs/2409.12191)). Token count ≈ **pixels / 784**. Images are resized so H and W are multiples of 28, and the processor rescales any image to fit `[min_pixels, max_pixels]` ([HF Qwen2-VL docs](https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl)). The model supports 4–16,384 visual tokens; `min_pixels=100*28*28`, `max_pixels=16384*28*28` at the architecture level, but the **AutoProcessor default `max_pixels` is only `28*28*1280 ≈ 1.0 MP`** — a common silent bottleneck.

**Why resolution matters for small text.** OmniDocBench's headline VLM failure mode is that "VLMs struggle with high-density documents… due to limitations in **input resolution and token length**," producing "**Missing Content in dense pages**" and "**Hallucinations in hard-to-recognize pages**" ([OmniDocBench, CVPR 2025](https://arxiv.org/abs/2412.07626)). Qwen's own ablation (Fig. 4) shows OCRBench/InfoVQA rising with image size "within a reasonable range." The reliability floor for glyph recognition is roughly **cap-height ≥ ~20 px** (x-height ~10-12 px).

**The math for docpipe (US-Letter, 11" long side; cap-height ≈ pt × DPI/72):**

| Long-side px | Eff. DPI | 6pt cap-height | 8pt cap-height | Image tokens (~area/784) |
|---|---|---|---|---|
| **2000 (current)** | 182 | **15.2 px (marginal)** | 20.2 px | ~3,940 |
| 2400 | 218 | 18.2 px | 24.2 px | ~5,680 |
| 2800 | 254 | 21.2 px (good) | 28.2 px | ~7,730 |
| 3300 | 300 | 25.0 px | 33.3 px | ~10,730 |
| 4077 | 370 (Qwen 16,384-tok ceiling) | 30.8 px | — | ~16,384 |
| **1140 (if server clamps to 1.0 MP)** | **104** | **8.6 px (unreadable)** | 11.5 px | ~1,280 |

Two consequences specific to docpipe:
- **The clamp cancels the DPI escalation.** `_zoom_for()` returns `max_long_px / long_pt` whenever `dpi·(long_pt/72) > max_long_px`. For letter, that condition is true at *both* 200 and 300 DPI, so the zoom (and output raster) is identical — the garbled-page retry gains **zero** resolution.
- **Reference points.** olmOCR renders fine-tune/inference pages at **max 1024 px long edge** (2048 px only for GPT-4o labeling) and budgets **~1,000 image tokens + ~1,800 anchor-text tokens ≈ 3,000 input tokens/page** ([olmOCR](https://arxiv.org/abs/2502.18443)). Nougat renders at **96 DPI → 896×672** ([Nougat](https://arxiv.org/abs/2308.13418)). docpipe already exceeds both because Nutanix guides are screenshot- and small-CLI-text-dense — that is the right call, but only if the server actually receives the pixels.

**Prescriptions (§1):**
- **P0 — Measure, don't assume.** Log `image_tokens` from `usage.prompt_tokens_details` (already parsed in `vlm_client.py:201`). A native 2000×1545 page should report **~3,900**. If it reports **~1,280**, the server is downscaling to its default `max_pixels` and small text is already lost.
- **P0 — Set server pixel bounds explicitly.** Launch vLLM/SGLang with `--mm-processor-kwargs '{"max_pixels": 8605440, "min_pixels": 200704}'` (that max = `28*28*10976 ≈ 8.6 MP`, headroom for a 300-DPI escalation) or pass `mm_processor_kwargs` per request via `extra_body` ([vLLM PR #9612](https://github.com/vllm-project/vllm/pull/9612), [issue #13099](https://github.com/vllm-project/vllm/issues/13099); SGLang has the same flag). This removes the "bounds not exposed" constraint that motivated the defensive clamp.
- **P1 — Raise base + fix escalation.** Move base to **`max_long_px = 2400`** (218 DPI, 6pt→18 px, ~5.7k tokens) and set the **escalation to raise the clamp, not just DPI**: `escalate_max_long_px = 3300` (300 DPI, ~10.7k tokens). Both stay well under the 262K context and the 16,384-token image cap. Pre-round the render to a multiple of 28 on each axis to avoid a second resample blurring glyphs.

---

## 2. Prompt engineering for faithful, non-hallucinated transcription

**Zero-shot instruction is the right default.** olmOCR, Nougat, and the Qwen-OCR line are all zero-shot-instructed (or fine-tuned), never image-few-shot — image exemplars cost ~1k tokens each and invite the model to copy exemplar content. If format drift appears, add a **text-only** micro-example (one 2×2 GFM table + one fenced command), never an image. Research caveat: "reliability should not depend on ad hoc prompting" — prompts are necessary but not sufficient, so they must be paired with the QA gates in §3 ([Seeing is Believing? Mitigating OCR Hallucinations](https://arxiv.org/abs/2506.20168)).

**Anchoring is the biggest un-used lever.** olmOCR's central result: injecting the PDF's own text/coordinates into the prompt yields "**significantly fewer hallucinations**," because "prompting with just the page image was prone to models completing unfinished sentences" ([olmOCR](https://arxiv.org/abs/2502.18443)). docpipe's corpus is **born-digital (WeasyPrint-generated)**, so its text layer is high quality — docpipe already extracts it for advisory overlap but only passes the *previous page's* tail to the model. **Feeding this page's salient text-layer lines as an anchor** (olmOCR-style, capped ~1-2k chars, framed as "reference, transcribe the image") would directly attack the two known issues: bare-unfenced-command emission and repetition. Caveat: born-digital reading order can be imperfect, so anchor ≠ ground truth.

**Table format: GFM vs HTML — docpipe's divergence is correct.** The general OCR-benchmark finding is that **HTML tables score higher** (OmniDocBench evaluates all tables as HTML for TEDS; [LightOnOCR](https://huggingface.co/blog/lightonai/lightonocr) reports switching training tables from Markdown to HTML "significantly improves benchmark scores") — *because HTML expresses `rowspan`/`colspan` and GFM structurally cannot* ([GFM tables have no merged-cell syntax](https://www.markdowntools.io/table-merge-cells)). **But docpipe's downstream parser silently drops raw HTML blocks**, so HTML would be *lost*. The prompt's rule — force GFM, **flatten merged cells by repeating the spanned value** — is the correct, contract-dictated mitigation. Two notes: (a) complex spanning tables are exactly where VLMs fail most (OmniDocBench), so flag these pages for the overlap check; (b) value-repetition fabricates per-row independence — acceptable for retrieval, worth a comment.

**Code fences.** The prompt's rules (fence *every* command incl. lone/reference-list commands, language hints, top-level after list items, always close fences) are aligned with the parser contract and with the known "bare command" failure. Keep them; back them with the deterministic post-fence pass in §3.

---

## 3. Hallucination & degenerate repetition at temperature 0

**Why VLM-OCR loops (mechanism):**
1. **Maximization is the root cause.** Greedy/beam decoding provably degenerates into repetition because "the probability of a repeated phrase **increases with each repetition**" — a positive feedback loop from the LM's unreliable distribution tail ([Holtzman, *The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751)). τ=0 is the extreme of maximization, so **temperature 0 maximizes loop risk.**
2. **Lost visual grounding.** When cross-attention can't localize on the page — dense/complex layout, small or low-contrast text — the decoder falls back on the LM prior and loops. LOCR's cross-attention heatmaps show Nougat "**cannot be focused on the correct position when the layout is complex, resulting in repetition degeneration**," cycling through regions ([LOCR](https://arxiv.org/abs/2403.02127)). Nougat itself notes that after one mistake "the model is not able to recover from the collapse" ([Nougat](https://arxiv.org/abs/2308.13418)).
3. **Prevalence scales with difficulty.** Nougat loops on **1.5%** in-domain pages; LOCR measures Nougat at **4.4% (arXiv), 8.1% (marketing), 13.2% (OOD quantum)**. Dense command-reference/cheat-sheet pages (docpipe's known weak spot) are the high-risk class.
4. **Hallucination proper:** when visual evidence is weak/absent the model "**defaults to linguistic priors rather than anchoring decisions to observable visual evidence**." The recommended posture is **uncertainty-aware refusal**: unrecognizable text "**should not be included… represented by a space to prevent any hallucination**" ([2506.20168](https://arxiv.org/abs/2506.20168)). FP8 weights+KV (docpipe's config) flatten top-token logit gaps, marginally increasing drift/tie-breaking into loops.

**Detection (layer these; docpipe has only line-based today):**
- Length/finish signals: `finish_reason == "length"` (docpipe's `truncated`) and exceeding `max_tokens` — olmOCR catches loops precisely when "output exceeds the maximum context length or does not validate against our schema."
- Nougat's **logit-variance** method: variance of logits over a sliding window (B=15); if it drops below a threshold (they use 6.75; half-threshold over the last 200 tokens at generation time) and stays there, classify as repetition. Requires logprobs (vLLM/SGLang can return them).
- Cheap text heuristics docpipe should add to the existing line detector: **zlib/gzip compression ratio** (degenerate text compresses ~5-10× more than prose), **char/token n-gram repetition ratio**, **max single-line length** (a 5,000-char line is pathological), and **intra-line phrase loops** (current `_repetition_flag` only catches whole-line repeats ≥50% or ≥8 consecutive identical lines — it misses `the the the…` and sub-line loops).
- **Text-layer cross-check (docpipe's best asset).** It already computes `token_overlap`. Promote it from advisory to a **soft flag when the text layer is long (>~200 words) but overlap is very low** — that signature is *missing content or hallucination*, not an image-only page. Also add an output/text-layer **length-ratio** upper bound (output ≫ text layer ⇒ probable fabrication/loop).

**Prevention:**
- **Add mild penalties.** vLLM/SGLang OpenAI servers accept `frequency_penalty` and `presence_penalty` natively; `repetition_penalty`, `top_k`, `min_p`, `no_repeat_ngram_size` via `extra_body` ([vLLM sampling params](https://docs.vllm.ai/en/latest/api/inference_params.html)). Add **`frequency_penalty ≈ 0.1-0.3`** — the cheapest loop guard. Keep it *mild*: tables, flattened merged cells, and CLI flags legitimately repeat tokens. **Avoid `no_repeat_ngram_size`** (or set ≥20) — hard n-gram bans corrupt tables/commands.
- **Reconsider τ=0.** olmOCR found "increasing generation temperature from τ=0.1 up to τ=0.8 **reduces the likelihood of repetitions**." Because MTP+FP8 already destroys determinism at τ=0, a small **τ=0.1-0.2 with `top_p≈0.9`** costs docpipe nothing in reproducibility and buys loop resistance. (Prod choice: keep τ=0 but add `frequency_penalty`, *or* τ=0.1 + top_p 0.9; either beats bare τ=0.)
- **Recover, olmOCR-style:** on a flagged page, re-enqueue with (now-functional) higher DPI *and* a nudged sampler; after N tries fall back to text-layer extraction rather than emit garbage.
- **Deterministic post-fence pass (fixes known issue a):** since bare commands are captured as top-level text (not dropped), a stage-4 regex pass can fence unfenced command-looking lines (`^\s*(nutanix@|<acropolis>|ncli |acli |aCLI|ncli>|\$ )…`) — a clean, model-independent fix for cheat-sheet pages.

---

## 4. Reading order & multi-column

This is the hardest VLM failure mode — OmniDocBench sees "a **clear drop in accuracy on multi-column and complex layouts**," with models missing content and mis-ordering. **docpipe is largely exempt: the corpus is single-column**, so investing in column-detection machinery is unwarranted. Residual risks are local: (a) side-by-side figure+caption, (b) key/value or Description/Command two-column reference tables, (c) callout boxes interrupting flow. The prompt already handles Description/Command pairs (prose then fenced block) and callouts (blockquotes). Keep single-column assumptions; do not build multi-column linearization. If any landscape/2-col appendix pages exist, let the QA overlap check flag ordering damage rather than pre-engineering for it.

---

## 5. Cross-page continuity (tables/lists/code across page breaks)

Note that olmOCR and Nougat both process pages **independently** and defer joining to post-processing. docpipe's **previous-page-tail + explicit CONTINUATION rule is more sophisticated** and appropriate. Best-practice guardrails:
- **Let the stitcher, not the model, own re-joining.** A GFM table continued on page N has no header row (the header was on page N-1), which is *invalid GFM in isolation* — the model cannot both "continue without a header" and emit valid standalone GFM. So the model's continuation is best-effort; **stage-3 seam-repair must be the authority** that re-attaches the header and merges the fragments, and **stage-4 fence-balancing** must be the final guarantor for code split across a break.
- **Make `prev_tail` structure-aware.** Ensure the tail carries enough to disambiguate an *open* structure: an unbalanced ```` ``` ```` count, an open table (trailing pipe rows), or a dangling list. A fixed character tail can truncate mid-structure and mislead the model. Compute the tail to include the last open block boundary.
- **Guard against the two continuation failure modes** the prompt already forbids (re-emitting a table header row; re-opening a fence) with a **stitch-time dedup** of a repeated header/fence at a seam, since the model won't always comply. The known page-1 WeasyPrint disclaimer is a *single-page* artifact — correctly a stage-4 cleanup, not a continuity concern.

---

## 6. Figure/screenshot description for retrieval

docpipe's `_FIGURE_ENRICHED` mode is already best-practice and matches multimodal-RAG guidance: a caption's retrieval value comes from **verbatim on-screen text** (menu paths, field/column labels, button/tab/dialog text, concrete values) and **named diagram components/relationships**, grounded strictly in what's legible ("never invent labels/numbers not legible"). The `*[Figure: …]*` line is top-level text, so it survives the parser (not trapped in a dropped construct). Refinements:
- **Prepend a stable subject noun-phrase** — what the screenshot *is* (e.g., "Prism Element VM dashboard showing…") — so the chunk has a retrievable subject in addition to the transcribed strings. Multimodal-RAG systems retrieve on the text caption and (optionally) show the image at generation time.
- **Keep the anti-fabrication clause** — enriched captioning is where "describe detail not actually visible" hallucination creeps in; the existing "describe only what you can read" guard is the correct brake. Pair with the §3 overlap check on figure-dense pages.
- For screenshot-dense guides, `enriched` is the right default over `minimal`/`skip`; the token cost is justified by retrieval recall on UI-driven Nutanix procedures.

---

## Consolidated prescriptions for docpipe

**P0 (correctness — do first):**
1. Log/verify `image_tokens` per page; a native 2000×1545 page must read ~3,900, not ~1,280. If ~1,280, the server is downscaling.
2. Launch vLLM/SGLang with `--mm-processor-kwargs '{"max_pixels": 8605440, "min_pixels": 200704}'` (or per-request `extra_body`) — the "bounds not exposed" assumption is false.
3. Fix the inert escalation: make the garbled retry raise `max_long_px` (e.g., 3300), not just `dpi`. As written, `escalate_dpi=300` re-renders an identical image.

**P1 (quality):**
4. Base render `max_long_px = 2400` (218 DPI; 6pt→18 px), escalation clamp 3300 (300 DPI).
5. Add `frequency_penalty: 0.1-0.3` to the chat body (and consider `temperature 0.1` + `top_p 0.9`). Do **not** add `no_repeat_ngram_size` for table/CLI corpora.
6. Strengthen `_repetition_flag`: add compression-ratio, char n-gram, and max-line-length checks (catches intra-line loops the line detector misses).
7. Promote `token_overlap` to a soft flag when text layer is long but overlap is low (missing-content/hallucination signature); add an output≫text-layer length-ratio guard.
8. Deterministic stage-4 pass to fence bare CLI lines (fixes the known cheat-sheet issue model-independently).

**P2 (upside):**
9. Adopt olmOCR-style **text-layer anchoring** for *this* page (born-digital text is high quality) to cut hallucination and bare-command emission at the source.
10. Ensure `prev_tail` is structure-aware (open fence/table/list) and that stage-3 stitch owns table-header re-attachment across seams.

---

## Sources

- olmOCR — *Unlocking Trillions of Tokens in PDFs with VLMs* (Allen AI): https://arxiv.org/abs/2502.18443 · PDF: https://olmocr.allenai.org/papers/olmocr.pdf
- Nougat — *Neural Optical Understanding for Academic Documents* (Blecher et al.): https://arxiv.org/abs/2308.13418
- LOCR — *Location-Guided Transformer for OCR* (Nougat repetition/cross-attention analysis): https://arxiv.org/abs/2403.02127
- Qwen2-VL Technical Report (Naive Dynamic Resolution, 14px patch / 2×2 merge / min-max pixels): https://arxiv.org/abs/2409.12191 · HF processor docs (default `max_pixels = 28*28*1280`): https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl
- Qwen2.5-VL Technical Report: https://arxiv.org/abs/2502.13923
- OmniDocBench (CVPR 2025) — table/reading-order/dense-page & pipeline-vs-VLM findings: https://arxiv.org/abs/2412.07626
- Holtzman et al. — *The Curious Case of Neural Text Degeneration* (why maximization loops): https://arxiv.org/abs/1904.09751
- *Seeing is Believing? Mitigating OCR Hallucinations in MLLMs* (language-prior fallback, abstention): https://arxiv.org/abs/2506.20168
- LlamaIndex — *Failure Modes That Break VLM-Powered OCR in Production* (repetition vs recitation, temp retry): https://www.llamaindex.ai/blog/engineering-insights-failure-modes-that-break-vlm-powered-ocr-in-production
- LightOnOCR-1B (HTML vs Markdown table training effect): https://huggingface.co/blog/lightonai/lightonocr
- GFM tables cannot express merged cells: https://www.markdowntools.io/table-merge-cells · https://formatarc.com/en/blog/gfm-table-cheatsheet/
- vLLM — Qwen `min_pixels`/`max_pixels` via `mm_processor_kwargs`: https://github.com/vllm-project/vllm/pull/9612 · https://github.com/vllm-project/vllm/issues/13099 · Sampling params (frequency/presence/repetition penalties): https://docs.vllm.ai/en/latest/api/inference_params.html

**Grounded docpipe references:** `docpipe/rasterize.py` (`_zoom_for` clamp), `docpipe/config.py` / `docpipe.toml` (`dpi=200`, `max_long_px=2000`, `escalate_dpi=300`, `temperature=0.0`, `max_tokens=6000`), `docpipe/vlm_client.py` (chat body sends only temperature/max_tokens; captures `image_tokens`), `docpipe/prompts.py` (GFM-forced, merged-cell flattening, continuation rule, enriched figures), `docpipe/validate.py` (`_repetition_flag`, `token_overlap`).



---

## Appendix 3: RAG Corpus Prep & Extraction-Quality Evaluation

# Preparing & Evaluating docpipe's PDF→Markdown Corpus for GraphRAG

Two research strands (A: corpus preparation for RAG/GraphRAG; B: extraction-quality evaluation), then tied into (1) a concrete eval harness for docpipe artifacts and (2) a definition of "good artifacts" for our GraphRAG ingestion. Load-bearing claims are cited inline; full source list at the end.

---

## A. Preparing extracted Markdown for retrieval (and GraphRAG)

### A1. Structure-aware vs. semantic chunking — structure wins for formal docs

The 2025 consensus has swung *away* from expensive embedding-similarity ("semantic") chunking and *toward* structure-aware splitting for documents that already carry logical structure. A NAACL 2025 study found the compute cost of semantic chunking is not justified by consistent gains — fixed ~200-word chunks matched or beat it; a 2026 benchmark of 7 strategies put recursive 512-token splitting first (~69%) and pure semantic chunking near the bottom (~54%) ([Firecrawl chunking survey](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)). Conversely, *structure/layout-aware* chunking that aligns boundaries to headings and section logic outperforms fixed-size splitting substantially on formal/technical corpora (e.g., adaptive/layout chunking 87% vs. 50% for fixed-token on a clinical task) ([ACM AI 2025, Semantic Layout Chunking](https://dl.acm.org/doi/10.1007/978-981-95-4969-6_3); [Databricks chunking guide](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)). Tabular content benefits from dedicated structure-aware chunkers that keep a table (and its header row) atomic rather than shredding it across chunks ([Structure-Aware Chunking for Tabular Data, arXiv:2605.00318](https://arxiv.org/pdf/2605.00318)).

**Implication for docpipe:** the highest-leverage move is not a clever runtime chunker — it is emitting Markdown whose ATX heading structure *is* the chunk plan. Our downstream parser already treats ATX headings as hard chunk boundaries, so docpipe's job is to make those boundaries land on true semantic sections.

### A2. Heading / hierarchy preservation

Markdown H1–H6 give free, deterministic chunk boundaries, and the winning pattern is to split at H2/H3 while *stamping each chunk with the heading path it lives under* — reported to lift retrieval accuracy 40–60% vs. naive splitting ([MDSpin, Markdown for RAG](https://www.mdspin.app/guides/markdown-for-rag); [AnythingMD](https://anythingmd.com/blog/markdown-for-rag-boosting-accuracy-reducing-costs)). Heading titles double as cheap, high-quality chunk context, which is exactly what Anthropic's **Contextual Retrieval** prepends to each chunk: a 50–100 token "where does this fit" string added before embedding + BM25 indexing cut failed retrievals up to 49% alone, and 67% combined with reranking (5.7%→1.9%) ([Anthropic, Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)). A clean heading tree gives us that context for free, and a broken one (missing H1, H1→H4 jumps, duplicate H1s) corrupts every chunk under it.

### A3. Keeping tables and code intact for retrieval

Tables and code must survive as atomic, well-formed units — code fenced so it isn't "mangled with narrative text," tables kept whole so the LLM can read row/column relationships ([Markdown-for-RAG guides above]). This matters doubly for docpipe because our **verified ingestion contract silently DROPS** fenced code and GFM tables that are *not top-level* (nested under a list item or wrapped in raw HTML). So "intact" for us has a precise, testable meaning: top-level GFM table, balanced fences, rectangular cell grid.

### A4. Metadata / provenance for citations

Best practice is a per-chunk metadata envelope: stable chunk ID, source-document ID, page range, and section title, aligned to natural structure ([Markdown-for-RAG guides]). Citation-grounded RAG systems go further and *enforce* that every generated claim resolves to a retrievable source span ([Citation-Enforced RAG, arXiv:2603.14170](https://arxiv.org/pdf/2603.14170)). docpipe's contract only consumes `title` / `version` / `last_edited` from frontmatter (+ title fallback to first H1), so provenance has to be carried in those fields plus the sha/source-pdf/page we already track in the manifest.

### A5. What makes a corpus *good for GraphRAG* specifically

GraphRAG indexing runs LLM entity extraction → relationship extraction → claims → graph → community detection → community summaries over each text unit ([Microsoft GraphRAG methods](https://microsoft.github.io/graphrag/index/methods/); [From Local to Global, arXiv:2404.16130](https://arxiv.org/html/2404.16130v2)). Three corpus properties dominate quality:

1. **Chunk size drives entity recall.** GPT-4 extracted *almost twice as many entity references from 600-token chunks as from 2400-token chunks*; GraphRAG's default is 600-token units with 100 overlap, and it uses multi-round "gleanings" (self-reflection) to recover missed entities from larger chunks ([arXiv:2404.16130](https://arxiv.org/html/2404.16130v2)). Structure-aligned sections that produce smaller, self-contained units therefore yield a denser, more complete graph.
2. **Clean text or a fragmented graph.** Removing page furniture, headers/footers, page numbers, boilerplate, and fixing broken hyphenation/encoding is described as *foundational* — "even minor inconsistencies… distort how entities and relationships are detected," producing a fragmented or misleading graph ([memgraph, text→entity graphs](https://memgraph.com/blog/unstructured-text-to-entity-graphs-rag-tool); [PremAI GraphRAG guide](https://blog.premai.io/graphrag-implementation-guide-entity-extraction-query-routing-when-it-beats-vector-rag-2026/)).
3. **Entity disambiguation / consistent terminology.** The hardest GraphRAG failure is entity linking — the same real-world entity written differently splits into multiple nodes, and open-domain extraction invents many near-duplicate relation types, breaking multi-hop queries ([ideasthesia GraphRAG lessons](https://www.ideasthesia.org/microsoft-graphrag-architecture-and-lessons-learned/); [PremAI guide](https://blog.premai.io/graphrag-implementation-guide-entity-extraction-query-routing-when-it-beats-vector-rag-2026/)). Consistent product/command naming and a `version` field to distinguish same-named entities across Nutanix releases directly attack this.

---

## B. Evaluating document-extraction quality

### B1. Reference-based metrics (the standard toolkit)

- **Normalized Edit Distance (NED) / CER.** Character Error Rate = Levenshtein(pred, ref)/|ref| at character level (WER is the word-level analogue). OmniDocBench uses NED as its text metric, averaged per-sample ([OmniDocBench, arXiv:2412.07626](https://arxiv.org/html/2412.07626v1)). Lower is better; it is the workhorse for free-text fidelity.
- **TEDS / TEDS-Struct for tables.** Tables are rendered to an HTML tree and scored by Tree-Edit-Distance-based Similarity: `TEDS(Ta,Tb) = 1 − EditDist(Ta,Tb) / max(|Ta|,|Tb|)`, where |T| is node count. **TEDS-Struct** ignores cell text and scores structure only; **TEDS(-Content)** also penalizes cell-text errors. Introduced with PubTabNet ([Zhong et al., arXiv:1911.10683](https://arxiv.org/abs/1911.10683)). A grid-based alternative, **GriTS**, better handles multi-hop cell misalignment ([GriTS](https://www.researchgate.net/publication/359435680_GriTS_Grid_table_similarity_metric_for_table_structure_recognition)).
- **Reading order.** OmniDocBench scores reading order as the **NED over the sequence of text blocks** (tables/figures/ignored blocks excluded) ([arXiv:2412.07626](https://arxiv.org/html/2412.07626v1)).
- **Layout / formulas.** Layout detection uses **mAP**; formulas use **CDM + NED + BLEU** ([OmniDocBench](https://arxiv.org/html/2412.07626v1)).

### B2. Benchmark methodologies

**OmniDocBench (CVPR 2025).** 981 real PDF pages across 9 doc types and 19 layout categories with 14 attribute labels (language, scan/watermark/colored-bg, table frame type, merged cells, rotation…), enabling *per-attribute* breakdowns. End-to-end evaluation matches predicted vs. GT blocks via an "Adjacency Search Match" (NED matrix → fuzzy substring match → merge adjacent paragraphs), then applies the per-type metrics above; headers/footers/page-numbers/captions are excluded from scoring due to inconsistent model conventions ([arXiv:2412.07626](https://arxiv.org/html/2412.07626v1)). Takeaway: **evaluate by content type and by page attribute**, not with one global score.

**olmOCR-bench / olmOCR 2 (Ai2, 2025).** A deliberately **reference-free, unit-test** benchmark: 1,402 PDF pages, 7,010 binary pass/fail tests, in 6 families ([arXiv:2510.19817](https://arxiv.org/html/2510.19817v1); [Ai2 blog](https://allenai.org/blog/olmocr-2); [repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench)):
- **Text Presence** — a 1–3 sentence span must appear (fuzzy match, optional positional constraint).
- **Text Absence** — boilerplate (headers/footers/page numbers) must *not* appear.
- **Natural Reading Order** — span A must precede span B, uninterrupted by intervening content.
- **Table Accuracy** — a target cell has the right value *and* the right relative position (neighbor up/down/left/right).
- **Math Formula Accuracy** — the formula must **render the same under KaTeX** (compare rendered DOM bounding boxes), not match LaTeX strings.
- **Baseline Robustness** — no long repeated n-grams and no off-target-language characters (degenerate-decode guard).

The explicit rationale: edit distance and LLM-judges *"reward/penalize OCR output in a manner that doesn't correlate with practical correctness"* and treat equivalent representations (e.g., caption before vs. after body) differently; binary property tests let "different-yet-equivalently-correct" outputs score the same, and the pass-fraction (0–1) doubles as a verifiable RL reward (olmOCR 2 hit 82.4) ([arXiv:2510.19817](https://arxiv.org/html/2510.19817v1)). Tests are auto-synthesized: VLM does layout analysis → semantic HTML → refine, then headers/footers become Absence tests, KaTeX equations become Math tests, table cells become Table tests. **This is the single most transferable idea for docpipe.**

### B3. Building a lightweight, mostly reference-free harness when there is no gold Markdown

Four complementary signal families, none of which need hand-written gold Markdown:

1. **Text-layer overlap (deterministic, free, partial reference).** The PDF's own extractable text layer is a free noisy reference. Token/character overlap gives a CER-adjacent fidelity proxy and a *reading-order* proxy (align the shared token order between VLM output and text layer; disagreement ≈ NED on reading order). It must stay **advisory** — image-only/screenshot pages legitimately have low overlap.
2. **Structural self-consistency (deterministic).** No reference at all — just check internal well-formedness: balanced code fences; every GFM table rectangular (constant pipe-column count per row → a mini **TEDS-Struct against a re-parse** of your own table); monotone heading tree (no H1→H4 jumps, exactly one H1); **round-trip stability** (Markdown → AST → re-serialize → parse again yields the same block structure). Tables specifically reward a **round-trip / grid check**: parse to a cell matrix and confirm it's rectangular and re-emittable.
3. **olmOCR-style property unit tests (deterministic).** Auto-generate cheap pass/fail assertions: **Absence** tests for known boilerplate (our surviving WeasyPrint disclaimer line, running headers/footers, page numbers); **Presence** tests for anchor strings mined from the PDF text layer; **Baseline Robustness** (no long repeated n-grams / off-language runs).
4. **LLM-as-judge, sampled and image-grounded (semantic, advisory).** Crucial nuance: docpipe is **not truly reference-free** — the rasterized page PNG *is* the ground truth. A **multimodal judge comparing the .md against the page image** is a reference-*based* check that needs no gold Markdown, and is the strongest semantic signal available. Use it on a *stratified sample* (table-dense, CLI-dense, screenshot-dense pages) for faithfulness/completeness/structure, and as a **table round-trip judge** ("does this Markdown table encode the same cells as the image?"). Keep it off the hard gate: LLM judges carry position bias, self-preference bias, and prompt sensitivity; mitigate with two differently-worded prompts and score agreement ([Wang et al., reference-free eval, arXiv:2304.00723](https://arxiv.org/pdf/2304.00723); [Wolfe, LLM-as-a-judge](https://cameronrwolfe.substack.com/p/llm-as-a-judge); [Margin-Adaptive Confidence Ranking, arXiv:2605.15416](https://arxiv.org/pdf/2605.15416)). For tables, "beyond string matching" work confirms LLM-as-judge + semantic cell-content matching catches errors TEDS/edit-distance miss ([arXiv:2603.18652](https://arxiv.org/pdf/2603.18652)).

---

## C. Recommended eval harness for docpipe artifacts

docpipe already has the right bones in `docpipe/validate.py`: `assess_page()` (empty / `garbled:*` / `short_vs_textlayer` flags), `token_overlap()` (advisory text-layer recall), and `DocCoverage` (page completeness). `rasterize.page_text()` gives the free text-layer reference, the Stage-1 PNG gives the image reference, and `report.py:build_report/report_dict` is where results surface. Build the harness as three deterministic tiers plus one sampled semantic tier, layered on those primitives — **most of it is reference-free and CI-cheap; only tier 4 touches a model.**

**Tier 0 — Contract conformance (HARD GATE; deterministic; blocks a doc from the corpus).** A violation here means *silent downstream data loss*, so fail closed. Assert on the final `.md`:
- Frontmatter present, parseable, at **byte 0**, with non-empty `title` (carry `version`/`last_edited` when known). *(complements `clean.build_frontmatter`)*
- **Exactly one H1** *(complements `clean.demote_extra_h1s` / `ensure_title_h1`)*; heading tree has no level jump >1.
- **No signal-bearing content before the first ATX heading** (downstream drops it) — flag if >N non-frontmatter chars precede heading 1.
- **Every fenced code block and GFM table is top-level** — not indented under a list item, not inside a raw-HTML block (both are silently dropped downstream).
- **Balanced code fences** *(post-hoc check of `clean.balance_fences`)*; no stray raw-HTML table/code wrappers.

**Tier 1 — Structural self-consistency (SOFT / QUARANTINE; deterministic; no reference).**
- Table grid check / TEDS-Struct-against-re-parse: every GFM table rectangular and round-trips; quarantine ragged tables.
- Markdown AST round-trip stable (parse→serialize→parse invariant).
- Repetition / degenerate-decode guard (`validate._repetition_flag` already covers `garbled:loop`/`repetition`; keep as gate).
- **Command-density-without-fence** heuristic → advisory only, to surface known-issue (a) cheat-sheet pages that emit bare unfenced commands (captured as top-level text — retrievable, but not marked as code).

**Tier 2 — Text-layer + property tests (SOFT; deterministic; free partial reference).**
- Promote `token_overlap` to a tracked per-page metric with a **soft floor** on *text-heavy* pages (guard by text-layer length so screenshot pages don't trip it — mirrors existing `short_vs_textlayer` gating logic).
- Reading-order proxy: order-agreement of shared tokens vs. the text layer (advisory NED).
- olmOCR-style **Absence tests**: assert the WeasyPrint disclaimer line, running headers/footers, and page numbers are gone (known-issue (c) becomes a deterministic catch).
- **Presence tests**: sample anchor phrases from the text layer and assert they survived.

**Tier 3 — Sampled image-grounded LLM judge (ADVISORY; semantic; model in loop).**
- Stratified sample of table-/CLI-/screenshot-dense pages: multimodal judge scores the `.md` against the **page PNG** for faithfulness/completeness/structure, plus a table round-trip judgment. Two-prompt consistency; report score + disagreement. Never a gate.

**Gate policy.** Tier 0 = hard fail (deterministic, safe). Tiers 1–2 = quarantine for human review, summarized in `report.py`. Tier 3 = telemetry/triage only. Handle known-issue (b) — vLLM MTP+FP8 nondeterminism at temp 0 — by **never gating on byte-identical reproducibility**; instead track semantic drift across re-runs or oxcart↔blackbird as an advisory signal (tolerance-based, not exact-match).

Report surface: extend `report_dict` with, per doc and per corpus, the Tier-0 pass rate, quarantine list with reasons, mean/min text-layer recall on text-heavy pages, table-validity rate, and boilerplate-leak count — the same "evaluate by content type and attribute" discipline OmniDocBench uses.

---

## D. What "good artifacts" means for our GraphRAG ingestion

A docpipe `.md` is *good for GraphRAG* when it satisfies the ingestion contract **and** maximizes graph quality:

1. **Every semantic section sits under a real ATX heading.** Headings are our hard chunk boundaries, and structure-aligned (hence smaller, self-contained) units maximize per-chunk entity-extraction recall — GraphRAG got ~2× the entities from 600- vs. 2400-token chunks ([arXiv:2404.16130](https://arxiv.org/html/2404.16130v2)). Heading titles also become free chunk context (à la Contextual Retrieval) and clean node/community labels.
2. **No orphan pre-heading content** — anything before the first heading is dropped downstream, so it must be zero-signal.
3. **Tables and code are top-level and well-formed** so they survive ingestion and are retrievable as atomic units (raggedness or list-nesting = silent loss).
4. **Text is clean of furniture** — no headers/footers, page numbers, disclaimers, or hyphenation breaks; boilerplate is exactly what fragments an entity graph ([memgraph](https://memgraph.com/blog/unstructured-text-to-entity-graphs-rag-tool); [PremAI](https://blog.premai.io/graphrag-implementation-guide-entity-extraction-query-routing-when-it-beats-vector-rag-2026/)). Stage-3 `stitch.strip_page_furniture`/de-hyphenation is doing GraphRAG-critical work, not cosmetics.
5. **Provenance in frontmatter** (`title`/`version`/`last_edited`, plus manifest sha/source-pdf/page) enables citations *and* entity disambiguation — the `version` field distinguishes same-named Nutanix entities across releases, directly countering GraphRAG's core entity-linking failure ([ideasthesia](https://www.ideasthesia.org/microsoft-graphrag-architecture-and-lessons-learned/)).
6. **Consistent, canonical terminology** (product names, CLI/nCLI/aCLI command spellings) so logically identical entities/relations collapse to one node/edge instead of fragmenting the graph.
7. **Self-contained sections.** Because Stage-2 writes running prose per page with previous-page continuity context, good sections read standalone (minimal dangling anaphora) → cleaner extraction and higher-quality relationships.

In short: docpipe's downstream contract and GraphRAG's quality needs point the same direction — **clean, top-level, heading-structured Markdown with provenance and canonical naming.** The recommended harness enforces the contract deterministically (Tier 0), guards fidelity and structure cheaply and reference-free (Tiers 1–2), and uses the page image as ground truth for a sampled semantic check (Tier 3) — no hand-authored gold Markdown required.

---

## Sources

**Evaluation / benchmarks**
- OmniDocBench (CVPR 2025) — [arXiv:2412.07626](https://arxiv.org/html/2412.07626v1) · [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)
- olmOCR 2 / olmOCR-bench — [arXiv:2510.19817](https://arxiv.org/html/2510.19817v1) · [Ai2 blog](https://allenai.org/blog/olmocr-2) · [bench repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [LlamaIndex review](https://www.llamaindex.ai/blog/olmocr-bench-review-insights-and-pitfalls-on-an-ocr-benchmark)
- TEDS / PubTabNet — [Zhong et al., arXiv:1911.10683](https://arxiv.org/abs/1911.10683) · GriTS — [ResearchGate 359435680](https://www.researchgate.net/publication/359435680_GriTS_Grid_table_similarity_metric_for_table_structure_recognition)
- Semantic table extraction eval "beyond string matching" — [arXiv:2603.18652](https://arxiv.org/pdf/2603.18652)
- Reference-free / LLM-as-judge caveats — [Wang et al., arXiv:2304.00723](https://arxiv.org/pdf/2304.00723) · [Wolfe, LLM-as-a-judge](https://cameronrwolfe.substack.com/p/llm-as-a-judge) · [Margin-Adaptive Confidence Ranking, arXiv:2605.15416](https://arxiv.org/pdf/2605.15416) · [OmniAI OCR benchmark](https://getomni.ai/blog/ocr-benchmark)

**Corpus preparation / RAG / GraphRAG**
- Chunking strategy surveys — [Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) · [Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- Structure/layout-aware chunking — [ACM AI 2025](https://dl.acm.org/doi/10.1007/978-981-95-4969-6_3) · [Structure-Aware Tabular Chunking, arXiv:2605.00318](https://arxiv.org/pdf/2605.00318)
- Anthropic Contextual Retrieval — [anthropic.com/engineering/contextual-retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- Markdown for RAG — [MDSpin](https://www.mdspin.app/guides/markdown-for-rag) · [AnythingMD](https://anythingmd.com/blog/markdown-for-rag-boosting-accuracy-reducing-costs)
- GraphRAG "From Local to Global" — [arXiv:2404.16130](https://arxiv.org/html/2404.16130v2) · [Microsoft GraphRAG methods](https://microsoft.github.io/graphrag/index/methods/)
- GraphRAG corpus quality / entity resolution — [PremAI guide](https://blog.premai.io/graphrag-implementation-guide-entity-extraction-query-routing-when-it-beats-vector-rag-2026/) · [ideasthesia lessons](https://www.ideasthesia.org/microsoft-graphrag-architecture-and-lessons-learned/) · [memgraph text→graph](https://memgraph.com/blog/unstructured-text-to-entity-graphs-rag-tool)
- Citation-enforced RAG — [arXiv:2603.14170](https://arxiv.org/pdf/2603.14170)

**docpipe grounding (local):** `docpipe/validate.py` (`assess_page`, `token_overlap`, `DocCoverage`), `docpipe/clean.py` (`balance_fences`, `demote_extra_h1s`, `ensure_title_h1`, `build_frontmatter`), `docpipe/stitch.py` (`strip_page_furniture`, `stitch_pages`), `docpipe/rasterize.py` (`page_text`), `docpipe/report.py` (`build_report`, `report_dict`).



---

## Appendix 4: Is Qwen3.6-27B Fit for Purpose?

# Is Qwen3.6-27B the right transcription engine for docpipe? — Evidence-based verdict

## Verdict up front

**Conditional YES — it is a defensible primary engine for *this* workload, but it is over-provisioned for pure OCR and carries a real hallucination/throughput tax. Keep it as the default, but (1) add olmOCR-style document anchoring, (2) harden against repetition/non-determinism, and (3) add a small specialized fallback for pure-table and command-cheat-sheet pages.** For docpipe's specific job — screenshot-dense Nutanix docs that need *semantic figure enrichment* plus a *precise custom Markdown contract* — a general VLM's steerability and figure-reasoning genuinely beat a pure-OCR model. But the evidence is unambiguous that on *pure document parsing accuracy per dollar*, 1–3B specialists (MinerU2.5, dots.ocr, MonkeyOCR, LightOnOCR) match or beat even 235B general VLMs. You are not buying better OCR with the 27B; you are buying figure understanding + instruction-following + one-model simplicity. If the corpus scales past ~tens of thousands of pages, that trade flips.

---

## 1. What "Qwen3.6-27B" actually is (and an evidentiary caveat)

docpipe's config (`docpipe.toml`, model_id `qwen36-27b-fp8-oxcart`, auto-probed from `/v1/models`) points at a **Qwen3.6-series VL model**. Qwen3.6 is a real early-2026 Alibaba release line (Qwen3.6-35B-A3B MoE and Qwen3.6-VL variants, ~April 2026) with document-OCR positioning ([aimlapi](https://aimlapi.com/blog/qwen-3-6-series-alibabas-open-source-llm-revolution-in-2026), [Alibaba Cloud](https://www.alibabacloud.com/blog/qwen3-6-plus-towards-real-world-agents_603005)). **I could not find published OmniDocBench/OCRBench numbers for the specific 27B-VL variant** — it is too new and thinly benchmarked. So all hard accuracy numbers below are anchored to the well-documented **Qwen3-VL** predecessor (technical report, Nov 2025, [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) and the public OmniDocBench leaderboard. The 27B dense sits between Qwen3-VL's 32B-dense and 30B-A3B-MoE tiers in capability, so those are the right analogs; treat the flagship-235B numbers as an *upper bound* the 27B will not exceed on OCR.

---

## 2. Evidence: the Qwen-VL family on document/OCR

**Qwen2.5-VL** ([arXiv:2502.13923](https://arxiv.org/abs/2502.13923)) introduced the architecture docpipe relies on: a **native dynamic-resolution ViT trained from scratch with window attention**, so a page is perceived at native resolution without tiling — exactly why docpipe can feed ~1700×2200px @200 DPI pages directly. It reports **96.4% DocVQA** and OmniDocBench edit distance **0.226 EN / 0.324 ZH** (72B), with explicit HTML-table/structured-extraction ability.

**Qwen3-VL** ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) adds what matters most for docpipe:
- **Native 256K-token interleaved context** (text+image+video) — docpipe pins 262K; ample headroom for the previous-page continuity tail.
- **OCR expanded to 32–39 languages**, with stated robustness to "low light, blur, and tilt" and rare/jargon characters.
- Enhanced interleaved-MRoPE + **DeepStack** multi-level ViT features (better fine-text localization).
- The report explicitly says repetition/language-mixing were targeted in post-training with "high-frequency penalties" — i.e., Qwen itself treats repetition as a known failure mode to be suppressed, not an solved one.

**OmniDocBench standing (the key numbers):**

| Model | Size | OmniDocBench v1.5 composite (higher=better) | v1.0 overall edit-dist EN/ZH (lower=better) |
|---|---|---|---|
| MinerU2.5-Pro | 1.2B | **95.69** (v1.6) | — |
| MinerU2.5 | 1.2B | 90.67 | 0.139 / 0.240 (MinerU2) |
| Gemini 3 Pro | — | 90.33 | — |
| **Qwen3-VL-235B** | 235B MoE | **89.15** | **0.155 / 0.207** |
| MonkeyOCR-pro | 3B | 88.85 | 0.138 / 0.206 |
| **dots.ocr** | 3B | 88.41 | **0.125 / 0.160** |
| Gemini-2.5-Pro | — | 88.03 | 0.148 / 0.212 |
| olmOCR | 7B | 81.79 | — |
| Qwen2.5-VL | 72B | — | 0.214 / 0.261 |
| GPT-4o | — | 75.02 | 0.233 / 0.399 |
| GOT-OCR2 | 0.58B | — | 0.287 / 0.411 |

Sources: [CodeSOTA OmniDocBench leaderboard](https://www.codesota.com/ocr/benchmark/omnidocbench), [dots.ocr paper](https://arxiv.org/html/2512.02498v1), [Qwen3-VL report](https://arxiv.org/abs/2511.21631). *(Two metric families exist — always separate them: v1.0 reports normalized edit distance, lower better; v1.5/v1.6 report a 0–100 composite, higher better. Do not compare across columns.)*

**The load-bearing takeaway:** a **235B general VLM (89.15) is essentially tied with a 3B specialist (dots.ocr 88.41, MonkeyOCR-pro 88.85) and *below* a 1.2B specialist (MinerU2.5 90.67)** on document parsing. A 27–32B general VLM will land *below* its own 235B on pure OCR. So on transcription-fidelity-per-parameter, the general VLM is strictly the wrong tool — the 27B is justified only by the non-OCR abilities in §5.

---

## 3. Evidence: the specialized systems

- **olmOCR / olmOCR-2** ([arXiv:2510.19817](https://arxiv.org/abs/2510.19817), [blog](https://allenai.org/blog/olmocr-2)) — itself a **Qwen2.5-VL-7B fine-tune**. **82.4 on olmOCR-Bench**, beating Marker (76.1) and MinerU (75.8). This is the strongest existence proof that the *right* base for doc-OCR is a mid-size VLM *fine-tuned for the task* — not a large general VLM used zero-shot. Two techniques are directly relevant to docpipe's known issues:
  - **Document anchoring:** injecting the PDF's own text-layer words + coordinates into the prompt "results in significantly fewer hallucinations… prompting with just the page image was prone to models completing unfinished sentences or inventing larger texts when the image data was ambiguous" ([olmOCR paper](https://arxiv.org/pdf/2502.18443)).
  - **Dynamic temperature scaling** (0.1→0.8, bumped whenever no EOS) to break repetition loops.
- **GOT-OCR2.0** (0.58B, [arXiv:2409.01704](https://arxiv.org/abs/2409.01704)) — elegant end-to-end OCR-2.0, 0.035 edit distance on *plain* text. But **1024×1024 max input** ([HF docs](https://huggingface.co/docs/transformers/model_doc/got_ocr2)) — undersized for docpipe's ~2000px screenshot pages without tiling — and it collapses on full complex pages (0.287/0.411 OmniDocBench). **Not viable as docpipe's primary.**
- **MinerU2.5** (1.2B decoupled VLM, [arXiv:2509.22186](https://arxiv.org/html/2509.22186v2)) — SOTA parsing (90.67), **2.12 pages/s on one A100** optimized, 2337 tok/s. Pipeline-style layout→content→reading-order; strong tables.
- **dots.ocr** (2.9B, [arXiv:2512.02498](https://arxiv.org/abs/2512.02498)) — single-VLM layout+content, **TableTEDS 88.6/89.0**, 100+ languages, up to 11M pixels. Outputs **layout JSON + Markdown (with HTML tables)**.
- **LightOnOCR-1B / -2** ([arXiv:2601.14251](https://arxiv.org/abs/2601.14251)) — SOTA on olmOCR-Bench while ~9× smaller, **5.71 pages/s per H100 (~493k pages/day) at <$0.01 per 1,000 pages**.

---

## 4. Head-to-head on the four axes

**Accuracy (pure transcription):** Specialists win or tie. dots.ocr/MinerU2.5/MonkeyOCR-pro (1–3B) ≥ Qwen3-VL-235B on OmniDocBench; a 27B general VLM lands somewhere in the high-80s composite at best — competitive, not leading. On **complex tables specifically**, TEDS-optimized specialists have a clear edge.

**Hallucination tendency:** This is the general VLM's structural weakness. "Seeing is Believing?" (NeurIPS 2025, [arXiv:2506.20168](https://arxiv.org/abs/2506.20168)) shows MLLMs **over-rely on language priors under visual degradation, producing semantically plausible but visually unsupported text** — precisely the failure that turns a mis-rendered CLI flag or config value into a confident fabrication. For a corpus feeding GraphRAG, a hallucinated `ncli`/`acli` argument propagates into downstream answers. Specialists trained *only* to transcribe hallucinate less by construction; olmOCR's whole design (anchoring + verifiable unit-test rewards) exists to suppress this. **This is docpipe's #1 risk and it is inherent to using a general VLM zero-shot.**

**Throughput / cost:** Not close. On the *same* 96GB GPU, a 27B-FP8 model (~27GB weights, `max_running=4`, up to 6000 output tok/page on dense spec tables) will run **1–2 orders of magnitude fewer pages/GPU-hour** than a 1–3B specialist that fits in <6GB and can run `max_running` 16–32+. Anchors: olmOCR-7B ≈ 3,400 tok/s/H100 (~10k pages < $2); LightOnOCR-1B ≈ 5.71 pages/s/H100 (<$10/million pages); MinerU2.5 ≈ 2.12 pages/s/A100. A 27B dense at ~4× the per-token compute of the 7B, emitting more tokens/page, plausibly costs **~4–8× olmOCR and ~20–40× a 1–3B specialist per page** (estimate, scaled from published 1B/7B figures — no measured 27B OCR number is published). For 936 pages this is irrelevant; at 50k+ pages it dominates.

**Operational fit (self-hosted vLLM/SGLang FP8):** Advantage general VLM *for the current setup*. Qwen3-VL/3.6 has first-class vLLM + SGLang support, FP8 weights+KV, native dynamic resolution (no tiling logic), and a 256K window — docpipe's dual oxcart(vLLM)/blackbird(SGLang) least-outstanding routing is a clean fit. A 27B-FP8 fits comfortably on 96GB with KV headroom at `max_running=4`. Specialists also serve on the same engines, but several emit **HTML tables and layout JSON** (dots.ocr, MinerU) — which collide with docpipe's parser contract (raw-HTML blocks and nested-in-list tables are *silently dropped*), so adopting one adds an HTML→GFM normalization layer.

---

## 5. Why a general VLM still wins *for docpipe specifically*

docpipe is **not a pure-OCR job**, and this is the crux of the "yes":

1. **Figure enrichment (`figures = enriched`).** The corpus is screenshot-dense; describing a screenshot's content as useful alt-text is a **VLM reasoning task**, not OCR. Specialists (dots.ocr, MinerU, GOT) *detect/box* figures but do not richly describe them. This is the single strongest pro-Qwen argument and specialists cannot replace it.
2. **A precise, unusual downstream Markdown contract.** docpipe needs top-level GFM tables + fenced code, ATX headings as hard chunk boundaries, single H1, YAML frontmatter at byte 0 — and must *avoid* raw HTML and nested-in-list tables (both dropped by the RAG parser). A general instruction-following VLM can be steered to emit exactly this dialect and make judgment calls ("is this a command? fence it"). Specialists emit fixed formats you must post-convert.
3. **CLI/nCLI/aCLI judgment + cross-page continuity.** Deciding what is a command vs prose and using the previous-page tail for seam repair are reasoning/steering tasks a general model does natively. (The known bare-unfenced-command issue is a *steering* weakness, mitigable — see §7 — not a reason to switch engines.)
4. **One model, one prompt, one deploy** vs a multi-model layout+OCR+table+reading-order pipeline. Real operational simplicity for a solo/small-team corpus build.

Note on *size*, though: since the 235B flagship only reaches ~89 on OmniDocBench, most of the figure-reasoning + steering benefit is likely available from a **smaller general VLM (e.g., an 8B-class Qwen3-VL/3.6)** at a fraction of the cost. **A/B the 27B against an 8B-class sibling** — if quality holds, that is the cheapest single win available.

---

## 6. Risks (ranked)

1. **Hallucination on ambiguous/degraded regions** — invents plausible CLI flags/values; corrupts GraphRAG answers. (Inherent; [arXiv:2506.20168](https://arxiv.org/abs/2506.20168).)
2. **Repetition loops at temperature 0** on dense pages — docpipe runs `temperature=0.0`, the highest-risk setting; Qwen's own report flags repetition as a suppressed-not-solved behavior, and olmOCR built dynamic-temperature specifically for this.
3. **Non-determinism** (your known issue b): vLLM MTP + FP8 is not bit-exact at temp 0 → non-reproducible corpus builds and noisy re-run diffs.
4. **Throughput/cost blow-up** if the corpus scales beyond low-thousands of pages.
5. **Table fidelity** on the densest spec tables trails dedicated TEDS-optimized models.

---

## 7. Mitigations & fallbacks (ranked by leverage, all evidence-backed)

**A. Add document anchoring (highest leverage).** These are digitally-generated WeasyPrint PDFs with a clean text layer, so extracting PyMuPDF words+coords is nearly free. Inject that text into the prompt alongside the image (olmOCR's method). This is the single best defense against hallucination *and* directly attacks your known issue (a) bare-command pages, because the anchor text carries the exact command strings. docpipe today only does *post-hoc advisory* text-overlap in stage 5 — **promote it to in-prompt anchoring.** ([olmOCR](https://arxiv.org/pdf/2502.18443))

**B. Turn the stage-5 text-layer overlap into an active arbiter.** When VLM output diverges from the anchor text beyond a threshold, flag/re-run rather than merely advising. This catches both hallucination and dropped/garbled command lines.

**C. Repetition + determinism hardening.** Disable MTP/speculative decoding for the transcription pass (accept lower speed for reproducibility); keep oxcart pinned (already done). On stage-5 repetition/garble detection, re-run with **temperature escalation** (0.1→0.2→…, mirroring olmOCR) *and* your existing `escalate_dpi=300`. The per-page artifact cache already makes accepted outputs stable across runs.

**D. Specialist fallback for pure-table and command-cheat-sheet pages (the honest "smaller model is better here").** Classify pages (dense-table or command-dense) and route those bodies to **dots.ocr (3B)** or **MinerU2.5 (1.2B)** — TableTEDS ~88–89 and far cheaper — while keeping the general VLM for figure-enriched and prose pages. This is a targeted two-model ensemble, not a wholesale switch, and it directly retires risks #1/#5 on the pages where they bite hardest.

**E. Table/HTML normalization pass (needed the moment any specialist is added, useful anyway).** Convert HTML tables → GFM and lift any nested-in-list tables to top level, so nothing is silently dropped by the RAG parser.

**F. Self-consistency for QA-flagged pages only.** Sample twice and reconcile against the anchor text for pages stage-5 flags; reserve for the few percent that fail QA, since it doubles cost.

---

## 8. Conditions — when the answer flips

- **Keep Qwen3.6-27B as primary** while: corpus is hundreds–low-thousands of pages, screenshot/figure-dense, needs semantic figure enrichment + a bespoke Markdown contract, and you value one-model simplicity. **This is exactly docpipe today → the choice is sound, conditional on mitigations A–C.**
- **Demote it to a figure-only specialist** (with dots.ocr/MinerU2.5/LightOnOCR as the text/table primary) if: the corpus scales to 100k+ pages, pages become predominantly dense tables/plain text with few figures, or cost/throughput start to dominate.
- **Cheapest immediate win regardless:** A/B the 27B against an **8B-class** general VLM in the same family — the leaderboard says you are almost certainly overpaying for OCR you can get from a much smaller model.

---

### Sources
- Qwen3-VL Technical Report — https://arxiv.org/abs/2511.21631
- Qwen2.5-VL Technical Report — https://arxiv.org/abs/2502.13923
- Qwen3.6 series overview — https://aimlapi.com/blog/qwen-3-6-series-alibabas-open-source-llm-revolution-in-2026 ; https://www.alibabacloud.com/blog/qwen3-6-plus-towards-real-world-agents_603005
- olmOCR 2 (paper) — https://arxiv.org/abs/2510.19817 ; (blog) https://allenai.org/blog/olmocr-2 ; (v1 paper, anchoring/temperature) https://arxiv.org/pdf/2502.18443
- GOT-OCR2.0 — https://arxiv.org/abs/2409.01704 ; https://huggingface.co/docs/transformers/model_doc/got_ocr2
- MinerU2.5 — https://arxiv.org/html/2509.22186v2 ; MinerU2.5-Pro https://arxiv.org/pdf/2604.04771
- dots.ocr — https://arxiv.org/abs/2512.02498
- LightOnOCR-1B/-2 — https://arxiv.org/abs/2601.14251
- OmniDocBench (benchmark) — https://arxiv.org/html/2412.07626v2 ; leaderboard https://www.codesota.com/ocr/benchmark/omnidocbench
- OCR hallucination / language priors — "Seeing is Believing?" (NeurIPS 2025) https://arxiv.org/abs/2506.20168

*(Note relayed to caller: docpipe's exact model is labeled "Qwen3.6-27B-FP8" and auto-probed from `/v1/models`; no public OmniDocBench numbers exist for that specific 27B-VL variant, so accuracy figures are anchored to the documented Qwen3-VL predecessor and the 235B flagship as an upper bound. All throughput numbers for the 27B are order-of-magnitude estimates scaled from published 1B/7B figures, not measured.)*

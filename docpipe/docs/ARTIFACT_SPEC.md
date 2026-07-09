# docpipe — EXPECTED OUTPUT ARTIFACT SPECIFICATION

**Scope:** the exact contents and shape of a single generated `.md` file (one per source PDF) that is *excellent* for GraphRAG ingestion. Every rule is grounded in (a) the verified downstream parser behavior and (b) the research findings. Load-bearing parser facts are cited as `file:line`.

---

## 0. The one-sentence contract

> A docpipe artifact is a **single UTF-8 Markdown file** whose **byte 0 is a YAML frontmatter block**, whose body is a **single-H1 ATX heading tree** in which **every semantic section lives under a real heading**, and in which **every code block and table is a well-formed, top-level (un-nested, non-HTML) block** — because anything that violates those four properties is **silently dropped or mis-chunked** by the ingestion parser, and every one of those properties simultaneously maximizes GraphRAG entity/relationship recall.

Four hard truths from the parser drive the entire spec (verified, `markdown_it_parser.py`):

| # | Parser behavior | Evidence | Consequence for the artifact |
|---|---|---|---|
| C1 | Frontmatter is matched **only at byte 0** by `^---\s*\n(.*?)\n---\s*\n`; only `title`/`version`/`last_edited` are read, the rest is preserved-but-inert. | `:62`, `:199-225`, `:791-808` | Frontmatter must be the literal first bytes, well-formed YAML, closed with `---`. |
| C2 | Sections are built by walking **top-level AST children only** (`for child in ast.children`, non-recursive); a heading **finalizes** the previous section. | `:582`, `:489-531` | ATX headings are **hard chunk boundaries**; heading tree = chunk plan. |
| C3 | Content is accumulated **only after** the first heading (`current_section` is `None` until then). | `:471`, `:521` | **Anything before the first H1 is DROPPED.** No orphan pre-heading content. |
| C4 | Only top-level `fence`/`code_block`/`table`/`list`/`paragraph`/`blockquote` are captured. Code/tables **nested in a list item** are not top-level (`_render_list_text` pulls only paragraphs + nested lists, `:414-448`); **raw HTML** has no handler despite `html:True` (`:122`, `:533-578`). | `:533-578` | Fenced code + GFM tables must be **top-level**; **never** indented under a list item; **never** HTML. Both silently vanish otherwise. |

A fifth, positive fact ties fidelity to graph quality: **CLI commands become first-class `Command` graph entities primarily from *fenced code blocks*** — `extract_commands()` Pattern 1 iterates `section["code_blocks"]` (`extract/commands.py:36`); the prose fallbacks (Patterns 2–3) are weaker. So fencing a command is not cosmetic — it is the difference between a command becoming a graph node or not.

There is **no per-document size cap** in the parser (confirmed; `chunk_assembler` `COMBINE_MAX_SECTIONS=12` only *merges* small sections, it does not cap doc size).

---

## 1. YAML frontmatter schema

### 1.1 Exact schema

Emitted at **byte 0**, opened and closed by a line that is exactly `---`, valid YAML, all values double-quoted scalars (docpipe emits via `_yaml_scalar`, `clean.py:114-121`).

```yaml
---
# ── PARSER-CONSUMED (read into Document metadata) ──────────────
title: "AHV Administration Guide"      # REQUIRED. Document title.
version: "6.5"                          # REQUIRED. Product/doc version.
last_edited: "2024-11-18"              # RECOMMENDED. Source doc date (ISO-8601).
# ── PROVENANCE (inert to parser; preserved in Document.frontmatter) ──
product: "AHV"                          # Nutanix product area.
source_pdf: "5c-book-of-ahv-administration.pdf"
sha256: "3f9a1c...e77b"                 # Content hash of the source PDF.
page_count: 84
dpi: 200
model_id: "qwen36-27b-fp8-oxcart"
endpoint: "oxcart"
pipeline_version: "0.1.0"
extracted_at: "2026-07-07T15:04:00+00:00"
---
```

### 1.2 Key-by-key rationale

| Key | Required | Parser use | Why it must be here |
|---|---|---|---|
| `title` | **Yes** | `frontmatter.get("title")` wins title resolution (`:791-795`). | Deterministic, faithful title independent of body H1 drift; becomes the Document node label and every chunk's `parent_path` root. |
| `version` | **Yes** | `document["version"]` (`:803`, default `"1.0"`). | **GraphRAG entity disambiguation:** same-named Nutanix entities (e.g. an `ncli` flag) across releases collapse into one node unless separated by version. This directly attacks GraphRAG's #1 failure mode (entity linking). |
| `last_edited` | **Recommended** | `document["last_edited"]` (`:805`, default `None`). | Recency/citation signal. `build_frontmatter` now emits this parser-consumed key from the extraction timestamp when a source document date is unavailable. |
| `product` | Provenance | Preserved in `frontmatter` dict, else inert. | Human/tooling filter; **note it is *not* `doc_category`** — `doc_category` is derived from the *directory* after `ingest/` in the path (`stages/parse.py:77-89`), so placing the file at `data/ingest/nutanix/<slug>/<slug>.md` is what yields `doc_category="nutanix"`. |
| `source_pdf`, `sha256`, `page_count` | Provenance | Inert (preserved). | **Citation spine:** resolves any chunk back to an exact source PDF and lets a re-ingest detect a changed source. `sha256` is the immutable identity. |
| `dpi`, `model_id`, `endpoint`, `pipeline_version`, `extracted_at` | Provenance | Inert (preserved). | Reproducibility/audit: which render + which model + which prompt version + when produced this artifact. Ties an artifact to the exact `(sha, dpi, model, prompt_version)` cache key (`artifacts.py:45`). |

### 1.3 Frontmatter hard rules

- **Byte 0, no BOM, no leading blank line.** The regex is anchored (`.match`, `:211`); one leading byte and the whole block reparses as body — and since it precedes the first heading, it is then *dropped* (C3).
- **Closed with `---` followed by a newline.** Body starts after a blank line (`clean.py:165` emits `{frontmatter}\n\n{body}`).
- **Valid YAML** (`yaml.safe_load`, `:217`). A parse error silently yields `{}` frontmatter → title falls back to body H1, `version` defaults, provenance lost. Double-quote all scalars; escape embedded quotes/backslashes.
- **Do not embed a literal `DocTag:` string** anywhere in the document — an inline `DocTag: X` in body content overrides doc-tag derivation (`stages/parse.py:68`).

---

## 2. Title rules

Resolution order (docpipe `resolve_title`, `clean.py:53-54`; parser mirror `:791-795`):

1. **Frontmatter `title`** (authoritative downstream).
2. Document's **own first H1** — taken *faithfully* from what the model transcribed (`first_h1_text`).
3. Embedded PDF metadata title.
4. Humanized filename slug.

Rules:
- The body **must open with exactly one H1**, and it must be the title (`ensure_title_h1`, `clean.py:72-79`). Frontmatter `title` and that H1 should be **identical strings** so citations, the Document node, and the first section agree.
- The H1 is **faithful, not invented** — transcribe the real cover/section title; never summarize or normalize it.
- One H1 per document; any later `#` is demoted to `##` (`demote_extra_h1s`, `clean.py:57-69`).

---

## 3. Heading-tree requirements

- **Single H1** at the top (title); everything else is `##`/`###`/… (C2, `clean.py`).
- **ATX only** (`#`, `##`, …). **No setext** (`Title\n====`) and **never bold-as-heading** (`**Section**` on its own line) — a bold line parses as a *paragraph*, not a heading, so it creates **no chunk boundary** and, if before the first real heading, is dropped (C3). This is a silent-loss trap.
- **Headings are hard chunk boundaries** (C2). Therefore each heading must mark a **true semantic section**, because the heading tree *is* the chunk plan. Structure-aligned sections yield smaller, self-contained units → the research shows GraphRAG extracts ~2× the entities from ~600-token vs ~2400-token chunks, so real section granularity directly raises graph density.
- **No orphan pre-heading content** (C3): between the frontmatter and the first `#` there must be **zero signal-bearing characters** (whitespace only). QA fails the artifact otherwise.
- **Monotone depth:** no level jump > 1 (no `#` → `###`). A jump corrupts `parent_path` for every descendant chunk (`:502-506`).
- **Heading path is free chunk context.** The parser stamps each section with `parent_path` (`"Installation > Prerequisites"`, `:505`) — a clean tree hands GraphRAG Contextual-Retrieval-style context for free; a broken tree poisons every chunk beneath it.
- **Continuation pages must not repeat the title** or re-open a section heading (prompt rule 1/8, `prompts.py`); the stitcher de-dupes a repeated table header/fence at a seam (`stitch.py:96-104`).

---

## 4. Code-block fidelity

**Requirements**

1. **Fenced, triple-backtick, with a language hint.** Never inline single-backticks for a command, never 4-space indentation (indentation makes an *indented code block* with no language, and inside a list item it is dropped entirely — C4).
2. **Top-level.** In a numbered procedure, write the step prose as the list item, then place the fence as a **top-level block immediately after** the list item — never indented under it (C4; prompt rule 4).
3. **Fence every command** — including a lone one-line command and each row of a command-reference list (prompt rule 3). Rationale is load-bearing: fenced blocks feed `Command` entity extraction Pattern 1 (`extract/commands.py:36`); an unfenced command reaches only the weaker prose patterns.
4. **Balanced fences.** Every ` ``` ` opens and closes. An unclosed fence makes the parser swallow the rest of the document as one code block. `balance_fences` (`clean.py:87-111`) closes a dangling fence before the next `##+` or at EOF, but the artifact must not rely on it.

**Language-hint convention** (prompt rule 3):

| Content | Hint |
|---|---|
| shell / nCLI / aCLI (`nutanix@cvm$ ncli …`, `<acropolis> net.list`) | `bash` |
| PowerShell | `powershell` |
| command output, REPL, config dumps, unknown syntax | `text` |

**Known issue (not a new bug):** dense command-reference / cheat-sheet pages sometimes emit **bare unfenced commands**. These are captured as **top-level paragraph text** — *retrievable, not dropped* — but they miss `code` metadata and the primary command-entity path. QA flags them **advisory** (command-density-without-fence); a deterministic Stage-4 regex pass can fence lines like `^\s*(nutanix@|<acropolis>|n?cli |acli |\$ )…`.

---

## 5. Table fidelity

**Requirements**

- **GFM pipe tables only. Never HTML.** Raw-HTML tables have no parser handler (C4) → the entire table vanishes. This is *the* reason docpipe diverges from OCR-benchmark convention (which favors HTML for `rowspan`/`colspan`): our downstream drops HTML, so GFM is contract-mandated (prompt rule 2).
- **Top-level**, never nested under a list item (C4).
- **Merged/spanning cells flattened** by **repeating the spanned value** across each affected cell, so no information is lost (GFM cannot express spans). Acceptable trade: it fabricates per-row independence, fine for retrieval.
- **Rectangular grid:** every row has the same pipe-column count, with a valid header + separator row (`|---|---|`). A ragged table mis-parses cells.
- **Atomic:** a table (and its header) stays whole; the stitcher re-attaches a header split across a page break and drops the repeated header (`stitch.py:96-104`).

**How to measure — TEDS.** Tree-Edit-Distance Similarity renders a table to an HTML tree and scores `TEDS(A,B) = 1 − EditDist(A,B) / max(|A|,|B|)` (node count); **TEDS-Struct** ignores cell text (structure only). Two applications for docpipe:

1. **Reference-free round-trip (hard gate):** parse the emitted GFM table to a cell matrix and re-serialize; **TEDS-Struct against the re-parse must equal 1.0** (rectangular, re-emittable). Any table below 1.0 is quarantined.
2. **Image-grounded sampled (advisory):** on a stratified sample of table-dense pages, a multimodal judge compares the table against the page PNG (the ground truth) — target **TEDS ≥ 0.90**; use **GriTS** as the cross-check for multi-hop cell-misalignment cases. Never a hard gate (known non-determinism, issue b).

---

## 6. Figure / screenshot description for retrieval

Default mode `enriched` (`prompts.py:33-41`), correct for this screenshot-dense corpus. Each figure becomes a **top-level italic bracketed line** — which survives the parser as a paragraph (not trapped in a dropped construct):

```
*[Figure: Prism Element VM dashboard — "Create VM" dialog. Fields: Name, vCPU(s)=2, Memory=8 GiB, Disk "SCSI.0 20 GiB". Buttons: Save, Cancel.]*
```

Anatomy for retrieval value:
1. **Lead with a stable subject noun-phrase** — *what the screenshot is* ("Prism Element VM dashboard…") — so the chunk has a retrievable subject, not just loose strings.
2. **Transcribe verbatim on-screen text**: menu paths, field/column labels, button/tab/dialog text, and concrete values shown.
3. **Name diagram components and their labelled relationships.**
4. **Anti-fabrication clause is mandatory:** describe only what is legibly visible; never invent labels, numbers, or UI text not present (enriched captioning is exactly where hallucination creeps in). This is grounded by the QA overlap check on figure-dense pages.

---

## 7. Provenance for citations

Provenance lives in three layers, each with a purpose:

- **Frontmatter provenance keys** (`source_pdf`, `sha256`, `page_count`, `version`, `last_edited`, `model_id`, `dpi`, `pipeline_version`, `extracted_at`) — preserved in `Document.frontmatter` (`:807`); resolve any chunk to an exact source and render build reproducible.
- **Path placement** — `doc_category` is derived from the directory after `ingest/` (`stages/parse.py:77-89`). Write to `data/ingest/nutanix/<slug>/<slug>.md` to stamp `doc_category="nutanix"` on the Document and every section.
- **`sha256` as immutable identity + `version` as the disambiguator** — together they let citation-grounded answers name a source *and* keep same-named entities from different releases as distinct graph nodes.

Citation-grade means: every generated claim can resolve to a retrievable span, and that span carries `title` + `version` + `source_pdf` + `parent_path`. The artifact supplies all four.

---

## 8. Body-content hygiene (GraphRAG-critical, not cosmetic)

- **No page furniture:** running headers/footers, page numbers, watermark bands (e.g. `AHV | Host Network Management | 60`) are stripped (`stitch.strip_page_furniture`, `:29-66`; prompt rule 7). Boilerplate is precisely what fragments an entity graph.
- **De-hyphenated across seams:** `config-` + `uration` → `configuration` (`stitch._merge_seam`, `:106-110`).
- **Callouts as blockquotes** with a bold label: `> **Note:** …`, `> **Warning:** …` (prompt rule 5) — survives as a top-level `blockquote` (`:569-578`).
- **List items = prose + nested lists only** (prompt rule 9): only paragraphs and nested lists survive inside a list item downstream (`_render_list_text`, `:414-448`). Anything structural (code/table) goes top-level after the item.
- **Canonical, consistent terminology** for product names and `ncli`/`acli`/`nCLI` spellings, so logically identical entities/relations collapse to one node/edge instead of fragmenting the graph.
- **Known surviving artifact (issue c, not novel):** a page-1 WeasyPrint disclaimer line can survive; it is a single-page artifact and is a Stage-4 cleanup / QA **Absence** test, not a continuity concern.

---

## 9. Golden annotated skeleton (Nutanix-flavored)

The **exact bytes** of an excellent artifact (annotation key follows). Everything below `---` is the file, starting at byte 0:

```markdown
---
title: "AHV Administration Guide"
version: "6.5"
last_edited: "2024-11-18"
product: "AHV"
source_pdf: "5c-book-of-ahv-administration.pdf"
sha256: "<source-pdf-sha256>"
page_count: 84
dpi: 200
model_id: "qwen36-27b-fp8-oxcart"
endpoint: "oxcart"
pipeline_version: "0.1.0"
extracted_at: "2026-07-07T15:04:00+00:00"
---

# AHV Administration Guide

## Host Network Management

AHV uses Open vSwitch (OVS) to connect the Controller VM, the hypervisor, and
guest VMs to the physical network. Each AHV host runs one OVS instance, and all
OVS instances form a single logical switch across the cluster.

### Virtual Switch Requirements

The following requirements apply to every virtual switch in the cluster.

| Parameter        | Default    | Configurable | Notes                          |
|------------------|------------|--------------|--------------------------------|
| MTU              | 1500 bytes | Yes          | Set per virtual switch         |
| Bond mode        | active-backup | Yes       | active-backup on all uplinks   |
| Bond mode        | active-backup | Yes       | balance-slb requires LACP off  |

> **Note:** Changing the bond mode briefly interrupts host connectivity. Perform
> the change during a maintenance window.

### Viewing Bridge and Bond Configuration

To display the current bridge and bond layout on a host:

1. Log on to the Controller VM with SSH.

```bash
nutanix@cvm$ acli net.list
```

2. Display the bridges configured on the AHV host.

```bash
nutanix@cvm$ manage_ovs show_bridges
```

The command prints one row per bridge:

```text
Bridge: br0
  bond: br0-up
    lacp: off
    interfaces: eth3 eth2
```

*[Figure: Prism Element "Network Configuration" page — Virtual Switch vs0 selected.
Uplink ports table columns: Host, Uplink Ports, Bond Type. Row: "AHV-1, eth2/eth3,
active-backup". Buttons: Edit, Delete, Create VS.]*
```

**Annotation key**

1. **Frontmatter at byte 0**, `---`-delimited, all scalars quoted → satisfies C1. `title`+`version`+`last_edited` are consumed; the rest is preserved provenance. (`last_edited` is emitted by `build_frontmatter`; source doc dates can replace the extraction timestamp when known.)
2. **Single H1** = `title` string, identical to frontmatter → §2/§3. It is the very first body block, so **no content is dropped to C3**.
3. **`##`/`###` ATX tree, monotone** (no level skips) → §3. Each heading is a real chunk boundary (C2) and a real semantic section, so `parent_path` = `"Host Network Management > Virtual Switch Requirements"` is clean.
4. **GFM table, top-level, rectangular**, with the **merged "Bond mode" cell flattened** by repeating the value across both rows → §5. No HTML → survives C4. Round-trip TEDS-Struct = 1.0.
5. **Callout as a blockquote with a bold label** → §8; survives as a top-level `blockquote`.
6. **Procedure pattern:** step prose is the list item; the `bash` fence sits **top-level immediately after** the item, *not indented under it* → satisfies C4 (indented-in-list code is dropped) and feeds `Command` entity extraction (`acli`, `manage_ovs`).
7. **`text` fence for command output** (unknown/REPL syntax) → §4 hint table; balanced fences throughout → §4.
8. **Enriched figure line**, top-level italic, leads with the subject noun-phrase then transcribes verbatim on-screen labels/values, invents nothing → §6.
9. **No page furniture, no pre-heading orphan text, no raw HTML, no bare unfenced command** anywhere → passes the Tier-0 gate below.

---

## 10. Per-artifact QA acceptance checklist

Four tiers. **Tier 0 = hard gate (fail ⇒ artifact must not enter the corpus)** because each Tier-0 violation is *silent* downstream data loss. Tiers 1–2 quarantine for review; Tier 3 is advisory telemetry. Thresholds are grounded in `validate.py` where noted; **[rec]** marks a recommended addition beyond today's code.

### Tier 0 — Contract conformance (deterministic HARD GATE)

| Check | PASS criteria | FAIL ⇒ | Metric / method |
|---|---|---|---|
| Frontmatter present @ byte 0 | `^---\s*\n(.*?)\n---\s*\n` matches at offset 0 **and** `yaml.safe_load` succeeds | Provenance lost; block may drop | Run the parser's own regex + YAML load |
| `title` non-empty | frontmatter `title` is a non-empty string | Title falls back / unstable | Key presence |
| Exactly one H1 | count of `#`-level headings == 1, and it is the first body block | Multi-title / dropped intro | AST heading scan |
| No orphan pre-heading content | zero non-whitespace chars between frontmatter close and first heading | Content **dropped** (C3) | Byte scan |
| Monotone heading depth | no level increase > 1 | Broken `parent_path` | Level-delta scan |
| Every code block top-level & fenced | no `fence`/`code_block` under a list/HTML ancestor; all fenced (no indented code blocks) | Code **dropped** (C4) | AST ancestor check |
| Every table top-level & GFM | no `table` under a list ancestor; **no raw HTML** table/block tokens | Table **dropped** (C4) | AST scan for `html_block`/`html_inline` + list-ancestor |
| Balanced fences | ` ``` ` count is even; no dangling fence at EOF | Rest-of-doc swallowed | Fence-parity count |
| ATX only | no setext headings; no bold-line-as-heading | Missing boundary | Token markup check |

### Tier 1 — Structural self-consistency (deterministic, SOFT / quarantine)

| Check | PASS criteria | Metric |
|---|---|---|
| Table grid / round-trip | every GFM table rectangular and re-emittable; **TEDS-Struct vs re-parse == 1.0** | TEDS-Struct |
| AST round-trip stable | parse → serialize → parse yields identical block structure | block-structure diff |
| No degenerate repetition | no single line > 50% of non-empty lines (when ≥ 6 lines); no ≥ 8 consecutive identical lines | `_repetition_flag` (`validate.py:44-59`) → `garbled:loop`/`garbled:repetition` |
| Alnum ratio sane | `alnum_ratio ≥ 0.55` | `_alnum_ratio` (`:36-41`, `:76`) → `garbled:low_alnum` |
| **[rec]** intra-line loop / compression guard | gzip ratio not pathological; no `the the the…`; max single-line length under bound (e.g. < 2000 chars) | zlib ratio + char n-gram + max-line-len |
| Command-density-without-fence | **advisory** flag when a page has many command-shaped lines but few fences (known issue a) | regex density |

### Tier 2 — Text-layer & property tests (deterministic, SOFT; free partial reference)

| Check | PASS criteria | Metric |
|---|---|---|
| Short-vs-text-layer | NOT (`text_layer > 200 chars` AND `body < 0.15 × text_layer`) | `short_vs_textlayer` (`validate.py:83`) |
| **[rec]** Text-layer recall floor (text-heavy pages) | on pages with text-layer > ~200 words, `token_overlap ≥ ~0.60`; low overlap on a text-heavy page ⇒ missing-content/hallucination flag | `token_overlap` (`:62-67`) — *currently advisory only* (`:85`); promote to soft flag |
| **[rec]** Output ≫ text-layer guard | output length not wildly greater than text-layer length | length-ratio upper bound (fabrication/loop signature) |
| **Absence** tests | WeasyPrint page-1 disclaimer (issue c), running headers/footers, and page numbers are **absent** from the artifact | substring/regex absence |
| **Presence** tests | anchor phrases sampled from the PDF text layer are present | fuzzy substring |

### Tier 3 — Image-grounded semantic judge (model-in-loop, ADVISORY only)

| Check | Method | Note |
|---|---|---|
| Faithfulness / completeness / structure | Multimodal judge compares `.md` against the page **PNG** (the true reference) on a stratified sample of table-/CLI-/screenshot-dense pages; two differently-worded prompts, report score + disagreement | Never a gate |
| Table round-trip judgment | "Does this Markdown table encode the same cells as the image?" → TEDS/GriTS-style | Advisory |
| Cross-run / cross-endpoint drift | tolerance-based semantic diff across re-runs or oxcart↔blackbird | **Never gate on byte-identical reproducibility** (known issue b: vLLM MTP+FP8 non-determinism at temp 0) |

### Document-level completeness (hard, `output.py` today)

- **Every page produced an `ok` artifact.** A doc with any failed/missing page is **not written** unless `--allow-incomplete` (`output.py:69-73`). Partial docs must not silently enter the corpus.

### Corpus-level report surface (extend `report_dict`)

Per doc and per corpus: **Tier-0 pass rate**, the **quarantine list with reasons**, **mean/min text-layer recall on text-heavy pages**, **table-validity (round-trip) rate**, and **boilerplate-leak count** — the same "evaluate by content type and attribute" discipline OmniDocBench uses.

---

## 11. Acceptance in one line

> An artifact is **accepted** iff it passes **all Tier-0 checks** and document-level completeness, with **zero Tier-1 quarantine flags**; Tier-2 soft-floor breaches and Tier-3 scores are recorded as telemetry and route the page to review, but do not by themselves block a byte-identical re-run (issue b is handled by tolerance, never exact-match).

**Key files backing this spec:** ingestion contract — `/Users/brennanconley/vibecode/wekadocs-matrix/src/ingestion/parsers/markdown_it_parser.py`, `/Users/brennanconley/vibecode/wekadocs-matrix/src/ingestion/stages/parse.py`, `/Users/brennanconley/vibecode/wekadocs-matrix/src/ingestion/extract/commands.py`; producer — `docpipe/clean.py`, `docpipe/prompts.py`, `docpipe/validate.py`, `docpipe/stitch.py`, `docpipe/output.py`, `docpipe/artifacts.py`, `docpipe/rasterize.py`, `docpipe/docpipe.toml`. Remaining optional follow-up: `token_overlap` is computed but left as a soft advisory signal where richer text-heavy-page scoring is desired.

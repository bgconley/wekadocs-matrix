
You’re right to be skeptical of “lots of tiny sections.”

**Over‑fragmentation hurts retrieval quality**—but swinging to very large chunks (2–4k tokens) can also dilute topical focus. The best path is a **combine‑first, split‑if‑needed strategy with guardrails** and measurement.

  

Below is a concrete, production‑ready plan you can execute now.

---

## **TL;DR (recommended plan)**

  

**Option C — Hierarchical Combine → (rare) Split → Chunk‑aware Retrieval**

1. **Keep your current H2/H3 parser** as the semantic skeleton.
    
2. **Combine adjacent micro‑sections** _within the same H2_ until reaching a **target window of ~800–1,500 tokens** (configurable), with an **absolute max of 7,900** (safety below Jina’s 8,192).
    
3. **Only split** when a single logical unit still exceeds the hard cap; use a **light (≤100 tokens) overlap** only in those split cases.
    
4. **Store provenance** (which original sections were merged) and add **NEXT_CHUNK** edges for adjacency.
    
5. **Retrieval returns combined chunks**, then **optionally expands to adjacent chunks** for full context before rerank.
    
6. **Instrument and verify**: size distribution, hit quality, zero‑loss integrity, and latency.
    

  

This yields **fewer, richer vectors** than your current 100–300‑token pieces without over‑averaging unrelated topics.

---

## **Why not “lots of small” or “single huge” chunks?**

- **Too small (≈100–300 tokens)**: weak semantic signal, many results needed to assemble an answer, higher storage/search overhead, brittle answers.
    
- **Too large (≥2–4k tokens everywhere)**: vectors can blend topics, hurting precision for pinpoint queries; slower ingestion and retrieval; more irrelevant text per hit.
    

  

**Sweet spot:** ~800–1,500 tokens typically captures a complete idea (procedure, concept, API sub‑section) without mixing disjoint topics. Keep it **configurable** so you can tune per corpus.

---

## **First, sanity‑check the numbers (important)**

  

There’s a mismatch in the report:

- A single doc is said to have **80,799 tokens** and **262 sections** → average ≈ **308 tokens/section**.
    
- Elsewhere, “**largest section 297 tokens**.” Those can’t both be true.
    

  

Before changing the pipeline, add a **sanity check** in the report step:

- Sum of section token_counts per doc **must equal** doc token_count (± tokenizer idiosyncrasies like special tokens).
    
- Report **p50/p75/p90/p99** token sizes per doc and **count of sections <200 tokens**.
    

  

If the doc truly averages ~300 with a max ~300, you **are** over‑fragmented.

---

## **The go‑forward plan in detail**

  

### **1) Add a “Combiner” stage (after parsing, before embedding)**

  

**Scope**

- Work on the **linearized H3 siblings under each H2** (don’t cross H2 by default).
    
- Optionally allow within‑H4 merges if your parser emits them.
    

  

**Heuristics (ordered, cheap to compute)**

- **Token budget:** accumulate until **target_min** ≤ size ≤ **target_max**
    
    - Default: target_min=800, target_max=1,500, absolute_max=7,900.
        
    
- **Don’t cross hard semantic breaks:** new H2, or headings that look like FAQ, Changelog, Glossary, Warnings, Examples—treat these as their own anchors that **attach** to a neighbor but don’t pull distant topics with them.
    
- **Keep structural blocks intact:** never split inside fenced code, tables, or admonitions; if a code block sits alone and is tiny, attach it to the preceding explanatory section.
    
- **Micro‑section absorption:** if a section is **<120 tokens**, greedily attach to its closest neighbor (prefer the previous one if it’s explanatory).
    
- **End‑of‑H2 smoothing:** if the final combined chunk under an H2 ends up **< target_min**, merge it into the previous chunk unless that would exceed absolute_max.
    

  

**Data you must emit per combined chunk**

- text (merged with clear separator, e.g., two newlines)
    
- token_count (measured with your XLM‑RoBERTa tokenizer)
    
- parent_h2_id, original_section_ids (ordered), order
    
- boundaries (start/end offsets or heading ids) for integrity checks
    

  

**Deterministic IDs**

id = sha256(document_id + join(original_section_ids))[:24] to preserve idempotency.

  

> This “Combiner” is O(n) over sections and adds negligible ingestion time given your current <10 ms/token-count performance.

---

### **2) Keep a minimal “Splitter” (fallback only)**

- Trigger only if a single logical unit blows past **7,900** tokens.
    
- Split at **sentence or paragraph boundaries**, use **≤100‑token overlap** only in these rare cases.
    
- Mark chunks with is_split=true, carry parent_section_id, and chain via **NEXT_CHUNK**.
    

---

### **3) Schema adjustments (lightweight)**

  

On Section (or Chunk) nodes add:

- is_combined: BOOLEAN
    
- is_split: BOOLEAN
    
- order: INT
    
- total_chunks: INT
    
- parent_section_id: STRING (the pre‑combination anchor)
    
- boundaries: JSON (ids or offsets)
    

  

Relationships:

- (:Section)-[:NEXT_CHUNK]->(:Section) for adjacency within a combined/split series.
    

  

Index:

- (parent_section_id, order) for fast reassembly.
    

---

### **4) Graph builder & vector store behavior**

- **Embed only the combined/split chunks**, not the original micro‑sections, to avoid doubling your Qdrant points.
    
- Keep the original sections in Neo4j for provenance/rendering only.
    
- Set embedding_provider=jina-ai (as you do) on the combined nodes.
    

---

### **5) Retrieval changes (chunk‑aware)**

1. **Hybrid search** (keep whatever you have—BM25 + vector is ideal).
    
2. Vector search returns chunks.
    
3. **Group by** **parent_section_id**; if the matched chunk is part of a sequence, **optionally pull ±1 adjacent chunk** (bounded expansion) when
    
    - query is long (≥12 tokens), or
        
    - top‑k scores are close (ambiguous)
        
    
4. **Rerank** by (vector_score + keyword_score) and **coverage** (fraction of the parent covered by selected chunks).
    
5. Return a **stitched answer context**: selected chunk(s) in canonical order, with headings preserved.
    

  

This gives you **cohesive context** without requiring big overlaps at index time.

---

### **6) Instrumentation & acceptance criteria**

  

**Ingestion metrics (emit to logs and your report)**

- Per doc: count_chunks, p50/p90/p99 token_count, ≤200 token count.
    
- Global: total Qdrant points before/after, average tokens per point.
    

  

**Correctness/integrity**

- **Zero‑loss check:** concatenating CombinedSection.original_section_ids in order must reproduce the original H2 subtree (use a SHA‑256 to compare).
    
- **Token safety:** assert all embedded chunks **≤ 7,900** tokens; alert otherwise.
    

  

**Retrieval quality (quick A/B)**

- Build a small query set (10–20 real queries).
    
- Compare **baseline** (current) vs **combined** on:
    
    - hit@1/hit@3 judged relevance,
        
    - average #chunks returned per answer,
        
    - answer assembly time,
        
    - editor hand‑offs (how often the answer needs adjacent sections).
        
    

  

**Performance**

- Ingestion: ensure <2× baseline time (it should be close to baseline).
    
- Retrieval: bounded adjacency expansion should keep latency tight.
    

---

## **Pseudocode sketch (combiner core)**

```
def combine_sections(h2_sections):
    chunks = []
    cur, cur_tokens = [], 0

    for s in h2_sections:  # s has .text, .id, .kind (code/table/etc.), .tokens
        t = s.tokens
        hard_break = s.is_h2_start or s.is_glossary_like or s.is_changelog_like

        if hard_break and cur:
            chunks.append(pack(cur))
            cur, cur_tokens = [], 0

        if t < 120 and cur:
            # absorb micro-section
            cur.append(s); cur_tokens += t
            continue

        if cur_tokens + t <= ABSOLUTE_MAX:
            cur.append(s); cur_tokens += t
            if cur_tokens >= TARGET_MAX:
                chunks.append(pack(cur))
                cur, cur_tokens = [], 0
        else:
            # finalize current and start new
            if cur: chunks.append(pack(cur))
            cur, cur_tokens = [s], t

    if cur:
        # end-of-H2 smoothing
        if cur_tokens < TARGET_MIN and chunks:
            merged = unpack(chunks.pop()) + cur
            if sum(x.tokens for x in merged) <= ABSOLUTE_MAX:
                chunks.append(pack(merged))
            else:
                chunks.append(pack(unpack(chunks[-1])))
                chunks.append(pack(cur))
        else:
            chunks.append(pack(cur))

    return chunks
```

_Constants (configurable):_ TARGET_MIN=800, TARGET_MAX=1,500, ABSOLUTE_MAX=7,900.

---

## **Rollout plan (low‑risk)**

1. **Add the combiner** behind a feature flag: COMBINE_SECTIONS=on.
    
2. **Re‑ingest a sample corpus** (the 5 test docs) and produce the same report you already generate, now with the new metrics.
    
3. **Run the quick A/B retrieval test** on your query set.
    
4. If metrics and judged relevance improve (they should), **re‑ingest production**. Keep the splitter behind SPLIT_FALLBACK=on.
    
5. **Monitor**: token distribution, vector count, retrieval latency; adjust TARGET_MIN/MAX if needed.
    

---

## **What this changes relative to your phase plan**

- Your Phase 2 (“Intelligent Splitter”) becomes **Phase 2a: Combiner (primary)** and **Phase 2b: Splitter (fallback)**.
    
- The **schema (Phase 3)** and **graph builder (Phase 4)** changes remain, but are now oriented around **combined** chunks.
    
- **Retrieval adjustments (Phase 5)**: keep exactly as you outlined—just operate on combined chunks.
    
- **Testing (Phase 6)**: add the zero‑loss + distribution checks above.
    

---

## **Final recommendation**

  

Proceed with **Option C**: **combine adjacent small sections into ~800–1,500‑token chunks, then split only when necessary**, and make retrieval chunk‑aware. This directly addresses your (correct) concern about context fragmentation while avoiding the pitfalls of very large chunks. Keep the thresholds configurable and verify with the quick A/B you can run today.

  

If you want, I can sketch the exact Neo4j updates and Qdrant upsert shape for the combined nodes and the NEXT_CHUNK edges next.
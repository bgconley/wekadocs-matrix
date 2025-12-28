# WekaDocs MCP Retrieval Playbook (KB + Graph)

Purpose: Guide MCP clients to assemble answers using the graph before fetching text, keeping responses compact and budget-aware.

Principles
- Progressive retrieval: plan → find seeds → expand graph → fetch minimal text.
- Budgets first: prefer cursors and byte caps; avoid bulk text.
- Product-shaped results: return IDs + previews, then selective excerpts.

Default Flow
1) kb.retrieve_evidence: preferred default for direct Q&A (returns quotes only)
2) kb.search: use for exploration or when you need to browse candidates
3) expand_neighbors / list_parents / list_children: explore the local neighborhood
4) describe_nodes: inspect lightweight projections to rank what’s worth reading
5) get_paths_between: connect concepts when relationships matter
6) get_entities_for_sections / get_sections_for_entities: pivot by entities to broaden/narrow
7) kb.read_excerpt / kb.expand_excerpt: fetch bounded excerpts (preferred)
8) get_section_text: legacy excerpt fetch when needed (keep 4–8KB)
9) compute_context_bundle: assemble a budgeted bundle when needed

Tool Hints
- kb.search: use top_k=5, max_per_doc=1, and previews to avoid bloat
- kb.read_excerpt: default max_tokens=300; use expand if you need more context
- expand_neighbors: 1–2 hops, filter rel_types, use cursor for pagination
- get_paths_between: keep max_hops ≤ 3 and max_paths small (≤ 10)
- describe_nodes: cheap summaries to triage before fetching text
- list_parents / list_children: navigate hierarchy without text
- get_section_text: default max_bytes_per 8KB; override only when necessary

Mini‑Recipes
- Fast Answer: kb.retrieve_evidence → respond with quotes + citations
- Neighborhood Summary: kb.search → expand_neighbors → describe_nodes → kb.read_excerpt (few)
- Connect Concepts: kb.search(A,B) → get_paths_between → describe_nodes → kb.read_excerpt (pivots)
- Task Context Bundle: seeds → compute_context_bundle with explicit token/byte budget

Budget Guidance
- Keep single MCP responses < 512KB; prefer multiple small calls
- Token budget per turn ~14k; avoid streaming full text unless essential
- Use cursors for paging; return next_cursor when available

Diagnostics
- When `kb.search` or `kb.retrieve_evidence` returns a `diagnostic_id`, prefer operator tools for inspection.
- MCP resources (if enabled via `MCP_DIAGNOSTICS_RESOURCES_ENABLED=true`) expose:
  - `wekadocs://diagnostics/{date}/{diagnostic_id}` (Markdown summary only)
- CLI fallback (works without MCP resources):
  - `python scripts/retrieval_diagnostics/show.py --id <diagnostic_id>`

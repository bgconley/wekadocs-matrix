# WekaDocs MCP Retrieval Playbook (Graph-First)

Purpose: Guide MCP clients to assemble answers using the graph before fetching text, keeping responses compact and budget-aware.

Principles
- Progressive retrieval: plan → find seeds → expand graph → fetch minimal text.
- Budgets first: prefer cursors and byte caps; avoid bulk text.
- Product-shaped results: return IDs + metadata, then selective excerpts.

Default Flow
1) search_sections (verbosity="snippet"): find relevant section/entity nodes (return ids/metadata only)
2) expand_neighbors / list_parents / list_children: explore the local neighborhood
3) describe_nodes: inspect lightweight projections to rank what’s worth reading
4) get_paths_between: connect concepts when relationships matter
5) get_entities_for_sections / get_sections_for_entities: pivot by entities to broaden/narrow
6) get_section_text: fetch a few key excerpts (4–8KB each, multiple small calls)
7) compute_context_bundle: assemble a budgeted bundle when needed

Tool Hints
- expand_neighbors: 1–2 hops, filter rel_types, use cursor for pagination
- get_paths_between: keep max_hops ≤ 3 and max_paths small (≤ 10)
- describe_nodes: cheap summaries to triage before fetching text
- list_parents / list_children: navigate hierarchy without text
- get_section_text: default max_bytes_per 8KB; override only when necessary

Mini‑Recipes
- Neighborhood Summary: search_sections → expand_neighbors → describe_nodes → get_section_text (few)
- Connect Concepts: search_sections(A,B) → get_paths_between → describe_nodes → get_section_text (pivots)
- Task Context Bundle: seeds → compute_context_bundle with explicit token/byte budget

Budget Guidance
- Keep single MCP responses < 512KB; prefer multiple small calls
- Token budget per turn ~14k; avoid streaming full text unless essential
- Use cursors for paging; return next_cursor when available

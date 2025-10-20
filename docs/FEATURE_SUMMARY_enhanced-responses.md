# Enhanced Responses Feature Summary

**Status:** Awaiting Approval
**Created:** 2025-10-18
**Effort:** 2.5 days
**Risk:** Low

---

## Quick Overview

**Problem:** LLMs get truncated 200-char snippets instead of full section text and graph context.

**Solution:**
1. Add `verbosity` parameter to `search_documentation` (snippet | full | graph)
2. Add new `traverse_relationships` tool for deep graph exploration

**Impact:** LLMs can formulate complete, context-rich answers instead of partial responses.

---

## What's Being Built

### Feature 1: Verbosity Levels

```json
// Current (snippet mode - default)
"evidence": [{"snippet": "Add clients that are always part of the clust..."}]

// New: full mode
"evidence": [{"full_text": "To configure persistent clients...", "title": "..."}]

// New: graph mode
"evidence": [{
  "full_text": "...",
  "related_entities": [{"name": "weka cluster add-client", "label": "Command"}],
  "related_sections": [{"title": "4. Create a cluster", "distance": 1}]
}]
```

### Feature 2: Graph Traversal Tool

```json
{
  "name": "traverse_relationships",
  "arguments": {
    "start_ids": ["6f37b...", "a9e5b..."],
    "rel_types": ["MENTIONS", "CONTAINS_STEP"],
    "max_depth": 2
  }
}

// Returns: nodes, relationships, paths
```

---

## Documents

| File | Purpose | Lines |
|------|---------|-------|
| `feature-spec-enhanced-responses.md` | What & Why | 455 |
| `implementation-plan-enhanced-responses.md` | How to Build | 886 |
| This file | Quick Reference | — |

**Read in order:**
1. This summary (you are here)
2. Feature spec (for business justification)
3. Implementation plan (for technical details)

---

## Key Decisions

### Why Option 3 (Verbosity) + Option 2 (Traverse)?

**Flexibility:**
- LLM can choose detail level based on query complexity
- Single tool call for most cases (verbosity)
- Deep dive available when needed (traverse)

**Backwards Compatibility:**
- Default `verbosity="snippet"` = current behavior
- Existing clients work without changes
- No breaking changes

**Performance:**
- snippet: P95 = 70ms (current)
- full: P95 < 100ms (target)
- graph: P95 < 150ms (target)
- traverse: P95 < 200ms (target)

All modes remain **< 500ms** (spec requirement).

---

## Task Breakdown

| Task | Component | Effort | Owner |
|------|-----------|--------|-------|
| E1 | Add verbosity to ResponseBuilder | 4h | Backend |
| E2 | Wire to QueryService & MCP | 2h | Backend |
| E3 | Neo4j relationship queries | 3h | Graph Eng |
| E4 | Implement traverse tool | 4h | Backend + Graph |
| E5 | Observability (metrics/logs) | 1h | SRE |
| E6 | Performance validation | 2h | QA |
| E7 | Documentation | 2h | Docs |
| **Total** | | **18h** | **(2.5 days)** |

---

## Testing Strategy

**Unit Tests:**
- 3 verbosity modes (snippet, full, graph)
- Depth limits & whitelist validation
- Size limits enforced

**Integration Tests:**
- E2E via `/mcp/tools/call`
- Chained `search` → `traverse` workflow

**Performance Tests:**
- P95 latency for each mode
- Load test: 1000 requests
- No memory leaks

**Test Files:**
- `tests/test_response_builder.py` (modes)
- `tests/test_traversal.py` (traversal)
- `tests/test_mcp_integration.py` (E2E)
- `tests/perf/test_verbosity_latency.py` (performance)

---

## Security & Safety

**Size Limits:**
- Full text: 32KB per section
- Total response: 64KB max
- Related entities: 20 per section
- Traversal nodes: 100 max

**Input Validation:**
- Verbosity: Enum validation (reject invalid)
- Relationship types: Whitelist only
- Max depth: Hard cap at 3 hops
- All Cypher: Parameterized (no injection)

**Rate Limits:**
- Same as existing: 60 req/min
- Apply to all modes equally
- Monitor P95 by verbosity level

---

## Rollout Plan

**Day 1 (Internal):**
- Deploy to dev
- Run full test suite
- Validate metrics

**Day 2 (Staging):**
- Deploy with feature flag
- Enable for internal users
- Collect feedback

**Day 3 (Production):**
- Gradual: 10% → 50% → 100%
- Monitor error rates
- Alert if P95 > 400ms

---

## Success Metrics (30 days)

**Adoption:**
- [ ] `verbosity=full` > 20% of queries
- [ ] `verbosity=graph` > 10% of queries
- [ ] `traverse_relationships` > 5% of sessions

**Quality:**
- [ ] No P95 regressions for snippet mode
- [ ] Error rate < 0.1%
- [ ] LLM answer completeness +40%

**Performance:**
- [x] snippet: P95 = 70ms (baseline)
- [ ] full: P95 < 100ms
- [ ] graph: P95 < 150ms
- [ ] traverse: P95 < 200ms

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Large responses hit token limits | Medium | High | Enforce 64KB max, document limits |
| Graph mode too slow | Low | Medium | Cache entities, limit to 20/section |
| Traversal abuse (scraping) | Low | Medium | Rate limits + audit logs |
| Neo4j load spike | Low | Low | Connection pooling + timeouts |

---

## Dependencies

**No external dependencies** - uses existing:
- Neo4j driver (already connected)
- Qdrant client (already connected)
- Phase 2 query modules (already built)
- Prometheus metrics (already instrumented)

**Only new code:**
- `src/query/traversal.py` (~200 lines)
- Extensions to existing files (see implementation plan)

---

## Comparison to Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Separate query-service container | Clean separation | Network hop, complexity | ❌ Rejected |
| Always return full text | Simple | Token waste, perf hit | ❌ Rejected |
| **Verbosity parameter** | Flexible, backwards compatible | Slightly more complex | ✅ **Selected** |
| **+ traverse tool** | Deep exploration available | Additional tool to maintain | ✅ **Selected** |

---

## Approval Checklist

Before proceeding to implementation:

- [ ] Review feature spec (problem statement, use cases)
- [ ] Review implementation plan (tasks, tests, rollout)
- [ ] Confirm effort estimate (2.5 days acceptable)
- [ ] Agree on performance targets (P95 < 500ms)
- [ ] Approve backwards compatibility approach
- [ ] Sign off on security measures (size limits, whitelist)

**Approved by:** _________________
**Date:** _________________

---

## Next Steps After Approval

1. Create feature branch: `feature/enhanced-responses`
2. Implement tasks E1-E7 in order
3. Run test suite after each task
4. Deploy to dev for validation
5. Create PR with:
   - All code changes
   - Test results (junit.xml)
   - Performance benchmarks
   - Updated MCP tool schemas

---

## Questions?

**Q: Will this break existing MCP clients?**
A: No. Default `verbosity="snippet"` maintains current behavior exactly.

**Q: What if an LLM requests `graph` mode for every query?**
A: We monitor via metrics and can add cost hints to tool description.

**Q: Can we cache related entities to speed up graph mode?**
A: Yes, planned in Task E3 with LRU cache (1000 entries).

**Q: What about vector search on entities (not just sections)?**
A: Out of scope for this feature. Would be Phase 4 enhancement.

**Q: How do we prevent traversal from being used to scrape the entire graph?**
A: Hard caps (100 nodes, depth 3), rate limits, relationship whitelist, audit logs.

---

**For detailed specs, see:**
- `/docs/feature-spec-enhanced-responses.md` (complete specification)
- `/docs/implementation-plan-enhanced-responses.md` (technical plan)

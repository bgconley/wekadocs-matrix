<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **wekadocs-matrix** (29861 symbols, 39505 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/wekadocs-matrix/context` | Codebase overview, check index freshness |
| `gitnexus://repo/wekadocs-matrix/clusters` | All functional areas |
| `gitnexus://repo/wekadocs-matrix/processes` | All execution flows |
| `gitnexus://repo/wekadocs-matrix/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Compaction / Resume Discipline

- After every context compaction, summary handoff, or resumed implementation turn, re-read this `AGENTS.md` file before continuing any work. Treat the refreshed file as the active repo-local operating contract.
- Do not rely on memory, a prior summary, or an earlier read of this file when a compaction or resume has occurred.

## Meaningful Red/Green Test Discipline

- Red/green tests must prove behavior, contracts, invariants, integration shape, or regression boundaries. A test whose red state is only "module does not exist", "symbol cannot be imported", or "file is missing" is not a meaningful RED test.
- When adding a new module, file, or public symbol, create the thinnest viable skeleton first if necessary, then write the RED test against the required behavior or contract so the failure demonstrates the real missing capability.
- Verify and record the RED failure before implementation. The expected failure should be an assertion or contract failure, not an import/setup/collection failure.
- Keep GREEN minimal: implement only enough behavior to satisfy the observed RED test, then rerun the focused test and any relevant surrounding checks.

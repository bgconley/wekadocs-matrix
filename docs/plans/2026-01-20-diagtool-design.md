# diagtool Design Document

**Date:** 2026-01-20
**Status:** Approved
**Author:** Claude + Brennan

## Overview

`diagtool` is a CLI utility for troubleshooting and analyzing MCP retrieval operations. It provides session replay, query deep-dives, pattern analysis, and live monitoring capabilities.

## Commands

### `diagtool replay`

Replay all tool calls from a session in chronological order.

```bash
diagtool replay                    # Show session picker
diagtool replay 1                  # Replay session #1 from picker
diagtool replay --last             # Most recent session
diagtool replay --date 2026-01-19  # Different day
diagtool replay 1 --detail         # Show provenance for each result
diagtool replay 1 --full           # Full JSON payloads
diagtool replay 1 --errors         # Only show errors
diagtool replay 1 --tool kb_search # Filter by tool name
```

**Session picker:** When no session specified, shows numbered table:
- `#` - Sequential number for easy reference
- `Time` - Session start time
- `Queries` - Number of search queries
- `Tools` - Total tool calls
- `Errors` - Error count (highlighted if > 0)
- `First Topic` - Inferred from first query text

**Detail levels:**
- Default: Compact summary per tool call
- `--detail`: Adds provenance panel (sources, pre/post rerank positions, scores)
- `--full`: Complete JSON request/response

### `diagtool query`

Deep dive into a single query diagnostic.

```bash
diagtool query --id <diagnostic-id>
diagtool query --last              # Most recent query
diagtool query 1.3                 # Session 1, query #3
diagtool query --id <id> --show-text    # Include chunk text
diagtool query --id <id> --compare <id> # Side-by-side comparison
diagtool query --id <id> --export json  # Export to file
```

**Output sections:**
1. Query info (original, rewritten, reason)
2. Pipeline visualization (stages with candidate counts and latencies)
3. All candidates table (scores per source, rerank impact, final ranking)
4. Timing breakdown (visual bar chart)

### `diagtool stats`

Aggregate statistics and pattern analysis.

```bash
diagtool stats                     # Today's summary
diagtool stats --date 2026-01-19   # Specific day
diagtool stats --week              # Last 7 days
diagtool stats --query "CORS"      # Filter by query pattern
diagtool stats --failed            # Only problematic queries
```

**Output sections:**
1. Overview (sessions, queries, tool calls, errors)
2. Pipeline health (per-component status, latency, success rate)
3. Latency distribution (p50, p90, p99 with visual bars)
4. Error summary (grouped by error type)
5. Query patterns (topic clustering with rerank impact analysis)

### `diagtool watch`

Live monitoring of MCP traffic.

```bash
diagtool watch                     # All traffic
diagtool watch --tools             # Only tool calls
diagtool watch --errors            # Only errors
diagtool watch --tool kb_search    # Specific tool
diagtool watch --excerpt-len 500   # Longer text excerpts
diagtool watch --no-excerpt        # Metadata only
diagtool watch --full              # Complete text
```

**Output:**
- Real-time streaming from MCP log
- Request/response pairs with timing
- Text excerpts from returned passages (default 280 chars)
- Running session stats at bottom
- Smart error hints (e.g., detecting type coercion issues)

## Visual Design Language

### Color Scheme (Semantic)

| Color   | Usage                                    |
|---------|------------------------------------------|
| Green   | Success, active/enabled, good values     |
| Red     | Errors, failures, warnings               |
| Yellow  | In-progress, waiting, caution            |
| Blue    | Informational, headers, labels           |
| Magenta | Highlights, important values, user input |
| Dim     | Secondary info, disabled, truncated      |

### Status Indicators

```
 ✓  Success (green)
 ✗  Error (red)
 ▶  Request/outgoing (blue)
 ◀  Response/incoming (green)
 ●  Active/enabled (green)
 ○  Disabled/inactive (dim)
 ⏳ Waiting/in-progress (yellow)
```

### Section Separation

- **Primary headers:** Double-line box with background color
- **Section headers:** Single-line bordered box
- **Timestamp dividers:** Thin rule with context
- **Continuation lines:** Vertical pipes for visual flow
- **Tables:** Bold blue headers, subtle alternating rows, right-aligned numbers

## Data Sources

| Source | Location | Contents |
|--------|----------|----------|
| Diagnostic JSONL | `reports/retrieval_diagnostics/{date}/retrieval_diagnostics.jsonl` | Full query data, scores, provenance |
| Diagnostic MD | `reports/retrieval_diagnostics/{date}/*.md` | Human-readable summaries |
| MCP Log | `~/Library/Logs/Claude/mcp-server-wekadocs-matrix.log` | Raw JSON-RPC traffic |

## Technology Stack

- **Python 3.11+**
- **Typer** - CLI framework with automatic help generation
- **Rich** - Terminal formatting, tables, colors, panels

## File Location

```
scripts/diagtool.py    # Single-file implementation
```

## Implementation Notes

### Session ID Resolution

Sessions are identified by `session_id` in diagnostic records. The tool:
1. Scans JSONL for unique session IDs
2. Groups queries by session
3. Assigns sequential numbers (1, 2, 3...) for easy reference
4. Extracts first query as "topic" hint

Users can reference sessions by:
- Number from picker (`1`, `2`)
- Raw session ID (`server-abc123...`)
- Shortcuts (`--last`, `--today`)

### Provenance Tracking

Each search result includes:
- `source`: Where candidate originated (dense/sparse/colbert/graph/bm25)
- `pre_rank`: Position before reranking
- `rerank_score`: Reranker confidence (0-1)
- `final_score`: Score after fusion
- `vector_scores`: Per-field scores (content, title, doc_title)

### Live Monitoring

The `watch` command:
1. Tails the MCP log file
2. Parses JSON-RPC messages
3. Matches request/response pairs by ID
4. Extracts and formats relevant fields
5. Streams to terminal with Rich Live display

## Future Enhancements

- Export to HTML report
- Comparison mode for A/B testing retrieval changes
- Integration with OpenTelemetry traces
- Saved filters/presets

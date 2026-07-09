# Release Proof Implementation Plan (Phase 6)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> This is **Slice D** — the final slice of the Refactor-Roadmap wrap-up (`docs/repo_audit/10_refactor_roadmap.md`, Phase 6: "Run fresh Nutanix ingest on a representative corpus; run golden retrieval/citation/evidence tests; compare fallback/degradation metrics; produce install-from-scratch proof"). Run **after A, B, C**.

**Goal:** Produce a reproducible release-proof artifact showing a from-scratch Nutanix install ingests a representative corpus and passes golden retrieval + citation/evidence acceptance with observable fallback metrics.

**Architecture:** **Reuse the existing eval harness** — `scripts/smoke_test_golden_queries.py` already loads the golden set, runs it through `QueryService`, and pass/fails at `hit_rate ≥ 0.7`; `scripts/run_canonical_retrieval_benchmark.py` and `scripts/evaluate_retrieval.py` complement it. The plan is split into **Part 1 (offline-preparable now)** — wire the curated Nutanix golden fixtures in, add a citation/evidence check, and write the install runbook + proof-capture script — and **Part 2 (requires live services)** — actually run the ingest + eval + install proof.

**Tech Stack:** Python, pytest, ripgrep, docker compose, the existing eval scripts, the model gateway (embedders/reranker/Qdrant/Neo4j/GLiNER).

**Execution note:** Create/overwrite files with the Write tool and edit with Edit (apply_patch-style), not shell heredoc redirection. **Commit messages must pass gitlint** (enforced at `commit-msg`): title ≤72 chars with a `(pN.N)` numeric scope, a blank line, then a `Phase N.N — <recap>` body whose lines are ≤80 chars. `git commit` examples are shown title-first — append the `-m "Phase 6.0 — …"` body when committing.

---

## Critical Context

- **Part 2 is LIVE-GATED.** It requires the full model stack (embedders, reranker, Qdrant, Neo4j, GLiNER, unified gateway — in this project the GPU host). **Do not run Part 2 steps unless the user confirms services are available.** Part 1 is fully offline-preparable and should be done first.
- **The golden fixtures were already curated to Nutanix** in the residue slice: `tests/fixtures/golden_query_set.yaml`, `tests/fixtures/baseline_query_set.yaml`, `tests/eval/queries.yaml`. This plan wires them into the runners and re-checks the acceptance bar for the new corpus.
- **Existing entrypoints:** ingest = `scripts/ingestctl`; reset = `scripts/reset_datastores.py`; golden eval = `scripts/smoke_test_golden_queries.py` (+ `scripts/run_canonical_retrieval_benchmark.py`); degradation = `src/mcp_server/retrieval_trace.py` (`RetrievalTraceBuilder`); install = `docker-compose.yml` (neo4j, qdrant, redis, mcp-server, ingestion-worker/service, alloy) + `deploy/scripts/`.
- **Open decision:** the `hit_rate ≥ 0.7` threshold was set for the WEKA corpus. After the Nutanix corpus swap it may need recalibration — treat the first live run as a **calibration** run, not a hard gate, and set the accepted threshold explicitly in the completion report.

---

## PART 1 — Offline Preparation (do now, no services)

### Task 1: Wire The Curated Nutanix Golden Fixtures Into The Runners

**Files:** Modify `scripts/smoke_test_golden_queries.py`, `scripts/run_canonical_retrieval_benchmark.py` (fixture paths/expectations only).

- [ ] **Step 1: Confirm the fixture wiring (already points at the curated set)**

`scripts/smoke_test_golden_queries.py` already defaults `--queries` to `tests/fixtures/golden_query_set.yaml` (line ~325), which was curated to Nutanix in the residue slice — so **no repathing is needed for the smoke test**. Verify the benchmark runner similarly:

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
rg -n "golden_query_set|baseline_query|queries.yaml|default=" scripts/smoke_test_golden_queries.py scripts/run_canonical_retrieval_benchmark.py | head
```

Only repath the benchmark runner if it still references an old/generic fixture. The substantive work is Step 2 (aligning expected titles/topics to the Nutanix corpus).

- [ ] **Step 2: Align expected doc-titles/topics to the Nutanix corpus**

The golden set's `expected_titles`/`expected_topics`/matched-doc assertions must reference Nutanix documents that will exist after ingest (Nutanix Files, Objects, Prism Central, NCI/AOS/AHV…). Update any residual generic expectations. Keep query `id`s stable.

- [ ] **Step 3: Offline sanity — fixtures parse and the runner imports/collects**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
python -c "import yaml,pathlib; [print('ok',p) or yaml.safe_load(pathlib.Path(p).read_text()) for p in ['tests/fixtures/golden_query_set.yaml','tests/eval/queries.yaml']]"
python -m py_compile scripts/smoke_test_golden_queries.py scripts/run_canonical_retrieval_benchmark.py
python scripts/smoke_test_golden_queries.py --help
```

Expected: fixtures parse; scripts compile; `--help` prints (proves arg wiring without touching services).

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test_golden_queries.py scripts/run_canonical_retrieval_benchmark.py tests/fixtures/golden_query_set.yaml tests/eval/queries.yaml
git commit -m "test(p6.0): point golden-eval runners at curated Nutanix fixtures" \
  -m "Phase 6.0 — align golden/benchmark runners + expectations to Nutanix corpus."
```

### Task 2: Add A Citation/Evidence Acceptance Check

**Files:** Create `scripts/eval/check_evidence_citations.py`; Create `tests/test_evidence_citation_scoring.py` (NOT under `tests/eval/` — that path is `.gitignore`d).

Phase 6 wants *citation/evidence* proof, not just retrieval hit@k. Build a small checker over `kb.retrieve_evidence` that asserts each curated query returns ≥1 quote with a non-empty `doc_tag`/`quote` (citation provenance), and score the pass rate.

- [ ] **Step 1: Write the scoring function (pure, offline-testable)**

Create `scripts/eval/check_evidence_citations.py` with a pure scorer plus a
small CLI. It accepts both the direct `quotes` shape and nested
`structured_content.quotes` / `evidence.quotes` shapes, because the release
proof will consume artifacts from more than one existing runner.

```python
#!/usr/bin/env python3
"""Score citation/evidence coverage from retrieval JSON artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _quotes_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(item.get("quotes"), list):
        return item["quotes"]
    structured = item.get("structured_content") or item.get("structuredContent") or {}
    if isinstance(structured, dict) and isinstance(structured.get("quotes"), list):
        return structured["quotes"]
    evidence = item.get("evidence") or {}
    if isinstance(evidence, dict) and isinstance(evidence.get("quotes"), list):
        return evidence["quotes"]
    return []


def _is_cited_quote(quote: dict[str, Any]) -> bool:
    quote_text = str(quote.get("quote") or quote.get("text") or "").strip()
    doc_tag = str(quote.get("doc_tag") or quote.get("docTag") or "").strip()
    return bool(quote_text and doc_tag)


def score_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    cited = 0
    uncited = 0
    details: list[dict[str, Any]] = []
    for idx, item in enumerate(results, start=1):
        query_id = item.get("id") or item.get("query_id") or f"q{idx}"
        quotes = _quotes_from_item(item)
        ok = any(_is_cited_quote(q) for q in quotes if isinstance(q, dict))
        if ok:
            cited += 1
        else:
            uncited += 1
        details.append({"id": query_id, "cited": ok, "quote_count": len(quotes)})
    total = cited + uncited
    return {
        "cited": cited,
        "uncited": uncited,
        "total": total,
        "pass_rate": cited / total if total else 0.0,
        "details": details,
    }


def _load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "queries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"No result list found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score citation evidence coverage.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args()

    score = score_evidence(_load_results(args.input))
    text = json.dumps(score, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if score["pass_rate"] >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Offline unit test with a fake evidence payload**

```python
# tests/test_evidence_citation_scoring.py
from scripts.eval.check_evidence_citations import score_evidence


def test_score_counts_cited_vs_uncited():
    fake = [
        {"id": "direct", "quotes": [{"quote": "x", "doc_tag": "nutanix/files"}]},
        {
            "id": "nested",
            "structured_content": {
                "quotes": [{"text": "y", "docTag": "nutanix/prism"}],
            },
        },
        {"id": "uncited", "quotes": [{"quote": "z"}]},
    ]
    s = score_evidence(fake)
    assert s["cited"] == 2
    assert s["uncited"] == 1
    assert s["pass_rate"] == 2 / 3
    assert [d["id"] for d in s["details"]] == ["direct", "nested", "uncited"]
```

```bash
python -m py_compile scripts/eval/check_evidence_citations.py
pytest tests/test_evidence_citation_scoring.py -q   # expect: 1 passed
```

- [ ] **Step 3: Commit**

```bash
git add scripts/eval/check_evidence_citations.py tests/test_evidence_citation_scoring.py
git commit -m "test(p6.0): offline-scored citation/evidence acceptance check" \
  -m "Phase 6.0 — add score_evidence() citation check + offline unit test."
```

### Task 3: Write The Install-From-Scratch Runbook + Proof-Capture Script

**Files:** Create `docs/RELEASE-PROOF.md` (runbook); Create `scripts/release_proof.sh` (orchestrator, guarded).

- [ ] **Step 1: Author the proof-capture script (does NOT auto-run live steps)**

`scripts/release_proof.sh` documents and sequences the live proof but requires an explicit `--live` flag to execute the service-touching steps; without it, it prints the plan (dry run). Use this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

LIVE=0
if [[ "${1:-}" == "--live" ]]; then
  LIVE=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--live]" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="${RELEASE_PROOF_ARTIFACT_DIR:-artifacts/release-proof/${STAMP}}"
CORPUS_PATH="${NUTANIX_CORPUS_PATH:-data/ingest/nutanix}"
GOLDEN_TOP_K="${GOLDEN_TOP_K:-10}"
CITATION_MIN_PASS_RATE="${CITATION_MIN_PASS_RATE:-0.8}"

run() {
  echo "+ $*"
  if [[ "$LIVE" == "1" ]]; then
    "$@"
  fi
}

if [[ "$LIVE" != "1" ]]; then
  echo "DRY RUN: pass --live to execute service-touching commands."
fi

run mkdir -p "$ARTIFACT_DIR"

run docker compose up -d
run python scripts/reset_datastores.py
run scripts/ingestctl ingest "$CORPUS_PATH" --tag nutanix --json

run python scripts/smoke_test_golden_queries.py \
  --queries tests/fixtures/golden_query_set.yaml \
  --top-k "$GOLDEN_TOP_K" \
  --report "$ARTIFACT_DIR/golden.json"

run python scripts/run_canonical_retrieval_benchmark.py \
  --queries tests/fixtures/canonical_retrieval_benchmark.yaml \
  --report "$ARTIFACT_DIR/canonical.json"

run python scripts/eval/check_evidence_citations.py \
  --input "$ARTIFACT_DIR/canonical.json" \
  --output "$ARTIFACT_DIR/citations.json" \
  --min-pass-rate "$CITATION_MIN_PASS_RATE"

echo "Release proof artifacts: $ARTIFACT_DIR"
```

Before committing this script, verify each invoked flag against `--help`. If a runner does not expose the exact flag shown above, update this plan's command and the script together; do not leave a known-bad runbook. The ingest step intentionally omits `ingestctl --watch`; that flag is currently an unimplemented stub that exits non-zero. Without `--watch` or `--no-wait`, `ingestctl ingest` uses its implemented monitoring path.

- [ ] **Step 2: Author `docs/RELEASE-PROOF.md`**

A runbook with this concrete structure:

````markdown
# Release Proof

This runbook captures the install-from-scratch proof for the Nutanix docs matrix.
It is split into offline checks and live proof because live proof requires the
model stack, Qdrant, Neo4j, Redis, and the ingestion worker.

## Prerequisites

- Branch: `wip/weka-to-nutanix-migration`
- Docker compose services available locally or on the configured GPU host.
- Required secrets exported: `NEO4J_PASSWORD`, `REDIS_PASSWORD`, `JWT_SECRET`.
- Nutanix corpus available at `${NUTANIX_CORPUS_PATH:-data/ingest/nutanix}`.

## Offline Checks

```bash
python -m py_compile scripts/smoke_test_golden_queries.py \
  scripts/run_canonical_retrieval_benchmark.py \
  scripts/eval/check_evidence_citations.py
pytest tests/test_evidence_citation_scoring.py -q
pytest tests/test_release_proof_script.py -q
bash -n scripts/release_proof.sh
bash scripts/release_proof.sh
NEO4J_PASSWORD=x REDIS_PASSWORD=x JWT_SECRET=x docker compose config >/dev/null
```

The release proof ingest step intentionally omits `ingestctl --watch` because
that flag is not implemented in the current CLI and exits non-zero. Without
`--watch` or `--no-wait`, `ingestctl ingest` uses its implemented progress
monitoring path.

## Live Proof

Run only after the user confirms services are available:

```bash
bash scripts/release_proof.sh --live
```

Artifacts land under `artifacts/release-proof/<timestamp>/` unless
`RELEASE_PROOF_ARTIFACT_DIR` is set.

## Acceptance Bars

- Golden retrieval hit rate: record the measured rate. The first Nutanix live
  run is calibration; do not weaken the bar silently.
- Citation/evidence pass rate: default `CITATION_MIN_PASS_RATE=0.8`.
- Completion report must record ingest counts, retrieval rate, citation rate,
  fallback/degradation events, artifact paths, and any calibrated threshold.

## Backup / Restore

Use the existing scripts under `deploy/scripts/` before destructive live runs
when preserving a prior local corpus matters.
````

- [ ] **Step 3: Offline verify the script is well-formed (dry run only)**

```bash
cd /Users/brennanconley/vibecode/wekadocs-matrix
python -m py_compile scripts/eval/check_evidence_citations.py
pytest tests/test_evidence_citation_scoring.py -q
python scripts/run_canonical_retrieval_benchmark.py --help
python scripts/ingestctl --help
bash -n scripts/release_proof.sh          # syntax check, no execution
bash scripts/release_proof.sh             # no --live flag => prints the dry plan, touches nothing
NEO4J_PASSWORD=x REDIS_PASSWORD=x JWT_SECRET=x docker compose config >/dev/null && echo "compose config valid"
```

Expected: citation test passes; command help prints; syntax OK; dry plan prints; compose config validates.

- [ ] **Step 4: Commit — Part 1 complete**

```bash
git add scripts/release_proof.sh docs/RELEASE-PROOF.md
git commit -m "docs(p6.0): install-from-scratch runbook + guarded proof-capture script" \
  -m "Phase 6.0 — add release-proof runbook + --live-guarded proof-capture script."
git push origin wip/weka-to-nutanix-migration
```

> **STOP HERE if live services are unavailable.** Part 1 is a self-contained, mergeable deliverable: the harness, fixtures, and runbook are ready to execute the moment the stack is up.

---

## PART 2 — Live Execution (requires the model stack — user must confirm services are up)

### Task 4: Clean-Volume Bring-Up (install-from-scratch proof)

- [ ] **Step 1:** From a clean checkout with empty volumes: `bash scripts/release_proof.sh --live` (or the documented `docker compose up -d` + health wait). Capture the bring-up log as the install proof. Then `python scripts/reset_datastores.py …` to guarantee empty datastores.

### Task 5: Fresh Nutanix Ingest

- [ ] **Step 1:** Ingest the representative Nutanix corpus via `scripts/ingestctl` (path/manifest per `docs/RELEASE-PROOF.md`). Capture the ingest report (`reports/ingest/<id>/ingest_report.md`) — record doc/section/entity counts.

### Task 6: Golden Retrieval + Citation/Evidence Eval

- [ ] **Step 1:** Run `python scripts/smoke_test_golden_queries.py --report artifacts/golden.json` (hit_rate) and `python scripts/run_canonical_retrieval_benchmark.py`. Run `python scripts/eval/check_evidence_citations.py` (citation pass_rate). Record both scores. **First run = calibration**: if hit_rate < 0.7 on the new corpus, investigate misaligned expectations vs a real regression before treating it as failure.

### Task 7: Fallback / Degradation Capture

- [ ] **Step 1:** For the golden queries, capture `RetrievalTraceBuilder` output (signal-pool/reranker/ColBERT/graph facets + any fallback/degradation events). Summarize which stages engaged vs degraded — this is the "compare fallback/degradation metrics" deliverable.

### Task 8: Completion Report

- [ ] **Step 1:** Write `docs/superpowers/findings/2026-07-<dd>-release-proof-completion.md` with: install-from-scratch log summary, ingest counts, golden hit_rate, citation pass_rate, the accepted thresholds (with calibration rationale), and the fallback/degradation summary. Attach/reference the `artifacts/`. Commit + push. **Do not claim quality beyond what the run measured.**

---

## Acceptance Criteria

**Part 1 (offline, must all hold):**
- Golden-eval runners load the curated Nutanix fixtures; expectations reference Nutanix docs; runners import/`--help` cleanly.
- Citation/evidence scorer exists with a passing offline unit test.
- `scripts/release_proof.sh` (guarded, dry-runs without `--live`) and `docs/RELEASE-PROOF.md` runbook exist; compose config validates.
- Part 1 committed and pushed as a standalone deliverable.

**Part 2 (live, when services available):**
- Clean-volume bring-up succeeds (install-from-scratch proof captured).
- Fresh Nutanix ingest completes with recorded counts.
- Golden hit_rate and citation pass_rate recorded; accepted thresholds documented with calibration rationale.
- Fallback/degradation summary captured from traces.
- Completion report committed; no unmeasured quality claims.

## Self-Review

- Reuses the existing eval harness rather than rebuilding it.
- Honest offline/live split: Part 1 is a complete, mergeable deliverable; Part 2 is explicitly gated on services and never auto-runs live steps without `--live`.
- The `0.7` threshold is treated as calibration-on-first-run, not an assumed pass bar for a changed corpus.
- No live model/datastore quality is claimed from offline work.

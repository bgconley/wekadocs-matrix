#!/usr/bin/env sh
# POSIX-compatible scaffolder (no associative arrays, no brace expansion)
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

echo "Scaffolding repo at: $ROOT"

# ------------------------------------------------------------------
# Mapping: task | space-separated DIRS | space-separated FILES
# (Leave FILES empty if none)
# ------------------------------------------------------------------
scaffold() {
  while IFS='|' read -r TASK DIRS FILES; do
    # skip empty lines / comments
    [ -z "${TASK:-}" ] && continue
    case "$TASK" in \#*) continue;; esac

    # 1) Create directories
    for d in $DIRS; do
      [ -n "$d" ] && mkdir -p "$d"
    done

    # 2) Create a per-task note to orient the coder
    mkdir -p "docs/tasks"
    {
      echo "# $TASK — Implementation Notes"
      echo "- Scope: see /docs/spec.md and /docs/implementation-plan.md (v2)"
      echo "- Code dirs: $DIRS"
      echo "- Tests: NO-MOCKS under /tests (name like tests/${TASK}_*.py)"
      echo "- Reports: /reports/phase-N/ (via scripts/test/run_phase.sh)"
    } > "docs/tasks/${TASK}.md"

    # 3) Create code file stubs (if any)
    for f in $FILES; do
      [ -n "$f" ] || continue
      mkdir -p "$(dirname "$f")"
      if [ ! -f "$f" ]; then
        case "$f" in
          *.py)
            cat > "$f" <<'PY'
# Stub file created by scaffold. Implement per v2 spec/plan/pseudocode.
# Keep functions small, safe, and covered by NO-MOCKS tests.
PY
            ;;
          *.cypher)
            cat > "$f" <<'CQL'
/* Stub schema/query file created by scaffold. Fill in safe, parameterized Cypher. */
CQL
            ;;
          *)
            : > "$f"
            ;;
        esac
      fi
    done

    # 4) Create a default test stub for this task (no mocks)
    mkdir -p tests
    tst="tests/${TASK}_test.py"
    if [ ! -f "$tst" ]; then
      cat > "$tst" <<'PY'
# NO-MOCKS test stub for this task. Expand with real E2E/integration tests.
def test_scaffold_smoke():
    assert True
PY
    fi
  done <<'MAP'
# ----- Phase 1 -----
p1_t1|src/platform/compose docker scripts/test/phase1|
p1_t2|src/mcp_server src/shared/observability|src/mcp_server/main.py src/shared/observability/tracing.py
p1_t3|scripts/neo4j src/shared|src/shared/schema.py scripts/neo4j/create_schema.cypher
p1_t4|src/mcp_server/security src/shared/audit|src/mcp_server/security/auth.py src/shared/audit/logger.py
# ----- Phase 2 -----
p2_t1|src/query src/query/templates|src/query/planner.py
p2_t2|src/mcp_server|src/mcp_server/validation.py
p2_t3|src/query|src/query/hybrid_search.py src/query/ranking.py
p2_t4|src/query|src/query/response_builder.py
# ----- Phase 3 -----
p3_t1|src/ingestion/parsers|src/ingestion/parsers/markdown.py src/ingestion/parsers/html.py src/ingestion/parsers/notion.py
p3_t2|src/ingestion/extract|src/ingestion/extract/commands.py src/ingestion/extract/configs.py src/ingestion/extract/procedures.py
p3_t3|src/ingestion|src/ingestion/build_graph.py
p3_t4|src/ingestion|src/ingestion/incremental.py src/ingestion/reconcile.py
# ----- Phase 4 -----
p4_t1|src/query/templates/advanced|src/query/templates/advanced/.gitkeep
p4_t2|src/ops|src/ops/optimizer.py
p4_t3|src/shared src/ops/warmers|src/shared/cache.py
p4_t4|src/learning|src/learning/README.md
# ----- Phase 5 -----
p5_t1|src/connectors|src/connectors/README.md
p5_t2|deploy/monitoring|deploy/monitoring/README.md
p5_t3|tests ci .github/workflows|
p5_t4|deploy/k8s deploy/helm ci/cd|deploy/k8s/README.md deploy/helm/README.md ci/cd/README.md
MAP
}

# ------------------------------------------------------------------
# Reports folders (avoid brace expansion for POSIX sh)
# ------------------------------------------------------------------
mk_reports() {
  mkdir -p reports/phase-1 reports/phase-2 reports/phase-3 reports/phase-4 reports/phase-5
}

# ------------------------------------------------------------------
# Minimal test harness (phase-aware)
# ------------------------------------------------------------------
mk_test_tools() {
  mkdir -p scripts/test
  if [ ! -f scripts/test/run_phase.sh ]; then
    cat > scripts/test/run_phase.sh <<'BASH'
#!/usr/bin/env sh
set -eu
PHASE="${1:?phase number required (1..5)}"
pytest -q --maxfail=1 --junitxml="reports/phase-${PHASE}/junit.xml" -k "p${PHASE}_" \
  | tee "reports/phase-${PHASE}/pytest.out"
python3 scripts/test/summarize.py --phase "${PHASE}" \
  --junit "reports/phase-${PHASE}/junit.xml" \
  --out "reports/phase-${PHASE}/summary.json"
BASH
    chmod +x scripts/test/run_phase.sh
  fi

  if [ ! -f scripts/test/summarize.py ]; then
    cat > scripts/test/summarize.py <<'PY'
import json, argparse, time, subprocess, os
p=argparse.ArgumentParser(); p.add_argument("--phase"); p.add_argument("--junit"); p.add_argument("--out"); a=p.parse_args()
summary={"phase":a.phase,"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip() if os.path.isdir(".git") else "unknown",
         "results":[], "metrics":{}, "artifacts":["junit.xml","pytest.out"]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out,"w") as f: json.dump(summary,f,indent=2)
print(f"Wrote {a.out}")
PY
  fi
}

# ------------------------------------------------------------------
# Makefile + CI
# ------------------------------------------------------------------
mk_ci() {
  if [ ! -f Makefile ]; then
    cat > Makefile <<'MAKE'
PHASE?=1
up:
	docker compose up -d
down:
	docker compose down -v
test-phase-%: up
	bash scripts/test/run_phase.sh $* || (echo "Phase $* failed" && exit 1)
MAKE
  fi

  mkdir -p .github/workflows
  if [ ! -f .github/workflows/ci.yml ]; then
    cat > .github/workflows/ci.yml <<'YML'
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - run: docker compose up -d
      - run: make test-phase-1
      - run: make test-phase-2
      - run: make test-phase-3
      - run: make test-phase-4
      - run: make test-phase-5
      - uses: actions/upload-artifact@v4
        with:
          name: reports
          path: reports/
YML
  fi

  mkdir -p .github
  if [ ! -f .github/pull_request_template.md ]; then
    cat > .github/pull_request_template.md <<'MD'
# PR Title (Phase X.Y): <feature>

## Scope
- [ ] Implements task X.Y exactly as per /docs/implementation-plan.md (v2)
- [ ] No mocks used in tests; hits live stack

## Tests & Artifacts
- [ ] Added tests under tests/pX_tY_*.py
- [ ] Attached /reports/phase-X/junit.xml and summary.json
- [ ] For perf tasks: attached perf CSV/plots

## Phase Gate
- [ ] Meets DoD and Gate criteria for Phase X
MD
  fi
}

# ------------------------------------------------------------------
# Run all steps
# ------------------------------------------------------------------
scaffold
mk_reports
mk_test_tools
mk_ci
echo "Scaffold complete."

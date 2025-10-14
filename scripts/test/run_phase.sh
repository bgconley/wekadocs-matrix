#!/usr/bin/env bash
# Phase-aware test runner with scoped coverage reporting.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <phase-number>"
  exit 1
fi

PHASE="$1"
REPORTS_DIR="reports/phase-${PHASE}"

case "${PHASE}" in
  1) COV_PKGS="src/mcp_server src/shared src/scripts/neo4j" ;;
  2) COV_PKGS="src/query src/shared" ;;
  3) COV_PKGS="src/ingestion src/shared" ;;
  4) COV_PKGS="src/query src/shared" ;;
  5) COV_PKGS="src/ops src/connectors src/shared" ;;
  *)
    echo "Unknown phase '${PHASE}'"
    exit 2
    ;;
esac

TEST_PATTERN="tests/p${PHASE}_*.py"
COV_ARGS=()
for pkg in ${COV_PKGS}; do
  COV_ARGS+=(--cov="${pkg}")
done

echo "=== Running Phase ${PHASE} Tests ==="
echo "Reports will be written to: ${REPORTS_DIR}"

mkdir -p "${REPORTS_DIR}"

PYTEST_LOG="${REPORTS_DIR}/pytest.out"

echo "Running tests..."
pytest \
  -q \
  --maxfail=1 \
  ${TEST_PATTERN} \
  --junit-xml="${REPORTS_DIR}/junit.xml" \
  "${COV_ARGS[@]}" \
  --cov-branch \
  --cov-report=xml:"${REPORTS_DIR}/coverage.xml" \
  --cov-report=html:"${REPORTS_DIR}/coverage" \
  --cov-report=term \
  | tee "${PYTEST_LOG}"

PYTEST_EXIT_CODE=${PIPESTATUS[0]}

python scripts/test/summarize.py \
  --phase "${PHASE}" \
  --junit "${REPORTS_DIR}/junit.xml" \
  --out "${REPORTS_DIR}/summary.json"

if [ "${PHASE}" = "3" ]; then
  if ! python scripts/test/check_phase3_metrics.py "${REPORTS_DIR}/summary.json"; then
    PYTEST_EXIT_CODE=1
  fi
fi

if [ -f .coveragerc ]; then
  GLOBAL_DIR="reports/global"
  echo "Generating global coverage report into ${GLOBAL_DIR}"
  mkdir -p "${GLOBAL_DIR}"
  coverage erase
  GLOBAL_PYTEST_LOG="${GLOBAL_DIR}/pytest.out"
  set +e
  COVERAGE_FILE="${GLOBAL_DIR}/.coverage" coverage run -m pytest -q \
    | tee "${GLOBAL_PYTEST_LOG}"
  GLOBAL_STATUS=${PIPESTATUS[0]}
  set -e
  coverage xml -o "${GLOBAL_DIR}/coverage.xml" || true
  coverage html -d "${GLOBAL_DIR}/coverage" >/dev/null 2>&1 || true
  if [ "${GLOBAL_STATUS}" -ne 0 ]; then
    echo "Global coverage run exited with status ${GLOBAL_STATUS}" >&2
  fi
fi

echo ""
echo "=== Reports Generated ==="
echo "JUnit XML: ${REPORTS_DIR}/junit.xml"
echo "Summary JSON: ${REPORTS_DIR}/summary.json"
echo "Coverage XML: ${REPORTS_DIR}/coverage.xml"
echo "Coverage HTML: ${REPORTS_DIR}/coverage/index.html"
echo ""

if [ "${PYTEST_EXIT_CODE}" -eq 0 ]; then
  echo "✅ All Phase ${PHASE} tests passed!"
else
  echo "❌ Some Phase ${PHASE} tests failed. Check reports for details."
fi

# --- Add to scripts/test/run_phase.sh ---
elif [ "$PHASE" = "4" ]; then
  echo "[PHASE-4] Running tests..."
  pytest -q --maxfail=1 --junitxml="reports/phase-4/junit.xml" -k "p4_" | tee "reports/phase-4/pytest.out"

  # Reuse summarizer if present; it's ok if this step is a no-op for now.
  if [ -f "scripts/test/summarize.py" ]; then
    python scripts/test/summarize.py --phase "4" \
      --junit "reports/phase-4/junit.xml" \
      --out "reports/phase-4/summary.json" || true
  fi

  # Phase‑4 gate: all p4_ tests must pass, EXPLAIN index seek checked, cache perf checked
  python scripts/test/check_phase4_metrics.py \
    --junit "reports/phase-4/junit.xml" \
    --summary "reports/phase-4/summary.json"
fi


exit "${PYTEST_EXIT_CODE}"

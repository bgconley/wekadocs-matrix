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

exit "${PYTEST_EXIT_CODE}"

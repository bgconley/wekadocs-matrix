#!/bin/bash
# Implements Phase 1 testing infrastructure
# See: /docs/implementation-plan.md → Phase 1 Tests & Artifacts
# Run tests for a specific phase and generate reports

set -e

# Parse arguments
PHASE=${1:-1}
REPORTS_DIR="reports/phase-${PHASE}"

echo "=== Running Phase ${PHASE} Tests ==="
echo "Reports will be written to: ${REPORTS_DIR}"

# Create reports directory
mkdir -p "${REPORTS_DIR}"

# Run pytest for the specific phase with coverage and XML output
echo "Running tests..."
pytest \
    tests/p${PHASE}_*.py \
    -v \
    --tb=short \
    --junit-xml="${REPORTS_DIR}/junit.xml" \
    --cov=src \
    --cov-report=html:"${REPORTS_DIR}/coverage" \
    --cov-report=term \
    | tee "${REPORTS_DIR}/test_output.log"

# Extract test results summary
echo ""
echo "=== Test Summary ==="
PYTEST_EXIT_CODE=$?

# Generate summary.json
python3 - <<EOF
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

reports_dir = Path("${REPORTS_DIR}")
junit_file = reports_dir / "junit.xml"

summary = {
    "phase": ${PHASE},
    "timestamp": datetime.utcnow().isoformat(),
    "success": ${PYTEST_EXIT_CODE} == 0,
    "tests": {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0
    },
    "test_details": []
}

if junit_file.exists():
    tree = ET.parse(junit_file)
    root = tree.getroot()

    # Get totals from testsuite element
    for testsuite in root.findall('testsuite'):
        summary["tests"]["total"] = int(testsuite.get('tests', 0))
        summary["tests"]["passed"] = summary["tests"]["total"] - int(testsuite.get('failures', 0)) - int(testsuite.get('errors', 0)) - int(testsuite.get('skipped', 0))
        summary["tests"]["failed"] = int(testsuite.get('failures', 0))
        summary["tests"]["errors"] = int(testsuite.get('errors', 0))
        summary["tests"]["skipped"] = int(testsuite.get('skipped', 0))
        summary["duration_seconds"] = float(testsuite.get('time', 0))

        # Get individual test details
        for testcase in testsuite.findall('testcase'):
            test_detail = {
                "name": testcase.get('name'),
                "classname": testcase.get('classname'),
                "duration": float(testcase.get('time', 0)),
                "status": "passed"
            }

            failure = testcase.find('failure')
            if failure is not None:
                test_detail["status"] = "failed"
                test_detail["message"] = failure.get('message', '')

            error = testcase.find('error')
            if error is not None:
                test_detail["status"] = "error"
                test_detail["message"] = error.get('message', '')

            skipped = testcase.find('skipped')
            if skipped is not None:
                test_detail["status"] = "skipped"
                test_detail["message"] = skipped.get('message', '')

            summary["test_details"].append(test_detail)

# Write summary
with open(reports_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Total tests: {summary['tests']['total']}")
print(f"Passed: {summary['tests']['passed']}")
print(f"Failed: {summary['tests']['failed']}")
print(f"Skipped: {summary['tests']['skipped']}")
print(f"Errors: {summary['tests']['errors']}")
EOF

echo ""
echo "=== Reports Generated ==="
echo "JUnit XML: ${REPORTS_DIR}/junit.xml"
echo "Summary JSON: ${REPORTS_DIR}/summary.json"
echo "Coverage HTML: ${REPORTS_DIR}/coverage/index.html"
echo ""

if [ ${PYTEST_EXIT_CODE} -eq 0 ]; then
    echo "✅ All Phase ${PHASE} tests passed!"
    exit 0
else
    echo "❌ Some Phase ${PHASE} tests failed. Check reports for details."
    exit 1
fi

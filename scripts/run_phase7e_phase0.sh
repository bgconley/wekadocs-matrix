#!/bin/bash
#
# Phase 7E.0 - Complete Phase 0 Execution
# Runs all validation and baseline tasks in sequence
#
# Usage:
#   ./scripts/run_phase7e_phase0.sh
#   ./scripts/run_phase7e_phase0.sh --skip-queries  # Skip baseline queries if no vectors

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Report directory
REPORT_DIR="reports/phase-7e"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Create report directory
mkdir -p "$REPORT_DIR"

# Parse arguments
SKIP_QUERIES=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-queries)
            SKIP_QUERIES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Phase 7E.0 - Validation & Baseline Establishment${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "Timestamp: ${TIMESTAMP}"
echo -e "Report directory: ${REPORT_DIR}"
echo ""

# Function to run task with error handling
run_task() {
    local task_num=$1
    local task_name=$2
    local command=$3

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Task ${task_num}: ${task_name}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if eval "$command"; then
        echo ""
        echo -e "${GREEN}✓ Task ${task_num} completed successfully${NC}"
        echo ""
        return 0
    else
        echo ""
        echo -e "${RED}✗ Task ${task_num} failed${NC}"
        echo ""
        return 1
    fi
}

# Track success/failure
TASKS_PASSED=0
TASKS_FAILED=0
FAILED_TASKS=()

# Task 0.1: Document Token Backfill
if run_task "0.1" "Document Token Backfill" \
    "python scripts/backfill_document_tokens.py --execute --report ${REPORT_DIR}/backfill-${TIMESTAMP}.json"; then
    ((TASKS_PASSED++))
else
    ((TASKS_FAILED++))
    FAILED_TASKS+=("0.1: Document Token Backfill")
fi

# Task 0.2: Token Accounting Validation
if run_task "0.2" "Token Accounting Validation" \
    "python scripts/validate_token_accounting.py --threshold 0.01 --report ${REPORT_DIR}/validation-${TIMESTAMP}.json"; then
    ((TASKS_PASSED++))
else
    ((TASKS_FAILED++))
    FAILED_TASKS+=("0.2: Token Accounting Validation")
fi

# Task 0.3: Baseline Distribution Analysis
if run_task "0.3" "Baseline Distribution Analysis" \
    "python scripts/baseline_distribution_analysis.py --report ${REPORT_DIR}/distribution-${TIMESTAMP}.json --markdown ${REPORT_DIR}/distribution-${TIMESTAMP}.md"; then
    ((TASKS_PASSED++))
else
    ((TASKS_FAILED++))
    FAILED_TASKS+=("0.3: Baseline Distribution Analysis")
fi

# Task 0.4: Baseline Query Execution (optional if --skip-queries)
if [ "$SKIP_QUERIES" = true ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Task 0.4: Baseline Query Execution${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}⊘ Skipped (--skip-queries flag)${NC}"
    echo ""
else
    if run_task "0.4" "Baseline Query Execution" \
        "python scripts/run_baseline_queries.py --queries tests/fixtures/baseline_query_set.yaml --report ${REPORT_DIR}/queries-${TIMESTAMP}.json --top-k 20"; then
        ((TASKS_PASSED++))
    else
        ((TASKS_FAILED++))
        FAILED_TASKS+=("0.4: Baseline Query Execution")
    fi
fi

# Summary
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}PHASE 0 SUMMARY${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "Timestamp: ${TIMESTAMP}"
echo -e "Tasks passed: ${GREEN}${TASKS_PASSED}${NC}"
echo -e "Tasks failed: ${RED}${TASKS_FAILED}${NC}"
echo ""

if [ ${TASKS_FAILED} -gt 0 ]; then
    echo -e "${RED}Failed tasks:${NC}"
    for task in "${FAILED_TASKS[@]}"; do
        echo -e "  ${RED}✗${NC} ${task}"
    done
    echo ""
fi

echo -e "Reports saved to: ${REPORT_DIR}/"
echo ""

# List generated reports
echo -e "${BLUE}Generated reports:${NC}"
ls -lh "${REPORT_DIR}"/*-${TIMESTAMP}.* 2>/dev/null || echo "  (none)"
echo ""

if [ ${TASKS_FAILED} -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 0 completed successfully${NC}"
    exit 0
else
    echo -e "${RED}✗ Phase 0 completed with ${TASKS_FAILED} failure(s)${NC}"
    exit 1
fi

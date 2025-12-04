#!/bin/bash
# Run all tests: unit, integration, and e2e
# Starts services if needed for integration/e2e tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd "${REPO_ROOT}"

UNIT_STATUS=0
INTEGRATION_STATUS=0
E2E_STATUS=0

# Run unit tests first (no services needed)
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Unit Tests${NC}"
echo -e "${BLUE}========================================${NC}"
"${SCRIPT_DIR}/test_unit.sh" || UNIT_STATUS=$?

# Start services for integration/e2e tests
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Services for Integration Tests${NC}"
echo -e "${BLUE}========================================${NC}"
"${SCRIPT_DIR}/start_services.sh"

# Run integration tests
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Integration Tests${NC}"
echo -e "${BLUE}========================================${NC}"
"${SCRIPT_DIR}/test_integration.sh" test || INTEGRATION_STATUS=$?

# Run e2e tests
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running E2E Tests${NC}"
echo -e "${BLUE}========================================${NC}"
"${SCRIPT_DIR}/test_e2e.sh" || E2E_STATUS=$?

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ $UNIT_STATUS -eq 0 ]; then
    echo -e "Unit Tests:        ${GREEN}✓ PASSED${NC}"
else
    echo -e "Unit Tests:        ${RED}✗ FAILED${NC}"
fi

if [ $INTEGRATION_STATUS -eq 0 ]; then
    echo -e "Integration Tests: ${GREEN}✓ PASSED${NC}"
else
    echo -e "Integration Tests: ${RED}✗ FAILED${NC}"
fi

if [ $E2E_STATUS -eq 0 ]; then
    echo -e "E2E Tests:         ${GREEN}✓ PASSED${NC}"
else
    echo -e "E2E Tests:         ${RED}✗ FAILED${NC}"
fi

# Exit with failure if any test suite failed
if [ $UNIT_STATUS -ne 0 ] || [ $INTEGRATION_STATUS -ne 0 ] || [ $E2E_STATUS -ne 0 ]; then
    exit 1
fi

echo ""
echo -e "${GREEN}All tests passed!${NC}"

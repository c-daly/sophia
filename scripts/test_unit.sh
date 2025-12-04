#!/bin/bash
# Run unit tests (no services required)

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

echo -e "${BLUE}Running Sophia unit tests...${NC}"
echo -e "${YELLOW}These tests run without external services (Neo4j, Milvus)${NC}"
echo ""

# Run unit tests only - these don't require any services
poetry run pytest tests/unit/ -v --tb=short "$@"

echo ""
echo -e "${GREEN}Unit tests complete.${NC}"
echo -e "To run integration tests: ${BLUE}./scripts/test_integration.sh${NC}"
echo -e "To run all tests:         ${BLUE}./scripts/test_all.sh${NC}"

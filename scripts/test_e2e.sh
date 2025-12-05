#!/bin/bash
# Run e2e tests (requires test stack with Sophia container)
# Start stack with: ./scripts/start_services.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Port configuration (4xxxx prefix for sophia)
SOPHIA_PORT="${SOPHIA_PORT:-48001}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-47474}"
MILVUS_METRICS_PORT="${MILVUS_METRICS_PORT:-47091}"

cd "${REPO_ROOT}"

# Check if e2e tests exist
if [ ! -d "tests/e2e" ] || [ -z "$(find tests/e2e -name 'test_*.py' 2>/dev/null)" ]; then
    echo -e "${YELLOW}No e2e tests found in tests/e2e/${NC}"
    exit 0
fi

# Export environment variables for tests
export SOPHIA_PORT="${SOPHIA_PORT}"
export SOPHIA_URL="http://localhost:${SOPHIA_PORT}"

# Check if test stack is running
echo -e "${BLUE}Checking test stack...${NC}"
if ! curl -s -f "http://localhost:${NEO4J_HTTP_PORT}/" > /dev/null 2>&1; then
    echo -e "${RED}Neo4j not running on port ${NEO4J_HTTP_PORT}. Start with: ./scripts/start_services.sh${NC}"
    exit 1
fi
if ! curl -s -f "http://localhost:${MILVUS_METRICS_PORT}/healthz" > /dev/null 2>&1; then
    echo -e "${RED}Milvus not running on port ${MILVUS_METRICS_PORT}. Start with: ./scripts/start_services.sh${NC}"
    exit 1
fi
if ! curl -s -f "http://localhost:${SOPHIA_PORT}/health" > /dev/null 2>&1; then
    echo -e "${RED}Sophia not running on port ${SOPHIA_PORT}. Start with: ./scripts/start_services.sh${NC}"
    exit 1
fi
echo -e "${GREEN}Test stack is running (Neo4j, Milvus, Sophia)${NC}"

# Run e2e tests
echo -e "${BLUE}Running Sophia e2e tests...${NC}"
poetry run pytest tests/e2e/ -v --tb=short -m e2e "$@"

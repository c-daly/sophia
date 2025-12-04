#!/bin/bash
# Stop test services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STACK_DIR="${REPO_ROOT}/tests/e2e/stack/sophia"
COMPOSE_FILE="${STACK_DIR}/docker-compose.test.yml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Stopping Sophia test services...${NC}"
docker compose -f "${COMPOSE_FILE}" down "$@"
echo -e "${GREEN}Services stopped${NC}"

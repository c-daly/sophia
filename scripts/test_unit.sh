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

# Run unit tests (exclude integration and e2e)
poetry run pytest tests/ \
    -m "not integration" \
    --ignore=tests/integration/ \
    --ignore=tests/e2e/ \
    -v --tb=short "$@"

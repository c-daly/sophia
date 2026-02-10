#!/bin/bash
# Run linting and formatting checks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Running ruff check..."
poetry run ruff check .

echo "Running black check..."
poetry run black --check .

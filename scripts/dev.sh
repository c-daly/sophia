#!/bin/bash
# Start Sophia development server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Starting Sophia dev server on port 47000..."
poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000 --reload

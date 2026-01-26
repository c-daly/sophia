#!/bin/bash
set -e
echo "=== Sophia Local Development Setup ==="

poetry install --with dev

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

poetry run python -c "from logos_config.ports import get_repo_ports; print(f'Sophia ports: {get_repo_ports(\"sophia\")}')"
echo "Setup complete. Run './scripts/run_tests.sh unit' to verify."

#!/bin/bash
set -e
echo "=== Sophia Local Development Setup ==="

poetry install --with dev

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# Install logos-foundry as editable from sibling checkout (for local dev only)
poetry run pip install -e ../logos

poetry run python -c "from logos_config.ports import get_repo_ports; print(f'Sophia ports: {get_repo_ports(\"sophia\")}')"
echo "Local dev setup complete. Verify: poetry run pip show logos-foundry"

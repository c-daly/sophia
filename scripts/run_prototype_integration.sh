#!/usr/bin/env bash
set -euo pipefail

COMPOSE=${COMPOSE_CMD:-"docker compose"}
COMPOSE_FILE=${COMPOSE_FILE:-"docker-compose.yml"}
SERVICES=("neo4j" "milvus-etcd" "milvus-minio" "milvus-standalone")
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-900} # 15 minutes default

cleanup() {
  echo "Stopping integration services..."
  $COMPOSE -f "$COMPOSE_FILE" down -v >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "Starting Neo4j + Milvus services for prototype integration tests..."
if ! $COMPOSE -f "$COMPOSE_FILE" up -d "${SERVICES[@]}"; then
  echo "Failed to start containers. Recent logs:"
  $COMPOSE -f "$COMPOSE_FILE" logs --tail=200 || true
  exit 1
fi

echo "Waiting for services to become healthy (timeout: ${HEALTH_TIMEOUT}s)..."
if ! timeout "${HEALTH_TIMEOUT}s" $COMPOSE -f "$COMPOSE_FILE" wait "${SERVICES[@]}"; then
  echo "Services failed to report healthy before timeout. Recent logs:"
  $COMPOSE -f "$COMPOSE_FILE" logs --tail=200 || true
  exit 1
fi

export NEO4J_URI=${NEO4J_URI:-"bolt://localhost:7687"}
export NEO4J_USER=${NEO4J_USER:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"sophiadev"}
export MILVUS_HOST=${MILVUS_HOST:-"localhost"}
export MILVUS_PORT=${MILVUS_PORT:-"19530"}
export RUN_PROTOTYPE_INTEGRATION=1

echo "Running prototype integration tests..."
poetry run pytest tests/api/test_prototype_integration.py -m integration "$@"

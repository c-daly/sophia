#!/bin/bash
# Start test services (Neo4j + Milvus) for integration/e2e testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STACK_DIR="${REPO_ROOT}/tests/e2e/stack/sophia"
COMPOSE_FILE="${STACK_DIR}/docker-compose.test.yml"

# Sophia uses offset ports to avoid conflicts with other LOGOS repos
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-37474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-37687}"
MILVUS_PORT="${MILVUS_PORT:-39530}"
MILVUS_METRICS_PORT="${MILVUS_METRICS_PORT:-39091}"

export NEO4J_HTTP_PORT NEO4J_BOLT_PORT MILVUS_PORT MILVUS_METRICS_PORT

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

compose() {
    docker compose -f "${COMPOSE_FILE}" "$@"
}

function stop_containers_on_ports() {
    local ports=(
        "${NEO4J_HTTP_PORT}"
        "${NEO4J_BOLT_PORT}"
        "${MILVUS_PORT}"
        "${MILVUS_METRICS_PORT}"
    )
    for port in "${ports[@]}"; do
        local container=$(docker ps --format '{{.ID}}' --filter "publish=${port}" 2>/dev/null)
        if [ -n "$container" ]; then
            echo -e "${YELLOW}Stopping container using port ${port}...${NC}"
            docker stop "$container" 2>/dev/null || true
        fi
    done
}

function wait_for_neo4j() {
    echo -n "Neo4j: "
    local max_attempts=30
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if compose exec -T neo4j cypher-shell -u neo4j -p neo4jtest "RETURN 1" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e "${RED}✗ Not ready (timeout)${NC}"
    return 1
}

function wait_for_milvus() {
    echo -n "Milvus: "
    local max_attempts=60
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:${MILVUS_METRICS_PORT}/healthz" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e "${RED}✗ Not ready (timeout)${NC}"
    return 1
}

echo -e "${BLUE}Stopping any containers using test ports...${NC}"
stop_containers_on_ports
compose down 2>/dev/null || true

echo -e "${BLUE}Starting Sophia test services...${NC}"
compose up -d

echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 5

wait_for_neo4j
wait_for_milvus

echo ""
echo -e "${GREEN}Services are ready!${NC}"
echo ""
echo "Port Configuration:"
echo "  Neo4j:  bolt://localhost:${NEO4J_BOLT_PORT} (http: ${NEO4J_HTTP_PORT})"
echo "  Milvus: localhost:${MILVUS_PORT} (health: ${MILVUS_METRICS_PORT})"
echo ""
echo "Run tests with:"
echo "  ./scripts/test_integration.sh"
echo "  ./scripts/test_e2e.sh"
echo "  ./scripts/test_all.sh"
echo ""
echo "Stop services with:"
echo "  ./scripts/stop_services.sh"

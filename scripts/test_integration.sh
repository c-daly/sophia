#!/bin/bash
set -e

# Sophia Integration Test Runner
# Runs Sophia API from source (like Apollo) with containerized backend services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_BASE="${REPO_ROOT}/containers/docker-compose.test.yml"

# Sophia ports use 47xxx prefix (logos_config.SOPHIA_PORTS)
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-47474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-47687}"
MILVUS_PORT="${MILVUS_PORT:-47530}"
MILVUS_METRICS_PORT="${MILVUS_METRICS_PORT:-47091}"
SOPHIA_PORT="${SOPHIA_PORT:-47000}"

NEO4J_CONTAINER="${NEO4J_CONTAINER:-sophia-test-neo4j}"
MILVUS_CONTAINER="${MILVUS_CONTAINER:-sophia-test-milvus}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jtest}"
SOPHIA_API_TOKEN="${SOPHIA_API_TOKEN:-test-token-for-sophia}"
SOPHIA_PID=""

export NEO4J_HTTP_PORT NEO4J_BOLT_PORT MILVUS_PORT MILVUS_METRICS_PORT SOPHIA_PORT
export NEO4J_CONTAINER MILVUS_CONTAINER NEO4J_USER NEO4J_PASSWORD SOPHIA_API_TOKEN

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function compose() {
    docker-compose -f "$COMPOSE_BASE" "$@"
}

function cleanup_sophia() {
    if [ -n "$SOPHIA_PID" ]; then
        echo -e "${YELLOW}Stopping Sophia API (PID: $SOPHIA_PID)...${NC}"
        kill $SOPHIA_PID 2>/dev/null || true
        wait $SOPHIA_PID 2>/dev/null || true
        SOPHIA_PID=""
    fi
}

function start_sophia_from_source() {
    echo -e "${BLUE}Starting Sophia API from source...${NC}"
    
    # Kill any existing process on the port
    local existing_pid=$(lsof -ti:$SOPHIA_PORT 2>/dev/null)
    if [ -n "$existing_pid" ]; then
        echo -e "${YELLOW}Killing existing process on port $SOPHIA_PORT (PID: $existing_pid)${NC}"
        kill $existing_pid 2>/dev/null || true
        sleep 2
    fi
    
    # Start Sophia API from source
    cd "$REPO_ROOT"
    poetry run uvicorn sophia.api.app:create_app \
        --factory \
        --host 0.0.0.0 \
        --port $SOPHIA_PORT \
        --log-level info \
        > /tmp/sophia-test-api.log 2>&1 &
    
    SOPHIA_PID=$!
    echo "Sophia API started (PID: $SOPHIA_PID)"
    
    # Wait for Sophia to be healthy
    echo -n "Sophia API: "
    local max_attempts=30
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:${SOPHIA_PORT}/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    
    echo -e "${RED}✗ Failed to start${NC}"
    echo "Last 20 lines of log:"
    tail -20 /tmp/sophia-test-api.log
    return 1
}

function start_services() {
    echo -e "${BLUE}Starting backend services (Neo4j, Milvus, Redis)...${NC}"
    compose up -d
    
    echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
    sleep 5
    
    # Check Neo4j
    echo -n "Neo4j: "
    local max_attempts=30
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if compose exec -T neo4j cypher-shell -u neo4j -p neo4jtest "RETURN 1" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            break
        fi
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}✗ Not ready (timeout)${NC}"
            return 1
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    # Check Milvus
    echo -n "Milvus: "
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:${MILVUS_METRICS_PORT}/healthz > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            break
        fi
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}✗ Not ready (timeout)${NC}"
            return 1
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    # Start Sophia from source
    start_sophia_from_source || return 1
}

function seed_data() {
    echo -e "${BLUE}Seeding test data...${NC}"
    echo -e "${YELLOW}Seeding pick-and-place scenario into Neo4j...${NC}"
    cd "$REPO_ROOT"
    poetry run python -m sophia.hcg_client.seeder
    echo -e "${GREEN}✓ Test data seeded${NC}"
}

function run_tests() {
    echo -e "${BLUE}Running Sophia integration tests...${NC}"
    echo -e "${YELLOW}NEO4J_URI=bolt://localhost:${NEO4J_BOLT_PORT}${NC}"
    echo -e "${YELLOW}MILVUS_HOST=localhost:${MILVUS_PORT}${NC}"
    echo -e "${YELLOW}SOPHIA_URL=http://localhost:${SOPHIA_PORT}${NC}"
    
    cd "$REPO_ROOT"
    export NEO4J_HOST="localhost"
    export NEO4J_USER NEO4J_PASSWORD NEO4J_BOLT_PORT NEO4J_HTTP_PORT
    export MILVUS_HOST="localhost"
    export MILVUS_PORT
    export SOPHIA_URL="http://localhost:${SOPHIA_PORT}"
    export SOPHIA_API_TOKEN
    export RUN_SOPHIA_INTEGRATION=1
    
    poetry run pytest tests/ -v \
        --cov=sophia \
        --cov-report=term \
        --cov-report=xml \
        --cov-fail-under=50 \
        -m "not requires_torch" \
        -r sS
}

function stop_services() {
    cleanup_sophia
    echo -e "${BLUE}Stopping backend services...${NC}"
    compose down
    echo -e "${GREEN}Services stopped${NC}"
}

trap cleanup_sophia EXIT INT TERM

# Main
case "${1:-test}" in
    up)
        start_services
        echo ""
        echo "Services running. Sophia API at http://localhost:${SOPHIA_PORT}"
        echo "Run tests with: poetry run pytest tests/"
        echo "Stop with: $0 down"
        ;;
    down)
        stop_services
        ;;
    seed)
        seed_data
        ;;
    test)
        start_services
        seed_data
        run_tests
        stop_services
        ;;
    *)
        echo "Usage: $0 [up|down|seed|test]"
        exit 1
        ;;
esac

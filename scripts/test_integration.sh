#!/bin/bash
# Sophia E2E/Integration test runner script with convenience commands

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/containers/docker-compose.test.yml"
COMPOSE_SOPHIA="${REPO_ROOT}/containers/docker-compose.test.sophia.yml"

# All sophia ports use 47xxx prefix (logos_config.SOPHIA_PORTS)
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-47474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-47687}"
MILVUS_PORT="${MILVUS_PORT:-47530}"
MILVUS_METRICS_PORT="${MILVUS_METRICS_PORT:-47091}"
SOPHIA_PORT="${SOPHIA_PORT:-47000}"

NEO4J_CONTAINER="${NEO4J_CONTAINER:-sophia-test-neo4j}"
MILVUS_CONTAINER="${MILVUS_CONTAINER:-sophia-test-milvus}"
SOPHIA_CONTAINER="${SOPHIA_CONTAINER:-sophia-test-api}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jtest}"
SOPHIA_API_TOKEN="${SOPHIA_API_TOKEN:-test-token-for-sophia}"

export NEO4J_HTTP_PORT NEO4J_BOLT_PORT MILVUS_PORT MILVUS_METRICS_PORT SOPHIA_PORT
export NEO4J_CONTAINER MILVUS_CONTAINER SOPHIA_CONTAINER NEO4J_USER NEO4J_PASSWORD SOPHIA_API_TOKEN

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

compose() {
    docker compose -f "${COMPOSE_FILE}" -f "${COMPOSE_SOPHIA}" "$@"
}

function print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  test       Run integration tests (default)"
    echo "  up         Start services only"
    echo "  down       Stop and remove services"
    echo "  logs       Show service logs"
    echo "  seed       Seed test data into Neo4j"
    echo "  status     Check service status"
    echo "  clean      Clean up everything including volumes"
    echo "  help       Show this help message"
    echo ""
    echo "Port Configuration (Sophia uses offset ports):"
    echo "  Neo4j:  ${NEO4J_BOLT_PORT} (bolt), ${NEO4J_HTTP_PORT} (http)"
    echo "  Milvus: ${MILVUS_PORT} (grpc), ${MILVUS_METRICS_PORT} (health)"
    echo "  Sophia: ${SOPHIA_PORT} (API)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run full test (start, seed, test)"
    echo "  $0 up              # Start services for manual testing"
    echo "  $0 seed            # Seed test data"
    echo "  $0 logs            # View logs"
    echo "  $0 down            # Stop services"
}

function stop_containers_on_ports() {
    local ports=(
        "${NEO4J_HTTP_PORT}"
        "${NEO4J_BOLT_PORT}"
        "${MILVUS_PORT}"
        "${MILVUS_METRICS_PORT}"
        "${SOPHIA_PORT}"
    )
    for port in "${ports[@]}"; do
        local container=$(docker ps --format '{{.ID}}' --filter "publish=${port}" 2>/dev/null)
        if [ -n "$container" ]; then
            echo -e "${YELLOW}Stopping container using port ${port}...${NC}"
            docker stop "$container" 2>/dev/null || true
        fi
    done
}

function start_services() {
    echo -e "${BLUE}Stopping any containers using test ports...${NC}"
    stop_containers_on_ports
    compose down 2>/dev/null || true
    
    echo -e "${BLUE}Starting Sophia test services...${NC}"
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
        echo -n "."
        sleep 2
        ((attempt++))
    done
    if [ $attempt -gt $max_attempts ]; then
        echo -e "${RED}✗ Not ready (timeout)${NC}"
    fi
    
    # Check Milvus
    echo -n "Milvus: "
    max_attempts=60
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:${MILVUS_METRICS_PORT}/healthz" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            break
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    if [ $attempt -gt $max_attempts ]; then
        echo -e "${RED}✗ Not ready (timeout)${NC}"
    fi
    
    # Check Sophia API
    echo -n "Sophia API: "
    max_attempts=60
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:${SOPHIA_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            break
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    if [ $attempt -gt $max_attempts ]; then
        echo -e "${RED}✗ Not ready (timeout)${NC}"
    fi
}

function stop_services() {
    echo -e "${BLUE}Stopping Sophia test services...${NC}"
    compose down
    echo -e "${GREEN}Services stopped${NC}"
}

function show_logs() {
    compose logs "$@"
}

function check_status() {
    echo -e "${BLUE}Service Status:${NC}"
    compose ps
    
    echo ""
    echo -e "${BLUE}Health Checks:${NC}"
    
    echo -n "Neo4j (bolt://localhost:${NEO4J_BOLT_PORT}): "
    if compose exec -T neo4j cypher-shell -u neo4j -p neo4jtest "RETURN 1" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Unhealthy${NC}"
    fi
    
    echo -n "Milvus (localhost:${MILVUS_PORT}): "
    if curl -s -f "http://localhost:${MILVUS_METRICS_PORT}/healthz" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Unhealthy${NC}"
    fi
    
    echo -n "Sophia API (localhost:${SOPHIA_PORT}): "
    if curl -s -f "http://localhost:${SOPHIA_PORT}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Unhealthy${NC}"
    fi
}

function seed_data() {
    echo -e "${BLUE}Seeding test data...${NC}"
    cd "${REPO_ROOT}"
    
    # Export environment variables for seeding
    export NEO4J_URI="bolt://localhost:${NEO4J_BOLT_PORT}"
    export NEO4J_USER NEO4J_PASSWORD
    
    echo -e "${YELLOW}Seeding pick-and-place scenario into Neo4j...${NC}"
    poetry run python -c "
from sophia.hcg_client import HCGClient
from sophia.hcg_client.seeder import seed_pick_and_place_data
import os

client = HCGClient(
    neo4j_uri=os.environ['NEO4J_URI'],
    neo4j_username=os.environ['NEO4J_USER'],
    neo4j_password=os.environ['NEO4J_PASSWORD'],
)
print('Clearing existing data...')
client.clear_all()
print('Seeding pick-and-place data...')
seed_pick_and_place_data(client)
# Verify seeding by counting nodes via direct Neo4j query
with client.driver.session() as session:
    result = session.run('MATCH (n:Node) RETURN count(n) as count')
    count = result.single()['count']
    print(f'Seeded {count} nodes')
client.close()
print('Done!')
"
    
    echo -e "${GREEN}✓ Test data seeded${NC}"
}

function run_tests() {
    echo -e "${BLUE}Running Sophia integration tests...${NC}"
    cd "${REPO_ROOT}"
    
    # Export environment variables for tests
    export NEO4J_URI="bolt://localhost:${NEO4J_BOLT_PORT}"
    export NEO4J_USER NEO4J_PASSWORD
    export MILVUS_HOST="localhost"
    export MILVUS_PORT
    export SOPHIA_URL="http://localhost:${SOPHIA_PORT}"
    export SOPHIA_API_TOKEN
    
    echo -e "${YELLOW}NEO4J_URI=${NEO4J_URI}${NC}"
    echo -e "${YELLOW}MILVUS_HOST=localhost:${MILVUS_PORT}${NC}"
    echo -e "${YELLOW}SOPHIA_URL=${SOPHIA_URL}${NC}"

    set +e
    poetry run pytest tests/integration/ -v --tb=short
    local test_status=$?
    set -e

    return $test_status
}

function clean_all() {
    echo -e "${YELLOW}Cleaning up all Sophia test resources...${NC}"
    compose down -v
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Main command handling
COMMAND="${1:-test}"

case "$COMMAND" in
    test)
        start_services
        seed_data
        run_tests
        ;;
    up)
        start_services
        ;;
    down)
        stop_services
        ;;
    seed)
        seed_data
        ;;
    logs)
        shift
        show_logs "$@"
        ;;
    status)
        check_status
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac

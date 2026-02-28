#!/bin/bash
# Sophia Test Runner - Unified test management script
#
# Usage: ./scripts/run_tests.sh [command] [options]
#
# This script manages the test pyramid:
#   unit        - Fast tests, no services needed (~5s)
#   integration - Requires Neo4j + Milvus (~30s)
#   e2e         - Full stack tests (~60s)
#   all         - Run everything

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

cd "${REPO_ROOT}"

# Port configuration (all 47xxx per logos_config.SOPHIA_PORTS)
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-47474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-47687}"
MILVUS_PORT="${MILVUS_PORT:-47530}"
MILVUS_METRICS_PORT="${MILVUS_METRICS_PORT:-47091}"

function show_help() {
  echo -e "${CYAN}Sophia Test Runner${NC}"
  echo ""
  echo "Usage: $0 <command> [pytest options]"
  echo ""
  echo -e "${YELLOW}Test Commands:${NC}"
  echo "  unit          Run unit tests (no services needed)"
  echo "  integration   Run integration tests (starts services automatically)"
  echo "  e2e           Run end-to-end tests (starts services automatically)"
  echo "  all           Run all tests (starts services automatically)"
  echo ""
  echo -e "${YELLOW}Service Commands:${NC}"
  echo "  up            Start test services (Neo4j, Milvus)"
  echo "  down          Stop test services"
  echo "  status        Check service status"
  echo "  logs          Show service logs"
  echo ""
  echo -e "${YELLOW}Quick Commands:${NC}"
  echo "  quick         Run unit tests only (fastest feedback)"
  echo "  ci            Run what CI runs (all tests with coverage)"
  echo ""
  echo -e "${YELLOW}Examples:${NC}"
  echo "  $0 unit                    # Fast unit tests"
  echo "  $0 integration             # Integration tests with services"
  echo "  $0 unit -k media           # Unit tests matching 'media'"
  echo "  $0 all --cov               # All tests with coverage"
  echo ""
  echo -e "${YELLOW}Test Distribution:${NC}"
  echo "  Unit:        tests/unit/        (~130 tests, ~5s)"
  echo "  Integration: tests/integration/ (~110 tests, ~30s)"
  echo "  E2E:         tests/e2e/         (~40 tests, ~60s)"
}

function check_services() {
  local neo4j_ok=false
  local milvus_ok=false

  if curl -s "http://localhost:${NEO4J_HTTP_PORT}/" >/dev/null 2>&1; then
    neo4j_ok=true
  fi

  if curl -s "http://localhost:${MILVUS_METRICS_PORT}/healthz" >/dev/null 2>&1; then
    milvus_ok=true
  fi

  if $neo4j_ok && $milvus_ok; then
    return 0
  else
    return 1
  fi
}

function start_services() {
  echo -e "${BLUE}Starting test services...${NC}"
  "${SCRIPT_DIR}/test_integration.sh" up
}

function stop_services() {
  echo -e "${BLUE}Stopping test services...${NC}"
  "${SCRIPT_DIR}/test_integration.sh" down
}

function show_status() {
  "${SCRIPT_DIR}/test_integration.sh" status
}

function show_logs() {
  "${SCRIPT_DIR}/test_integration.sh" logs
}

function run_unit() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Running Unit Tests (no services required)${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  poetry run pytest tests/unit/ -v --tb=short "$@"
}

function run_integration() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Running Integration Tests${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  if ! check_services; then
    echo -e "${YELLOW}Services not running. Starting them...${NC}"
    start_services
  fi

  poetry run pytest tests/integration/ -v --tb=short "$@"
}

function run_e2e() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Running E2E Tests${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  if ! check_services; then
    echo -e "${YELLOW}Services not running. Starting them...${NC}"
    start_services
  fi

  poetry run pytest tests/e2e/ -v --tb=short "$@"
}

function run_all() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Running All Tests${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  if ! check_services; then
    echo -e "${YELLOW}Services not running. Starting them...${NC}"
    start_services
  fi

  poetry run pytest tests/ -v --tb=short "$@"
}

function run_ci() {
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Running CI Tests (with coverage)${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  if ! check_services; then
    echo -e "${YELLOW}Services not running. Starting them...${NC}"
    start_services
  fi

  poetry run pytest tests/ \
    --cov=src/sophia \
    --cov-report=term-missing \
    --cov-fail-under=50 \
    -v --tb=short -r sS "$@"
}

# Main command dispatch
case "${1:-help}" in
unit | u)
  shift
  run_unit "$@"
  ;;
integration | int | i)
  shift
  run_integration "$@"
  ;;
e2e | end-to-end)
  shift
  run_e2e "$@"
  ;;
all | a)
  shift
  run_all "$@"
  ;;
quick | q)
  shift
  run_unit "$@"
  ;;
ci)
  shift
  run_ci "$@"
  ;;
up | start)
  start_services
  ;;
down | stop)
  stop_services
  ;;
status | st)
  show_status
  ;;
logs | log)
  show_logs
  ;;
help | --help | -h)
  show_help
  ;;
*)
  echo -e "${RED}Unknown command: $1${NC}"
  echo ""
  show_help
  exit 1
  ;;
esac

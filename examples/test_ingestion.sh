#!/usr/bin/env bash
# Test script for Hermes proposal ingestion endpoint

set -e

BASE_URL="${SOPHIA_BASE_URL:-http://localhost:8000}"

echo "Testing Hermes Proposal Ingestion Endpoint"
echo "==========================================="
echo ""
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health check
echo "Test 1: Health check (should work without auth)"
curl -s -X GET "$BASE_URL/health" | jq .
echo ""

# Test 2: Ingest minimal proposal
echo "Test 2: Ingest minimal proposal"
RESPONSE=$(curl -s -X POST "$BASE_URL/ingest/hermes_proposal" \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_id": "test_minimal_001",
    "llm_provider": "openai",
    "model": "gpt-4",
    "generated_at": "2025-11-23T12:00:00Z",
    "confidence": 0.75
  }')

echo "$RESPONSE" | jq .

if echo "$RESPONSE" | jq -e '.status == "accepted"' > /dev/null; then
    echo "✓ Minimal proposal accepted"
else
    echo "✗ Minimal proposal failed"
    exit 1
fi
echo ""

# Test 3: Ingest full proposal with all optional fields
echo "Test 3: Ingest full proposal with plan steps, states, and tool calls"
RESPONSE=$(curl -s -X POST "$BASE_URL/ingest/hermes_proposal" \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_id": "test_full_002",
    "source_service": "hermes",
    "llm_provider": "openai",
    "model": "gpt-4",
    "generated_at": "2025-11-23T12:00:00Z",
    "confidence": 0.85,
    "raw_text": "Move the red block to the bin",
    "plan_steps": [
      {
        "action": "move_to_red_block",
        "target": "red_block",
        "parameters": {}
      },
      {
        "action": "grasp_red_block",
        "target": "red_block",
        "parameters": {"force": 0.5}
      },
      {
        "action": "move_to_bin",
        "target": "bin",
        "parameters": {}
      }
    ],
    "imagined_states": [
      {
        "state_id": "state_1",
        "entities": {"red_block": {"location": "table"}}
      },
      {
        "state_id": "state_2",
        "entities": {"red_block": {"location": "bin"}}
      }
    ],
    "diagnostics": {
      "reasoning": "Block needs to be moved from table to bin"
    },
    "tool_calls": [
      {
        "tool": "get_object_location",
        "parameters": {"object_id": "red_block"}
      }
    ],
    "metadata": {
      "session_id": "test_session_123",
      "user_id": "test_user"
    }
  }')

echo "$RESPONSE" | jq .

if echo "$RESPONSE" | jq -e '.status == "accepted"' > /dev/null; then
    NODE_COUNT=$(echo "$RESPONSE" | jq '.stored_node_ids | length')
    echo "✓ Full proposal accepted with $NODE_COUNT nodes created"
    
    # Should have 1 proposal + 3 plan steps + 2 states + 1 tool call = 7 nodes
    if [ "$NODE_COUNT" -eq 7 ]; then
        echo "✓ Correct number of nodes created (7)"
    else
        echo "✗ Expected 7 nodes, got $NODE_COUNT"
        exit 1
    fi
else
    echo "✗ Full proposal failed"
    exit 1
fi
echo ""

# Test 4: Test validation error (missing required field)
echo "Test 4: Test validation error (missing required fields)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/ingest/hermes_proposal" \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_id": "test_invalid_003"
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

echo "$BODY" | jq .

if [ "$HTTP_CODE" = "422" ]; then
    echo "✓ Validation error returned 422 as expected"
else
    echo "✗ Expected HTTP 422, got $HTTP_CODE"
    exit 1
fi
echo ""

# Test 5: Test confidence out of range
echo "Test 5: Test confidence validation (out of range)"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/ingest/hermes_proposal" \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_id": "test_invalid_004",
    "llm_provider": "openai",
    "model": "gpt-4",
    "generated_at": "2025-11-23T12:00:00Z",
    "confidence": 1.5
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

echo "$BODY" | jq .

if [ "$HTTP_CODE" = "422" ]; then
    echo "✓ Invalid confidence returned 422 as expected"
else
    echo "✗ Expected HTTP 422, got $HTTP_CODE"
    exit 1
fi
echo ""

echo "==========================================="
echo "All ingestion tests passed! ✓"
echo "==========================================="

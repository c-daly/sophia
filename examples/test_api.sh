#!/bin/bash
# Integration test script for Sophia API
# This script demonstrates how to use the API endpoints

set -e

API_URL="${API_URL:-http://localhost:8000}"
API_TOKEN="${SOPHIA_API_TOKEN:-test-token}"

echo "========================================="
echo "Sophia API Integration Test"
echo "========================================="
echo ""

# Health check
echo "1. Testing /health endpoint (no auth required)..."
curl -s "$API_URL/health" | jq .
echo ""

# Test authentication
echo "2. Testing authentication on /plan..."
echo "   a) Without token (should fail):"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/plan" \
  -H "Content-Type: application/json" \
  -d '{"goal": {"description": "test", "target_state": "test"}}')
echo "   HTTP Status: $HTTP_CODE (expected: 403)"
echo ""

echo "   b) With valid token (should succeed):"
curl -s -X POST "$API_URL/plan" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"goal": {"description": "test goal", "target_state": "test_state"}}' | jq .
echo ""

# Test /imagine endpoint
echo "3. Testing /imagine endpoint..."
curl -s -X POST "$API_URL/imagine" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "cwm_g_imagery": [{"type": "visual", "content": "red block"}],
    "cwm_e_emotion_tags": ["curious", "focused"],
    "horizon": 3,
    "model_version": "v1.0",
    "assumptions": ["block is graspable"]
  }' | jq . || echo "Note: /imagine requires HCG (Neo4j+Milvus) to be running"
echo ""

# Test /execute endpoint
echo "4. Testing /execute endpoint (dry run)..."
curl -s -X POST "$API_URL/execute" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "test-plan-123",
    "dry_run": true
  }' | jq .
echo ""

echo "========================================="
echo "Integration test complete!"
echo "========================================="

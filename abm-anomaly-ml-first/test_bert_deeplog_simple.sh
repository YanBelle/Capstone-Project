#!/bin/bash

# Simple BERT-DeepLog System Test Script
# Tests the API endpoints using curl

set -e

echo "🧪 BERT-DeepLog System Test Suite"
echo "=================================="

API_URL="http://localhost:8000"
DASHBOARD_URL="http://localhost"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass_test() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail_test() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

info() {
    echo -e "${BLUE}ℹ️ ${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠️ ${NC} $1"
}

# Test 1: API Health Check
echo ""
info "Testing API health..."
if curl -s -f "$API_URL/api/v1/health" > /dev/null; then
    pass_test "API health check"
else
    fail_test "API health check"
fi

# Test 2: BERT-DeepLog Model Info
echo ""
info "Testing BERT-DeepLog model info endpoint..."
MODEL_INFO=$(curl -s "$API_URL/api/v1/bert-deeplog/model-info")
if echo "$MODEL_INFO" | grep -q "model_available"; then
    pass_test "BERT-DeepLog model info endpoint"
    echo "   Model available: $(echo "$MODEL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_available'])")"
else
    fail_test "BERT-DeepLog model info endpoint"
fi

# Test 3: BERT-DeepLog Prediction (should fail without training)
echo ""
info "Testing BERT-DeepLog prediction endpoint (expecting 'model not trained' error)..."
PREDICTION_RESULT=$(curl -s -X POST "$API_URL/api/v1/bert-deeplog/predict" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "session_text": "CARD INSERTED PIN ENTERED CASH DISPENSED CARD TAKEN"}')

if echo "$PREDICTION_RESULT" | grep -q "Model not trained"; then
    pass_test "BERT-DeepLog prediction endpoint (correctly reports untrained model)"
else
    fail_test "BERT-DeepLog prediction endpoint response"
    echo "   Response: $PREDICTION_RESULT"
fi

# Test 4: Training History
echo ""
info "Testing BERT-DeepLog training history endpoint..."
TRAINING_HISTORY=$(curl -s "$API_URL/api/v1/bert-deeplog/training-history")
if echo "$TRAINING_HISTORY" | grep -q "training_history"; then
    pass_test "BERT-DeepLog training history endpoint"
else
    fail_test "BERT-DeepLog training history endpoint"
fi

# Test 5: Prediction Cache
echo ""
info "Testing BERT-DeepLog prediction cache endpoint..."
CACHE_INFO=$(curl -s "$API_URL/api/v1/bert-deeplog/prediction-cache")
if echo "$CACHE_INFO" | grep -q "total_cached_predictions"; then
    pass_test "BERT-DeepLog prediction cache endpoint"
else
    fail_test "BERT-DeepLog prediction cache endpoint"
fi

# Test 6: Dashboard Accessibility
echo ""
info "Testing dashboard accessibility..."
if curl -s -f "$DASHBOARD_URL/" > /dev/null; then
    pass_test "Dashboard is accessible"
else
    fail_test "Dashboard accessibility"
fi

# Test 7: DeepLog Dashboard Route
echo ""
info "Testing DeepLog dashboard route..."
if curl -s -f "$DASHBOARD_URL/dashboard/deeplog" > /dev/null; then
    pass_test "DeepLog dashboard route is accessible"
else
    fail_test "DeepLog dashboard route accessibility"
fi

# Test 8: Sample Training Data (basic endpoint test)
echo ""
info "Testing BERT-DeepLog training endpoint with invalid data (expecting error)..."
TRAINING_RESULT=$(curl -s -X POST "$API_URL/api/v1/bert-deeplog/train" \
  -H "Content-Type: application/json" \
  -d '{"sessions": [], "validation_split": 0.2}')

if echo "$TRAINING_RESULT" | grep -q "detail"; then
    pass_test "BERT-DeepLog training endpoint (correctly handles invalid data)"
else
    fail_test "BERT-DeepLog training endpoint response"
fi

# Summary
echo ""
echo "=================================="
echo "🎯 Test Summary"
echo "=================================="
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed! BERT-DeepLog system is operational.${NC}"
    echo ""
    echo "🌐 Access Points:"
    echo "  • API Documentation: $API_URL/docs"
    echo "  • DeepLog Dashboard: $DASHBOARD_URL/dashboard/deeplog"
    echo "  • Main Dashboard: $DASHBOARD_URL/dashboard"
    echo ""
    echo "🚀 Next Steps:"
    echo "  1. Access the DeepLog dashboard to train a model"
    echo "  2. Upload training data or use sample data"
    echo "  3. Configure training parameters and start training"
    echo "  4. Test predictions on the trained model"
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  $TESTS_FAILED test(s) failed. Check the output above for details.${NC}"
    exit 1
fi

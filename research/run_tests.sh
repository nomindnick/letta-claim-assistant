#!/bin/bash
# Run POC test scripts for Letta integration

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "Letta Integration POC Test Runner"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please activate your virtual environment first"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local requires_server=$3
    
    echo -e "${YELLOW}Running: $test_name${NC}"
    echo "----------------------------------------"
    
    if [ "$requires_server" = "true" ]; then
        # Check if Letta server is running
        if ! curl -s http://localhost:8283/health > /dev/null 2>&1; then
            echo -e "${YELLOW}Starting Letta server...${NC}"
            letta server --port 8283 > /tmp/letta_server.log 2>&1 &
            SERVER_PID=$!
            
            # Wait for server to start
            for i in {1..30}; do
                if curl -s http://localhost:8283/health > /dev/null 2>&1; then
                    echo -e "${GREEN}Server started (PID: $SERVER_PID)${NC}"
                    break
                fi
                sleep 1
            done
        else
            echo -e "${GREEN}Server already running${NC}"
        fi
    fi
    
    # Run the test
    python $test_file
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
    fi
    
    echo ""
    return $TEST_RESULT
}

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0

# Test 1: Server connectivity (starts its own server)
echo "=================================================="
echo "Test 1: Server Connectivity"
echo "=================================================="
run_test "Server Connectivity" "research/test_letta_server.py" false
if [ $? -eq 0 ]; then ((PASSED_TESTS++)); fi
((TOTAL_TESTS++))

# Test 2: Ollama integration (requires server)
echo "=================================================="
echo "Test 2: Ollama Integration"
echo "=================================================="
run_test "Ollama Integration" "research/test_letta_ollama.py" true
if [ $? -eq 0 ]; then ((PASSED_TESTS++)); fi
((TOTAL_TESTS++))

# Test 3: Memory operations (requires server)
echo "=================================================="
echo "Test 3: Memory Operations"
echo "=================================================="
run_test "Memory Operations" "research/test_letta_memory.py" true
if [ $? -eq 0 ]; then ((PASSED_TESTS++)); fi
((TOTAL_TESTS++))

# Test 4: Migration patterns (requires server)
echo "=================================================="
echo "Test 4: Migration Patterns"
echo "=================================================="
run_test "Migration Patterns" "research/test_letta_migration.py" true
if [ $? -eq 0 ]; then ((PASSED_TESTS++)); fi
((TOTAL_TESTS++))

# Test 5: Gemini integration (optional, requires API key)
if [ ! -z "$GEMINI_API_KEY" ]; then
    echo "=================================================="
    echo "Test 5: Gemini Integration"
    echo "=================================================="
    run_test "Gemini Integration" "research/test_letta_gemini.py" true
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); fi
    ((TOTAL_TESTS++))
else
    echo -e "${YELLOW}Skipping Gemini test (no API key)${NC}"
fi

# Summary
echo "=================================================="
echo "TEST SUMMARY"
echo "=================================================="
echo -e "Tests Passed: ${GREEN}$PASSED_TESTS${NC} / $TOTAL_TESTS"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
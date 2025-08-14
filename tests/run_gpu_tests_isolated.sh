#!/bin/bash

# Run each GPU test in a separate process to avoid the one-GPU-manager-per-process limitation
# This script uses GoogleTest's built-in test discovery to find and run tests individually

set -e

BUILD_DIR="${BUILD_DIR:-build}"
TEST_BINARY="$BUILD_DIR/mad_escape_tests"

if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: Test binary not found at $TEST_BINARY"
    echo "Please build the tests first with: make -C build mad_escape_tests"
    exit 1
fi

echo "=================================================="
echo "Running GPU Tests in Isolated Processes"
echo "=================================================="
echo ""
echo "NOTE: Each test runs in a separate process to avoid"
echo "the Madrona limitation of one GPU manager per process."
echo "Each test will take ~45 seconds for NVRTC compilation."
echo ""

# Get list of all GPU tests
GPU_TESTS=$($TEST_BINARY --gtest_list_tests 2>/dev/null | awk '
    /GPU.*Test\.$/ { suite=$1; next }
    suite && /^  / { 
        # Remove leading spaces and comments
        test=$1
        gsub(/^[ \t]+/, "", test)
        gsub(/#.*$/, "", test)
        # Print suite.test format for gtest_filter
        print suite test
    }
')

# Count tests
TEST_COUNT=$(echo "$GPU_TESTS" | wc -l)
echo "Found $TEST_COUNT GPU tests to run"
echo "Estimated total time: $(($TEST_COUNT * 45)) seconds"
echo ""

# Track results
PASSED=0
FAILED=0
FAILED_TESTS=""

# Run each test in a separate process
CURRENT=0
for TEST in $GPU_TESTS; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TEST_COUNT] Running: $TEST"
    echo "----------------------------------------"
    
    if ALLOW_GPU_TESTS_IN_SUITE=1 $TEST_BINARY --gtest_filter="$TEST" --gtest_color=yes 2>&1 | tail -5; then
        echo "✓ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "✗ FAILED"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS\n  - $TEST"
    fi
    echo ""
done

# Summary
echo "=================================================="
echo "GPU Test Summary"
echo "=================================================="
echo "Total: $TEST_COUNT"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    echo -e "$FAILED_TESTS"
    exit 1
else
    echo ""
    echo "All GPU tests passed!"
fi
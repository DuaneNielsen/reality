#!/bin/bash

# Script to run headless multiple times with an optional level file
# Usage: ./test_headless_loop.sh [num_runs] [level_file]

# Get number of runs from parameter, default to 10
NUM_RUNS=${1:-10}
LEVEL_FILE=${2:-}

# Build command based on whether level file is provided
if [ -n "$LEVEL_FILE" ]; then
    LOAD_FLAG="--load $LEVEL_FILE"
    echo "Running headless $NUM_RUNS times with level: $LEVEL_FILE"
else
    LOAD_FLAG=""
    echo "Running headless $NUM_RUNS times with default level"
fi
echo "================================================"

# Counters
SUCCESS_COUNT=0
FAIL_COUNT=0
SEGFAULT_COUNT=0
ASSERTION_COUNT=0

for i in $(seq 1 $NUM_RUNS); do
    echo -n "Run $i/$NUM_RUNS: "
    
    # Run headless and capture output and exit code
    OUTPUT=$(./build/headless --num-worlds 4 --num-steps 5000 --seed 42 $LOAD_FLAG --rand-actions 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        # Extract FPS from output
        FPS=$(echo "$OUTPUT" | grep "FPS:" | cut -d' ' -f2)
        echo "SUCCESS (FPS: $FPS)"
        ((SUCCESS_COUNT++))
    elif [ $EXIT_CODE -eq 139 ]; then
        echo "SEGFAULT!"
        ((SEGFAULT_COUNT++))
        ((FAIL_COUNT++))
    elif [ $EXIT_CODE -eq 134 ]; then
        echo "ASSERTION FAILURE!"
        ((ASSERTION_COUNT++))
        ((FAIL_COUNT++))
        # Show the assertion message if available
        echo "$OUTPUT" | grep -i "assert" | head -1
    else
        echo "FAILED (Exit code: $EXIT_CODE)"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "================================================"
echo "Summary:"
echo "  Successful runs: $SUCCESS_COUNT/$NUM_RUNS"
echo "  Failed runs:     $FAIL_COUNT/$NUM_RUNS"
if [ $SEGFAULT_COUNT -gt 0 ]; then
    echo "    - Segfaults:    $SEGFAULT_COUNT"
fi
if [ $ASSERTION_COUNT -gt 0 ]; then
    echo "    - Assertions:   $ASSERTION_COUNT"
fi

# Exit with failure if any runs failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
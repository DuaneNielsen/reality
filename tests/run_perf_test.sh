#!/bin/bash
# Run performance tests and save results

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="build/perf_results/$TIMESTAMP"

echo "Performance Test Suite - $TIMESTAMP"
echo "========================================"
echo

# Ensure build directory exists
mkdir -p build/perf_results

# CPU test
echo "1. Running CPU benchmark (1024 worlds, 1000 steps)..."
echo "   Output: $OUTPUT_DIR/cpu/"
echo

if uv run python scripts/sim_bench.py \
    --num-worlds 1024 \
    --num-steps 1000 \
    --check-baseline \
    --save-profile \
    --output-dir "$OUTPUT_DIR/cpu"; then
    echo "   CPU test: PASSED"
    cpu_status=0
else
    cpu_exit=$?
    if [ $cpu_exit -eq 2 ]; then
        echo "   CPU test: WARNING - Performance below warning threshold"
        cpu_status=2
    else
        echo "   CPU test: FAILED - Performance below minimum threshold"
        cpu_status=1
    fi
fi

echo
echo "========================================"
echo

# GPU test (only if GPU is available)
if command -v nvidia-smi &> /dev/null; then
    echo "2. Running GPU benchmark (8192 worlds, 1000 steps)..."
    echo "   Output: $OUTPUT_DIR/gpu/"
    echo
    
    if uv run python scripts/sim_bench.py \
        --num-worlds 8192 \
        --num-steps 1000 \
        --gpu-id 0 \
        --check-baseline \
        --save-profile \
        --output-dir "$OUTPUT_DIR/gpu"; then
        echo "   GPU test: PASSED"
        gpu_status=0
    else
        gpu_exit=$?
        if [ $gpu_exit -eq 2 ]; then
            echo "   GPU test: WARNING - Performance below warning threshold"
            gpu_status=2
        else
            echo "   GPU test: FAILED - Performance below minimum threshold"
            gpu_status=1
        fi
    fi
else
    echo "2. Skipping GPU benchmark (no GPU detected)"
    gpu_status=0
fi

echo
echo "========================================"
echo "Test Summary:"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo

# Determine overall exit status
if [ ${cpu_status:-0} -eq 0 ] && [ ${gpu_status:-0} -eq 0 ]; then
    echo "Overall: ALL TESTS PASSED"
    exit 0
elif [ ${cpu_status:-0} -eq 1 ] || [ ${gpu_status:-0} -eq 1 ]; then
    echo "Overall: TESTS FAILED"
    exit 1
else
    echo "Overall: TESTS PASSED WITH WARNINGS"
    exit 2
fi
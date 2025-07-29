#!/bin/bash
# Quick test - assumes project is already built

echo "=== Quick CPU Test ==="

# Test headless and capture output
echo "Headless: 10 worlds, 100 steps"
output=$(./build/headless --cpu --num-worlds 10 --num-steps 100 --rand-actions 2>&1)

# Display the output
echo "$output"

# Extract FPS value using grep and awk
fps=$(echo "$output" | grep "FPS:" | awk '{print $2}' | cut -d'.' -f1)

if [ -n "$fps" ]; then
    echo ""
    echo "Checking FPS performance..."
    echo "Extracted FPS: $fps"
    
    # Expected range: 270000 to 310000
    min_fps=270000
    max_fps=310000
    
    if [ "$fps" -ge "$min_fps" ] && [ "$fps" -le "$max_fps" ]; then
        echo "✓ FPS is within expected range (270,000 - 310,000)"
    else
        echo "✗ WARNING: FPS $fps is outside expected range (270,000 - 310,000)"
        echo "  This might indicate a performance issue"
    fi
else
    echo "✗ ERROR: Could not extract FPS from output"
fi

echo ""

# Test viewer (auto-closes after 3 seconds)
echo "Viewer: 4 worlds (3 second test)"
timeout 3s ./build/viewer --num-worlds 4 --cpu || true

echo ""
echo "=== Tests complete ==="

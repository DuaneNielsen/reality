#!/bin/bash
# Quick test - assumes project is already built

echo "=== Quick CPU Test ==="

# Test headless with our new level format
echo "Headless: 10 worlds, 100 steps with compiled level"
output=$(./build/headless --load levels/quick_test.lvl -n 10 -s 100 --rand-actions 2>&1)

# Display the output
echo "$output"

# Extract FPS value using grep and awk
fps=$(echo "$output" | grep "FPS:" | awk '{print $2}' | cut -d'.' -f1)

if [ -n "$fps" ]; then
    echo ""
    echo "Checking FPS performance..."
    echo "Extracted FPS: $fps"
    
    # Expected range: 200000 to 310000
    min_fps=200000
    max_fps=310000
    
    if [ "$fps" -ge "$min_fps" ] && [ "$fps" -le "$max_fps" ]; then
        echo "✓ FPS is within expected range (200,000 - 310,000)"
    else
        echo "✗ WARNING: FPS $fps is outside expected range (200,000 - 310,000)"
        echo "  This might indicate a performance issue"
    fi
else
    echo "✗ ERROR: Could not extract FPS from output"
fi

echo ""

# Test viewer with compiled level (auto-closes after 3 seconds)
echo "Viewer: 4 worlds (3 second test) with compiled level"
timeout 3s ./build/viewer --load levels/simple_room.lvl -n 4 --cpu || true

echo ""
echo "=== Tests complete ==="
#!/bin/bash
# Simple smoke test for Madrona Escape Room
# Tests basic CPU functionality for headless and viewer modes

set -e  # Exit on error

echo "=== Madrona Escape Room Smoke Test ==="
echo

# Clean and build
echo "1. Cleaning old build..."
rm -rf build
mkdir build

echo "2. Configuring with CMake..."
cd build
/opt/cmake/bin/cmake .. > /dev/null 2>&1
cd ..

echo "3. Building project..."
make -C build -j8 > /dev/null 2>&1
echo "   Build successful!"
echo

# Test headless mode
echo "4. Testing headless mode (CPU)..."
echo "   Running 10 worlds for 100 steps..."
output=$(./build/headless --cpu --num-worlds 10 --num-steps 100 --rand-actions 2>&1)
echo "$output"

# Extract and verify FPS
fps=$(echo "$output" | grep "FPS:" | awk '{print $2}' | cut -d'.' -f1)
if [ -n "$fps" ]; then
    echo ""
    echo "   Checking FPS performance..."
    echo "   Extracted FPS: $fps"
    
    # Expected range: 270000 to 310000
    min_fps=270000
    max_fps=310000
    
    if [ "$fps" -ge "$min_fps" ] && [ "$fps" -le "$max_fps" ]; then
        echo "   ✓ FPS is within expected range (270,000 - 310,000)"
    else
        echo "   ✗ WARNING: FPS $fps is outside expected range (270,000 - 310,000)"
        echo "     This might indicate a performance issue"
        exit 1
    fi
else
    echo "   ✗ ERROR: Could not extract FPS from output"
    exit 1
fi
echo

# Test viewer
echo "5. Testing viewer (CPU)..."
echo "   Starting viewer with 4 worlds..."
echo "   Press 'Esc' or close window to continue test"
timeout 5s ./build/viewer --num-worlds 4 --cpu || true
echo

# Test Python import
echo "6. Testing Python bindings..."
uv run python -c "import madrona_escape_room; print('   Python import successful!')"
echo

echo "=== All smoke tests passed! ==="

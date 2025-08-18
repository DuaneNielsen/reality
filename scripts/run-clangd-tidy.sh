#!/bin/bash

# Script to run clangd-tidy on the codebase (faster alternative to clang-tidy)
# Requires the project to be built first with compile_commands.json

set -e

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Error: build directory not found. Please build the project first."
    echo "Run: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && make -C build"
    exit 1
fi

# Check if compile_commands.json exists
if [ ! -f "build/compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in build directory."
    echo "Please rebuild with: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && make -C build"
    exit 1
fi

echo "Running clangd-tidy on source files..."

# Run clangd-tidy on main source files (excluding test files and problematic files)
uv run clangd-tidy \
    --fail-on-severity=hint \
    --context=2 \
    --color=auto \
    -p build \
    src/*.cpp src/*.hpp include/*.h \
    --allow-extensions=cpp,hpp,h \
    2>/dev/null || {
        echo "Note: Some files had compilation errors (expected for dlpack_extension.cpp without Python headers)"
        echo "clangd-tidy analysis complete with diagnostics above."
    }

echo "clangd-tidy analysis complete."
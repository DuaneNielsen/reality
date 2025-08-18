#!/bin/bash

# Script to run clang-tidy on the codebase
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

# Create symlink to compile_commands.json in root directory
if [ ! -L "compile_commands.json" ]; then
    ln -s build/compile_commands.json compile_commands.json
fi

echo "Running clang-tidy on source files..."

# Find all C++ source files in src/ and include/ directories
find src include -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | \
    grep -v test | \
    xargs clang-tidy --format-style=none

echo "clang-tidy analysis complete."
#!/bin/bash
# Fix submodule build artifact noise in git status --porcelain
# Run this script after cloning or resetting submodules

set -e

echo "Configuring nested submodules to ignore build artifacts..."

cd external/madrona

for sub in external/fast_float external/googletest external/madrona-deps external/madrona-toolchain external/meshoptimizer external/nanobind external/simdjson; do 
    echo "Setting $sub to ignore all changes"
    git config submodule.$sub.ignore all
done

echo "Done! git -C external/madrona status --porcelain should now be clean"
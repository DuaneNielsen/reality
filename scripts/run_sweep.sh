#!/bin/bash

# Exit on any error
set -e

# Parse arguments and use environment variable or default
PROJECT="${WANDB_PROJECT:-madrona-escape-room-dev}"
CACHED=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --cached)
            CACHED=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--project PROJECT_NAME] [--cached]"
            echo "  --project: Specify wandb project name"
            echo "  --cached: Keep existing CUDA kernel cache (skip deletion)"
            echo "Or set WANDB_PROJECT environment variable"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_CONFIG="$SCRIPT_DIR/sweep_config.yaml"
TEMP_SWEEP_CONFIG="$SCRIPT_DIR/temp_sweep_config.yaml"

# Create a temporary config with the specified project
sed "s/^project:.*/project: $PROJECT/" "$SWEEP_CONFIG" > "$TEMP_SWEEP_CONFIG"

echo "Setting up CUDA kernel cache..."
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p build

# Delete existing cache to ensure fresh compilation with correct optimization
# (unless --cached flag is used)
if [ "$CACHED" = false ]; then
    if [ -f "$PROJECT_ROOT/build/madrona_kernels.cache" ]; then
        echo "Removing existing CUDA kernel cache to ensure correct optimization level"
        rm -f "$PROJECT_ROOT/build/madrona_kernels.cache"
    fi
else
    if [ -f "$PROJECT_ROOT/build/madrona_kernels.cache" ]; then
        echo "Using existing CUDA kernel cache (--cached flag specified)"
    else
        echo "No existing CUDA kernel cache found, will be created"
    fi
fi

export MADRONA_MWGPU_KERNEL_CACHE="$PROJECT_ROOT/build/madrona_kernels.cache"
echo "Kernel cache enabled: $MADRONA_MWGPU_KERNEL_CACHE"

echo "Compiling levels before starting sweep..."
if ! uv run python levels/compile_levels.py; then
    echo "Error: Level compilation failed"
    rm "$TEMP_SWEEP_CONFIG"
    exit 1
fi

echo "Creating sweep for project '$PROJECT' from: $SWEEP_CONFIG"

# Create the sweep and extract the agent command
SWEEP_OUTPUT=$(uv run wandb sweep "$TEMP_SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Clean up temporary config
rm "$TEMP_SWEEP_CONFIG"

# Extract the sweep ID
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | tail -1 | awk '{print $NF}')

if [[ -z "$SWEEP_ID" ]]; then
    echo "Error: Could not find sweep ID in output"
    exit 1
fi

echo "Starting sweep agent for project '$PROJECT'..."
echo "Press Ctrl+C to stop"

# Execute the agent command with proper signal handling
exec uv run wandb agent "$SWEEP_ID"
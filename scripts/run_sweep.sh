#!/bin/bash

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_CONFIG="$SCRIPT_DIR/sweep_config.yaml"

echo "Creating sweep from: $SWEEP_CONFIG"

# Create the sweep and extract the agent command
SWEEP_OUTPUT=$(uv run wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract the sweep ID
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | tail -1 | awk '{print $NF}')

if [[ -z "$SWEEP_ID" ]]; then
    echo "Error: Could not find sweep ID in output"
    exit 1
fi

echo "Starting sweep agent..."
echo "Press Ctrl+C to stop"

# Execute the agent command with proper signal handling
exec uv run wandb agent "$SWEEP_ID"
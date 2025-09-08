#!/bin/bash

# Exit on any error
set -e

# Parse arguments and use environment variable or default
PROJECT="${WANDB_PROJECT:-madrona-escape-room-dev}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--project PROJECT_NAME]"
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
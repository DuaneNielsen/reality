#!/bin/bash

# Exit on any error
set -e

# Check if wandb run identifier provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <wandb_run_identifier> [viewer_options...]"
    echo "Run identifier can be:"
    echo "  - Human-friendly name: frosty-sweep-1, cosmic-sweep-63"
    echo "  - Partial name: cosmic-sweep (will lookup exact run hash)"
    echo "  - Run hash: i0jv5zou"
    echo "Examples:"
    echo "  $0 cosmic-sweep-63"
    echo "  $0 i0jv5zou --fps 30"
    exit 1
fi

RUN_IDENTIFIER="$1"
shift  # Remove first argument, rest are passed to viewer

PROJECT_NAME="${WANDB_PROJECT:-madrona-escape-room-dev}"

echo "Looking for wandb run: $RUN_IDENTIFIER in project $PROJECT_NAME"

# First, lookup the run hash
LOOKUP_RESULT=$(uv run python scripts/wandb_utils.py lookup "$RUN_IDENTIFIER" --project "$PROJECT_NAME")

if [[ "$LOOKUP_RESULT" == "NOT_FOUND" ]]; then
    echo "Error: No wandb run found matching '$RUN_IDENTIFIER' in project $PROJECT_NAME"
    echo ""
    echo "Try one of these options:"
    echo "1. Use a run hash directly (8 character string like 'i0jv5zou')"
    echo "2. Set WANDB_PROJECT environment variable to the correct project"
    echo "3. Check available local runs: ls wandb/"
    echo "4. List remote runs: uv run python scripts/infer_from_wandb.py --list --project '$PROJECT_NAME'"
    exit 1
elif [[ "$LOOKUP_RESULT" == "DUPLICATES" ]]; then
    echo "Error: Multiple runs found matching '$RUN_IDENTIFIER' in project $PROJECT_NAME"
    echo "Please be more specific with the run name"
    echo "Use: uv run python scripts/infer_from_wandb.py --list '$RUN_IDENTIFIER' --project '$PROJECT_NAME'"
    exit 1
elif [[ "$LOOKUP_RESULT" =~ ^ERROR: ]]; then
    echo "$LOOKUP_RESULT"
    exit 1
fi

RUN_HASH="$LOOKUP_RESULT"
echo "Found run hash: $RUN_HASH"

# Find the local wandb run directory
WANDB_RUN_DIR_RESULT=$(uv run python scripts/wandb_utils.py find_run_dir "$RUN_HASH")

if [[ "$WANDB_RUN_DIR_RESULT" == "NOT_FOUND" ]]; then
    echo "Error: No local wandb directory found for run hash $RUN_HASH"
    echo "Run 'wandb sync' or download the run data locally first"
    echo "Expected directory: wandb/run-*-$RUN_HASH"
    exit 1
elif [[ "$WANDB_RUN_DIR_RESULT" =~ ^ERROR: ]]; then
    echo "$WANDB_RUN_DIR_RESULT"
    exit 1
fi

WANDB_RUN_DIR="$WANDB_RUN_DIR_RESULT"
echo "Found wandb run directory: $WANDB_RUN_DIR"

# Look for train.rec file in the checkpoints directory
RECORDING_RESULT=$(uv run python scripts/wandb_utils.py find_recording "$WANDB_RUN_DIR")

if [[ "$RECORDING_RESULT" == "NOT_FOUND" ]]; then
    echo ""
    echo "No train.rec file found in the checkpoint directory."
    echo "This means the training run was not recorded with the --record flag."
    echo ""
    echo "To replay this run, you would need to:"
    echo "1. Re-run training with --record flag, or"
    echo "2. Run inference to generate a recording file using:"
    echo "   uv run python scripts/infer_from_wandb.py $RUN_HASH"
    echo ""
    echo "Available files in checkpoint directory:"
    ls -la "$WANDB_RUN_DIR/files/checkpoints/" 2>/dev/null || echo "  (directory not accessible)"
    exit 1
elif [[ "$RECORDING_RESULT" =~ ^ERROR: ]]; then
    echo "$RECORDING_RESULT"
    exit 1
fi

TRAINING_RECORDING="$RECORDING_RESULT"
echo "Found training recording: $(basename "$TRAINING_RECORDING")"

# Check if build directory exists and has viewer
if [[ ! -f "./build/viewer" ]]; then
    echo "Error: Viewer not found at ./build/viewer"
    echo "Please build the project first with: ./build.sh"
    exit 1
fi

echo ""
echo "Starting viewer to replay training recording..."
echo "Recording: $(basename "$TRAINING_RECORDING")"
echo "Press Ctrl+C to stop"
echo ""

# Launch viewer in replay mode with any additional arguments
exec ./build/viewer --replay "$TRAINING_RECORDING" --auto-reset "$@"
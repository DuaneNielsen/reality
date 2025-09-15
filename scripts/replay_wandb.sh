#!/bin/bash

# Exit on any error
set -e

# Default number of steps for inference
DEFAULT_NUM_STEPS=2000

# Parse command line arguments
RUN_IDENTIFIER=""
NUM_STEPS=$DEFAULT_NUM_STEPS
VIEWER_ARGS=()

# Check if wandb run identifier provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <wandb_run_identifier> [options...] [viewer_options...]"
    echo "Run identifier can be:"
    echo "  - Human-friendly name: frosty-sweep-1, cosmic-sweep-63"
    echo "  - Partial name: cosmic-sweep (will lookup exact run hash)"
    echo "  - Run hash: i0jv5zou"
    echo ""
    echo "Options:"
    echo "  --num-steps <N>     Number of steps to record during inference (default: $DEFAULT_NUM_STEPS)"
    echo ""
    echo "Examples:"
    echo "  $0 cosmic-sweep-63"
    echo "  $0 cosmic-sweep-63 --num-steps 5000"
    echo "  $0 i0jv5zou --num-steps 1000 --fps 30"
    exit 1
fi

RUN_IDENTIFIER="$1"
shift  # Remove first argument

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        *)
            # All other arguments are passed to viewer
            VIEWER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Find wandb run directory by hash or name
echo "Looking for wandb run: $RUN_IDENTIFIER"

# Try to find by exact hash match first
WANDB_RUN_DIRS=(wandb/*-"$RUN_IDENTIFIER")
if [[ ! -d "${WANDB_RUN_DIRS[0]}" ]]; then
    # Try to find by run name
    WANDB_RUN_DIRS=(wandb/"$RUN_IDENTIFIER"-*)
    if [[ ! -d "${WANDB_RUN_DIRS[0]}" ]]; then
        # Try partial match
        WANDB_RUN_DIRS=(wandb/*"$RUN_IDENTIFIER"*)
        if [[ ! -d "${WANDB_RUN_DIRS[0]}" ]]; then
            # No local directory found, try to lookup the run hash using the API
            echo "No local directory found, trying to lookup run hash..."
            PROJECT_NAME="${WANDB_PROJECT:-madrona-escape-room-dev}"
            
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
            elif [[ "$LOOKUP_RESULT" =~ ^[a-z0-9]{8}$ ]]; then
                # Valid hash returned, try to find local directory with this hash
                echo "Found run hash: $LOOKUP_RESULT"
                WANDB_RUN_DIRS=(wandb/*-"$LOOKUP_RESULT")
                if [[ ! -d "${WANDB_RUN_DIRS[0]}" ]]; then
                    echo "Error: Found run '$RUN_IDENTIFIER' (hash: $LOOKUP_RESULT) but no local wandb directory"
                    echo "Run 'wandb sync' or download the run data locally first"
                    echo "Expected directory: wandb/run-*-$LOOKUP_RESULT"
                    exit 1
                fi
            else
                echo "Error: No wandb run found matching '$RUN_IDENTIFIER' in project $PROJECT_NAME"
                echo "Lookup result: $LOOKUP_RESULT"
                echo ""
                echo "Available local runs:"
                ls -1 wandb/ | grep "run-" | head -10
                echo ""
                echo "To see all runs in wandb project:"
                echo "uv run python scripts/infer_from_wandb.py --list --project '$PROJECT_NAME'"
                exit 1
            fi
        fi
    fi
fi

WANDB_RUN_DIR="${WANDB_RUN_DIRS[0]}"
echo "Found wandb run directory: $WANDB_RUN_DIR"

# Look for .rec files in the checkpoints directory
CHECKPOINTS_DIR="$WANDB_RUN_DIR/files/checkpoints"
if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    echo "Error: Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
fi

# Find .pth checkpoint files first to determine the latest checkpoint
CKPT_FILES=("$CHECKPOINTS_DIR"/*.pth)
if [[ ! -f "${CKPT_FILES[0]}" ]]; then
    echo "Error: No checkpoint files (.pth) found in $CHECKPOINTS_DIR"
    echo "Available files:"
    ls -la "$CHECKPOINTS_DIR"
    exit 1
fi

# Use the latest checkpoint
LATEST_CKPT=$(ls -t "${CKPT_FILES[@]}" | head -1)
CKPT_NAME=$(basename "$LATEST_CKPT" .pth)
echo "Found latest checkpoint: $CKPT_NAME"

# Check if corresponding .rec file exists for the latest checkpoint
EXPECTED_REC_FILE="$CHECKPOINTS_DIR/$CKPT_NAME.rec"
if [[ ! -f "$EXPECTED_REC_FILE" ]]; then
    echo "No .rec file found for latest checkpoint: $CKPT_NAME"
    echo "Generating recording file through inference..."
    
    # Extract the run hash from the directory path
    RUN_HASH=$(basename "$WANDB_RUN_DIR" | sed 's/.*-//')
    
    echo "Running inference to generate recording..."
    echo "Steps: $NUM_STEPS"
    echo "This may take a few minutes..."

    # Run inference using the hash (which will find the run and generate .rec file)
    # Use project from environment variable or default to madrona-escape-room-dev
    PROJECT_NAME="${WANDB_PROJECT:-madrona-escape-room-dev}"
    echo "Using project: $PROJECT_NAME"
    
    if ! uv run python scripts/infer_from_wandb.py "$RUN_HASH" --num-steps "$NUM_STEPS" --project "$PROJECT_NAME"; then
        echo "Error: Failed to run inference"
        exit 1
    fi
    
    echo "Inference completed, checking for generated .rec file..."
    
    # Check if the expected .rec file was generated
    if [[ ! -f "$EXPECTED_REC_FILE" ]]; then
        echo "Error: Expected .rec file not generated: $EXPECTED_REC_FILE"
        echo "Available .rec files:"
        ls -la "$CHECKPOINTS_DIR"/*.rec 2>/dev/null || echo "None found"
        exit 1
    fi
    
    echo "Generated recording file successfully: $(basename "$EXPECTED_REC_FILE")"
fi

# Use the .rec file corresponding to the latest checkpoint
LATEST_REC="$EXPECTED_REC_FILE"

# Show information about other available .rec files if they exist
OTHER_REC_FILES=("$CHECKPOINTS_DIR"/*.rec)
if [[ -f "${OTHER_REC_FILES[0]}" ]]; then
    NUM_REC_FILES=${#OTHER_REC_FILES[@]}
    if [[ $NUM_REC_FILES -gt 1 ]]; then
        echo "Found $NUM_REC_FILES recording files in total:"
        for rec_file in "${OTHER_REC_FILES[@]}"; do
            if [[ "$rec_file" == "$LATEST_REC" ]]; then
                echo "  $(basename "$rec_file") [LATEST - USING THIS]"
            else
                echo "  $(basename "$rec_file")"
            fi
        done
    fi
fi

echo "Using recording file: $LATEST_REC"

# Check if build directory exists and has viewer
if [[ ! -f "./build/viewer" ]]; then
    echo "Error: Viewer not found at ./build/viewer"
    echo "Please build the project first with: ./build.sh"
    exit 1
fi

echo "Starting viewer in replay mode..."
echo "Recording: $(basename "$LATEST_REC")"
echo "Press Ctrl+C to stop"
echo

# Launch viewer in replay mode with any additional arguments
exec ./build/viewer --replay "$LATEST_REC" --auto-reset "${VIEWER_ARGS[@]}"
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
            
            LOOKUP_RESULT=$(uv run python scripts/infer_from_wandb.py --lookup "$RUN_IDENTIFIER" --project "$PROJECT_NAME")
            
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

# Find .rec files
REC_FILES=("$CHECKPOINTS_DIR"/*.rec)
if [[ ! -f "${REC_FILES[0]}" ]]; then
    echo "No .rec files found in $CHECKPOINTS_DIR"
    echo "Looking for checkpoint files to run inference..."
    
    # Find .pth checkpoint files
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
    echo "Found checkpoint: $CKPT_NAME"
    
    # Extract the run hash from the directory path
    RUN_HASH=$(basename "$WANDB_RUN_DIR" | sed 's/.*-//')
    
    echo "Running inference to generate recording..."
    echo "This may take a few minutes..."
    
    # Run inference using the hash (which will find the run and generate .rec file)
    # Use project from environment variable or default to madrona-escape-room-dev
    PROJECT_NAME="${WANDB_PROJECT:-madrona-escape-room-dev}"
    echo "Using project: $PROJECT_NAME"
    
    if ! uv run python scripts/infer_from_wandb.py "$RUN_HASH" --num-steps 1000 --project "$PROJECT_NAME"; then
        echo "Error: Failed to run inference"
        exit 1
    fi
    
    echo "Inference completed, looking for generated .rec file..."
    
    # Refresh the .rec files list
    REC_FILES=("$CHECKPOINTS_DIR"/*.rec)
    if [[ ! -f "${REC_FILES[0]}" ]]; then
        echo "Error: No .rec file generated after inference"
        exit 1
    fi
    
    echo "Generated recording file successfully!"
fi

# If multiple .rec files, use the latest one
if [[ ${#REC_FILES[@]} -gt 1 ]]; then
    echo "Found ${#REC_FILES[@]} recording files, using the latest:"
    for rec_file in "${REC_FILES[@]}"; do
        echo "  $(basename "$rec_file")"
    done
    # Sort by modification time and take the latest
    LATEST_REC=$(ls -t "${REC_FILES[@]}" | head -1)
else
    LATEST_REC="${REC_FILES[0]}"
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
exec ./build/viewer --replay "$LATEST_REC" "$@"
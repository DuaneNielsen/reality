#!/bin/bash

# Debug script to run inference and capture trajectory logging
RUN_DIR="wandb/run-20250926_165849-aowz3q2a/files/checkpoints"

echo "Running inference with debug logging..."
echo "RUN_DIR: $RUN_DIR"

# Run inference to capture expected checksum values during recording
# Filter to only capture position debug output and spawn debug
uv run python scripts/infer.py \
    --ckpt-path "$RUN_DIR/3300.pth" \
    --recording-path "$RUN_DIR/3300_debug.rec" \
    --num-worlds 50 \
    --num-steps 1000 \
    --num-channels 1024 \
    --level-file "$RUN_DIR/four_room_progression_32_levels_spawn_random.lvl" \
    --track-world 31 \
    2>&1 | grep -E "World (31|32|38)|Entity.*LeafID|SPAWN_DEBUG|RESET_DEBUG" > inference_debug_filtered.log

echo "Inference complete. Now running viewer for replay with trajectory tracking..."

# Run viewer to capture step-by-step trajectory during replay
# Track World 31 Agent 0 to get detailed position data
timeout 120s ./build/viewer --replay "$RUN_DIR/3300_debug.rec" \
    --track --track-world 31 --track-agent 0 --track-file "world31_replay_trajectory.csv" \
    2>&1 | grep -E "World (31|32|38)|Entity.*LeafID|SPAWN_DEBUG|Checksum mismatch|Episode step" > viewer_debug_filtered.log

echo "Debug run complete."
echo "Check inference_debug_filtered.log for recording debug output"
echo "Check viewer_debug_filtered.log for replay debug output"
echo "Check world31_recording_trajectory.csv for step-by-step World 31 recording trajectory"
echo "Check world31_replay_trajectory.csv for step-by-step World 31 replay trajectory"
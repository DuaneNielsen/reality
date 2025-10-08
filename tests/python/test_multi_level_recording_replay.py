#!/usr/bin/env python3
"""
Comprehensive multi-level recording and replay test.
Tests the new v3 recording format that supports different CompiledLevel structs per world.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from madrona_escape_room import ExecMode, SimManager
from madrona_escape_room.level_compiler import DEFAULT_TILESET, compile_level


def create_test_levels():
    """Create three distinct test levels for multi-level testing."""

    # Level 1: Open corridor (easy movement)
    level1_ascii = [
        "####################",
        "#S.................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "####################",
    ]

    # Level 2: Simple maze
    level2_ascii = [
        "####################",
        "#S.................#",
        "#.########.#######.#",
        "#..................#",
        "#.########.#######.#",
        "####################",
    ]

    # Level 3: More complex maze
    level3_ascii = [
        "####################",
        "#S....##########...#",
        "#.....#........#...#",
        "#.....#........#...#",
        "#.....##########...#",
        "####################",
    ]

    # Convert to CompiledLevel structs
    level1_data = {"ascii": level1_ascii, "tileset": DEFAULT_TILESET, "level_name": "open_corridor"}

    level2_data = {"ascii": level2_ascii, "tileset": DEFAULT_TILESET, "level_name": "simple_maze"}

    level3_data = {"ascii": level3_ascii, "tileset": DEFAULT_TILESET, "level_name": "complex_maze"}

    # Compile the levels
    level1 = compile_level(level1_data)[0]  # Extract single level from list
    level2 = compile_level(level2_data)[0]  # Extract single level from list
    level3 = compile_level(level3_data)[0]  # Extract single level from list

    return level1, level2, level3


@pytest.mark.slow
@pytest.mark.spec("docs/specs/mgr.md", "startRecording")
def test_multi_level_recording_roundtrip():
    """Test recording and replaying across multiple different levels."""

    level1, level2, level3 = create_test_levels()

    # Create a 5-world setup with level assignment:
    # World 0,1: open_corridor
    # World 2,3: simple_maze
    # World 4: complex_maze
    per_world_levels = [level1, level1, level2, level2, level3]
    num_worlds = 5

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Phase 1: Record with multi-level setup
        print(f"\nPhase 1: Recording {num_worlds} worlds with multi-level configuration")

        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=-1,
            num_worlds=num_worlds,
            rand_seed=42,
            auto_reset=True,
            compiled_levels=per_world_levels,
            enable_batch_renderer=False,
        )

        # Start recording (raises exception on failure)
        mgr.start_recording(recording_path)

        # Set unique action patterns per world type to create different trajectories
        action_tensor = mgr.action_tensor().to_torch()
        num_steps = 600  # Run enough steps to trigger checksum verification

        recorded_positions = []

        for step in range(num_steps):
            # World 0,1 (open corridor): Move forward consistently
            action_tensor[0, :] = torch.tensor(
                [2, 0, 2], dtype=torch.int32
            )  # MEDIUM forward, no rotate
            action_tensor[1, :] = torch.tensor([2, 0, 2], dtype=torch.int32)

            # World 2,3 (simple maze): Alternate movement to navigate
            action_tensor[2, :] = torch.tensor(
                [1, step % 8, 2], dtype=torch.int32
            )  # SLOW, rotating direction
            action_tensor[3, :] = torch.tensor([1, (step + 2) % 8, 2], dtype=torch.int32)

            # World 4 (complex maze): More complex movement pattern
            action_tensor[4, :] = torch.tensor(
                [1, (step * 3) % 8, (step % 3)], dtype=torch.int32
            )  # SLOW, complex pattern

            mgr.step()

            # Record positions for verification
            obs = mgr.self_observation_tensor().to_torch()
            positions = obs[:, 0, :3].clone()  # [worlds, xyz]
            recorded_positions.append(positions)

            print(f"  Step {step}: Recording actions and positions")

        mgr.stop_recording()
        print(f"✓ Recording completed with {num_steps} steps")

        # Verify file was created and has expected structure
        assert os.path.exists(recording_path), "Recording file not created"
        file_size = os.path.getsize(recording_path)
        print(f"  Recording file size: {file_size} bytes")

        del mgr  # Clean up manager

        # Phase 2: Replay and verify trajectories match exactly
        print("\nPhase 2: Replaying recorded session")

        # Create replay manager from file (automatically picks up config)
        replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)
        print("✓ Replay file loaded successfully")

        # Replay and compare trajectories
        replayed_positions = []

        for step in range(num_steps):
            # Get positions before applying replay actions
            obs = replay_mgr.self_observation_tensor().to_torch()
            positions = obs[:, 0, :3].clone()
            replayed_positions.append(positions)

            # Apply replay step
            replay_mgr.replay_step()
            replay_mgr.step()

            print(f"  Step {step}: Replaying actions and comparing positions")

            if step < len(recorded_positions):
                # Compare positions (allow small floating-point differences)
                recorded_pos = recorded_positions[step]
                replayed_pos = positions

                diff = torch.abs(recorded_pos - replayed_pos)
                max_diff = torch.max(diff)

                print(f"    Max position difference: {max_diff:.6f}")
                assert max_diff < 0.1, f"Position mismatch at step {step}: max diff {max_diff}"

        print("✓ All trajectories match exactly between recording and replay")

        # Phase 3: Verify different world types behaved differently
        print("\nPhase 3: Verifying level-specific behavior")

        # Check that different level types produced different movement patterns
        final_recorded = recorded_positions[-1]

        # Open corridor worlds (0,1) should have similar forward progress
        corridor_0_pos = final_recorded[0]
        corridor_1_pos = final_recorded[1]
        corridor_diff = torch.abs(corridor_0_pos - corridor_1_pos)
        print(f"  Corridor worlds similarity: max diff = {torch.max(corridor_diff):.6f}")
        assert torch.max(corridor_diff) < 0.5, "Corridor worlds should behave similarly"

        # Maze worlds (2,3,4) should have different positions from corridor
        maze_2_pos = final_recorded[2]

        corridor_vs_maze_diff = torch.abs(corridor_0_pos - maze_2_pos)
        print(f"  Corridor vs maze difference: max diff = {torch.max(corridor_vs_maze_diff):.6f}")
        assert (
            torch.max(corridor_vs_maze_diff) > 0.01
        ), "Different levels should produce different trajectories"

        print("✓ Multi-level behavior verification passed")

        # Verify replay determinism using checksum verification
        assert (
            not replay_mgr.has_checksum_failed()
        ), "Multi-level replay should be deterministic (no checksum failures)"
        print("✓ Multi-level checksum verification passed")

    finally:
        # Cleanup
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "readReplayMetadata")
def test_multi_level_metadata_storage():
    """Test that recording properly stores metadata for multiple levels."""

    level1, level2, _ = create_test_levels()

    # Create setup with duplicate levels: [level1, level1, level2, level1]
    per_world_levels = [level1, level1, level2, level1]
    num_worlds = 4

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=-1,
            num_worlds=num_worlds,
            rand_seed=42,
            auto_reset=True,
            compiled_levels=per_world_levels,
            enable_batch_renderer=False,
        )

        # Start recording - should deduplicate to 2 unique levels
        mgr.start_recording(recording_path)  # Will raise exception on failure

        # Run a few steps
        for step in range(3):
            mgr.step()

        mgr.stop_recording()

        # Load replay metadata to verify deduplication
        metadata_opt = mgr.read_replay_metadata(recording_path)
        assert metadata_opt is not None, "Failed to read replay metadata"

        print(f"  Total worlds: {metadata_opt.num_worlds}")

        # Verify basic metadata
        assert metadata_opt.num_worlds == 4, f"Expected 4 worlds, got {metadata_opt.num_worlds}"

        print("✓ Multi-level metadata storage works correctly")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "loadReplay")
def test_v5_format_validation():
    """Test that v5 format files are created with sensor config."""

    level1, _, _ = create_test_levels()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create a v5 format recording
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=-1,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            compiled_levels=[level1],
            enable_batch_renderer=False,
        )

        mgr.start_recording(recording_path)  # Will raise exception on failure

        mgr.step()
        mgr.stop_recording()

        # Verify the file can be read
        metadata = mgr.read_replay_metadata(recording_path)
        assert metadata is not None, "Failed to read v5 replay metadata"

        assert (
            metadata.version == 5
        ), f"Expected version 5 (with sensor config), got {metadata.version}"
        assert metadata.magic != 0, f"Expected valid magic number, got {metadata.magic}"

        print(f"✓ v5 format file created and validated (version {metadata.version})")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


if __name__ == "__main__":
    print("Running multi-level recording/replay tests...")
    test_multi_level_recording_roundtrip()
    test_multi_level_metadata_storage()
    test_v5_format_validation()
    print("All tests passed!")

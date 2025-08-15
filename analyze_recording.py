#!/usr/bin/env python3
"""Analyze the binary structure of a recording file."""

import struct
import sys


def read_recording(filename):
    """Read and analyze a recording file."""
    with open(filename, "rb") as f:
        # Read ReplayMetadata (64 bytes)
        magic = f.read(4)
        version = struct.unpack("I", f.read(4))[0]
        env_name = f.read(20).decode("ascii").rstrip("\x00")
        _ = struct.unpack("Q", f.read(8))[0]  # start_time (unused)
        _ = struct.unpack("Q", f.read(8))[0]  # end_time (unused)
        num_worlds = struct.unpack("I", f.read(4))[0]
        num_agents = struct.unpack("I", f.read(4))[0]
        num_steps = struct.unpack("I", f.read(4))[0]
        actions_per_step = struct.unpack("I", f.read(4))[0]
        seed = struct.unpack("I", f.read(4))[0]

        print("=== ReplayMetadata (64 bytes) ===")
        print(f"Magic: {magic}")
        print(f"Version: {version}")
        print(f"Env: {env_name}")
        print(f"Worlds: {num_worlds}, Agents: {num_agents}")
        print(f"Steps: {num_steps}, Actions per step: {actions_per_step}")
        print(f"Seed: {seed}")
        print()

        # Read CompiledLevel (8216 bytes)
        level_start = f.tell()
        width = struct.unpack("I", f.read(4))[0]
        height = struct.unpack("I", f.read(4))[0]
        num_tiles = struct.unpack("I", f.read(4))[0]
        padding = struct.unpack("I", f.read(4))[0]

        print(f"=== CompiledLevel at offset {level_start} ===")
        print(f"Width: {width}, Height: {height}")
        print(f"Num tiles: {num_tiles}")
        print(f"Padding: {padding}")

        # Skip the rest of CompiledLevel (2048 * 4 bytes = 8192 bytes for tiles)
        f.seek(level_start + 8216)

        actions_start = f.tell()
        print(f"\n=== Actions start at offset {actions_start} ===")

        # Read first few actions as int32_t
        print("\nFirst 10 steps of actions (as int32_t):")
        for step in range(min(10, num_steps)):
            step_actions = []
            for world in range(num_worlds):
                # Read 3 int32_t values per world
                move_amount = struct.unpack("i", f.read(4))[0]
                move_angle = struct.unpack("i", f.read(4))[0]
                rotate = struct.unpack("i", f.read(4))[0]
                step_actions.append((move_amount, move_angle, rotate))
            print(f"Step {step}: {step_actions}")

        # Check file size
        f.seek(0, 2)  # Go to end
        file_size = f.tell()
        expected_actions_size = num_steps * num_worlds * 3 * 4  # 3 int32_t per world
        expected_total = 64 + 8216 + expected_actions_size

        print("\n=== File size analysis ===")
        print(f"Actual file size: {file_size}")
        print(f"Expected: {expected_total}")
        print("  Metadata: 64 bytes")
        print("  Level: 8216 bytes")
        print(f"  Actions: {expected_actions_size} bytes ({num_steps} * {num_worlds} * 3 * 4)")

        if file_size != expected_total:
            print(f"WARNING: Size mismatch! Difference: {file_size - expected_total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_recording.py <recording_file>")
        sys.exit(1)

    read_recording(sys.argv[1])

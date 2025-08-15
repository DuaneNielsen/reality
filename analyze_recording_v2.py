#!/usr/bin/env python3
"""Analyze the binary structure of a recording file - Version 2."""

import struct
import sys


def read_recording(filename):
    """Read and analyze a recording file with correct structure sizes."""
    with open(filename, "rb") as f:
        # Read ReplayMetadata (136 bytes total)
        # uint32_t magic
        # uint32_t version
        # char sim_name[64]
        # uint32_t num_worlds
        # uint32_t num_agents_per_world
        # uint32_t num_steps
        # uint32_t actions_per_step
        # uint64_t timestamp
        # uint32_t seed
        # uint32_t reserved[8] (32 bytes)

        magic_bytes = f.read(4)
        magic = struct.unpack("<I", magic_bytes)[0]
        version = struct.unpack("<I", f.read(4))[0]
        sim_name = f.read(64).decode("ascii").rstrip("\x00")
        num_worlds = struct.unpack("<I", f.read(4))[0]
        num_agents = struct.unpack("<I", f.read(4))[0]
        num_steps = struct.unpack("<I", f.read(4))[0]
        actions_per_step = struct.unpack("<I", f.read(4))[0]
        timestamp = struct.unpack("<Q", f.read(8))[0]
        seed = struct.unpack("<I", f.read(4))[0]
        _ = f.read(32)  # reserved (8 * 4 bytes, unused)

        print("=== ReplayMetadata (136 bytes) ===")
        print(f"Magic: 0x{magic:08x} (bytes: {magic_bytes.hex()})")
        print(f"Version: {version}")
        print(f"Sim name: '{sim_name}'")
        print(f"Worlds: {num_worlds}, Agents per world: {num_agents}")
        print(f"Steps: {num_steps}, Actions per step: {actions_per_step}")
        print(f"Timestamp: {timestamp}")
        print(f"Seed: {seed}")
        print()

        # Read CompiledLevel (8216 bytes)
        level_start = f.tell()
        print(f"=== CompiledLevel at offset {level_start} ===")

        width = struct.unpack("<I", f.read(4))[0]
        height = struct.unpack("<I", f.read(4))[0]
        num_tiles = struct.unpack("<I", f.read(4))[0]
        padding = struct.unpack("<I", f.read(4))[0]

        print(f"Width: {width}, Height: {height}")
        print(f"Num tiles: {num_tiles}")
        print(f"Padding: {padding}")

        # Skip tile data (2048 * 4 bytes)
        f.seek(level_start + 8216)

        actions_start = f.tell()
        print(f"\n=== Actions start at offset {actions_start} ===")

        # Check how many steps we can actually read
        f.seek(0, 2)  # Go to end
        file_size = f.tell()
        f.seek(actions_start)  # Go back to actions

        bytes_for_actions = file_size - actions_start
        bytes_per_step = num_worlds * 3 * 4  # 3 int32_t per world
        actual_steps = bytes_for_actions // bytes_per_step if bytes_per_step > 0 else 0

        print(f"File has {bytes_for_actions} bytes for actions")
        print(f"Each step needs {bytes_per_step} bytes ({num_worlds} worlds * 3 actions * 4 bytes)")
        print(f"Can read {actual_steps} complete steps")

        if num_steps > 0:
            print(f"\nMetadata claims {num_steps} steps")
            if actual_steps < num_steps:
                print(f"WARNING: File only has data for {actual_steps} steps!")

        # Read first few actions
        if actual_steps > 0:
            print(f"\nFirst {min(10, actual_steps)} steps of actions (as int32_t):")
            for step in range(min(10, actual_steps)):
                step_actions = []
                for world in range(num_worlds):
                    # Read 3 int32_t values per world
                    move_amount = struct.unpack("<i", f.read(4))[0]
                    move_angle = struct.unpack("<i", f.read(4))[0]
                    rotate = struct.unpack("<i", f.read(4))[0]
                    step_actions.append((move_amount, move_angle, rotate))
                print(f"Step {step}: {step_actions}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_recording_v2.py <recording_file>")
        sys.exit(1)

    read_recording(sys.argv[1])

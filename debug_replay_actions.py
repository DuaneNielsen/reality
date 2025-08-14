#!/usr/bin/env python3
"""Debug script to analyze action data in recording files."""

import struct
import sys


def read_recording_metadata(filename):
    """Read the metadata from a recording file."""
    with open(filename, "rb") as f:
        # Read header
        magic = f.read(4)
        version = struct.unpack("I", f.read(4))[0]

        # Read environment name
        env_name = f.read(20).decode("ascii").rstrip("\x00")

        # Read timestamps
        start_time = struct.unpack("Q", f.read(8))[0]
        end_time = struct.unpack("Q", f.read(8))[0]

        # Read metadata
        num_worlds = struct.unpack("I", f.read(4))[0]
        num_agents = struct.unpack("I", f.read(4))[0]
        num_steps = struct.unpack("I", f.read(4))[0]
        action_size = struct.unpack("I", f.read(4))[0]

        return {
            "magic": magic,
            "version": version,
            "env_name": env_name,
            "start_time": start_time,
            "end_time": end_time,
            "num_worlds": num_worlds,
            "num_agents": num_agents,
            "num_steps": num_steps,
            "action_size": action_size,
            "header_size": 64,  # Total header size
        }


def read_actions(filename, metadata, num_steps_to_read=10):
    """Read the first N steps of actions from the file."""
    with open(filename, "rb") as f:
        # Skip header
        f.seek(metadata["header_size"])

        actions = []
        num_worlds = metadata["num_worlds"]
        action_size = metadata["action_size"]

        for step in range(min(num_steps_to_read, metadata["num_steps"])):
            step_actions = []
            for world in range(num_worlds):
                # Read action_size floats for this world
                world_actions = []
                for _ in range(action_size):
                    val = struct.unpack("f", f.read(4))[0]
                    world_actions.append(val)
                step_actions.append(world_actions)
            actions.append(step_actions)

        return actions


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_replay_actions.py <recording_file>")
        sys.exit(1)

    filename = sys.argv[1]

    # Read metadata
    metadata = read_recording_metadata(filename)
    print("Recording metadata:")
    print(f"  Environment: {metadata['env_name']}")
    print(f"  Worlds: {metadata['num_worlds']}")
    print(f"  Agents: {metadata['num_agents']}")
    print(f"  Steps: {metadata['num_steps']}")
    print(f"  Action size: {metadata['action_size']}")
    print()

    # Read first 10 steps of actions
    actions = read_actions(filename, metadata, 10)

    print("First 10 steps of actions:")
    for step_idx, step_actions in enumerate(actions):
        print(f"Step {step_idx}:")
        for world_idx, world_actions in enumerate(step_actions):
            print(f"  World {world_idx}: {world_actions}")
        print()


if __name__ == "__main__":
    main()

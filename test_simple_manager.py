#!/usr/bin/env python3
"""
Simple test to check if manager creation works.
"""

import madrona_escape_room


def test_simple():
    print("Creating manager...")
    mgr = madrona_escape_room.SimManager(
        exec_mode=madrona_escape_room.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
        compiled_levels=madrona_escape_room.create_default_level(),
    )
    print("Manager created successfully!")

    print("Testing step...")
    mgr.step()
    print("Step successful!")

    print("Destroying manager...")
    mgr.destroy()
    print("Manager destroyed successfully!")


if __name__ == "__main__":
    test_simple()

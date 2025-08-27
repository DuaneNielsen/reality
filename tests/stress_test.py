#!/usr/bin/env python3
"""
Stress test for Madrona Escape Room using Python bindings.
Tests the simulation with configurable iterations and level files.
"""

import argparse
import os
import sys
import time

import numpy as np

# Add parent directory to path to import madrona_escape_room
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from madrona_escape_room import ExecMode, SimManager
from madrona_escape_room.default_level import create_default_level


def run_stress_test(iterations=100, level_file=None, num_worlds=32):
    """
    Run stress test with specified iterations and level.

    Args:
        iterations: Number of simulation steps to run
        level_file: Optional .lvl file to load, uses Python default if None
        num_worlds: Number of parallel worlds to simulate
    """
    print("Running stress test:")
    print(f"  Iterations: {iterations}")
    print(f"  Worlds: {num_worlds}")

    if level_file:
        print(f"  Level: {level_file}")
        if not os.path.exists(level_file):
            print(f"Error: Level file '{level_file}' not found")
            return False

        # For codegen branch, we can't easily load .lvl files
        # Just use the default level instead
        print("  Note: Loading .lvl files not fully supported on codegen branch")
        print("  Using Python default level instead")
        level = create_default_level()
    else:
        print("  Level: Python default level")
        level = create_default_level()

    # Create manager with the level
    print(f"\nInitializing SimManager with {num_worlds} worlds...")
    try:
        # Codegen branch API
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=num_worlds,
            rand_seed=5,
            auto_reset=True,
            compiled_levels=[level],  # Note: plural 'levels'
        )
    except Exception as e:
        print(f"Error creating SimManager: {e}")
        return False

    print(f"Starting {iterations} iterations...")

    # Get action tensor for random actions
    action_tensor = mgr.action_tensor()
    actions = action_tensor.to_numpy()

    # Progress tracking
    report_interval = max(1, iterations // 10)  # Report 10 times

    # Timing
    start_time = time.time()

    try:
        for i in range(iterations):
            # Set random actions
            # Action space: [move_amount (0-3), move_angle (0-7), rotate (0-4)]
            # Check tensor dimensions
            if len(actions.shape) == 3:
                # 3D tensor: [worlds, agents, actions]
                actions[:, :, 0] = np.random.randint(0, 4, size=(num_worlds, 1))  # move_amount
                actions[:, :, 1] = np.random.randint(0, 8, size=(num_worlds, 1))  # move_angle
                actions[:, :, 2] = np.random.randint(0, 5, size=(num_worlds, 1))  # rotate
            else:
                # 2D tensor: [worlds, actions]
                actions[:, 0] = np.random.randint(0, 4, size=num_worlds)  # move_amount
                actions[:, 1] = np.random.randint(0, 8, size=num_worlds)  # move_angle
                actions[:, 2] = np.random.randint(0, 5, size=num_worlds)  # rotate

            # Step simulation
            mgr.step()

            # Report progress
            if (i + 1) % report_interval == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) * num_worlds / elapsed
                print(f"  Completed {i + 1}/{iterations} iterations - FPS: {fps:.0f}")

        # Final timing
        total_time = time.time() - start_time
        total_steps = iterations * num_worlds
        fps = total_steps / total_time

        print(f"\n✅ Success! Completed {iterations} iterations without crashes.")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Total steps: {total_steps:,} (iterations × worlds)")
        print(f"  Average FPS: {fps:.0f}")
        return True

    except Exception as e:
        print(f"\n❌ Error at iteration {i + 1}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        del mgr


def main():
    parser = argparse.ArgumentParser(description="Stress test for Madrona Escape Room simulation")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations to run (default: 100)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Level file to load (.lvl format). Uses Python default level if not specified",
    )
    parser.add_argument(
        "--worlds", type=int, default=32, help="Number of parallel worlds (default: 32)"
    )

    args = parser.parse_args()

    # Run the stress test
    success = run_stress_test(
        iterations=args.iterations, level_file=args.load, num_worlds=args.worlds
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to investigate non-determinism in replay system.

Performs controlled rotation actions and compares trajectory outputs
to identify the source of non-deterministic behavior.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import madrona_escape_room
from madrona_escape_room.manager import SimManager
from tests.python.test_helpers import AgentController, ObservationReader


def save_trajectory_to_csv(filename: str, trajectory_data: list):
    """Save trajectory data to CSV file"""
    with open(filename, "w") as f:
        # Write header
        f.write("step,pos_x,pos_y,pos_z,rotation,max_y_progress,reward,done,termination_reason\n")

        # Write data
        for step, data in enumerate(trajectory_data):
            f.write(
                f"{step},{data['pos_x']:.6f},{data['pos_y']:.6f},{data['pos_z']:.6f},"
                f"{data['rotation']:.6f},{data['max_y_progress']:.6f},{data['reward']:.3f},"
                f"{data['done']},{data['termination_reason']}\n"
            )


def run_rotation_test(
    test_name: str,
    num_steps: int = 25,
    use_recording: bool = False,
    recording_path: str = None,
    replay_mode: bool = False,
) -> list:
    """
    Run a controlled rotation test and return trajectory data.

    Args:
        test_name: Name for this test run
        num_steps: Number of simulation steps to run
        use_recording: Whether to record actions
        recording_path: Path to recording file (if recording)
        replay_mode: Whether this is a replay run

    Returns:
        List of trajectory data dictionaries
    """
    print(f"\n=== {test_name} ===")

    # Create manager
    if replay_mode and recording_path:
        print(f"Creating SimManager from replay: {recording_path}")
        mgr = SimManager.from_replay(recording_path, madrona_escape_room.ExecMode.CPU)
    else:
        print("Creating new SimManager")
        mgr = SimManager(
            exec_mode=madrona_escape_room.ExecMode.CPU,
            gpu_id=0,  # Required parameter
            num_worlds=1,  # Single world for simplicity
            rand_seed=42,  # Fixed seed for determinism
            auto_reset=False,  # Manual reset control
        )

    # Start recording if requested
    if use_recording and recording_path and not replay_mode:
        print(f"Starting recording: {recording_path}")
        mgr.start_recording(recording_path)

    # Enable trajectory logging to file for detailed analysis
    trajectory_file = f"{test_name.lower().replace(' ', '_')}_trajectory.csv"
    print(f"Enabling trajectory logging: {trajectory_file}")
    mgr.enable_trajectory_logging(0, 0, trajectory_file)

    # Initialize helpers
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset to ensure clean start
    print("Resetting world...")
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[0] = 1
    mgr.step()
    reset_tensor[0] = 0

    trajectory_data = []

    # Execute controlled rotation sequence
    for step in range(num_steps):
        # Set actions based on step
        controller.reset_actions()  # Clear all actions, set rotation to NONE

        if step < 10:
            # Steps 0-9: Rotate left slowly
            controller.rotate_only(rotation=madrona_escape_room.action.rotate.SLOW_LEFT)
            action_desc = "SLOW_LEFT"
        elif step < 15:
            # Steps 10-14: Stop rotation
            controller.rotate_only(rotation=madrona_escape_room.action.rotate.NONE)
            action_desc = "NONE"
        else:
            # Steps 15+: Rotate right slowly
            controller.rotate_only(rotation=madrona_escape_room.action.rotate.SLOW_RIGHT)
            action_desc = "SLOW_RIGHT"

        # Execute step
        if replay_mode:
            # For replay, need to call both replay_step() and step()
            replay_complete = mgr.replay_step()
            if not replay_complete:
                mgr.step()
            else:
                print(f"Replay completed at step {step}")
                break
        else:
            mgr.step()

        # Collect trajectory data
        pos = observer.get_normalized_position(0, 0)
        rotation = observer.get_rotation(0, 0)
        max_y = observer.get_max_y_progress(0, 0)
        reward = observer.get_reward(0, 0)
        done = observer.get_done_flag(0, 0)
        termination_reason = observer.get_termination_reason(0, 0)

        data = {
            "pos_x": pos[0],
            "pos_y": pos[1],
            "pos_z": pos[2],
            "rotation": rotation,
            "max_y_progress": max_y,
            "reward": reward,
            "done": done,
            "termination_reason": termination_reason,
        }
        trajectory_data.append(data)

        print(
            f"Step {step:2d}: {action_desc:10s} | "
            f"Pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) | "
            f"Rot={rotation:.3f} | Done={done}"
        )

        if done:
            print(f"Episode ended at step {step}, termination reason: {termination_reason}")
            break

    # Stop recording if active
    if use_recording and recording_path and not replay_mode:
        print("Stopping recording")
        mgr.stop_recording()

    # Disable trajectory logging
    mgr.disable_trajectory_logging()

    # Save trajectory data to CSV
    csv_filename = f"{test_name.lower().replace(' ', '_')}_data.csv"
    save_trajectory_to_csv(csv_filename, trajectory_data)
    print(f"Saved trajectory data: {csv_filename}")

    return trajectory_data


def compare_trajectories(traj1: list, traj2: list, name1: str, name2: str) -> bool:
    """Compare two trajectory datasets and report differences"""
    print(f"\n=== Comparing {name1} vs {name2} ===")

    if len(traj1) != len(traj2):
        print(f"❌ Length mismatch: {name1}={len(traj1)} steps, {name2}={len(traj2)} steps")
        return False

    max_steps = min(len(traj1), len(traj2))
    differences_found = False

    for step in range(max_steps):
        d1, d2 = traj1[step], traj2[step]

        # Check each field for differences
        diffs = []

        # Position differences (tolerance for floating point)
        pos_tol = 1e-6
        if abs(d1["pos_x"] - d2["pos_x"]) > pos_tol:
            diffs.append(f"pos_x: {d1['pos_x']:.6f} vs {d2['pos_x']:.6f}")
        if abs(d1["pos_y"] - d2["pos_y"]) > pos_tol:
            diffs.append(f"pos_y: {d1['pos_y']:.6f} vs {d2['pos_y']:.6f}")
        if abs(d1["pos_z"] - d2["pos_z"]) > pos_tol:
            diffs.append(f"pos_z: {d1['pos_z']:.6f} vs {d2['pos_z']:.6f}")

        # Rotation differences
        rot_tol = 1e-6
        if abs(d1["rotation"] - d2["rotation"]) > rot_tol:
            diffs.append(f"rotation: {d1['rotation']:.6f} vs {d2['rotation']:.6f}")

        # Other fields (exact match expected)
        if d1["max_y_progress"] != d2["max_y_progress"]:
            diffs.append(f"max_y_progress: {d1['max_y_progress']} vs {d2['max_y_progress']}")
        if d1["reward"] != d2["reward"]:
            diffs.append(f"reward: {d1['reward']} vs {d2['reward']}")
        if d1["done"] != d2["done"]:
            diffs.append(f"done: {d1['done']} vs {d2['done']}")
        if d1["termination_reason"] != d2["termination_reason"]:
            diffs.append(
                f"termination_reason: {d1['termination_reason']} vs {d2['termination_reason']}"
            )

        if diffs:
            if not differences_found:
                print(f"❌ First difference at step {step}:")
                differences_found = True
            else:
                print(f"❌ Step {step} differences:")
            for diff in diffs:
                print(f"    {diff}")

    if not differences_found:
        print("✅ Trajectories are identical")
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description="Test rotation determinism in Madrona simulation")
    parser.add_argument("--num-steps", type=int, default=25, help="Number of simulation steps")
    parser.add_argument(
        "--test-direct", action="store_true", help="Test direct simulation determinism (run twice)"
    )
    parser.add_argument(
        "--test-replay",
        action="store_true",
        help="Test replay determinism (record once, replay twice)",
    )
    parser.add_argument(
        "--test-recording-vs-replay",
        action="store_true",
        help="Test recording vs replay consistency",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if not any([args.test_direct, args.test_replay, args.test_recording_vs_replay, args.all]):
        print("Please specify at least one test to run. Use --help for options.")
        return

    # Create timestamped output directory
    timestamp = int(time.time())
    output_dir = f"rotation_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"Output directory: {output_dir}")

    results = {}

    # Test 1: Direct simulation determinism
    if args.test_direct or args.all:
        print("\n" + "=" * 60)
        print("TEST 1: Direct Simulation Determinism")
        print("=" * 60)

        traj1 = run_rotation_test("Direct Run 1", args.num_steps)
        traj2 = run_rotation_test("Direct Run 2", args.num_steps)

        results["direct_deterministic"] = compare_trajectories(
            traj1, traj2, "Direct Run 1", "Direct Run 2"
        )

    # Test 2: Replay determinism
    if args.test_replay or args.all:
        print("\n" + "=" * 60)
        print("TEST 2: Replay Determinism")
        print("=" * 60)

        recording_path = "rotation_test.rec"

        # Record once
        print("Recording simulation...")
        run_rotation_test(
            "Recording", args.num_steps, use_recording=True, recording_path=recording_path
        )

        # Replay twice
        print("\nReplaying simulation twice...")
        replay1 = run_rotation_test(
            "Replay 1", args.num_steps, recording_path=recording_path, replay_mode=True
        )
        replay2 = run_rotation_test(
            "Replay 2", args.num_steps, recording_path=recording_path, replay_mode=True
        )

        results["replay_deterministic"] = compare_trajectories(
            replay1, replay2, "Replay 1", "Replay 2"
        )

    # Test 3: Recording vs Replay consistency
    if args.test_recording_vs_replay or args.all:
        print("\n" + "=" * 60)
        print("TEST 3: Recording vs Replay Consistency")
        print("=" * 60)

        recording_path = "consistency_test.rec"

        # Record
        recording_traj = run_rotation_test(
            "Recording for Consistency",
            args.num_steps,
            use_recording=True,
            recording_path=recording_path,
        )

        # Replay
        replay_traj = run_rotation_test(
            "Replay for Consistency",
            args.num_steps,
            recording_path=recording_path,
            replay_mode=True,
        )

        results["recording_replay_consistent"] = compare_trajectories(
            recording_traj, replay_traj, "Recording", "Replay"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30s}: {status}")

    # Diagnosis
    print("\nDIAGNOSIS:")

    if "direct_deterministic" in results:
        if not results["direct_deterministic"]:
            print("❌ Direct simulation is non-deterministic - physics simulation issue")
        else:
            print("✅ Direct simulation is deterministic")

    if "replay_deterministic" in results:
        if not results["replay_deterministic"]:
            print("❌ Replay system is non-deterministic - replay implementation issue")
        else:
            print("✅ Replay system is deterministic")

    if "recording_replay_consistent" in results:
        if not results["recording_replay_consistent"]:
            print("❌ Recording and replay produce different results - serialization issue")
        else:
            print("✅ Recording and replay are consistent")

    print(f"\nAll output files saved in: {os.getcwd()}")


if __name__ == "__main__":
    main()

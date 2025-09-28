"""
Test script to investigate non-determinism in replay system.

Uses pytest fixtures and AgentController to perform controlled rotation tests
and identify the source of non-deterministic behavior.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from test_helpers import AgentController, ObservationReader

import madrona_escape_room


def run_rotation_sequence(mgr, num_steps: int = 25) -> list:
    """
    Run a controlled rotation sequence and return trajectory data.

    Args:
        mgr: SimManager instance
        num_steps: Number of simulation steps to run

    Returns:
        List of trajectory data dictionaries
    """
    # Initialize helpers
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset to ensure clean start - but don't step yet
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[0] = 1

    trajectory_data = []

    # Execute controlled rotation sequence
    for step in range(num_steps):
        # Set actions based on step BEFORE stepping
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

        # Execute step AFTER setting action
        mgr.step()
        reset_tensor[0] = 0  # Clear reset flag after first step

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
        if abs(d1["max_y_progress"] - d2["max_y_progress"]) > 1e-6:
            diffs.append(f"max_y_progress: {d1['max_y_progress']} vs {d2['max_y_progress']}")
        if abs(d1["reward"] - d2["reward"]) > 1e-6:
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


def test_direct_simulation_determinism(cpu_manager):
    """Test that direct simulation (no recording/replay) is deterministic"""
    print("\n" + "=" * 60)
    print("TEST: Direct Simulation Determinism")
    print("=" * 60)

    mgr = cpu_manager

    # Run the same rotation sequence twice
    print("Running first simulation...")
    traj1 = run_rotation_sequence(mgr, num_steps=25)

    print("\nRunning second simulation...")
    traj2 = run_rotation_sequence(mgr, num_steps=25)

    # Compare results
    is_deterministic = compare_trajectories(traj1, traj2, "Direct Run 1", "Direct Run 2")

    if is_deterministic:
        print("✅ Direct simulation is deterministic")
    else:
        print("❌ Direct simulation is non-deterministic - physics simulation issue")

    # For pytest compatibility, don't return anything
    assert is_deterministic, "Direct simulation should be deterministic"


def test_replay_determinism(cpu_manager):
    """Test that replay system is deterministic (replay same recording twice)"""
    print("\n" + "=" * 60)
    print("TEST: Replay Determinism")
    print("=" * 60)

    mgr = cpu_manager
    recording_path = "rotation_test.rec"

    # Record the simulation
    print("Recording simulation...")
    mgr.start_recording(recording_path)
    run_rotation_sequence(mgr, num_steps=25)
    mgr.stop_recording()

    # Create two replay managers and run them
    print("\nCreating first replay manager...")
    replay_mgr1 = madrona_escape_room.SimManager.from_replay(
        recording_path, madrona_escape_room.ExecMode.CPU
    )

    print("Creating second replay manager...")
    replay_mgr2 = madrona_escape_room.SimManager.from_replay(
        recording_path, madrona_escape_room.ExecMode.CPU
    )

    # Run replays
    print("Running first replay...")
    traj_replay1 = run_replay_sequence(replay_mgr1, num_steps=25)

    print("Running second replay...")
    traj_replay2 = run_replay_sequence(replay_mgr2, num_steps=25)

    # Compare replays
    is_deterministic = compare_trajectories(traj_replay1, traj_replay2, "Replay 1", "Replay 2")

    if is_deterministic:
        print("✅ Replay system is deterministic")
    else:
        print("❌ Replay system is non-deterministic - replay implementation issue")

    assert is_deterministic, "Replay system should be deterministic"


def run_replay_sequence(mgr, num_steps: int = 25) -> list:
    """
    Run a replay sequence and return trajectory data.

    Args:
        mgr: SimManager instance in replay mode
        num_steps: Number of simulation steps to run

    Returns:
        List of trajectory data dictionaries
    """
    # Initialize observer (no controller needed for replay)
    observer = ObservationReader(mgr)

    trajectory_data = []

    # Execute replay sequence
    for step in range(num_steps):
        # For replay, need to call both replay_step() and step()
        replay_complete = mgr.replay_step()
        if not replay_complete:
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
            f"Step {step:2d}: "
            f"Pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) | "
            f"Rot={rotation:.3f} | Done={done}"
        )

        if replay_complete:
            print(f"Replay completed at step {step}")
            break

        if done:
            print(f"Episode ended at step {step}, termination reason: {termination_reason}")
            break

    return trajectory_data


def test_3_step_action_offset(cpu_manager):
    """
    Minimal 3-step test to isolate the action offset issue.
    Expected: A, B, C
    Actual replay: A, A, B (one step behind)
    """
    print("\n=== 3-Step Action Offset Test ===")

    mgr = cpu_manager
    mgr.start_recording("test_3_step_offset.rec")

    # Set action A (SLOW_LEFT)
    action_tensor = mgr.action_tensor().to_torch()
    action_tensor[0, 0] = 0  # move_amount
    action_tensor[0, 1] = 0  # move_angle
    action_tensor[0, 2] = 1  # SLOW_LEFT
    print("   Step 1: Set action A (SLOW_LEFT=1)")
    mgr.step()

    # Set action B (NONE)
    action_tensor[0, 0] = 0  # move_amount
    action_tensor[0, 1] = 0  # move_angle
    action_tensor[0, 2] = 2  # NONE
    print("   Step 2: Set action B (NONE=2)")
    mgr.step()

    # Set action C (SLOW_RIGHT)
    action_tensor[0, 0] = 0  # move_amount
    action_tensor[0, 1] = 0  # move_angle
    action_tensor[0, 2] = 3  # SLOW_RIGHT
    print("   Step 3: Set action C (SLOW_RIGHT=3)")
    mgr.step()

    mgr.stop_recording()

    # Replay using the debug output from the existing logging
    print("\n2. REPLAY:")
    replay_mgr = madrona_escape_room.SimManager.from_replay(
        "test_3_step_offset.rec", madrona_escape_room.ExecMode.CPU
    )

    for step in range(3):
        replay_complete = replay_mgr.replay_step()
        if not replay_complete:
            replay_mgr.step()
            print(f"   Step {step+1}: Replay step completed")
        else:
            break

    print("Expected: Step 1=A(1), Step 2=B(2), Step 3=C(3)")
    print("If offset exists: Step 1=A(1), Step 2=A(1), Step 3=B(2)")
    print("Check the debug output above to see what actions were loaded at each step")


def test_recording_vs_replay_consistency(cpu_manager):
    """Test that recording and replay produce consistent results"""
    print("\n" + "=" * 60)
    print("TEST: Recording vs Replay Consistency")
    print("=" * 60)

    mgr = cpu_manager
    recording_path = "consistency_test.rec"

    # Record the simulation
    print("Recording simulation...")
    mgr.start_recording(recording_path)

    # Enable trajectory logging for recording
    print("Enabling trajectory logging for recording...")
    mgr.enable_trajectory_logging(0, 0, "recording_for_consistency_trajectory.csv")

    traj_recording = run_rotation_sequence(mgr, num_steps=25)

    # Disable trajectory logging
    mgr.disable_trajectory_logging()
    mgr.stop_recording()

    # Create replay manager and run it
    print("Creating replay manager...")
    replay_mgr = madrona_escape_room.SimManager.from_replay(
        recording_path, madrona_escape_room.ExecMode.CPU
    )

    # Enable trajectory logging for replay
    print("Enabling trajectory logging for replay...")
    replay_mgr.enable_trajectory_logging(0, 0, "replay_for_consistency_trajectory.csv")

    print("Running replay...")
    traj_replay = run_replay_sequence(replay_mgr, num_steps=25)

    # Disable trajectory logging
    replay_mgr.disable_trajectory_logging()

    # Compare recording vs replay
    is_consistent = compare_trajectories(traj_recording, traj_replay, "Recording", "Replay")

    if is_consistent:
        print("✅ Recording and replay are consistent")
    else:
        print("❌ Recording and replay produce different results - serialization issue")

    assert is_consistent, "Recording and replay should be consistent"


if __name__ == "__main__":
    # Allow running as script for debugging
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))

"""
Test rotation determinism using only Manager's built-in trajectory logging.
This replaces the old multi-system approach with a single, consistent trajectory system.
"""

import os
import tempfile

import pytest

import madrona_escape_room


class AgentController:
    """Helper to set agent actions via tensor interface"""

    def __init__(self, mgr):
        self.mgr = mgr
        self.action_tensor = mgr.action_tensor().to_torch()

    def reset_actions(self):
        """Reset all actions to default (no movement, no rotation)"""
        self.action_tensor[0, 0] = 0  # move_amount = STOP
        self.action_tensor[0, 1] = 0  # move_angle = FORWARD
        self.action_tensor[0, 2] = 2  # rotate = NONE

    def rotate_only(self, rotation):
        """Set only rotation action, keeping movement at zero"""
        self.action_tensor[0, 0] = 0  # move_amount = STOP
        self.action_tensor[0, 1] = 0  # move_angle = FORWARD
        self.action_tensor[0, 2] = rotation


def run_controlled_sequence(mgr, trajectory_file: str, num_steps: int = 25):
    """
    Run a controlled rotation sequence using Manager's trajectory logging.

    Args:
        mgr: SimManager instance
        trajectory_file: Path to write trajectory CSV
        num_steps: Number of simulation steps to run
    """
    controller = AgentController(mgr)

    # Reset to clean start
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[0] = 1

    # Enable Manager's trajectory logging (automatically logs initial state)
    mgr.enable_trajectory_logging(0, 0, trajectory_file)

    # Execute controlled rotation sequence
    for step in range(num_steps):
        controller.reset_actions()

        if step < 10:
            # Steps 0-9: Rotate left slowly
            controller.rotate_only(madrona_escape_room.action.rotate.SLOW_LEFT)
        elif step < 15:
            # Steps 10-14: Stop rotation
            controller.rotate_only(madrona_escape_room.action.rotate.NONE)
        else:
            # Steps 15+: Rotate right slowly
            controller.rotate_only(madrona_escape_room.action.rotate.SLOW_RIGHT)

        mgr.step()
        reset_tensor[0] = 0  # Clear reset flag after first step

    mgr.disable_trajectory_logging()


def run_replay_sequence(mgr, trajectory_file: str, num_steps: int = 25):
    """
    Run replay sequence using Manager's trajectory logging.

    Args:
        mgr: SimManager instance in replay mode
        trajectory_file: Path to write trajectory CSV
        num_steps: Number of simulation steps to run
    """
    # Enable Manager's trajectory logging (automatically logs initial state)
    mgr.enable_trajectory_logging(0, 0, trajectory_file)

    # Execute replay sequence
    for step in range(num_steps):
        replay_complete = mgr.replay_step()
        if not replay_complete:
            mgr.step()
        else:
            break

    mgr.disable_trajectory_logging()


def compare_trajectory_files(file1: str, file2: str) -> bool:
    """
    Compare two trajectory CSV files line by line.

    Args:
        file1, file2: Paths to trajectory CSV files

    Returns:
        True if files are identical, False otherwise
    """
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"❌ Missing trajectory files: {file1} or {file2}")
        return False

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print(f"❌ Different line counts: {len(lines1)} vs {len(lines2)}")
        return False

    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1.strip() != line2.strip():
            print(f"❌ Line {i+1} differs:")
            print(f"  File1: {line1.strip()}")
            print(f"  File2: {line2.strip()}")
            return False

    print(f"✅ Trajectory files match ({len(lines1)} lines)")
    return True


def test_recording_vs_replay_consistency(cpu_manager):
    """Test that recording and replay produce identical trajectories"""
    print("\n" + "=" * 60)
    print("TEST: Recording vs Replay Trajectory Consistency")
    print("=" * 60)

    mgr = cpu_manager

    with tempfile.TemporaryDirectory() as tmpdir:
        recording_path = os.path.join(tmpdir, "consistency_test.rec")
        recording_traj = os.path.join(tmpdir, "recording_trajectory.csv")
        replay_traj = os.path.join(tmpdir, "replay_trajectory.csv")

        # Phase 1: Record the simulation
        print("Recording simulation...")
        mgr.start_recording(recording_path)
        run_controlled_sequence(mgr, recording_traj, num_steps=25)
        mgr.stop_recording()

        # Phase 2: Replay the simulation
        print("Creating replay manager...")
        replay_mgr = madrona_escape_room.SimManager.from_replay(
            recording_path, madrona_escape_room.ExecMode.CPU
        )

        print("Running replay...")
        run_replay_sequence(replay_mgr, replay_traj, num_steps=25)

        # Phase 3: Compare trajectory files
        print("Comparing trajectories...")
        is_consistent = compare_trajectory_files(recording_traj, replay_traj)

        if is_consistent:
            print("✅ Recording and replay trajectories are identical")
        else:
            print("❌ Recording and replay trajectories differ")
            # Print first few lines of each for debugging
            print("\nFirst 5 lines of recording trajectory:")
            with open(recording_traj, "r") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(f"  {i+1}: {line.strip()}")

            print("\nFirst 5 lines of replay trajectory:")
            with open(replay_traj, "r") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(f"  {i+1}: {line.strip()}")

        assert is_consistent, "Recording and replay trajectories should be identical"


def test_multiple_replay_consistency(cpu_manager):
    """Test that multiple replays of the same recording produce identical results"""
    print("\n" + "=" * 60)
    print("TEST: Multiple Replay Consistency")
    print("=" * 60)

    mgr = cpu_manager

    with tempfile.TemporaryDirectory() as tmpdir:
        recording_path = os.path.join(tmpdir, "multi_replay_test.rec")
        traj1 = os.path.join(tmpdir, "replay1_trajectory.csv")
        traj2 = os.path.join(tmpdir, "replay2_trajectory.csv")

        # Record the simulation
        print("Recording simulation...")
        mgr.start_recording(recording_path)
        run_controlled_sequence(mgr, os.path.join(tmpdir, "original_trajectory.csv"), num_steps=15)
        mgr.stop_recording()

        # First replay
        print("Running first replay...")
        replay_mgr1 = madrona_escape_room.SimManager.from_replay(
            recording_path, madrona_escape_room.ExecMode.CPU
        )
        run_replay_sequence(replay_mgr1, traj1, num_steps=15)

        # Second replay
        print("Running second replay...")
        replay_mgr2 = madrona_escape_room.SimManager.from_replay(
            recording_path, madrona_escape_room.ExecMode.CPU
        )
        run_replay_sequence(replay_mgr2, traj2, num_steps=15)

        # Compare the two replays
        print("Comparing replay trajectories...")
        is_consistent = compare_trajectory_files(traj1, traj2)

        if is_consistent:
            print("✅ Multiple replays produce identical trajectories")
        else:
            print("❌ Multiple replays produce different trajectories")

        assert is_consistent, "Multiple replays should produce identical trajectories"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))

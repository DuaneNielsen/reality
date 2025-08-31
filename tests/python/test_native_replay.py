#!/usr/bin/env python3
"""
Test native C++ replay functionality through Python bindings.
Tests the replay methods added to SimManager class.
"""

import os
import tempfile
import warnings

import pytest


@pytest.fixture(autouse=True)
def suppress_replay_warnings():
    """Suppress UserWarnings about replay loading for these legacy tests"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Loading replay into existing manager.*", category=UserWarning
        )
        yield


def create_test_recording(mgr, num_steps=5, seed=42):
    """Helper function to create a test recording file"""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    # Create recording
    mgr.start_recording(recording_path, seed=seed)

    action_tensor = mgr.action_tensor().to_torch()

    for step in range(num_steps):
        # Set predictable actions
        action_tensor.fill_(0)
        action_tensor[:, 0] = (step % 3) + 1  # move_amount cycles 1,2,3
        action_tensor[:, 1] = step % 8  # move_angle cycles through directions
        action_tensor[:, 2] = 2  # rotate = NONE

        mgr.step()

    mgr.stop_recording()
    return recording_path


def test_replay_lifecycle(cpu_manager):
    """Test basic replay load/unload cycle"""
    mgr = cpu_manager

    # Create a test recording
    recording_path = create_test_recording(mgr, num_steps=3)

    try:
        # Initially should not have replay
        assert not mgr.has_replay()

        # Test new interface alongside old
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)
        assert replay_mgr.has_replay()

        # Also test old interface for compatibility
        mgr.load_replay(recording_path)
        assert mgr.has_replay()

        # Get step counts
        current, total = mgr.get_replay_step_count()
        assert current == 0  # Haven't started replaying yet
        assert total == 3  # Should match our recording

        print(f"Loaded replay: {current}/{total} steps")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_step_through(cpu_manager):
    """Test stepping through a replay"""
    mgr = cpu_manager

    # Create a test recording
    num_steps = 5
    recording_path = create_test_recording(mgr, num_steps=num_steps)

    try:
        # Load replay
        mgr.load_replay(recording_path)

        # Step through the replay
        steps_taken = 0
        while True:
            current, total = mgr.get_replay_step_count()
            assert current == steps_taken
            assert total == num_steps

            # Execute replay step (sets actions for this step)
            finished = mgr.replay_step()
            steps_taken += 1

            if finished:
                break

            # Run the actual simulation step with the replay actions
            mgr.step()

            # Should not be finished until we've done all steps
            assert steps_taken <= num_steps

        # Should have taken exactly num_steps steps
        assert steps_taken == num_steps

        # Final step count check
        current, total = mgr.get_replay_step_count()
        assert current == num_steps
        assert total == num_steps

        print(f"Successfully stepped through {steps_taken} replay steps")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_beyond_end(cpu_manager):
    """Test behavior when stepping beyond replay end"""
    mgr = cpu_manager

    # Create a short recording
    recording_path = create_test_recording(mgr, num_steps=2)

    try:
        mgr.load_replay(recording_path)

        # Step through entire replay
        finished1 = mgr.replay_step()  # Step 1
        assert not finished1
        mgr.step()  # Run simulation with step 1 actions

        finished2 = mgr.replay_step()  # Step 2
        assert finished2  # Should finish on step 2 (just consumed last action)

        # Try stepping beyond end
        finished3 = mgr.replay_step()  # Beyond end
        assert finished3  # Should still report finished

        current, total = mgr.get_replay_step_count()
        assert current == 2
        assert total == 2

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_error_handling(cpu_manager):
    """Test error conditions in replay"""
    mgr = cpu_manager

    # Test loading non-existent file
    with pytest.raises(RuntimeError):
        mgr.load_replay("/path/that/does/not/exist.bin")

    # Test loading invalid file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # Write some invalid data
        f.write(b"not a valid replay file")
        invalid_path = f.name

    try:
        with pytest.raises(RuntimeError):
            mgr.load_replay(invalid_path)
    finally:
        os.unlink(invalid_path)

    # Test replay operations without loaded replay
    assert not mgr.has_replay()

    # These should handle gracefully or return appropriate values
    current, total = mgr.get_replay_step_count()
    # Exact behavior depends on implementation, but should not crash

    _finished = mgr.replay_step()
    # Should indicate no replay available (likely True = finished)


def test_replay_multiple_loads(cpu_manager):
    """Test loading multiple replay files"""
    mgr = cpu_manager

    # Create two different recordings
    recording1 = create_test_recording(mgr, num_steps=3, seed=123)
    recording2 = create_test_recording(mgr, num_steps=7, seed=456)

    try:
        # Load first replay
        mgr.load_replay(recording1)
        current1, total1 = mgr.get_replay_step_count()
        assert total1 == 3

        # Load second replay (should replace first)
        mgr.load_replay(recording2)
        current2, total2 = mgr.get_replay_step_count()
        assert total2 == 7
        assert current2 == 0  # Should reset to beginning

    finally:
        for path in [recording1, recording2]:
            if os.path.exists(path):
                os.unlink(path)


def test_replay_with_different_worlds(cpu_manager):
    """Test replay with different world configurations"""
    mgr = cpu_manager

    # The manager was created with specific num_worlds
    action_tensor = mgr.action_tensor().to_torch()
    num_worlds = action_tensor.shape[0]

    # Create recording with current world count
    recording_path = create_test_recording(mgr, num_steps=4)

    try:
        mgr.load_replay(recording_path)

        # Should load successfully
        assert mgr.has_replay()

        current, total = mgr.get_replay_step_count()
        assert total == 4

        # Try a few replay steps
        finished1 = mgr.replay_step()
        assert not finished1
        mgr.step()  # Run simulation with replay step 1

        finished2 = mgr.replay_step()
        assert not finished2
        mgr.step()  # Run simulation with replay step 2

        print(f"Replay working with {num_worlds} worlds")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_state_consistency(cpu_manager):
    """Test that replay state remains consistent during simulation"""
    mgr = cpu_manager

    recording_path = create_test_recording(mgr, num_steps=3)

    try:
        mgr.load_replay(recording_path)

        # State should be consistent through various operations
        assert mgr.has_replay()

        # Access tensors while replay is loaded
        _action_tensor = mgr.action_tensor().to_torch()
        assert mgr.has_replay()

        _obs_tensor = mgr.self_observation_tensor().to_torch()
        assert mgr.has_replay()

        # Step replay
        finished = mgr.replay_step()
        assert mgr.has_replay()
        assert not finished
        mgr.step()  # Run simulation with replay actions

        # Check counts
        current, total = mgr.get_replay_step_count()
        assert current == 1
        assert total == 3
        assert mgr.has_replay()

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)

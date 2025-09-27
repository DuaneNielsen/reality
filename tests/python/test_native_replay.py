#!/usr/bin/env python3
"""
Test native C++ replay functionality through Python bindings.
Tests the replay methods added to SimManager class.
"""

import os
import tempfile
import warnings

import pytest
import torch


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
    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    # Create recording (seed parameter removed - uses manager's seed)
    mgr.start_recording(recording_path)

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

        # Test from_replay interface
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)
        assert replay_mgr.has_replay()

        # Get step counts
        current, total = replay_mgr.get_replay_step_count()
        assert current == 0  # Haven't started replaying yet
        assert total == 3  # Should match our recording

        print(f"Loaded replay: {current}/{total} steps")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_step_through(cpu_manager):
    """Test stepping through a replay"""
    mgr = cpu_manager

    # Create a test recording with enough steps to trigger checksum verification (>200)
    num_steps = 600
    recording_path = create_test_recording(mgr, num_steps=num_steps)

    try:
        # Load replay
        import madrona_escape_room as mer

        mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

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

        # Verify replay determinism using checksum verification
        assert (
            not mgr.has_checksum_failed()
        ), "Replay should be deterministic (no checksum failures)"

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_beyond_end(cpu_manager):
    """Test behavior when stepping beyond replay end"""
    mgr = cpu_manager

    # Create a short recording
    recording_path = create_test_recording(mgr, num_steps=2)

    try:
        import madrona_escape_room as mer

        mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

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

        # Note: Only 1 step was executed, so checksum verification (every 200 steps) won't trigger

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_error_handling(cpu_manager):
    """Test error conditions in replay"""
    mgr = cpu_manager

    # Test loading non-existent file
    import madrona_escape_room as mer

    with pytest.raises(RuntimeError):
        mer.SimManager.from_replay("/path/that/does/not/exist.rec", mer.ExecMode.CPU)

    # Test loading invalid file
    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        # Write some invalid data
        f.write(b"not a valid replay file")
        invalid_path = f.name

    try:
        with pytest.raises(RuntimeError):
            mer.SimManager.from_replay(invalid_path, mer.ExecMode.CPU)
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

    # Create two different recordings using separate fresh managers
    # (since recording requires fresh simulation state)
    import madrona_escape_room as mer

    # Create first recording with a fresh manager
    temp_mgr1 = mer.SimManager(
        exec_mode=mer.ExecMode.CPU, gpu_id=0, num_worlds=4, rand_seed=123, auto_reset=True
    )
    recording1 = create_test_recording(temp_mgr1, num_steps=3, seed=123)

    # Create second recording with another fresh manager
    temp_mgr2 = mer.SimManager(
        exec_mode=mer.ExecMode.CPU, gpu_id=0, num_worlds=4, rand_seed=456, auto_reset=True
    )
    recording2 = create_test_recording(temp_mgr2, num_steps=7, seed=456)

    try:
        # Create replay manager for first recording
        mgr1 = mer.SimManager.from_replay(recording1, mer.ExecMode.CPU)
        current1, total1 = mgr1.get_replay_step_count()
        assert total1 == 3

        # Create replay manager for second recording
        mgr2 = mer.SimManager.from_replay(recording2, mer.ExecMode.CPU)
        current2, total2 = mgr2.get_replay_step_count()
        assert total2 == 7
        assert current2 == 0  # Should start at beginning

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

    # Create recording with current world count (>200 steps for checksum verification)
    recording_path = create_test_recording(mgr, num_steps=600)

    try:
        import madrona_escape_room as mer

        mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

        # Should load successfully
        assert mgr.has_replay()

        current, total = mgr.get_replay_step_count()
        assert total == 600

        # Run full replay to test checksum verification with different world configurations
        step_count = 0
        while step_count < 600:
            replay_complete = mgr.replay_step()
            if not replay_complete:
                mgr.step()  # Execute simulation step with checksum verification
            step_count += 1
            if replay_complete:
                break

        print(f"Replay working with {num_worlds} worlds ({step_count} steps)")

        # Verify replay determinism using checksum verification
        assert (
            not mgr.has_checksum_failed()
        ), "Replay should be deterministic (no checksum failures)"

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_state_consistency(cpu_manager):
    """Test that replay state remains consistent during simulation"""
    mgr = cpu_manager

    recording_path = create_test_recording(mgr, num_steps=600)

    try:
        import madrona_escape_room as mer

        mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

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
        assert total == 600  # Updated to match the new recording length
        assert mgr.has_replay()

        # Note: Only 1 step was replayed, so checksum verification (every 200 steps) won't trigger

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/sim.md", "Step 2: Reset Agent Physics and Spawning")
def test_replay_spawn_position_determinism():
    """Test that random spawn positions are deterministic during replay"""
    import madrona_escape_room as mer

    # Create a level with random spawning enabled
    level = mer.create_default_level()
    level.spawn_random = True

    # Record with random spawning
    recording_mgr = mer.SimManager(
        exec_mode=mer.ExecMode.CPU,
        num_worlds=4,
        compiled_levels=[level],
        gpu_id=0,
        rand_seed=123,  # Fixed seed for recording
        auto_reset=True,
    )

    # Capture initial spawn positions from recording BEFORE any steps
    obs_tensor = recording_mgr.self_observation_tensor().to_torch()
    recording_positions = []
    for world_idx in range(4):
        pos = obs_tensor[world_idx, 0, :3].clone()  # [x, y, z] for agent 0
        recording_positions.append(pos)
        print(f"Recording - World {world_idx} spawn: {pos}")

    # Now create the recording (this will step the simulation)
    num_steps = 1
    recording_path = create_test_recording(recording_mgr, num_steps=num_steps, seed=123)

    try:
        # Now replay the same recording multiple times
        replay_positions_run1 = []
        replay_positions_run2 = []

        # First replay run
        replay_mgr1 = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

        obs1 = replay_mgr1.self_observation_tensor().to_torch()
        for world_idx in range(4):
            pos = obs1[world_idx, 0, :3].clone()
            replay_positions_run1.append(pos)
            print(f"Replay Run 1 - World {world_idx} spawn: {pos}")

        # Second replay run (separate manager instance)
        replay_mgr2 = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)
        obs2 = replay_mgr2.self_observation_tensor().to_torch()
        for world_idx in range(4):
            pos = obs2[world_idx, 0, :3].clone()
            replay_positions_run2.append(pos)
            print(f"Replay Run 2 - World {world_idx} spawn: {pos}")

        # Test: All replay runs should have identical spawn positions
        for world_idx in range(4):
            # Replay run 1 should match replay run 2
            assert torch.allclose(
                replay_positions_run1[world_idx], replay_positions_run2[world_idx], atol=0.001
            ), (
                f"World {world_idx}: Replay runs have different spawn positions: "
                f"{replay_positions_run1[world_idx]} vs {replay_positions_run2[world_idx]}"
            )

            # Both replay runs should match the original recording
            assert torch.allclose(
                recording_positions[world_idx], replay_positions_run1[world_idx], atol=0.001
            ), (
                f"World {world_idx}: Replay spawn doesn't match recording spawn: "
                f"Recording: {recording_positions[world_idx]}, "
                f"Replay: {replay_positions_run1[world_idx]}"
            )
            print(f"✓ World {world_idx}: Spawn positions are deterministic")

        print("Spawn position determinism test completed")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/sim.md", "Step 2: Reset Agent Physics and Spawning")
def test_replay_spawn_position_bug_simple():
    """Simplified test that demonstrates the spawn position bug clearly"""
    import madrona_escape_room as mer

    # Create a level with random spawning enabled
    level = mer.create_default_level()
    level.spawn_random = True

    # Record a single step
    recording_mgr = mer.SimManager(
        exec_mode=mer.ExecMode.CPU,
        num_worlds=1,  # Use single world for clarity
        compiled_levels=[level],
        gpu_id=0,
        rand_seed=42,
        auto_reset=True,
    )

    # Get spawn position from recording BEFORE any steps are taken
    recording_obs = recording_mgr.self_observation_tensor().to_torch()
    recording_spawn = recording_obs[0, 0, :3].clone()  # World 0, agent 0, [x,y,z]

    recording_path = create_test_recording(recording_mgr, num_steps=1)

    try:
        # Now replay and get spawn position
        replay_mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)
        replay_obs = replay_mgr.self_observation_tensor().to_torch()
        replay_spawn = replay_obs[0, 0, :3].clone()

        print(f"Recording spawn: {recording_spawn}")
        print(f"Replay spawn:    {replay_spawn}")
        print(f"Difference:      {(recording_spawn - replay_spawn).abs()}")

        # This assertion should pass for deterministic replay but currently fails
        if not torch.allclose(recording_spawn, replay_spawn, atol=0.001):
            pytest.fail(
                f"SPAWN DETERMINISM BUG: Replay spawn position {replay_spawn} "
                f"differs from recording spawn position {recording_spawn}. "
                f"Max difference: {(recording_spawn - replay_spawn).abs().max():.6f}. "
                f"This indicates random spawn positions are not using deterministic RNG."
            )

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/sim.md", "Step 2: Reset Agent Physics and Spawning")
def test_spawn_positions_are_random_between_episodes():
    """Test that spawn_random=True produces different positions across episodes"""
    import madrona_escape_room as mer

    # Create a level with random spawning enabled
    level = mer.create_default_level()
    level.spawn_random = True

    # Create manager with auto_reset to test episode transitions
    mgr = mer.SimManager(
        exec_mode=mer.ExecMode.CPU,
        num_worlds=1,
        compiled_levels=[level],
        gpu_id=0,
        rand_seed=42,
        auto_reset=True,
    )

    # Collect spawn positions from multiple episodes
    positions = []

    # Get initial position
    obs = mgr.self_observation_tensor().to_torch()
    initial_pos = obs[0, 0, :3].clone()
    positions.append(initial_pos)

    # Trigger resets and collect positions
    action_tensor = mgr.action_tensor().to_torch()
    action_tensor.fill_(0)

    for i in range(4):  # Get 4 more positions after resets
        mgr.step()  # This should trigger reset due to auto_reset
        obs = mgr.self_observation_tensor().to_torch()
        pos = obs[0, 0, :3].clone()
        positions.append(pos)

    # Verify positions are different between episodes
    all_same = True
    for i in range(1, len(positions)):
        diff = (positions[0] - positions[i]).abs().max()
        if diff > 0.001:  # Threshold for "different enough"
            all_same = False
            break

    assert not all_same, (
        f"Random spawn positions are not varying between episodes. "
        f"All positions: {positions}. "
        f"With spawn_random=True, agents should spawn at different locations across episodes."
    )


def test_replay_large_scale_with_autoreset():
    """
    Test replay accuracy for large-scale simulations with auto-reset enabled.
    Tests 32 worlds running up to 2000 steps with auto-reset.
    This is a comprehensive test for replay determinism at scale.
    """
    import madrona_escape_room as mer

    # Use higher world count and step count to test scalability
    num_worlds = 32
    num_steps = 2000

    # Create a manager with auto-reset enabled
    recording_mgr = mer.SimManager(
        exec_mode=mer.ExecMode.CPU,
        gpu_id=0,
        num_worlds=num_worlds,
        rand_seed=7777,  # Fixed seed for deterministic recording
        auto_reset=True,  # Enable auto-reset for episode transitions
    )

    # Create recording with more complex action patterns
    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    print(f"Creating large-scale recording: {num_worlds} worlds, {num_steps} steps")
    recording_mgr.start_recording(recording_path)

    action_tensor = recording_mgr.action_tensor().to_torch()

    # Use more varied action patterns to test different scenarios
    for step in range(num_steps):
        action_tensor.fill_(0)

        # Cycle through different movement patterns for each world
        for world_idx in range(num_worlds):
            # Varied movement patterns
            action_tensor[world_idx, 0] = (step + world_idx) % 4  # move_amount 0-3
            action_tensor[world_idx, 1] = (step * 2 + world_idx) % 8  # move_angle 0-7
            action_tensor[world_idx, 2] = (step + world_idx * 2) % 5  # rotate 0-4

        recording_mgr.step()

        # Progress indicator for long recording
        if (step + 1) % 400 == 0:
            print(f"Recording progress: {step + 1}/{num_steps} steps")

    recording_mgr.stop_recording()
    print("Recording completed")

    try:
        # Load replay
        print("Loading replay for verification...")
        replay_mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

        # Verify basic replay properties
        assert replay_mgr.has_replay()
        current, total = replay_mgr.get_replay_step_count()
        assert current == 0
        assert total == num_steps

        print(f"Replay loaded: {total} steps, {num_worlds} worlds")

        # Step through the entire replay
        print("Stepping through replay...")
        steps_completed = 0

        while steps_completed < num_steps:
            # Get replay actions for this step
            finished = replay_mgr.replay_step()

            # Execute simulation step with replay actions
            replay_mgr.step()
            steps_completed += 1

            # Progress indicator
            if steps_completed % 400 == 0:
                print(f"Replay progress: {steps_completed}/{num_steps} steps")

                # Verify step count consistency
                current, total = replay_mgr.get_replay_step_count()
                assert (
                    current == steps_completed
                ), f"Step count mismatch: {current} != {steps_completed}"

            # Check if we should finish after this step
            if finished:
                break

        # Verify we completed all steps
        assert (
            steps_completed == num_steps
        ), f"Completed {steps_completed} steps, expected {num_steps}"

        # Verify final replay state
        current, total = replay_mgr.get_replay_step_count()
        assert current == num_steps
        assert total == num_steps

        # Critical: Verify replay determinism through checksum verification
        # With 2000 steps and checksum interval of 200, we should have had 10 checksum verifications
        if replay_mgr.has_checksum_failed():
            pytest.fail(
                "Large-scale replay failed checksum verification. "
                f"This indicates non-deterministic behavior with {num_worlds} worlds "
                f"over {num_steps} steps. "
                "Auto-reset configuration may not be properly stored/loaded, or there may be other "
                "sources of non-determinism in the simulation."
            )

        print(f"✓ Large-scale replay test passed: {num_worlds} worlds, {num_steps} steps")
        print("✓ Checksum verification passed - replay is deterministic")

        # The fact that checksum verification passed proves that the auto-reset setting
        # was properly stored and loaded from replay metadata. If it wasn't, we would have
        # seen checksum mismatches due to different episode reset behaviors.
        print("✓ Auto-reset setting properly preserved (verified by checksum validation)")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)
            print("Cleanup completed")

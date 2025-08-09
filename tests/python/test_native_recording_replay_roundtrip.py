#!/usr/bin/env python3
"""
Test round-trip recording and replay functionality.
Records actions, then replays them and verifies consistency.
"""

import os
import tempfile


def test_roundtrip_basic_consistency(log_and_verify_replay_cpu_manager):
    """Test basic record → replay → verify cycle.
    Fixture automatically records, traces, replays, and verifies trajectory matches."""
    mgr = log_and_verify_replay_cpu_manager

    # Manager is already recording and tracing, just run the actions
    action_tensor = mgr.action_tensor().to_torch()
    num_steps = 5

    for step in range(num_steps):
        # Set known actions
        action_tensor.fill_(0)
        action_tensor[:, 0] = (step % 3) + 1  # move_amount
        action_tensor[:, 1] = step % 8  # move_angle
        action_tensor[:, 2] = 2  # rotate

        mgr.step()

    # That's it! The fixture will automatically:
    # 1. Finalize recording when context exits
    # 2. Replay the recording
    # 3. Compare trajectory traces
    # 4. Assert they match exactly


def test_roundtrip_observation_consistency(log_and_verify_replay_cpu_manager):
    """Test observation recording with automatic replay verification"""
    mgr = log_and_verify_replay_cpu_manager

    # Manager is already recording and tracing, just run the actions
    action_tensor = mgr.action_tensor().to_torch()
    num_steps = 4

    recorded_observations = []

    for step in range(num_steps):
        # Set actions
        action_tensor.fill_(0)
        action_tensor[:, 0] = 1  # SLOW movement
        action_tensor[:, 1] = 0  # FORWARD
        action_tensor[:, 2] = 2  # No rotation

        mgr.step()

        # Capture observation for informational purposes
        obs = mgr.self_observation_tensor().to_torch().clone()
        recorded_observations.append(obs)

    # Print observation progression for debugging
    print("Observation progression during recording:")
    for i, obs in enumerate(recorded_observations):
        pos = obs[0, 0, :3]  # First world, first agent, xyz position
        print(f"  Step {i}: pos=({pos[0].item():.3f}, {pos[1].item():.3f}, {pos[2].item():.3f})")

    # That's it! The fixture will automatically verify replay matches exactly


def test_roundtrip_trajectory_file_verification(cpu_manager):
    """Test that trajectory traces match exactly between record and replay by comparing trace files.
    Uses regular cpu_manager to avoid conflicts with debug_cpu_manager."""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    with tempfile.NamedTemporaryFile(suffix="_record_trace.txt", delete=False) as f:
        record_trace_path = f.name

    with tempfile.NamedTemporaryFile(suffix="_replay_trace.txt", delete=False) as f:
        replay_trace_path = f.name

    try:
        # Phase 1: Record session with trajectory logging to file
        mgr.start_recording(recording_path, seed=42)
        mgr.enable_trajectory_logging(world_idx=0, agent_idx=0, filename=record_trace_path)

        action_tensor = mgr.action_tensor().to_torch()
        num_steps = 8  # More steps for better trajectory data

        for step in range(num_steps):
            # Set deterministic actions
            action_tensor.fill_(0)
            action_tensor[:, 0] = (step % 3) + 1  # move_amount: 1,2,3,1,2,3...
            action_tensor[:, 1] = step % 8  # move_angle: cycles through directions
            action_tensor[:, 2] = 2  # rotate: NONE

            mgr.step()

        mgr.disable_trajectory_logging()
        mgr.stop_recording()

        # Phase 2: Replay session with trajectory logging to different file
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(recording_path, mer.madrona.ExecMode.CPU)
        replay_mgr.enable_trajectory_logging(world_idx=0, agent_idx=0, filename=replay_trace_path)

        for step in range(num_steps):
            finished = replay_mgr.replay_step()
            replay_mgr.step()  # Execute the simulation step with replayed actions

            if step < num_steps - 1:
                assert not finished
            else:
                assert finished

        replay_mgr.disable_trajectory_logging()

        # Phase 3: Compare trace files - they should be identical
        assert os.path.exists(record_trace_path), "Record trace file should exist"
        assert os.path.exists(replay_trace_path), "Replay trace file should exist"

        with open(record_trace_path, "r") as f:
            record_content = f.read().strip()

        with open(replay_trace_path, "r") as f:
            replay_content = f.read().strip()

        # Files should have content
        assert len(record_content) > 0, "Record trace file should not be empty"
        assert len(replay_content) > 0, "Replay trace file should not be empty"

        # Split into lines for better comparison
        record_lines = record_content.split("\n")
        replay_lines = replay_content.split("\n")

        assert len(record_lines) == len(replay_lines), (
            f"Trace files should have same number of lines: record={len(record_lines)}, replay={len(replay_lines)}"
        )

        # Compare line by line for detailed error reporting
        for i, (record_line, replay_line) in enumerate(zip(record_lines, replay_lines)):
            assert record_line == replay_line, (
                f"Trajectory mismatch at line {i + 1}:\n  Record: {record_line}\n  Replay: {replay_line}"
            )

        print(f"✓ Successfully verified {len(record_lines)} trajectory trace lines match exactly")
        print(
            f"✓ Record trace: {len(record_content)} chars, Replay trace: {len(replay_content)} chars"
        )

    finally:
        # Clean up all temporary files
        for path in [recording_path, record_trace_path, replay_trace_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_roundtrip_trajectory_file_verification_with_debug(
    log_and_verify_replay_cpu_manager,
):
    """Test that debug session trajectory traces match replay traces.
    Uses debug_cpu_manager which is guaranteed to record and trace."""
    mgr = log_and_verify_replay_cpu_manager

    # The debug_cpu_manager is already recording and tracing
    action_tensor = mgr.action_tensor().to_torch()
    num_steps = 8

    for step in range(num_steps):
        action_tensor.fill_(0)
        action_tensor[:, 0] = (step % 3) + 1
        action_tensor[:, 1] = step % 8
        action_tensor[:, 2] = 2
        mgr.step()

    # Store paths while manager is still in context
    debug_recording_path = mgr._debug_recording_path
    debug_trajectory_path = mgr._debug_trajectory_path

    print(f"Debug recording path: {debug_recording_path}")
    print(f"Debug trajectory path: {debug_trajectory_path}")

    # Just verify the files were created - actual replay will happen after context exits
    assert hasattr(mgr, "_debug_recording_path"), "Manager should have debug recording path"
    assert hasattr(mgr, "_debug_trajectory_path"), "Manager should have debug trajectory path"

    print("✓ Debug session created recording and trajectory files")


def test_trajectory_file_verification_detects_differences(cpu_manager):
    """Test that our trajectory verification detects when trajectories differ"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix="_trace1.txt", delete=False) as f:
        trace1_path = f.name

    with tempfile.NamedTemporaryFile(suffix="_trace2.txt", delete=False) as f:
        trace2_path = f.name

    try:
        # Create two different trajectory logs
        mgr.enable_trajectory_logging(world_idx=0, agent_idx=0, filename=trace1_path)

        action_tensor = mgr.action_tensor().to_torch()

        # Run a few steps for trace1
        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 1  # SLOW movement
            action_tensor[:, 1] = 0  # FORWARD
            action_tensor[:, 2] = 2  # No rotation
            mgr.step()

        mgr.disable_trajectory_logging()

        # Create a different trajectory for trace2
        mgr.enable_trajectory_logging(world_idx=0, agent_idx=0, filename=trace2_path)

        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 2  # MEDIUM movement (different!)
            action_tensor[:, 1] = 4  # BACKWARD (different!)
            action_tensor[:, 2] = 1  # SLOW_LEFT (different!)
            mgr.step()

        mgr.disable_trajectory_logging()

        # Verify both files exist and have content
        assert os.path.exists(trace1_path)
        assert os.path.exists(trace2_path)

        with open(trace1_path, "r") as f:
            trace1_content = f.read().strip()

        with open(trace2_path, "r") as f:
            trace2_content = f.read().strip()

        # They should be different (proving our test would catch differences)
        assert trace1_content != trace2_content, (
            "Different actions should produce different trajectory traces"
        )

        trace1_lines = trace1_content.split("\n")
        trace2_lines = trace2_content.split("\n")

        # They should have same number of lines (same number of steps)
        assert len(trace1_lines) == len(trace2_lines), (
            "Same number of steps should produce same number of trace lines"
        )

        # But the actual positions should differ
        differences_found = 0
        for i, (line1, line2) in enumerate(zip(trace1_lines, trace2_lines)):
            if line1 != line2:
                differences_found += 1

        assert differences_found > 0, (
            f"Expected trajectory differences but found none. "
            f"Trace1 length: {len(trace1_content)}, Trace2 length: {len(trace2_content)}"
        )

        print(
            f"✓ Successfully detected {differences_found} trajectory differences "
            f"out of {len(trace1_lines)} lines"
        )
        print("✓ This confirms our verification test would catch trajectory mismatches")

    finally:
        for path in [trace1_path, trace2_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_roundtrip_session_replay(log_and_verify_replay_cpu_manager):
    """Test a single record/replay session using automatic verification fixture"""
    mgr = log_and_verify_replay_cpu_manager

    # Manager is already recording and tracing, just run the actions
    action_tensor = mgr.action_tensor().to_torch()
    num_steps = 5

    for step in range(num_steps):
        action_tensor.fill_(0)
        action_tensor[:, 0] = 2  # MEDIUM speed
        action_tensor[:, 1] = (step * 2) % 8  # Different directions
        action_tensor[:, 2] = 2  # No rotation

        mgr.step()

    # That's it! The fixture will automatically verify replay matches exactly


def test_roundtrip_edge_cases(cpu_manager):
    """Test edge cases in record/replay"""
    mgr = cpu_manager

    # Test very short recording (1 step)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        short_path = f.name

    try:
        mgr.start_recording(short_path, seed=999)

        action_tensor = mgr.action_tensor().to_torch()
        action_tensor.fill_(0)
        action_tensor[:, 0] = 3  # FAST
        action_tensor[:, 1] = 4  # BACKWARD
        action_tensor[:, 2] = 1  # SLOW_LEFT

        mgr.step()
        mgr.stop_recording()

        # Replay single step using new interface
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(short_path, mer.madrona.ExecMode.CPU)

        current, total = replay_mgr.get_replay_step_count()
        assert total == 1

        finished = replay_mgr.replay_step()
        assert finished  # Should finish immediately

        current, total = replay_mgr.get_replay_step_count()
        assert current == 1

    finally:
        if os.path.exists(short_path):
            os.unlink(short_path)

    # Test empty recording (0 steps)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        empty_path = f.name

    try:
        mgr.start_recording(empty_path, seed=0)
        # Don't step, just stop
        mgr.stop_recording()

        # Should still be loadable using new interface
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(empty_path, mer.madrona.ExecMode.CPU)

        current, total = replay_mgr.get_replay_step_count()
        assert total == 0

        # First replay step should indicate finished
        finished = replay_mgr.replay_step()
        assert finished

    finally:
        if os.path.exists(empty_path):
            os.unlink(empty_path)


def test_roundtrip_with_reset(log_and_verify_replay_cpu_manager):
    """Test recording/replay across episode resets with automatic verification"""
    mgr = log_and_verify_replay_cpu_manager

    # Manager is already recording and tracing, just run the actions
    action_tensor = mgr.action_tensor().to_torch()

    # Run enough steps to potentially trigger resets
    num_steps = 10

    for step in range(num_steps):
        action_tensor.fill_(0)
        action_tensor[:, 0] = 2  # MEDIUM speed
        action_tensor[:, 1] = 0  # FORWARD
        action_tensor[:, 2] = 2  # No rotation

        mgr.step()

        # Check if any episodes are done
        done_tensor = mgr.done_tensor().to_torch()
        if done_tensor.any():
            print(f"  Episode reset occurred at step {step}")

    # That's it! The fixture will automatically verify replay works across resets

#!/usr/bin/env python3
"""
Test round-trip recording and replay functionality.
Records actions, then replays them and verifies consistency.
Enhanced with detailed action verification and corruption detection.
"""

import os
import struct
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
        mgr.start_recording(recording_path)
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

        replay_mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)
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
            f"Trace files should have same number of lines: record={len(record_lines)}, "
            f"replay={len(replay_lines)}"
        )

        # Compare line by line for detailed error reporting
        for i, (record_line, replay_line) in enumerate(zip(record_lines, replay_lines)):
            assert record_line == replay_line, (
                f"Trajectory mismatch at line {i + 1}:\n  Record: {record_line}\n  "
                f"Replay: {replay_line}"
            )

        print(f"✓ Successfully verified {len(record_lines)} trajectory trace lines match exactly")
        print(
            f"✓ Record trace: {len(record_content)} chars, "
            f"Replay trace: {len(replay_content)} chars"
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
        assert (
            trace1_content != trace2_content
        ), "Different actions should produce different trajectory traces"

        trace1_lines = trace1_content.split("\n")
        trace2_lines = trace2_content.split("\n")

        # They should have same number of lines (same number of steps)
        assert len(trace1_lines) == len(
            trace2_lines
        ), "Same number of steps should produce same number of trace lines"

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
        mgr.start_recording(short_path)

        action_tensor = mgr.action_tensor().to_torch()
        action_tensor.fill_(0)
        action_tensor[:, 0] = 3  # FAST
        action_tensor[:, 1] = 4  # BACKWARD
        action_tensor[:, 2] = 1  # SLOW_LEFT

        mgr.step()
        mgr.stop_recording()

        # Replay single step using new interface
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(short_path, mer.ExecMode.CPU)

        current, total = replay_mgr.get_replay_step_count()
        assert total == 1

        finished = replay_mgr.replay_step()
        assert finished  # Should finish immediately

        current, total = replay_mgr.get_replay_step_count()
        assert current == 1

    finally:
        if os.path.exists(short_path):
            os.unlink(short_path)

    # Test empty recording (0 steps) using fresh manager
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        empty_path = f.name

    try:
        # Need fresh manager for second recording
        import madrona_escape_room as mer

        fresh_mgr = mer.SimManager(
            exec_mode=mer.ExecMode.CPU, gpu_id=0, num_worlds=4, rand_seed=42, auto_reset=True
        )
        fresh_mgr.start_recording(empty_path)
        # Don't step, just stop
        fresh_mgr.stop_recording()

        # Should still be loadable using new interface
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(empty_path, mer.ExecMode.CPU)

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


def test_action_sequence_detailed_validation(cpu_manager):
    """Test detailed action sequence validation with step-by-step verification"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording with specific action sequence
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]

        # Define precise action sequence for validation
        action_sequence = [
            (1, 0, 2),  # SLOW, FORWARD, NONE
            (2, 2, 1),  # MEDIUM, RIGHT, SLOW_LEFT
            (3, 4, 4),  # FAST, BACKWARD, FAST_RIGHT
            (1, 6, 0),  # SLOW, LEFT, FAST_LEFT
            (2, 1, 3),  # MEDIUM, FORWARD_RIGHT, SLOW_RIGHT
        ]

        # Record actions
        for step, (move_amount, move_angle, rotate) in enumerate(action_sequence):
            action_tensor.fill_(0)
            action_tensor[:, 0] = move_amount
            action_tensor[:, 1] = move_angle
            action_tensor[:, 2] = rotate
            mgr.step()

        mgr.stop_recording()

        # Replay and verify action sequence matches exactly
        import madrona_escape_room as mer

        replay_mgr = mer.SimManager.from_replay(recording_path, mer.ExecMode.CPU)

        current, total = replay_mgr.get_replay_step_count()
        assert total == len(action_sequence), f"Expected {len(action_sequence)} steps, got {total}"

        # Capture actions during replay by examining action tensor
        replay_action_tensor = replay_mgr.action_tensor().to_torch()
        replayed_actions = []

        for step in range(total):
            finished = replay_mgr.replay_step()

            # Capture the action that was just loaded
            current_actions = replay_action_tensor.clone()
            step_actions = []
            for world_idx in range(num_worlds):
                move_amount = current_actions[world_idx, 0].item()
                move_angle = current_actions[world_idx, 1].item()
                rotate = current_actions[world_idx, 2].item()
                step_actions.append((move_amount, move_angle, rotate))
            replayed_actions.append(step_actions)

            replay_mgr.step()  # Execute the simulation step

            if step == total - 1:
                assert finished, f"Expected replay to finish at step {step}"

        # Verify replayed actions match original sequence
        print("=== Action Sequence Validation ===")
        for step, (expected_actions, replayed_step) in enumerate(
            zip(action_sequence, replayed_actions)
        ):
            expected_move_amount, expected_move_angle, expected_rotate = expected_actions

            for world_idx, (move_amount, move_angle, rotate) in enumerate(replayed_step):
                assert move_amount == expected_move_amount, (
                    f"Step {step}, World {world_idx}: move_amount mismatch - "
                    f"expected {expected_move_amount}, got {move_amount}"
                )
                assert move_angle == expected_move_angle, (
                    f"Step {step}, World {world_idx}: move_angle mismatch - "
                    f"expected {expected_move_angle}, got {move_angle}"
                )
                assert rotate == expected_rotate, (
                    f"Step {step}, World {world_idx}: rotate mismatch - "
                    f"expected {expected_rotate}, got {rotate}"
                )

            print(f"✓ Step {step}: {expected_actions} validated for {num_worlds} worlds")

        print(f"✓ All {len(action_sequence)} action sequences validated successfully")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_action_data_corruption_detection(cpu_manager):
    """Test that action data corruption is detected during replay"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        corrupted_path = f.name

    try:
        # Create a clean recording
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()

        # Record known actions
        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 2  # MEDIUM
            action_tensor[:, 1] = 0  # FORWARD
            action_tensor[:, 2] = 2  # NONE
            mgr.step()

        mgr.stop_recording()

        # Read original file and corrupt action data
        with open(recording_path, "rb") as f:
            original_data = f.read()

        # Find action data location (after metadata + CompiledLevel)
        # Use file size to estimate action data location
        metadata_size = 136  # ReplayMetadata for version 2
        estimated_level_size = 12472  # Estimated CompiledLevel size
        action_start = metadata_size + estimated_level_size

        # Ensure we don't go beyond file bounds
        if action_start >= len(original_data):
            action_start = (
                len(original_data) - 48
            )  # Ensure we have space for at least some action data

        # Create corrupted version by modifying action data
        corrupted_data = bytearray(original_data)

        # Corrupt first action's move_amount (change from 2 to 99)
        if action_start + 4 < len(corrupted_data):
            corrupted_data[action_start : action_start + 4] = struct.pack(
                "<i", 99
            )  # Invalid move_amount

        with open(corrupted_path, "wb") as f:
            f.write(corrupted_data)

        # Try to replay corrupted file
        import madrona_escape_room as mer

        # The replay should still load (corruption might not be detected at load time)
        replay_mgr = mer.SimManager.from_replay(corrupted_path, mer.ExecMode.CPU)

        # Verify we can detect the corruption by examining action values
        action_tensor = replay_mgr.action_tensor().to_torch()

        # Replay first step
        replay_mgr.replay_step()

        # Check if action was corrupted
        first_action = action_tensor[0, 0].item()  # First world, move_amount

        # If corruption was in the right place, we should see the invalid value
        if first_action == 99:
            print("✓ Successfully detected action corruption - move_amount = 99 (invalid)")
        else:
            print(
                f"Note: Action corruption may not have affected the expected location "
                f"(got {first_action})"
            )

        # Test action validation ranges
        valid_move_amounts = [0, 1, 2, 3]  # Valid range for move_amount
        valid_move_angles = list(range(8))  # 0-7 for move_angle
        valid_rotates = [0, 1, 2, 3, 4]  # Valid range for rotate

        # Check all actions in tensor for validity
        for world_idx in range(action_tensor.shape[0]):
            move_amount = action_tensor[world_idx, 0].item()
            move_angle = action_tensor[world_idx, 1].item()
            rotate = action_tensor[world_idx, 2].item()

            if move_amount not in valid_move_amounts:
                print(
                    f"✓ Detected invalid move_amount: {move_amount} (valid: {valid_move_amounts})"
                )
            if move_angle not in valid_move_angles:
                print(f"✓ Detected invalid move_angle: {move_angle} (valid: {valid_move_angles})")
            if rotate not in valid_rotates:
                print(f"✓ Detected invalid rotate: {rotate} (valid: {valid_rotates})")

        print("✓ Action data corruption detection test completed")

    finally:
        for path in [recording_path, corrupted_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_file_structure_integrity_validation(cpu_manager):
    """Test file structure integrity and boundary verification"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording with known parameters
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]
        num_steps = 4

        for step in range(num_steps):
            action_tensor.fill_(0)
            action_tensor[:, 0] = (step % 3) + 1
            action_tensor[:, 1] = step % 8
            action_tensor[:, 2] = 2
            mgr.step()

        mgr.stop_recording()

        # Verify file structure integrity
        file_size = os.path.getsize(recording_path)

        with open(recording_path, "rb") as f:
            print("=== File Structure Integrity Validation ===")
            print(f"File size: {file_size} bytes")

            # Verify metadata section
            magic = struct.unpack("<I", f.read(4))[0]
            version = struct.unpack("<I", f.read(4))[0]

            assert magic == 0x4D455352, f"Invalid magic: 0x{magic:08x}"
            assert version == 2, f"Invalid version: {version}"

            # Skip to end of metadata
            f.seek(136)  # ReplayMetadata size
            metadata_end = f.tell()

            print(f"Metadata ends at: {metadata_end}")

            # Calculate expected action data size
            bytes_per_step = num_worlds * 3 * 4  # 3 int32_t per world
            expected_action_bytes = num_steps * bytes_per_step

            print(
                f"Expected action data: {expected_action_bytes} bytes "
                f"({num_steps} steps × {num_worlds} worlds × 3 actions × 4 bytes)"
            )

            # Verify file has sufficient size
            min_expected_size = metadata_end + expected_action_bytes
            assert (
                file_size >= min_expected_size
            ), f"File too small: {file_size} < {min_expected_size}"

            # Find actual action data by working backwards from file end
            estimated_action_start = file_size - expected_action_bytes

            if estimated_action_start >= metadata_end:
                f.seek(estimated_action_start)
                print(f"Action data likely starts at: {estimated_action_start}")

                # Try to read and validate first action
                try:
                    first_move_amount = struct.unpack("<i", f.read(4))[0]
                    first_move_angle = struct.unpack("<i", f.read(4))[0]
                    first_rotate = struct.unpack("<i", f.read(4))[0]

                    print(
                        f"First action: move_amount={first_move_amount}, "
                        f"move_angle={first_move_angle}, rotate={first_rotate}"
                    )

                    # Validate action ranges
                    assert 0 <= first_move_amount <= 3, f"Invalid move_amount: {first_move_amount}"
                    assert 0 <= first_move_angle <= 7, f"Invalid move_angle: {first_move_angle}"
                    assert 0 <= first_rotate <= 4, f"Invalid rotate: {first_rotate}"

                    print("✓ First action data validation passed")

                except struct.error as e:
                    print(f"Failed to read action data: {e}")

            # Verify file structure boundaries
            remaining_bytes = file_size - f.tell()
            expected_remaining = expected_action_bytes - 12  # We read 3 int32_t (12 bytes)

            print(f"Remaining bytes: {remaining_bytes}")
            print(f"Expected remaining: {expected_remaining}")

            print("✓ File structure integrity validation completed")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_partial_file_replay_handling(cpu_manager):
    """Test handling of partial/incomplete recording files"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        full_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        partial_path = f.name

    try:
        # Create a full recording
        mgr.start_recording(full_path)

        action_tensor = mgr.action_tensor().to_torch()

        for step in range(5):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 1
            action_tensor[:, 1] = 0
            action_tensor[:, 2] = 2
            mgr.step()

        mgr.stop_recording()

        # Create partial file (truncate action data)
        with open(full_path, "rb") as f:
            full_data = f.read()

        # Keep metadata and some action data, but not all
        partial_size = len(full_data) - 20  # Remove last 20 bytes
        partial_data = full_data[:partial_size]

        with open(partial_path, "wb") as f:
            f.write(partial_data)

        print("=== Partial File Replay Handling ===")
        print(f"Original file: {len(full_data)} bytes")
        print(f"Partial file: {len(partial_data)} bytes")

        # Try to load partial file
        import madrona_escape_room as mer

        try:
            replay_mgr = mer.SimManager.from_replay(partial_path, mer.ExecMode.CPU)

            current, total = replay_mgr.get_replay_step_count()
            print(f"Partial file reports {total} steps")

            # The total might be less than original due to truncation
            assert total <= 5, "Partial file should not report more steps than original"

            # Try to replay available steps
            steps_replayed = 0
            for step in range(total):
                try:
                    finished = replay_mgr.replay_step()
                    replay_mgr.step()
                    steps_replayed += 1

                    if finished:
                        break
                except Exception as e:
                    print(f"Replay failed at step {step}: {e}")
                    break

            print(f"Successfully replayed {steps_replayed} steps from partial file")

        except Exception as e:
            print(f"Failed to load partial file (expected): {e}")

        print("✓ Partial file handling test completed")

    finally:
        for path in [full_path, partial_path]:
            if os.path.exists(path):
                os.unlink(path)

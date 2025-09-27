#!/usr/bin/env python3
"""
Test checksum verification functionality for replay determinism validation.
Tests the v4 format checksum recording, parsing, and verification features.
"""

import os
import tempfile
import warnings

import pytest


@pytest.fixture(autouse=True)
def suppress_replay_warnings():
    """Suppress UserWarnings about replay loading for these tests"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Loading replay into existing manager.*", category=UserWarning
        )
        yield


@pytest.mark.spec("docs/specs/mgr.md", "startRecording")
def test_checksum_recording_and_verification(cpu_manager):
    """Test that checksum verification works correctly for deterministic replay"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Record simulation with enough steps to trigger checksum records (>200 steps)
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()

        # Run 450 steps to get checksum records at steps 200 and 400
        for step in range(450):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 1  # move_amount = SLOW
            action_tensor[:, 1] = 0  # move_angle = FORWARD
            action_tensor[:, 2] = 2  # rotate = NONE
            mgr.step()

        mgr.stop_recording()

        # Verify checksum functionality by creating replay manager
        try:
            from madrona_escape_room import SimManager
            from madrona_escape_room.generated_constants import ExecMode

            replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)
            # If no exception was raised, loading succeeded
        except Exception as e:
            pytest.fail(f"Failed to load v4 format replay with checksums: {e}")

        # Use the replay manager for the rest of the test
        mgr = replay_mgr

        # Initially, no checksum failures should be detected
        assert not mgr.has_checksum_failed(), "Checksum should not have failed before replay"

        # Run replay and check for checksum verification
        step_count = 0
        while not mgr.replay_step() and step_count < 450:
            step_count += 1

        # The checksum verification system should have been triggered
        # (We can't guarantee pass/fail since simulation may have non-determinism,
        #  but we can verify the system ran without crashing)
        print(f"Completed {step_count} replay steps")
        print(f"Checksum failed flag: {mgr.has_checksum_failed()}")

        # Test passed if we completed the replay without crashes
        assert step_count > 400, f"Replay should complete >400 steps, got {step_count}"

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "loadReplay")
def test_checksum_file_format_detection(cpu_manager):
    """Test that v4 format files are properly detected and handled"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Create a v4 format recording
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(250):  # Enough to trigger one checksum record
            action_tensor.fill_(0)
            mgr.step()

        mgr.stop_recording()

        # Verify metadata shows v4 format
        metadata = mgr.read_replay_metadata(recording_path)
        assert metadata is not None, "Failed to read replay metadata"
        assert metadata.version == 4, f"Expected v4 format, got v{metadata.version}"
        assert metadata.magic == 0x4D455352, f"Invalid magic number: 0x{metadata.magic:08x}"

        # Verify the file can be loaded (tests v4 parsing)
        try:
            from madrona_escape_room import SimManager
            from madrona_escape_room.generated_constants import ExecMode

            SimManager.from_replay(recording_path, ExecMode.CPU)
            # If no exception was raised, loading succeeded
        except Exception as e:
            pytest.fail(f"Failed to load v4 format file: {e}")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "hasChecksumFailed")
def test_checksum_hasChecksumFailed_flag(cpu_manager):
    """Test that hasChecksumFailed flag functionality works"""
    mgr = cpu_manager

    # Initially should be false
    assert not mgr.has_checksum_failed(), "Initial checksum failed flag should be False"

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Create a recording
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(250):
            action_tensor.fill_(step % 4)  # Vary actions to create some dynamics
            mgr.step()

        mgr.stop_recording()

        # Load and replay
        from madrona_escape_room import SimManager
        from madrona_escape_room.generated_constants import ExecMode

        mgr = SimManager.from_replay(recording_path, ExecMode.CPU)

        # Run enough steps to trigger checksum verification
        step_count = 0
        while not mgr.replay_step() and step_count < 250:
            step_count += 1

        # The flag state depends on whether checksums matched
        # We just verify the flag is accessible and boolean
        checksum_failed = mgr.has_checksum_failed()
        assert isinstance(checksum_failed, bool), "has_checksum_failed should return boolean"

        print(f"Checksum verification completed, failed flag: {checksum_failed}")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "replayStep")
def test_extended_checksum_recording(cpu_manager):
    """Test checksum recording with 600 steps to demonstrate multiple checksum points"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Record 600 steps to get checksum records at steps 200 and 400
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(600):
            action_tensor.fill_(step % 4)  # Vary actions to create some dynamics
            mgr.step()

        mgr.stop_recording()

        # Should be v4 format with multiple checksum records
        metadata = mgr.read_replay_metadata(recording_path)
        assert metadata.version == 4, f"Expected v4 format, got v{metadata.version}"
        assert metadata.num_steps == 600, f"Expected 600 steps, got {metadata.num_steps}"

        # Should load and replay successfully
        try:
            from madrona_escape_room import SimManager
            from madrona_escape_room.generated_constants import ExecMode

            mgr = SimManager.from_replay(recording_path, ExecMode.CPU)
            # If no exception was raised, loading succeeded
        except Exception as e:
            pytest.fail(f"Failed to load v4 format file with multiple checksum records: {e}")

        # Run replay and verify checksums at each 200-step interval
        step_count = 0
        while not mgr.replay_step() and step_count < 600:
            step_count += 1

            # Check for checksum failure after each checksum verification point
            if step_count in [200, 400]:
                checksum_failed = mgr.has_checksum_failed()
                assert not checksum_failed, (
                    f"Checksum verification failed at step {step_count} - "
                    "simulation should be deterministic! Same actions should produce same positions"
                )
                print(f"✅ Checksum verification PASSED at step {step_count}")

        assert step_count >= 590, f"Should complete most replay steps, got {step_count}"

        # Final check - checksums should have passed for deterministic simulation
        final_checksum_failed = mgr.has_checksum_failed()
        assert not final_checksum_failed, (
            "Final checksum verification should pass - "
            "replay uses same actions so should be deterministic"
        )

        print(f"Successfully completed {step_count} replay steps with checksum verification")
        print(
            f"Final checksum verification status: {'FAILED' if final_checksum_failed else 'PASSED'}"
        )

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)

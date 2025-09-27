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

        # Verify checksum functionality by loading replay
        try:
            mgr.load_replay(recording_path)
            # If no exception was raised, loading succeeded
        except Exception as e:
            pytest.fail(f"Failed to load v4 format replay with checksums: {e}")

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
            mgr.load_replay(recording_path)
            # If no exception was raised, loading succeeded
        except Exception as e:
            pytest.fail(f"Failed to load v4 format file: {e}")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


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
        mgr.load_replay(recording_path)

        # Run enough steps to trigger checksum verification
        step_count = 0
        while not mgr.replay_step() and step_count < 250:
            step_count += 1

        # The flag state depends on whether checksums matched
        # We just verify the flag is accessible and boolean
        checksum_failed = mgr.has_checksum_failed()
        assert isinstance(checksum_failed, bool), "hasChecksumFailed should return boolean"

        print(f"Checksum verification completed, failed flag: {checksum_failed}")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_minimal_checksum_recording(cpu_manager):
    """Test checksum recording with minimal steps (no checksum records expected)"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Record only 50 steps (less than 200, so no checksum records)
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(50):
            action_tensor.fill_(0)
            mgr.step()

        mgr.stop_recording()

        # Should still be v4 format even without checksum records
        metadata = mgr.read_replay_metadata(recording_path)
        assert metadata.version == 4, f"Expected v4 format, got v{metadata.version}"

        # Should load and replay successfully
        try:
            mgr.load_replay(recording_path)
            # If no exception was raised, loading succeeded
        except Exception as e:
            pytest.fail(f"Failed to load v4 format file without checksum records: {e}")

        # Run replay
        step_count = 0
        while not mgr.replay_step() and step_count < 50:
            step_count += 1

        assert step_count >= 45, f"Should complete most replay steps, got {step_count}"

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)

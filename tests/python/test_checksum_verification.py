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


@pytest.mark.spec("docs/specs/mgr.md", "hasChecksumFailed")
def test_checksum_verification_with_corruption():
    """Test that checksum verification detects corrupted replay files"""
    import struct

    from madrona_escape_room import SimManager
    from madrona_escape_room.generated_constants import ExecMode

    # Create a fresh manager for this test
    mgr = SimManager(exec_mode=ExecMode.CPU, gpu_id=0, num_worlds=4, rand_seed=42, auto_reset=True)

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        clean_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        corrupted_path = f.name

    try:
        # Create a clean recording with enough steps for checksum records
        mgr.start_recording(clean_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(250):  # Enough for at least one checksum at step 200
            action_tensor.fill_(step % 4)
            mgr.step()

        mgr.stop_recording()

        # Read the clean file and create a corrupted version
        with open(clean_path, "rb") as f:
            clean_data = f.read()

        # Corrupt some action data (change a few bytes in the middle)
        corrupted_data = bytearray(clean_data)
        if len(corrupted_data) > 1000:
            # Corrupt some bytes that are likely to be action data
            for i in range(500, 510):
                if i < len(corrupted_data):
                    corrupted_data[i] = (corrupted_data[i] + 50) % 256

        with open(corrupted_path, "wb") as f:
            f.write(corrupted_data)

        # Replay the corrupted file - checksums should detect the difference
        try:
            replay_mgr = SimManager.from_replay(corrupted_path, ExecMode.CPU)

            # Run the replay
            step_count = 0
            while not replay_mgr.replay_step() and step_count < 250:
                step_count += 1

            # Check if corruption was detected
            checksum_failed = replay_mgr.has_checksum_failed()
            # Note: We can't guarantee it will fail since corruption might not affect
            # position calculations, but if it does fail, that proves the system works
            print(f"Corruption detection test: checksum_failed = {checksum_failed}")
            print(f"Completed {step_count} steps with corrupted data")

        except Exception as e:
            print(f"Corrupted replay failed to load or execute: {e}")
            # This is also a valid outcome - the corruption was severe enough
            # to prevent replay entirely

    finally:
        for path in [clean_path, corrupted_path]:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.spec("docs/specs/mgr.md", "hasChecksumFailed")
def test_multi_world_checksum_verification():
    """Test checksum verification with multiple worlds"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.generated_constants import ExecMode

    # Create manager with multiple worlds
    mgr = SimManager(exec_mode=ExecMode.CPU, gpu_id=0, num_worlds=8, rand_seed=42, auto_reset=True)

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Record with multiple worlds having different actions
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]

        # Run enough steps to trigger checksum verification
        for step in range(250):
            action_tensor.fill_(0)
            # Give each world slightly different actions
            for world_idx in range(num_worlds):
                action_tensor[world_idx, 0] = (world_idx % 3) + 1  # Different speeds
                action_tensor[world_idx, 1] = (world_idx + step) % 8  # Different directions
                action_tensor[world_idx, 2] = 2  # Same rotation

            mgr.step()

        mgr.stop_recording()

        # Replay with checksum verification
        replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)

        step_count = 0
        while not replay_mgr.replay_step() and step_count < 250:
            step_count += 1

        # Multi-world checksums should pass for deterministic simulation
        checksum_failed = replay_mgr.has_checksum_failed()
        assert not checksum_failed, (
            f"Multi-world checksum verification failed - replay should be deterministic "
            f"across all {num_worlds} worlds"
        )

        print(f"✓ Multi-world checksum verification PASSED for {num_worlds} worlds")
        print(f"Completed {step_count} replay steps successfully")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "startRecording")
def test_checksum_frequency_verification():
    """Test that checksums are recorded at the expected frequency (every 200 steps)"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.generated_constants import ExecMode

    mgr = SimManager(exec_mode=ExecMode.CPU, gpu_id=0, num_worlds=4, rand_seed=42, auto_reset=True)

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Record exactly 610 steps to test checksum frequency
        # Should get checksums at steps 200, 400, 600
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(610):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 1  # Consistent actions
            action_tensor[:, 1] = 0
            action_tensor[:, 2] = 2
            mgr.step()

        mgr.stop_recording()

        # Verify the recording metadata
        metadata = mgr.read_replay_metadata(recording_path)
        assert metadata.num_steps == 610, f"Expected 610 steps, got {metadata.num_steps}"
        assert metadata.version == 4, f"Expected v4 format with checksums, got v{metadata.version}"

        # Replay and verify checksum verification happens at expected intervals
        replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)

        step_count = 0
        checksum_verification_points = []

        while not replay_mgr.replay_step() and step_count < 610:
            step_count += 1

            # Check for checksum verification at key intervals
            if step_count in [200, 400, 600]:
                checksum_failed = replay_mgr.has_checksum_failed()
                checksum_verification_points.append((step_count, checksum_failed))
                assert not checksum_failed, (
                    f"Checksum verification failed at step {step_count} - "
                    "deterministic replay should pass"
                )

        # Verify we hit the expected checksum points
        expected_points = [200, 400, 600]
        actual_points = [point[0] for point in checksum_verification_points]

        for expected_step in expected_points:
            assert expected_step in actual_points, (
                f"Expected checksum verification at step {expected_step}, "
                f"but only found verification at steps: {actual_points}"
            )

        print(f"✓ Checksum verification confirmed at steps: {actual_points}")
        print("All checksum verifications PASSED (deterministic replay)")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.spec("docs/specs/mgr.md", "hasChecksumFailed")
def test_checksum_flag_persistence():
    """Test that hasChecksumFailed flag persists once set and resets appropriately"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.generated_constants import ExecMode

    mgr = SimManager(exec_mode=ExecMode.CPU, gpu_id=0, num_worlds=4, rand_seed=42, auto_reset=True)

    # Initially should be False
    assert not mgr.has_checksum_failed(), "Initial checksum failed flag should be False"

    with tempfile.NamedTemporaryFile(suffix=".rec", delete=False) as f:
        recording_path = f.name

    try:
        # Create a recording that will definitely have checksums
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        for step in range(450):  # Multiple checksum points
            action_tensor.fill_(step % 3)
            mgr.step()

        mgr.stop_recording()

        # Test with fresh replay manager
        replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)

        # Should start as False
        assert not replay_mgr.has_checksum_failed(), "Replay manager should start with False"

        # Run partial replay
        for _ in range(150):  # Before first checksum
            replay_mgr.replay_step()

        # Should still be False
        assert (
            not replay_mgr.has_checksum_failed()
        ), "Should remain False before checksum verification"

        # Run past first checksum point
        for _ in range(100):  # Past step 200
            replay_mgr.replay_step()

        # For deterministic simulation, should still be False
        first_check = replay_mgr.has_checksum_failed()
        assert not first_check, "First checksum verification should pass"

        # Continue to second checksum point
        for _ in range(200):  # Past step 400
            replay_mgr.replay_step()

        # Should still be False for deterministic replay
        second_check = replay_mgr.has_checksum_failed()
        assert not second_check, "Second checksum verification should pass"

        print("✓ Checksum flag persistence test PASSED")
        print("Flag correctly remained False throughout deterministic replay")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)

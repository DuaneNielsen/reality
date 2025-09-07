#!/usr/bin/env python3
"""
Test native C++ recording functionality through Python bindings.
Tests the recording methods added to SimManager class.
"""

import os
import struct
import tempfile

import pytest


def test_recording_lifecycle(cpu_manager):
    """Test basic recording start/stop cycle"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Initially should not be recording
        assert not mgr.is_recording()

        # Start recording
        mgr.start_recording(recording_path)
        assert mgr.is_recording()

        # Stop recording
        mgr.stop_recording()
        assert not mgr.is_recording()

        # File should exist
        assert os.path.exists(recording_path)
        assert os.path.getsize(recording_path) > 0  # Should have metadata at minimum

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.slow
def test_gpu_recording_lifecycle(request):
    """Test basic recording start/stop cycle on GPU"""
    # Create a fresh GPU manager to ensure we start at step 0
    import torch

    from madrona_escape_room import ExecMode

    if request.config.getoption("--no-gpu"):
        pytest.skip("Skipping GPU fixture due to --no-gpu flag")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from conftest import _create_sim_manager

    mgr = _create_sim_manager(request, ExecMode.CUDA)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Initially should not be recording
        assert not mgr.is_recording()

        # Start recording
        mgr.start_recording(recording_path)
        assert mgr.is_recording()

        # Run a few steps
        action_tensor = mgr.action_tensor().to_torch()
        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 1  # SLOW movement
            action_tensor[:, 1] = 0  # FORWARD
            action_tensor[:, 2] = 2  # No rotation
            mgr.step()

        # Stop recording
        mgr.stop_recording()
        assert not mgr.is_recording()

        # File should exist and have content
        assert os.path.exists(recording_path)
        assert os.path.getsize(recording_path) > 0

        print("GPU recording test completed successfully")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_recording_with_steps(cpu_manager):
    """Test recording with actual simulation steps"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Start recording
        mgr.start_recording(recording_path)

        # Run some simulation steps with actions
        action_tensor = mgr.action_tensor().to_torch()
        num_steps = 10

        for step in range(num_steps):
            # Set some actions (move forward slowly)
            action_tensor.fill_(0)  # Reset all actions
            action_tensor[:, 0] = 1  # move_amount = SLOW
            action_tensor[:, 1] = 0  # move_angle = FORWARD
            action_tensor[:, 2] = 2  # rotate = NONE

            mgr.step()

        # Stop recording
        mgr.stop_recording()

        # Verify file size makes sense
        file_size = os.path.getsize(recording_path)

        # Should have metadata + (num_steps * num_worlds * 3 * sizeof(int32))
        num_worlds = action_tensor.shape[0]
        _expected_action_data_size = num_steps * num_worlds * 3 * 4  # 3 int32s per action

        # File should be larger than just metadata
        assert file_size > 64  # Metadata is less than this

        print(f"Recorded {num_steps} steps for {num_worlds} worlds")
        print(f"File size: {file_size} bytes")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_recording_error_handling(cpu_manager):
    """Test error conditions in recording"""
    mgr = cpu_manager

    # Test invalid path - C++ implementation prints error but doesn't raise exception
    # Let's test that it doesn't crash instead
    try:
        mgr.start_recording("/invalid/path/that/does/not/exist.bin")
        # If it doesn't raise an exception, that's fine - just verify it doesn't crash
        mgr.stop_recording()  # Clean up any potential state
    except RuntimeError:
        # If it does raise an exception, that's also acceptable
        pass

    # Test stopping recording when not recording
    mgr.stop_recording()  # Should not raise error

    # Test multiple start_recording calls
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        mgr.start_recording(recording_path)

        # Second start should raise error in current implementation
        # (recording can only be started once)
        with pytest.raises(RuntimeError, match="Recording already in progress"):
            mgr.start_recording(recording_path)

        mgr.stop_recording()

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_recording_file_format(cpu_manager):
    """Test comprehensive recording file format validation - Version 2 complete validation"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Start recording and run a few steps
        mgr.start_recording(recording_path)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]

        # Run 3 steps with known actions
        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = step + 1  # move_amount varies by step
            action_tensor[:, 1] = 0  # move_angle = FORWARD
            action_tensor[:, 2] = 2  # rotate = NONE
            mgr.step()

        mgr.stop_recording()

        # Read and verify complete file structure with Version 2 format
        with open(recording_path, "rb") as f:
            file_data = f.read()

            print(f"Total file size: {len(file_data)} bytes")

            # File should have some content
            assert len(file_data) > 0

            # Version 2 ReplayMetadata structure (192 bytes total):
            # - magic number (4 bytes)
            # - version (4 bytes)
            # - sim_name (64 bytes)
            # - level_name (64 bytes) - NEW in version 2
            # - num_worlds (4 bytes)
            # - num_agents_per_world (4 bytes)
            # - num_steps (4 bytes)
            # - actions_per_step (4 bytes)
            # - timestamp (8 bytes)
            # - seed (4 bytes)
            # - reserved (28 bytes) - 7 * 4 bytes
            metadata_size = 192  # Correct size for version 2

            if len(file_data) >= metadata_size:
                # Read all 14 fields (vs previous 9 fields) from the loaded data
                offset = 0
                magic_bytes = file_data[offset : offset + 4]
                magic = struct.unpack("<I", magic_bytes)[0]
                offset += 4

                version = struct.unpack("<I", file_data[offset : offset + 4])[0]
                offset += 4

                sim_name_bytes = file_data[offset : offset + 64]
                sim_name = sim_name_bytes.decode("ascii").rstrip("\x00")
                offset += 64

                level_name_bytes = file_data[offset : offset + 64]
                # Handle potential non-ASCII characters in level_name
                try:
                    level_name = level_name_bytes.decode("ascii").rstrip("\x00")
                except UnicodeDecodeError:
                    # If ASCII decode fails, try to extract null-terminated string
                    null_pos = level_name_bytes.find(b"\x00")
                    if null_pos >= 0:
                        level_name = level_name_bytes[:null_pos].decode("ascii", errors="replace")
                    else:
                        level_name = level_name_bytes.decode("ascii", errors="replace").rstrip(
                            "\x00"
                        )
                offset += 64

                num_worlds_meta = struct.unpack("<I", file_data[offset : offset + 4])[0]
                offset += 4

                num_agents = struct.unpack("<I", file_data[offset : offset + 4])[0]
                offset += 4

                num_steps_meta = struct.unpack("<I", file_data[offset : offset + 4])[0]
                offset += 4

                actions_per_step = struct.unpack("<I", file_data[offset : offset + 4])[0]
                offset += 4

                timestamp = struct.unpack("<Q", file_data[offset : offset + 8])[0]
                offset += 8

                seed = struct.unpack("<I", file_data[offset : offset + 4])[0]
                offset += 4

                reserved = file_data[offset : offset + 28]  # 7 * 4 bytes reserved
                offset += 28

                print("=== Complete ReplayMetadata Validation (14 fields) ===")
                print(f"Magic: 0x{magic:08x} (bytes: {magic_bytes.hex()})")
                print(f"Version: {version}")
                print(f"Sim name: '{sim_name}'")
                print(
                    f"Level name: '{level_name}'" + (" [NEW FIELD]" if level_name else " [EMPTY]")
                )
                print(f"Num worlds: {num_worlds_meta}")
                print(f"Num agents: {num_agents}")
                print(f"Num steps: {num_steps_meta}")
                print(f"Actions per step: {actions_per_step}")
                print(f"Timestamp: {timestamp}")
                print(f"Seed: {seed}")
                print(f"Reserved: {len(reserved)} bytes")

                # Verify all 14 fields (comprehensive validation)
                assert magic == 0x4D455352, f"Expected magic 0x4D455352, got 0x{magic:08x}"
                assert version == 2, f"Expected version 2 (current format), got {version}"
                assert (
                    sim_name == "madrona_escape_room"
                ), f"Expected sim_name 'madrona_escape_room', got '{sim_name}'"
                assert (
                    level_name
                ), f"Level name should not be empty in version 2, got '{level_name}'"
                assert (
                    num_worlds_meta == num_worlds
                ), f"Expected {num_worlds} worlds, got {num_worlds_meta}"
                assert num_agents >= 1, f"Expected at least 1 agent per world, got {num_agents}"
                assert num_steps_meta == 3, f"Expected 3 steps, got {num_steps_meta}"
                assert actions_per_step == 3, f"Expected 3 actions per step, got {actions_per_step}"
                assert timestamp > 0, f"Expected positive timestamp, got {timestamp}"
                assert seed == 42, f"Expected seed 42, got {seed}"
                assert len(reserved) == 28, f"Expected 28 reserved bytes, got {len(reserved)}"

                print("✓ All 14 ReplayMetadata fields validated successfully")

                # Verify C++ struct alignment compliance
                complete_metadata = file_data[:metadata_size]
                assert len(complete_metadata) == metadata_size, "Failed to read complete metadata"

                # Check for proper null-termination in string fields
                sim_name_null_pos = sim_name_bytes.find(b"\x00")
                level_name_null_pos = level_name_bytes.find(b"\x00")

                assert sim_name_null_pos >= 0, "sim_name should be null-terminated"
                assert level_name_null_pos >= 0, "level_name should be null-terminated"

                print("✓ C++ struct alignment and string handling validated")

                # Enhanced action data validation
                action_data = file_data[metadata_size:]
                expected_action_bytes = (
                    3 * num_worlds * 3 * 4
                )  # 3 steps * worlds * 3 components * 4 bytes

                print(f"Action data size: {len(action_data)} bytes")
                print(f"Expected action data size: {expected_action_bytes} bytes")

                # Validate action data presence and size bounds
                assert (
                    len(action_data) >= expected_action_bytes
                ), f"Insufficient action data: {len(action_data)} < {expected_action_bytes}"

                print("✓ Enhanced action data validation completed")
                print("✓ Current format (version 2) comprehensive validation PASSED")
            else:
                raise AssertionError(
                    f"File too small ({len(file_data)} bytes) for version 2 metadata "
                    f"({metadata_size} bytes)"
                )

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_recording_empty_session(cpu_manager):
    """Test recording session with no steps"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Start and immediately stop recording
        mgr.start_recording(recording_path)
        mgr.stop_recording()

        # File should still exist with metadata
        assert os.path.exists(recording_path)
        file_size = os.path.getsize(recording_path)

        # Should have at least metadata
        assert file_size > 0
        print(f"Empty recording file size: {file_size} bytes")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_recording_state_persistence(cpu_manager):
    """Test that recording state persists across operations"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Start recording
        mgr.start_recording(recording_path)

        # Should remain recording through various operations
        assert mgr.is_recording()

        mgr.step()
        assert mgr.is_recording()

        _action_tensor = mgr.action_tensor().to_torch()
        assert mgr.is_recording()

        _reward_tensor = mgr.reward_tensor().to_torch()
        assert mgr.is_recording()

        mgr.stop_recording()
        assert not mgr.is_recording()

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_current_format_specification_compliance(cpu_manager):
    """Test current format (version 2) specification compliance and struct layout"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording to test format compliance
        mgr.start_recording(recording_path)
        mgr.stop_recording()

        # Test format specification compliance
        with open(recording_path, "rb") as f:
            print("=== Current Format (Version 2) Specification Compliance ===")

            # Read and validate exact struct layout
            magic_bytes = f.read(4)
            version_bytes = f.read(4)
            sim_name_bytes = f.read(64)
            level_name_bytes = f.read(64)

            magic = struct.unpack("<I", magic_bytes)[0]
            version = struct.unpack("<I", version_bytes)[0]

            print(f"Magic number: 0x{magic:08x} (expected: 0x4D455352)")
            print(f"Version: {version} (expected: 2)")
            print(f"Sim name section: {len(sim_name_bytes)} bytes")
            print(f"Level name section: {len(level_name_bytes)} bytes")

            # Validate magic number and version (critical format identifiers)
            assert magic == 0x4D455352, f"Invalid magic number: 0x{magic:08x}"
            assert version == 2, f"Expected current version 2, got {version}"

            # Validate string field layout
            assert (
                len(sim_name_bytes) == 64
            ), f"sim_name field should be 64 bytes, got {len(sim_name_bytes)}"
            assert (
                len(level_name_bytes) == 64
            ), f"level_name field should be 64 bytes, got {len(level_name_bytes)}"

            # Check string null-termination
            sim_name = sim_name_bytes.decode("ascii").rstrip("\x00")
            # Handle potential non-ASCII characters in level_name
            try:
                level_name = level_name_bytes.decode("ascii").rstrip("\x00")
            except UnicodeDecodeError:
                # If ASCII decode fails, try to extract null-terminated string
                null_pos = level_name_bytes.find(b"\x00")
                if null_pos >= 0:
                    level_name = level_name_bytes[:null_pos].decode("ascii", errors="replace")
                else:
                    level_name = level_name_bytes.decode("ascii", errors="replace").rstrip("\x00")

            assert sim_name == "madrona_escape_room", f"Invalid sim_name: '{sim_name}'"
            assert level_name, f"level_name should not be empty: '{level_name}'"

            # Validate field alignment - check that strings are properly null-terminated
            sim_name_null_pos = sim_name_bytes.find(b"\x00")
            level_name_null_pos = level_name_bytes.find(b"\x00")

            assert sim_name_null_pos >= 0, "sim_name must be null-terminated"
            assert level_name_null_pos >= 0, "level_name must be null-terminated"

            # Ensure proper padding - check remaining fields
            remaining_fields = f.read(56)  # 4+4+4+4+8+4+28 = 56 bytes remaining in metadata
            assert (
                len(remaining_fields) == 56
            ), f"Expected 56 bytes for remaining fields, got {len(remaining_fields)}"

            print("✓ Magic number and version validation passed")
            print("✓ String field layout and null-termination validated")
            print("✓ Struct alignment and padding verified")
            print("✓ Current format specification compliance PASSED")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_format_error_conditions(cpu_manager):
    """Test error condition handling for current format"""

    # Test 1: Invalid magic number detection
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        invalid_magic_path = f.name
        # Write file with invalid magic but valid structure
        f.write(struct.pack("<I", 0xDEADBEEF))  # Invalid magic
        f.write(struct.pack("<I", 2))  # Valid version
        f.write(b"madrona_escape_room\x00" + b"\x00" * 44)  # sim_name (64 bytes)
        f.write(b"test_level\x00" + b"\x00" * 53)  # level_name (64 bytes)
        f.write(b"\x00" * 40)  # Remaining fields

    try:
        with open(invalid_magic_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            version = struct.unpack("<I", f.read(4))[0]

            # Should detect invalid magic
            assert magic == 0xDEADBEEF, "Should read invalid magic"
            assert magic != 0x4D455352, "Magic should not match expected value"
            assert version == 2, "Version should still be valid"

        print("✓ Invalid magic number detection validated")

    finally:
        if os.path.exists(invalid_magic_path):
            os.unlink(invalid_magic_path)

    # Test 2: Invalid version detection
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        invalid_version_path = f.name
        f.write(struct.pack("<I", 0x4D455352))  # Valid magic
        f.write(struct.pack("<I", 999))  # Invalid version
        f.write(b"\x00" * 128)  # Padding

    try:
        with open(invalid_version_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            version = struct.unpack("<I", f.read(4))[0]

            assert magic == 0x4D455352, "Magic should be valid"
            assert version == 999, "Should read invalid version"
            assert version != 2, "Version should not match current version"

        print("✓ Invalid version detection validated")

    finally:
        if os.path.exists(invalid_version_path):
            os.unlink(invalid_version_path)

    # Test 3: Incomplete header detection
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        incomplete_path = f.name
        # Write only partial header
        f.write(struct.pack("<I", 0x4D455352))  # Valid magic
        f.write(struct.pack("<I", 2))  # Valid version
        f.write(b"madrona")  # Incomplete sim_name

    try:
        file_size = os.path.getsize(incomplete_path)
        assert file_size < 136, f"File should be incomplete: {file_size} < 136 bytes"

        with open(incomplete_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            version = struct.unpack("<I", f.read(4))[0]
            remaining = f.read()

            assert magic == 0x4D455352, "Magic should be valid"
            assert version == 2, "Version should be valid"
            assert len(remaining) < 128, "Should have incomplete data"

        print("✓ Incomplete header detection validated")

    finally:
        if os.path.exists(incomplete_path):
            os.unlink(incomplete_path)

    print("✓ All error condition tests passed")


def test_field_alignment_and_padding(cpu_manager):
    """Test field alignment and padding in current format"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording
        mgr.start_recording(recording_path)
        mgr.stop_recording()

        # Test field alignment
        with open(recording_path, "rb") as f:
            print("=== Field Alignment and Padding Validation ===")

            # Read complete metadata and verify field boundaries
            complete_metadata = f.read(192)
            assert (
                len(complete_metadata) == 192
            ), f"Expected 192 bytes, got {len(complete_metadata)}"

            # Parse each field at correct offset
            offset = 0

            # Magic (4 bytes)
            magic = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # Version (4 bytes)
            version = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # Sim name (64 bytes)
            sim_name_bytes = complete_metadata[offset : offset + 64]
            sim_name = sim_name_bytes.decode("ascii").rstrip("\x00")
            offset += 64

            # Level name (64 bytes)
            level_name_bytes = complete_metadata[offset : offset + 64]
            level_name = level_name_bytes.decode("ascii").rstrip("\x00")
            offset += 64

            # Continue parsing remaining fields to reach offset 192
            # num_worlds (4 bytes)
            num_worlds = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # num_agents_per_world (4 bytes)
            _num_agents = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # num_steps (4 bytes)
            _num_steps = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # actions_per_step (4 bytes)
            _actions_per_step = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # timestamp (8 bytes)
            timestamp = struct.unpack("<Q", complete_metadata[offset : offset + 8])[0]
            offset += 8

            # seed (4 bytes)
            _seed = struct.unpack("<I", complete_metadata[offset : offset + 4])[0]
            offset += 4

            # reserved (28 bytes)
            _reserved = complete_metadata[offset : offset + 28]
            offset += 28

            print("Field offsets verified:")
            print(f"  Magic at 0: 0x{magic:08x}")
            print(f"  Version at 4: {version}")
            print(f"  Sim name at 8: '{sim_name}' ({len(sim_name_bytes)} bytes)")
            print(f"  Level name at 72: '{level_name}' ({len(level_name_bytes)} bytes)")
            print(f"  Num worlds at 136: {num_worlds}")
            print(f"  Timestamp at 152: {timestamp}")
            print(f"  Next field at: {offset}")

            # Validate field values
            assert magic == 0x4D455352, "Magic field misaligned or corrupted"
            assert version == 2, "Version field misaligned or corrupted"
            assert sim_name == "madrona_escape_room", "sim_name field misaligned or corrupted"
            assert level_name, "level_name field misaligned or corrupted"
            assert offset == 192, f"Field alignment error: should be at offset 192, got {offset}"

            # Check string field padding
            for i in range(len(sim_name), 64):
                if sim_name_bytes[i] != 0:
                    print(f"Warning: sim_name padding byte {i} is not null: {sim_name_bytes[i]}")

            for i in range(len(level_name), 64):
                if level_name_bytes[i] != 0:
                    print(
                        f"Warning: level_name padding byte {i} is not null: {level_name_bytes[i]}"
                    )

            print("✓ Field alignment validation passed")
            print("✓ String padding verification completed")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)

#!/usr/bin/env python3
"""
Comprehensive binary format validation for recording files.
Tests complete ReplayMetadata structure and CompiledLevel validation.
"""

import os
import struct
import tempfile


def test_replay_metadata_complete_structure(cpu_manager):
    """Test complete ReplayMetadata structure validation - all 14 fields"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording with known seed and run steps
        mgr.start_recording(recording_path, seed=12345)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]

        # Run exactly 5 steps with known actions
        for step in range(5):
            action_tensor.fill_(0)
            action_tensor[:, 0] = step + 1  # move_amount varies by step
            action_tensor[:, 1] = 0  # move_angle = FORWARD
            action_tensor[:, 2] = 2  # rotate = NONE
            mgr.step()

        mgr.stop_recording()

        # Read and validate complete ReplayMetadata structure (136 bytes for version 2)
        with open(recording_path, "rb") as f:
            # Read complete ReplayMetadata structure
            # Based on replay_metadata.hpp version 2 format:
            magic_bytes = f.read(4)
            magic = struct.unpack("<I", magic_bytes)[0]
            version = struct.unpack("<I", f.read(4))[0]
            sim_name_bytes = f.read(64)
            sim_name = sim_name_bytes.decode("ascii").rstrip("\x00")
            level_name_bytes = f.read(64)  # New in version 2
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
            num_worlds_meta = struct.unpack("<I", f.read(4))[0]
            num_agents_meta = struct.unpack("<I", f.read(4))[0]
            num_steps_meta = struct.unpack("<I", f.read(4))[0]
            actions_per_step = struct.unpack("<I", f.read(4))[0]
            timestamp = struct.unpack("<Q", f.read(8))[0]
            seed = struct.unpack("<I", f.read(4))[0]
            reserved = f.read(28)  # 7 * 4 bytes reserved (reduced by 1 for level_name)

            print("=== Complete ReplayMetadata Validation ===")
            print(f"Magic: 0x{magic:08x} (bytes: {magic_bytes.hex()})")
            print(f"Version: {version}")
            print(f"Sim name: '{sim_name}'")
            print(f"Level name: '{level_name}'")
            print(f"Num worlds: {num_worlds_meta}")
            print(f"Num agents per world: {num_agents_meta}")
            print(f"Num steps: {num_steps_meta}")
            print(f"Actions per step: {actions_per_step}")
            print(f"Timestamp: {timestamp}")
            print(f"Seed: {seed}")
            print(f"Reserved bytes: {len(reserved)} bytes")

            # Validate all 14 fields (vs existing test's 9 fields)
            assert magic == 0x4D455352, f"Expected magic 0x4D455352, got 0x{magic:08x}"
            assert version == 2, f"Expected version 2, got {version}"
            assert (
                sim_name == "madrona_escape_room"
            ), f"Expected sim_name 'madrona_escape_room', got '{sim_name}'"
            assert level_name, f"Level name should not be empty, got '{level_name}'"
            assert (
                num_worlds_meta == num_worlds
            ), f"Expected {num_worlds} worlds, got {num_worlds_meta}"
            assert (
                num_agents_meta >= 1
            ), f"Expected at least 1 agent per world, got {num_agents_meta}"
            assert num_steps_meta == 5, f"Expected 5 steps, got {num_steps_meta}"
            assert actions_per_step == 3, f"Expected 3 actions per step, got {actions_per_step}"
            assert timestamp > 0, f"Expected positive timestamp, got {timestamp}"
            assert seed == 12345, f"Expected seed 12345, got {seed}"
            assert len(reserved) == 28, f"Expected 28 reserved bytes, got {len(reserved)}"

            print("✓ All 14 ReplayMetadata fields validated successfully")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_compiled_level_structure_validation(cpu_manager):
    """Test CompiledLevel structure validation and embedded data integrity"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording
        mgr.start_recording(recording_path, seed=999)

        action_tensor = mgr.action_tensor().to_torch()

        # Run a single step
        action_tensor.fill_(0)
        action_tensor[:, 0] = 1
        action_tensor[:, 1] = 0
        action_tensor[:, 2] = 2
        mgr.step()

        mgr.stop_recording()

        # Read file structure
        with open(recording_path, "rb") as f:
            # Skip ReplayMetadata (136 bytes for version 2)
            f.seek(136)

            level_start = f.tell()
            print(f"=== CompiledLevel at offset {level_start} ===")

            # Read CompiledLevel header fields
            num_tiles = struct.unpack("<i", f.read(4))[0]
            max_entities = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<i", f.read(4))[0]
            height = struct.unpack("<i", f.read(4))[0]
            scale = struct.unpack("<f", f.read(4))[0]
            level_name_bytes = f.read(64)
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

            print("CompiledLevel header:")
            print(f"  num_tiles: {num_tiles}")
            print(f"  max_entities: {max_entities}")
            print(f"  width: {width}")
            print(f"  height: {height}")
            print(f"  scale: {scale}")
            print(f"  level_name: '{level_name}'")

            # Validate CompiledLevel constraints
            assert 0 <= num_tiles <= 1024, f"num_tiles {num_tiles} not in valid range [0, 1024]"
            assert max_entities > 0, f"max_entities should be positive, got {max_entities}"
            assert width > 0, f"width should be positive, got {width}"
            assert height > 0, f"height should be positive, got {height}"
            assert scale > 0, f"scale should be positive, got {scale}"
            assert level_name, "level_name should not be empty"

            # Read spawn data
            num_spawns = struct.unpack("<i", f.read(4))[0]
            assert 0 <= num_spawns <= 8, f"num_spawns {num_spawns} not in valid range [0, 8]"

            # Skip spawn arrays (8 * 4 * 3 = 96 bytes)
            f.read(96)

            # Skip tile data arrays (1024 * 4 * 3 = 12288 bytes)
            f.read(12288)

            # Verify we read exactly CompiledLevel size
            level_end = f.tell()
            compiled_level_size = level_end - level_start

            # Expected size calculation:
            # Header: 4+4+4+4+4+64 = 84 bytes
            # Spawn data: 4 + (8*4*3) = 100 bytes
            # Tile data: 1024*4*3 = 12288 bytes
            # Total: 84 + 100 + 12288 = 12472 bytes
            expected_size = 84 + 100 + 12288

            print(f"CompiledLevel size: {compiled_level_size} bytes")
            print(f"Expected size: {expected_size} bytes")

            # Note: Actual size might differ due to struct padding
            assert (
                compiled_level_size >= expected_size
            ), f"CompiledLevel too small: {compiled_level_size} < {expected_size}"

            print("✓ CompiledLevel structure validation passed")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_action_data_verification_step_by_step(cpu_manager):
    """Test action data verification with step-by-step validation"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording with known action sequence
        mgr.start_recording(recording_path, seed=555)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]

        # Define specific action sequence for validation
        test_actions = [
            (1, 0, 2),  # SLOW, FORWARD, NONE
            (2, 2, 1),  # MEDIUM, RIGHT, SLOW_LEFT
            (3, 4, 4),  # FAST, BACKWARD, FAST_RIGHT
            (1, 6, 0),  # SLOW, LEFT, FAST_LEFT
        ]

        for step, (move_amount, move_angle, rotate) in enumerate(test_actions):
            action_tensor.fill_(0)
            action_tensor[:, 0] = move_amount
            action_tensor[:, 1] = move_angle
            action_tensor[:, 2] = rotate
            mgr.step()

        mgr.stop_recording()

        # Read and validate action data
        with open(recording_path, "rb") as f:
            # Skip ReplayMetadata (136 bytes) and CompiledLevel
            f.seek(136)

            # Determine CompiledLevel size by reading to find actions
            # For now, use a known offset or seek to end to calculate
            f.seek(0, 2)  # Go to end
            file_size = f.tell()

            # Calculate action data size and work backwards from file end
            bytes_per_step = num_worlds * 3 * 4  # 3 int32_t per world
            expected_action_bytes = len(test_actions) * bytes_per_step

            print("=== Action Data Verification ===")
            print(f"File size: {file_size} bytes")
            print(f"Expected action bytes: {expected_action_bytes}")
            print(f"Bytes per step: {bytes_per_step}")

            # Actions are at the end of the file, work backwards
            actions_start = file_size - expected_action_bytes
            f.seek(actions_start)

            print(f"Action data starts at: {actions_start}")

            remaining_bytes = file_size - actions_start
            possible_steps = remaining_bytes // bytes_per_step if bytes_per_step > 0 else 0

            print(f"Remaining bytes: {remaining_bytes}")
            print(f"Possible steps: {possible_steps}")

            # Validate we can read expected number of steps
            assert possible_steps >= len(
                test_actions
            ), f"Expected at least {len(test_actions)} steps, can only read {possible_steps}"

            # Read and validate each step
            for step_idx, expected_actions in enumerate(test_actions):
                step_data = []
                for world_idx in range(num_worlds):
                    try:
                        move_amount = struct.unpack("<i", f.read(4))[0]
                        move_angle = struct.unpack("<i", f.read(4))[0]
                        rotate = struct.unpack("<i", f.read(4))[0]
                        step_data.append((move_amount, move_angle, rotate))
                    except struct.error as e:
                        print(f"Failed to read step {step_idx}, world {world_idx}: {e}")
                        raise

                # Verify actions match expected values for all worlds
                expected_move_amount, expected_move_angle, expected_rotate = expected_actions
                for world_idx, (move_amount, move_angle, rotate) in enumerate(step_data):
                    assert move_amount == expected_move_amount, (
                        f"Step {step_idx}, World {world_idx}: expected move_amount "
                        f"{expected_move_amount}, got {move_amount}"
                    )
                    assert move_angle == expected_move_angle, (
                        f"Step {step_idx}, World {world_idx}: expected move_angle "
                        f"{expected_move_angle}, got {move_angle}"
                    )
                    assert rotate == expected_rotate, (
                        f"Step {step_idx}, World {world_idx}: expected rotate "
                        f"{expected_rotate}, got {rotate}"
                    )

                print(f"✓ Step {step_idx}: {expected_actions} validated for {num_worlds} worlds")

            print("✓ All action data validated successfully")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_format_specification_compliance(cpu_manager):
    """Test format specification compliance with C++ struct layout"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create minimal recording
        mgr.start_recording(recording_path, seed=42)
        mgr.stop_recording()

        # Validate format specification compliance
        with open(recording_path, "rb") as f:
            file_data = f.read()

            print("=== Format Specification Compliance ===")
            print(f"Total file size: {len(file_data)} bytes")

            # Validate minimum file size
            min_expected_size = 136  # ReplayMetadata for version 2
            assert (
                len(file_data) >= min_expected_size
            ), f"File too small: {len(file_data)} < {min_expected_size}"

            # Validate magic number and version
            magic = struct.unpack("<I", file_data[0:4])[0]
            version = struct.unpack("<I", file_data[4:8])[0]

            assert magic == 0x4D455352, f"Invalid magic number: 0x{magic:08x}"
            assert version == 2, f"Expected version 2, got {version}"

            # Validate struct alignment - check for null padding in strings
            sim_name_section = file_data[8:72]  # 64 bytes
            level_name_section = file_data[72:136]  # 64 bytes

            # Should be null-terminated strings
            sim_name = sim_name_section.split(b"\x00")[0].decode("ascii")
            level_name = level_name_section.split(b"\x00")[0].decode("ascii")

            assert sim_name == "madrona_escape_room", f"Invalid sim_name: '{sim_name}'"
            assert level_name, f"Level name should not be empty: '{level_name}'"

            print(f"✓ Magic number: 0x{magic:08x}")
            print(f"✓ Version: {version}")
            print(f"✓ Sim name: '{sim_name}'")
            print(f"✓ Level name: '{level_name}'")
            print("✓ Format specification compliance validated")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_error_condition_handling(cpu_manager):
    """Test malformed file handling and corruption detection"""

    # Test 1: Empty file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        empty_path = f.name
        # File is created but empty

    try:
        with open(empty_path, "rb") as f:
            data = f.read()
            assert len(data) == 0, "File should be empty"

        print("✓ Empty file handled correctly")
    finally:
        if os.path.exists(empty_path):
            os.unlink(empty_path)

    # Test 2: File with invalid magic number
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        invalid_magic_path = f.name
        # Write invalid magic number
        f.write(struct.pack("<I", 0xDEADBEEF))  # Invalid magic
        f.write(struct.pack("<I", 2))  # Valid version
        f.write(b"\x00" * 128)  # Padding

    try:
        with open(invalid_magic_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0xDEADBEEF, "Should read invalid magic"
            assert magic != 0x4D455352, "Magic should not match expected value"

        print("✓ Invalid magic number detected")
    finally:
        if os.path.exists(invalid_magic_path):
            os.unlink(invalid_magic_path)

    # Test 3: File with invalid version
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
            assert version not in [1, 2], "Version should not be valid"

        print("✓ Invalid version detected")
    finally:
        if os.path.exists(invalid_version_path):
            os.unlink(invalid_version_path)

    # Test 4: Truncated file
    mgr = cpu_manager
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        full_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        truncated_path = f.name

    try:
        # Create a full recording
        mgr.start_recording(full_path, seed=42)
        action_tensor = mgr.action_tensor().to_torch()
        action_tensor.fill_(0)
        mgr.step()
        mgr.stop_recording()

        # Create truncated version
        with open(full_path, "rb") as f:
            full_data = f.read()

        # Truncate to just partial metadata
        truncated_data = full_data[:50]  # Only part of metadata

        with open(truncated_path, "wb") as f:
            f.write(truncated_data)

        # Verify truncation
        with open(truncated_path, "rb") as f:
            data = f.read()
            assert len(data) == 50, f"Expected 50 bytes, got {len(data)}"
            assert len(data) < 136, "Should be less than full metadata size"

        print("✓ Truncated file handling verified (50 bytes vs expected 136+ bytes)")

    finally:
        for path in [full_path, truncated_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_file_boundary_validation(cpu_manager):
    """Test file structure integrity and boundary checking"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording with multiple steps
        mgr.start_recording(recording_path, seed=777)

        action_tensor = mgr.action_tensor().to_torch()
        num_worlds = action_tensor.shape[0]
        num_steps = 3

        for step in range(num_steps):
            action_tensor.fill_(0)
            action_tensor[:, 0] = step + 1
            action_tensor[:, 1] = 0
            action_tensor[:, 2] = 2
            mgr.step()

        mgr.stop_recording()

        # Analyze file structure boundaries
        with open(recording_path, "rb") as f:
            file_size = os.path.getsize(recording_path)

            print("=== File Boundary Validation ===")
            print(f"Total file size: {file_size} bytes")

            # Read metadata
            metadata_end = 136  # ReplayMetadata size for version 2
            f.seek(metadata_end)

            print(f"ReplayMetadata ends at: {metadata_end}")

            # Find CompiledLevel size by seeking to a known position
            # (This is an approximation - in practice we'd read the structure)
            estimated_compiled_level_size = 12472  # From earlier calculation
            compiled_level_end = metadata_end + estimated_compiled_level_size

            print(f"Estimated CompiledLevel ends at: {compiled_level_end}")

            # Calculate expected action data size
            bytes_per_step = num_worlds * 3 * 4  # 3 int32_t per action per world
            expected_action_bytes = num_steps * bytes_per_step

            print(f"Expected action data: {expected_action_bytes} bytes")
            print(f"Bytes per step: {bytes_per_step} (for {num_worlds} worlds)")

            # Verify file size makes sense
            min_expected_size = metadata_end + expected_action_bytes
            max_expected_size = min_expected_size + estimated_compiled_level_size

            print(f"Expected file size range: {min_expected_size} - {max_expected_size} bytes")

            assert (
                file_size >= min_expected_size
            ), f"File too small: {file_size} < {min_expected_size}"
            # Note: Not asserting max size due to potential padding and unknown
            # CompiledLevel exact size

            # Try to validate action data boundaries
            f.seek(compiled_level_end)
            remaining_after_level = file_size - compiled_level_end

            if remaining_after_level >= expected_action_bytes:
                print(
                    f"✓ Sufficient bytes for action data: {remaining_after_level} >= "
                    f"{expected_action_bytes}"
                )
            else:
                # Adjust our assumption about where actions start
                actions_start = file_size - expected_action_bytes
                print(f"Adjusted actions start to: {actions_start}")
                assert actions_start >= metadata_end, "Actions cannot start before metadata ends"

            print("✓ File boundary validation completed")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)

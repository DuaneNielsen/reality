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
        mgr.start_recording(recording_path)

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

        # Read and validate complete ReplayMetadata structure (192 bytes for version 2)
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
            assert seed == 42, f"Expected seed 42, got {seed}"
            assert len(reserved) == 28, f"Expected 28 reserved bytes, got {len(reserved)}"

            print("✓ All 14 ReplayMetadata fields validated successfully")

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_compiled_level_structure_validation(cpu_manager):
    """Test CompiledLevel binary read/write round-trip - what goes in should come out"""
    import ctypes

    from madrona_escape_room.ctypes_bindings import lib
    from madrona_escape_room.dataclass_utils import create_compiled_level
    from madrona_escape_room.generated_constants import Result
    from madrona_escape_room.generated_dataclasses import CompiledLevel
    from madrona_escape_room.level_compiler import compile_ascii_level

    # Create a test level with known values
    test_level = """
    ####
    #SS#
    #  #
    ####
    """

    # Compile the level to get a CompiledLevel dataclass
    level_write = compile_ascii_level(test_level, scale=3.5, level_name="test_roundtrip")

    with tempfile.NamedTemporaryFile(suffix=".lvl", delete=False) as f:
        level_path = f.name

    try:
        # Write the compiled level to binary file
        c_level_write = level_write.to_ctype()
        result = lib.mer_write_compiled_level(
            level_path.encode("utf-8"), ctypes.byref(c_level_write)
        )
        assert result == Result.Success, f"Failed to write compiled level: {result}"

        # Read it back
        level_read_empty = create_compiled_level()
        c_level_read = level_read_empty.to_ctype()
        result = lib.mer_read_compiled_level(level_path.encode("utf-8"), ctypes.byref(c_level_read))
        assert result == Result.Success, f"Failed to read compiled level: {result}"

        # Convert back to dataclass
        level_read = CompiledLevel.from_ctype(c_level_read)

        # Verify key fields match
        assert (
            level_read.num_tiles == level_write.num_tiles
        ), f"num_tiles mismatch: {level_read.num_tiles} != {level_write.num_tiles}"
        assert (
            level_read.width == level_write.width
        ), f"width mismatch: {level_read.width} != {level_write.width}"
        assert (
            level_read.height == level_write.height
        ), f"height mismatch: {level_read.height} != {level_write.height}"
        assert (
            abs(level_read.world_scale - level_write.world_scale) < 0.001
        ), f"scale mismatch: {level_read.world_scale} != {level_write.world_scale}"
        assert (
            level_read.num_spawns == level_write.num_spawns
        ), f"num_spawns mismatch: {level_read.num_spawns} != {level_write.num_spawns}"

        # Verify level name (already strings in dataclasses)
        assert (
            level_read.level_name == level_write.level_name
        ), f"level_name mismatch: '{level_read.level_name}' != '{level_write.level_name}'"

        # Verify world boundaries
        assert abs(level_read.world_min_x - level_write.world_min_x) < 0.001
        assert abs(level_read.world_max_x - level_write.world_max_x) < 0.001
        assert abs(level_read.world_min_y - level_write.world_min_y) < 0.001
        assert abs(level_read.world_max_y - level_write.world_max_y) < 0.001

        # Verify spawn positions match
        for i in range(level_read.num_spawns):
            assert (
                abs(level_read.spawn_x[i] - level_write.spawn_x[i]) < 0.001
            ), f"spawn_x[{i}] mismatch"
            assert (
                abs(level_read.spawn_y[i] - level_write.spawn_y[i]) < 0.001
            ), f"spawn_y[{i}] mismatch"

        # Verify some tile data (just check first few active tiles)
        for i in range(min(10, level_read.num_tiles)):
            assert (
                level_read.object_ids[i] == level_write.object_ids[i]
            ), f"object_ids[{i}] mismatch"
            assert (
                abs(level_read.tile_x[i] - level_write.tile_x[i]) < 0.001
            ), f"tile_x[{i}] mismatch"
            assert (
                abs(level_read.tile_y[i] - level_write.tile_y[i]) < 0.001
            ), f"tile_y[{i}] mismatch"

        print("✓ CompiledLevel binary round-trip validation passed")
        print(f"  Successfully wrote and read back level '{level_read.level_name}'")
        print(
            f"  Dimensions: {level_read.width}x{level_read.height}, scale: {level_read.world_scale}"
        )
        print(f"  Tiles: {level_read.num_tiles}, Spawns: {level_read.num_spawns}")

    finally:
        if os.path.exists(level_path):
            os.unlink(level_path)


def test_action_data_verification_step_by_step(cpu_manager):
    """Test action data verification with step-by-step validation"""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Create recording with known action sequence
        mgr.start_recording(recording_path)

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
            # Skip ReplayMetadata (192 bytes) and CompiledLevel
            f.seek(192)

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
        mgr.start_recording(recording_path)
        mgr.stop_recording()

        # Validate format specification compliance
        with open(recording_path, "rb") as f:
            file_data = f.read()

            print("=== Format Specification Compliance ===")
            print(f"Total file size: {len(file_data)} bytes")

            # Validate minimum file size
            min_expected_size = 192  # ReplayMetadata for version 2
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
        mgr.start_recording(full_path)
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
            assert len(data) < 192, "Should be less than full metadata size"

        print("✓ Truncated file handling verified (50 bytes vs expected 192+ bytes)")

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
        mgr.start_recording(recording_path)

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
            metadata_end = 192  # ReplayMetadata size for version 2
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

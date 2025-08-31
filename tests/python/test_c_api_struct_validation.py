"""
Test C API struct validation and memory layout with pytest framework
"""

import ctypes

import pytest

from madrona_escape_room.ctypes_bindings import (
    ManagerConfig,
    MER_CompiledLevel,
    create_manager_with_levels,
    lib,
)
from madrona_escape_room.level_compiler import compile_ascii_level


class TestCAPIStructValidation:
    """Test C API struct validation and direct C API calls"""

    def test_c_api_validation_function(self):
        """Test C API validation function directly"""
        level = """######
#S...#
#.C..#
######"""

        compiled = compile_ascii_level(level)

        # Use the library and struct already loaded by ctypes_bindings
        # Convert compiled level to ctypes structure
        level_struct = compiled.to_ctype()

        # The library functions are already configured in ctypes_bindings

        # Validate
        result = lib.mer_validate_compiled_level(ctypes.byref(level_struct))

        if result != 0:  # MER_SUCCESS = 0
            error_msg = lib.mer_result_to_string(result)
            if error_msg:
                error_str = error_msg.decode("utf-8")
            else:
                error_str = f"Error code {result}"
            pytest.fail(f"C API validation failed: {error_str}")

    def test_manager_creation_with_c_api(self):
        """Test creating manager through direct C API calls"""
        level = """########
#S.....#
#......#
########"""

        compiled = compile_ascii_level(level)

        # Use the module's existing ManagerConfig
        config = ManagerConfig()
        config.exec_mode = 0  # CPU
        config.gpu_id = 0
        config.num_worlds = 1
        config.rand_seed = 42
        config.auto_reset = True
        config.enable_batch_renderer = False
        config.batch_render_view_width = 64
        config.batch_render_view_height = 64

        # Create manager using the module's helper
        handle = ctypes.c_void_p()
        result, c_config, levels_array = create_manager_with_levels(
            ctypes.byref(handle), config, compiled
        )

        if result != 0:
            error_msg = lib.mer_result_to_string(result)
            if error_msg:
                error_str = error_msg.decode("utf-8")
            else:
                error_str = f"Error code {result}"
            pytest.fail(f"Manager creation failed: {error_str}")

        assert handle.value != 0, "Manager handle should not be null"

        # Clean up
        result = lib.mer_destroy_manager(handle)
        assert result == 0, "Manager destruction should succeed"

    def test_struct_memory_layout(self):
        """Test struct matches C API expected size and field access"""
        # Already imports from ctypes_bindings correctly
        from madrona_escape_room.ctypes_bindings import lib

        # Get expected size from C API
        expected_size = lib.mer_get_compiled_level_size()

        # Create dataclass and convert to ctypes
        struct = MER_CompiledLevel()
        c_struct = struct.to_ctype()
        actual_size = ctypes.sizeof(c_struct)

        assert actual_size == expected_size, f"Size mismatch: {actual_size} != {expected_size}"

        # Test field access on ctypes version
        c_struct.num_tiles = 42
        c_struct.max_entities = 100
        c_struct.width = 10
        c_struct.height = 8
        c_struct.world_scale = 1.5

        assert c_struct.num_tiles == 42
        assert c_struct.max_entities == 100
        assert c_struct.width == 10
        assert c_struct.height == 8
        assert abs(c_struct.world_scale - 1.5) < 0.001

        # Test array bounds (use 1023 as max since arrays are 1024 elements)
        c_struct.object_ids[0] = 1
        c_struct.object_ids[1023] = 2
        c_struct.tile_x[0] = 3.14
        c_struct.tile_x[1023] = 2.71
        c_struct.tile_y[0] = -1.0
        c_struct.tile_y[1023] = 1.0

        assert c_struct.object_ids[0] == 1
        assert c_struct.object_ids[1023] == 2
        assert abs(c_struct.tile_x[0] - 3.14) < 0.001
        assert abs(c_struct.tile_x[1023] - 2.71) < 0.001
        assert abs(c_struct.tile_y[0] + 1.0) < 0.001
        assert abs(c_struct.tile_y[1023] - 1.0) < 0.001

    def test_multiple_level_types_c_validation(self):
        """Test different level types with C API validation"""
        test_levels = [
            # Simple room
            """####
#S.#
####""",
            # Room with obstacles
            """######
#S...#
#.CC.#
######""",
            # Larger level
            """##########
#S.......#
#..####..#
#........#
##########""",
        ]

        # Use the library already loaded by ctypes_bindings
        # No need to manually define structs or function signatures

        for i, level_ascii in enumerate(test_levels):
            compiled = compile_ascii_level(level_ascii)

            # Convert to ctypes struct using the module's conversion
            struct = compiled.to_ctype()

            # Validate with C API (already configured in ctypes_bindings)
            result = lib.mer_validate_compiled_level(ctypes.byref(struct))
            assert result == 0, f"Level {i} validation failed with code {result}"

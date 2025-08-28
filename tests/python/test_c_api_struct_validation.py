"""
Test C API struct validation and memory layout with pytest framework
"""

import ctypes

import pytest

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

        # Load C API library directly
        lib = ctypes.CDLL("./build/libmadrona_escape_room_c_api.so")

        # Define struct manually - must match C++ CompiledLevel exactly
        class MER_CompiledLevel(ctypes.Structure):
            _fields_ = [
                ("num_tiles", ctypes.c_int32),
                ("max_entities", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("world_scale", ctypes.c_float),
                ("done_on_collide", ctypes.c_bool),
                ("level_name", ctypes.c_char * 64),  # MAX_LEVEL_NAME_LENGTH
                # World boundaries
                ("world_min_x", ctypes.c_float),
                ("world_max_x", ctypes.c_float),
                ("world_min_y", ctypes.c_float),
                ("world_max_y", ctypes.c_float),
                ("world_min_z", ctypes.c_float),
                ("world_max_z", ctypes.c_float),
                ("num_spawns", ctypes.c_int32),
                ("spawn_x", ctypes.c_float * 8),  # MAX_SPAWNS
                ("spawn_y", ctypes.c_float * 8),
                ("spawn_facing", ctypes.c_float * 8),
                ("object_ids", ctypes.c_int32 * 1024),  # MAX_TILES
                ("tile_x", ctypes.c_float * 1024),
                ("tile_y", ctypes.c_float * 1024),
                ("tile_z", ctypes.c_float * 1024),
                ("tile_persistent", ctypes.c_bool * 1024),
                ("tile_render_only", ctypes.c_bool * 1024),
                ("tile_entity_type", ctypes.c_int32 * 1024),
                ("tile_response_type", ctypes.c_int32 * 1024),
                ("tile_scale_x", ctypes.c_float * 1024),
                ("tile_scale_y", ctypes.c_float * 1024),
                ("tile_scale_z", ctypes.c_float * 1024),
                ("tile_rot_w", ctypes.c_float * 1024),
                ("tile_rot_x", ctypes.c_float * 1024),
                ("tile_rot_y", ctypes.c_float * 1024),
                ("tile_rot_z", ctypes.c_float * 1024),
                ("tile_rand_x", ctypes.c_float * 1024),
                ("tile_rand_y", ctypes.c_float * 1024),
                ("tile_rand_z", ctypes.c_float * 1024),
                ("tile_rand_rot_z", ctypes.c_float * 1024),
                ("tile_rand_scale_x", ctypes.c_float * 1024),
                ("tile_rand_scale_y", ctypes.c_float * 1024),
                ("tile_rand_scale_z", ctypes.c_float * 1024),
            ]

        # Populate struct
        level_struct = MER_CompiledLevel()
        level_struct.num_tiles = compiled.num_tiles
        level_struct.max_entities = compiled.max_entities
        level_struct.width = compiled.width
        level_struct.height = compiled.height
        level_struct.world_scale = compiled.world_scale
        level_struct.done_on_collide = compiled.done_on_collide
        level_struct.level_name = compiled.level_name
        # Set world boundaries
        level_struct.world_min_x = compiled.world_min_x
        level_struct.world_max_x = compiled.world_max_x
        level_struct.world_min_y = compiled.world_min_y
        level_struct.world_max_y = compiled.world_max_y
        level_struct.world_min_z = compiled.world_min_z
        level_struct.world_max_z = compiled.world_max_z
        level_struct.num_spawns = compiled.num_spawns

        for i in range(8):  # MAX_SPAWNS
            level_struct.spawn_x[i] = compiled.spawn_x[i]
            level_struct.spawn_y[i] = compiled.spawn_y[i]
            level_struct.spawn_facing[i] = compiled.spawn_facing[i]

        for i in range(1024):  # MAX_TILES
            level_struct.object_ids[i] = compiled.object_ids[i]
            level_struct.tile_x[i] = compiled.tile_x[i]
            level_struct.tile_y[i] = compiled.tile_y[i]
            level_struct.tile_z[i] = compiled.tile_z[i]
            level_struct.tile_persistent[i] = compiled.tile_persistent[i]
            level_struct.tile_render_only[i] = compiled.tile_render_only[i]
            level_struct.tile_entity_type[i] = compiled.tile_entity_type[i]
            level_struct.tile_response_type[i] = compiled.tile_response_type[i]
            level_struct.tile_scale_x[i] = compiled.tile_scale_x[i]
            level_struct.tile_scale_y[i] = compiled.tile_scale_y[i]
            level_struct.tile_scale_z[i] = compiled.tile_scale_z[i]
            level_struct.tile_rot_w[i] = compiled.tile_rotation[i][0]
            level_struct.tile_rot_x[i] = compiled.tile_rotation[i][1]
            level_struct.tile_rot_y[i] = compiled.tile_rotation[i][2]
            level_struct.tile_rot_z[i] = compiled.tile_rotation[i][3]
            # New randomization fields - default to 0 if not present
            level_struct.tile_rand_x[i] = compiled.tile_rand_x[i]
            level_struct.tile_rand_y[i] = compiled.tile_rand_y[i]
            level_struct.tile_rand_z[i] = compiled.tile_rand_z[i]
            level_struct.tile_rand_rot_z[i] = compiled.tile_rand_rot_z[i]
            level_struct.tile_rand_scale_x[i] = compiled.tile_rand_scale_x[i]
            level_struct.tile_rand_scale_y[i] = compiled.tile_rand_scale_y[i]
            level_struct.tile_rand_scale_z[i] = compiled.tile_rand_scale_z[i]

        # Set up C API validation
        lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(MER_CompiledLevel)]
        lib.mer_validate_compiled_level.restype = ctypes.c_int
        lib.mer_result_to_string.argtypes = [ctypes.c_int]
        lib.mer_result_to_string.restype = ctypes.c_char_p

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

        # Load library
        lib = ctypes.CDLL("./build/libmadrona_escape_room_c_api.so")

        # Define structs - must match C++ CompiledLevel exactly
        class MER_CompiledLevel(ctypes.Structure):
            _fields_ = [
                ("num_tiles", ctypes.c_int32),
                ("max_entities", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("world_scale", ctypes.c_float),
                ("done_on_collide", ctypes.c_bool),
                ("level_name", ctypes.c_char * 64),  # MAX_LEVEL_NAME_LENGTH
                # World boundaries
                ("world_min_x", ctypes.c_float),
                ("world_max_x", ctypes.c_float),
                ("world_min_y", ctypes.c_float),
                ("world_max_y", ctypes.c_float),
                ("world_min_z", ctypes.c_float),
                ("world_max_z", ctypes.c_float),
                ("num_spawns", ctypes.c_int32),
                ("spawn_x", ctypes.c_float * 8),  # MAX_SPAWNS
                ("spawn_y", ctypes.c_float * 8),
                ("spawn_facing", ctypes.c_float * 8),
                ("object_ids", ctypes.c_int32 * 1024),  # MAX_TILES
                ("tile_x", ctypes.c_float * 1024),
                ("tile_y", ctypes.c_float * 1024),
                ("tile_z", ctypes.c_float * 1024),
                ("tile_persistent", ctypes.c_bool * 1024),
                ("tile_render_only", ctypes.c_bool * 1024),
                ("tile_entity_type", ctypes.c_int32 * 1024),
                ("tile_response_type", ctypes.c_int32 * 1024),
                ("tile_scale_x", ctypes.c_float * 1024),
                ("tile_scale_y", ctypes.c_float * 1024),
                ("tile_scale_z", ctypes.c_float * 1024),
                ("tile_rot_w", ctypes.c_float * 1024),
                ("tile_rot_x", ctypes.c_float * 1024),
                ("tile_rot_y", ctypes.c_float * 1024),
                ("tile_rot_z", ctypes.c_float * 1024),
                ("tile_rand_x", ctypes.c_float * 1024),
                ("tile_rand_y", ctypes.c_float * 1024),
                ("tile_rand_z", ctypes.c_float * 1024),
                ("tile_rand_rot_z", ctypes.c_float * 1024),
                ("tile_rand_scale_x", ctypes.c_float * 1024),
                ("tile_rand_scale_y", ctypes.c_float * 1024),
                ("tile_rand_scale_z", ctypes.c_float * 1024),
            ]

        class MER_ManagerConfig(ctypes.Structure):
            _fields_ = [
                ("exec_mode", ctypes.c_int),
                ("gpu_id", ctypes.c_int),
                ("num_worlds", ctypes.c_uint32),
                ("rand_seed", ctypes.c_uint32),
                ("auto_reset", ctypes.c_bool),
                ("enable_batch_renderer", ctypes.c_bool),
                ("batch_render_view_width", ctypes.c_uint32),
                ("batch_render_view_height", ctypes.c_uint32),
            ]

        # Create and populate level struct
        level_struct = MER_CompiledLevel()
        level_struct.num_tiles = compiled.num_tiles
        level_struct.max_entities = compiled.max_entities
        level_struct.width = compiled.width
        level_struct.height = compiled.height
        level_struct.world_scale = compiled.world_scale
        level_struct.done_on_collide = compiled.done_on_collide
        level_struct.level_name = compiled.level_name
        # Set world boundaries
        level_struct.world_min_x = compiled.world_min_x
        level_struct.world_max_x = compiled.world_max_x
        level_struct.world_min_y = compiled.world_min_y
        level_struct.world_max_y = compiled.world_max_y
        level_struct.world_min_z = compiled.world_min_z
        level_struct.world_max_z = compiled.world_max_z
        level_struct.num_spawns = compiled.num_spawns

        for i in range(8):  # MAX_SPAWNS
            level_struct.spawn_x[i] = compiled.spawn_x[i]
            level_struct.spawn_y[i] = compiled.spawn_y[i]
            level_struct.spawn_facing[i] = compiled.spawn_facing[i]

        for i in range(1024):  # MAX_TILES
            level_struct.object_ids[i] = compiled.object_ids[i]
            level_struct.tile_x[i] = compiled.tile_x[i]
            level_struct.tile_y[i] = compiled.tile_y[i]
            level_struct.tile_z[i] = compiled.tile_z[i]
            level_struct.tile_persistent[i] = compiled.tile_persistent[i]
            level_struct.tile_render_only[i] = compiled.tile_render_only[i]
            level_struct.tile_entity_type[i] = compiled.tile_entity_type[i]
            level_struct.tile_response_type[i] = compiled.tile_response_type[i]
            level_struct.tile_scale_x[i] = compiled.tile_scale_x[i]
            level_struct.tile_scale_y[i] = compiled.tile_scale_y[i]
            level_struct.tile_scale_z[i] = compiled.tile_scale_z[i]
            level_struct.tile_rot_w[i] = compiled.tile_rotation[i][0]
            level_struct.tile_rot_x[i] = compiled.tile_rotation[i][1]
            level_struct.tile_rot_y[i] = compiled.tile_rotation[i][2]
            level_struct.tile_rot_z[i] = compiled.tile_rotation[i][3]
            # New randomization fields - default to 0 if not present
            level_struct.tile_rand_x[i] = compiled.tile_rand_x[i]
            level_struct.tile_rand_y[i] = compiled.tile_rand_y[i]
            level_struct.tile_rand_z[i] = compiled.tile_rand_z[i]
            level_struct.tile_rand_rot_z[i] = compiled.tile_rand_rot_z[i]
            level_struct.tile_rand_scale_x[i] = compiled.tile_rand_scale_x[i]
            level_struct.tile_rand_scale_y[i] = compiled.tile_rand_scale_y[i]
            level_struct.tile_rand_scale_z[i] = compiled.tile_rand_scale_z[i]

        # Create config
        config = MER_ManagerConfig()
        config.exec_mode = 0  # CPU
        config.gpu_id = 0
        config.num_worlds = 1
        config.rand_seed = 42
        config.auto_reset = True
        config.enable_batch_renderer = False
        config.batch_render_view_width = 64
        config.batch_render_view_height = 64

        # Create level array
        LevelArray = MER_CompiledLevel * 1
        levels_array = LevelArray(level_struct)

        # Set up function signatures
        lib.mer_create_manager.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(MER_ManagerConfig),
            ctypes.POINTER(MER_CompiledLevel),
            ctypes.c_uint32,
        ]
        lib.mer_create_manager.restype = ctypes.c_int
        lib.mer_destroy_manager.argtypes = [ctypes.c_void_p]
        lib.mer_destroy_manager.restype = ctypes.c_int
        lib.mer_result_to_string.argtypes = [ctypes.c_int]
        lib.mer_result_to_string.restype = ctypes.c_char_p

        # Create manager
        handle = ctypes.c_void_p()
        result = lib.mer_create_manager(ctypes.byref(handle), ctypes.byref(config), levels_array, 1)

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
        from madrona_escape_room.ctypes_bindings import MER_CompiledLevel, lib

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

        lib = ctypes.CDLL("./build/libmadrona_escape_room_c_api.so")

        class MER_CompiledLevel(ctypes.Structure):
            _fields_ = [
                ("num_tiles", ctypes.c_int32),
                ("max_entities", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("world_scale", ctypes.c_float),
                ("done_on_collide", ctypes.c_bool),
                ("object_ids", ctypes.c_int32 * 256),
                ("tile_x", ctypes.c_float * 256),
                ("tile_y", ctypes.c_float * 256),
            ]

        lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(MER_CompiledLevel)]
        lib.mer_validate_compiled_level.restype = ctypes.c_int

        for i, level_ascii in enumerate(test_levels):
            compiled = compile_ascii_level(level_ascii)

            # Convert to struct
            struct = MER_CompiledLevel()
            struct.num_tiles = compiled.num_tiles
            struct.max_entities = compiled.max_entities
            struct.width = compiled.width
            struct.height = compiled.height
            struct.world_scale = compiled.world_scale

            for j in range(256):
                struct.object_ids[j] = compiled.object_ids[j]
                struct.tile_x[j] = compiled.tile_x[j]
                struct.tile_y[j] = compiled.tile_y[j]

            # Validate with C API
            result = lib.mer_validate_compiled_level(ctypes.byref(struct))
            assert result == 0, f"Level {i} validation failed with code {result}"

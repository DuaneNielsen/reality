"""
Test C API struct validation and memory layout with pytest framework
"""

import ctypes

import pytest

from madrona_escape_room.level_compiler import compile_level


class TestCAPIStructValidation:
    """Test C API struct validation and direct C API calls"""

    def test_c_api_validation_function(self):
        """Test C API validation function directly"""
        level = """######
#S...#
#.C..#
######"""

        compiled = compile_level(level)

        # Load C API library directly
        lib = ctypes.CDLL("./build/libmadrona_escape_room_c_api.so")

        # Define struct manually - must match C++ CompiledLevel exactly
        class MER_CompiledLevel(ctypes.Structure):
            _fields_ = [
                ("num_tiles", ctypes.c_int32),
                ("max_entities", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("scale", ctypes.c_float),
                ("level_name", ctypes.c_char * 64),  # MAX_LEVEL_NAME_LENGTH
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
            ]

        # Populate struct
        level_struct = MER_CompiledLevel()
        level_struct.num_tiles = compiled["num_tiles"]
        level_struct.max_entities = compiled["max_entities"]
        level_struct.width = compiled["width"]
        level_struct.height = compiled["height"]
        level_struct.scale = compiled["scale"]
        level_struct.level_name = compiled["level_name"].encode("utf-8")
        level_struct.num_spawns = compiled["num_spawns"]

        for i in range(8):  # MAX_SPAWNS
            level_struct.spawn_x[i] = compiled["spawn_x"][i]
            level_struct.spawn_y[i] = compiled["spawn_y"][i]
            level_struct.spawn_facing[i] = compiled["spawn_facing"][i]

        for i in range(1024):  # MAX_TILES
            level_struct.object_ids[i] = compiled["object_ids"][i]
            level_struct.tile_x[i] = compiled["tile_x"][i]
            level_struct.tile_y[i] = compiled["tile_y"][i]
            level_struct.tile_z[i] = compiled["tile_z"][i]
            level_struct.tile_persistent[i] = compiled["tile_persistent"][i]
            level_struct.tile_render_only[i] = compiled["tile_render_only"][i]
            level_struct.tile_entity_type[i] = compiled["tile_entity_type"][i]
            level_struct.tile_response_type[i] = compiled["tile_response_type"][i]
            level_struct.tile_scale_x[i] = compiled["tile_scale_x"][i]
            level_struct.tile_scale_y[i] = compiled["tile_scale_y"][i]
            level_struct.tile_scale_z[i] = compiled["tile_scale_z"][i]
            level_struct.tile_rot_w[i] = compiled["tile_rot_w"][i]
            level_struct.tile_rot_x[i] = compiled["tile_rot_x"][i]
            level_struct.tile_rot_y[i] = compiled["tile_rot_y"][i]
            level_struct.tile_rot_z[i] = compiled["tile_rot_z"][i]
            # New randomization fields - default to 0 if not present
            level_struct.tile_rand_x[i] = compiled.get("tile_rand_x", [0.0] * 1024)[i]
            level_struct.tile_rand_y[i] = compiled.get("tile_rand_y", [0.0] * 1024)[i]
            level_struct.tile_rand_z[i] = compiled.get("tile_rand_z", [0.0] * 1024)[i]
            level_struct.tile_rand_rot_z[i] = compiled.get("tile_rand_rot_z", [0.0] * 1024)[i]

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

        compiled = compile_level(level)

        # Load library
        lib = ctypes.CDLL("./build/libmadrona_escape_room_c_api.so")

        # Define structs - must match C++ CompiledLevel exactly
        class MER_CompiledLevel(ctypes.Structure):
            _fields_ = [
                ("num_tiles", ctypes.c_int32),
                ("max_entities", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("scale", ctypes.c_float),
                ("level_name", ctypes.c_char * 64),  # MAX_LEVEL_NAME_LENGTH
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
        level_struct.num_tiles = compiled["num_tiles"]
        level_struct.max_entities = compiled["max_entities"]
        level_struct.width = compiled["width"]
        level_struct.height = compiled["height"]
        level_struct.scale = compiled["scale"]
        level_struct.level_name = compiled["level_name"].encode("utf-8")
        level_struct.num_spawns = compiled["num_spawns"]

        for i in range(8):  # MAX_SPAWNS
            level_struct.spawn_x[i] = compiled["spawn_x"][i]
            level_struct.spawn_y[i] = compiled["spawn_y"][i]
            level_struct.spawn_facing[i] = compiled["spawn_facing"][i]

        for i in range(1024):  # MAX_TILES
            level_struct.object_ids[i] = compiled["object_ids"][i]
            level_struct.tile_x[i] = compiled["tile_x"][i]
            level_struct.tile_y[i] = compiled["tile_y"][i]
            level_struct.tile_z[i] = compiled["tile_z"][i]
            level_struct.tile_persistent[i] = compiled["tile_persistent"][i]
            level_struct.tile_render_only[i] = compiled["tile_render_only"][i]
            level_struct.tile_entity_type[i] = compiled["tile_entity_type"][i]
            level_struct.tile_response_type[i] = compiled["tile_response_type"][i]
            level_struct.tile_scale_x[i] = compiled["tile_scale_x"][i]
            level_struct.tile_scale_y[i] = compiled["tile_scale_y"][i]
            level_struct.tile_scale_z[i] = compiled["tile_scale_z"][i]
            level_struct.tile_rot_w[i] = compiled["tile_rot_w"][i]
            level_struct.tile_rot_x[i] = compiled["tile_rot_x"][i]
            level_struct.tile_rot_y[i] = compiled["tile_rot_y"][i]
            level_struct.tile_rot_z[i] = compiled["tile_rot_z"][i]
            # New randomization fields - default to 0 if not present
            level_struct.tile_rand_x[i] = compiled.get("tile_rand_x", [0.0] * 1024)[i]
            level_struct.tile_rand_y[i] = compiled.get("tile_rand_y", [0.0] * 1024)[i]
            level_struct.tile_rand_z[i] = compiled.get("tile_rand_z", [0.0] * 1024)[i]
            level_struct.tile_rand_rot_z[i] = compiled.get("tile_rand_rot_z", [0.0] * 1024)[i]

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
        """Test struct memory layout and field access"""
        from madrona_escape_room.ctypes_bindings import MER_CompiledLevel

        # Create struct and test basic properties
        struct = MER_CompiledLevel()
        struct_size = ctypes.sizeof(struct)

        # Expected minimum size: 5 int32/float fields + 3 arrays of 256 elements
        expected_min_size = 4 * 5 + 256 * 4 * 3  # 20 + 3072 = 3092 bytes minimum
        assert (
            struct_size >= expected_min_size
        ), f"Struct too small: {struct_size} < {expected_min_size}"

        # Test field access
        struct.num_tiles = 42
        struct.max_entities = 100
        struct.width = 10
        struct.height = 8
        struct.scale = 1.5

        assert struct.num_tiles == 42
        assert struct.max_entities == 100
        assert struct.width == 10
        assert struct.height == 8
        assert abs(struct.scale - 1.5) < 0.001

        # Test array bounds
        struct.object_ids[0] = 1
        struct.object_ids[255] = 2
        struct.tile_x[0] = 3.14
        struct.tile_x[255] = 2.71
        struct.tile_y[0] = -1.0
        struct.tile_y[255] = 1.0

        assert struct.object_ids[0] == 1
        assert struct.object_ids[255] == 2
        assert abs(struct.tile_x[0] - 3.14) < 0.001
        assert abs(struct.tile_x[255] - 2.71) < 0.001
        assert abs(struct.tile_y[0] + 1.0) < 0.001
        assert abs(struct.tile_y[255] - 1.0) < 0.001

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
                ("scale", ctypes.c_float),
                ("object_ids", ctypes.c_int32 * 256),
                ("tile_x", ctypes.c_float * 256),
                ("tile_y", ctypes.c_float * 256),
            ]

        lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(MER_CompiledLevel)]
        lib.mer_validate_compiled_level.restype = ctypes.c_int

        for i, level_ascii in enumerate(test_levels):
            compiled = compile_level(level_ascii)

            # Convert to struct
            struct = MER_CompiledLevel()
            struct.num_tiles = compiled["num_tiles"]
            struct.max_entities = compiled["max_entities"]
            struct.width = compiled["width"]
            struct.height = compiled["height"]
            struct.scale = compiled["scale"]

            for j in range(256):
                struct.object_ids[j] = compiled["object_ids"][j]
                struct.tile_x[j] = compiled["tile_x"][j]
                struct.tile_y[j] = compiled["tile_y"][j]

            # Validate with C API
            result = lib.mer_validate_compiled_level(ctypes.byref(struct))
            assert result == 0, f"Level {i} validation failed with code {result}"

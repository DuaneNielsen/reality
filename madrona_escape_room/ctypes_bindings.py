"""
ctypes bindings for Madrona Escape Room C API
Direct replacement for CFFI bindings to resolve library loading issues
"""

import ctypes
import os
from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
)


# Find the shared library
def _find_library():
    """Find the Madrona Escape Room C API library"""
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Try different possible library names and locations
    possible_paths = [
        # First try the build directory (most likely location)
        os.path.join(module_dir, "..", "build", "libmadrona_escape_room_c_api.so"),
        # Then try the package directory
        os.path.join(module_dir, "libmadrona_escape_room_c_api.so"),
        # Then try relative to project root
        os.path.join(module_dir, "..", "..", "build", "libmadrona_escape_room_c_api.so"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    # If not found, try to load by name (system will search LD_LIBRARY_PATH)
    return "libmadrona_escape_room_c_api.so"


# Load the library
_lib_path = _find_library()


# Set up environment for bundled libraries (like the original code does)
def _setup_library_path():
    """Set up LD_LIBRARY_PATH to include directories with dependencies"""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(module_dir, "..", "build")

    paths_to_add = []
    if os.path.exists(build_dir):
        paths_to_add.append(os.path.abspath(build_dir))

    # Add package directory for bundled libraries
    paths_to_add.append(module_dir)

    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path in paths_to_add:
        if path not in current_path:
            if current_path:
                os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = path
            current_path = os.environ["LD_LIBRARY_PATH"]


_setup_library_path()

try:
    # Load our library normally
    lib = ctypes.CDLL(_lib_path)
except OSError as e:
    raise ImportError(
        f"Failed to load Madrona Escape Room C API library from '{_lib_path}'. "
        f"Make sure the library is built and in the correct location. "
        f"You may need to run: make -C build -j8\n"
        f"Original error: {e}"
    )

# Error codes enum
MER_SUCCESS = 0
MER_ERROR_NULL_POINTER = -1
MER_ERROR_INVALID_PARAMETER = -2
MER_ERROR_ALLOCATION_FAILED = -3
MER_ERROR_NOT_INITIALIZED = -4
MER_ERROR_CUDA_FAILURE = -5
MER_ERROR_FILE_NOT_FOUND = -6
MER_ERROR_INVALID_FILE = -7

# Execution modes enum
MER_EXEC_MODE_CPU = 0
MER_EXEC_MODE_CUDA = 1

# Tensor element types enum
MER_TENSOR_TYPE_UINT8 = 0
MER_TENSOR_TYPE_INT8 = 1
MER_TENSOR_TYPE_INT16 = 2
MER_TENSOR_TYPE_INT32 = 3
MER_TENSOR_TYPE_INT64 = 4
MER_TENSOR_TYPE_FLOAT16 = 5
MER_TENSOR_TYPE_FLOAT32 = 6

# Observation size constants
MER_SELF_OBSERVATION_SIZE = 5
MER_STEPS_REMAINING_SIZE = 1
MER_AGENT_ID_SIZE = 1
MER_TOTAL_OBSERVATION_SIZE = (
    MER_SELF_OBSERVATION_SIZE + MER_STEPS_REMAINING_SIZE + MER_AGENT_ID_SIZE
)

# Simulation parameter constants
MER_NUM_AGENTS = 1
MER_NUM_ROOMS = 1
MER_MAX_ENTITIES_PER_ROOM = 6
MER_EPISODE_LENGTH = 200

# Action constants - Move amount
MER_MOVE_STOP = 0
MER_MOVE_SLOW = 1
MER_MOVE_MEDIUM = 2
MER_MOVE_FAST = 3

# Action constants - Move angle (8 directions)
MER_MOVE_FORWARD = 0
MER_MOVE_FORWARD_RIGHT = 1
MER_MOVE_RIGHT = 2
MER_MOVE_BACKWARD_RIGHT = 3
MER_MOVE_BACKWARD = 4
MER_MOVE_BACKWARD_LEFT = 5
MER_MOVE_LEFT = 6
MER_MOVE_FORWARD_LEFT = 7

# Action constants - Rotation
MER_ROTATE_FAST_LEFT = 0
MER_ROTATE_SLOW_LEFT = 1
MER_ROTATE_NONE = 2
MER_ROTATE_SLOW_RIGHT = 3
MER_ROTATE_FAST_RIGHT = 4

# Manager handle type (opaque pointer)
MER_ManagerHandle = c_void_p


# Replay metadata structure
class MER_ReplayMetadata(Structure):
    _fields_ = [
        ("num_worlds", c_uint32),
        ("num_agents_per_world", c_uint32),
        ("num_steps", c_uint32),
        ("seed", c_uint32),
        ("sim_name", c_char * 64),
        ("timestamp", c_uint64),
    ]


# Get MAX_TILES from C API - now returns the current C++ constant
def _get_max_tiles():
    """Get MAX_TILES from C++ CompiledLevel::MAX_TILES via C API"""
    try:
        # Try to get it from the C API if library is loaded
        if "lib" in globals() and hasattr(lib, "mer_get_max_tiles"):
            lib.mer_get_max_tiles.restype = c_int32
            return lib.mer_get_max_tiles()
    except Exception:
        pass
    # Fallback to hardcoded value matching C++ CompiledLevel::MAX_TILES
    return 1024


def _get_max_spawns():
    """Get MAX_SPAWNS from C++ CompiledLevel::MAX_SPAWNS via C API"""
    try:
        # Try to get it from the C API if library is loaded
        if "lib" in globals() and hasattr(lib, "mer_get_max_spawns"):
            lib.mer_get_max_spawns.restype = c_int32
            return lib.mer_get_max_spawns()
    except Exception:
        pass
    # Fallback to hardcoded value matching C++ CompiledLevel::MAX_SPAWNS
    return 8


MAX_TILES = _get_max_tiles()
MAX_SPAWNS = _get_max_spawns()


class MER_CompiledLevel(Structure):
    _fields_ = [
        ("num_tiles", c_int32),
        ("max_entities", c_int32),
        ("width", c_int32),
        ("height", c_int32),
        ("scale", c_float),
        ("level_name", c_char * 64),  # MAX_LEVEL_NAME_LENGTH = 64
        # World boundaries in world units
        ("world_min_x", c_float),
        ("world_max_x", c_float),
        ("world_min_y", c_float),
        ("world_max_y", c_float),
        ("world_min_z", c_float),
        ("world_max_z", c_float),
        ("num_spawns", c_int32),
        ("spawn_x", c_float * 8),  # MAX_SPAWNS = 8
        ("spawn_y", c_float * 8),  # MAX_SPAWNS = 8
        ("spawn_facing", c_float * 8),  # MAX_SPAWNS = 8, agent facing angles in radians
        ("object_ids", c_int32 * MAX_TILES),
        ("tile_x", c_float * MAX_TILES),
        ("tile_y", c_float * MAX_TILES),
        ("tile_z", c_float * MAX_TILES),
        ("tile_persistent", c_bool * MAX_TILES),
        ("tile_render_only", c_bool * MAX_TILES),
        ("tile_entity_type", c_int32 * MAX_TILES),
        ("tile_response_type", c_int32 * MAX_TILES),
        ("tile_scale_x", c_float * MAX_TILES),
        ("tile_scale_y", c_float * MAX_TILES),
        ("tile_scale_z", c_float * MAX_TILES),
        ("tile_rot_w", c_float * MAX_TILES),
        ("tile_rot_x", c_float * MAX_TILES),
        ("tile_rot_y", c_float * MAX_TILES),
        ("tile_rot_z", c_float * MAX_TILES),
        ("tile_rand_x", c_float * MAX_TILES),
        ("tile_rand_y", c_float * MAX_TILES),
        ("tile_rand_z", c_float * MAX_TILES),
        ("tile_rand_rot_z", c_float * MAX_TILES),
    ]


# Manager configuration structure
class MER_ManagerConfig(Structure):
    _fields_ = [
        ("exec_mode", c_int),
        ("gpu_id", c_int),
        ("num_worlds", c_uint32),
        ("rand_seed", c_uint32),
        ("auto_reset", c_bool),
        ("enable_batch_renderer", c_bool),
        ("batch_render_view_width", c_uint32),
        ("batch_render_view_height", c_uint32),
    ]


# Tensor structure
class MER_Tensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("dimensions", c_int64 * 16),
        ("num_dimensions", c_int32),
        ("element_type", c_int),
        ("num_bytes", c_int64),
        ("gpu_id", c_int32),
    ]


# Function signatures
# Manager lifecycle functions
lib.mer_create_manager.argtypes = [
    POINTER(MER_ManagerHandle),
    POINTER(MER_ManagerConfig),
    POINTER(MER_CompiledLevel),  # Array of compiled levels (one per world), NULL for default
    c_uint32,  # Length of compiled_levels array
]
lib.mer_create_manager.restype = c_int

# Level validation functions
lib.mer_validate_compiled_level.argtypes = [POINTER(MER_CompiledLevel)]
lib.mer_validate_compiled_level.restype = c_int

lib.mer_destroy_manager.argtypes = [MER_ManagerHandle]
lib.mer_destroy_manager.restype = c_int

# Simulation functions
lib.mer_step.argtypes = [MER_ManagerHandle]
lib.mer_step.restype = c_int

# Tensor access functions
lib.mer_get_reset_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_reset_tensor.restype = c_int

lib.mer_get_action_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_action_tensor.restype = c_int

lib.mer_get_reward_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_reward_tensor.restype = c_int

lib.mer_get_done_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_done_tensor.restype = c_int

lib.mer_get_self_observation_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_self_observation_tensor.restype = c_int

lib.mer_get_steps_remaining_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_steps_remaining_tensor.restype = c_int

lib.mer_get_progress_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_progress_tensor.restype = c_int

lib.mer_get_rgb_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_rgb_tensor.restype = c_int

lib.mer_get_depth_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_depth_tensor.restype = c_int

# Control functions (for viewer)
lib.mer_trigger_reset.argtypes = [MER_ManagerHandle, c_int32]
lib.mer_trigger_reset.restype = c_int

lib.mer_set_action.argtypes = [MER_ManagerHandle, c_int32, c_int32, c_int32, c_int32]
lib.mer_set_action.restype = c_int

# Trajectory logging
lib.mer_enable_trajectory_logging.argtypes = [
    MER_ManagerHandle,
    c_int32,
    c_int32,
    c_char_p,
]
lib.mer_enable_trajectory_logging.restype = c_int

lib.mer_disable_trajectory_logging.argtypes = [MER_ManagerHandle]
lib.mer_disable_trajectory_logging.restype = c_int

# Recording functionality
lib.mer_start_recording.argtypes = [MER_ManagerHandle, c_char_p, c_uint32]
lib.mer_start_recording.restype = c_int

lib.mer_stop_recording.argtypes = [MER_ManagerHandle]
lib.mer_stop_recording.restype = c_int

lib.mer_is_recording.argtypes = [MER_ManagerHandle, POINTER(c_bool)]
lib.mer_is_recording.restype = c_int

# Replay metadata reading (static function)
lib.mer_read_replay_metadata.argtypes = [c_char_p, POINTER(MER_ReplayMetadata)]
lib.mer_read_replay_metadata.restype = c_int

# Replay functionality
lib.mer_load_replay.argtypes = [MER_ManagerHandle, c_char_p]
lib.mer_load_replay.restype = c_int

lib.mer_has_replay.argtypes = [MER_ManagerHandle, POINTER(c_bool)]
lib.mer_has_replay.restype = c_int

lib.mer_replay_step.argtypes = [MER_ManagerHandle, POINTER(c_bool)]
lib.mer_replay_step.restype = c_int

lib.mer_get_replay_step_count.argtypes = [
    MER_ManagerHandle,
    POINTER(c_uint32),
    POINTER(c_uint32),
]
lib.mer_get_replay_step_count.restype = c_int

# Utility functions
lib.mer_result_to_string.argtypes = [c_int]
lib.mer_result_to_string.restype = c_char_p

# Binary I/O functions
lib.mer_write_compiled_level.argtypes = [c_char_p, POINTER(MER_CompiledLevel)]
lib.mer_write_compiled_level.restype = c_int

lib.mer_read_compiled_level.argtypes = [c_char_p, POINTER(MER_CompiledLevel)]
lib.mer_read_compiled_level.restype = c_int


class CTypesLib:
    """Wrapper class to provide CFFI-like interface for easier migration"""

    def __init__(self):
        self.lib = lib

    def new(self, type_name):
        """Create a new ctypes object similar to ffi.new()"""
        if type_name == "MER_ManagerConfig*":
            return POINTER(MER_ManagerConfig)(MER_ManagerConfig())
        elif type_name == "MER_ManagerHandle*":
            return POINTER(MER_ManagerHandle)(MER_ManagerHandle())
        elif type_name == "MER_Tensor*":
            return POINTER(MER_Tensor)(MER_Tensor())
        elif type_name == "MER_CompiledLevel*":
            return POINTER(MER_CompiledLevel)(MER_CompiledLevel())
        else:
            raise ValueError(f"Unknown type: {type_name}")

    def string(self, c_char_ptr):
        """Convert C string to Python string similar to ffi.string()"""
        if c_char_ptr:
            return c_char_ptr.decode("utf-8")
        return ""

    def cast(self, typename, value):
        """Cast value to pointer type similar to ffi.cast()"""
        if typename == "uintptr_t":
            return ctypes.cast(value, ctypes.c_void_p).value
        else:
            raise ValueError(f"Unknown cast type: {typename}")

    def buffer(self, data_ptr, size):
        """Create a buffer from pointer and size similar to ffi.buffer()"""
        # Convert void pointer to array of bytes
        ArrayType = ctypes.c_uint8 * size
        array_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ArrayType))
        return array_ptr.contents

    @property
    def NULL(self):
        """NULL pointer"""
        return None


# Helper functions for compiled level integration
def dict_to_compiled_level(compiled_dict):
    """
    Convert compiled level dictionary to MER_CompiledLevel ctypes structure.

    The compiled_dict contains arrays sized exactly for the level's dimensions
    (width × height), but the ctypes structure uses fixed MAX_TILES arrays.
    We copy the actual data and zero-fill the remaining slots.

    Args:
        compiled_dict: Output from level_compiler.compile_level()

    Returns:
        MER_CompiledLevel: ctypes structure ready for C API
    """
    level = MER_CompiledLevel()
    level.num_tiles = compiled_dict["num_tiles"]
    level.max_entities = compiled_dict["max_entities"]
    level.width = compiled_dict["width"]
    level.height = compiled_dict["height"]
    level.scale = compiled_dict["scale"]

    # Copy level name (ensure it's null-terminated and fits in 64 bytes)
    level_name = compiled_dict.get("level_name", "unknown_level")
    level_name_bytes = level_name.encode("utf-8")[:63]  # Leave room for null terminator
    level.level_name = level_name_bytes

    # Copy world boundaries
    level.world_min_x = compiled_dict.get("world_min_x", -20.0)
    level.world_max_x = compiled_dict.get("world_max_x", 20.0)
    level.world_min_y = compiled_dict.get("world_min_y", -20.0)
    level.world_max_y = compiled_dict.get("world_max_y", 20.0)
    level.world_min_z = compiled_dict.get("world_min_z", 0.0)
    level.world_max_z = compiled_dict.get("world_max_z", 25.0)

    # Copy spawn data
    level.num_spawns = compiled_dict["num_spawns"]
    for i in range(8):  # MAX_SPAWNS
        level.spawn_x[i] = compiled_dict["spawn_x"][i]
        level.spawn_y[i] = compiled_dict["spawn_y"][i]
        level.spawn_facing[i] = compiled_dict.get("spawn_facing", [0.0] * 8)[i]

    # Get the actual array size for this level
    array_size = compiled_dict["array_size"]

    # Validate that the compiled level data fits in our fixed-size arrays
    if array_size > MAX_TILES:
        raise ValueError(f"Level too large: needs {array_size} tiles but MAX_TILES is {MAX_TILES}")

    # Copy actual data from compiler-calculated arrays
    for i in range(array_size):
        level.object_ids[i] = compiled_dict["object_ids"][i]
        level.tile_x[i] = compiled_dict["tile_x"][i]
        level.tile_y[i] = compiled_dict["tile_y"][i]
        level.tile_z[i] = compiled_dict["tile_z"][i]
        level.tile_persistent[i] = compiled_dict["tile_persistent"][i]
        level.tile_render_only[i] = compiled_dict["tile_render_only"][i]
        level.tile_entity_type[i] = compiled_dict["tile_entity_type"][i]
        level.tile_response_type[i] = compiled_dict["tile_response_type"][i]

        # Transform data
        level.tile_scale_x[i] = compiled_dict["tile_scale_x"][i]
        level.tile_scale_y[i] = compiled_dict["tile_scale_y"][i]
        level.tile_scale_z[i] = compiled_dict["tile_scale_z"][i]
        level.tile_rot_w[i] = compiled_dict["tile_rot_w"][i]
        level.tile_rot_x[i] = compiled_dict["tile_rot_x"][i]
        level.tile_rot_y[i] = compiled_dict["tile_rot_y"][i]
        level.tile_rot_z[i] = compiled_dict["tile_rot_z"][i]

        # Randomization data (optional - default to 0 if not present)
        level.tile_rand_x[i] = (
            compiled_dict.get("tile_rand_x", [0.0] * array_size)[i]
            if "tile_rand_x" in compiled_dict
            else 0.0
        )
        level.tile_rand_y[i] = (
            compiled_dict.get("tile_rand_y", [0.0] * array_size)[i]
            if "tile_rand_y" in compiled_dict
            else 0.0
        )
        level.tile_rand_z[i] = (
            compiled_dict.get("tile_rand_z", [0.0] * array_size)[i]
            if "tile_rand_z" in compiled_dict
            else 0.0
        )
        level.tile_rand_rot_z[i] = (
            compiled_dict.get("tile_rand_rot_z", [0.0] * array_size)[i]
            if "tile_rand_rot_z" in compiled_dict
            else 0.0
        )

    # Zero-fill remaining slots (important for deterministic behavior)
    for i in range(array_size, MAX_TILES):
        level.object_ids[i] = 0  # TILE_EMPTY
        level.tile_x[i] = 0.0
        level.tile_y[i] = 0.0
        level.tile_z[i] = 0.0
        level.tile_persistent[i] = False
        level.tile_render_only[i] = False
        level.tile_entity_type[i] = 0
        level.tile_response_type[i] = 0
        level.tile_scale_x[i] = 1.0
        level.tile_scale_y[i] = 1.0
        level.tile_scale_z[i] = 1.0
        level.tile_rot_w[i] = 1.0
        level.tile_rot_x[i] = 0.0
        level.tile_rot_y[i] = 0.0
        level.tile_rot_z[i] = 0.0
        level.tile_rand_x[i] = 0.0
        level.tile_rand_y[i] = 0.0
        level.tile_rand_z[i] = 0.0
        level.tile_rand_rot_z[i] = 0.0

    return level


def compiled_level_to_dict(level):
    """
    Convert MER_CompiledLevel ctypes structure to dictionary.
    Reverse of dict_to_compiled_level.

    Args:
        level: MER_CompiledLevel ctypes structure

    Returns:
        Dict matching compile_level() output format
    """
    # Get actual array size for this level
    array_size = level.width * level.height

    return {
        # Header fields
        "num_tiles": level.num_tiles,
        "max_entities": level.max_entities,
        "width": level.width,
        "height": level.height,
        "scale": level.scale,
        "level_name": level.level_name.decode("utf-8").rstrip("\x00"),
        # World boundaries
        "world_min_x": level.world_min_x,
        "world_max_x": level.world_max_x,
        "world_min_y": level.world_min_y,
        "world_max_y": level.world_max_y,
        "world_min_z": level.world_min_z,
        "world_max_z": level.world_max_z,
        # Spawn data
        "num_spawns": level.num_spawns,
        "spawn_x": list(level.spawn_x),
        "spawn_y": list(level.spawn_y),
        "spawn_facing": list(level.spawn_facing),
        # Tile arrays (only copy actual data, not padding)
        "object_ids": list(level.object_ids[:array_size]),
        "tile_x": list(level.tile_x[:array_size]),
        "tile_y": list(level.tile_y[:array_size]),
        "tile_z": list(level.tile_z[:array_size]),
        "tile_persistent": list(level.tile_persistent[:array_size]),
        "tile_render_only": list(level.tile_render_only[:array_size]),
        "tile_entity_type": list(level.tile_entity_type[:array_size]),
        "tile_response_type": list(level.tile_response_type[:array_size]),
        # Transform arrays
        "tile_scale_x": list(level.tile_scale_x[:array_size]),
        "tile_scale_y": list(level.tile_scale_y[:array_size]),
        "tile_scale_z": list(level.tile_scale_z[:array_size]),
        "tile_rot_w": list(level.tile_rot_w[:array_size]),
        "tile_rot_x": list(level.tile_rot_x[:array_size]),
        "tile_rot_y": list(level.tile_rot_y[:array_size]),
        "tile_rot_z": list(level.tile_rot_z[:array_size]),
        # Randomization arrays
        "tile_rand_x": list(level.tile_rand_x[:array_size]),
        "tile_rand_y": list(level.tile_rand_y[:array_size]),
        "tile_rand_z": list(level.tile_rand_z[:array_size]),
        "tile_rand_rot_z": list(level.tile_rand_rot_z[:array_size]),
        # Metadata
        "array_size": array_size,
    }


def validate_compiled_level_ctypes(level):
    """
    Validate compiled level using C API validation function.

    Args:
        level: MER_CompiledLevel structure

    Raises:
        ValueError: If validation fails
    """
    result = lib.mer_validate_compiled_level(ctypes.byref(level))
    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result)
        if error_msg:
            error_msg = error_msg.decode("utf-8")
        else:
            error_msg = f"Error code {result}"
        raise ValueError(f"Compiled level validation failed: {error_msg}")


def create_compiled_levels_array(compiled_level_dicts):
    """
    Create ctypes array of compiled levels from list of dictionaries.

    Args:
        compiled_level_dicts: List of compiled level dictionaries

    Returns:
        tuple: (ctypes array, array length)
    """
    if not compiled_level_dicts:
        return None, 0

    # Convert each dict to ctypes structure
    levels = []
    for compiled_dict in compiled_level_dicts:
        level = dict_to_compiled_level(compiled_dict)
        validate_compiled_level_ctypes(level)
        levels.append(level)

    # Create ctypes array
    ArrayType = MER_CompiledLevel * len(levels)
    levels_array = ArrayType(*levels)

    return levels_array, len(levels)


def get_physics_assets_list():
    """Get list of physics asset names."""
    if "lib" in globals() and hasattr(lib, "mer_get_physics_assets_count"):
        lib.mer_get_physics_assets_count.restype = c_int32
        lib.mer_get_physics_asset_name.restype = c_char_p
        lib.mer_get_physics_asset_name.argtypes = [c_int32]

        count = lib.mer_get_physics_assets_count()
        names = []
        for i in range(count):
            name = lib.mer_get_physics_asset_name(i)
            if name:
                names.append(name.decode("utf-8"))
        return names
    return []


def get_render_assets_list():
    """Get list of render asset names."""
    if "lib" in globals() and hasattr(lib, "mer_get_render_assets_count"):
        lib.mer_get_render_assets_count.restype = c_int32
        lib.mer_get_render_asset_name.restype = c_char_p
        lib.mer_get_render_asset_name.argtypes = [c_int32]

        count = lib.mer_get_render_assets_count()
        names = []
        for i in range(count):
            name = lib.mer_get_render_asset_name(i)
            if name:
                names.append(name.decode("utf-8"))
        return names
    return []


def get_physics_asset_object_id(name):
    """Get object ID for a physics asset by name."""
    if "lib" in globals() and hasattr(lib, "mer_get_physics_asset_object_id"):
        lib.mer_get_physics_asset_object_id.restype = c_int32
        lib.mer_get_physics_asset_object_id.argtypes = [c_char_p]
        return lib.mer_get_physics_asset_object_id(name.encode("utf-8"))
    return -1


def get_render_asset_object_id(name):
    """Get object ID for a render asset by name."""
    if "lib" in globals() and hasattr(lib, "mer_get_render_asset_object_id"):
        lib.mer_get_render_asset_object_id.restype = c_int32
        lib.mer_get_render_asset_object_id.argtypes = [c_char_p]
        return lib.mer_get_render_asset_object_id(name.encode("utf-8"))
    return -1


# Create instances for compatibility
ctypes_lib = CTypesLib()

# Export the main objects that will replace ffi and lib
ffi = ctypes_lib
lib = ctypes_lib.lib

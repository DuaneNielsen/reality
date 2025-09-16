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
    c_size_t,
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

# Import all constants from generated file
from .generated_constants import action, consts

# Create convenience aliases for nested namespaces
math = consts.math
physics = consts.physics
rendering = consts.rendering

# These constants are now available from generated_constants.py:
# - Error codes (MER_SUCCESS, MER_ERROR_*, etc) from madEscape::Result enum
# - Execution modes (MER_EXEC_MODE_CPU/CUDA) from madrona::ExecMode
# The C API uses these enums directly via void pointers

# Manager handle type (opaque pointer)
MER_ManagerHandle = c_void_p


# Import constants
from .generated_constants import limits

# Import API boundary structs - these are the ONLY structs we use now
# ECS components (Action, SelfObservation, Done, Reward, Progress, StepsTaken)
# are accessed through tensor exports, not direct struct manipulation
from .generated_dataclasses import (
    CompiledLevel,
    ManagerConfig,
    ReplayMetadata,
    to_ctypes,
)


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


# Helper function to create manager with proper level array handling
def create_manager_with_levels(handle_ptr, config, compiled_levels):
    """
    Wrapper to properly handle compiled levels list when creating manager.

    Args:
        handle_ptr: Pointer to MER_ManagerHandle
        config: ManagerConfig dataclass (not a pointer)
        compiled_levels: None, single CompiledLevel (dataclass), or list of CompiledLevel

    Returns:
        Tuple of (result_code, c_config, levels_array) where c_config and
        levels_array must be kept alive
    """
    # Convert config dataclass to ctypes
    c_config = config.to_ctype()
    from ctypes import pointer

    config_ptr = pointer(c_config)

    if compiled_levels is None:
        # No levels - use default
        return lib.mer_create_manager(handle_ptr, config_ptr, None, 0), c_config, None

    # Convert single level to list
    if not isinstance(compiled_levels, list):
        compiled_levels = [compiled_levels]

    if not compiled_levels:
        # Empty list - use default
        return lib.mer_create_manager(handle_ptr, config_ptr, None, 0), c_config, None

    # Convert dataclasses to ctypes
    ctypes_levels = [level.to_ctype() for level in compiled_levels]

    # Get the ctypes class from the first converted level
    CTypesCompiledLevel = type(ctypes_levels[0])

    # Create ctypes array
    num_levels = len(ctypes_levels)
    ArrayType = CTypesCompiledLevel * num_levels
    levels_array = ArrayType()

    # Copy each level into the array
    for i, level in enumerate(ctypes_levels):
        levels_array[i] = level

    print(f"[DEBUG ctypes_bindings] Passing {num_levels} levels to C API")
    print(
        f"[DEBUG ctypes_bindings] First level spawn: "
        f"({levels_array[0].spawn_x[0]}, {levels_array[0].spawn_y[0]})"
    )

    # Pass array pointer to C API
    from ctypes import POINTER, c_void_p, cast

    levels_ptr = cast(levels_array, c_void_p)

    result = lib.mer_create_manager(handle_ptr, config_ptr, levels_ptr, num_levels)

    # Return result, config, and array (all must be kept alive!)
    return result, c_config, levels_array


# Function signatures
# Manager lifecycle functions
lib.mer_create_manager.argtypes = [
    POINTER(MER_ManagerHandle),
    c_void_p,  # Direct ManagerConfig pointer (now auto-generated)
    c_void_p,  # Direct CompiledLevel pointer from Python (NULL for default)
    c_uint32,  # Length of compiled_levels array
]
lib.mer_create_manager.restype = c_int

lib.mer_create_manager_from_replay.argtypes = [
    POINTER(MER_ManagerHandle),
    c_char_p,  # filepath
    c_int32,  # exec_mode (0=CPU, 1=CUDA)
    c_int32,  # gpu_id
    c_bool,  # enable_batch_renderer
]
lib.mer_create_manager_from_replay.restype = c_int

# Level validation functions
lib.mer_validate_compiled_level.argtypes = [c_void_p]  # Direct CompiledLevel pointer
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

lib.mer_get_termination_reason_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_termination_reason_tensor.restype = c_int

lib.mer_get_self_observation_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_self_observation_tensor.restype = c_int

lib.mer_get_compass_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_compass_tensor.restype = c_int

lib.mer_get_lidar_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_lidar_tensor.restype = c_int

lib.mer_get_steps_taken_tensor.argtypes = [MER_ManagerHandle, POINTER(MER_Tensor)]
lib.mer_get_steps_taken_tensor.restype = c_int

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
lib.mer_start_recording.argtypes = [MER_ManagerHandle, c_char_p]
lib.mer_start_recording.restype = c_int

lib.mer_stop_recording.argtypes = [MER_ManagerHandle]
lib.mer_stop_recording.restype = c_int

lib.mer_is_recording.argtypes = [MER_ManagerHandle, POINTER(c_bool)]
lib.mer_is_recording.restype = c_int

# Replay metadata reading (static function)
lib.mer_read_replay_metadata.argtypes = [c_char_p, c_void_p]  # Direct ReplayMetadata pointer
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

# Binary I/O functions (unified format)
lib.mer_write_compiled_levels.argtypes = [
    c_char_p,
    c_void_p,
    c_uint32,
]  # filepath, compiled_levels, num_levels
lib.mer_write_compiled_levels.restype = c_int

lib.mer_read_compiled_levels.argtypes = [
    c_char_p,
    c_void_p,
    POINTER(c_uint32),
    c_uint32,
]  # filepath, out_levels, out_num_levels, max_levels
lib.mer_read_compiled_levels.restype = c_int

# Get CompiledLevel size for validation
lib.mer_get_compiled_level_size.argtypes = []
lib.mer_get_compiled_level_size.restype = c_size_t


def validate_compiled_level(level):
    """
    Validate compiled level using C API validation function.

    Args:
        level: CompiledLevel dataclass

    Raises:
        ValueError: If validation fails
    """
    # Convert dataclass to ctypes if needed
    if hasattr(level, "to_ctype"):
        c_level = level.to_ctype()
    else:
        c_level = level

    result = lib.mer_validate_compiled_level(ctypes.byref(c_level))
    if result != 0:  # Success = 0
        error_msg = lib.mer_result_to_string(result)
        if error_msg:
            error_msg = error_msg.decode("utf-8")
        else:
            error_msg = f"Error code {result}"
        raise ValueError(f"Compiled level validation failed: {error_msg}")


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


# Export CompiledLevel as MER_CompiledLevel for backward compatibility with tests
MER_CompiledLevel = CompiledLevel

"""
ctypes bindings for Madrona Escape Room C API
Direct replacement for CFFI bindings to resolve library loading issues
"""

import os
import ctypes
from ctypes import Structure, c_int, c_int32, c_int64, c_uint32, c_bool, c_void_p, c_char_p, POINTER

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
    
    current_path = os.environ.get('LD_LIBRARY_PATH', '')
    for path in paths_to_add:
        if path not in current_path:
            if current_path:
                os.environ['LD_LIBRARY_PATH'] = f"{path}:{current_path}"
            else:
                os.environ['LD_LIBRARY_PATH'] = path
            current_path = os.environ['LD_LIBRARY_PATH']

_setup_library_path()

try:
    # Load our library normally
    lib = ctypes.CDLL(_lib_path)
except OSError as e:
    raise ImportError(
        f"Failed to load Madrona Escape Room C API library from '{_lib_path}'. "
        f"Make sure the library is built and in the correct location. "
        f"You may need to run: make -C build -j$(nproc)\n"
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
MER_TOTAL_OBSERVATION_SIZE = MER_SELF_OBSERVATION_SIZE + MER_STEPS_REMAINING_SIZE + MER_AGENT_ID_SIZE

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
lib.mer_create_manager.argtypes = [POINTER(MER_ManagerHandle), POINTER(MER_ManagerConfig)]
lib.mer_create_manager.restype = c_int

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
lib.mer_enable_trajectory_logging.argtypes = [MER_ManagerHandle, c_int32, c_int32, c_char_p]
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

# Replay functionality
lib.mer_load_replay.argtypes = [MER_ManagerHandle, c_char_p]
lib.mer_load_replay.restype = c_int

lib.mer_has_replay.argtypes = [MER_ManagerHandle, POINTER(c_bool)]
lib.mer_has_replay.restype = c_int

lib.mer_replay_step.argtypes = [MER_ManagerHandle, POINTER(c_bool)]
lib.mer_replay_step.restype = c_int

lib.mer_get_replay_step_count.argtypes = [MER_ManagerHandle, POINTER(c_uint32), POINTER(c_uint32)]
lib.mer_get_replay_step_count.restype = c_int

# Utility functions
lib.mer_result_to_string.argtypes = [c_int]
lib.mer_result_to_string.restype = c_char_p


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
        else:
            raise ValueError(f"Unknown type: {type_name}")
    
    def string(self, c_char_ptr):
        """Convert C string to Python string similar to ffi.string()"""
        if c_char_ptr:
            return c_char_ptr.decode('utf-8')
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


# Create instances for compatibility
ctypes_lib = CTypesLib()

# Export the main objects that will replace ffi and lib
ffi = ctypes_lib
lib = ctypes_lib.lib
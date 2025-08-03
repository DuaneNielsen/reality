"""
Madrona Escape Room Python bindings using ctypes
Provides the same API as the original nanobind version
"""

import numpy as np
from enum import IntEnum
import os
import sys

# Set up environment for bundled libraries before importing ctypes module
_module_dir = os.path.dirname(os.path.abspath(__file__))
_current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if _module_dir not in _current_ld_path:
    os.environ['LD_LIBRARY_PATH'] = f"{_module_dir}:{_current_ld_path}"

# Import the ctypes module
try:
    from .ctypes_bindings import ffi, lib
except ImportError as e:
    raise ImportError(
        f"ctypes bindings failed to load library. "
        f"Please ensure the C API library is built: make -C build -j$(nproc)\n"
        f"Original error: {e}"
    )

# Import constants from ctypes_bindings
from .ctypes_bindings import (
    MER_SUCCESS, MER_SELF_OBSERVATION_SIZE, MER_STEPS_REMAINING_SIZE, 
    MER_AGENT_ID_SIZE, MER_TOTAL_OBSERVATION_SIZE, MER_NUM_AGENTS,
    MER_MOVE_STOP, MER_MOVE_SLOW, MER_MOVE_MEDIUM, MER_MOVE_FAST,
    MER_MOVE_FORWARD, MER_MOVE_FORWARD_RIGHT, MER_MOVE_RIGHT, MER_MOVE_BACKWARD_RIGHT,
    MER_MOVE_BACKWARD, MER_MOVE_BACKWARD_LEFT, MER_MOVE_LEFT, MER_MOVE_FORWARD_LEFT,
    MER_ROTATE_FAST_LEFT, MER_ROTATE_SLOW_LEFT, MER_ROTATE_NONE, 
    MER_ROTATE_SLOW_RIGHT, MER_ROTATE_FAST_RIGHT,
    MER_TENSOR_TYPE_UINT8, MER_TENSOR_TYPE_INT8, MER_TENSOR_TYPE_INT16,
    MER_TENSOR_TYPE_INT32, MER_TENSOR_TYPE_INT64, MER_TENSOR_TYPE_FLOAT16,
    MER_TENSOR_TYPE_FLOAT32, MER_EXEC_MODE_CPU, MER_EXEC_MODE_CUDA
)

# Export constants
SELF_OBSERVATION_SIZE = MER_SELF_OBSERVATION_SIZE
STEPS_REMAINING_SIZE = MER_STEPS_REMAINING_SIZE
AGENT_ID_SIZE = MER_AGENT_ID_SIZE
TOTAL_OBSERVATION_SIZE = MER_TOTAL_OBSERVATION_SIZE
NUM_AGENTS = MER_NUM_AGENTS

# Create action submodule equivalents
class action:
    class move_amount:
        STOP = MER_MOVE_STOP
        SLOW = MER_MOVE_SLOW
        MEDIUM = MER_MOVE_MEDIUM
        FAST = MER_MOVE_FAST
    
    class move_angle:
        FORWARD = MER_MOVE_FORWARD
        FORWARD_RIGHT = MER_MOVE_FORWARD_RIGHT
        RIGHT = MER_MOVE_RIGHT
        BACKWARD_RIGHT = MER_MOVE_BACKWARD_RIGHT
        BACKWARD = MER_MOVE_BACKWARD
        BACKWARD_LEFT = MER_MOVE_BACKWARD_LEFT
        LEFT = MER_MOVE_LEFT
        FORWARD_LEFT = MER_MOVE_FORWARD_LEFT
    
    class rotate:
        FAST_LEFT = MER_ROTATE_FAST_LEFT
        SLOW_LEFT = MER_ROTATE_SLOW_LEFT
        NONE = MER_ROTATE_NONE
        SLOW_RIGHT = MER_ROTATE_SLOW_RIGHT
        FAST_RIGHT = MER_ROTATE_FAST_RIGHT

# Madrona submodule for compatibility
class madrona:
    class ExecMode(IntEnum):
        CPU = MER_EXEC_MODE_CPU
        CUDA = MER_EXEC_MODE_CUDA
    
    class Tensor:
        """Wrapper for tensor data from C API - provides zero-copy access"""
        def __init__(self, c_tensor):
            # Store the raw C tensor struct for direct access
            self._tensor = c_tensor
            self._element_type = c_tensor.element_type
            self._gpu_id = c_tensor.gpu_id
        
        def devicePtr(self):
            """Get raw device pointer (for compatibility)"""
            return self._tensor.data
        
        def type(self):
            """Get element type"""
            return self._element_type
        
        def isOnGPU(self):
            """Check if tensor is on GPU"""
            return self._gpu_id != -1
        
        def gpuID(self):
            """Get GPU ID (-1 for CPU)"""
            return self._gpu_id
        
        def numDims(self):
            """Get number of dimensions"""
            return self._tensor.num_dimensions
        
        def dims(self):
            """Get dimensions array"""
            return [self._tensor.dimensions[i] for i in range(self._tensor.num_dimensions)]
        
        def numBytesPerItem(self):
            """Get number of bytes per item"""
            size_map = {
                MER_TENSOR_TYPE_UINT8: 1,
                MER_TENSOR_TYPE_INT8: 1,
                MER_TENSOR_TYPE_INT16: 2,
                MER_TENSOR_TYPE_INT32: 4,
                MER_TENSOR_TYPE_INT64: 8,
                MER_TENSOR_TYPE_FLOAT16: 2,
                MER_TENSOR_TYPE_FLOAT32: 4,
            }
            return size_map.get(self._element_type, 4)
        
        def to_numpy(self):
            """Convert to numpy array (zero-copy view, CPU only)"""
            import numpy as np
            import ctypes
            
            # Check if tensor is on GPU
            if self.isOnGPU():
                raise RuntimeError(
                    "Cannot convert GPU tensor to numpy array directly. "
                    "Use to_torch() or torch.from_dlpack() for GPU tensors."
                )
            
            # Map element types to numpy dtypes
            dtype_map = {
                MER_TENSOR_TYPE_UINT8: np.uint8,
                MER_TENSOR_TYPE_INT8: np.int8,
                MER_TENSOR_TYPE_INT16: np.int16,
                MER_TENSOR_TYPE_INT32: np.int32,
                MER_TENSOR_TYPE_INT64: np.int64,
                MER_TENSOR_TYPE_FLOAT16: np.float16,
                MER_TENSOR_TYPE_FLOAT32: np.float32,
            }
            
            np_dtype = dtype_map[self._element_type]
            shape = self.dims()
            
            # Create a numpy array view of the data without copying
            # Convert void pointer to array of bytes for ctypes
            ArrayType = ctypes.c_uint8 * self._tensor.num_bytes
            array_ptr = ctypes.cast(self._tensor.data, ctypes.POINTER(ArrayType))
            buffer = array_ptr.contents
            
            array = np.frombuffer(buffer, dtype=np_dtype).reshape(shape)
            
            # The array is already a view, not a copy
            array.flags.writeable = True
            return array
        
        def to_torch(self):
            """Convert to PyTorch tensor (zero-copy when possible)"""
            import torch
            
            # For GPU tensors, prioritize DLPack protocol for zero-copy
            if self.isOnGPU():
                try:
                    # Try DLPack first (this is the main goal of our work)
                    return torch.from_dlpack(self)
                except Exception as e:
                    # Fallback: not ideal but for compatibility, create a zero tensor
                    import warnings
                    warnings.warn(
                        f"GPU DLPack conversion failed ({e}), creating placeholder tensor. "
                        "Real data access not available without DLPack support.",
                        UserWarning
                    )
                    # Map element types to PyTorch dtypes
                    dtype_map = {
                        MER_TENSOR_TYPE_UINT8: torch.uint8,
                        MER_TENSOR_TYPE_INT8: torch.int8,
                        MER_TENSOR_TYPE_INT16: torch.int16,
                        MER_TENSOR_TYPE_INT32: torch.int32,
                        MER_TENSOR_TYPE_INT64: torch.int64,
                        MER_TENSOR_TYPE_FLOAT16: torch.float16,
                        MER_TENSOR_TYPE_FLOAT32: torch.float32,
                    }
                    torch_dtype = dtype_map[self._element_type]
                    shape = self.dims()
                    device = torch.device(f'cuda:{self._gpu_id}')
                    return torch.zeros(shape, dtype=torch_dtype, device=device)
            else:
                # For CPU tensors, use numpy conversion (zero-copy)
                numpy_view = self.to_numpy()
                return torch.from_numpy(numpy_view)
        
        def __dlpack__(self, stream=None):
            """Create DLPack capsule for PyTorch consumption"""
            import ctypes
            
            try:
                import _madrona_escape_room_dlpack as dlpack_ext
            except ImportError:
                # Fallback to to_torch() if DLPack extension not available
                import warnings
                warnings.warn(
                    "DLPack extension not available, falling back to to_torch(). "
                    "Install DLPack extension for optimal performance.",
                    UserWarning
                )
                return self.to_torch()
            
            # Get tensor parameters
            data_ptr = ctypes.cast(self._tensor.data, ctypes.c_void_p).value
            shape = self.dims()
            dtype = self._element_type
            device_type = 2 if self.isOnGPU() else 1  # 2=CUDA, 1=CPU
            device_id = self.gpuID() if self.isOnGPU() else 0
            
            # Create DLPack capsule
            return dlpack_ext.create_dlpack_capsule(
                data_ptr, shape, dtype, device_type, device_id
            )
        
        def __dlpack_device__(self):
            """Return device info tuple for DLPack protocol"""
            try:
                import _madrona_escape_room_dlpack as dlpack_ext
            except ImportError:
                # Return device info directly if extension not available
                device_type = 2 if self.isOnGPU() else 1  # 2=CUDA, 1=CPU
                device_id = self.gpuID() if self.isOnGPU() else 0
                return (device_type, device_id)
            
            device_type = 2 if self.isOnGPU() else 1  # 2=CUDA, 1=CPU
            device_id = self.gpuID() if self.isOnGPU() else 0
            
            return dlpack_ext.get_dlpack_device(device_type, device_id)
        
        @property
        def shape(self):
            return tuple(self.dims())
        
        @property
        def dtype(self):
            # Return numpy dtype for compatibility
            import numpy as np
            dtype_map = {
                MER_TENSOR_TYPE_UINT8: np.uint8,
                MER_TENSOR_TYPE_INT8: np.int8,
                MER_TENSOR_TYPE_INT16: np.int16,
                MER_TENSOR_TYPE_INT32: np.int32,
                MER_TENSOR_TYPE_INT64: np.int64,
                MER_TENSOR_TYPE_FLOAT16: np.float16,
                MER_TENSOR_TYPE_FLOAT32: np.float32,
            }
            return dtype_map.get(self._element_type, np.float32)

def _check_result(result):
    """Check C API result and raise exception if error"""
    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result)
        if error_msg:
            error_str = error_msg.decode('utf-8')
        else:
            error_str = f"Unknown error code: {result}"
        raise RuntimeError(f"Madrona Escape Room error: {error_str}")

class SimManager:
    """Main simulation manager class"""
    
    def __init__(self, exec_mode, gpu_id, num_worlds, rand_seed, auto_reset, 
                 enable_batch_renderer=False):
        from .ctypes_bindings import MER_ManagerConfig, MER_ManagerHandle
        from ctypes import byref
        
        # Create config
        config = MER_ManagerConfig()
        config.exec_mode = exec_mode.value if isinstance(exec_mode, madrona.ExecMode) else exec_mode
        config.gpu_id = gpu_id
        config.num_worlds = num_worlds
        config.rand_seed = rand_seed
        config.auto_reset = auto_reset
        config.enable_batch_renderer = enable_batch_renderer
        config.batch_render_view_width = 64
        config.batch_render_view_height = 64
        
        # Create handle
        self._handle = MER_ManagerHandle()
        result = lib.mer_create_manager(byref(self._handle), byref(config))
        _check_result(result)
        
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            lib.mer_destroy_manager(self._handle)
    
    def step(self):
        """Run one simulation step"""
        result = lib.mer_step(self._handle)
        _check_result(result)
    
    def _get_tensor(self, getter_func):
        """Helper to get tensor from C API"""
        from .ctypes_bindings import MER_Tensor
        from ctypes import byref
        
        c_tensor = MER_Tensor()
        result = getter_func(self._handle, byref(c_tensor))
        _check_result(result)
        return madrona.Tensor(c_tensor)
    
    def reset_tensor(self):
        return self._get_tensor(lib.mer_get_reset_tensor)
    
    def action_tensor(self):
        return self._get_tensor(lib.mer_get_action_tensor)
    
    def reward_tensor(self):
        return self._get_tensor(lib.mer_get_reward_tensor)
    
    def done_tensor(self):
        return self._get_tensor(lib.mer_get_done_tensor)
    
    def self_observation_tensor(self):
        return self._get_tensor(lib.mer_get_self_observation_tensor)
    
    def steps_remaining_tensor(self):
        return self._get_tensor(lib.mer_get_steps_remaining_tensor)
    
    def progress_tensor(self):
        return self._get_tensor(lib.mer_get_progress_tensor)
    
    def rgb_tensor(self):
        return self._get_tensor(lib.mer_get_rgb_tensor)
    
    def depth_tensor(self):
        return self._get_tensor(lib.mer_get_depth_tensor)
    
    def enable_trajectory_logging(self, world_idx, agent_idx, filename=None):
        """Enable trajectory logging for a specific agent"""
        if filename is not None:
            filename_bytes = filename.encode('utf-8')
            result = lib.mer_enable_trajectory_logging(
                self._handle, world_idx, agent_idx, filename_bytes
            )
        else:
            result = lib.mer_enable_trajectory_logging(
                self._handle, world_idx, agent_idx, None
            )
        _check_result(result)
    
    def disable_trajectory_logging(self):
        """Disable trajectory logging"""
        result = lib.mer_disable_trajectory_logging(self._handle)
        _check_result(result)
    
    # Recording functionality
    def start_recording(self, filepath, seed=None):
        """Start recording actions to a binary file
        
        Args:
            filepath: Path where to save the recording
            seed: Random seed to store in metadata (uses current manager seed if None)
        """
        from ctypes import c_uint32
        
        if seed is None:
            # Use a default seed - we don't have access to the manager's current seed
            # so we'll use 0 as a placeholder
            seed = 0
            
        filepath_bytes = filepath.encode('utf-8')
        result = lib.mer_start_recording(self._handle, filepath_bytes, c_uint32(seed))
        _check_result(result)
    
    def stop_recording(self):
        """Stop recording actions"""
        result = lib.mer_stop_recording(self._handle)
        _check_result(result)
    
    def is_recording(self):
        """Check if currently recording
        
        Returns:
            bool: True if recording is active
        """
        from ctypes import c_bool, byref
        
        is_recording = c_bool()
        result = lib.mer_is_recording(self._handle, byref(is_recording))
        _check_result(result)
        return is_recording.value
    
    # Replay functionality
    def load_replay(self, filepath):
        """Load a replay file for playback
        
        Args:
            filepath: Path to the replay file
        """
        filepath_bytes = filepath.encode('utf-8')
        result = lib.mer_load_replay(self._handle, filepath_bytes)
        _check_result(result)
    
    def has_replay(self):
        """Check if a replay is currently loaded
        
        Returns:
            bool: True if replay is loaded
        """
        from ctypes import c_bool, byref
        
        has_replay = c_bool()
        result = lib.mer_has_replay(self._handle, byref(has_replay))
        _check_result(result)
        return has_replay.value
    
    def replay_step(self):
        """Execute one step of replay
        
        Returns:
            bool: True if replay finished (no more steps), False if more steps remain
        """
        from ctypes import c_bool, byref
        
        finished = c_bool()
        result = lib.mer_replay_step(self._handle, byref(finished))
        _check_result(result)
        return finished.value
    
    def get_replay_step_count(self):
        """Get current and total step counts for loaded replay
        
        Returns:
            tuple: (current_step, total_steps)
        """
        from ctypes import c_uint32, byref
        
        current_step = c_uint32()
        total_steps = c_uint32()
        result = lib.mer_get_replay_step_count(self._handle, byref(current_step), byref(total_steps))
        _check_result(result)
        return (current_step.value, total_steps.value)

# Re-export madrona submodule for compatibility
__all__ = [
    'SimManager',
    'madrona',
    'action',
    'SELF_OBSERVATION_SIZE',
    'STEPS_REMAINING_SIZE', 
    'AGENT_ID_SIZE',
    'TOTAL_OBSERVATION_SIZE',
    'NUM_AGENTS',
]
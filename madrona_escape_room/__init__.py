"""
Madrona Escape Room Python bindings using CFFI
Provides the same API as the original nanobind version
"""

import numpy as np
from enum import IntEnum
import os
import sys

# Set up environment for bundled libraries before importing CFFI module
_module_dir = os.path.dirname(os.path.abspath(__file__))
_current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if _module_dir not in _current_ld_path:
    os.environ['LD_LIBRARY_PATH'] = f"{_module_dir}:{_current_ld_path}"

# Import the CFFI module
try:
    from ._madrona_escape_room_cffi import ffi, lib
except ImportError as e:
    raise ImportError(
        f"CFFI bindings not built or missing dependencies. "
        f"Please run: uv run python setup_build.py\n"
        f"Original error: {e}"
    )

# Export constants
SELF_OBSERVATION_SIZE = lib.MER_SELF_OBSERVATION_SIZE
STEPS_REMAINING_SIZE = lib.MER_STEPS_REMAINING_SIZE
AGENT_ID_SIZE = lib.MER_AGENT_ID_SIZE
TOTAL_OBSERVATION_SIZE = lib.MER_TOTAL_OBSERVATION_SIZE
NUM_AGENTS = lib.MER_NUM_AGENTS

# Create action submodule equivalents
class action:
    class move_amount:
        STOP = lib.MER_MOVE_STOP
        SLOW = lib.MER_MOVE_SLOW
        MEDIUM = lib.MER_MOVE_MEDIUM
        FAST = lib.MER_MOVE_FAST
    
    class move_angle:
        FORWARD = lib.MER_MOVE_FORWARD
        FORWARD_RIGHT = lib.MER_MOVE_FORWARD_RIGHT
        RIGHT = lib.MER_MOVE_RIGHT
        BACKWARD_RIGHT = lib.MER_MOVE_BACKWARD_RIGHT
        BACKWARD = lib.MER_MOVE_BACKWARD
        BACKWARD_LEFT = lib.MER_MOVE_BACKWARD_LEFT
        LEFT = lib.MER_MOVE_LEFT
        FORWARD_LEFT = lib.MER_MOVE_FORWARD_LEFT
    
    class rotate:
        FAST_LEFT = lib.MER_ROTATE_FAST_LEFT
        SLOW_LEFT = lib.MER_ROTATE_SLOW_LEFT
        NONE = lib.MER_ROTATE_NONE
        SLOW_RIGHT = lib.MER_ROTATE_SLOW_RIGHT
        FAST_RIGHT = lib.MER_ROTATE_FAST_RIGHT

# Madrona submodule for compatibility
class madrona:
    class ExecMode(IntEnum):
        CPU = lib.MER_EXEC_MODE_CPU
        CUDA = lib.MER_EXEC_MODE_CUDA
    
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
                lib.MER_TENSOR_TYPE_UINT8: 1,
                lib.MER_TENSOR_TYPE_INT8: 1,
                lib.MER_TENSOR_TYPE_INT16: 2,
                lib.MER_TENSOR_TYPE_INT32: 4,
                lib.MER_TENSOR_TYPE_INT64: 8,
                lib.MER_TENSOR_TYPE_FLOAT16: 2,
                lib.MER_TENSOR_TYPE_FLOAT32: 4,
            }
            return size_map.get(self._element_type, 4)
        
        def to_numpy(self):
            """Convert to numpy array (zero-copy view)"""
            import numpy as np
            
            # Map element types to numpy dtypes
            dtype_map = {
                lib.MER_TENSOR_TYPE_UINT8: np.uint8,
                lib.MER_TENSOR_TYPE_INT8: np.int8,
                lib.MER_TENSOR_TYPE_INT16: np.int16,
                lib.MER_TENSOR_TYPE_INT32: np.int32,
                lib.MER_TENSOR_TYPE_INT64: np.int64,
                lib.MER_TENSOR_TYPE_FLOAT16: np.float16,
                lib.MER_TENSOR_TYPE_FLOAT32: np.float32,
            }
            
            np_dtype = dtype_map[self._element_type]
            shape = self.dims()
            
            # Create a numpy array view of the data without copying
            buffer = ffi.buffer(self._tensor.data, self._tensor.num_bytes)
            array = np.frombuffer(buffer, dtype=np_dtype).reshape(shape)
            
            # The array is already a view, not a copy
            array.flags.writeable = True
            return array
        
        def to_torch(self):
            """Convert to PyTorch tensor (zero-copy)"""
            import torch
            
            # Map element types to PyTorch dtypes
            dtype_map = {
                lib.MER_TENSOR_TYPE_UINT8: torch.uint8,
                lib.MER_TENSOR_TYPE_INT8: torch.int8,
                lib.MER_TENSOR_TYPE_INT16: torch.int16,
                lib.MER_TENSOR_TYPE_INT32: torch.int32,
                lib.MER_TENSOR_TYPE_INT64: torch.int64,
                lib.MER_TENSOR_TYPE_FLOAT16: torch.float16,
                lib.MER_TENSOR_TYPE_FLOAT32: torch.float32,
            }
            
            torch_dtype = dtype_map[self._element_type]
            shape = self.dims()
            
            # Get pointer value as integer
            ptr_value = int(ffi.cast("uintptr_t", self._tensor.data))
            
            # Determine device and create tensor
            if self.isOnGPU():
                # For GPU tensors, we need to create a CUDA tensor
                # This is zero-copy - PyTorch will use the CUDA memory directly
                device = torch.device(f'cuda:{self._gpu_id}')
                
                # Calculate total number of elements
                total_elements = 1
                for dim in shape:
                    total_elements *= dim
                
                # Create storage from CUDA pointer
                # Note: This uses PyTorch's internal API to wrap external CUDA memory
                import torch.cuda
                storage = torch.cuda._CudaBase.__new__(torch.cuda.ByteStorage)
                torch.cuda._CudaBase.__init__(storage, total_elements * self.numBytesPerItem(), 
                                            ptr_value, allocator=None, 
                                            device=device.index, resizable=False)
                
                # Create tensor from storage
                tensor = torch.empty(0, dtype=torch_dtype, device=device)
                tensor.set_(storage, 0, shape, tuple())
                
                # Ensure correct dtype view
                if torch_dtype != torch.uint8:
                    tensor = tensor.view(torch_dtype)
                    tensor = tensor.reshape(shape)
            else:
                # For CPU tensors, use from_numpy which maintains zero-copy
                # when the numpy array is already a view (which ours is)
                numpy_view = self.to_numpy()
                tensor = torch.from_numpy(numpy_view)
                
            return tensor
        
        @property
        def shape(self):
            return tuple(self.dims())
        
        @property
        def dtype(self):
            # Return numpy dtype for compatibility
            import numpy as np
            dtype_map = {
                lib.MER_TENSOR_TYPE_UINT8: np.uint8,
                lib.MER_TENSOR_TYPE_INT8: np.int8,
                lib.MER_TENSOR_TYPE_INT16: np.int16,
                lib.MER_TENSOR_TYPE_INT32: np.int32,
                lib.MER_TENSOR_TYPE_INT64: np.int64,
                lib.MER_TENSOR_TYPE_FLOAT16: np.float16,
                lib.MER_TENSOR_TYPE_FLOAT32: np.float32,
            }
            return dtype_map.get(self._element_type, np.float32)

def _check_result(result):
    """Check C API result and raise exception if error"""
    if result != lib.MER_SUCCESS:
        error_msg = ffi.string(lib.mer_result_to_string(result)).decode('utf-8')
        raise RuntimeError(f"Madrona Escape Room error: {error_msg}")

class SimManager:
    """Main simulation manager class"""
    
    def __init__(self, exec_mode, gpu_id, num_worlds, rand_seed, auto_reset, 
                 enable_batch_renderer=False):
        # Create config
        config = ffi.new("MER_ManagerConfig*")
        config.exec_mode = exec_mode.value if isinstance(exec_mode, madrona.ExecMode) else exec_mode
        config.gpu_id = gpu_id
        config.num_worlds = num_worlds
        config.rand_seed = rand_seed
        config.auto_reset = auto_reset
        config.enable_batch_renderer = enable_batch_renderer
        config.batch_render_view_width = 64
        config.batch_render_view_height = 64
        
        # Create handle
        self._handle_ptr = ffi.new("MER_ManagerHandle*")
        result = lib.mer_create_manager(self._handle_ptr, config)
        _check_result(result)
        
        self._handle = self._handle_ptr[0]
        
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            lib.mer_destroy_manager(self._handle)
    
    def step(self):
        """Run one simulation step"""
        result = lib.mer_step(self._handle)
        _check_result(result)
    
    def _get_tensor(self, getter_func):
        """Helper to get tensor from C API"""
        c_tensor = ffi.new("MER_Tensor*")
        result = getter_func(self._handle, c_tensor)
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
                self._handle, world_idx, agent_idx, ffi.NULL
            )
        _check_result(result)
    
    def disable_trajectory_logging(self):
        """Disable trajectory logging"""
        result = lib.mer_disable_trajectory_logging(self._handle)
        _check_result(result)

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
import numpy as np

from .generated_constants import TensorElementType


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
            TensorElementType.UInt8: 1,
            TensorElementType.Int8: 1,
            TensorElementType.Int16: 2,
            TensorElementType.Int32: 4,
            TensorElementType.Int64: 8,
            TensorElementType.Float16: 2,
            TensorElementType.Float32: 4,
        }
        return size_map.get(self._element_type, 4)

    def to_numpy(self):
        """Convert to numpy array (zero-copy view, CPU only)"""
        import ctypes

        # Check if tensor is on GPU
        if self.isOnGPU():
            raise RuntimeError(
                "Cannot convert GPU tensor to numpy array directly. "
                "Use to_torch() or torch.from_dlpack() for GPU tensors."
            )

        # Map element types to numpy dtypes
        dtype_map = {
            TensorElementType.UInt8: np.uint8,
            TensorElementType.Int8: np.int8,
            TensorElementType.Int16: np.int16,
            TensorElementType.Int32: np.int32,
            TensorElementType.Int64: np.int64,
            TensorElementType.Float16: np.float16,
            TensorElementType.Float32: np.float32,
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
        """Convert to PyTorch tensor using DLPack (zero-copy)"""
        import torch

        return torch.from_dlpack(self)

    def __dlpack__(self, stream=None):
        """Create DLPack capsule for PyTorch consumption"""
        import ctypes

        try:
            from . import _madrona_escape_room_dlpack as dlpack_ext
        except ImportError:
            # Raise ImportError so tests can properly skip
            raise ImportError("DLPack extension module not found")

        # Get tensor parameters
        data_ptr = ctypes.cast(self._tensor.data, ctypes.c_void_p).value
        shape = self.dims()
        dtype = self._element_type
        device_type = 2 if self.isOnGPU() else 1  # 2=CUDA, 1=CPU
        device_id = self.gpuID() if self.isOnGPU() else 0

        # Create DLPack capsule
        return dlpack_ext.create_dlpack_capsule(data_ptr, shape, dtype, device_type, device_id)

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
        dtype_map = {
            TensorElementType.UInt8: np.uint8,
            TensorElementType.Int8: np.int8,
            TensorElementType.Int16: np.int16,
            TensorElementType.Int32: np.int32,
            TensorElementType.Int64: np.int64,
            TensorElementType.Float16: np.float16,
            TensorElementType.Float32: np.float32,
        }
        return dtype_map.get(self._element_type, np.float32)

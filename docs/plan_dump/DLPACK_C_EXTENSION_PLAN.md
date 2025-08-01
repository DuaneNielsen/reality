# DLPack Python C Extension Module Implementation Plan

## Overview
Create a separate Python C extension module (`_madrona_escape_room_dlpack`) that provides DLPack protocol support for the existing CFFI-based tensor system. This approach isolates DLPack complexity while maintaining clean separation from the main CFFI bindings.

## Architecture

```
┌─────────────────────────────────┐
│ Python User Code                │
│ torch.from_dlpack(tensor)       │
└─────────────────────────────────┘
                 │
┌─────────────────────────────────┐
│ madrona_escape_room.__init__.py │
│ Tensor.__dlpack__()             │
│ Tensor.__dlpack_device__()      │
└─────────────────────────────────┘
                 │
┌─────────────────────────────────┐
│ _madrona_escape_room_dlpack.so  │
│ (C Extension Module)            │
│ - create_dlpack_capsule()       │
│ - get_dlpack_device()           │
└─────────────────────────────────┘
                 │
┌─────────────────────────────────┐
│ _madrona_escape_room_cffi.so    │
│ (CFFI Module - existing)        │
│ - tensor data, shape, dtype     │
└─────────────────────────────────┘
```

## Implementation Steps

### Phase 1: C Extension Module Core (High Priority)

1. **Create C Extension Structure**
   - `src/dlpack_extension.cpp` - Main C extension implementation
   - `scripts/build_dlpack_extension.py` - Build script for the extension
   - Update `pyproject.toml` to include the C extension in build

2. **DLPack Core Functions**
   - `create_dlpack_capsule(data_ptr, shape, dtype, device_type, device_id)` - Creates PyCapsule with DLManagedTensor
   - `get_dlpack_device(device_type, device_id)` - Returns device tuple for `__dlpack_device__`
   - Proper DLManagedTensor struct creation with cleanup callbacks

3. **Memory Management**
   - DLManagedTensor deleter that ONLY frees DLPack metadata (not Madrona's data)
   - Reference counting to prevent double-free
   - CUDA vs CPU memory handling

### Phase 2: Python Integration (High Priority)

4. **Update Tensor Class in `__init__.py`**
   - Add `__dlpack__()` method that calls C extension
   - Add `__dlpack_device__()` method  
   - Import and use the DLPack extension module
   - Maintain backward compatibility with existing `to_torch()` method

5. **Error Handling**
   - Proper exception translation from C to Python
   - Validation of tensor parameters before capsule creation
   - Graceful fallback if DLPack extension fails to load

### Phase 3: Build System Integration (Medium Priority)

6. **Update Build Configuration**
   - Modify `pyproject.toml` to build both CFFI and C extension modules
   - Ensure proper compiler flags and Python include paths
   - Handle CUDA SDK detection if needed for GPU tensors

7. **Package Structure**
   - Both `_madrona_escape_room_cffi.so` and `_madrona_escape_room_dlpack.so` bundled
   - Update import logic to handle missing DLPack extension gracefully
   - Maintain existing CFFI functionality unchanged

### Phase 4: Testing & Validation (Medium Priority)

8. **DLPack Protocol Tests**
   - Test `__dlpack__()` and `__dlpack_device__()` methods exist
   - Verify PyCapsule creation and validity
   - Test PyTorch `torch.from_dlpack()` integration
   - Test both CPU and GPU tensors (if GPU support needed)

9. **Integration Tests**
   - Update existing test suite to use DLPack when available
   - Backward compatibility tests with `to_torch()` method
   - Performance comparison between DLPack and numpy conversion paths

## Technical Details

### DLPack C Extension Implementation

```cpp
// Core function signatures
PyObject* create_dlpack_capsule(PyObject* self, PyObject* args);
PyObject* get_dlpack_device(PyObject* self, PyObject* args);

// DLPack structures (based on nanobind reference)
typedef struct {
    int32_t device_type;  // 1=CPU, 2=CUDA
    int32_t device_id;
} DLDevice;

typedef struct {
    uint8_t code;    // 0=Int, 1=UInt, 2=Float
    uint8_t bits;    // 8, 16, 32, 64
    uint16_t lanes;  // Always 1 for scalars
} DLDataType;

typedef struct {
    void* data;           // Madrona tensor data pointer
    DLDevice device;      // Device info
    int ndim;            // Number of dimensions  
    DLDataType dtype;    // Element type
    int64_t* shape;      // Dimensions array (malloc'd)
    int64_t* strides;    // NULL for C-contiguous
    uint64_t byte_offset; // Always 0
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;    // Context for cleanup
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;
```

### Python Integration Points

```python
# In madrona_escape_room/__init__.py
class Tensor:
    def __dlpack__(self, stream=None):
        """Create DLPack capsule for PyTorch consumption"""
        import _madrona_escape_room_dlpack as dlpack_ext
        return dlpack_ext.create_dlpack_capsule(
            int(ffi.cast("uintptr_t", self._tensor.data)),
            self.dims(),
            self._element_type, 
            2 if self.isOnGPU() else 1,  # device_type
            self.gpuID() if self.isOnGPU() else 0  # device_id
        )
    
    def __dlpack_device__(self):
        """Return device info tuple"""
        import _madrona_escape_room_dlpack as dlpack_ext
        return dlpack_ext.get_dlpack_device(
            2 if self.isOnGPU() else 1,  # device_type  
            self.gpuID() if self.isOnGPU() else 0  # device_id
        )
```

## File Structure

```
src/
├── dlpack_extension.cpp           # New C extension
├── madrona_escape_room_c_api.cpp  # Existing CFFI wrapper
└── ...

scripts/
├── build_dlpack_extension.py      # New build script  
├── build_cffi.py                  # Existing CFFI build
└── ...

madrona_escape_room/
├── __init__.py                     # Updated with DLPack methods
├── _madrona_escape_room_cffi.so    # Existing CFFI module
└── _madrona_escape_room_dlpack.so  # New DLPack module
```

## Benefits

1. **Clean Separation**: DLPack complexity isolated in focused C extension
2. **Maintainability**: Each module has single responsibility
3. **Robust PyCapsule Creation**: Uses Python C API properly
4. **Minimal Dependencies**: No additional external dependencies
5. **Backward Compatibility**: Existing CFFI functionality unchanged
6. **Performance**: Zero-copy DLPack integration with PyTorch

## Success Criteria

- `torch.from_dlpack(tensor)` works with CFFI tensors
- All existing tests continue to pass
- DLPack tests pass for both CPU and GPU tensors
- Clean build process with both modules
- No dependency hell or version conflicts
# Python Bindings Architecture

This document provides a detailed technical overview of the Python bindings architecture for the Madrona Escape Room environment.

## Table of Contents
- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Architecture Components](#architecture-components)
- [Build Process](#build-process)
- [Memory Management](#memory-management)
- [Packaging and Distribution](#packaging-and-distribution)
- [Technical Decisions](#technical-decisions)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## Overview

The Python bindings provide a bridge between the C++ Madrona simulation engine and Python/PyTorch for machine learning applications. The architecture was designed with these key requirements:

1. **No RTTI/Exceptions**: Madrona engine is compiled without RTTI and exceptions for performance
2. **Zero-Copy Tensor Access**: Direct memory access between C++ and Python/PyTorch
3. **Self-Contained Packaging**: Bundle all dependencies for easy distribution
4. **API Compatibility**: Maintain the same Python API as the original nanobind version

## Technology Stack

### Core Technologies

1. **CFFI (C Foreign Function Interface)**
   - Version: Latest (1.15+)
   - Role: Generate Python bindings from C headers
   - Why: Works without C++ RTTI, better than ctypes for complex APIs

2. **C++ to C Wrapper Layer**
   - Pure C API wrapping C++ classes
   - Error handling via return codes instead of exceptions
   - Opaque handle pattern for object management

3. **Shared Libraries**
   - `libmadrona_escape_room_c_api.so`: Main C wrapper library
   - `_madrona_escape_room_cffi.cpython-*.so`: CFFI-generated Python extension
   - Bundled dependencies: embree, dxcompiler, shader compiler, memory manager

### Build Tools

- **CMake**: C++ build system (3.18+)
- **Make**: Build automation
- **Python setuptools**: Python packaging
- **uv**: Python package management

## Architecture Components

### 1. C++ Core (Madrona Engine)

The simulation engine written in C++ with these characteristics:
- Entity Component System (ECS) architecture
- No RTTI (`-fno-rtti`)
- No exceptions (`-fno-exceptions`)
- Custom memory management
- GPU and CPU execution modes

### 2. C API Wrapper Layer

**Header**: `include/madrona_escape_room_c_api.h`
```c
// Opaque handle for C++ Manager object
typedef struct MER_Manager MER_Manager;

// Error codes instead of exceptions
typedef enum {
    MER_SUCCESS = 0,
    MER_ERROR_INVALID_PARAMETER = -1,
    MER_ERROR_OUT_OF_MEMORY = -2,
    MER_ERROR_CUDA_ERROR = -3,
    MER_ERROR_UNKNOWN = -999
} MER_ErrorCode;

// C-compatible tensor descriptor
typedef struct {
    void* data;
    int64_t dimensions[16];
    int32_t num_dimensions;
    MER_TensorElementType element_type;
    int64_t num_bytes;
    int32_t gpu_id;
} MER_Tensor;
```

**Implementation**: `src/madrona_escape_room_c_api.cpp`
- Wraps C++ `Manager` class methods
- Translates exceptions to error codes
- Manages object lifetime via handles
- Converts between C++ and C tensor representations

### 3. CFFI Binding Layer

**Build Script**: `scripts/build_cffi.py`
```python
# Parse C header (with preprocessing)
processed_header = preprocess_header(header_content)

# Define CFFI module
ffi = FFI()
ffi.cdef(processed_header)
ffi.set_source(
    "_madrona_escape_room_cffi",
    '#include "madrona_escape_room_c_api.h"',
    include_dirs=[include_dir],
    library_dirs=[lib_dir],
    libraries=["madrona_escape_room_c_api"],
    extra_link_args=["-Wl,-rpath,$ORIGIN"]  # Find libs in same dir
)
```

### 4. Python Wrapper Layer

**Module**: `madrona_escape_room/__init__.py`

Provides Pythonic API and zero-copy tensor access:

```python
class Tensor:
    """Zero-copy tensor wrapper"""
    def __init__(self, c_tensor):
        self._tensor = c_tensor
        self._cached_numpy = None
        self._cached_torch = None
    
    def to_numpy(self):
        if self._cached_numpy is None:
            # Create view without copying data
            self._cached_numpy = np.frombuffer(
                ffi.buffer(self._tensor.data, self._tensor.num_bytes),
                dtype=self._get_numpy_dtype()
            ).reshape(self.dimensions())
        return self._cached_numpy
    
    def to_torch(self):
        if self._cached_torch is None:
            numpy_array = self.to_numpy()
            if self.gpu_id() >= 0:
                # GPU tensor - use torch CUDA tensor
                self._cached_torch = torch.as_tensor(
                    numpy_array, 
                    device=f'cuda:{self.gpu_id()}'
                )
            else:
                # CPU tensor - zero-copy from numpy
                self._cached_torch = torch.from_numpy(numpy_array)
        return self._cached_torch
```

## Build Process

### 1. C++ Library Build
```bash
# CMake configuration
cmake -B build

# Compile C++ libraries
make -C build -j$(nproc)
```

This produces:
- Static libraries (`.a`) for Madrona components
- Shared library `libmadrona_escape_room_c_api.so`
- External dependencies (embree, dxcompiler)

### 2. CFFI Module Build
```bash
# Run build script
python scripts/build_cffi.py
```

Steps:
1. Parse C header file
2. Remove preprocessor directives for CFFI
3. Generate C extension code
4. Compile with Python headers
5. Link against C wrapper library

### 3. Package Assembly
```bash
# Full build and install
python setup_build.py
```

This script:
1. Builds C++ libraries via CMake
2. Builds CFFI module
3. Copies all libraries to package directory
4. Sets up proper rpath for library loading
5. Installs package with pip

## Memory Management

### Zero-Copy Design

The bindings achieve zero-copy by:

1. **Direct Memory Access**: Python/PyTorch tensors created as views over C++ memory
2. **Reference Counting**: Python holds references to C++ manager to prevent deallocation
3. **Memory Layout**: C++ uses row-major layout compatible with NumPy/PyTorch

### GPU Memory

For GPU tensors:
- Memory allocated on GPU via CUDA
- CPU-side pointer provided for metadata
- PyTorch CUDA tensors created from device pointers
- Proper device context management

### Lifetime Management

```python
class SimManager:
    def __init__(self, ...):
        # Create C++ manager via C API
        self._handle = lib.MER_Manager_create(...)
        
        # Cache tensor wrappers (they reference same memory)
        self._action_tensor = None
        self._obs_tensor = None
    
    def __del__(self):
        # Clean up C++ object
        if hasattr(self, '_handle') and self._handle:
            lib.MER_Manager_destroy(self._handle)
```

## Packaging and Distribution

### Library Dependencies

The package bundles these shared libraries:

```
madrona_escape_room/
├── __init__.py                    # Python module
├── _madrona_escape_room_cffi.so   # CFFI extension
├── libmadrona_escape_room_c_api.so # C wrapper
├── libembree4.so.4                # Intel Embree (ray tracing)
├── libdxcompiler.so               # DirectX compiler
├── libmadrona_render_shader_compiler.so # Shader compiler
└── libmadrona_std_mem.so          # Memory management
```

### Dynamic Loading

Library loading strategy:

1. **LD_LIBRARY_PATH Setup**: Add package directory to library search path
2. **RPATH Configuration**: CFFI module has `$ORIGIN` rpath to find libraries
3. **Bundled Dependencies**: All required libraries shipped with package

```python
# In __init__.py
import os
_module_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['LD_LIBRARY_PATH'] = f"{_module_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
```

### Setup Configuration

**setup.py**:
```python
setup(
    name="madrona-escape-room",
    packages=find_packages(),
    package_data={
        "madrona_escape_room": [
            "*.so",          # All shared libraries
            "*.so.*",        # Versioned libraries
            "*.pyi",         # Type stubs
        ],
    },
    install_requires=["torch>=2.0.0", "numpy>=1.20.0", "cffi>=1.15.0"],
)
```

## Technical Decisions

### Why CFFI over ctypes?

1. **Better C Header Parsing**: CFFI can parse complex C headers automatically
2. **Performance**: CFFI generates compiled extensions, faster than ctypes
3. **Type Safety**: Better type checking at build time
4. **NumPy Integration**: Better buffer protocol support

### Why Not Cython/pybind11/nanobind?

1. **No RTTI Requirement**: These tools typically require RTTI
2. **Exception Handling**: Madrona compiled without exceptions
3. **Simplicity**: CFFI requires only C API, no C++ complexity

### Static vs Dynamic Linking

We use a hybrid approach:
- **Static linking**: Most Madrona libraries linked statically into C wrapper
- **Dynamic linking**: External dependencies (embree, dxcompiler) bundled as shared libraries
- **Reasoning**: Reduces number of shared libraries while handling pre-built dependencies

### CUDA Library Conflicts and Resolution

#### The Problem

When using PyTorch with CUDA support alongside Madrona's CUDA bindings, we encountered a critical issue:
- PyTorch bundles its own CUDA libraries (typically CUDA 12.6 as of late 2024)
- Our system uses CUDA 12.8 for Madrona compilation
- Python extensions search their site-packages before system libraries
- This causes symbol resolution failures when PyTorch is imported first

The error manifests as:
```
undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

#### Root Cause Analysis

1. **Library Search Order**: When our C extension loads, it searches for dependencies in this order:
   - RPATH/RUNPATH paths
   - LD_LIBRARY_PATH
   - Python site-packages (where PyTorch lives)
   - System library paths (/usr/lib, /usr/local/cuda/lib64)

2. **PyTorch's Bundled Libraries**: PyTorch includes:
   - `libnvJitLink.so.12` (CUDA 12.6 version)
   - `libnvrtc.so.12` and other CUDA runtime libraries
   - These libraries lack symbols for CUDA 12.8 (e.g., `__nvJitLinkCreate_12_8`)

3. **Import Order Sensitivity**: 
   - `import torch` then `import madrona_escape_room` → Fails (PyTorch's libs loaded first)
   - `import madrona_escape_room` then `import torch` → Works (correct libs loaded first)

#### The Solution: Static Linking

We resolved this by statically linking CUDA libraries that have version conflicts:

```cmake
# In src/CMakeLists.txt
if(CUDAToolkit_FOUND)
    # Find static versions of CUDA libraries
    find_library(NVJITLINK_STATIC_LIB nvJitLink_static 
        PATHS ${CUDAToolkit_LIBRARY_DIR}
        NO_DEFAULT_PATH)
    
    find_library(NVPTXCOMPILER_STATIC_LIB nvptxcompiler_static
        PATHS ${CUDAToolkit_LIBRARY_DIR}
        NO_DEFAULT_PATH)
    
    if(NVJITLINK_STATIC_LIB AND NVPTXCOMPILER_STATIC_LIB)
        target_link_libraries(madrona_escape_room_c_api PRIVATE
            CUDA::nvrtc
            CUDA::cuda_driver
            ${NVJITLINK_STATIC_LIB}      # Static linking
            ${NVPTXCOMPILER_STATIC_LIB}   # Static linking
        )
    endif()
endif()
```

Benefits of this approach:
- **No Runtime Conflicts**: Static symbols are resolved at link time
- **Import Order Independence**: Works regardless of import order
- **No LD_LIBRARY_PATH Hacks**: Clean solution without environment manipulation
- **Version Stability**: Our code always uses the CUDA version it was built with

#### Alternative Solutions (Not Recommended)

1. **LD_LIBRARY_PATH Manipulation**: Setting system paths before Python paths
   - Fragile and affects entire Python process
   - Can break other packages

2. **Import Order Requirements**: Requiring users to import in specific order
   - Poor user experience
   - Easy to break accidentally

3. **Shipping CUDA Libraries**: Bundling our own CUDA libraries
   - Large package size
   - Potential conflicts with system CUDA

### Error Handling Strategy

Since exceptions are disabled:
1. C API returns error codes
2. Python wrapper checks return values
3. Converts errors to Python exceptions
4. Preserves error context via error messages

```python
def _check_error(result):
    if result != lib.MER_SUCCESS:
        error_map = {
            lib.MER_ERROR_INVALID_PARAMETER: ValueError,
            lib.MER_ERROR_OUT_OF_MEMORY: MemoryError,
            lib.MER_ERROR_CUDA_ERROR: RuntimeError,
        }
        exc_type = error_map.get(result, RuntimeError)
        raise exc_type(f"Madrona error code: {result}")
```

## Performance Considerations

### Zero-Copy Verification

Ensure zero-copy by checking memory addresses:
```python
tensor = mgr.action_tensor()
numpy_arr = tensor.to_numpy()
torch_tensor = tensor.to_torch()

# Verify same memory
assert numpy_arr.__array_interface__['data'][0] == torch_tensor.data_ptr()
```

### Overhead Analysis

CFFI overhead is minimal:
- Function call overhead: ~100ns per call
- No data copying for tensor access
- Compiled extension (not interpreted)
- Direct memory access for large tensors

### GPU Performance

GPU tensor access maintains zero-copy:
- Device pointers passed directly
- No CPU-GPU transfers for tensor access
- PyTorch CUDA tensors created from device memory
- Synchronization handled by simulation step

## Troubleshooting

### CUDA Version Mismatch Errors

**Symptom**: Import fails with undefined symbol errors like:
```
undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

**Diagnosis**:
1. Check your CUDA version: `nvcc --version`
2. Check PyTorch's CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Check which libraries are loaded: 
   ```python
   import ctypes
   ctypes.CDLL("libnvJitLink.so.12", mode=ctypes.RTLD_GLOBAL)
   # Then check /proc/self/maps for actual library loaded
   ```

**Solution**: Rebuild with static CUDA libraries (see CUDA Library Conflicts section above)

### Missing Libraries at Runtime

**Symptom**: Import fails with library not found errors

**Diagnosis**:
```bash
ldd build/libmadrona_escape_room_c_api.so
# Check for "not found" entries
```

**Solutions**:
1. Ensure all libraries are copied during build: `python setup_build.py`
2. Check package contains all libraries: `ls madrona_escape_room/*.so*`
3. Verify RPATH is set correctly: `readelf -d madrona_escape_room/_madrona_escape_room_cffi*.so | grep RPATH`

### Symbol Visibility Issues

**Symptom**: CFFI can't find symbols in C API

**Diagnosis**:
```bash
nm -D build/libmadrona_escape_room_c_api.so | grep MER_
# Should show exported symbols
```

**Solution**: Ensure proper export macros in C header:
```c
#define MER_EXPORT __attribute__((visibility("default")))
```

## Future Improvements

1. **Type Annotations**: Generate `.pyi` files for better IDE support
2. **Async API**: Support for async simulation stepping
3. **Multi-GPU**: Better multi-GPU distribution support
4. **Wheels**: Pre-built binary wheels for common platforms
5. **Memory Pools**: Custom allocators for better performance
6. **CUDA Version Detection**: Automatically detect and link appropriate CUDA version
7. **Better Error Messages**: More descriptive error messages for common issues
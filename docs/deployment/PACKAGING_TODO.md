# Packaging TODO

## ✅ COMPLETED: Basic Library Packaging
**Status**: Fixed in setup.py - all 5 required libraries now get copied during build

## 🚨 OUTSTANDING PROBLEMS

### 1. CUDA 12.8+ Build Failures

**Technical Problem**: 
Build system fails with CUDA 12.8+ installations due to version-specific dependencies in the build chain.

**Build Failure Pattern**:
```bash
# CUDA 12.8 build attempt
export CUDA_HOME=/usr/local/cuda-12.8
cmake -B build -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
# Fails at compile time with version incompatibility errors
```

**Root Cause Analysis Needed**:
- Which specific CUDA components require 12.5?
- Is this a Madrona submodule limitation or our build configuration?
- Are there specific NVRTC API changes in 12.8+ that break compilation?
- What symbols/functions are missing or changed between 12.5 and 12.8?

**Technical Investigation Required**:
- Analyze exact build error messages with CUDA 12.8
- Determine if issue is in NVRTC, CUDA runtime, or driver API usage
- Check Madrona's CUDA version requirements and compatibility matrix

### 2. System CUDA vs PyTorch CUDA Symbol Version Conflicts

**Technical Problem**: 
ctypes C API library compiled against system CUDA creates runtime symbol conflicts with PyTorch's bundled CUDA.

**Symbol Version Mismatch Details**:
```bash
# Build time (system CUDA 12.5)
libmadrona_escape_room_c_api.so -> requires __nvJitLinkCreate_12_5

# Runtime (PyTorch CUDA 12.4)  
PyTorch's libnvJitLink.so.12 -> provides __nvJitLinkCreate_12_4
Result: undefined symbol: __nvJitLinkCreate_12_5
```

**Technical Investigation Needed**:
- Exact symbol dependencies our C API library requires
- How PyTorch determines which CUDA libraries to bundle
- Whether PyTorch CUDA libraries have full symbol compatibility matrices
- Library loading order and symbol resolution behavior in ctypes context

**Library Detection Investigation**:
```bash
# What CUDA libraries does our C API actually link against?
ldd build/libmadrona_escape_room_c_api.so | grep -i cuda

# What symbols does it require?
nm -D build/libmadrona_escape_room_c_api.so | grep nvJit

# What PyTorch CUDA libraries are available?
python -c "
import torch, os
torch_path = torch.__path__[0]
nvidia_path = os.path.join(os.path.dirname(torch_path), 'nvidia')
print(f'PyTorch CUDA libs: {nvidia_path}')
"
find $(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__path__[0]), 'nvidia'))") -name "*.so*" | grep -i nvjit
```

**Build System Analysis Needed**:
- How does CMake's `find_package(CUDAToolkit)` determine which CUDA to use?
- Can we detect and build against PyTorch's CUDA installation instead?
- What happens if we set `CUDA_TOOLKIT_ROOT_DIR` to PyTorch's CUDA path?
- Are there CMake variables that control CUDA library linking paths?

### 3. NVRTC Dynamic Library Loading Issues

**Technical Problem**:
NVRTC performs dynamic library loading at runtime for `libnvrtc-builtins.so.X.Y`, which fails when version mismatches occur.

**Dynamic Loading Details**:
```cpp
// NVRTC internally does something like:
dlopen("libnvrtc-builtins.so.12.5", RTLD_LAZY);
// Hardcoded version string based on compile-time CUDA version
```

**Investigation Required**:
- How does NVRTC determine which builtin library version to load?
- Is the version string compile-time determined or runtime detected?
- Can we intercept or redirect NVRTC's dynamic loading?
- What environment variables or runtime configuration affect NVRTC library search?

**Library Path Analysis**:
```bash
# Where does NVRTC search for builtin libraries?
LD_DEBUG=libs ./build/headless --cuda 0 -n 1 -s 10 2>&1 | grep nvrtc-builtins

# What versions are available in different CUDA installations?
find /usr/local/cuda* -name "libnvrtc-builtins*" 2>/dev/null
ls -la /usr/local/cuda-12.*/targets/x86_64-linux/lib/libnvrtc-builtins*
```

**Research Questions**:
- Does NVRTC version detection come from compile-time macros or runtime detection?
- Can we build NVRTC to use a generic version or runtime-detected version?
- Is there a way to make NVRTC version-agnostic for builtin library loading?


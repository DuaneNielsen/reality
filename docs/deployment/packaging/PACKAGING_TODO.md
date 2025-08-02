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

## Required Fixes

### 1. Fix BuildPyWithLibrary Class in setup.py

**Current (incomplete):**
```python
class BuildPyWithLibrary(build_py):
    def run(self):
        super().run()
        
        # Copy the C library to the package
        c_lib = Path("build/libmadrona_escape_room_c_api.so")
        if c_lib.exists():
            # Only copies main library - MISSING DEPENDENCIES!
            shutil.copy2(c_lib, dest)
```

**Needs to be:**
```python
class BuildPyWithLibrary(build_py):
    def run(self):
        super().run()
        
        # ALL libraries that ctypes needs
        required_libs = [
            "libmadrona_escape_room_c_api.so",  # Main C API (ctypes loads this)
            "libembree4.so.4",                   # Embree ray tracing  
            "libdxcompiler.so",                  # DirectX shader compiler
            "libmadrona_render_shader_compiler.so", # Madrona rendering
            "libmadrona_std_mem.so"              # Madrona memory management
        ]
        
        build_dir = Path("build")
        for package in self.packages:
            if package == "madrona_escape_room":
                package_dir = Path(self.build_lib) / package
                
                for lib_name in required_libs:
                    lib_path = build_dir / lib_name
                    if lib_path.exists():
                        dest = package_dir / lib_name
                        print(f"Copying {lib_name} to {dest}")
                        shutil.copy2(lib_path, dest)
                    else:
                        print(f"Warning: {lib_name} not found in build directory")
```

### 2. Test the Fix

After fixing setup.py:

```bash
# Clean rebuild
rm -rf build/ dist/ *.egg-info/
make -C build -j$(nproc)

# Build package  
python -m build

# Test in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/madrona_escape_room-*.whl

# Verify it works
python -c "
import madrona_escape_room as mer
mgr = mer.SimManager(mer.madrona.ExecMode.CPU, 0, 1, 42, True)
print('✅ Packaged ctypes bindings work!')
"
```

### 3. Verify Library Inclusion

Check that wheel contains all libraries:
```bash
unzip -l dist/madrona_escape_room-*.whl | grep "\.so"
```

Should show:
- `libmadrona_escape_room_c_api.so` ✅ Main library
- `libembree4.so.4` ❌ Currently missing  
- `libdxcompiler.so` ❌ Currently missing
- `libmadrona_render_shader_compiler.so` ❌ Currently missing
- `libmadrona_std_mem.so` ❌ Currently missing
- `_madrona_escape_room_dlpack.cpython-*.so` ✅ DLPack extension

## Why This Matters

Without all libraries packaged:
- ✅ Development works (finds libraries in build/ directory)
- ❌ Distribution fails (missing dependencies when installed)
- ❌ Users get "library not found" errors
- ❌ ctypes.CDLL() fails to load main library due to missing dependencies

## Priority: HIGH
This is the final step to make ctypes bindings distributable. The implementation is correct, just the packaging is incomplete.

## Estimated Time: 15 minutes
Simple fix to copy all required libraries during build.
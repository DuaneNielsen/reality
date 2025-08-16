# Debugging CUDA Linking Issues in Python Extensions

This guide documents a systematic approach to debugging CUDA library linking conflicts, particularly when integrating custom C++ extensions with PyTorch.

## The Problem We Solved

**Symptom**: `ImportError: undefined symbol: __nvJitLinkCreate_12_5, version libnvJitLink.so.12`

**Root Cause**: Our library was compiled against system CUDA 12.5, but PyTorch 2.5.1 shipped with older CUDA 12.4 libraries. Since CUDA backward compatibility doesn't work in reverse (older runtime can't run newer-compiled code), the 12.4 runtime couldn't provide the 12.5 symbols our library needed.

**The Fix**: Upgrading to PyTorch 2.7.1 (CUDA 12.6) provided a newer runtime that could satisfy our library's 12.5 symbol requirements through CUDA's backward compatibility.

## Understanding CUDA Backward Compatibility

### The Rule
- ✅ **Newer CUDA runtime** can run code compiled against **older CUDA versions**
- ❌ **Older CUDA runtime** cannot run code compiled against **newer CUDA versions**
- ⚠️ **Compatibility window is limited** - typically 2-3 major versions, not indefinite

### Our Specific Case
```
System CUDA: 12.5 (our library compiled against this)
PyTorch 2.5.1: CUDA 12.4 runtime (too old - can't run 12.5 compiled code)
PyTorch 2.7.1: CUDA 12.6 runtime (newer - can run 12.5 compiled code)
```

### Symbol Versioning Pattern
CUDA uses versioned symbols like `__nvJitLinkCreate_X_Y` where:
- `X_Y` represents the CUDA version (e.g., `12_5` = CUDA 12.5)
- Runtime libraries provide symbols for their version AND older versions within the compatibility window
- Your compiled code requests the specific version it was built against

## Systematic Debugging Approach

### 1. Identify the Version Mismatch Pattern

CUDA linking errors typically show:
```
undefined symbol: __nvJitLinkCreate_12_5, version libnvJitLink.so.12
```

This tells you:
- **Required version**: 12.5 (what your code needs)  
- **Available library**: libnvJitLink.so.12 (but wrong symbol version)

### 2. Check All CUDA Versions in Your Environment

```bash
# System CUDA version (what you compiled against)
nvcc --version
cat /usr/local/cuda/version.txt  # If it exists

# Driver version (compatibility upper bound)
nvidia-smi

# PyTorch's bundled CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# What CUDA packages did PyTorch install?
python -c "import pkg_resources; [print(f'{pkg.key}: {pkg.version}') for pkg in pkg_resources.working_set if 'nvidia' in pkg.key]"
```

### 3. Trace Library Loading with LD_DEBUG

```bash
# Test your extension in isolation (should work)
LD_DEBUG=libs python -c "import your_extension" 2>&1 | grep -i nvjit

# Test with PyTorch first (likely fails)
LD_DEBUG=libs python -c "import torch; import your_extension" 2>&1 | grep -A10 -B5 nvjit
```

**What to look for**:
- Which nvJitLink library gets loaded first
- Whether the search finds the right library but fails on symbol resolution
- RPATH/RUNPATH search order conflicts

### 4. Analyze Symbol Availability

```bash
# What symbols does your library need?
nm -D /path/to/your/extension.so | grep nvJit
objdump -T /path/to/your/extension.so | grep nvJit

# What symbols does PyTorch's nvJitLink provide?
nm -D $(python -c "import torch; print(torch.__path__[0])")/../nvidia/nvjitlink/lib/libnvJitLink.so.12 | grep nvJit

# Compare system CUDA vs PyTorch CUDA symbols
nm -D /usr/local/cuda/lib64/libnvJitLink.so.12 | grep nvJit
```

### 5. Verify Backward Compatibility Window

Not all CUDA versions are backward compatible indefinitely. Check NVIDIA's documentation for your specific versions:

```bash
# Check what your system supports
nvidia-smi --query-gpu=compute_cap --format=csv
# Cross-reference with CUDA compatibility matrix
```

## Solution Strategies (In Order of Preference)

### Solution 1: Update PyTorch to Newer CUDA Version

**Best approach**: Ensure PyTorch's CUDA version is newer than your compilation target.

```toml
# pyproject.toml - Pin to version with compatible CUDA
dependencies = [
    "torch>=2.7.0",  # CUDA 12.6, backward compatible with 12.5
]
```

**Verification**:
```bash
# After upgrade, check versions align
python -c "
import torch
torch_cuda = tuple(map(int, torch.version.cuda.split('.')))
print(f'PyTorch CUDA: {torch_cuda}')
# Your compiled version should be <= PyTorch version
"
```

### Solution 2: Rebuild Against PyTorch's CUDA Version

**When to use**: If you can't upgrade PyTorch (e.g., other dependencies)

```bash
# Find PyTorch's CUDA installation
python -c "
import torch, os
nvidia_path = os.path.join(os.path.dirname(torch.__path__[0]), 'nvidia')
print(f'PyTorch CUDA path: {nvidia_path}')
"

# Rebuild with PyTorch's CUDA
export CUDA_HOME=/path/to/pytorch/nvidia/cuda_runtime
export CUDA_ROOT=$CUDA_HOME
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ...
```

### Solution 3: Library Loading Order Control

**Advanced technique**: Force your extension to establish the CUDA environment first.

```python
# In your package's __init__.py
import sys
import ctypes
import os

# Preload CUDA libraries with global symbols
original_flags = sys.getdlopenflags()
sys.setdlopenflags(original_flags | ctypes.RTLD_GLOBAL)

try:
    # Load your extension first to establish CUDA environment
    from . import your_extension_module
finally:
    # Restore flags before other imports
    sys.setdlopenflags(original_flags)

# Now safe to import PyTorch
```

### Solution 4: Wheel Bundling (Distribution)

**For package distribution**: Bundle compatible CUDA libraries directly.

```bash
# Use auditwheel to bundle dependencies
pip install auditwheel
python -m build --wheel
auditwheel repair dist/your_package-*.whl -w dist/

# Verify bundled libraries
auditwheel show dist/your_package-*-linux_x86_64.whl
```

## CMake Best Practices for CUDA Compatibility

### Portable CUDA Detection

```cmake
# Use FindCUDAToolkit (CMake 3.17+) for better portability
find_package(CUDAToolkit REQUIRED)

# Set RPATH to find CUDA libraries at runtime
set_target_properties(your_target PROPERTIES
    INSTALL_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Link against modern CUDA targets
target_link_libraries(your_target PRIVATE 
    CUDA::cudart 
    CUDA::nvrtc
    # CUDA::nvjitlink  # If explicitly needed
)
```

### Version-Aware Configuration

```cmake
# Check CUDA version compatibility
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "CUDA 12.0+ required for compatibility with modern PyTorch")
endif()

# Set appropriate compute capabilities
set_target_properties(your_target PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80;86;89;90"  # Adjust for your needs
)
```

## Debugging Tools Quick Reference

### Runtime Analysis
```bash
# Comprehensive library loading trace
LD_DEBUG=all your_program 2>&1 | grep -E "(nvjit|cuda)" > debug.log

# Symbol resolution debugging  
LD_DEBUG=symbols,bindings your_program 2>&1 | grep nvJit

# Library search path debugging
LD_DEBUG=libs your_program 2>&1 | grep "search path"
```

### Static Analysis
```bash
# Check library dependencies
ldd /path/to/your/extension.so
readelf -d /path/to/your/extension.so | grep -E "(RPATH|RUNPATH|NEEDED)"

# Symbol table analysis
nm -D /path/to/library.so | grep nvJit
objdump -T /path/to/library.so | grep nvJit

# Version information
objdump -p /path/to/library.so | grep VERSION
```

## Prevention Strategies

### 1. Version Matrix Testing

Test your extension against multiple PyTorch/CUDA combinations:

```yaml
# CI matrix example
strategy:
  matrix:
    pytorch-version: ["2.6.0", "2.7.0", "2.7.1"]
    system-cuda: ["12.4", "12.5", "12.6"]
    # Test backward compatibility boundaries
```

### 2. Environment Validation

Add runtime checks to your extension:

```python
def validate_cuda_environment():
    import torch
    
    # Get versions
    pytorch_cuda = tuple(map(int, torch.version.cuda.split('.')))
    
    # Your extension's required CUDA version (set during build)
    required_cuda = (12, 5)  # Example
    
    if pytorch_cuda < required_cuda:
        raise RuntimeError(
            f"PyTorch CUDA {pytorch_cuda} is older than required {required_cuda}. "
            f"Please upgrade PyTorch or rebuild extension with older CUDA."
        )
```

### 3. Documentation

Document CUDA compatibility requirements:

```markdown
## CUDA Compatibility

This package requires:
- System CUDA: 12.5+
- PyTorch with CUDA 12.5+ (e.g., PyTorch 2.7.1+)

### Compatibility Matrix
| PyTorch Version | CUDA Version | Compatible |
|----------------|--------------|------------|
| 2.5.1          | 12.4         | ❌ Too old |
| 2.7.1          | 12.6         | ✅ Works   |
```

## Common Pitfalls

1. **Assuming indefinite backward compatibility** - CUDA compatibility windows are limited
2. **Ignoring patch versions** - Sometimes critical for symbol availability  
3. **Not testing import order** - Issues often only appear when PyTorch loads first
4. **Hardcoding CUDA paths** - Breaks on different systems/containers
5. **Forgetting about Docker/containers** - CUDA driver compatibility differs from runtime

## Key Takeaways

1. **CUDA backward compatibility is directional** - newer runtime supports older compiled code, not vice versa
2. **Symbol versioning matters** - `__nvJitLinkCreate_12_5` vs `__nvJitLinkCreate_12_4` are different symbols
3. **PyTorch ships its own CUDA** - Don't assume it matches your system CUDA
4. **Library loading order affects symbol resolution** - First loaded library often wins
5. **LD_DEBUG is your best friend** - Shows exactly what's happening during library loading

The systematic approach we used here - identifying the version mismatch, understanding CUDA's backward compatibility rules, and upgrading to a compatible PyTorch version - is generally applicable to similar CUDA linking issues.
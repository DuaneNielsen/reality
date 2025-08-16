# CUDA Setup Guide

## Overview

This guide addresses common CUDA library issues encountered when running the Madrona Escape Room simulation in CUDA mode.

## CUDA Version Support

**Currently Supported**: CUDA 12.5 only

**Important Notes**:
- **CUDA 12.6**: Not viable due to an LLVM bug in NVRTC
- **CUDA 12.8**: Not currently supported - additional work required for compatibility
- **Other versions**: Untested and likely incompatible

## Problem Description

When running the headless executable with CUDA mode, you may encounter the following error:

```
nvrtc: error: failed to open libnvrtc-builtins.so.12.5.
Make sure that libnvrtc-builtins.so.12.5 is installed correctly.

Error at /home/duane/madrona/madrona_escape_room/external/madrona/src/mw/cpp_compile.cpp:100
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
```

## Root Cause

The issue occurs because:

1. **Dynamic Library Loading**: The NVRTC (NVIDIA Runtime Compilation) library dynamically loads `libnvrtc-builtins.so.12.5` at runtime, unlike other CUDA libraries that are linked at compile time.

2. **Specific Version Requirement**: The system is hardcoded to work with CUDA 12.5 specifically.

3. **Missing Library Path**: The required CUDA 12.5 libraries exist on the system but are not in the system's library search path.

## Prerequisites

Before proceeding, ensure you have CUDA 12.5 installed:

```bash
ls -la /usr/local/cuda-12.5/targets/x86_64-linux/lib/libnvrtc-builtins*
```

If CUDA 12.5 is not installed, you must install it before proceeding.

## Diagnosis Steps

### 1. Check Linked Libraries
```bash
ldd ./headless | grep -i cuda
```

This shows which CUDA libraries the executable is linked against and their paths.

### 2. Check Current Library Path
```bash
echo $LD_LIBRARY_PATH
```

### 3. Check System Library Cache
```bash
ldconfig -p | grep nvrtc-builtins
```

This shows which versions of `libnvrtc-builtins` are available in the system cache.

### 4. Verify CUDA 12.5 Installation
```bash
ls -la /usr/local/cuda-12.5/targets/x86_64-linux/lib/libnvrtc-builtins*
```

You should see files like:
```
libnvrtc-builtins.so -> libnvrtc-builtins.so.12.5
libnvrtc-builtins.so.12.5 -> libnvrtc-builtins.so.12.5.40
libnvrtc-builtins.so.12.5.40
```

## Solution: Update System Library Cache

### Step 1: Create CUDA 12.5 Library Configuration File

```bash
sudo bash -c 'echo "/usr/local/cuda-12.5/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda-12-5.conf'
```

### Step 2: Update Library Cache

```bash
sudo ldconfig
```

### Step 3: Verify the Fix

```bash
ldconfig -p | grep nvrtc-builtins
```

You should now see CUDA 12.5 libraries listed in the output:
```
libnvrtc-builtins.so.12.5 (libc6,x86-64) => /usr/local/cuda-12.5/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.5
```

### Step 4: Test CUDA Mode

```bash
./headless --cuda 0 --num-worlds 1 --num-steps 100 --rand-actions
```

## Alternative Solutions

### Temporary Fix: Environment Variable

If you don't have sudo access, you can temporarily add the CUDA 12.5 library path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
./headless --cuda 0 --num-worlds 1 --num-steps 100 --rand-actions
```

### Permanent User Fix: Shell Profile

Add to your `~/.bashrc` or `~/.profile`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## Common Mistakes

### Using Other CUDA Versions

**Do not** create configuration files for other CUDA versions:

```bash
# DON'T DO THIS - CUDA 12.6 has LLVM bugs
# sudo bash -c 'echo "/usr/local/cuda-12.6/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda-12-6.conf'

# DON'T DO THIS - CUDA 12.8 is not supported yet
# sudo bash -c 'echo "/usr/local/cuda-12.8/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda-12-8.conf'
```

### Using Generic CUDA Symlink

Avoid using `/usr/local/cuda/lib64` if it points to a non-12.5 version:

```bash
# Check what version the symlink points to
ls -la /usr/local/cuda
```

## Verification

After applying the fix, you should see output similar to:

```
Executing 100 Steps x 1 Worlds (CUDA)
Compiling GPU engine code:
Initialization finished
FPS: 1668
```

## Troubleshooting

### Issue: Wrong CUDA Version Installed
If you only have CUDA 12.6 or 12.8 installed, you must install CUDA 12.5:
1. Download CUDA 12.5 from NVIDIA's archive
2. Install it alongside existing versions
3. Follow the setup steps above

### Issue: Permission Denied
If you get permission errors when creating the configuration file, ensure you have sudo access or use the environment variable method.

### Issue: Still Getting Library Errors
1. Verify CUDA 12.5 is actually installed at `/usr/local/cuda-12.5/`
2. Check if the library file actually exists
3. Try the environment variable method as a test
4. Ensure no other CUDA paths are interfering in `LD_LIBRARY_PATH`

### Issue: Performance Differences
- CPU mode may be faster for small workloads (few worlds/steps)
- GPU mode scales better with larger workloads (many worlds)
- Initial GPU compilation adds overhead but subsequent runs are faster

## Future CUDA Support

To add support for newer CUDA versions (like 12.8):
1. Update the Madrona submodule to a version that supports the target CUDA version
2. Test compilation and runtime compatibility
3. Update build configuration if needed
4. Verify NVRTC functionality works correctly

CUDA 12.6 support is blocked by NVRTC LLVM bugs and should be avoided.

## Best Practices

1. **Stick to CUDA 12.5**: Don't attempt to use other versions
2. **System-wide Setup**: Use the `ldconfig` method for permanent, system-wide fixes
3. **Testing**: Always test both CPU and CUDA modes after setup changes
4. **Clean Environment**: Avoid mixing CUDA versions in environment variables
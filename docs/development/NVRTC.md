# NVRTC Performance Optimization Guide

## Overview

NVRTC (NVIDIA Runtime Compilation) is the just-in-time compiler that compiles CUDA C++ code to GPU kernels at runtime. In the Madrona Escape Room project, NVRTC compilation can take 30+ seconds on startup, making development and testing slow.

This guide covers optimization strategies to reduce NVRTC compilation time from 30+ seconds to under 2 seconds.

## Understanding the Compilation Chain

```
C++/CUDA → PTX → SASS → GPU Execution
         ↑      ↑      ↑
      NVRTC   Driver   Hardware
     (compile (JIT    (actual
      time)   time)   execution)
```

- **NVRTC**: Compiles C++ to PTX (Parallel Thread Execution) - the slow step
- **Driver JIT**: Compiles PTX to SASS (native GPU assembly) - relatively fast
- **PTX**: NVIDIA's intermediate assembly language, human-readable and portable

## Madrona Kernel Cache System

**Best Solution**: Madrona has a built-in kernel cache system that completely bypasses NVRTC recompilation.

### Environment Variables

```bash
# Enable kernel cache (stores compiled kernels on disk)
export MADRONA_MWGPU_KERNEL_CACHE=/path/to/cache/kernels.cache

# Use debug mode for faster initial compilation
export MADRONA_MWGPU_FORCE_DEBUG=1

# Enable verbose compilation output (for debugging)
export MADRONA_MWGPU_VERBOSE_COMPILE=1
```

### Performance Results

- **First run**: ~30 seconds (builds 58MB cache file)
- **Subsequent runs**: ~1.6 seconds (20x speedup!)
- **Cache persistence**: Survives between sessions

### Implementation in Test Tracker

The kernel cache is automatically enabled in `tests/test_tracker.py` for GPU tests:

```python
# In test_tracker.py for GPU tests
cache_path = self.base_dir / "build" / "madrona_kernels.cache"
env["MADRONA_MWGPU_KERNEL_CACHE"] = str(cache_path)
env["MADRONA_MWGPU_FORCE_DEBUG"] = "1"
```

## Alternative NVRTC Optimizations

If kernel caching isn't available, these options can help:

### 1. Compilation Mode Selection

```cpp
// In CompileConfig
CompileConfig cfg {
    .optMode = CompileConfig::OptMode::Debug     // Fastest compilation
    .optMode = CompileConfig::OptMode::Optimize  // Balanced
    .optMode = CompileConfig::OptMode::LTO       // Slowest (default)
};
```

### 2. CUDA Compilation Cache

```bash
# Enable CUDA's built-in compilation cache
export CUDA_CACHE_PATH=/dev/shm/cuda_cache  # Use RAM disk for speed
export CUDA_CACHE_MAXSIZE=4294967296        # 4GB cache limit
```

### 3. NVRTC Compilation Flags

Common flags for faster compilation:

```cpp
const char* fast_flags[] = {
    "--device-debug",           // Debug mode (faster compilation)
    "--use_fast_math",         // Fast math operations
    "--opt-level=1",           // Lower optimization level
    "--gpu-architecture=compute_75"  // Target specific architecture
};
```

## Debugging Compilation Issues

### Verbose Output

Enable verbose compilation to see what's being compiled:

```bash
export MADRONA_MWGPU_VERBOSE_COMPILE=1
./build/your_gpu_executable
```

This shows:
- Compiler flags being used
- Files being compiled
- Compilation time breakdown

### Common Issues

**Long compilation times:**
- Check if cache is enabled and writable
- Verify debug mode is enabled for development
- Ensure cache path exists and has sufficient space

**Cache not working:**
- Verify `MADRONA_MWGPU_KERNEL_CACHE` points to writable location
- Check that cache file isn't corrupted (delete to regenerate)
- Ensure consistent compilation flags between runs

## Integration Checklist

To enable kernel caching in your application:

1. **Set environment variable**:
   ```bash
   export MADRONA_MWGPU_KERNEL_CACHE=/path/to/cache.cache
   ```

2. **Enable debug mode for development**:
   ```bash
   export MADRONA_MWGPU_FORCE_DEBUG=1
   ```

3. **Verify cache creation**:
   - First run creates ~58MB cache file
   - Subsequent runs skip "Compiling GPU engine code" message

4. **Add to CI/build scripts**:
   ```bash
   mkdir -p build/cache
   export MADRONA_MWGPU_KERNEL_CACHE=build/cache/kernels.cache
   ```

## Technical Details

### Cache File Format

The kernel cache stores:
- Compiled GPU kernels (PTX and binary)
- Compilation metadata and checksums
- Source code hashes for invalidation

### Cache Invalidation

Cache is invalidated when:
- Source code changes (detected via hash)
- Compilation flags change
- CUDA version changes
- GPU architecture changes

### Memory Usage

- **Cache file**: ~58MB on disk
- **Runtime memory**: Minimal additional overhead
- **Build memory**: Same as without cache

## Troubleshooting

### Cache Permission Issues
```bash
# Fix permissions
chmod 644 /path/to/kernels.cache
chown $USER:$USER /path/to/kernels.cache
```

### Cache Corruption
```bash
# Delete and regenerate
rm /path/to/kernels.cache
# Next run will recreate cache
```

### Still Slow After Cache?
- Check if correct environment variables are set
- Verify cache file exists and is recent
- Enable verbose output to see if cache is being used

## Performance Monitoring

Track compilation performance:

```bash
# Time first run (cache creation)
time ./your_gpu_executable

# Time second run (cache usage)  
time ./your_gpu_executable
```

Expected results:
- First run: 30+ seconds
- Second run: <2 seconds
- Cache file: ~58MB created

## References

- [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Madrona CompileConfig Reference](../../../external/madrona/include/madrona/mw_gpu.hpp)
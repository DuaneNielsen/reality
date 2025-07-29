# CFFI Bindings for Madrona Escape Room

## Overview
This document describes the CFFI-based Python bindings that replace the original nanobind bindings. The CFFI bindings were created to avoid RTTI and exception requirements that are incompatible with the Madrona engine.

## Architecture

### Components
1. **C API Wrapper** (`include/madrona_escape_room_c_api.h`, `src/madrona_escape_room_c_api.cpp`)
   - Pure C interface wrapping the C++ Manager class
   - No RTTI or exceptions used
   - Error handling via return codes
   - Opaque handle pattern for C++ objects

2. **CFFI Module** (`scripts/build_cffi.py`)
   - Builds Python extension using CFFI
   - Parses C header and generates bindings
   - Links against C wrapper library

3. **Python Wrapper** (`madrona_escape_room/__init__.py`)
   - Provides same API as nanobind version
   - Implements zero-copy tensor access
   - Handles library loading and environment setup

## Key Features

### Zero-Copy Tensor Access
- Direct memory access without copying data
- Support for both CPU and GPU tensors
- PyTorch integration maintains zero-copy semantics

### Bundled Dependencies
The package bundles all required shared libraries:
- `libmadrona_escape_room_c_api.so` - Main C wrapper
- `libembree4.so.4` - Intel Embree ray tracing
- `libdxcompiler.so` - DirectX shader compiler
- `libmadrona_render_shader_compiler.so` - Madrona shader compiler
- `libmadrona_std_mem.so` - Madrona memory management

### Error Handling
- C API uses error codes instead of exceptions
- Python wrapper converts error codes to exceptions
- All errors properly propagated to Python

## Building

```bash
# Build everything
uv run python setup_build.py

# Install package
uv pip install -e .
```

## API Compatibility
The CFFI bindings maintain full API compatibility with the original nanobind version:

```python
import madrona_escape_room

# Create manager
mgr = madrona_escape_room.SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=1024,
    rand_seed=42,
    auto_reset=True
)

# Access tensors (zero-copy)
actions = mgr.action_tensor().to_torch()
observations = mgr.self_observation_tensor().to_torch()
rewards = mgr.reward_tensor().to_torch()

# Run simulation
mgr.step()
```

## Testing
The CFFI bindings pass all existing tests:
```bash
# Run CPU tests
uv run --extra test pytest tests/python/ -v --no-gpu

# Run GPU tests (after CPU tests pass)
uv run --extra test pytest tests/python/ -v -k "gpu"
```

## Performance
The CFFI bindings maintain zero-copy semantics and have minimal overhead compared to the original nanobind implementation.
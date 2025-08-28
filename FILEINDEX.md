# File Index

## Core C++ Source Files (src/)

**Core Simulation**
- `src/consts.hpp` - game constants, action values, physics parameters, rendering settings
- `src/types.hpp` - ECS component definitions, CompiledLevel structure, Action/Observation types
- `src/mgr.cpp` - simulation manager, asset loading, tensor exports, recording/replay functionality
- `src/mgr.hpp` - manager class interface, configuration, tensor access methods
- `src/sim.cpp` - ECS simulation core, systems registration, task graph, physics integration 
- `src/sim.hpp` - simulation class, world state, export IDs, engine context

**Level Generation**
- `src/level_gen.cpp` - world generation from compiled level data, entity creation
- `src/level_gen.hpp` - level generation function declarations

**Asset Management**
- `src/asset_ids.hpp` - asset ID constants for objects (cube, wall, agent, plane, etc.)
- `src/asset_registry.hpp` - asset information registry and loading utilities
- `src/asset_data.cpp` - static asset data and material definitions

**Applications**
- `src/headless.cpp` - command-line headless simulation runner with recording/replay
- `src/viewer.cpp` - interactive 3D viewer with controls and recording
- `src/viewer_core.cpp` - core viewer logic separated from UI
- `src/viewer_core.hpp` - viewer state machine and functionality

**Utilities**
- `src/dlpack_extension.cpp` - DLPack tensor format support for Python interop
- `src/madrona_escape_room_c_api.cpp` - C API wrapper for Python bindings
- `src/struct_export.cpp` - C++ struct to Python export utilities
- `src/file_inspector.cpp` - level file inspection tool
- `src/default_level.cpp` - default level generation for testing
- `src/replay_loader.hpp` - replay file format and loading utilities
- `src/test_c_wrapper_cpu.cpp` - C API CPU testing program
- `src/test_c_wrapper_gpu.cpp` - C API GPU testing program

## Python Package (madrona_escape_room/)

**Main Interface**
- `madrona_escape_room/__init__.py` - package entry point, imports SimManager and constants
- `madrona_escape_room/manager.py` - SimManager class, simulation lifecycle, tensor access
- `madrona_escape_room/ctypes_bindings.py` - low-level C API bindings using ctypes
- `madrona_escape_room/tensor.py` - tensor wrapper for PyTorch integration

**Level Creation**
- `madrona_escape_room/default_level.py` - default level generator for Python
- `madrona_escape_room/level_compiler.py` - ASCII to CompiledLevel converter

**Generated Files**
- `madrona_escape_room/generated_constants.py` - auto-generated constants from C++ headers
- `madrona_escape_room/generated_dataclasses.py` - auto-generated Python dataclasses
- `madrona_escape_room/generated_structs.py` - auto-generated struct bindings

**Utilities**
- `madrona_escape_room/dataclass_utils.py` - utilities for working with dataclasses
- `madrona_escape_room/level_compiler_old.py` - legacy level compiler

## Code Generation (codegen/)

- `codegen/generate_dataclass_structs.py` - generates Python dataclasses from binary using pahole
- `codegen/generate_python_constants.py` - generates constants from C++ headers using libclang
- `codegen/dump_ast.py` - debugging tool for AST inspection

## Build Configuration

- `CMakeLists.txt` - main CMake build configuration
- `src/CMakeLists.txt` - source build configuration with GPU/CPU compilation

## Development Documentation

- `docs/development/CPP_CODING_STANDARDS.md` - C++ coding standards and GPU/CPU code separation
- `docs/development/TESTING_GUIDE.md` - Python testing guide with pytest
- `docs/development/CPP_TESTING_GUIDE.md` - C++ testing with GoogleTest
- `docs/development/GDB_GUIDE.md` - debugging guide using GDB
- `docs/architecture/ECS_ARCHITECTURE.md` - Entity Component System architecture
- `docs/tools/VIEWER_GUIDE.md` - interactive viewer usage guide
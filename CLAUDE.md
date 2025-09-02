# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious
- Engineering safety:  Never fail to mention failing tests, even if they are incidental to the current object, failing tests are always a problem and should never be overlooked.

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

#includes
@src/const.hpp
@madrona_escape_room/generated_constants.py
@FILEINDEX.md
@docs/development/CPP_CODING_STANDARDS.md

## Overview

This is a Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine. It implements a navigation environment where agents explore and try to maximize their forward progress through the world.

**Tech Stack:**

- C++ (core simulation using Entity Component System pattern)
- Python (PyTorch-based PPO training)
- CMake build system

# Headless Mode Quick Reference

Headless mode runs simulation without graphics for benchmarking, testing, or server deployment.

## to see the options run

```bash
./build/headless --help
```

# Essential Commands

### Building the project

use the project-builder subagent to build the project

Manual build command (fast, parallel, quiet):
```bash
make -C build -j16 -s
```

### Running the Simulation

to visualize the simulation for the user, run the viewer, it will create a playable simulation that the user can use to interact eith the simulation
read the viewer guide in [VIEWER_GUIDE.md](docs/tools/VIEWER_GUIDE.md) for commands on how to run the viewer

### Running python code

ALWAYS use uv when running python code or managing packages

```
uv run python ...
```

```
uv run python default_level.py 
```

### Development

#### Python coding standards

IMPORTANT: when writing tests for python code ALWAYS use the pytest framework by writing the test in the tests/python directory, and following pytest conventions

GOOD:

test/python/test_my_code.py

BAD:

test_my_code.py <---- DO NOT DO THIS

IMPORTANT: NEVER use sys.path.insert(..) ALWAYS place modules in the correct locations so they can be imported correctly by python

#### C++ coding standards

# Testing

### Unit tests

This project uses pytest for python tests and GoogleTest for C++ tests
The pytests are documented in [TESTING_GUIDE.md](docs/development/TESTING_GUIDE.md)
c++ unit tests are documented in [CPP_TESTING_GUIDE.md](tests/cpp/README.md)

IMPORTANT:  Whenever tests fail, ALWAYS report it using the standardized format below.  We may decide to take no action, but we should ALWAYS ALWAYS mention the failure in the correct format first.

### Test Output Format

**CRITICAL FOR CONTEXT PRESERVATION**: This exact format enables test identification and debugging after context resets. Without precise test paths (`test_file.py::test_name`) and reproduction commands, failing tests cannot be located or fixed in future sessions.

When reporting test results, use this exact format immediately:

**"Do NOT make up test names - only use the real test names from the actual pytest output."**

"When reporting test results, you MUST use this EXACT format with actual test file names and test function names from the real test output:"

## Test Results Summary

### ✅ C++ Tests: X/X passed

### ❌ Python Tests: X passed, X failed, X skipped

**Failed Tests:**

- ❌ `test_file.py::test_name` - Brief error description
- ❌ `test_file.py::test_name` - Brief error description

**Reproduction:**

```bash
uv run --group dev pytest tests/python/test_file.py::test_name -v
```

### Example Test Output:

## Test Results Summary

### ✅ C++ Tests: 18/18 passed

### ❌ Python Tests: 145 passed, 2 failed, 15 skipped

**Failed Tests:**

- ❌ `test_bindings.py::test_deterministic_actions` - Position comparison failure
- ❌ `test_spawn_locations.py::test_single_spawn_center` - Wrong spawn X: -20.0 (expected -6.25)

**Reproduction:**

```bash
uv run --group dev pytest tests/python/test_bindings.py::test_deterministic_actions -v
uv run --group dev pytest tests/python/test_spawn_locations.py::test_single_spawn_center -v
```

### Why This Format Is Mandatory

1. **Context Reset Recovery**: After context resets, only exact test paths allow continuation of debugging work
2. **Direct Reproduction**: The bash commands can be copied and run without modification
3. **150+ Test Navigation**: With 150+ Python tests and 18 C++ tests, vague descriptions like "tensor tests failed" are useless

**Without this format**: Information needed to fix tests is permanently lost when context resets occur.



### Unit test best practice

**IMPORTANT**: when writing unit tests use the cpu_manger fixtures in conftest.py, DO NOT call the SimManager directly, unless you are specfically testing the SimManager initialization itself
**IMPORTANT**: when creating plans to implement new features, the plan must ALWAYS include tests in the appropriate unit test framework

### Rules when fixing CPP tests

**IMPORTANT**:  After you make c++ code changes, **you must call the build agent**to build the project!

### Debugging using GDB

If the user asks to "debug the code", or "debug it" or generally references "debugging" then interpret this as a request to use the GDB tool to gather information about the behaviour of the program, and follow the following procedure

1. read the file [GDB_GUIDE.md](docs/development/GDB_GUIDE.md)
2. use the debug tool in your MCP library to gather information on the problem at hand, or study the code

# Documentation

## Documentation creation rules

- when proposing new documents, always add them to the docs folder
- when the user asks save a plan to a file it should be written to docs\plan_dump

# Scratch files

When creating a python file that we don't want to keep in the repo, but just want to use for the purposes of a one-time test or to study/understand code better.  Create it in the ./scratch folder. 

# Code Classification System

The codebase uses a three-tier classification system to help developers understand what needs to be modified:

### [BOILERPLATE]

Pure Madrona framework code that should never be changed. This includes:

- CPU/GPU execution infrastructure
- Memory management systems
- Rendering pipeline setup
- Base class structures

### [REQUIRED_INTERFACE]

Methods and structures that every Madrona environment must implement:

- `loadPhysicsObjects()` - Load collision meshes and configure physics
- `loadRenderObjects()` - Load visual assets and materials
- `triggerReset()` - Reset episode state (for episodic environments)
- `setAction()` - Accept actions from the policy
- Tensor export methods - Define observation/action spaces
- Reset and action buffers - Required for episodic RL

### [GAME_SPECIFIC]

Implementation details unique to this escape room game:

- Action structure fields (moveAmount, moveAngle, rotate, grab)
- Observation tensor types and shapes
- Object types and their physics properties
- Material colors and textures
- Game constants (see Game-Specific Constants section below)

When creating a new environment:

1. Keep all `[BOILERPLATE]` code unchanged
2. Implement all `[REQUIRED_INTERFACE]` methods with your game's logic
3. Replace all `[GAME_SPECIFIC]` code with your game's details

# ECS Architecture

If you need to make modifications to the ECS system, read the documentation in [ECS_ARCHITECTURE.md](docs/architecture/ECS_ARCHITECTURE.md)

# Source Code File Descriptions

## Core Source Files (./src/)

- **CMakeLists.txt**: Build configuration for libraries, executables, and linking dependencies
- **consts.hpp**: Game constants and action value definitions for discrete agent controls
- **dlpack_extension.cpp**: DLPack tensor format support for Python tensor interoperability
- **headless.cpp**: Command-line headless simulation runner with recording/replay capabilities
- **level_gen.cpp**: Level generation from ASCII/compiled level data and entity creation
- **level_gen.hpp**: Level generation function declarations
- **madrona_escape_room_c_api.cpp**: C API wrapper for Python bindings using ctypes
- **mgr.cpp**: Manager class implementation for simulation lifecycle and tensor exports
- **mgr.hpp**: Manager class header with configuration and interface declarations
- **replay_metadata.hpp**: Replay file format structures and utility functions
- **sim.cpp**: Core ECS simulation with system registration and task graph setup
- **sim.hpp**: Simulation class and enum definitions (TaskGraphID, ExportID, SimObject)
- **sim.inl**: Template implementations for renderable entity creation/destruction
- **test_c_wrapper_cpu.cpp**: C API test program for CPU execution validation
- **test_c_wrapper_gpu.cpp**: C API test program with embedded level data for GPU testing
- **types.hpp**: ECS component definitions and archetype structures
- **viewer.cpp**: Interactive 3D viewer with recording/replay and agent control
- **viewer_core.cpp**: Testable viewer logic separated from UI concerns
- **viewer_core.hpp**: Header for viewer state machine and core functionality
- **viewer_orig.cpp**: Original viewer implementation (legacy)

## Scripts (./scripts/)

- **run-clangd-tidy.sh**: Fast static analysis and magic number detection

## Python Package Files (./madrona_escape_room/madrona_escape_room/)

- **__init__.py**: Main Python package entry point with SimManager class and ctypes bindings
- **__init__.pyi**: Type stubs for Python package public API and SimManager interface
- **ctypes_bindings.py**: Low-level ctypes wrapper for C API library loading and function bindings
- **level_compiler.py**: ASCII to CompiledLevel converter for test-driven level creation
- **madrona.pyi**: Type stubs for Madrona framework compatibility layer

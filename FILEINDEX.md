# File Index

## Core C++ Source Files (src/)

src/asset_data.cpp - static asset data and material definitions
src/camera_controller.cpp - free-fly and tracking camera implementations
src/default_level.cpp - procedural level generation with walls and obstacles
src/dlpack_extension.cpp - DLPack tensor format support for Python interop
src/file_inspector.cpp - CLI tool for inspecting recording and level files
src/headless.cpp - command-line headless simulation runner with recording/replay
src/level_gen.cpp - level generation from ASCII/compiled data and entity creation
src/level_io.cpp - compiled level file I/O and binary format parsing
src/madrona_escape_room_c_api.cpp - C API wrapper for Python bindings using ctypes
src/mgr.cpp - manager class implementation for simulation lifecycle and tensor exports
src/sim.cpp - core ECS simulation with system registration and task graph setup
src/struct_export.cpp - dummy file to export API boundary structures for pahole extraction
src/viewer_core.cpp - testable viewer logic separated from UI concerns
src/viewer.cpp - interactive 3D viewer with recording/replay and agent control

## C++ Header Files (src/)

src/asset_ids.hpp - asset ID constants and asset reference system
src/asset_registry.hpp - asset data structures and material definitions
src/camera_controller.hpp - camera control interface with multiple controller types
src/consts.hpp - game constants and action value definitions
src/level_gen.hpp - level generation function declarations
src/level_io.hpp - compiled level file I/O function declarations
src/mgr.hpp - manager class header with configuration and interface declarations
src/sim.hpp - simulation class and enum definitions (TaskGraphID, ExportID, SimObject)
src/types.hpp - ECS component definitions and archetype structures
src/viewer_core.hpp - header for viewer state machine and core functionality

## Python Package Files (madrona_escape_room/)

madrona_escape_room/ctypes_bindings.py - low-level ctypes wrapper for C API library loading
madrona_escape_room/dataclass_utils.py - utility functions for creating properly initialized dataclass instances
madrona_escape_room/default_level.py - default level generator creating 16x16 rooms with walls
madrona_escape_room/generated_constants.py - auto-generated Python constants from C++ headers
madrona_escape_room/generated_dataclasses.py - auto-generated Python dataclasses from C++ structs
madrona_escape_room/generated_madrona_constants.py - auto-generated madrona framework constants
madrona_escape_room/__init__.py - main Python package entry point with SimManager class
madrona_escape_room/level_compiler.py - ASCII to CompiledLevel converter for test-driven level creation
madrona_escape_room/level_io.py - binary level file I/O using C API
madrona_escape_room/manager.py - Python interface to C++ simulation manager
madrona_escape_room/sensor_config.py - sensor configuration presets for RGB depth and lidar sensors
madrona_escape_room/tensor.py - tensor wrapper providing zero-copy access to C API tensors

## Code Generation Scripts (codegen/)

codegen/dump_ast.py - dumps libclang AST for debugging constant extraction
codegen/generate_dataclass_structs.py - generates Python dataclasses from C++ structs using pahole
codegen/generate_python_constants.py - generates Python constants from C++ headers using libclang

## Documentation and Specs (docs/)

docs/specs/episodic_ema_tracker.md - specification for episodic EMA tracking
docs/specs/mgr.md - manager module specification and API documentation
docs/specs/sim.md - simulation module specification and architecture

## Test Files (tests/)

tests/README.md - testing procedures and test execution guide
tests/cpp/README.md - C++ unit testing guide with GoogleTest
tests/python/README.md - Python testing guide with pytest

## Training Package (train_src/madrona_escape_room_learn/)

train_src/madrona_escape_room_learn/action.py - discrete action distributions for multi-component actions
train_src/madrona_escape_room_learn/actor_critic.py - neural network architecture for policy and value functions
train_src/madrona_escape_room_learn/amp.py - automatic mixed precision training utilities
train_src/madrona_escape_room_learn/cfg.py - configuration classes for training and simulation
train_src/madrona_escape_room_learn/__init__.py - training package initialization
train_src/madrona_escape_room_learn/learning_state.py - state management for training process
train_src/madrona_escape_room_learn/models.py - neural network model definitions
train_src/madrona_escape_room_learn/moving_avg.py - exponential moving average utilities
train_src/madrona_escape_room_learn/profile.py - performance profiling decorators
train_src/madrona_escape_room_learn/rnn.py - recurrent neural network components
train_src/madrona_escape_room_learn/rollouts.py - experience collection and rollout management
train_src/madrona_escape_room_learn/sim_interface_adapter.py - adapter for simulation interface compatibility
train_src/madrona_escape_room_learn/train.py - main PPO training loop implementation

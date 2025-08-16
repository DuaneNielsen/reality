# Python Test Suite

This directory contains the Python test suite for the Madrona Escape Room environment. The tests use pytest and cover core functionality, bindings, level compilation, and various simulation features.

## Test Structure

### Test Configuration
- **conftest.py** - pytest configuration with fixtures for CPU/GPU managers, recording capabilities, and replay testing

### Core Functionality Tests
- **test_00_ctypes_cpu_only.py** - Import order independence tests between PyTorch and madrona_escape_room
- **test_bindings.py** - Comprehensive Python bindings test covering tensor operations, simulation steps, and GPU functionality  
- **test_ctypes_basic.py** - Basic ctypes bindings test for CPU/GPU manager creation

### Level System Tests
- **test_ascii_level_compiler.py** - ASCII level compilation pipeline from ASCII art to C API structs
- **test_level_compiler_c_api.py** - Level compiler C API integration tests
- **test_spawn_locations.py** - Tests for agent spawn point validation and positioning

### Simulation System Tests
- **test_movement_system.py** - Agent movement including forward movement, strafing, and rotation
- **test_reward_system.py** - Reward system testing for Y-progress based scoring at episode end

### Recording and Replay Tests
- **test_native_recording.py** - Native recording functionality tests
- **test_native_recording_gpu.py** - GPU-specific recording tests
- **test_native_recording_replay_roundtrip.py** - Recording/replay roundtrip validation
- **test_native_replay.py** - Replay system functionality tests

### Tensor and Memory Tests
- **test_dlpack.py** - DLPack tensor format interoperability
- **test_dlpack_implementation.py** - DLPack implementation details
- **test_simple_dlpack.py** - Basic DLPack functionality
- **test_ctypes_dlpack.py** - DLPack integration with ctypes
- **test_zero_copy.py** - Zero-copy tensor operations

### API Validation Tests
- **test_c_api_struct_validation.py** - C API struct validation and memory layout
- **test_helpers.py** - Test utilities for agent control and observation reading

### Environment Integration Tests
- **test_env_wrapper.py** - Environment wrapper functionality
- **test_torchrl_env_wrapper.py** - TorchRL environment wrapper tests
- **test_torchrl_zero_copy.py** - TorchRL zero-copy tensor operations

### MCP Integration Tests
- **test_fastmcp_madrona_repl.py** - Fast MCP Madrona REPL integration tests

### Test Data
- **data/reward_test.bin** - Binary test data for reward system validation

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest tests/python/

# Run specific test file
pytest tests/python/test_bindings.py

# Run with verbose output
pytest tests/python/ -v
```

### Recording and Visualization
```bash
# Record actions during tests for later replay
pytest tests/python/ --record-actions

# Record and automatically launch viewer
pytest tests/python/ --record-actions --visualize

# Enable trajectory logging to files
pytest tests/python/ --trace-trajectories
```

### GPU Testing
```bash
# Skip GPU tests if CUDA unavailable
pytest tests/python/ --no-gpu

# Run only GPU tests
pytest tests/python/ -k "gpu"
```

## Key Features

### Fixtures
- **cpu_manager** - Function-scoped CPU SimManager with optional recording
- **gpu_manager** - Session-scoped GPU SimManager  
- **log_and_verify_replay_cpu_manager** - CPU manager with automatic replay verification
- **test_manager_from_replay** - Factory for creating managers from replay files

### Test Markers
- Use `@pytest.mark.custom_level("level_ascii")` to provide custom levels to tests

### Recording Capabilities
Tests can optionally record actions and trajectories:
- Actions saved to `.bin` files for viewer replay
- Trajectories logged to `.txt` files for analysis
- Automatic replay verification in specialized fixtures

### GPU Support
GPU tests automatically skip when CUDA is unavailable and can be explicitly disabled with `--no-gpu`.

## Test Organization

Tests are organized by functionality:
- **Core functionality** - Basic imports, manager creation, tensor operations
- **Level system** - ASCII compilation, spawn validation, level loading
- **Simulation** - Movement, rewards, physics, episode management  
- **Integration** - Recording/replay, environment wrappers, external APIs
- **Performance** - Memory layout, zero-copy operations, DLPack compatibility

Each test file focuses on a specific subsystem and includes both positive and negative test cases where appropriate.
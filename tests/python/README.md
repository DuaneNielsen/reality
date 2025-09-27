# Python Test Suite

This directory contains the Python test suite for the Madrona Escape Room environment with **33 test files**. The tests use pytest and cover core functionality, bindings, level compilation, simulation features, recording/replay, and comprehensive system validation.

## Key Constraint: GPU Manager Limitation

**Critical**: Madrona only supports **one GPU manager at a time** per process.

```python
# ❌ Multiple GPU managers - will fail
mgr1 = SimManager(exec_mode=ExecMode.CUDA, ...)
mgr2 = SimManager(exec_mode=ExecMode.CUDA, ...)  # FAILS

# ✅ Use session fixture instead  
def test_gpu_feature(gpu_manager):
    mgr = gpu_manager  # Shared session manager
```

**Why**: CUDA static initialization causes deadlocks with multiple GPU managers.

## Test Structure

### Test Configuration
- **conftest.py** - pytest configuration with fixtures for CPU/GPU managers, recording capabilities, and replay testing

### Core Functionality Tests
- **test_00_ctypes_cpu_only.py** - Import order independence tests between PyTorch and madrona_escape_room
- **test_bindings.py** - Comprehensive Python bindings test covering tensor operations, simulation steps, and GPU functionality  
- **test_ctypes_basic.py** - Basic ctypes bindings test for CPU/GPU manager creation
- **test_asset_descriptors_api.py** - C API functions for physics and render asset descriptors

### Level System Tests
- **test_ascii_level_compiler.py** - ASCII level compilation pipeline from ASCII art to C API structs
- **test_level_compiler_c_api.py** - Level compiler C API integration tests
- **test_spawn_locations.py** - Tests for agent spawn point validation and positioning
- **test_world_boundaries.py** - World boundary calculations for different level sizes and scales
- **test_tileset_compiler.py** - Tileset compilation functionality
- **test_default_level_binary_comparison.py** - Binary comparison of default level generation

### Simulation System Tests
- **test_movement_system.py** - Agent movement including forward movement, strafing, and rotation
- **test_reward_system.py** - Reward system testing for Y-progress based scoring at episode end
- **test_collision_termination.py** - Collision-based episode termination with per-tile collision flags
- **test_done_tensor_reset.py** - Done tensor reset behavior and episode length handling
- **test_comprehensive_reset_episode_counter_done_flag.py** - Comprehensive testing of reset/episode counter/done flag interactions

### Recording and Replay Tests
- **test_native_recording.py** - Native recording functionality tests
- **test_native_recording_gpu.py** - GPU-specific recording tests
- **test_native_recording_replay_roundtrip.py** - Recording/replay roundtrip validation using checksum verification
- **test_native_replay.py** - Replay system functionality tests
- **test_recording_binary_format.py** - Binary recording format validation
- **test_checksum_verification.py** - Comprehensive checksum-based determinism validation

### Tensor and Memory Tests
- **test_dlpack.py** - DLPack tensor format interoperability
- **test_dlpack_implementation.py** - DLPack implementation details
- **test_simple_dlpack.py** - Basic DLPack functionality
- **test_ctypes_dlpack.py** - DLPack integration with ctypes
- **test_zero_copy.py** - Zero-copy tensor operations

### API Validation Tests
- **test_c_api_struct_validation.py** - C API struct validation and memory layout
- **test_helpers.py** - Test utilities for agent control and observation reading
- **test_custom_level_fixture.py** - Custom level fixture functionality testing

### Environment Integration Tests
- **test_env_wrapper.py** - Environment wrapper functionality
- **test_torchrl_env_wrapper.py** - TorchRL environment wrapper tests
- **test_torchrl_zero_copy.py** - TorchRL zero-copy tensor operations

### Performance and Stress Tests
- **test_stress.py** - Stress testing with configurable iterations and world counts

### MCP Integration Tests
- **test_fastmcp_madrona_repl.py** - Fast MCP Madrona REPL integration tests

### Test Data
- **data/reward_test.bin** - Binary test data for reward system validation

## Test Execution Order

```bash
# 1. CPU tests first
uv run --group dev pytest tests/python/ -v --no-gpu

# 2. GPU tests after CPU tests pass
uv run --group dev pytest tests/python/ -v -k "gpu"
```

## Fixtures

### Main Test Fixtures
- **cpu_manager** - Function-scoped CPU SimManager (new per test, supports custom levels)
- **gpu_manager** - Session-scoped GPU SimManager (shared across all GPU tests)
- **gpu_env** - Session-scoped GPU TorchRL environment
- **log_and_verify_replay_cpu_manager** - CPU manager with automatic checksum-based replay verification
- **test_manager_from_replay** - Factory for creating managers from replay files

## Custom Level Testing

The `cpu_manager` fixture supports custom ASCII levels using the `@pytest.mark.ascii_level()` decorator:

### Basic Usage
```python
# Define ASCII level with walls and spawn points
TEST_LEVEL = """
##########
#S.......#
#........#
#...#....#
#...#....#
#........#
##########
"""

@pytest.mark.ascii_level(TEST_LEVEL)
def test_with_custom_level(cpu_manager):
    """Test uses the custom level instead of default"""
    mgr = cpu_manager  # Manager initialized with TEST_LEVEL
    # Test your custom level behavior
```

### Key Features
- **Spawn Points**: Mark agent spawn locations with 'S' 
- **Walls**: Use '#' for walls and obstacles
- **Empty Space**: Use '.' for walkable areas
- **Auto-compilation**: Levels are automatically compiled with scale=2.5

### Example: Testing Movement Constraints
```python
# Level with narrow corridor
CORRIDOR_LEVEL = """
################################
#S.............................#
#..............................#
#############........###########
#..............................#
################################
"""

@pytest.mark.ascii_level(CORRIDOR_LEVEL)
def test_corridor_navigation(cpu_manager):
    """Test agent navigation through narrow passage"""
    mgr = cpu_manager
    # Agent must navigate through the gap in the wall
```

### Notes
- Only works with `cpu_manager` fixture (function-scoped)
- Module-scoped fixtures don't support per-test custom levels
- GPU tests should not use custom levels (violates single GPU manager constraint)
- JSON levels can also be used with `@pytest.mark.json_level()` for more complex level definitions

## Running Tests

### Basic Test Execution
```bash
# Run all tests
uv run --group dev pytest tests/python/

# Run specific test file
uv run --group dev pytest tests/python/test_bindings.py

# Run with verbose output
uv run --group dev pytest tests/python/ -v

# Run quick tests (skip slow stress tests and GPU tests)  
uv run --group dev pytest tests/python/ -m "not slow"
```

### Test Flags

#### Basic Flags
```bash
--no-gpu              # Skip all GPU tests
```

#### Debugging and Visualization Flags
```bash
--record-actions      # Record actions to binary files for viewer replay
--visualize           # Auto-launch viewer after tests (requires --record-actions)
```

#### Usage Examples
```bash
# Record actions for debugging
uv run --group dev pytest tests/python/test_reward_system.py --record-actions

# Record actions and auto-launch viewer
uv run --group dev pytest tests/python/test_reward_system.py --record-actions --visualize

# Debug specific test
uv run --group dev pytest tests/python/test_reward_system.py::test_forward_movement_reward -v --record-actions
```

### Output Files
When using debugging flags, files are created in `test_recordings/` with automatic naming:
- **Actions**: `test_recordings/test_module.py__test_function_actions.bin`

Example:
```bash
uv run --group dev pytest tests/python/test_reward_system.py::test_forward_movement_reward --record-actions
# Creates:
# test_recordings/test_reward_system.py__test_forward_movement_reward_actions.bin
```

### Checksum-Based Replay Verification

The test suite now uses **checksum verification** for determinism validation instead of trajectory file comparison:

#### Key Benefits
- **Simpler**: Single boolean flag vs complex file comparison
- **Faster**: No file I/O for trajectory traces
- **More reliable**: Built into recording format vs external files
- **Automatic**: Checksums calculated every 200 steps during replay
- **Self-contained**: No manual file management required

#### Features
- **Automatic verification**: Tests using `log_and_verify_replay_cpu_manager` automatically verify replay determinism
- **Built-in checksums**: v4 format recordings embed position checksums every 200 steps
- **Simple API**: Check determinism with `mgr.has_checksum_failed()` boolean flag
- **Multi-world support**: Verifies determinism across all simulation worlds
- **Corruption detection**: Detects non-deterministic replays or corrupted files

#### Usage Examples
```python
# Modern approach - automatic checksum verification
def test_deterministic_replay(log_and_verify_replay_cpu_manager):
    mgr = log_and_verify_replay_cpu_manager
    # Run your test actions...
    # Fixture automatically verifies determinism on exit using checksums

# Manual checksum verification
def test_manual_checksum_check(cpu_manager):
    mgr = cpu_manager
    mgr.start_recording("recording.bin")
    # ... run simulation ...
    mgr.stop_recording()

    replay_mgr = SimManager.from_replay("recording.bin", ExecMode.CPU)
    # Run replay...
    assert not replay_mgr.has_checksum_failed(), "Replay should be deterministic"
```

#### Migration from Trajectory Verification
The previous trajectory logging approach has been replaced:
- ❌ **Old**: Complex trajectory file comparison, manual file management
- ✅ **New**: Simple boolean flag, automatic checksum verification
- **Tests**: Comprehensive checksum verification in `test_checksum_verification.py`

### GPU Testing
```bash
# Skip GPU tests if CUDA unavailable
uv run --group dev pytest tests/python/ --no-gpu

# Run only GPU tests (~11 GPU test files, expect ~60s NVRTC compilation on first GPU test)
uv run --group dev pytest tests/python/ -k "gpu" -v

# GPU tests are marked as 'slow' due to NVRTC compilation time
uv run --group dev pytest tests/python/ -m "slow" -k "gpu"
```

### Performance and Stress Testing
```bash
# Run only stress tests
uv run --group dev pytest tests/python/ -k "stress" -v

# Run all stress tests including slow ones
uv run --group dev pytest tests/python/test_stress.py -v

# Stress test with action recording for analysis
uv run --group dev pytest tests/python/test_stress.py --record-actions -v
```

## Writing GPU Tests

### Correct Pattern
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_functionality(gpu_manager):
    """Test GPU features using shared manager"""
    mgr = gpu_manager
    
    # Use the shared manager for testing
    obs_tensor = mgr.self_observation_tensor()
    assert obs_tensor.isOnGPU()
```

### Common Mistakes
```python
# ❌ Creates new GPU manager - will conflict
def test_bad_gpu_test():
    mgr = SimManager(exec_mode=ExecMode.CUDA, ...)  # FAILS

# ❌ Missing pytest.mark.skipif for CUDA
def test_missing_skip(gpu_manager):  # Will fail if no CUDA

# ❌ Not using fixture parameter
def test_ignores_fixture(gpu_manager):
    mgr = SimManager(exec_mode=ExecMode.CUDA, ...)  # Ignores fixture, creates new manager
```

## Agent Movement Controls

### CRITICAL: Always Use Helper Functions

**❌ DO NOT manually set action tensors:**
```python
# WRONG - Agent will move in circles due to unset rotation
actions = mgr.action_tensor().to_torch()
actions[:] = 0
actions[:, 0] = 3  # FAST movement
actions[:, 1] = 0  # FORWARD - but rotation not set!
```

**✅ Use AgentController helper class:**
```python
from test_helpers import AgentController

controller = AgentController(mgr)
controller.reset_actions()  # Sets rotation to NONE (no spin)
controller.move_forward(speed=consts.action.move_amount.FAST)
```

### Action Components
The action tensor has 3 components: `[move_amount, move_angle, rotate]`
- **move_amount**: `STOP=0, SLOW=1, MEDIUM=2, FAST=3`
- **move_angle**: `FORWARD=0, FORWARD_RIGHT=1, RIGHT=2, BACKWARD_RIGHT=3, BACKWARD=4, BACKWARD_LEFT=5, LEFT=6, FORWARD_LEFT=7`
- **rotate**: `FAST_LEFT=0, SLOW_LEFT=1, NONE=2, SLOW_RIGHT=3, FAST_RIGHT=4`

### Common Movement Patterns
```python
controller = AgentController(mgr)

# Essential: Reset actions before setting new ones
controller.reset_actions()  # Sets rotate=NONE to prevent spinning

# Basic movement
controller.move_forward(speed=consts.action.move_amount.MEDIUM)
controller.move_backward(speed=consts.action.move_amount.SLOW)
controller.strafe_left(speed=consts.action.move_amount.FAST)
controller.strafe_right(speed=consts.action.move_amount.MEDIUM)
controller.stop()

# Rotation without movement
controller.rotate_only(rotation=consts.action.rotate.SLOW_LEFT)

# Custom actions for specific worlds
controller.set_custom_action(
    world_idx=0,
    move_amount=consts.action.move_amount.FAST,
    move_angle=consts.action.move_angle.FORWARD_RIGHT,
    rotate=consts.action.rotate.NONE
)
```

### Why This Matters
- **Unset rotation causes circular movement** - agents spin while moving
- **Manual tensor manipulation is error-prone** - easy to forget rotation component
- **Helper functions ensure consistent behavior** - all components properly set
- **Makes tests more readable** - clear intent vs cryptic tensor indices

## Best Practices

**Do's:**
- **Always use `AgentController` for movement** - prevents circular movement bugs
- Use `gpu_manager` fixture for all GPU tests
- Add CUDA availability checks to GPU tests  
- Test CPU before GPU functionality
- Fix warnings immediately - tests should use `assert` not `return` statements
- Use custom levels with `cpu_manager` for specific test scenarios

**Don'ts:**
- **Manually set action tensors** - use AgentController instead
- Create GPU managers directly in tests
- Mix session/function-scoped GPU fixtures
- Forget `@pytest.mark.skipif` for GPU tests
- Return values from test functions (causes `PytestReturnNotNoneWarning`)
- Use custom levels with GPU tests

## Test Markers
- `@pytest.mark.ascii_level("level_ascii")` - Provide custom ASCII levels to tests
- `@pytest.mark.json_level("level_json")` - Provide custom JSON levels to tests
- `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` - Skip tests when CUDA unavailable
- `@pytest.mark.slow` - Mark tests that take significant time (GPU compilation, stress tests)

## Recording and Verification Capabilities
Tests can optionally record actions and verify determinism:
- Actions saved to `.bin` files for viewer replay
- Built-in checksum verification for determinism validation
- Automatic replay verification in specialized fixtures
- Use `--record-actions` flag for debugging
- No manual trajectory file management required

## Debugging

**GPU test failures** (`Fatal Python error`, deadlocks): Check if test creates its own GPU manager instead of using the `gpu_manager` fixture.

## Known Limitations

### GPU Replay Factory Method

**Issue**: The new `SimManager.from_replay()` factory method cannot be used in GPU tests because it creates a fresh GPU manager, violating the "one GPU manager per process" constraint.

**Current Status**: 
- CPU tests: Use `SimManager.from_replay()` (no warnings)
- GPU tests: Continue using `mgr.load_replay()` (with warnings) 

**Workaround**: GPU replay tests use the existing session-scoped `gpu_manager` fixture with the legacy `load_replay()` method.

**Future**: Consider adding GPU-compatible replay factory that reuses existing GPU context.
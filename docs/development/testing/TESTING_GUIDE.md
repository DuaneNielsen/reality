# Testing Guide

Testing patterns and best practices for the Madrona Escape Room project.

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

## Test Execution Order

```bash
# 1. CPU tests first
uv run --group dev pytest tests/python/ -v --no-gpu

# 2. GPU tests after CPU tests pass
uv run --group dev pytest tests/python/ -v -k "gpu"
```

## Fixtures

- `gpu_manager`: Session-scoped GPU SimManager (shared across all GPU tests)
- `gpu_env`: Session-scoped GPU TorchRL environment  
- `cpu_manager`: Function-scoped CPU manager (new per test, supports custom levels)

## Custom Level Testing

The `cpu_manager` fixture supports custom ASCII levels using the `@pytest.mark.custom_level()` decorator:

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

@pytest.mark.custom_level(TEST_LEVEL)
def test_with_custom_level(cpu_manager):
    """Test uses the custom level instead of default"""
    mgr = cpu_manager  # Manager initialized with TEST_LEVEL
    # Test your custom level behavior
```

### Key Features
- **Spawn Points**: Mark agent spawn locations with 'S' 
- **Walls**: Use '#' for walls and obstacles
- **Empty Space**: Use '.' for walkable areas
- **Auto-compilation**: Levels are automatically compiled with scale=2.0

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

@pytest.mark.custom_level(CORRIDOR_LEVEL)
def test_corridor_navigation(cpu_manager):
    """Test agent navigation through narrow passage"""
    mgr = cpu_manager
    # Agent must navigate through the gap in the wall
```

### Notes
- Only works with `cpu_manager` fixture (function-scoped)
- Module-scoped fixtures don't support per-test custom levels
- GPU tests should not use custom levels (violates single GPU manager constraint)

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

## Test Flags

### Basic Flags
```bash
--no-gpu              # Skip all GPU tests
```

### Debugging and Visualization Flags
```bash
--record-actions      # Record actions to binary files for viewer replay
--trace-trajectories  # Log agent trajectories to text files
--visualize           # Auto-launch viewer after tests (requires --record-actions)
```

### Usage Examples
```bash
# Record actions for debugging
uv run --group dev pytest tests/python/test_reward_system.py --record-actions

# Record both actions and trajectories
uv run --group dev pytest tests/python/test_reward_system.py --record-actions --trace-trajectories

# Record, trace, and auto-launch viewer
uv run --group dev pytest tests/python/test_reward_system.py --record-actions --trace-trajectories --visualize

# Debug specific test
uv run --group dev pytest tests/python/test_reward_system.py::test_forward_movement_reward -v --record-actions
```

### Output Files
When using debugging flags, files are created in `test_recordings/` with automatic naming:
- **Actions**: `test_recordings/test_module.py__test_function_actions.bin`  
- **Trajectories**: `test_recordings/test_module.py__test_function_actions_trajectory.txt`

Example:
```bash
uv run --group dev pytest tests/python/test_reward_system.py::test_forward_movement_reward --record-actions --trace-trajectories
# Creates:
# test_recordings/test_reward_system.py__test_forward_movement_reward_actions.bin
# test_recordings/test_reward_system.py__test_forward_movement_reward_actions_trajectory.txt
```

## Best Practices

**Do's:**
- Use `gpu_manager` fixture for all GPU tests
- Add CUDA availability checks to GPU tests  
- Test CPU before GPU functionality
- Fix warnings immediately - tests should use `assert` not `return` statements

**Don'ts:**
- Create GPU managers directly in tests
- Mix session/function-scoped GPU fixtures
- Forget `@pytest.mark.skipif` for GPU tests
- Return values from test functions (causes `PytestReturnNotNoneWarning`)

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
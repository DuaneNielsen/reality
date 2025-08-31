# Testing Guide

Testing patterns and best practices for the Madrona Escape Room project.

## Test Structure Overview

The project uses multiple test frameworks:
- **Python Tests**: pytest for high-level integration and behavior testing
- **C++ CPU Tests**: GoogleTest for fast unit testing
- **C++ GPU Tests**: Separated into fast and stress test executables

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

### Python Tests
```bash
# 1. CPU tests first
uv run --group dev pytest tests/python/ -v --no-gpu

# 2. GPU tests after CPU tests pass
uv run --group dev pytest tests/python/ -v -k "gpu"
```

### C++ Tests
```bash
# Fast CPU tests (~30 seconds)
./build/mad_escape_tests

# Fast GPU tests (~1-2 minutes, shared managers)
./build/mad_escape_gpu_tests  

# GPU stress tests (~8+ minutes, individual managers)
./build/mad_escape_gpu_stress_tests

# Or use CMake targets:
make run_cpp_tests           # CPU tests
make run_gpu_tests          # Fast GPU tests  
make run_gpu_stress_tests   # Comprehensive GPU tests
```

### C++ GPU Test Strategy

**Two-tier approach** for optimal development workflow:

1. **mad_escape_gpu_tests** - Daily development testing
   - 8 tests using shared managers with reset-based testing
   - Minimal NVRTC compilation overhead
   - ~1-2 minutes execution time
   - Tests: Basic functionality, tensor validation, reset behavior

2. **mad_escape_gpu_stress_tests** - Comprehensive validation  
   - 7 tests requiring individual manager compilation
   - Full NVRTC compilation per test (~40+ seconds each)
   - ~8+ minutes execution time
   - Tests: Large world counts, memory stress, configuration variations

## Fixtures

### Python Test Fixtures
- `gpu_manager`: Session-scoped GPU SimManager (shared across all GPU tests)
- `gpu_env`: Session-scoped GPU TorchRL environment  
- `cpu_manager`: Function-scoped CPU manager (new per test, supports custom levels)

### C++ Test Fixtures
- `ReusableGPUManagerTest`: Shared GPU manager with reset-based isolation
- `CustomGPUManagerTest`: Creates new managers for configuration-specific tests  
- `MadronaTestBase`: Base fixture with C API bindings (legacy)

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

### Python GPU Tests - Correct Pattern
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_functionality(gpu_manager):
    """Test GPU features using shared manager"""
    mgr = gpu_manager
    
    # Use the shared manager for testing
    obs_tensor = mgr.self_observation_tensor()
    assert obs_tensor.isOnGPU()
```

### C++ GPU Tests - Fast Tests Pattern
```cpp
// Fast tests using shared manager (mad_escape_gpu_tests)
TEST_F(ReusableGPUManagerTest, BasicFunctionality) {
    ASSERT_NE(mgr, nullptr);
    
    // Test using shared manager - fast execution
    mgr->step();
    
    auto action_tensor = GetActionTensor();
    EXPECT_GE(action_tensor.gpuID(), 0);
}

// Reset is automatic between tests via ResetAllWorlds()
TEST_F(ReusableGPUManagerTest, TensorValidation) {
    // Fresh state due to reset, no manager recreation needed
    auto tensor = GetActionTensor();
    std::vector<int64_t> expected = {config.numWorlds, 3};
    EXPECT_TRUE(ValidateTensorShape(tensor, expected));
}
```

### C++ GPU Tests - Stress Tests Pattern
```cpp
// Stress tests with custom managers (mad_escape_gpu_stress_tests)
TEST_F(CustomGPUManagerTest, LargeWorldCount) {
    config.numWorlds = 1024;  // Custom configuration
    
    ASSERT_TRUE(CreateManager());  // Creates new manager (~40s compilation)
    EXPECT_NE(custom_manager, nullptr);
    
    // Test with custom configuration
    auto action_tensor = custom_manager->actionTensor();
    // ... test large world functionality
}
```

### Common Mistakes

#### Python Tests
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

#### C++ Tests
```cpp
// ❌ Wrong test file - expensive test in fast suite
TEST_F(CustomGPUManagerTest, ExpensiveTest) {
    // This belongs in mad_escape_gpu_stress_tests, not mad_escape_gpu_tests
    config.numWorlds = 1024;
    CreateManager();  // Causes long compilation in fast test suite
}

// ❌ Not using shared manager efficiently
TEST_F(ReusableGPUManagerTest, WastefulTest) {
    CreateSharedManager();  // Unnecessary - manager already created in SetUp()
}

// ❌ Forgetting to use helper methods
TEST_F(ReusableGPUManagerTest, ManualTensorAccess) {
    auto tensor = mgr->actionTensor();  // Use GetActionTensor() instead
}
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

## Test Scripts

The `tests/` directory contains several utility scripts for running tests and benchmarks:

### Quick Testing Script
```bash
tests/quicktest.sh
```
**Purpose**: Run both C++ and Python tests in brief mode, only showing failures
- C++ tests: Uses `--gtest_brief=1` to minimize output
- Python tests: Uses `-q -m "not slow"` to skip slow tests and reduce output
- Ideal for rapid development iteration

### Performance Testing Script  
```bash
tests/run_perf_test.sh
```
**Purpose**: Comprehensive performance benchmarking with baseline comparison
- **CPU Benchmark**: 1024 worlds, 1000 steps
- **GPU Benchmark**: 8192 worlds, 1000 steps (if GPU available)  
- **Features**:
  - Checks performance against baselines
  - Saves detailed profiling data to `build/perf_results/TIMESTAMP/`
  - Generates HTML profile reports using `pyinstrument`
  - Returns exit codes: 0 (pass), 1 (fail), 2 (warning)
  - Automatically detects GPU availability
- **Requirements**: Requires `pyinstrument>=4.0.0` (included in dev dependencies)

**Usage**:
```bash
# Run full performance test suite
./tests/run_perf_test.sh

# Results saved to build/perf_results/YYYYMMDD_HHMMSS/
# HTML profiles saved to /tmp/sim_bench_profile_*.html

# Run smaller benchmark for quick testing
uv run python scripts/sim_bench.py --num-worlds 64 --num-steps 100 --check-baseline
```

### Stability Testing Script
```bash
tests/test_headless_loop.sh [num_runs] [level_file]
```
**Purpose**: Test simulation stability by running headless mode multiple times
- **Default**: 10 runs with default level
- **Parameters**:
  - `num_runs`: Number of test iterations (default: 10)  
  - `level_file`: Optional custom level file
- **Tracks**: Success rate, segfaults, assertion failures
- **Configuration**: 4 worlds, 5000 steps, seed 42, random actions

**Usage**:
```bash
# Run 10 times with default level
./tests/test_headless_loop.sh

# Run 25 times with custom level
./tests/test_headless_loop.sh 25 custom_level.bin

# Check specific level stability
./tests/test_headless_loop.sh 50 test_recordings/debug_level.bin
```

**Example Output**:
```
Run 1/10: SUCCESS (FPS: 2450.5)
Run 2/10: SUCCESS (FPS: 2445.2)
...
Summary:
  Successful runs: 9/10
  Failed runs:     1/10
    - Assertions:   1
```

## Best Practices

**Do's:**
- Use `gpu_manager` fixture for all GPU tests
- Add CUDA availability checks to GPU tests  
- Test CPU before GPU functionality
- Fix warnings immediately - tests should use `assert` not `return` statements
- Use `tests/quicktest.sh` for rapid development testing
- Run `tests/run_perf_test.sh` before performance-critical commits
- Use `tests/test_headless_loop.sh` to validate simulation stability

**Don'ts:**
- Create GPU managers directly in tests
- Mix session/function-scoped GPU fixtures
- Forget `@pytest.mark.skipif` for GPU tests
- Return values from test functions (causes `PytestReturnNotNoneWarning`)
- Skip stability testing for simulation core changes

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
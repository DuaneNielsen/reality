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
uv run --extra test pytest tests/python/ -v --no-gpu

# 2. GPU tests after CPU tests pass
uv run --extra test pytest tests/python/ -v -k "gpu"
```

## Fixtures

- `gpu_manager`: Session-scoped GPU SimManager (shared across all GPU tests)
- `gpu_env`: Session-scoped GPU TorchRL environment  
- `cpu_manager`: Function-scoped CPU manager (new per test)

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

```bash
--no-gpu          # Skip all GPU tests
--record-actions  # Record actions for viewer replay
--visualize       # Auto-launch viewer after tests
```

## Best Practices

**Do's:**
- Use `gpu_manager` fixture for all GPU tests
- Add CUDA availability checks to GPU tests  
- Test CPU before GPU functionality

**Don'ts:**
- Create GPU managers directly in tests
- Mix session/function-scoped GPU fixtures
- Forget `@pytest.mark.skipif` for GPU tests

## Debugging

**GPU test failures** (`Fatal Python error`, deadlocks): Check if test creates its own GPU manager instead of using the `gpu_manager` fixture.
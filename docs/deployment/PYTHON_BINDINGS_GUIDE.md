# Python Bindings Guide

This document provides a comprehensive guide to the Python bindings for the Madrona Escape Room environment, including installation, API reference, and advanced usage patterns.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Context Managers](#context-managers)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)

## Overview

The Python bindings provide a bridge between the C++ Madrona simulation engine and Python/PyTorch for machine learning applications. The current implementation uses:

- **C API**: Pure C wrapper around the C++ Manager class
- **ctypes**: Python's built-in foreign function library for calling C functions
- **Zero-copy tensors**: Direct memory access between C++ and Python/PyTorch
- **Context managers**: Convenient Python patterns for recording and debugging

### Key Features

1. **No Dependencies**: Built-in ctypes eliminates external binding library dependencies
2. **Zero-Copy Tensor Access**: Direct memory access between C++ and Python/PyTorch
3. **Self-Contained**: All dependencies bundled for easy distribution
4. **Context Managers**: Pythonic patterns for recording and trajectory logging
5. **PyTorch Integration**: Seamless tensor conversion with automatic device handling

## Installation

### Building from Source

```bash
# Build the C++ components
mkdir build
cmake -B build
make -C build -j8

# Install Python package
uv pip install -e .
```

### Verifying Installation

```bash
# Run basic tests
uv run --extra test pytest tests/python/test_bindings.py -v --no-gpu

# Run GPU tests (after CPU tests pass)
uv run --extra test pytest tests/python/test_bindings.py -v -k "gpu"
```

## Architecture

### Technology Stack

```
┌─────────────────────────────────────┐
│           Python Application        │
├─────────────────────────────────────┤
│         Python Bindings             │
│  (__init__.py + ctypes_bindings.py) │
├─────────────────────────────────────┤
│           C API Layer               │
│    (madrona_escape_room_c_api.*)    │
├─────────────────────────────────────┤
│         C++ Manager Class           │
│           (mgr.cpp/hpp)             │
├─────────────────────────────────────┤
│        Madrona ECS Engine           │
│          (simulation core)          │
└─────────────────────────────────────┘
```

### Components

1. **C++ Core**: The Madrona simulation engine with ECS architecture
2. **C API Wrapper**: Pure C interface wrapping C++ Manager methods
3. **ctypes Bindings**: Python module using ctypes to call C functions
4. **Python Wrapper**: High-level Pythonic API with context managers

### Error Handling

- C API returns error codes instead of exceptions (since C++ compiled without exceptions)
- Python wrapper converts error codes to appropriate Python exceptions
- All operations are checked for errors and raise meaningful exceptions

## API Reference

### SimManager Class

The main entry point for simulation control.

```python
import madrona_escape_room
from madrona_escape_room import SimManager

# Create manager
sim = SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,  # or CUDA
    gpu_id=0,                    # GPU device ID (ignored for CPU mode)
    num_worlds=1024,             # Number of parallel simulation worlds
    rand_seed=42,                # Random seed
    auto_reset=True,             # Automatically reset episodes when done
    enable_batch_renderer=False  # Enable rendering (CPU mode only)
)
```

### Core Methods

#### Simulation Control
```python
sim.step()                       # Advance simulation by one timestep
```

#### Tensor Access
All tensor methods return `madrona.Tensor` objects that can be converted to PyTorch tensors:

```python
# Get tensors
reset_tensor = sim.reset_tensor()           # [num_worlds] - reset flags
action_tensor = sim.action_tensor()         # [num_worlds, num_agents, 4] - actions
reward_tensor = sim.reward_tensor()         # [num_worlds, num_agents, 1] - rewards
done_tensor = sim.done_tensor()             # [num_worlds, num_agents, 1] - done flags
obs_tensor = sim.self_observation_tensor()  # [num_worlds, num_agents, 5] - observations
steps_tensor = sim.steps_remaining_tensor() # [num_worlds, num_agents, 1] - time remaining
progress_tensor = sim.progress_tensor()     # [num_worlds, num_agents, 1] - progress metric

# Convert to PyTorch (zero-copy when possible)
actions = action_tensor.to_torch()
rewards = reward_tensor.to_torch()
dones = done_tensor.to_torch()

# Convert to NumPy (CPU only, zero-copy)
if not action_tensor.isOnGPU():
    actions_np = action_tensor.to_numpy()
```

#### Rendering (CPU Mode Only)
```python
# Enable batch renderer during creation
sim = SimManager(..., enable_batch_renderer=True)

# Access rendered images
rgb_tensor = sim.rgb_tensor()      # [num_worlds, height, width, 3] - RGB images
depth_tensor = sim.depth_tensor()  # [num_worlds, height, width, 1] - depth maps
```

### Action Space

Actions are 3-dimensional integer arrays:

```python
# Action tensor shape: [num_worlds, 3]
actions = sim.action_tensor().to_torch()

# Action components (3 components per agent)
actions[..., 0] = move_amount    # 0-3: STOP, SLOW, MEDIUM, FAST
actions[..., 1] = move_angle     # 0-7: 8 directions (FORWARD, FORWARD_RIGHT, etc.)
actions[..., 2] = rotate         # 0-4: rotation (FAST_LEFT, SLOW_LEFT, NONE, SLOW_RIGHT, FAST_RIGHT)
# Note: grab action has been removed from current version
```

### Constants

```python
# Use provided constants for clarity
import madrona_escape_room as mer

# Movement amounts
actions[..., 0] = mer.action.move_amount.FAST

# Movement directions
actions[..., 1] = mer.action.move_angle.FORWARD

# Rotation
actions[..., 2] = mer.action.rotate.SLOW_RIGHT
```

### Resetting Worlds

```python
# Set reset flag to trigger episode reset
reset_tensor = sim.reset_tensor().to_torch()
reset_tensor[0] = 1  # Reset world 0
sim.step()           # Reset happens on next step
```

## Context Managers

The bindings provide convenient context managers for common debugging and development tasks.

### Recording Context Manager

Records all actions during the context for later replay:

```python
# Record actions to a file
with sim.recording("demo.bin", seed=42):
    for step in range(1000):
        # Set actions
        actions = sim.action_tensor().to_torch()
        actions.fill_(0)  # Example: set all actions to zero
        
        # Step simulation
        sim.step()

# File "demo.bin" now contains recorded actions and seed
# Replay with viewer (automatically uses correct parameters):
# ./build/viewer --replay demo.bin
```

#### Understanding Seeds in Recording

**Seeds are critical for deterministic replay**. When you record actions, the simulation's random seed is saved in the recording file metadata. This ensures that:

1. **Level Generation**: The same random levels are generated during replay
2. **Physics Determinism**: Random physics events (if any) occur identically
3. **Exact Reproduction**: The replay produces identical simulation states

```python
# The seed parameter determines the simulation's randomness
with sim.recording("demo.bin", seed=12345):
    # This recording will be deterministic with seed 12345
    for step in range(1000):
        sim.step()

# Different seeds create different recordings
with sim.recording("demo_different.bin", seed=67890):
    # This creates a completely different simulation sequence
    for step in range(1000):
        sim.step()

# If you omit the seed, a default is used (may not be deterministic)
with sim.recording("demo_default.bin"):  # Uses seed=0 by default
    for step in range(1000):
        sim.step()
```

**Important**: The seed in the recording file must match the environment's initial seed for proper replay. The viewer automatically reads the seed from the recording file and initializes the simulation with the same seed, ensuring perfect determinism.

#### Foolproof Replay with Factory Method

**NEW**: Use `SimManager.from_replay()` to eliminate all configuration errors:

```python
# Only specify what you care about - everything else comes from replay file
sim = SimManager.from_replay(
    "demo.bin",                         # Path to replay file  
    exec_mode=madrona.ExecMode.CUDA,    # Your choice: CPU or CUDA
    gpu_id=0,                           # Your choice: which GPU
    enable_batch_renderer=False         # Your choice: rendering on/off
)

# Manager is perfectly configured and ready to replay
print(f"Loaded replay with {sim._num_worlds} worlds")  # Automatically set
current, total = sim.get_replay_step_count()
print(f"Replay has {total} steps")

# Step through replay - guaranteed to work correctly
for step in range(total):
    finished = sim.replay_step()
    if finished:
        break
    
    # Access any tensors normally
    obs = sim.self_observation_tensor().to_torch()
    actions = sim.action_tensor().to_torch() 
    rewards = sim.reward_tensor().to_torch()
```

**Benefits**:
- **Impossible to misconfigure** - all settings come from replay file
- **No surprises** - you specify only execution preferences  
- **Works like viewer** - same automatic configuration as viewer/headless
- **Zero-error replay** - guaranteed bit-for-bit identical results

**Note**: In testing environments with session-scoped GPU managers (to avoid the "one GPU manager per process" limitation), you may still need to use the legacy `load_replay()` method for GPU tests. See the testing guide for details.

### Trajectory Logging Context Manager

Logs agent positions for debugging movement:

```python
# Log trajectory to file
with sim.trajectory_logging(world_idx=0, agent_idx=0, filename="trajectory.txt"):
    for step in range(100):
        sim.step()

# File "trajectory.txt" now contains position data
# Or log to stdout by omitting filename:
with sim.trajectory_logging(world_idx=0, agent_idx=0):
    for step in range(100):
        sim.step()
```

### Debug Session Context Manager

Combines recording and trajectory logging:

```python
# Complete debug session
with sim.debug_session("debug_run", enable_recording=True, enable_tracing=True, seed=42):
    for step in range(500):
        # Your simulation code here
        sim.step()

# Creates:
# - debug_run.bin (action recording)
# - debug_run_trajectory.txt (position log)
```

### Manual Context Management

You can also use the methods directly:

```python
# Start recording
sim.start_recording("demo.bin", seed=42)

# Check if recording
if sim.is_recording():
    print("Recording active")

# Stop recording
sim.stop_recording()

# Trajectory logging
sim.enable_trajectory_logging(world_idx=0, agent_idx=0, filename="positions.txt")
sim.disable_trajectory_logging()
```

## Advanced Usage

### Training Loop Example

```python
import torch
import madrona_escape_room as mer

# Create simulation
sim = mer.SimManager(
    exec_mode=mer.madrona.ExecMode.CUDA,
    gpu_id=0,
    num_worlds=8192,
    rand_seed=42,
    auto_reset=True
)

# Get tensor references (these persist across steps)
actions = sim.action_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
dones = sim.done_tensor().to_torch()
observations = sim.self_observation_tensor().to_torch()

# Training loop
for step in range(10000):
    # Get actions from policy
    with torch.no_grad():
        # Flatten for policy input: [num_worlds * num_agents, obs_dim]
        obs_flat = observations.view(-1, observations.shape[-1])
        action_logits = policy(obs_flat)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions_flat = action_dist.sample()
        
        # Reshape back to [num_worlds, num_agents, action_dim]
        actions[:] = actions_flat.view(actions.shape)
    
    # Step simulation
    sim.step()
    
    # Training logic here...
    # rewards and dones tensors are automatically updated
```

### Zero-Copy Verification

Ensure zero-copy access is working:

```python
tensor = sim.action_tensor()
numpy_arr = tensor.to_numpy()  # CPU only
torch_tensor = tensor.to_torch()

# Verify same memory (CPU tensors)
if not tensor.isOnGPU():
    assert numpy_arr.__array_interface__['data'][0] == torch_tensor.data_ptr()
    print("Zero-copy verified!")
```

### GPU Memory Management

```python
# GPU tensors automatically use correct device
if sim.action_tensor().isOnGPU():
    device_id = sim.action_tensor().gpuID()
    print(f"Tensors are on cuda:{device_id}")
    
    # PyTorch tensors automatically on correct device
    torch_tensor = sim.action_tensor().to_torch()
    assert torch_tensor.device == torch.device(f'cuda:{device_id}')
```

### Replay Functionality

**Recommended**: Use `SimManager.from_replay()` for foolproof replay that automatically configures everything correctly.

```python
# RECOMMENDED: Create manager automatically configured from replay file
sim = mer.SimManager.from_replay(
    "demo.bin",                           # Replay file path
    exec_mode=mer.madrona.ExecMode.CPU,   # Your execution preference
    gpu_id=0                              # Your GPU preference
)
# Manager is automatically configured with correct num_worlds, seed, auto_reset

# Alternative: Read metadata first, then create manually
metadata = mer.SimManager.read_replay_metadata("demo.bin")
print(f"Replay has {metadata['num_worlds']} worlds, seed {metadata['seed']}")

sim = mer.SimManager(
    exec_mode=mer.madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=metadata['num_worlds'],    # From replay file
    rand_seed=metadata['seed'],           # From replay file
    auto_reset=True                       # Always true for replay
)
sim.load_replay("demo.bin")  # Will warn about using factory method instead

# Check if replay is loaded
if sim.has_replay():
    # Get step information
    current, total = sim.get_replay_step_count()
    print(f"Replay: step {current}/{total}")
    
    # Step through replay
    while True:
        finished = sim.replay_step()
        if finished:
            break
        
        # Access simulation state during replay
        observations = sim.self_observation_tensor().to_torch()
        # ... process observations ...
```

#### Replay File Format

Recording files contain:
- **Header**: Metadata including the original seed used during recording
- **Action Data**: Sequence of actions for each simulation step
- **World Configuration**: Number of worlds and other settings

When `load_replay()` is called, the simulation automatically:
1. Reads the seed from the file header
2. Reinitializes the simulation with that seed
3. Resets all worlds to match the initial state during recording

This ensures that replays are **bit-for-bit identical** to the original recording.

## Troubleshooting

### Library Loading Issues

**Problem**: `ImportError` when importing the module

**Solutions**:
1. Ensure the C++ library is built: `make -C build -j8`
2. Check library exists: `ls build/libmadrona_escape_room_c_api.so`
3. Verify LD_LIBRARY_PATH includes the build directory

### CUDA Issues

**Problem**: GPU tensors not working or memory errors

**Solutions**:
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Use smaller batch sizes if running out of memory
4. Ensure GPU ID is valid: `torch.cuda.device_count()`

### Zero-Copy Not Working

**Problem**: Performance is slow due to memory copying

**Solutions**:
1. Verify you're using `.to_torch()` not `.to_numpy().to_torch()`
2. For GPU tensors, ensure DLPack support is available
3. Check tensor memory addresses for CPU tensors (see verification example above)

### Shape Mismatches

**Problem**: Tensor shapes don't match expected dimensions

**Common shapes**:
- Actions: `[num_worlds, num_agents, 4]`
- Observations: `[num_worlds, num_agents, 5]`
- Rewards: `[num_worlds, num_agents, 1]`
- Done flags: `[num_worlds, num_agents, 1]`

**Solution**: Always check tensor shapes before use:
```python
print(f"Action tensor shape: {sim.action_tensor().shape}")
```

### Replay Issues

**Problem**: Replay produces different results than the original recording

**SOLUTION**: Use `SimManager.from_replay()` to eliminate configuration issues:

```python
# FOOLPROOF - automatically configures everything correctly
sim = SimManager.from_replay("demo.bin", exec_mode=ExecMode.CPU)
```

**Legacy troubleshooting** (if you must use the old `load_replay()` method):

1. **Configuration mismatch**:
   ```python
   # Read metadata first to get correct configuration
   metadata = SimManager.read_replay_metadata("demo.bin")
   sim = SimManager(
       exec_mode=ExecMode.CPU,
       num_worlds=metadata['num_worlds'],    # Must match
       rand_seed=metadata['seed'],           # Must match
       auto_reset=True,                      # Always true for replay
   )
   sim.load_replay("demo.bin")  # Will still warn to use factory method
   ```

2. **Viewer vs Python behavior**:
   ```bash
   # Viewer automatically handles configuration
   ./build/viewer --replay demo.bin  # Always works
   
   # Python requires correct configuration OR use factory method
   ```

3. **File corruption**:
   ```python
   # Check metadata first
   try:
       metadata = SimManager.read_replay_metadata("demo.bin")
       print(f"Valid replay: {metadata['num_worlds']} worlds")
   except Exception as e:
       print(f"Corrupt replay file: {e}")
   ```

4. **Determinism verification**:
   ```python
   # Test replay determinism
   sim1 = SimManager.from_replay("demo.bin", ExecMode.CPU)
   sim1.replay_step()
   obs1 = sim1.self_observation_tensor().to_torch().clone()
   
   sim2 = SimManager.from_replay("demo.bin", ExecMode.CPU)
   sim2.replay_step()
   obs2 = sim2.self_observation_tensor().to_torch()
   
   assert torch.equal(obs1, obs2), "Replay not deterministic!"
   ```

## Performance Notes

### Memory Access Patterns

1. **Tensor Reuse**: Get tensor references once and reuse them
   ```python
   # Good: Get reference once
   actions = sim.action_tensor().to_torch()
   for step in range(1000):
       actions[:] = new_actions  # Reuse tensor
       sim.step()
   
   # Bad: Get tensor every step
   for step in range(1000):
       actions = sim.action_tensor().to_torch()  # Creates new wrapper each time
       sim.step()
   ```

2. **In-place Operations**: Modify tensors in-place when possible
   ```python
   actions = sim.action_tensor().to_torch()
   actions.fill_(0)        # Good: in-place
   actions[:] = new_data   # Good: in-place assignment
   actions = torch.zeros_like(actions)  # Bad: creates new tensor
   ```

### GPU Optimization

1. **Batch Size**: Use larger batch sizes (num_worlds) for better GPU utilization
2. **Memory Coalescing**: Access tensor data in contiguous patterns
3. **Device Synchronization**: Minimize CPU-GPU synchronization points

### Profiling

Use PyTorch profiler to identify bottlenecks:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for step in range(100):
        # Your simulation code
        sim.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## API Compatibility

This ctypes-based implementation maintains full API compatibility with the original nanobind version. Code written for the nanobind bindings should work without modification.

### Migration Notes

If migrating from older versions:

1. **Import paths remain the same**: `from madrona_escape_room import SimManager`
2. **Method signatures unchanged**: All method parameters and return types are identical
3. **Tensor interface preserved**: `.to_torch()` and `.to_numpy()` methods work as before
4. **Constants available**: All action and configuration constants remain available

The only differences are internal implementation details and improved error handling.
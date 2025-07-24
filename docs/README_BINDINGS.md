# Madrona Escape Room Python Bindings

This document describes the Python bindings for the Madrona Escape Room simulator.

## Installation

```bash
# Build the project
mkdir build && cd build
/opt/cmake/bin/cmake ..
make -j$(nproc)
cd ..

# Install Python package with uv
pip install -e .
```

## Running Tests

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run all tests
uv run --extra test pytest test_bindings.py -v

# Run only CPU tests
uv run --extra test pytest test_bindings.py -v -k "not gpu"
```

## API Reference

### Creating a SimManager

```python
import madrona_escape_room
from madrona_escape_room import SimManager

# CPU mode
sim = SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=1024,
    rand_seed=42,
    auto_reset=True,
    enable_batch_renderer=False
)

# GPU mode
sim = SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CUDA,
    gpu_id=0,
    num_worlds=8192,
    rand_seed=42,
    auto_reset=True,
    enable_batch_renderer=False
)
```

### Available Methods

- `step()` - Advance simulation by one timestep
- `reset_tensor()` - Access reset flags for worlds (shape: [num_worlds])
- `action_tensor()` - Access agent actions (shape: [num_worlds, num_agents, 4])
- `reward_tensor()` - Access rewards (shape: [num_worlds, num_agents, 1])
- `done_tensor()` - Access episode done flags (shape: [num_worlds, num_agents, 1])
- `self_observation_tensor()` - Access agent's own state
- `partner_observations_tensor()` - Access other agents' states
- `room_entity_observations_tensor()` - Access room objects
- `lidar_tensor()` - Access lidar sensor data (shape: [num_worlds, num_agents, 30, 2])
- `steps_remaining_tensor()` - Access episode time remaining
- `rgb_tensor()` - Access RGB render output (if batch renderer enabled)
- `depth_tensor()` - Access depth render output (if batch renderer enabled)

### Converting to PyTorch Tensors

All tensor methods return Madrona tensors. Convert to PyTorch tensors using `.to_torch()`:

```python
# Get PyTorch tensors
actions = sim.action_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
dones = sim.done_tensor().to_torch()

# Tensors are in shape [num_worlds, num_agents, ...]
# For training, often flatten to [num_worlds * num_agents, ...]
actions = actions.view(-1, *actions.shape[2:])
rewards = rewards.view(-1, *rewards.shape[2:])
dones = dones.view(-1, *dones.shape[2:])
```

### Action Space

Actions are 4-dimensional integers:
- `actions[..., 0]` - Movement amount (0-3)
- `actions[..., 1]` - Movement angle (0-7) 
- `actions[..., 2]` - Rotation (0-4)
- `actions[..., 3]` - Grab (0-1)

### Resetting Worlds

Use the reset tensor to trigger episode resets:

```python
reset_tensor = sim.reset_tensor().to_torch()
reset_tensor[0] = 1  # Reset world 0
sim.step()  # Reset happens on next step
```

### Example Training Loop

```python
# Initialize
actions = sim.action_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
dones = sim.done_tensor().to_torch()

# Training loop
for step in range(num_steps):
    # Get actions from policy
    actions[:] = policy.get_actions(observations)
    
    # Step simulation
    sim.step()
    
    # Process rewards and dones
    # ... training logic ...
```

## Important Notes

1. **Tensor Shapes**: All tensors have shape `[num_worlds, num_agents, ...]`
2. **Device Placement**: GPU tensors are automatically on the correct CUDA device
3. **Zero-Copy**: Tensors provide direct access to simulation memory
4. **Thread Safety**: SimManager is not thread-safe - use one per thread
5. **State Persistence**: Manager maintains state between calls

## Verified Functionality

All core functionality has been tested:
- ✓ CPU and GPU execution modes
- ✓ Tensor access and shapes
- ✓ Simulation stepping
- ✓ Reset functionality
- ✓ Memory layout compatibility with PyTorch
- ✓ Multi-step execution
- ✓ State persistence
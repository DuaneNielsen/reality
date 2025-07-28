# Madrona Escape Room Python Bindings

This document describes the Python bindings for the Madrona Escape Room simulator.

## Installation

```bash
# Build the project
mkdir build
/opt/cmake/bin/cmake -B build
make -C build -j$(nproc)

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
- `enable_trajectory_logging(world_idx, agent_idx, filename=None)` - Enable position logging for debugging
- `disable_trajectory_logging()` - Disable position logging

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

## Advanced Binding Patterns

### Optional String Parameters

When creating Python bindings for C++ functions with optional string parameters, special care is needed:

```cpp
// C++ function with optional string parameter
void myFunction(int required, const char* optional = nullptr);

// INCORRECT: Won't work with keyword arguments
.def("my_function", &myFunction, 
     nb::arg("required"), nb::arg("optional") = nb::none())

// CORRECT: Use wrapper with std::optional<std::string>
.def("my_function", 
     [](MyClass& self, int required, std::optional<std::string> optional) {
         if (optional.has_value()) {
             self.myFunction(required, optional->c_str());
         } else {
             self.myFunction(required, nullptr);
         }
     },
     nb::arg("required"), nb::arg("optional") = nb::none())
```

**Key learnings:**
- nanobind cannot bind `std::optional<T*>` where T has a special type caster (like `const char*`)
- Use `std::optional<std::string>` in the binding layer and convert to `const char*`
- Always include `<nanobind/stl/string.h>` and `<nanobind/stl/optional.h>` for STL type support

### Coding standard

Ensure functions work with both positional and keyword arguments:

1. **All parameters must be annotated** when using `nb::arg()`
2. **Use consistent naming** between C++ parameter names and Python argument names
3. **Test both calling styles** in your test suite:
   ```python
   # Both should work
   mgr.my_function(42, "value")                    # Positional
   mgr.my_function(required=42, optional="value")  # Keyword
   mgr.my_function(42, optional="value")           # Mixed
   ```

### Common Pitfalls

1. **Missing STL headers**: Always include the appropriate nanobind STL headers for type conversion
2. **Type caster conflicts**: Some types (like `const char*`) have special handling that conflicts with `std::optional`
3. **Partial annotations**: If you use `nb::arg()` for one parameter, you must use it for all parameters
4. **C++ function signatures**: When using `std::optional` in C++, remember to include `<optional>` in the header file

### Example: Trajectory Logging Implementation

Here's how the trajectory logging feature was implemented with optional file parameter:

```cpp
// mgr.hpp
#include <optional>
void enableTrajectoryLogging(int32_t world_idx, int32_t agent_idx, 
                           std::optional<const char*> filename = std::nullopt);

// bindings.cpp
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

.def("enable_trajectory_logging", 
     [](Manager &mgr, int32_t world_idx, int32_t agent_idx, 
        std::optional<std::string> filename) {
         if (filename.has_value()) {
             mgr.enableTrajectoryLogging(world_idx, agent_idx, filename->c_str());
         } else {
             mgr.enableTrajectoryLogging(world_idx, agent_idx, std::nullopt);
         }
     },
     nb::arg("world_idx"), nb::arg("agent_idx"), nb::arg("filename") = nb::none(),
     "Enable trajectory logging for a specific agent")
```

## Verified Functionality

All core functionality has been tested:
- ✓ CPU and GPU execution modes
- ✓ Tensor access and shapes
- ✓ Simulation stepping
- ✓ Reset functionality
- ✓ Memory layout compatibility with PyTorch
- ✓ Multi-step execution
- ✓ State persistence
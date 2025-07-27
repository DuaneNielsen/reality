# How to Export an ECS Component

This guide shows how to make an ECS component accessible from Python.

## Steps

### 1. Add to ExportID enum (src/sim.hpp)
Add your component name before `NumExports`:
```cpp
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    StepsRemaining,
    Progress,  // Add your component here
    NumExports,
};
```

### 2. Register the export (src/sim.cpp)
In `registerTypes()`, add:
```cpp
registry.exportColumn<Agent, Progress>(
    (uint32_t)ExportID::Progress);
```

### 3. Add tensor accessor to Manager (src/mgr.hpp)
Add public method:
```cpp
madrona::py::Tensor progressTensor() const;
```

### 4. Implement tensor accessor (src/mgr.cpp)
Add implementation:
```cpp
Tensor Manager::progressTensor() const
{
    return impl_->exportTensor(ExportID::Progress,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   1,  // Progress has 1 float
                               });
}
```

### 5. Add Python binding (src/bindings.cpp)
In the Manager class bindings, add:
```cpp
.def("progress_tensor", &Manager::progressTensor)
```

### 6. Document Python Usage

Create a descriptive file like `docs/using_progress_component.md` containing a Python usage example:

```python
import madrona_escape_room
import numpy as np

# Create manager
mgr = madrona_escape_room.Manager(
    exec_mode=madrona_escape_room.ExecMode.CPU,
    gpu_id=0,
    num_worlds=10,
    rand_seed=42,
    auto_reset=True
)

# Run some steps
for i in range(100):
    mgr.step()

# Access the progress tensor
progress = mgr.progress_tensor()
# Shape: [num_worlds, num_agents, 1]

# Get progress for world 0, agent 0
agent_progress = progress[0, 0, 0]
print(f"Agent 0 progress (maxY): {agent_progress}")

# Get average progress across all agents
avg_progress = np.mean(progress)
print(f"Average progress: {avg_progress}")

# Find best performing agent
best_world, best_agent = np.unravel_index(
    np.argmax(progress[..., 0]), 
    progress[..., 0].shape
)
print(f"Best agent: World {best_world}, Agent {best_agent} with progress {progress[best_world, best_agent, 0]}")
```

### 7. Build the project

```bash
cd build
make -j$(nproc)
cd ..
# The Python package automatically picks up the changes
```

IMPORTANT: don't forget to cd .. after build completes successfully!

### 8. Add test to tests/python/test_bindings.py

```python
def test_progress_tensor(manager_config):
    """Test that progress tensor is accessible and has correct shape."""
    mgr = madrona_escape_room.Manager(**manager_config)
    
    # Get progress tensor
    progress = mgr.progress_tensor()
    
    # Check shape
    expected_shape = (manager_config['num_worlds'], madrona_escape_room.NUM_AGENTS, 1)
    assert progress.shape == expected_shape, f"Expected shape {expected_shape}, got {progress.shape}"
    
    # Check initial values are reasonable (should be near spawn position)
    assert np.all(progress >= 0), "Progress values should be non-negative"
    assert np.all(progress < 10), "Initial progress should be less than 10"
    
    # Run some steps and verify progress updates
    initial_progress = progress.copy()
    for _ in range(50):
        mgr.step()
    
    final_progress = mgr.progress_tensor()
    
    # At least some agents should have made progress
    assert np.any(final_progress > initial_progress), "Some agents should have made progress"
    
    # Progress should never decrease (it tracks maximum Y)
    assert np.all(final_progress >= initial_progress), "Progress should never decrease"
```

### 9. Run the tests and fix any issues

```bash
# Run the specific test
uv run --extra test pytest tests/python/test_bindings.py::test_progress_tensor -v

# Or run all binding tests
uv run --extra test pytest tests/python/test_bindings.py -v

# Common issues and fixes:
# - ImportError: rebuild with 'cd build && make -j$(nproc) && cd ..'
# - Shape mismatch: verify tensor dimensions in step 4
# - Missing attribute: ensure Python binding was added in step 5
```

## Notes
- Component must be part of an archetype to be exported
- Tensor shape depends on component structure
- Float components use `TensorElementType::Float32`
- Integer components use `TensorElementType::Int32`
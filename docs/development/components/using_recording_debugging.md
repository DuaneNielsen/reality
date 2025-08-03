# Recording and Debugging with Context Managers

This document describes the context manager API for recording actions and trajectory logging in Madrona Escape Room, providing clean, Pythonic debugging workflows.

## Overview

The context manager API provides automatic resource management for:
- **Action Recording**: Save agent actions to binary files for replay
- **Trajectory Logging**: Track agent positions and progress to files
- **Combined Debugging**: Both recording and tracking in one session

## Context Managers Available

### Individual Context Managers

#### Recording Context Manager
```python
import madrona_escape_room

mgr = madrona_escape_room.SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=4,
    rand_seed=42,
    auto_reset=True
)

# Record actions with automatic cleanup
with mgr.recording("my_actions.bin", seed=42):
    for i in range(200):
        mgr.step()
# Recording automatically stops when context exits
```

#### Trajectory Logging Context Manager
```python
# Log trajectory to file
with mgr.trajectory_logging(world_idx=0, agent_idx=0, filename="trajectory.txt"):
    for i in range(200):
        mgr.step()
# Logging automatically stops and file is closed

# Log trajectory to stdout (no filename)
with mgr.trajectory_logging(world_idx=0, agent_idx=0):
    for i in range(200):
        mgr.step()
```

### Master Debug Session

The `DebugSession` context manager combines both recording and trajectory logging:

```python
# Both recording and trajectory logging
with mgr.debug_session("debug_session"):
    for i in range(200):
        mgr.step()
# Creates: debug_session.bin and debug_session_trajectory.txt

# Only recording (disable trajectory logging)
with mgr.debug_session("recording_only", enable_tracing=False):
    for i in range(200):
        mgr.step()
# Creates: recording_only.bin

# Only trajectory logging (disable recording)
with mgr.debug_session("tracing_only", enable_recording=False):
    for i in range(200):
        mgr.step()
# Creates: tracing_only_trajectory.txt

# Custom seed for recording
with mgr.debug_session("custom_seed", seed=123):
    for i in range(200):
        mgr.step()
```

## pytest Integration

### Command Line Flags

The testing framework supports automatic recording and tracing:

```bash
# Record actions for all tests
pytest --record-actions

# Enable trajectory tracing for all tests  
pytest --trace-trajectories

# Both recording and tracing
pytest --record-actions --trace-trajectories

# With automatic visualization
pytest --record-actions --visualize

# Run specific test with debugging
pytest tests/python/test_reward_system.py::test_forward_movement_reward --record-actions --trace-trajectories
```

### Automatic File Naming

Files are automatically named based on the test:
- **Recording**: `test_recordings/test_reward_system.py__test_forward_movement_reward_actions.bin`
- **Trajectory**: `test_recordings/test_reward_system.py__test_forward_movement_reward_actions_trajectory.txt`

### Example Test Debugging Workflow

```bash
# 1. Run test with recording and tracing
pytest tests/python/test_reward_system.py::test_forward_movement_reward -v --record-actions --trace-trajectories

# 2. Check the generated files
ls test_recordings/
# test_reward_system.py__test_forward_movement_reward_actions.bin
# test_reward_system.py__test_forward_movement_reward_actions_trajectory.txt

# 3. Replay in viewer
./build/viewer --num-worlds 4 --replay test_recordings/test_reward_system.py__test_forward_movement_reward_actions.bin

# 4. Analyze trajectory data
cat test_recordings/test_reward_system.py__test_forward_movement_reward_actions_trajectory.txt
```

## Advanced Usage

### Nested Context Managers

You can nest context managers for fine-grained control:

```python
with mgr.recording("outer_recording.bin"):
    # Initial setup
    for i in range(50):
        mgr.step()
    
    # Enable trajectory logging for specific section
    with mgr.trajectory_logging(0, 0, "middle_trajectory.txt"):
        for i in range(100):
            mgr.step()
    
    # Continue recording without trajectory
    for i in range(50):
        mgr.step()
```

### Exception Safety

Context managers ensure proper cleanup even when exceptions occur:

```python
try:
    with mgr.debug_session("crash_test"):
        for i in range(100):
            mgr.step()
            if i == 50:
                raise ValueError("Simulated crash")
except ValueError:
    pass
# Recording and trajectory logging are properly stopped and files closed
```

### Custom Recording Parameters

```python
# Different seeds for different sessions
with mgr.recording("session1.bin", seed=42):
    # Run scenario 1
    pass

with mgr.recording("session2.bin", seed=123):
    # Run scenario 2 with different seed
    pass
```

## Integration with Viewer

### Automatic Visualization

Tests can automatically launch the viewer:

```bash
pytest test_example.py --record-actions --visualize
```

This will:
1. Run the test with recording enabled
2. Display the recording file path
3. Automatically launch the viewer to replay the actions

### Manual Viewer Usage

```bash
# View recorded actions
./build/viewer --num-worlds 4 --replay test_recordings/my_test_actions.bin

# View with trajectory tracking enabled
./build/viewer --num-worlds 4 --replay test_recordings/my_test_actions.bin --track
```

## Output Formats

### Recording Files (.bin)

Binary files containing:
- Action sequences
- Metadata (seed, number of worlds, step count)
- Compatible with viewer replay

### Trajectory Files (.txt)

Text files with format:
```
Step    0: World 0 Agent 0: pos=(0.00,5.00,1.62) rot=0.0° progress=0.00
Step    1: World 0 Agent 0: pos=(0.00,5.12,1.62) rot=0.0° progress=0.01
Step    2: World 0 Agent 0: pos=(0.00,5.24,1.62) rot=0.0° progress=0.02
```

Fields:
- **Step**: Simulation step number
- **World/Agent**: Indices of tracked world and agent  
- **pos**: Global position (x, y, z)
- **rot**: Agent rotation in degrees
- **progress**: Progress value (normalized max Y reached)

## Performance Considerations

- Context managers add minimal overhead
- Recording writes binary data efficiently
- Trajectory logging writes text on each step (slower)
- Use recording for performance-critical scenarios
- Use trajectory logging for detailed debugging

## Backward Compatibility

All existing API methods remain available:

```python
# Old API still works
mgr.start_recording("file.bin", seed=42)
mgr.enable_trajectory_logging(0, 0, "traj.txt")
# ... run simulation ...
mgr.stop_recording()
mgr.disable_trajectory_logging()

# New context manager API (recommended)
with mgr.debug_session("file"):
    # ... run simulation ...
```

## Error Handling

Context managers handle common error scenarios:

```python
# Graceful handling of missing files/permissions
try:
    with mgr.recording("/invalid/path/file.bin"):
        mgr.step()
except RuntimeError as e:
    print(f"Recording failed: {e}")

# Context manager ensures cleanup even on errors
with mgr.debug_session("test"):
    # Even if this fails, recording/logging will be stopped
    mgr.step()
```
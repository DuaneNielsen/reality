# Madrona Python REPL MCP Server

This guide explains how to use the Madrona Python REPL through the Model Context Protocol (MCP) for interactive development. The server provides a persistent Python session with pre-loaded Madrona environment, making debugging and exploration much more efficient than writing standalone scripts.

## Benefits

- **Interactive debugging** - Test hypotheses quickly without writing full scripts
- **Persistent state** - Variables and managers stay loaded between executions  
- **Exploratory development** - Incrementally build up complex test scenarios
- **Real-time inspection** - Examine tensor states, agent positions, and rewards interactively
- **Performance profiling** - Quick benchmarks without separate benchmark scripts

## Prerequisites

- Madrona Escape Room project built (`make -C build -j8 -s`)
- Python package installed (`uv pip install -e .`)
- `uv` package manager installed

## Claude Code (Automatic Setup)

**If you're using Claude Code, setup is automatic!** The project includes a session hook that automatically installs the MCP server when you start a session.

### Verification
You can verify the server is installed by checking:
```bash
claude mcp list
# Should show: madrona_repl: uv run python scripts/fastmcp_madrona_server.py - ✓ Healthy
```

## Manual Setup (Other Tools)

For users of other MCP-compatible tools, install manually:

### Installation Commands

```bash
# Navigate to project directory
cd /path/to/madrona_escape_room

# Add the FastMCP Madrona server
your-mcp-client add madrona_repl uv run python -- scripts/fastmcp_madrona_server.py
```

### Verification
Test the server works:
```bash
# Test server startup
uv run python scripts/fastmcp_madrona_server.py
# Should start without errors (Ctrl+C to exit)
```

## Usage

The MCP server provides two tools:

### `execute_python(code, reset=False)`
Execute Python code with pre-loaded Madrona environment (SimManager, madrona, numpy).

### `list_variables()`
List all variables in the current Python session with types and shapes.

## Getting Started

### Basic Manager Creation

```python
# Create a SimManager for testing
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=2)
obs = mgr.self_observation_tensor().to_numpy()
actions = mgr.action_tensor().to_numpy()

print(f"Manager created with {len(obs)} worlds")
print(f"Observation shape: {obs.shape}")
print(f"Action shape: {actions.shape}")
```

### Canonical Workflow

```python
# Standard simulation workflow
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=2)
obs = mgr.self_observation_tensor().to_numpy()
actions = mgr.action_tensor().to_numpy()
actions[0, :] = [1, 0, 0]  # move_amount, move_angle, rotate
mgr.step()
pos = obs[0, 0, :3]  # world 0, agent 0 position
print(f"Agent position: {pos}")
```

### Interactive Exploration

```python
# Check current variables
list_variables()

# Inspect agent states
for i in range(len(obs)):
    agent_pos = obs[i, 0, :3]  # world i, agent 0, xyz position
    print(f"World {i} agent position: {agent_pos}")

# Test different actions
actions[:] = 0
actions[:, 0] = [1.0, 0.5, 0.0]  # Move forward with slight turn
mgr.step()

# Check rewards
rewards = mgr.reward_tensor().to_numpy()
print(f"Rewards: {rewards.flatten()}")
```

## Common Use Cases

### 1. Reward System Development

```python
# Test reward triggers
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=1)
obs = mgr.self_observation_tensor().to_numpy()
actions = mgr.action_tensor().to_numpy()
rewards = mgr.reward_tensor().to_numpy()

# Move forward until we get a reward
step_count = 0
while rewards[0, 0] == 0 and step_count < 200:
    actions[:] = 0
    actions[0, 0] = [1.0, 0, 0]  # Move forward
    mgr.step()
    step_count += 1
    
    if step_count % 20 == 0:
        pos = obs[0, 0, :3]
        print(f"Step {step_count}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
print(f"First reward after {step_count} steps: {rewards[0, 0]}")
```

### 2. Action Space Exploration

```python
# Test each action dimension
action_names = ['moveAmount', 'moveAngle', 'rotate']
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=1)
obs = mgr.self_observation_tensor().to_numpy()
actions = mgr.action_tensor().to_numpy()

for i, name in enumerate(action_names):
    mgr.reset()
    pos_before = obs[0, 0, :3].copy()
    rot_before = obs[0, 0, 4]  # rotation angle
    
    # Test this action dimension
    actions[:] = 0
    actions[0, 0, i] = 1.0
    mgr.step()
    
    pos_after = obs[0, 0, :3].copy()
    rot_after = obs[0, 0, 4]
    
    pos_delta = pos_after - pos_before
    rot_delta = rot_after - rot_before
    
    print(f"{name:12}: pos Δ=({pos_delta[0]:+.3f}, {pos_delta[1]:+.3f}, {pos_delta[2]:+.3f}), "
          f"rot Δ={rot_delta:+.3f}")
```

### 3. Performance Benchmarking

```python
import time

# CPU benchmark
print("CPU Benchmark:")
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=1024)
actions = mgr.action_tensor().to_numpy()

start_time = time.time()
for _ in range(100):
    actions[:] = np.random.uniform(-1, 1, actions.shape)
    mgr.step()
end_time = time.time()

elapsed = end_time - start_time
sims_per_sec = (1024 * 100) / elapsed
print(f"CPU: {sims_per_sec:.0f} simulations/second")

# GPU benchmark (if available)
try:
    print("\nGPU Benchmark:")
    mgr_gpu = SimManager(exec_mode=madrona.ExecMode.CUDA, gpu_id=0, num_worlds=8192)
    actions_gpu = mgr_gpu.action_tensor().to_numpy()
    
    start_time = time.time()
    for _ in range(100):
        actions_gpu[:] = np.random.uniform(-1, 1, actions_gpu.shape)
        mgr_gpu.step()
    end_time = time.time()
    
    elapsed = end_time - start_time
    sims_per_sec = (8192 * 100) / elapsed
    print(f"GPU: {sims_per_sec:.0f} simulations/second")
except Exception as e:
    print(f"GPU not available: {e}")
```

### 4. Episode Analysis

```python
# Run a complete episode and analyze
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=4)
obs = mgr.self_observation_tensor().to_numpy()
actions = mgr.action_tensor().to_numpy()
rewards = mgr.reward_tensor().to_numpy()
done = mgr.done_tensor().to_numpy()

mgr.reset()
episode_rewards = []
episode_positions = []

for step in range(200):
    # Random actions
    actions[:] = np.random.uniform(-1, 1, actions.shape)
    mgr.step()
    
    # Collect data
    episode_rewards.append(rewards.copy())
    episode_positions.append(obs[:, 0, :3].copy())  # All agents' positions
    
    # Check if any episodes finished
    if done.any():
        finished_worlds = np.where(done.flatten())[0]
        print(f"Step {step}: Episodes finished in worlds: {finished_worlds}")

# Analyze results
total_rewards = np.array(episode_rewards).sum(axis=0)
final_positions = episode_positions[-1]

print(f"\nEpisode Summary:")
for i in range(len(total_rewards)):
    print(f"  World {i}: Total reward = {total_rewards[i, 0]:.3f}, "
          f"Final pos = ({final_positions[i, 0]:.2f}, {final_positions[i, 1]:.2f}, {final_positions[i, 2]:.2f})")
```

## Session Management

### Reset Session
```python
# Reset the Python session (clears all variables)
execute_python("", reset=True)
```

### Check Variables
```python
# See what's currently defined
list_variables()
```

### Save Important State
```python
# Save manager state before experimenting
saved_obs = obs.copy()
saved_rewards = rewards.copy()
saved_actions = actions.copy()

# Later restore if needed
obs[:] = saved_obs
rewards[:] = saved_rewards
actions[:] = saved_actions
```

## Integration with Other Tools

### With Viewer
```python
# Record actions for viewer replay
import numpy as np

# Set up scenario
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=1)
actions = mgr.action_tensor().to_numpy()

# Create interesting action sequence
action_sequence = []
for step in range(100):
    if step < 20:
        action = [1.0, 0, 0]  # Move forward
    elif step < 40:
        action = [1.0, 0.5, 0]  # Turn right
    else:
        action = [0.5, -0.5, 0]  # Turn left slowly
    
    actions[0, 0] = action
    mgr.step()
    action_sequence.append(action.copy())

print("Action sequence created - can be used with viewer")
```

### With Tests
```python
# Prototype test scenarios interactively
def test_forward_movement_reward():
    mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=1)
    obs = mgr.self_observation_tensor().to_numpy()
    actions = mgr.action_tensor().to_numpy()
    rewards = mgr.reward_tensor().to_numpy()
    
    mgr.reset()
    initial_y = obs[0, 0, 1]
    
    # Move forward
    actions[:] = 0
    actions[0, 0, 0] = 1.0  # moveAmount = 1.0
    mgr.step()
    
    final_y = obs[0, 0, 1]
    reward = rewards[0, 0]
    
    print(f"Y position: {initial_y:.3f} -> {final_y:.3f}")
    print(f"Reward: {reward:.3f}")
    
    return final_y > initial_y and reward > 0

# Test the function
result = test_forward_movement_reward()
print(f"Test passed: {result}")
```

## Troubleshooting

### Server Issues
```bash
# Check server status
claude mcp list

# Test server manually
uv run python scripts/fastmcp_madrona_server.py

# Reinstall server
claude mcp remove madrona_repl
claude mcp add madrona_repl uv run python -- scripts/fastmcp_madrona_server.py
```

### Import/Build Issues
```bash
# Rebuild project
make -C build clean
make -C build -j8 -s

# Reinstall Python package
uv pip install -e . --force-reinstall

# Test imports
python -c "import madrona_escape_room; print('OK')"
```

### Performance Issues
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Best Practices

1. **Start small** - Use 1-4 worlds for initial exploration
2. **Save state** - Store important variables before risky experiments  
3. **Use PyTorch operations** - Leverage PyTorch tensors for efficiency and GPU compatibility
4. **Check shapes** - Always verify tensor dimensions match expectations
5. **Reset when needed** - Clear session state when switching contexts

## See Also

- [Testing Guide](../testing/TESTING_GUIDE.md) - Formal testing patterns
- [GDB Guide](../debugging/GDB_GUIDE.md) - Step-through debugging  
- [Viewer Guide](../../tools/VIEWER_GUIDE.md) - 3D visualization
- [Headless Mode](../../deployment/headless/HEADLESS_QUICK_REFERENCE.md) - Command-line simulation
- [ENVIRONMENT.md](../../../ENVIRONMENT.md) - Complete action/observation space details
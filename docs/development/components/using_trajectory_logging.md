# Using Trajectory Logging

This document describes how to use the trajectory logging feature in Madrona Escape Room to monitor agent positions and progress during simulation.

## Overview

Trajectory logging allows you to track a specific agent's position and progress throughout the simulation. When enabled, it prints the agent's coordinates, rotation, and progress value to stdout at each simulation step.

## Python API

### Enabling Trajectory Logging

```python
import madrona_escape_room

# Create manager
mgr = madrona_escape_room.SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=10,
    rand_seed=42,
    auto_reset=True
)

# Enable logging for world 0, agent 0 to stdout
mgr.enable_trajectory_logging(world_idx=0, agent_idx=0)

# Run simulation - trajectory will be printed to stdout
for i in range(100):
    mgr.step()
```

### Logging to File

You can optionally log trajectories to a file instead of stdout:

```python
# Enable logging to a specific file
mgr.enable_trajectory_logging(world_idx=0, agent_idx=0, filename="agent_trajectory.txt")

# Run simulation - trajectory will be written to file
for i in range(100):
    mgr.step()

# Always disable logging when done to close the file
mgr.disable_trajectory_logging()
```

**Important notes about file logging:**
- The file is opened in write mode ('w'), overwriting any existing content
- Always call `disable_trajectory_logging()` to properly close the file
- If the file cannot be opened, an error is printed and logging falls back to stdout

### Disabling Trajectory Logging

```python
# Disable trajectory logging (closes file if logging to file)
mgr.disable_trajectory_logging()
```

## Output Format

The trajectory logging outputs the following format to stdout:
```
Step    0: World 0 Agent 0: pos=(0.00,5.00,1.62) rot=0.0° progress=0.00
Step    1: World 0 Agent 0: pos=(0.00,5.12,1.62) rot=0.0° progress=0.01
Step    2: World 0 Agent 0: pos=(0.00,5.24,1.62) rot=0.0° progress=0.02
...
```

Fields:
- **Step**: Current simulation step since logging was enabled
- **World/Agent**: Indices of the tracked world and agent
- **pos**: Global position (x, y, z) in world coordinates
- **rot**: Agent rotation in degrees (converted from radians)
- **progress**: Current progress value (normalized maximum Y position reached)

## Use Cases

### 1. Debugging Agent Behavior
Track an agent to understand its movement patterns:
```python
# Enable logging for a poorly performing agent
worst_world, worst_agent = find_worst_performing_agent()
mgr.enable_trajectory_logging(worst_world, worst_agent)

# Run for a few steps to see behavior
for _ in range(50):
    mgr.step()
    
mgr.disable_trajectory_logging()
```

### 2. Training Diagnostics
Monitor agent progress during training:
```python
# During training loop
if episode % 100 == 0:  # Every 100 episodes
    mgr.enable_trajectory_logging(0, 0)  # Track first agent
    
# Run episode
for step in range(episode_length):
    actions = policy.get_actions(observations)
    mgr.step()
    
mgr.disable_trajectory_logging()
```

### 3. Comparing Policies
Track the same agent with different policies:
```python
# Test policy A
load_policy(policy_a)
mgr.enable_trajectory_logging(0, 0)
run_episode()
mgr.disable_trajectory_logging()

# Reset and test policy B
mgr.triggerReset(0)
load_policy(policy_b)
mgr.enable_trajectory_logging(0, 0)
run_episode()
mgr.disable_trajectory_logging()
```

## Performance Considerations

- Trajectory logging writes output at every step, which can impact performance
- Use sparingly during training - primarily for debugging
- Two options for capturing output:
  1. **Direct file logging** (recommended):
     ```python
     mgr.enable_trajectory_logging(0, 0, filename="trajectory.txt")
     ```
  2. **Shell redirection** (for stdout logging):
     ```bash
     python my_script.py > trajectory_log.txt
     ```
- File logging is more efficient than stdout redirection in most cases

## Integration with Other Tools

### Viewer Integration
The interactive viewer (./build/viewer) also supports trajectory tracking:
- Press 'T' to toggle tracking for the current world
- Use `--track` command line option: `./build/viewer 4 --cpu --track 0 0`

### Headless Mode
Trajectory logging works in headless mode:
```bash
./build/headless CPU 100 1000 --track-agent 0 0
```

## Example: Trajectory Analysis Script

```python
import madrona_escape_room
import matplotlib.pyplot as plt

# Create manager
mgr = madrona_escape_room.SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=1,
    rand_seed=42,
    auto_reset=True
)

# Log trajectory to file
trajectory_file = "agent_trajectory.log"
mgr.enable_trajectory_logging(0, 0, filename=trajectory_file)

for _ in range(200):
    mgr.step()

mgr.disable_trajectory_logging()

# Parse trajectory data from file
trajectory_data = []
with open(trajectory_file, 'r') as f:
    for line in f:
        if 'Step' in line:
            # Extract position data
            parts = line.split()
            step = int(parts[1].rstrip(':'))
            x = float(parts[6].replace('pos=(', '').rstrip(','))
            y = float(parts[7].rstrip(','))
            z = float(parts[8].rstrip(')'))
            progress = float(parts[10].replace('progress=', ''))
            trajectory_data.append((step, x, y, z, progress))

# Plot trajectory
if trajectory_data:
    steps, xs, ys, zs, progress = zip(*trajectory_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot XY position
    ax1.plot(xs, ys, 'b-', label='Agent Path')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Agent Trajectory (Top View)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot progress over time
    ax2.plot(steps, progress, 'g-', label='Progress')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Progress (Max Y)')
    ax2.set_title('Agent Progress Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('agent_trajectory.png')
    print(f"Saved trajectory plot with {len(trajectory_data)} data points")
```

## Notes

- Only one agent can be tracked at a time
- Logging state is not preserved across manager destruction
- Output can go to either stdout or a specified file
- When logging to file, the file is overwritten each time logging is enabled
- The step counter resets when logging is re-enabled
- Always call `disable_trajectory_logging()` to ensure files are properly closed
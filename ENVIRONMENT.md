# Madrona Escape Room Environment Specification

This document describes the action space, observation space, and basic usage of the Madrona Escape Room environment.

## Action Space

The environment uses discrete actions with 3 components per agent:

### Action Components

| Component | Range | Description |
|-----------|--------|-------------|
| **moveAmount** | [0, 3] | Movement speed |
| **moveAngle** | [0, 7] | Movement direction (relative to agent) |
| **rotate** | [0, 4] | Rotation speed |

### Movement Amount Values
- `0` = STOP - No movement
- `1` = SLOW - Slow movement speed
- `2` = MEDIUM - Medium movement speed  
- `3` = FAST - Fast movement speed

### Movement Angle Values (Agent-Relative)
- `0` = FORWARD - Move forward relative to agent's facing direction
- `1` = FORWARD_RIGHT - Move diagonally forward-right
- `2` = RIGHT - Move right (strafe)
- `3` = BACKWARD_RIGHT - Move diagonally backward-right
- `4` = BACKWARD - Move backward
- `5` = BACKWARD_LEFT - Move diagonally backward-left
- `6` = LEFT - Move left (strafe)
- `7` = FORWARD_LEFT - Move diagonally forward-left

**Important**: Movement angles are relative to the agent's current facing direction, not world coordinates. For example, moveAngle=6 (LEFT) means the agent moves 90° to their left, regardless of their orientation in the world.

### Rotation Values
- `0` = FAST_LEFT - Fast counter-clockwise rotation
- `1` = SLOW_LEFT - Slow counter-clockwise rotation
- `2` = NONE - No rotation (default/center value)
- `3` = SLOW_RIGHT - Slow clockwise rotation
- `4` = FAST_RIGHT - Fast clockwise rotation

## Observation Space

Each agent receives the following observations:

### Self Observation (5 values)
- **Position** (3 values): `[x, y, z]` - Agent's position in world coordinates
  - Normalized by world dimensions
- **Max Y** (1 value): Maximum Y coordinate reached (progress indicator)
  - Normalized by world length
- **Rotation** (1 value): Agent's rotation angle (theta)
  - Normalized by π

## Reward Structure

- **Progress Reward**: Given at episode end based on maximum Y position reached
- **Normalized**: Reward = maxY / worldLength
- **Per-Level Bonus**: Additional reward for completing rooms (if applicable)

## Episode Mechanics

- **Episode Length**: 200 steps (default)
- **Auto-Reset**: Episodes automatically reset when done flag is set
- **Done Condition**: Set when steps remaining reaches 0

## Python Usage Example

```python
import torch
import madrona_escape_room

# Create the simulation manager
mgr = madrona_escape_room.SimManager(
    exec_mode=madrona_escape_room.madrona.ExecMode.CPU,  # or CUDA
    gpu_id=-1,           # -1 for CPU, 0+ for GPU device
    num_worlds=1024,     # Number of parallel worlds
    rand_seed=42,        # Random seed
    auto_reset=True,     # Automatically reset episodes
    enable_batch_renderer=False  # Set True for rendering
)

# Get tensor references (zero-copy views)
action_tensor = mgr.action_tensor().to_torch()        # Shape: [num_worlds, num_agents, 3]
obs_tensor = mgr.self_observation_tensor().to_torch() # Shape: [num_worlds, num_agents, 5]
reward_tensor = mgr.reward_tensor().to_torch()        # Shape: [num_worlds, num_agents]
done_tensor = mgr.done_tensor().to_torch()            # Shape: [num_worlds, num_agents]

# Main training loop
for step in range(10000):
    # Set actions for all agents
    # Example: Move all agents forward at medium speed
    action_tensor[:, :, 0] = 2  # moveAmount = MEDIUM
    action_tensor[:, :, 1] = 0  # moveAngle = FORWARD
    action_tensor[:, :, 2] = 2  # rotate = NONE
    
    # Step the simulation
    mgr.step()
    
    # Read observations and rewards
    observations = obs_tensor.clone()  # Clone if you need to store
    rewards = reward_tensor.clone()
    dones = done_tensor.clone()
    
    # Process observations
    positions = observations[:, :, :3]  # x, y, z positions
    max_y_progress = observations[:, :, 3]  # Progress indicator
    rotations = observations[:, :, 4]  # Agent rotations
    
    # Your training logic here
    # ...
```

## Advanced Example: Agent-Specific Actions

```python
# Control specific agents differently
num_worlds = action_tensor.shape[0]
num_agents = action_tensor.shape[1]

for world_idx in range(num_worlds):
    for agent_idx in range(num_agents):
        if agent_idx == 0:
            # Agent 0: Move forward fast
            action_tensor[world_idx, agent_idx, 0] = 3  # FAST
            action_tensor[world_idx, agent_idx, 1] = 0  # FORWARD
            action_tensor[world_idx, agent_idx, 2] = 2  # NO ROTATION
        else:
            # Agent 1: Strafe left slowly while rotating right
            action_tensor[world_idx, agent_idx, 0] = 1  # SLOW
            action_tensor[world_idx, agent_idx, 1] = 6  # LEFT
            action_tensor[world_idx, agent_idx, 2] = 3  # SLOW_RIGHT

mgr.step()
```

## Using Action Constants

```python
# Import action constants for better readability
from madrona_escape_room import action

# Set actions using named constants
action_tensor[:, :, 0] = action.move_amount.MEDIUM
action_tensor[:, :, 1] = action.move_angle.FORWARD
action_tensor[:, :, 2] = action.rotate.NONE

# Example: Make agent strafe right
action_tensor[0, 0, 0] = action.move_amount.FAST
action_tensor[0, 0, 1] = action.move_angle.RIGHT
action_tensor[0, 0, 2] = action.rotate.NONE
```

## Zero-Copy Tensor Access

The environment provides zero-copy access to simulation data through DLPack:

```python
# Direct PyTorch tensor access (zero-copy)
obs_torch = mgr.self_observation_tensor().to_torch()

# Or use DLPack protocol
obs_dlpack = torch.from_dlpack(mgr.self_observation_tensor())

# Both tensors share the same memory with the simulation
# Modifications to these tensors directly affect the simulation
```

## Key Environment Parameters

- **World Dimensions**: 18 x 10 units (width x length)
- **Number of Agents**: 2 per world
- **Physics Timestep**: 0.04 seconds
- **Physics Substeps**: 4 per frame
- **Agent Speed**: 8 units/second (at FAST setting)
- **Rotation Speed**: 5 radians/second (at FAST setting)

## Tips for Using the Environment

1. **Strafing**: To make an agent strafe (move sideways), use moveAngle=2 (RIGHT) or moveAngle=6 (LEFT) while keeping rotation constant.

2. **Smooth Turning**: Combine forward movement with rotation for car-like motion:
   ```python
   action_tensor[:, :, 0] = 2  # MEDIUM speed
   action_tensor[:, :, 1] = 0  # FORWARD
   action_tensor[:, :, 2] = 3  # SLOW_RIGHT rotation
   ```

3. **Diagonal Movement**: Use the diagonal moveAngle values (1, 3, 5, 7) for 45° movement.

4. **Stop and Rotate**: Set moveAmount=0 to rotate in place without moving.

5. **Batch Operations**: The environment is optimized for batch operations. Set actions for all worlds simultaneously for best performance.
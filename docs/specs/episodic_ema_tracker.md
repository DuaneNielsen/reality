# Episodic Exponential Moving Average Tracker Specification

## Overview

A PyTorch-based tool for tracking episode statistics using exponential moving averages (EMA) in batch environments. Efficiently handles multiple parallel environments and provides smooth, responsive metrics for reinforcement learning training.

## Core Features

- **Batch Processing**: Handles N parallel environments simultaneously
- **Episode Tracking**: Accumulates rewards and counts steps until episode termination
- **EMA Statistics**: Maintains exponential moving averages for smooth metric tracking
- **Min/Max Tracking**: Records episodic minimum and maximum values
- **PyTorch Integration**: Native tensor operations for GPU acceleration

## Interface Specification

### Class: `EpisodicEMATracker`

```python
class EpisodicEMATracker(torch.nn.Module):
    def __init__(self,
                 num_envs: int,
                 alpha: float = 0.01,
                 device: torch.device = None):
        """
        Initialize the episodic EMA tracker.

        Args:
            num_envs: Number of parallel environments
            alpha: EMA decay factor (0 < alpha <= 1). Lower values = more smoothing
            device: PyTorch device for tensor operations

        Note:
            Must inherit from torch.nn.Module to avoid memory transfers and
            enable proper device management via registered buffers.
        """

    def step_update(self,
                   rewards: torch.Tensor,
                   dones: torch.Tensor) -> dict:
        """
        Update tracker with batch of rewards and done flags.

        Args:
            rewards: Tensor of shape (N,) containing per-environment rewards
            dones: Tensor of shape (N,) containing per-environment done flags

        Returns:
            dict: Statistics for completed episodes (if any)
        """

    def get_statistics(self) -> dict:
        """
        Get current EMA statistics.

        Returns:
            dict: Current smoothed statistics
        """

    def reset_env(self, env_indices: torch.Tensor):
        """
        Reset specific environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
```

## Data Structures

### Internal State
```python
# Per-environment episode accumulators (registered buffers)
self.episode_rewards: torch.Tensor  # Shape: (num_envs,)
self.episode_lengths: torch.Tensor  # Shape: (num_envs,)

# EMA statistics (via EMATracker modules)
self.ema_reward_tracker: EMATracker
self.ema_length_tracker: EMATracker

# Episode extremes tracking (registered buffers)
self.episode_reward_min: torch.Tensor  # Scalar tensor
self.episode_reward_max: torch.Tensor  # Scalar tensor
self.episode_length_min: torch.Tensor  # Scalar tensor
self.episode_length_max: torch.Tensor  # Scalar tensor

# Metadata (registered buffers)
self.episodes_completed: torch.Tensor  # Scalar tensor
self.total_steps: torch.Tensor         # Scalar tensor
```

### Return Values

#### `step_update()` Return Dict
```python
{
    # Only present when episodes complete
    "completed_episodes": {
        "count": int,                    # Number of episodes completed this step
        "rewards": torch.Tensor,         # Rewards of completed episodes
        "lengths": torch.Tensor,         # Lengths of completed episodes
        "env_indices": torch.Tensor      # Which environments completed
    }
}
```

#### `get_statistics()` Return Dict
```python
{
    # Wandb-compatible key names for direct logging
    "episodes/reward_ema": float,     # Smoothed average episode reward
    "episodes/length_ema": float,     # Smoothed average episode length
    "episodes/reward_min": float,     # Minimum episode reward seen
    "episodes/reward_max": float,     # Maximum episode reward seen
    "episodes/length_min": int,       # Minimum episode length seen
    "episodes/length_max": int,       # Maximum episode length seen
    "episodes/completed": int,        # Total episodes completed
    "episodes/total_steps": int       # Total steps across all environments
}
```

## Algorithm Details

### EMA Update Formula
```
ema_new = alpha * new_value + (1 - alpha) * ema_old
```

### Step Processing Logic
1. **Accumulate**: Add rewards to running episode totals, increment episode lengths
2. **Detect Completion**: Check `dones` tensor for completed episodes
3. **Update Statistics**: For completed episodes:
   - Update EMA values using episode totals
   - Update min/max extremes
   - Reset accumulators for completed environments
4. **Return Results**: Provide completed episode information

### Min/Max Tracking
- Initialize with 0.0 for rewards and 0 for lengths
- Update with actual min/max values once episodes start completing
- Update using `torch.min()` and `torch.max()` operations
- Persistent across all episodes

## Usage Example

```python
# Initialize tracker
tracker = EpisodicEMATracker(num_envs=64, alpha=0.01)

# Training loop
for step in range(num_steps):
    actions = policy(observations)
    observations, rewards, dones, infos = env.step(actions)

    # Update tracker
    completed = tracker.step_update(rewards, dones)

    # Log completed episodes
    if "completed_episodes" in completed:
        print(f"Step {step}: {completed['completed_episodes']['count']} episodes completed")

    # Direct wandb logging every 100 steps
    if step % 100 == 0:
        episode_stats = tracker.get_statistics()
        wandb.log(episode_stats, step=step)  # Direct logging with wandb-compatible keys
```

## Performance Considerations

- **Vectorized Operations**: Use PyTorch tensor operations for batch processing
- **GPU Compatibility**: All operations should be device-agnostic
- **Memory Efficiency**: Minimize tensor allocations in hot paths
- **Conditional Updates**: Only update EMA when episodes actually complete

## Device Management

### nn.Module Architecture
- **Required**: Must inherit from `torch.nn.Module`
- **Registered Buffers**: All internal state stored as registered buffers for automatic device handling
- **No Memory Transfers**: Avoids unnecessary CPU-GPU transfers during updates

### Device Behavior
- **Internal Processing**: All computations occur on the module's device
- **Input Handling**: Input tensors automatically moved to module device
- **Return Values**: Statistics returned as Python primitives (CPU values) for logging/monitoring
- **Buffer Management**: Use `self.register_buffer()` for all persistent tensors

### Example Device Usage
```python
# GPU usage
tracker = EpisodicEMATracker(num_envs=64, device=torch.device('cuda'))

# CPU tensors automatically moved to GPU
cpu_rewards = torch.tensor([1.0, 2.0, ...])  # CPU
cpu_dones = torch.tensor([False, True, ...]) # CPU

result = tracker.step_update(cpu_rewards, cpu_dones)  # Auto device handling
stats = tracker.get_statistics()  # Returns CPU values for logging
```

## Configuration Parameters

### Alpha Selection Guidelines
- `alpha = 0.001`: Very smooth, slow to respond (1000-episode effective window)
- `alpha = 0.01`: Smooth, moderate response (100-episode effective window)
- `alpha = 0.1`: Responsive, less smooth (10-episode effective window)
- `alpha = 1.0`: No smoothing, just current episode values

### Device Handling
- Default to CPU if no device specified
- Inherit device from input tensors when possible
- Support explicit device specification in constructor
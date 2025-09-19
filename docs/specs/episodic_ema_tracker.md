# Episodic Exponential Moving Average Tracker Specification

## Overview

A PyTorch-based tool for tracking episode statistics using exponential moving averages (EMA) in batch environments. Efficiently handles multiple parallel environments and provides smooth, responsive metrics for reinforcement learning training. Supports termination reason tracking to monitor how episodes end over the training process.

## Core Features

- **Batch Processing**: Handles N parallel environments simultaneously
- **Episode Tracking**: Accumulates rewards and counts steps until episode termination
- **EMA Statistics**: Maintains exponential moving averages for smooth metric tracking
- **Termination Reason Tracking**: Monitors episode termination causes with count-based probability tracking
- **Performance Optimized**: Streamlined implementation for minimal computational overhead
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
                   dones: torch.Tensor,
                   termination_reasons: torch.Tensor = None) -> dict:
        """
        Update tracker with batch of rewards and done flags.

        Args:
            rewards: Tensor of shape (N,) containing per-environment rewards
            dones: Tensor of shape (N,) containing per-environment done flags
            termination_reasons: Optional tensor of shape (N,) containing termination reasons for completed episodes

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

# Termination reason tracking (registered buffers)
self.termination_counts: torch.Tensor      # Shape: (num_termination_reasons,)
self.total_terminations: torch.Tensor      # Scalar tensor
# Note: EMA termination trackers kept for backward compatibility but not used in statistics

# Episode extremes tracking (registered buffers)
# Min/max tracking removed for performance optimization

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
    # Termination reason probabilities (count-based, not EMA)
    "episodes/termination_time_limit_prob": float,        # Count-based probability of time limit termination
    "episodes/termination_progress_complete_prob": float, # Count-based probability of progress completion
    "episodes/termination_collision_death_prob": float,   # Count-based probability of collision death
    # Min/max tracking removed for performance optimization
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
   - Update EMA values using batch means for performance
   - Update termination reason counts for probability calculation (if provided)
   - Reset accumulators for completed environments
4. **Return Results**: Provide completed episode information

### Performance Optimizations
- Early exit when no episodes complete (most common case)
- Batch EMA updates using mean of completed episodes
- Optimized device handling to avoid unnecessary tensor transfers
- Min/max tracking removed for maximum performance

## Termination Reason Constants

```python
from enum import IntEnum

class TerminationReason(IntEnum):
    """
    Termination reasons for episodes based on reward_termination_system.md
    """
    TIME_LIMIT = 0          # Episode ends after 200 steps (episodeLen)
    PROGRESS_COMPLETE = 1   # Agent reaches normalized_progress >= 1.0
    COLLISION_DEATH = 2     # Agent dies from collision
```

## Usage Examples

### Basic EMA Tracking

```python
# Initialize basic tracker
tracker = EpisodicEMATracker(num_envs=64, alpha=0.01)

# Training loop
for step in range(num_steps):
    actions = policy(observations)
    observations, rewards, dones, infos = env.step(actions)

    # Extract termination reasons from infos (optional)
    termination_reasons = None
    if dones.any():
        # Example: extract from environment info
        termination_reasons = torch.tensor([
            info.get('termination_reason', TerminationReason.TIME_LIMIT)
            for info in infos
        ])

    # Update tracker with termination reasons
    completed = tracker.step_update(rewards, dones, termination_reasons)

    # Log completed episodes
    if "completed_episodes" in completed:
        print(f"Step {step}: {completed['completed_episodes']['count']} episodes completed")

    # Direct wandb logging every 100 steps
    if step % 100 == 0:
        episode_stats = tracker.get_statistics()
        wandb.log(episode_stats, step=step)  # Direct logging with wandb-compatible keys

        # New termination probability stats are automatically included:
        # episodes/termination_time_limit_prob
        # episodes/termination_progress_complete_prob
        # episodes/termination_collision_death_prob
```

### Enhanced Tracking with Histograms

```python
# Initialize enhanced tracker with custom bins
reward_bins = [0, 5, 10, 20, 50, 100, 200]  # Custom reward ranges
length_bins = [1, 10, 25, 50, 100, 200]     # Custom length ranges

tracker = EpisodicEMATrackerWithHistogram(
    num_envs=64,
    alpha=0.01,
    reward_bins=reward_bins,
    length_bins=length_bins
)

# Training loop
for step in range(num_steps):
    actions = policy(observations)
    observations, rewards, dones, infos = env.step(actions)

    # Extract termination reasons from infos (optional)
    termination_reasons = None
    if dones.any():
        termination_reasons = torch.tensor([
            info.get('termination_reason', TerminationReason.TIME_LIMIT)
            for info in infos
        ])

    # Update tracker (both EMA and histograms)
    completed = tracker.step_update(rewards, dones, termination_reasons)

    # Regular EMA logging
    if step % 100 == 0:
        ema_stats = tracker.get_statistics()
        wandb.log(ema_stats, step=step)

    # Periodic histogram logging (less frequent due to larger data)
    if step % 1000 == 0:
        histogram_data = tracker.get_histograms()

        # Log histograms in wandb-compatible format
        wandb.log({
            "episode_rewards": wandb.Histogram(histogram_data["histograms/reward_counts"]),
            "episode_lengths": wandb.Histogram(histogram_data["histograms/length_counts"]),
            "reward_distribution": histogram_data["histograms/reward_distribution"],
            "length_distribution": histogram_data["histograms/length_distribution"]
        }, step=step)
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

## Enhanced Histogram Functionality

### FastVectorizedHistogram

An ultra-fast histogram implementation optimized for batch updates:

```python
class FastVectorizedHistogram(torch.nn.Module):
    def __init__(self, bin_edges, device=None):
        """
        Initialize histogram with pre-defined bin edges.

        Args:
            bin_edges: List or tensor of bin edge values (e.g., [0, 10, 20, 50, 100])
            device: PyTorch device for tensor operations
        """

    def update_batch(self, values):
        """Vectorized batch update using torch.bincount for maximum efficiency."""

    def get_probabilities(self):
        """Get normalized histogram as probabilities."""

    def get_counts(self):
        """Get raw bin counts."""

    def reset(self):
        """Reset all bin counts to zero."""
```

### EpisodicEMATrackerWithHistogram

Enhanced tracker that combines EMA statistics with histogram analysis:

```python
class EpisodicEMATrackerWithHistogram(EpisodicEMATracker):
    def __init__(self, num_envs: int, alpha: float = 0.01, device: torch.device = None,
                 reward_bins=None, length_bins=None):
        """
        Initialize enhanced tracker with histogram support.

        Args:
            reward_bins: List of bin edges for reward histogram
            length_bins: List of bin edges for length histogram
        """

    def get_statistics(self) -> dict:
        """Get EMA statistics only (clean separation from histograms)."""

    def get_histograms(self) -> dict:
        """Get histogram data in wandb-compatible format."""

    def reset_histograms(self):
        """Reset histogram counts while preserving EMA state."""
```

### Histogram Data Format

The `get_histograms()` method returns data compatible with `wandb.Histogram`:

```python
histograms = tracker.get_histograms()

# Direct wandb logging
wandb.log({
    "episode_rewards": wandb.Histogram(histograms["histograms/reward_counts"]),
    "episode_lengths": wandb.Histogram(histograms["histograms/length_counts"])
}, step=step)

# Or using the distribution arrays
wandb.log({
    "reward_distribution": histograms["histograms/reward_distribution"],
    "length_distribution": histograms["histograms/length_distribution"]
}, step=step)
```

#### Returned Keys

```python
{
    # Raw counts for wandb.Histogram()
    "histograms/reward_counts": np.array,     # [n1, n2, n3, ...] bin counts
    "histograms/length_counts": np.array,     # [n1, n2, n3, ...] bin counts

    # Normalized probability distributions
    "histograms/reward_distribution": np.array,  # [p1, p2, p3, ...] probabilities
    "histograms/length_distribution": np.array,  # [p1, p2, p3, ...] probabilities

    # Bin edge information for interpretation
    "histograms/reward_bin_edges": np.array,     # [0, 10, 20, 50, 100] edges
    "histograms/length_bin_edges": np.array      # [1, 10, 25, 50, 200] edges
}
```

### Default Bin Configurations

**Reward Bins**: `[0, 1, 5, 10, 20, 50, 100, 200]` (7 bins)
- Captures low, medium, and high reward ranges
- Logarithmic-like spacing for better resolution at lower values

**Length Bins**: `[1, 10, 25, 50, 100, 150, 200, 300]` (7 bins)
- Covers short to very long episodes
- Higher resolution for typical episode lengths (10-100 steps)

### Performance Characteristics

- **O(1) per episode completion** for histogram updates
- **Vectorized batch processing** handles multiple simultaneous completions
- **Memory efficient** - only stores bin counts, not individual values
- **Device agnostic** - automatic CPU/GPU tensor movement

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
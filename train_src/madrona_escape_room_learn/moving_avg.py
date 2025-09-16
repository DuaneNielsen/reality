from enum import IntEnum

import torch
import torch.nn as nn


class TerminationReason(IntEnum):
    """
    Termination reasons for episodes based on reward_termination_system.md
    """

    TIME_LIMIT = 0  # Episode ends after 200 steps (episodeLen)
    PROGRESS_COMPLETE = 1  # Agent reaches normalized_progress >= 1.0
    COLLISION_DEATH = 2  # Agent dies from collision


# Simple Exponential Moving Average tracker for scalar metrics
class EMATracker(nn.Module):
    def __init__(self, decay, disable=False):
        super().__init__()

        self.disable = disable
        if disable:
            return

        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer("one_minus_decay", 1 - self.decay)
        self.register_buffer("ema", torch.zeros([], dtype=torch.float32))
        self.register_buffer("N", torch.zeros([], dtype=torch.int64))

    def update(self, value):
        """Update EMA with a new value"""
        if self.disable:
            return

        self.N.add_(1)
        # Optimized EMA update: ema = decay * ema + (1-decay) * value
        self.ema.mul_(self.decay).add_(value, alpha=self.one_minus_decay)


# Exponential Moving Average mean and variance estimator for
# values and observations
class EMANormalizer(nn.Module):
    def __init__(self, decay, eps=1e-5, disable=False):
        super().__init__()

        self.disable = disable
        if disable:
            return

        self.eps = eps

        # Current parameter estimates
        self.register_buffer("mu", torch.zeros([], dtype=torch.float32))
        self.register_buffer("inv_sigma", torch.zeros([], dtype=torch.float32))
        self.register_buffer("sigma", torch.zeros([], dtype=torch.float32))

        # Intermediate values used to compute the moving average
        # decay and one_minus_decay don't strictly need to be tensors, but it's
        # critically important for floating point precision that
        # one_minus_decay is computed in fp32 rather than fp64 to
        # match the bias_correction computation below
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer("one_minus_decay", 1 - self.decay)

        self.register_buffer("mu_biased", torch.zeros([], dtype=torch.float32))
        self.register_buffer("sigma_sq_biased", torch.zeros([], dtype=torch.float32))
        self.register_buffer("N", torch.zeros([], dtype=torch.int64))

        nn.init.constant_(self.mu, 0)
        nn.init.constant_(self.inv_sigma, 0)
        nn.init.constant_(self.sigma, 0)

        nn.init.constant_(self.mu_biased, 0)
        nn.init.constant_(self.sigma_sq_biased, 0)
        nn.init.constant_(self.N, 0)

    def forward(self, amp, x):
        if self.disable:
            return x

        with amp.disable():
            if self.training:
                x_f32 = x.to(dtype=torch.float32)

                self.N.add_(1)
                bias_correction = -torch.expm1(self.N * torch.log(self.decay))

                self.mu_biased.mul_(self.decay).addcmul_(x_f32.mean(), self.one_minus_decay)

                new_mu = self.mu_biased / bias_correction

                # prev_mu needs to be unbiased (bias_correction only accounts
                # for the initial EMA with 0), since otherwise variance would
                # be off by a squared factor.
                # On the first iteration, simply treat x's variance as the
                # full estimate of variance
                if self.N == 1:
                    prev_mu = new_mu
                else:
                    prev_mu = self.mu

                sigma_sq_new = torch.mean((x_f32 - prev_mu) * (x_f32 - new_mu))

                self.sigma_sq_biased.mul_(self.decay).addcmul_(sigma_sq_new, self.one_minus_decay)

                sigma_sq = self.sigma_sq_biased / bias_correction

                # Write out new unbiased params
                self.mu = new_mu
                self.inv_sigma = torch.rsqrt(torch.clamp(sigma_sq, min=self.eps))
                self.sigma = torch.reciprocal(self.inv_sigma)

            return torch.addcmul(
                -self.mu * self.inv_sigma,
                x,
                self.inv_sigma,
            ).to(dtype=x.dtype)

    def invert(self, amp, normalized_x):
        if self.disable:
            return normalized_x

        with amp.disable():
            return torch.addcmul(
                self.mu,
                normalized_x.to(dtype=torch.float32),
                self.sigma,
            ).to(dtype=normalized_x.dtype)


# Episodic Exponential Moving Average tracker for batch environments
class EpisodicEMATracker(torch.nn.Module):
    """
    Episodic EMA tracker for batch reinforcement learning environments.

    Tracks episode statistics using exponential moving averages across multiple
    parallel environments. Efficiently handles episode boundaries and provides
    smooth metrics for training monitoring.
    """

    def __init__(
        self, num_envs: int, alpha: float = 0.01, device: torch.device = None, disable: bool = False
    ):
        """
        Initialize the episodic EMA tracker.

        Args:
            num_envs: Number of parallel environments
            alpha: EMA decay factor (0 < alpha <= 1). Lower values = more smoothing
            device: PyTorch device for tensor operations
            disable: If True, disable all tracking for performance testing

        Note:
            Inherits from torch.nn.Module to avoid memory transfers and
            enable proper device management via registered buffers.
        """
        super().__init__()

        self.num_envs = num_envs
        self.alpha = alpha
        self.disable = disable

        if self.disable:
            return

        # Per-environment episode accumulators (registered buffers for device handling)
        self.register_buffer("episode_rewards", torch.zeros(num_envs))
        self.register_buffer("episode_lengths", torch.zeros(num_envs, dtype=torch.int64))

        # EMA trackers for episode statistics
        # EMATracker uses decay parameter, so we pass (1-alpha) to get alpha*new + (1-alpha)*old
        self.ema_reward_tracker = EMATracker(1 - alpha)
        self.ema_length_tracker = EMATracker(1 - alpha)

        # Termination reason tracking
        num_termination_reasons = len(TerminationReason)
        self.register_buffer(
            "termination_counts", torch.zeros(num_termination_reasons, dtype=torch.int64)
        )
        self.register_buffer("total_terminations", torch.tensor(0, dtype=torch.int64))

        # EMA trackers for termination reason probabilities
        self.ema_termination_trackers = torch.nn.ModuleList(
            [EMATracker(1 - alpha) for _ in range(num_termination_reasons)]
        )

        # Removed min/max tracking for performance

        # Metadata (registered buffers)
        self.register_buffer("episodes_completed", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.int64))

        # Move to device if specified
        if device is not None:
            self.to(device)

    def step_update(
        self, rewards: torch.Tensor, dones: torch.Tensor, termination_reasons: torch.Tensor = None
    ) -> dict:
        """
        Update tracker with batch of rewards and done flags.

        Args:
            rewards: Tensor of shape (N,) containing per-environment rewards
            dones: Tensor of shape (N,) containing per-environment done flags
            termination_reasons: Optional tensor of shape (N,) containing termination reasons
                for completed episodes

        Returns:
            dict: Statistics for completed episodes (if any)
        """
        if self.disable:
            return {}

        # Optimize: avoid device moves if already on correct device
        if rewards.device != self.episode_rewards.device:
            rewards = rewards.to(self.episode_rewards.device)
        if dones.device != self.episode_rewards.device:
            dones = dones.to(self.episode_rewards.device)

        # Accumulate rewards and increment lengths
        self.episode_rewards += rewards
        self.episode_lengths += 1
        self.total_steps += self.num_envs

        # Early exit if no episodes completed (most common case)
        if not dones.any():
            return {}

        # Find completed episodes - use nonzero for better performance
        completed_mask = dones.bool()
        completed_indices = completed_mask.nonzero(as_tuple=False).squeeze(1)

        # Get completed episode data
        completed_rewards = self.episode_rewards[completed_mask]
        completed_lengths = self.episode_lengths[completed_mask]

        # Vectorized EMA updates (completely avoid Python loops and .item() calls)
        num_completed = len(completed_rewards)
        if num_completed > 0:
            # Batch update EMA with mean of completed episodes (much faster)
            mean_completed_reward = completed_rewards.mean()
            mean_completed_length = completed_lengths.float().mean()

            # Update EMAs with batch means (single tensor operation)
            self.ema_reward_tracker.update(mean_completed_reward)
            self.ema_length_tracker.update(mean_completed_length)

            # Update termination reason tracking
            if termination_reasons is not None:
                # Get termination reasons for completed episodes
                completed_termination_reasons = termination_reasons[completed_mask]

                # Update EMA for termination reason probabilities using batch probabilities
                for reason_idx in range(len(TerminationReason)):
                    # Calculate probability for this batch of completed episodes
                    reason_count = (completed_termination_reasons == reason_idx).sum()
                    batch_prob = reason_count.float() / num_completed if num_completed > 0 else 0.0

                    # Update EMA with this batch's probability
                    self.ema_termination_trackers[reason_idx].update(batch_prob)

                # Update cumulative counts for debugging/monitoring
                for reason_idx in range(len(TerminationReason)):
                    reason_count = (completed_termination_reasons == reason_idx).sum()
                    self.termination_counts[reason_idx] += reason_count

                self.total_terminations += num_completed

            # Removed min/max updates for performance

            # Update episode count
            self.episodes_completed += num_completed

            # Reset completed environments
            self.episode_rewards[completed_mask] = 0
            self.episode_lengths[completed_mask] = 0

            # Return completed episode info (avoid unnecessary clones)
            return {
                "completed_episodes": {
                    "count": num_completed,
                    "rewards": completed_rewards,
                    "lengths": completed_lengths,
                    "env_indices": completed_indices,
                }
            }

        return {}

    def get_statistics(self) -> dict:
        """
        Get current EMA statistics with wandb-compatible key names.

        Returns:
            dict: Current smoothed statistics as CPU primitives with wandb key names
        """
        if self.disable:
            return {}

        stats = {
            # Episode tracking (matching wandb log structure from train.py)
            "episodes/reward_ema": self.ema_reward_tracker.ema.item()
            if self.ema_reward_tracker.N > 0
            else 0.0,
            "episodes/length_ema": self.ema_length_tracker.ema.item()
            if self.ema_length_tracker.N > 0
            else 0.0,
            # Removed min/max statistics for performance
            "episodes/completed": self.episodes_completed.item(),
            "episodes/total_steps": self.total_steps.item(),
        }

        # Add termination reason probability EMAs using grouping pattern
        termination_reason_names = ["time_limit", "progress_complete", "collision_death"]
        for reason_idx, reason_name in enumerate(termination_reason_names):
            tracker = self.ema_termination_trackers[reason_idx]
            stats[f"episodes/termination/{reason_name}_prob"] = (
                tracker.ema.item() if tracker.N > 0 else 0.0
            )

        return stats

    def reset_env(self, env_indices: torch.Tensor):
        """
        Reset specific environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
        if self.disable:
            return
        env_indices = env_indices.to(self.episode_rewards.device)
        self.episode_rewards[env_indices] = 0
        self.episode_lengths[env_indices] = 0


# Fast vectorized histogram implementation
class FastVectorizedHistogram(torch.nn.Module):
    """
    Ultra-fast histogram using torch.searchsorted and torch.bincount.

    Uses pre-computed bin edges for maximum efficiency in batch updates.
    """

    def __init__(self, bin_edges, device=None):
        """
        Initialize histogram with pre-defined bin edges.

        Args:
            bin_edges: List or tensor of bin edge values (e.g., [0, 10, 20, 50, 100])
            device: PyTorch device for tensor operations
        """
        super().__init__()
        bin_edges = torch.tensor(bin_edges, dtype=torch.float32)
        bins = torch.zeros(len(bin_edges) - 1, dtype=torch.int64)

        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bins", bins)

        # Move to device if specified
        if device is not None:
            self.to(device)

    def update_batch(self, values):
        """
        Update histogram with batch of values - fully vectorized.

        Args:
            values: Tensor of values to add to histogram
        """
        if len(values) == 0:
            return

        # Optimize: avoid device move if already on correct device
        if values.device != self.bin_edges.device:
            values = values.to(self.bin_edges.device)

        # Find which bin each value belongs to using binary search
        bin_indices = torch.searchsorted(self.bin_edges[1:], values, right=False)

        # Clamp to valid range (handle values outside bin edges)
        bin_indices = torch.clamp(bin_indices, 0, len(self.bins) - 1)

        # Count occurrences of each bin index
        counts = torch.bincount(bin_indices, minlength=len(self.bins))

        # Add to existing bins
        self.bins += counts[: len(self.bins)]

    def get_probabilities(self):
        """Get normalized histogram as probabilities."""
        total = self.bins.sum()
        return self.bins.float() / total if total > 0 else self.bins.float()

    def get_counts(self):
        """Get raw bin counts."""
        return self.bins.clone()

    def reset(self):
        """Reset all bin counts to zero."""
        self.bins.zero_()


# Enhanced EMA tracker with histogram support
class EpisodicEMATrackerWithHistogram(EpisodicEMATracker):
    """
    Enhanced episodic EMA tracker that also maintains histograms of episode rewards and lengths.

    Provides detailed distribution analysis alongside EMA statistics for comprehensive
    episode performance monitoring.
    """

    def __init__(
        self,
        num_envs: int,
        alpha: float = 0.01,
        device: torch.device = None,
        reward_bins=None,
        length_bins=None,
        disable: bool = False,
    ):
        """
        Initialize enhanced tracker with histogram support.

        Args:
            num_envs: Number of parallel environments
            alpha: EMA decay factor (0 < alpha <= 1)
            device: PyTorch device for tensor operations
            reward_bins: List of bin edges for reward histogram (e.g., [0, 5, 10, 20, 50, 100])
            length_bins: List of bin edges for length histogram (e.g., [1, 10, 25, 50, 100, 200])
            disable: If True, disable all tracking for performance testing
        """
        super().__init__(num_envs, alpha, device, disable)

        if self.disable:
            return

        # Default reward bins (adjust based on your reward range)
        if reward_bins is None:
            reward_bins = [0, 1, 5, 10, 20, 50, 100, 200]

        # Default length bins (adjust based on your episode lengths)
        if length_bins is None:
            length_bins = [1, 10, 25, 50, 100, 150, 200, 300]

        # Initialize histograms
        self.reward_histogram = FastVectorizedHistogram(reward_bins, device)
        self.length_histogram = FastVectorizedHistogram(length_bins, device)

        # Store bin edges for statistics reporting
        self.reward_bin_edges = self.reward_histogram.bin_edges
        self.length_bin_edges = self.length_histogram.bin_edges

    def step_update(
        self, rewards: torch.Tensor, dones: torch.Tensor, termination_reasons: torch.Tensor = None
    ) -> dict:
        """
        Update tracker and histograms with batch of rewards and done flags.

        Args:
            rewards: Tensor of shape (N,) containing per-environment rewards
            dones: Tensor of shape (N,) containing per-environment done flags
            termination_reasons: Optional tensor of shape (N,) containing termination reasons
                for completed episodes

        Returns:
            dict: Statistics for completed episodes (if any) with histogram data
        """
        # Call parent method for EMA tracking
        result = super().step_update(rewards, dones, termination_reasons)

        # Update histograms if episodes completed
        if "completed_episodes" in result:
            completed_data = result["completed_episodes"]
            completed_rewards = completed_data["rewards"]
            completed_lengths = completed_data["lengths"]

            # Vectorized histogram updates
            self.reward_histogram.update_batch(completed_rewards)
            self.length_histogram.update_batch(completed_lengths.float())

        return result

    def get_statistics(self) -> dict:
        """
        Get current EMA statistics (without histogram data).

        Returns:
            dict: Base EMA statistics from parent class
        """
        return super().get_statistics()

    def get_histograms(self) -> dict:
        """
        Get histogram distributions and counts.

        Returns:
            dict: Histogram data with probability distributions, counts, and bin edges
        """
        if self.disable:
            return {}
        # Get histogram statistics
        reward_probs = self.reward_histogram.get_probabilities()
        length_probs = self.length_histogram.get_probabilities()
        reward_counts = self.reward_histogram.get_counts()
        length_counts = self.length_histogram.get_counts()

        return {
            # Probability distributions
            "histograms/reward_distribution": reward_probs.cpu().numpy(),
            "histograms/length_distribution": length_probs.cpu().numpy(),
            # Raw counts for debugging
            "histograms/reward_counts": reward_counts.cpu().numpy(),
            "histograms/length_counts": length_counts.cpu().numpy(),
            # Bin edges for interpretation
            "histograms/reward_bin_edges": self.reward_bin_edges.cpu().numpy(),
            "histograms/length_bin_edges": self.length_bin_edges.cpu().numpy(),
        }

    def reset_histograms(self):
        """Reset histogram counts while preserving EMA state."""
        if self.disable:
            return
        self.reward_histogram.reset()
        self.length_histogram.reset()

    def reset_env(self, env_indices: torch.Tensor):
        """Reset specific environments (inherits from parent)."""
        super().reset_env(env_indices)

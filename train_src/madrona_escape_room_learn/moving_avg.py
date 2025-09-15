import torch
import torch.nn as nn


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
        self.ema.mul_(self.decay).add_(value * self.one_minus_decay)


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

    def __init__(self, num_envs: int, alpha: float = 0.01, device: torch.device = None):
        """
        Initialize the episodic EMA tracker.

        Args:
            num_envs: Number of parallel environments
            alpha: EMA decay factor (0 < alpha <= 1). Lower values = more smoothing
            device: PyTorch device for tensor operations

        Note:
            Inherits from torch.nn.Module to avoid memory transfers and
            enable proper device management via registered buffers.
        """
        super().__init__()

        self.num_envs = num_envs
        self.alpha = alpha

        # Per-environment episode accumulators (registered buffers for device handling)
        self.register_buffer("episode_rewards", torch.zeros(num_envs))
        self.register_buffer("episode_lengths", torch.zeros(num_envs, dtype=torch.int64))

        # EMA trackers for episode statistics
        # EMATracker uses decay parameter, so we pass (1-alpha) to get alpha*new + (1-alpha)*old
        self.ema_reward_tracker = EMATracker(1 - alpha)
        self.ema_length_tracker = EMATracker(1 - alpha)

        # Episode extremes tracking (registered buffers)
        self.register_buffer("episode_reward_min", torch.tensor(float("inf")))
        self.register_buffer("episode_reward_max", torch.tensor(float("-inf")))
        self.register_buffer("episode_length_min", torch.tensor(float("inf")))
        self.register_buffer("episode_length_max", torch.tensor(0))

        # Metadata (registered buffers)
        self.register_buffer("episodes_completed", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.int64))

        # Move to device if specified
        if device is not None:
            self.to(device)

    def step_update(self, rewards: torch.Tensor, dones: torch.Tensor) -> dict:
        """
        Update tracker with batch of rewards and done flags.

        Args:
            rewards: Tensor of shape (N,) containing per-environment rewards
            dones: Tensor of shape (N,) containing per-environment done flags

        Returns:
            dict: Statistics for completed episodes (if any)
        """
        # Move tensors to device (auto-handled by nn.Module)
        rewards = rewards.to(self.episode_rewards.device)
        dones = dones.to(self.episode_rewards.device)

        # Accumulate rewards and increment lengths
        self.episode_rewards += rewards
        self.episode_lengths += 1
        self.total_steps += self.num_envs

        # Find completed episodes
        completed_mask = dones.bool()
        completed_indices = torch.where(completed_mask)[0]

        result = {}

        if len(completed_indices) > 0:
            # Get completed episode data
            completed_rewards = self.episode_rewards[completed_mask]
            completed_lengths = self.episode_lengths[completed_mask]

            # Update EMA statistics
            for reward in completed_rewards:
                self.ema_reward_tracker.update(reward.item())
            for length in completed_lengths:
                self.ema_length_tracker.update(length.item())

            # Update min/max statistics
            if len(completed_rewards) > 0:
                batch_reward_min = completed_rewards.min()
                batch_reward_max = completed_rewards.max()
                batch_length_min = completed_lengths.min()
                batch_length_max = completed_lengths.max()

                self.episode_reward_min = torch.min(self.episode_reward_min, batch_reward_min)
                self.episode_reward_max = torch.max(self.episode_reward_max, batch_reward_max)
                self.episode_length_min = torch.min(self.episode_length_min, batch_length_min)
                self.episode_length_max = torch.max(self.episode_length_max, batch_length_max)

            # Update episode count
            self.episodes_completed += len(completed_indices)

            # Reset completed environments
            self.episode_rewards[completed_mask] = 0
            self.episode_lengths[completed_mask] = 0

            # Return completed episode info
            result["completed_episodes"] = {
                "count": len(completed_indices),
                "rewards": completed_rewards.clone(),
                "lengths": completed_lengths.clone(),
                "env_indices": completed_indices.clone(),
            }

        return result

    def get_statistics(self) -> dict:
        """
        Get current EMA statistics with wandb-compatible key names.

        Returns:
            dict: Current smoothed statistics as CPU primitives with wandb key names
        """
        return {
            # Episode tracking (matching wandb log structure from train.py)
            "episodes/reward_ema": self.ema_reward_tracker.ema.item()
            if self.ema_reward_tracker.N > 0
            else 0.0,
            "episodes/length_ema": self.ema_length_tracker.ema.item()
            if self.ema_length_tracker.N > 0
            else 0.0,
            "episodes/reward_min": self.episode_reward_min.item()
            if not torch.isinf(self.episode_reward_min)
            else 0.0,
            "episodes/reward_max": self.episode_reward_max.item()
            if not torch.isinf(self.episode_reward_max)
            else 0.0,
            "episodes/length_min": int(self.episode_length_min.item())
            if not torch.isinf(self.episode_length_min)
            else 0,
            "episodes/length_max": int(self.episode_length_max.item()),
            "episodes/completed": self.episodes_completed.item(),
            "episodes/total_steps": self.total_steps.item(),
        }

    def reset_env(self, env_indices: torch.Tensor):
        """
        Reset specific environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
        env_indices = env_indices.to(self.episode_rewards.device)
        self.episode_rewards[env_indices] = 0
        self.episode_lengths[env_indices] = 0

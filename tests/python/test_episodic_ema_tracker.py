import os
import sys

import numpy as np
import pytest
import torch

# Add train_src to path for importing the tracker classes
train_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "train_src")
if train_src_path not in sys.path:
    sys.path.insert(0, train_src_path)

from madrona_escape_room_learn.moving_avg import EpisodicEMATracker


class TestEpisodicEMATracker:
    """Test suite for EpisodicEMATracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = EpisodicEMATracker(num_envs=4, alpha=0.1)

        assert tracker.num_envs == 4
        assert tracker.alpha == 0.1
        assert tracker.episodes_completed == 0
        assert tracker.total_steps == 0
        assert torch.equal(tracker.episode_rewards, torch.zeros(4))
        assert torch.equal(tracker.episode_lengths, torch.zeros(4, dtype=torch.int64))

    def test_single_step_no_done(self):
        """Test single step without episode completion."""
        tracker = EpisodicEMATracker(num_envs=3, alpha=0.1)

        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([False, False, False])

        result = tracker.step_update(rewards, dones)

        # No episodes completed
        assert "completed_episodes" not in result

        # Episode accumulators updated
        assert torch.equal(tracker.episode_rewards, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(tracker.episode_lengths, torch.tensor([1, 1, 1]))

        # Total steps updated
        assert tracker.total_steps == 3
        assert tracker.episodes_completed == 0

    def test_single_episode_completion(self):
        """Test single episode completion."""
        tracker = EpisodicEMATracker(num_envs=3, alpha=0.1)

        # Step 1: accumulate rewards
        rewards1 = torch.tensor([1.0, 2.0, 3.0])
        dones1 = torch.tensor([False, False, False])
        tracker.step_update(rewards1, dones1)

        # Step 2: complete episode in env 1
        rewards2 = torch.tensor([1.5, 1.0, 2.0])
        dones2 = torch.tensor([False, True, False])
        result = tracker.step_update(rewards2, dones2)

        # Check completed episode info
        assert "completed_episodes" in result
        completed = result["completed_episodes"]
        assert completed["count"] == 1
        assert torch.equal(completed["rewards"], torch.tensor([3.0]))  # 2.0 + 1.0
        assert torch.equal(completed["lengths"], torch.tensor([2]))
        assert torch.equal(completed["env_indices"], torch.tensor([1]))

        # Check environment reset
        expected_rewards = torch.tensor([2.5, 0.0, 5.0])  # env 1 reset to 0
        expected_lengths = torch.tensor([2, 0, 2])  # env 1 reset to 0
        assert torch.equal(tracker.episode_rewards, expected_rewards)
        assert torch.equal(tracker.episode_lengths, expected_lengths)

        # Check metadata
        assert tracker.episodes_completed == 1
        assert tracker.total_steps == 6

    def test_multiple_episodes_completion(self):
        """Test multiple episodes completing in same step."""
        tracker = EpisodicEMATracker(num_envs=4, alpha=0.1)

        # Accumulate some rewards
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dones = torch.tensor([False, False, False, False])
        tracker.step_update(rewards, dones)

        # Complete episodes in envs 0 and 2
        rewards = torch.tensor([0.5, 1.0, 1.5, 2.0])
        dones = torch.tensor([True, False, True, False])
        result = tracker.step_update(rewards, dones)

        # Check completed episodes
        completed = result["completed_episodes"]
        assert completed["count"] == 2
        assert torch.equal(completed["rewards"], torch.tensor([1.5, 4.5]))  # envs 0,2
        assert torch.equal(completed["lengths"], torch.tensor([2, 2]))
        assert torch.equal(completed["env_indices"], torch.tensor([0, 2]))

        # Check environment states
        expected_rewards = torch.tensor([0.0, 3.0, 0.0, 6.0])
        expected_lengths = torch.tensor([0, 2, 0, 2])
        assert torch.equal(tracker.episode_rewards, expected_rewards)
        assert torch.equal(tracker.episode_lengths, expected_lengths)

    def test_ema_computation(self):
        """Test EMA computation over multiple episodes."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.5)  # High alpha for easier testing

        # Complete first episode
        tracker.step_update(torch.tensor([10.0, 0.0]), torch.tensor([True, False]))
        stats = tracker.get_statistics()
        # With alpha=0.5: ema = 0.5 * 10.0 + (1-0.5) * 0 = 5.0
        assert abs(stats["episodes/reward_ema"] - 5.0) < 1e-6
        assert abs(stats["episodes/length_ema"] - 0.5) < 1e-6  # 0.5 * 1 + 0.5 * 0

        # Complete second episode with different values
        tracker.step_update(torch.tensor([0.0, 20.0]), torch.tensor([False, True]))
        stats = tracker.get_statistics()

        # EMA after second episode: 0.5 * 20.0 + 0.5 * 5.0 = 12.5
        assert abs(stats["episodes/reward_ema"] - 12.5) < 1e-6
        # EMA length: 0.5 * 2 + 0.5 * 0.5 = 1.25
        assert abs(stats["episodes/length_ema"] - 1.25) < 1e-6

    def test_min_max_tracking(self):
        """Test min/max statistics tracking."""
        tracker = EpisodicEMATracker(num_envs=3, alpha=0.1)

        # Complete episodes with varying rewards and lengths
        # Episode 1: reward=5, length=1
        tracker.step_update(torch.tensor([5.0, 0.0, 0.0]), torch.tensor([True, False, False]))

        # Episode 2: reward=10, length=3
        # Step 2: env 1 accumulates, length becomes 2
        tracker.step_update(torch.tensor([0.0, 5.0, 0.0]), torch.tensor([False, False, False]))
        # Step 3: env 1 completes with length 3
        tracker.step_update(torch.tensor([0.0, 5.0, 0.0]), torch.tensor([False, True, False]))

        # Episode 3: reward=1, length=4
        # Step 4: env 2 accumulates, length becomes 4
        tracker.step_update(torch.tensor([0.0, 0.0, 0.5]), torch.tensor([False, False, False]))
        # Step 5: env 2 completes with length 5
        tracker.step_update(torch.tensor([0.0, 0.0, 0.5]), torch.tensor([False, False, True]))

        stats = tracker.get_statistics()
        assert stats["episodes/reward_min"] == 1.0
        assert stats["episodes/reward_max"] == 10.0  # Episode 2: 5.0 + 5.0 = 10.0
        assert stats["episodes/length_min"] == 1
        assert (
            stats["episodes/length_max"] == 5
        )  # Episode 1: len=1, Episode 2: len=3, Episode 3: len=5
        assert stats["episodes/completed"] == 3

    def test_reset_env(self):
        """Test manual environment reset."""
        tracker = EpisodicEMATracker(num_envs=4, alpha=0.1)

        # Accumulate some rewards
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dones = torch.tensor([False, False, False, False])
        tracker.step_update(rewards, dones)

        # Reset environments 1 and 3
        tracker.reset_env(torch.tensor([1, 3]))

        expected_rewards = torch.tensor([1.0, 0.0, 3.0, 0.0])
        expected_lengths = torch.tensor([1, 0, 1, 0])
        assert torch.equal(tracker.episode_rewards, expected_rewards)
        assert torch.equal(tracker.episode_lengths, expected_lengths)

    def test_empty_batch(self):
        """Test handling of empty reward/done batches."""
        tracker = EpisodicEMATracker(num_envs=0, alpha=0.1)

        result = tracker.step_update(torch.tensor([]), torch.tensor([], dtype=torch.bool))
        assert "completed_episodes" not in result
        assert tracker.total_steps == 0

    def test_all_environments_done(self):
        """Test all environments completing episodes simultaneously."""
        tracker = EpisodicEMATracker(num_envs=3, alpha=0.1)

        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([True, True, True])
        result = tracker.step_update(rewards, dones)

        completed = result["completed_episodes"]
        assert completed["count"] == 3
        assert torch.equal(completed["rewards"], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(completed["lengths"], torch.tensor([1, 1, 1]))

        # All environments should be reset
        assert torch.equal(tracker.episode_rewards, torch.zeros(3))
        assert torch.equal(tracker.episode_lengths, torch.zeros(3, dtype=torch.int64))

    def test_device_handling(self):
        """Test proper device handling for tensors."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tracker = EpisodicEMATracker(num_envs=2, alpha=0.1, device=device)

            # CPU tensors should be moved to device
            rewards = torch.tensor([1.0, 2.0])  # CPU tensor
            dones = torch.tensor([False, True])  # CPU tensor

            result = tracker.step_update(rewards, dones)

            # Internal state should be on correct device (using nn.Module device handling)
            assert tracker.episode_rewards.device.type == device.type
            assert tracker.episode_lengths.device.type == device.type
            assert tracker.episode_reward_min.device.type == device.type
            assert tracker.episodes_completed.device.type == device.type

            # Returned tensors should be on correct device
            if "completed_episodes" in result:
                completed = result["completed_episodes"]
                assert completed["rewards"].device.type == device.type
                assert completed["lengths"].device.type == device.type
                assert completed["env_indices"].device.type == device.type

    def test_edge_case_zero_rewards(self):
        """Test handling of zero rewards."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        # Complete episode with zero reward
        rewards = torch.tensor([0.0, 1.0])
        dones = torch.tensor([True, False])
        result = tracker.step_update(rewards, dones)

        completed = result["completed_episodes"]
        assert torch.equal(completed["rewards"], torch.tensor([0.0]))

        stats = tracker.get_statistics()
        assert stats["episodes/reward_min"] == 0.0
        assert stats["episodes/reward_max"] == 0.0

    def test_edge_case_negative_rewards(self):
        """Test handling of negative rewards."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        # Complete episodes with negative rewards
        rewards = torch.tensor([-5.0, 3.0])
        dones = torch.tensor([True, True])
        tracker.step_update(rewards, dones)

        stats = tracker.get_statistics()
        assert stats["episodes/reward_min"] == -5.0
        assert stats["episodes/reward_max"] == 3.0

    def test_statistics_before_episodes(self):
        """Test statistics when no episodes have completed."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        stats = tracker.get_statistics()
        assert stats["episodes/reward_ema"] == 0.0
        assert stats["episodes/length_ema"] == 0.0
        assert stats["episodes/reward_min"] == 0.0
        assert stats["episodes/reward_max"] == 0.0
        assert stats["episodes/length_min"] == 0
        assert stats["episodes/length_max"] == 0
        assert stats["episodes/completed"] == 0
        assert stats["episodes/total_steps"] == 0

    def test_series_of_single_step_episodes(self):
        """Test a series of single-step episodes to verify EMA computation."""
        tracker = EpisodicEMATracker(num_envs=3, alpha=0.2)  # Higher alpha for easier verification

        # Episode 1: All environments complete with rewards [5, 10, 15]
        result = tracker.step_update(
            torch.tensor([5.0, 10.0, 15.0]), torch.tensor([True, True, True])
        )

        completed = result["completed_episodes"]
        assert completed["count"] == 3
        assert torch.equal(completed["rewards"], torch.tensor([5.0, 10.0, 15.0]))
        assert torch.equal(completed["lengths"], torch.tensor([1, 1, 1]))

        stats = tracker.get_statistics()
        # EMA updates sequentially:
        # After episode 1 (reward=5): ema = 0.2*5 + 0.8*0 = 1.0
        # After episode 2 (reward=10): ema = 0.2*10 + 0.8*1.0 = 2.8
        # After episode 3 (reward=15): ema = 0.2*15 + 0.8*2.8 = 5.24
        assert abs(stats["episodes/reward_ema"] - 5.24) < 1e-6
        # Length EMA: each episode has length 1
        # After 3 episodes: final ema = 0.2*1 + 0.8*(0.2*1 + 0.8*(0.2*1)) = 0.2 + 0.8*0.36 = 0.488
        assert abs(stats["episodes/length_ema"] - 0.488) < 1e-6
        assert stats["episodes/reward_min"] == 5.0
        assert stats["episodes/reward_max"] == 15.0
        assert stats["episodes/length_min"] == 1
        assert stats["episodes/length_max"] == 1
        assert stats["episodes/completed"] == 3

        # Episode 2: All environments complete with rewards [2, 4, 6]
        result = tracker.step_update(
            torch.tensor([2.0, 4.0, 6.0]), torch.tensor([True, True, True])
        )

        completed = result["completed_episodes"]
        assert completed["count"] == 3
        assert torch.equal(completed["rewards"], torch.tensor([2.0, 4.0, 6.0]))
        assert torch.equal(completed["lengths"], torch.tensor([1, 1, 1]))

        stats = tracker.get_statistics()
        # After 6 episodes total, EMA should reflect the running average
        # The exact values depend on the order of updates within each step
        assert stats["episodes/reward_min"] == 2.0  # New minimum
        assert stats["episodes/reward_max"] == 15.0  # Still the maximum
        assert stats["episodes/length_min"] == 1
        assert stats["episodes/length_max"] == 1
        assert stats["episodes/completed"] == 6

        # Episode 3: Mixed completion pattern
        result = tracker.step_update(
            torch.tensor([100.0, 8.0, 12.0]), torch.tensor([True, False, True])
        )

        completed = result["completed_episodes"]
        assert completed["count"] == 2
        assert torch.equal(completed["rewards"], torch.tensor([100.0, 12.0]))
        assert torch.equal(completed["lengths"], torch.tensor([1, 1]))

        stats = tracker.get_statistics()
        assert stats["episodes/reward_min"] == 2.0
        assert stats["episodes/reward_max"] == 100.0  # New maximum
        assert stats["episodes/length_min"] == 1
        assert stats["episodes/length_max"] == 1
        assert stats["episodes/completed"] == 8

    def test_device_behavior_specification(self):
        """Test that internal processing occurs on module device while statistics return CPU values."""
        if torch.cuda.is_available():
            # Test with GPU device
            device = torch.device("cuda")
            tracker = EpisodicEMATracker(num_envs=2, alpha=0.1, device=device)

            # Verify all internal buffers are on GPU
            assert tracker.episode_rewards.device.type == "cuda"
            assert tracker.episode_lengths.device.type == "cuda"
            assert tracker.episode_reward_min.device.type == "cuda"
            assert tracker.episodes_completed.device.type == "cuda"

            # CPU input tensors
            cpu_rewards = torch.tensor([5.0, 10.0])  # CPU
            cpu_dones = torch.tensor([True, True])  # CPU

            # Verify inputs are on CPU before update
            assert cpu_rewards.device.type == "cpu"
            assert cpu_dones.device.type == "cpu"

            # Update (should auto-move inputs to GPU for processing)
            result = tracker.step_update(cpu_rewards, cpu_dones)

            # Verify internal buffers are still on GPU and updated
            assert tracker.episode_rewards.device.type == "cuda"
            assert tracker.episodes_completed.item() == 2  # Updated

            # Verify returned tensors are on GPU (same device as module)
            completed = result["completed_episodes"]
            assert completed["rewards"].device.type == "cuda"
            assert completed["lengths"].device.type == "cuda"

            # CRITICAL TEST: Verify get_statistics() returns CPU primitives
            stats = tracker.get_statistics()

            # All statistics should be Python primitives (not tensors)
            assert isinstance(stats["episodes/reward_ema"], float)
            assert isinstance(stats["episodes/length_ema"], float)
            assert isinstance(stats["episodes/reward_min"], float)
            assert isinstance(stats["episodes/reward_max"], float)
            assert isinstance(stats["episodes/length_min"], int)
            assert isinstance(stats["episodes/length_max"], int)
            assert isinstance(stats["episodes/completed"], int)
            assert isinstance(stats["episodes/total_steps"], int)

            # Verify values are correct (not just type)
            assert stats["episodes/reward_min"] == 5.0
            assert stats["episodes/reward_max"] == 10.0
            assert stats["episodes/length_min"] == 1
            assert stats["episodes/length_max"] == 1
            assert stats["episodes/completed"] == 2

        # Also test CPU-only behavior for completeness
        cpu_tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        # Update with CPU tensors
        result = cpu_tracker.step_update(torch.tensor([3.0, 7.0]), torch.tensor([True, True]))

        # Verify statistics are still Python primitives (not tensors)
        stats = cpu_tracker.get_statistics()
        assert isinstance(stats["episodes/reward_ema"], float)
        assert isinstance(stats["episodes/completed"], int)
        assert stats["episodes/completed"] == 2

    def test_long_episode_simulation(self):
        """Test simulation of longer episodes with realistic rewards."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.01)

        # Simulate a 100-step episode
        for step in range(99):
            rewards = torch.tensor([0.1, 0.2])  # Small incremental rewards
            dones = torch.tensor([False, False])
            result = tracker.step_update(rewards, dones)
            assert "completed_episodes" not in result

        # Complete episodes
        rewards = torch.tensor([1.0, 2.0])  # Final rewards
        dones = torch.tensor([True, True])
        result = tracker.step_update(rewards, dones)

        completed = result["completed_episodes"]
        assert completed["count"] == 2
        # Episode rewards should be accumulated: 99*0.1 + 1.0 = 10.9, 99*0.2 + 2.0 = 21.8
        expected_rewards = torch.tensor([10.9, 21.8])
        assert torch.allclose(completed["rewards"], expected_rewards, atol=1e-6)
        assert torch.equal(completed["lengths"], torch.tensor([100, 100]))

        stats = tracker.get_statistics()
        assert stats["episodes/length_min"] == 100
        assert stats["episodes/length_max"] == 100
        assert stats["episodes/completed"] == 2
        assert stats["episodes/total_steps"] == 200


if __name__ == "__main__":
    pytest.main([__file__])

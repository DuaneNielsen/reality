import os
import sys

import numpy as np
import pytest
import torch

# Add train_src to path for importing the tracker classes
train_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "train_src")
if train_src_path not in sys.path:
    sys.path.insert(0, train_src_path)

from madrona_escape_room_learn.moving_avg import (
    EpisodicEMATracker,
    EpisodicEMATrackerWithHistogram,
    FastVectorizedHistogram,
)


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
        """Test device behavior: processing on module device, statistics return CPU values."""
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


class TestFastVectorizedHistogram:
    """Test suite for FastVectorizedHistogram."""

    def test_histogram_initialization(self):
        """Test histogram initialization with custom bin edges."""
        bin_edges = [0, 10, 20, 50, 100]
        histogram = FastVectorizedHistogram(bin_edges)

        assert len(histogram.bins) == 4  # num_bins = len(edges) - 1
        assert torch.equal(
            histogram.bin_edges, torch.tensor([0, 10, 20, 50, 100], dtype=torch.float32)
        )
        assert torch.equal(histogram.bins, torch.zeros(4, dtype=torch.int64))

    def test_single_value_update(self):
        """Test updating histogram with single values."""
        bin_edges = [0, 10, 20, 50, 100]
        histogram = FastVectorizedHistogram(bin_edges)

        # Test values in different bins
        histogram.update_batch(torch.tensor([5.0]))  # Bin 0 (0-10)
        histogram.update_batch(torch.tensor([15.0]))  # Bin 1 (10-20)
        histogram.update_batch(torch.tensor([75.0]))  # Bin 3 (50-100)

        expected_counts = torch.tensor([1, 1, 0, 1], dtype=torch.int64)
        assert torch.equal(histogram.bins, expected_counts)

    def test_batch_update(self):
        """Test vectorized batch updates."""
        bin_edges = [0, 10, 20, 50, 100]
        histogram = FastVectorizedHistogram(bin_edges)

        # Batch update with multiple values
        values = torch.tensor([5.0, 15.0, 25.0, 75.0, 8.0, 12.0])
        histogram.update_batch(values)

        # Expected: [5, 8] -> bin 0, [15, 12] -> bin 1, [25] -> bin 2, [75] -> bin 3
        expected_counts = torch.tensor([2, 2, 1, 1], dtype=torch.int64)
        assert torch.equal(histogram.bins, expected_counts)

    def test_edge_cases(self):
        """Test edge cases like empty batches and out-of-range values."""
        bin_edges = [0, 10, 20, 50, 100]
        histogram = FastVectorizedHistogram(bin_edges)

        # Empty batch
        histogram.update_batch(torch.tensor([]))
        assert torch.equal(histogram.bins, torch.zeros(4, dtype=torch.int64))

        # Out-of-range values (should be clamped)
        histogram.update_batch(torch.tensor([-5.0, 150.0]))  # Below min, above max
        expected_counts = torch.tensor(
            [1, 0, 0, 1], dtype=torch.int64
        )  # Clamped to first and last bins
        assert torch.equal(histogram.bins, expected_counts)

    def test_probabilities(self):
        """Test probability calculation."""
        bin_edges = [0, 10, 20, 50, 100]
        histogram = FastVectorizedHistogram(bin_edges)

        # Add some values
        values = torch.tensor([5.0, 15.0, 15.0, 25.0])  # 1, 2, 0, 1 in bins
        histogram.update_batch(values)

        probs = histogram.get_probabilities()
        expected_probs = torch.tensor([1 / 4, 2 / 4, 1 / 4, 0 / 4], dtype=torch.float32)
        assert torch.allclose(probs, expected_probs)

    def test_reset(self):
        """Test histogram reset functionality."""
        bin_edges = [0, 10, 20, 50, 100]
        histogram = FastVectorizedHistogram(bin_edges)

        # Add some data
        histogram.update_batch(torch.tensor([5.0, 15.0, 25.0]))
        assert not torch.equal(histogram.bins, torch.zeros(4, dtype=torch.int64))

        # Reset
        histogram.reset()
        assert torch.equal(histogram.bins, torch.zeros(4, dtype=torch.int64))

    def test_device_handling(self):
        """Test device handling for histogram."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            bin_edges = [0, 10, 20, 50, 100]
            histogram = FastVectorizedHistogram(bin_edges, device=device)

            # Verify histogram is on GPU
            assert histogram.bin_edges.device.type == "cuda"
            assert histogram.bins.device.type == "cuda"

            # CPU input should work
            cpu_values = torch.tensor([5.0, 15.0])
            histogram.update_batch(cpu_values)

            # Verify histogram was updated
            assert histogram.bins.sum() == 2


class TestEpisodicEMATrackerWithHistogram:
    """Test suite for EpisodicEMATrackerWithHistogram."""

    def test_initialization_with_default_bins(self):
        """Test initialization with default bin configurations."""
        tracker = EpisodicEMATrackerWithHistogram(num_envs=3, alpha=0.1)

        # Check that histograms are initialized
        assert hasattr(tracker, "reward_histogram")
        assert hasattr(tracker, "length_histogram")

        # Check default bins
        assert len(tracker.reward_histogram.bins) == 7  # Default reward bins
        assert len(tracker.length_histogram.bins) == 7  # Default length bins

    def test_initialization_with_custom_bins(self):
        """Test initialization with custom bin configurations."""
        reward_bins = [0, 5, 10, 25, 100]
        length_bins = [1, 10, 50, 200]

        tracker = EpisodicEMATrackerWithHistogram(
            num_envs=2, alpha=0.1, reward_bins=reward_bins, length_bins=length_bins
        )

        assert len(tracker.reward_histogram.bins) == 4  # 5 edges = 4 bins
        assert len(tracker.length_histogram.bins) == 3  # 4 edges = 3 bins
        assert torch.equal(tracker.reward_bin_edges, torch.tensor(reward_bins, dtype=torch.float32))
        assert torch.equal(tracker.length_bin_edges, torch.tensor(length_bins, dtype=torch.float32))

    def test_histogram_updates_on_episode_completion(self):
        """Test that histograms are updated when episodes complete."""
        reward_bins = [0, 10, 20, 50, 100]
        length_bins = [1, 5, 10, 20, 50]

        tracker = EpisodicEMATrackerWithHistogram(
            num_envs=3, alpha=0.1, reward_bins=reward_bins, length_bins=length_bins
        )

        # Episode 1: rewards [8, 15, 75] -> bins [0, 1, 3], lengths all 1 -> bin 0
        tracker.step_update(torch.tensor([8.0, 15.0, 75.0]), torch.tensor([True, True, True]))

        histograms = tracker.get_histograms()

        # Check reward histogram
        expected_reward_counts = torch.tensor([1, 1, 0, 1], dtype=torch.int64)
        assert torch.equal(
            torch.tensor(histograms["histograms/reward_counts"]), expected_reward_counts
        )

        # Check length histogram (all episodes length 1)
        expected_length_counts = torch.tensor([3, 0, 0, 0], dtype=torch.int64)
        assert torch.equal(
            torch.tensor(histograms["histograms/length_counts"]), expected_length_counts
        )

    def test_separate_statistics_and_histograms(self):
        """Test that get_statistics() and get_histograms() return separate data."""
        tracker = EpisodicEMATrackerWithHistogram(num_envs=2, alpha=0.1)

        # Complete an episode
        tracker.step_update(torch.tensor([10.0, 20.0]), torch.tensor([True, True]))

        # Get statistics (should be EMA data only)
        stats = tracker.get_statistics()
        histogram_keys = [k for k in stats.keys() if k.startswith("histograms/")]
        assert len(histogram_keys) == 0  # No histogram data in statistics

        # Get histograms (should have histogram data)
        histograms = tracker.get_histograms()
        histogram_keys = [k for k in histograms.keys() if k.startswith("histograms/")]
        assert (
            len(histogram_keys) == 6
        )  # Distribution, counts, and bin edges for both reward and length

        # Verify expected keys in histograms
        expected_keys = {
            "histograms/reward_distribution",
            "histograms/length_distribution",
            "histograms/reward_counts",
            "histograms/length_counts",
            "histograms/reward_bin_edges",
            "histograms/length_bin_edges",
        }
        assert set(histograms.keys()) == expected_keys

    def test_histogram_reset_preserves_ema(self):
        """Test that resetting histograms preserves EMA statistics."""
        tracker = EpisodicEMATrackerWithHistogram(num_envs=2, alpha=0.2)

        # Complete some episodes
        tracker.step_update(torch.tensor([10.0, 20.0]), torch.tensor([True, True]))
        tracker.step_update(torch.tensor([15.0, 25.0]), torch.tensor([True, True]))

        # Get statistics before reset
        stats_before = tracker.get_statistics()
        histograms_before = tracker.get_histograms()

        # Reset histograms
        tracker.reset_histograms()

        # Get statistics after reset
        stats_after = tracker.get_statistics()
        histograms_after = tracker.get_histograms()

        # EMA statistics should be preserved
        assert stats_before["episodes/reward_ema"] == stats_after["episodes/reward_ema"]
        assert stats_before["episodes/length_ema"] == stats_after["episodes/length_ema"]
        assert stats_before["episodes/completed"] == stats_after["episodes/completed"]

        # Histogram counts should be reset
        assert np.sum(histograms_after["histograms/reward_counts"]) == 0
        assert np.sum(histograms_after["histograms/length_counts"]) == 0
        assert np.sum(histograms_before["histograms/reward_counts"]) > 0
        assert np.sum(histograms_before["histograms/length_counts"]) > 0

    def test_long_episodes_histogram_tracking(self):
        """Test histogram tracking for episodes of varying lengths."""
        length_bins = [1, 5, 10, 25, 50]
        tracker = EpisodicEMATrackerWithHistogram(num_envs=3, alpha=0.1, length_bins=length_bins)

        # Create episodes of different lengths
        # Episode 1: length 1 (immediate completion)
        tracker.step_update(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([True, False, False]))

        # Episode 2: length 7 (env 1 completes after 6 more steps)
        for _ in range(6):
            tracker.step_update(torch.tensor([0.0, 1.0, 1.0]), torch.tensor([False, False, False]))
        tracker.step_update(torch.tensor([0.0, 1.0, 1.0]), torch.tensor([False, True, False]))

        # Episode 3: length 15 (env 2 completes after 14 more steps)
        for _ in range(14):
            tracker.step_update(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([False, False, False]))
        tracker.step_update(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([False, False, True]))

        histograms = tracker.get_histograms()
        length_counts = histograms["histograms/length_counts"]

        # Expected: 1 episode in bin 0 (length 1), 1 in bin 1 (length 7), 1 in bin 2 (length 15)
        expected_counts = np.array([1, 1, 1, 0])  # bins: [1-5), [5-10), [10-25), [25-50)
        assert np.array_equal(length_counts, expected_counts)

    def test_device_handling_with_histograms(self):
        """Test device handling for tracker with histograms."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tracker = EpisodicEMATrackerWithHistogram(num_envs=2, alpha=0.1, device=device)

            # Verify histograms are on GPU
            assert tracker.reward_histogram.bin_edges.device.type == "cuda"
            assert tracker.length_histogram.bin_edges.device.type == "cuda"

            # CPU inputs should work
            cpu_rewards = torch.tensor([10.0, 20.0])
            cpu_dones = torch.tensor([True, True])

            tracker.step_update(cpu_rewards, cpu_dones)
            histograms = tracker.get_histograms()

            # Verify histograms were updated
            assert np.sum(histograms["histograms/reward_counts"]) == 2
            assert np.sum(histograms["histograms/length_counts"]) == 2

            # Statistics should still return CPU values
            stats = tracker.get_statistics()
            assert isinstance(stats["episodes/reward_ema"], float)
            assert isinstance(stats["episodes/completed"], int)

    def test_disable_flag_functionality(self):
        """Test that disable flag prevents all tracking and returns empty dicts."""
        # Test basic EMA tracker with disable=True
        disabled_tracker = EpisodicEMATracker(num_envs=4, alpha=0.1, disable=True)

        # Test step_update returns empty dict
        result = disabled_tracker.step_update(
            torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([True, True, True, True])
        )
        assert result == {}

        # Test get_statistics returns empty dict
        stats = disabled_tracker.get_statistics()
        assert stats == {}

        # Test reset_env does nothing (should not crash)
        disabled_tracker.reset_env(torch.tensor([0, 1]))

        # Test histogram tracker with disable=True
        disabled_hist_tracker = EpisodicEMATrackerWithHistogram(num_envs=2, alpha=0.1, disable=True)

        # Test step_update returns empty dict
        result = disabled_hist_tracker.step_update(
            torch.tensor([5.0, 10.0]), torch.tensor([True, True])
        )
        assert result == {}

        # Test get_statistics returns empty dict
        stats = disabled_hist_tracker.get_statistics()
        assert stats == {}

        # Test get_histograms returns empty dict
        histograms = disabled_hist_tracker.get_histograms()
        assert histograms == {}

        # Test reset_histograms does nothing (should not crash)
        disabled_hist_tracker.reset_histograms()

    def test_disable_flag_performance_benefit(self):
        """Test that disabled tracker provides significant performance benefit."""
        import time

        num_envs = 100
        num_steps = 50

        # Test disabled tracker performance
        disabled_tracker = EpisodicEMATrackerWithHistogram(num_envs=num_envs, disable=True)

        start_time = time.time()
        for i in range(num_steps):
            rewards = torch.randn(num_envs) * 0.1
            dones = torch.rand(num_envs) < 0.1  # 10% done rate
            disabled_tracker.step_update(rewards, dones)
            disabled_tracker.get_statistics()
            disabled_tracker.get_histograms()
        disabled_time = time.time() - start_time

        # Test enabled tracker performance
        enabled_tracker = EpisodicEMATrackerWithHistogram(num_envs=num_envs, disable=False)

        start_time = time.time()
        for i in range(num_steps):
            rewards = torch.randn(num_envs) * 0.1
            dones = torch.rand(num_envs) < 0.1  # 10% done rate
            enabled_tracker.step_update(rewards, dones)
            enabled_tracker.get_statistics()
            enabled_tracker.get_histograms()
        enabled_time = time.time() - start_time

        # Disabled should be significantly faster (at least 10x)
        speedup = enabled_time / disabled_time
        assert speedup > 10.0, f"Expected >10x speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__])

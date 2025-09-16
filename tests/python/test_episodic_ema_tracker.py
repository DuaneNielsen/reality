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
    EMATracker,
    EpisodicEMATracker,
    EpisodicEMATrackerWithHistogram,
    FastVectorizedHistogram,
    TerminationReason,
)


# Basic EMATracker tests
def test_ema_tracker_basic():
    """Test basic EMATracker functionality"""
    ema = EMATracker(decay=0.9)

    # Initially should be 0
    assert ema.ema.item() == 0.0
    assert ema.N.item() == 0

    # First update should set EMA to the value
    ema.update(10)
    assert abs(ema.ema.item() - 10.0 * 0.1) < 1e-6  # First update: 0 * 0.9 + 10 * 0.1 = 1.0
    assert ema.N.item() == 1

    # Second update should apply EMA formula
    ema.update(20)
    expected = 1.0 * 0.9 + 20.0 * 0.1  # 2.9
    assert abs(ema.ema.item() - expected) < 1e-6
    assert ema.N.item() == 2


def test_ema_tracker_disabled():
    """Test EMATracker when disabled"""
    ema = EMATracker(decay=0.9, disable=True)

    ema.update(10)
    ema.update(20)

    # When disabled, tensors aren't created, so we can't access them
    # This is expected behavior - disabled trackers don't track anything


def test_ema_tracker_device():
    """Test EMATracker can be moved to device"""
    ema = EMATracker(decay=0.9)

    # Move to CPU (should work on any system)
    ema_cpu = ema.to("cpu")

    ema_cpu.update(15)
    assert abs(ema_cpu.ema.item() - 15.0 * 0.1) < 1e-6


def test_episode_length_ema_sequence():
    """Test EMA with a sequence of episode lengths"""
    ema = EMATracker(decay=0.9)

    # Simulate a sequence of episode lengths
    episode_lengths = [10, 15, 20, 25, 30]  # Increasing episode lengths (improving agent)

    for length in episode_lengths:
        ema.update(length)

    # After 5 updates, we should have a reasonable EMA
    final_ema = ema.ema.item()
    assert ema.N.item() == 5
    assert 0 <= final_ema <= 30  # Should be somewhere in the range
    # With high decay (0.9), the EMA will be weighted toward earlier values
    # Manual calculation: 10 -> 0.9*10+0.1*15=10.5 -> 0.9*10.5+0.1*20=11.45 -> etc.

    print(f"Episode length EMA after sequence {episode_lengths}: {final_ema:.2f}")


def test_episode_length_ema_recent_bias():
    """Test EMA with different decay values showing recent bias"""
    # Test with high decay (conservative, biased toward old data)
    ema_conservative = EMATracker(decay=0.95)

    # Test with low decay (responsive, biased toward recent data)
    ema_responsive = EMATracker(decay=0.7)

    # Start with short episodes (bad performance)
    initial_episodes = [5, 8, 12, 10, 7]  # Short episodes (hitting obstacles)

    for length in initial_episodes:
        ema_conservative.update(length)
        ema_responsive.update(length)

    conservative_after_bad = ema_conservative.ema.item()
    responsive_after_bad = ema_responsive.ema.item()

    # Now agent improves - long episodes
    improved_episodes = [45, 60, 80, 90, 100]  # Much longer episodes (good navigation)

    for length in improved_episodes:
        ema_conservative.update(length)
        ema_responsive.update(length)

    conservative_after_good = ema_conservative.ema.item()
    responsive_after_good = ema_responsive.ema.item()

    # The responsive EMA should be much higher (closer to recent good performance)
    # The conservative EMA should still be dragged down by the initial bad performance
    assert responsive_after_good > conservative_after_good
    assert responsive_after_good > 70  # Should be closer to recent high values
    assert conservative_after_good < responsive_after_good  # Should be lower due to history

    print(
        f"After bad episodes - Conservative: {conservative_after_bad:.1f}, "
        f"Responsive: {responsive_after_bad:.1f}"
    )
    print(
        f"After good episodes - Conservative: {conservative_after_good:.1f}, "
        f"Responsive: {responsive_after_good:.1f}"
    )
    print(
        f"Responsive EMA adapted {responsive_after_good - responsive_after_bad:.1f} steps "
        f"vs Conservative {conservative_after_good - conservative_after_bad:.1f} steps"
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

    def test_episode_completion_tracking(self):
        """Test episode completion tracking (min/max removed for performance)."""
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
        # Min/max tracking removed for performance - only check EMA and counts
        assert stats["episodes/completed"] == 3
        assert "episodes/reward_ema" in stats
        assert "episodes/length_ema" in stats

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

        # Min/max tracking removed for performance - no additional assertions needed

    def test_edge_case_negative_rewards(self):
        """Test handling of negative rewards."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        # Complete episodes with negative rewards
        rewards = torch.tensor([-5.0, 3.0])
        dones = torch.tensor([True, True])
        tracker.step_update(rewards, dones)

        # Min/max tracking removed for performance - no additional assertions needed

    def test_statistics_before_episodes(self):
        """Test statistics when no episodes have completed."""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        stats = tracker.get_statistics()
        assert stats["episodes/reward_ema"] == 0.0
        assert stats["episodes/length_ema"] == 0.0
        # Min/max tracking removed for performance
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
        # EMA now updates with batch mean due to performance optimization:
        # Mean of [5, 10, 15] = 10.0
        # EMA update: ema = 0.2*10.0 + 0.8*0 = 2.0
        assert abs(stats["episodes/reward_ema"] - 2.0) < 1e-6
        # Length EMA: all episodes have length 1, so mean = 1.0
        # EMA update: ema = 0.2*1.0 + 0.8*0 = 0.2
        assert abs(stats["episodes/length_ema"] - 0.2) < 1e-6
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
        # Second batch mean: [2, 4, 6] = 4.0
        # EMA update: ema = 0.2*4.0 + 0.8*2.0 = 0.8 + 1.6 = 2.4
        assert abs(stats["episodes/reward_ema"] - 2.4) < 1e-6
        # Length EMA: mean = 1.0, ema = 0.2*1.0 + 0.8*0.2 = 0.2 + 0.16 = 0.36
        assert abs(stats["episodes/length_ema"] - 0.36) < 1e-6
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
            # Min/max tracking removed for performance
            assert isinstance(stats["episodes/completed"], int)
            assert isinstance(stats["episodes/total_steps"], int)

            # Verify values are correct (not just type)
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

        # Disabled should be significantly faster (at least 5x)
        speedup = enabled_time / disabled_time
        assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"


def test_rollout_manager_initialization(cpu_manager):
    """Test that RolloutManager can be created with episode length EMA"""
    from madrona_escape_room_learn.actor_critic import RecurrentStateConfig
    from madrona_escape_room_learn.amp import AMPState
    from madrona_escape_room_learn.cfg import SimInterface
    from madrona_escape_room_learn.rollouts import RolloutManager

    mgr = cpu_manager

    # Create minimal simulation interface
    sim = SimInterface(
        step=mgr.step,
        obs=[mgr.self_observation_tensor().to_torch()],
        actions=mgr.action_tensor().to_torch(),
        dones=mgr.done_tensor().to_torch(),
        rewards=mgr.reward_tensor().to_torch(),
        termination_reasons=mgr.termination_reason_tensor().to_torch(),
        manager=mgr,
    )

    dev = torch.device("cpu")
    amp = AMPState(dev, enable_mixed_precision=False)
    recurrent_cfg = RecurrentStateConfig(shapes=[])

    # Create rollout manager with episode length tracking
    rollout_mgr = RolloutManager(
        dev=dev,
        sim=sim,
        steps_per_update=20,
        num_bptt_chunks=2,
        amp=amp,
        recurrent_cfg=recurrent_cfg,
        episode_length_ema_decay=0.95,
    )

    # Verify EMA tracker was created and initialized properly
    assert hasattr(rollout_mgr, "episode_length_ema")
    assert rollout_mgr.episode_length_ema.N.item() == 0
    assert rollout_mgr.episode_length_ema.ema.item() == 0.0

    # Test manual EMA updates to verify it works
    rollout_mgr.episode_length_ema.update(50)
    rollout_mgr.episode_length_ema.update(100)

    assert rollout_mgr.episode_length_ema.N.item() == 2
    ema_value = rollout_mgr.episode_length_ema.ema.item()
    # First update: 0 * 0.95 + 50 * 0.05 = 2.5
    # Second update: 2.5 * 0.95 + 100 * 0.05 = 7.375
    expected = 7.375
    assert abs(ema_value - expected) < 1e-5

    print(f"RolloutManager episode length EMA test passed: {ema_value:.2f}")


# Termination Reason Tracking Tests


class TestTerminationReasonTracking:
    """Test suite for termination reason tracking in EpisodicEMATracker."""

    def test_termination_reason_enum(self):
        """Test that TerminationReason enum has expected values"""
        assert TerminationReason.TIME_LIMIT == 0
        assert TerminationReason.PROGRESS_COMPLETE == 1
        assert TerminationReason.COLLISION_DEATH == 2
        assert len(TerminationReason) == 3

    def test_termination_tracking_basic(self):
        """Test basic termination reason tracking functionality"""
        # Initialize tracker with 4 environments, high alpha for fast response
        tracker = EpisodicEMATracker(num_envs=4, alpha=0.5)

        # Simulate first batch: 2 episodes complete with different termination reasons
        rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
        dones = torch.tensor([True, True, False, False])  # First 2 environments complete
        termination_reasons = torch.tensor(
            [
                TerminationReason.TIME_LIMIT,  # env 0: time limit
                TerminationReason.PROGRESS_COMPLETE,  # env 1: progress complete
                TerminationReason.TIME_LIMIT,  # env 2: not used (not done)
                TerminationReason.COLLISION_DEATH,  # env 3: not used (not done)
            ]
        )

        # Expected batch probabilities: 50% TIME_LIMIT, 50% PROGRESS_COMPLETE, 0% COLLISION_DEATH
        tracker.step_update(rewards, dones, termination_reasons)
        stats = tracker.get_statistics()

        # Verify probabilities are not all zero
        time_limit_prob = stats.get("episodes/termination_time_limit_prob", 0.0)
        progress_prob = stats.get("episodes/termination_progress_complete_prob", 0.0)
        collision_prob = stats.get("episodes/termination_collision_death_prob", 0.0)

        assert time_limit_prob > 0, "TIME_LIMIT probability should be > 0"
        assert progress_prob > 0, "PROGRESS_COMPLETE probability should be > 0"
        assert collision_prob == 0, "COLLISION_DEATH probability should be 0"

    def test_termination_tracking_ema_update(self):
        """Test that EMA updating works correctly across multiple batches"""
        tracker = EpisodicEMATracker(num_envs=4, alpha=0.5)

        # Batch 1: 50% TIME_LIMIT, 50% PROGRESS_COMPLETE
        rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
        dones = torch.tensor([True, True, False, False])
        termination_reasons = torch.tensor(
            [
                TerminationReason.TIME_LIMIT,
                TerminationReason.PROGRESS_COMPLETE,
                TerminationReason.TIME_LIMIT,
                TerminationReason.COLLISION_DEATH,
            ]
        )

        tracker.step_update(rewards, dones, termination_reasons)
        stats1 = tracker.get_statistics()

        # Batch 2: 33% TIME_LIMIT, 0% PROGRESS_COMPLETE, 67% COLLISION_DEATH
        rewards = torch.tensor([0.5, 1.0, 0.2, 2.0])
        dones = torch.tensor([False, True, True, True])
        termination_reasons = torch.tensor(
            [
                TerminationReason.TIME_LIMIT,
                TerminationReason.COLLISION_DEATH,
                TerminationReason.COLLISION_DEATH,
                TerminationReason.TIME_LIMIT,
            ]
        )

        tracker.step_update(rewards, dones, termination_reasons)
        stats2 = tracker.get_statistics()

        # Verify EMA is updating (values should change between batches)
        assert (
            stats1["episodes/termination_time_limit_prob"]
            != stats2["episodes/termination_time_limit_prob"]
        )
        assert (
            stats1["episodes/termination_collision_death_prob"]
            != stats2["episodes/termination_collision_death_prob"]
        )

        # All probabilities should be > 0 after both batches
        assert stats2["episodes/termination_time_limit_prob"] > 0
        assert stats2["episodes/termination_collision_death_prob"] > 0

    def test_termination_tracking_optional_parameter(self):
        """Test that termination_reasons parameter is optional"""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)

        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([True, True])

        # Should work without termination_reasons
        tracker.step_update(rewards, dones)
        stats = tracker.get_statistics()

        # All termination probabilities should be 0 when not provided
        assert stats["episodes/termination_time_limit_prob"] == 0.0
        assert stats["episodes/termination_progress_complete_prob"] == 0.0
        assert stats["episodes/termination_collision_death_prob"] == 0.0

    def test_termination_tracking_no_completed_episodes(self):
        """Test that tracker handles case where no episodes complete"""
        tracker = EpisodicEMATracker(num_envs=3, alpha=0.1)

        rewards = torch.tensor([1.0, 2.0, 0.5])
        dones = torch.tensor([False, False, False])  # No episodes complete
        termination_reasons = torch.tensor(
            [
                TerminationReason.TIME_LIMIT,
                TerminationReason.PROGRESS_COMPLETE,
                TerminationReason.COLLISION_DEATH,
            ]
        )

        tracker.step_update(rewards, dones, termination_reasons)
        stats = tracker.get_statistics()

        # All probabilities should remain 0
        assert stats["episodes/termination_time_limit_prob"] == 0.0
        assert stats["episodes/termination_progress_complete_prob"] == 0.0
        assert stats["episodes/termination_collision_death_prob"] == 0.0

    def test_termination_tracking_statistics_keys(self):
        """Test that all expected statistics keys are present"""
        tracker = EpisodicEMATracker(num_envs=2, alpha=0.1)
        stats = tracker.get_statistics()

        expected_keys = [
            "episodes/reward_ema",
            "episodes/length_ema",
            "episodes/termination_time_limit_prob",
            "episodes/termination_progress_complete_prob",
            "episodes/termination_collision_death_prob",
            "episodes/completed",
            "episodes/total_steps",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing expected key: {key}"


if __name__ == "__main__":
    pytest.main([__file__])

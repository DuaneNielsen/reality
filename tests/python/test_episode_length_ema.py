import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from train_src.madrona_escape_room_learn.moving_avg import EMATracker


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

    assert ema.ema.item() == 0.0
    assert ema.N.item() == 0


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


def test_rollout_manager_initialization(cpu_manager):
    """Test that RolloutManager can be created with episode length EMA"""
    from train_src.madrona_escape_room_learn.actor_critic import RecurrentStateConfig
    from train_src.madrona_escape_room_learn.amp import AMPState
    from train_src.madrona_escape_room_learn.cfg import SimInterface
    from train_src.madrona_escape_room_learn.rollouts import RolloutManager

    mgr = cpu_manager

    # Create minimal simulation interface
    sim = SimInterface(
        step=mgr.step,
        obs=[mgr.self_observation_tensor().to_torch()],
        actions=mgr.action_tensor().to_torch(),
        dones=mgr.done_tensor().to_torch(),
        rewards=mgr.reward_tensor().to_torch(),
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

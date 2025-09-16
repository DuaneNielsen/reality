#!/usr/bin/env python3
"""
Test termination reason tracking in EpisodicEMATracker
"""

import os
import sys

import pytest
import torch

# Add the train_src directory to path so we can import moving_avg
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "train_src"))

from madrona_escape_room_learn.moving_avg import EpisodicEMATracker, TerminationReason


def test_termination_reason_enum():
    """Test that TerminationReason enum has expected values"""
    assert TerminationReason.TIME_LIMIT == 0
    assert TerminationReason.PROGRESS_COMPLETE == 1
    assert TerminationReason.COLLISION_DEATH == 2
    assert len(TerminationReason) == 3


def test_termination_tracking_basic():
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


def test_termination_tracking_ema_update():
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


def test_termination_tracking_optional_parameter():
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


def test_termination_tracking_no_completed_episodes():
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


def test_termination_tracking_statistics_keys():
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

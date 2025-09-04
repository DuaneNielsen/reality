#!/usr/bin/env python3
"""
Test reward system for Madrona Escape Room.
Tests that rewards are only given at episode end based on normalized Y progress.
"""

import numpy as np
import pytest
import torch
from test_helpers import AgentController, ObservationReader, reset_world

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define level with spawn at southernmost center position
TEST_LEVEL_SOUTH_SPAWN = """
################################
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#...............S..............#
################################
"""


@pytest.mark.ascii_level(TEST_LEVEL_SOUTH_SPAWN)
def test_forward_movement_reward(cpu_manager):
    """Test reward for consistent forward movement"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Move forward at moderate speed
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=1.5)

    # Track initial position
    initial_y = observer.get_position(0)[1]
    print(f"Starting Y position: {initial_y:.2f}")

    # Run for 190 steps
    for i in range(190):
        controller.step()

        # Check no reward during episode
        reward = observer.get_reward(0)
        assert reward == 0.0, f"Reward should be 0 during episode, got {reward} at step {i + 1}"

        # Print progress every 50 steps
        if i % 50 == 0:
            pos = observer.get_position(0)
            max_y = observer.get_max_y_progress(0)
            print(f"Step {i}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Max Y progress={max_y:.3f}")

    # Final steps to trigger reward
    for _ in range(10):
        controller.step()

    # Check final state
    final_pos = observer.get_position(0)
    final_reward = observer.get_reward(0)
    max_y_progress = observer.get_max_y_progress(0)

    print("\nFinal state:")
    y_movement = final_pos[1] - initial_y
    print(f"  Position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f} (moved Y by {y_movement:.2f})")
    print(f"  Max Y progress: {max_y_progress:.3f}")
    print(f"  Final reward: {final_reward:.3f}")

    # Verify reward matches progress
    assert final_reward > 0.0, "Should receive positive reward for forward movement"
    assert observer.get_done_flag(0), "Episode should be done"
    assert observer.get_steps_remaining(0) == 0, "Steps should be exhausted"


# Define custom level with walls that limit forward progress
TEST_LEVEL_WITH_WALLS = """
################################
#..............................#
#..............................#
#..............................#
#############........###########
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#...............S..............#
################################
"""


@pytest.mark.ascii_level(TEST_LEVEL_WITH_WALLS)
def test_reward_normalization(cpu_manager):
    """Test that rewards are properly normalized by world length"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0 - manager already has the custom level
    reset_world(mgr, 0)

    # Move forward consistently
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=1.5)

    # Run until episode end
    while observer.get_steps_remaining(0) > 0:
        controller.step()

    final_reward = observer.get_reward(0)
    max_y = observer.get_max_y_progress(0)

    print("Normalization test:")
    print(f"  Max Y (normalized): {max_y:.3f}")
    print(f"  Reward: {final_reward:.3f}")

    # Rewards are normalized to [0, 1] based on Y progress through the world
    assert 0.0 < final_reward <= 1.0, "Reward should be normalized between 0 and 1"

    # The observation maxY and reward should now be consistent after the C++ fix
    assert abs(final_reward - max_y) < 0.01, "Reward and observation maxY should match"

    # With the gap in the wall at row 4, the agent can pass through and reach
    # nearly the end of the level (about 90% progress)
    assert 0.85 < final_reward < 0.95, (
        "Reward should reflect nearly complete progress through the gap"
    )


# NOTE: Removed test_recorded_actions_reward - replaced by comprehensive native recording tests

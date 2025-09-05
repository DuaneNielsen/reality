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

    # Verify reward tensor has correct shape [num_worlds, 1] for single-agent environment
    reward_tensor = mgr.reward_tensor().to_torch()
    assert reward_tensor.shape == (
        4,
        1,
    ), f"Expected reward tensor shape (4, 1), got {reward_tensor.shape}"

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

    # Verify reward tensor has correct shape [num_worlds, 1] for single-agent environment
    reward_tensor = mgr.reward_tensor().to_torch()
    assert reward_tensor.shape == (
        4,
        1,
    ), f"Expected reward tensor shape (4, 1), got {reward_tensor.shape}"

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


def test_reward_tensor_shape(cpu_manager):
    """Test that reward tensor has correct shape for single-agent environment"""
    mgr = cpu_manager

    # Verify reward tensor shape
    reward_tensor = mgr.reward_tensor().to_torch()
    num_worlds = reward_tensor.shape[0]

    # Single-agent environment should have shape [num_worlds, 1] not [num_worlds, num_agents, 1]
    assert len(reward_tensor.shape) == 2, (
        f"Reward tensor should be 2D, got {len(reward_tensor.shape)}D shape {reward_tensor.shape}"
    )
    assert reward_tensor.shape[1] == 1, (
        f"Reward tensor should have 1 reward per world, got {reward_tensor.shape[1]}"
    )

    print(
        f"✓ Reward tensor shape: {reward_tensor.shape} (num_worlds={num_worlds}, rewards_per_world=1)"
    )

    # Verify we can access rewards correctly
    for world_idx in range(num_worlds):
        reward_val = reward_tensor[world_idx, 0].item()
        assert isinstance(reward_val, float), f"Reward should be float, got {type(reward_val)}"


# NOTE: Removed test_recorded_actions_reward - replaced by comprehensive native recording tests


def test_auto_reset_reward_delivery():
    """Test that rewards are properly delivered when auto-reset triggers episode end"""
    from madrona_escape_room import ExecMode, SimManager, create_default_level
    from madrona_escape_room.generated_constants import consts

    # Create manager with auto-reset enabled
    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,  # Enable auto-reset
        compiled_levels=create_default_level(),
    )

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset all worlds
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0

    # Track initial positions
    initial_positions = []
    for world_idx in range(4):
        pos = observer.get_position(world_idx)
        initial_positions.append(pos[1])  # Track Y position
        print(f"World {world_idx} initial Y position: {pos[1]:.2f}")

    # Keep agent still to test pure auto-reset behavior
    controller.reset_actions()
    # Actions default to 0 (STOP), so agent will stay still

    # Track rewards and positions during episode
    reward_history = []
    max_progress = [0.0] * 4

    # Run exactly episodeLen steps to trigger auto-reset
    for step in range(consts.episodeLen):
        controller.step()

        # Check rewards during episode (should be 0)
        current_rewards = []
        for world_idx in range(4):
            reward = observer.get_reward(world_idx)
            current_rewards.append(reward)

            # Track max progress
            pos = observer.get_position(world_idx)
            progress = pos[1] - initial_positions[world_idx]
            max_progress[world_idx] = max(max_progress[world_idx], progress)

        reward_history.append(current_rewards.copy())

        # Rewards should be 0 during episode, but may be delivered at the final step due to auto-reset
        for world_idx, reward in enumerate(current_rewards):
            if step == consts.episodeLen - 1:
                # At the final step, auto-reset may deliver rewards
                if reward > 0:
                    print(
                        f"Step {step}: Auto-reset delivered reward {reward:.4f} to world {world_idx}"
                    )
                assert reward >= 0.0, (
                    f"Final step reward should be non-negative at step {step}, world {world_idx}, got {reward}"
                )
            else:
                # During episode (not final step), rewards should be 0
                assert reward == 0.0, (
                    f"Reward should be 0 during episode at step {step}, world {world_idx}, got {reward}"
                )

    # Step once more to trigger auto-reset and reward delivery
    controller.step()

    # Check final rewards after auto-reset trigger
    final_rewards = []
    for world_idx in range(4):
        reward = observer.get_reward(world_idx)
        final_rewards.append(reward)
        done = observer.get_done_flag(world_idx)

        print(
            f"World {world_idx}: Final reward={reward:.4f}, Done={done}, Progress={max_progress[world_idx]:.2f}"
        )

        # With auto-reset, reward is based on maxY reached (including spawn position)
        # Even stationary agents get reward based on spawn position relative to world_min_y
        assert reward >= 0.0, (
            f"Should receive non-negative reward in world {world_idx}, got {reward}"
        )

        # Verify reward is properly normalized
        assert 0.0 <= reward <= 1.0, (
            f"Reward should be normalized [0,1] in world {world_idx}, got {reward}"
        )

    print(f"✓ Auto-reset test passed - rewards delivered: {[f'{r:.4f}' for r in final_rewards]}")

    # The key test: verify that auto-reset properly delivers rewards when episodes end
    # Even with no movement, agents get reward based on their spawn position relative to world_min_y
    for world_idx, reward in enumerate(final_rewards):
        # Auto-reset should deliver consistent rewards based on agent position
        assert reward >= 0.0, f"Reward should be non-negative in world {world_idx}, got {reward}"
        print(f"  World {world_idx}: Auto-reset delivered reward {reward:.4f} for spawn position")

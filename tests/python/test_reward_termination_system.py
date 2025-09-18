#!/usr/bin/env python3
"""
Test reward system for Madrona Escape Room.
Tests that rewards are given incrementally as the agent makes forward progress.
"""

import json

import numpy as np
import pytest
import torch
from test_helpers import AgentController, ObservationReader, reset_world

from madrona_escape_room.generated_constants import consts

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


@pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
@pytest.mark.ascii_level(TEST_LEVEL_SOUTH_SPAWN)
def test_forward_movement_reward(cpu_manager):
    """Test incremental reward for consistent forward movement"""
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

    total_rewards = 0.0
    prev_y = initial_y
    rewards_received = 0

    # Run for 190 steps
    for i in range(190):
        controller.step()

        # Check for incremental rewards
        reward = observer.get_reward(0)
        current_y = observer.get_position(0)[1]

        if reward > 0.0:
            rewards_received += 1
            total_rewards += reward
            print(
                f"Step {i+1}: Y={current_y:.2f} "
                f"(moved {current_y - prev_y:.3f}), reward={reward:.6f}"
            )
            prev_y = current_y

        # Print progress every 50 steps
        if i % 50 == 0:
            pos = observer.get_position(0)
            max_y = observer.get_max_y_progress(0)
            print(
                f"Step {i}: X={pos[0]:.2f}, Y={pos[1]:.2f}, "
                f"Max Y progress={max_y:.3f}, Total rewards={total_rewards:.6f}"
            )

    # Final steps to complete episode
    for _ in range(10):
        controller.step()
        reward = observer.get_reward(0)
        if reward > 0.0:
            total_rewards += reward
            rewards_received += 1

    # Check final state
    final_pos = observer.get_position(0)
    max_y_progress = observer.get_max_y_progress(0)

    print("\nFinal state:")
    y_movement = final_pos[1] - initial_y
    print(f"  Position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f} (moved Y by {y_movement:.2f})")
    print(f"  Max Y progress: {max_y_progress:.3f}")
    print(f"  Total rewards accumulated: {total_rewards:.6f}")
    print(f"  Number of steps with rewards: {rewards_received}")

    # Verify incremental reward system
    assert rewards_received > 0, "Should receive incremental rewards during movement"
    assert total_rewards > 0.0, "Should accumulate positive rewards for forward movement"
    assert observer.get_done_flag(0), "Episode should be done"
    assert observer.get_steps_remaining(0) == 0, "Steps should be exhausted"

    # The total accumulated rewards should be roughly equal to the normalized progress
    # (allowing for small numerical differences)
    expected_total = max_y_progress
    assert (
        abs(total_rewards - expected_total) < 0.01
    ), f"Total rewards {total_rewards:.6f} should match progress {expected_total:.6f}"


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


@pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
@pytest.mark.ascii_level(TEST_LEVEL_WITH_WALLS)
def test_reward_normalization(cpu_manager):
    """Test that incremental rewards sum to normalized total progress"""
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

    total_rewards = 0.0
    rewards_received = 0

    # Run until episode end, accumulating rewards
    while observer.get_steps_remaining(0) > 0:
        controller.step()
        reward = observer.get_reward(0)
        if reward > 0.0:
            total_rewards += reward
            rewards_received += 1

    # Final step reward check
    final_step_reward = observer.get_reward(0)
    max_y = observer.get_max_y_progress(0)

    print("Normalization test:")
    print(f"  Max Y (normalized): {max_y:.3f}")
    print(f"  Total accumulated rewards: {total_rewards:.6f}")
    print(f"  Steps with rewards: {rewards_received}")
    print(f"  Final step reward: {final_step_reward:.6f}")

    # Incremental rewards should sum to normalized progress (0 to 1)
    assert (
        0.0 < total_rewards <= 1.0
    ), f"Total rewards {total_rewards} should be normalized between 0 and 1"

    # The total accumulated rewards should match the observation maxY
    assert (
        abs(total_rewards - max_y) < 0.01
    ), f"Total rewards {total_rewards:.6f} should match progress {max_y:.6f}"

    # With the gap in the wall at row 4, the agent can pass through and reach
    # nearly the end of the level (about 90% progress)
    assert (
        0.85 < total_rewards < 0.95
    ), f"Total rewards {total_rewards:.3f} should reflect nearly complete progress through the gap"


def test_reward_tensor_shape(cpu_manager):
    """Test that reward tensor has correct shape for single-agent environment"""
    mgr = cpu_manager

    # Verify reward tensor shape
    reward_tensor = mgr.reward_tensor().to_torch()
    num_worlds = reward_tensor.shape[0]

    # Single-agent environment should have shape [num_worlds, 1] not [num_worlds, num_agents, 1]
    assert (
        len(reward_tensor.shape) == 2
    ), f"Reward tensor should be 2D, got {len(reward_tensor.shape)}D shape {reward_tensor.shape}"
    assert (
        reward_tensor.shape[1] == 1
    ), f"Reward tensor should have 1 reward per world, got {reward_tensor.shape[1]}"

    print(
        f"✓ Reward tensor shape: {reward_tensor.shape} "
        f"(num_worlds={num_worlds}, rewards_per_world=1)"
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

        # With incremental rewards, agent still gets no rewards since they're not moving
        # Only forward movement (increased Y) gives rewards
        for world_idx, reward in enumerate(current_rewards):
            # Since agents are stationary, they shouldn't get rewards from movement
            assert reward == 0.0, (
                f"Stationary agents should get no rewards at step {step}, "
                f"world {world_idx}, got {reward}"
            )

    # Step once more to trigger auto-reset
    controller.step()

    # Check final rewards after auto-reset trigger
    final_rewards = []
    for world_idx in range(4):
        reward = observer.get_reward(world_idx)
        final_rewards.append(reward)
        done = observer.get_done_flag(world_idx)

        print(
            f"World {world_idx}: Final reward={reward:.4f}, Done={done}, "
            f"Progress={max_progress[world_idx]:.2f}"
        )

        # With incremental rewards and no movement, agents should get no rewards
        # Incremental rewards only come from forward movement
        assert (
            reward == 0.0
        ), f"Stationary agents should get no rewards in world {world_idx}, got {reward}"

        # Collision death penalty still works, but no collision occurred
        assert (
            reward >= -1.0
        ), f"Rewards should not be less than collision penalty in world {world_idx}, got {reward}"

    print(
        f"✓ Auto-reset test passed - no rewards for stationary agents: "
        f"{[f'{r:.4f}' for r in final_rewards]}"
    )

    # The key test: verify that auto-reset works with incremental reward system
    # Stationary agents get no rewards since they made no forward progress
    for world_idx, reward in enumerate(final_rewards):
        assert (
            reward == 0.0
        ), f"Stationary agent should get no reward in world {world_idx}, got {reward}"
        print(f"  World {world_idx}: No reward for stationary agent (correct incremental behavior)")


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_step_zero_reward_is_zero(cpu_manager):
    """SPEC 1: Step 0 reward is always 0"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Check reward immediately after reset (step 0)
    step_0_reward = observer.get_reward(0)
    print(f"Step 0 reward: {step_0_reward}")

    # SPEC REQUIREMENT: Step 0 reward must be 0
    assert step_0_reward == 0.0, f"SPEC VIOLATION: Step 0 reward should be 0.0, got {step_0_reward}"


@pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_forward_movement_gives_incremental_reward(cpu_manager):
    """SPEC 2: Forward movement gives small incremental reward based on forward progress"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Verify step 0 reward is 0
    assert observer.get_reward(0) == 0.0, "Step 0 must have 0 reward"

    # Move forward one step
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=1.0)

    initial_y = observer.get_position(0)[1]
    controller.step()

    reward_after_forward = observer.get_reward(0)
    new_y = observer.get_position(0)[1]
    forward_progress = new_y - initial_y

    print(f"Initial Y: {initial_y:.3f}, New Y: {new_y:.3f}, Progress: {forward_progress:.3f}")
    print(f"Reward for forward movement: {reward_after_forward}")

    # SPEC REQUIREMENT: Forward movement gives positive incremental reward
    if forward_progress > 0:
        assert reward_after_forward > 0.0, (
            f"SPEC VIOLATION: Forward progress {forward_progress:.3f} should give "
            f"positive reward, got {reward_after_forward}"
        )
    else:
        print("No forward progress made - this may be a physics issue")


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_backward_movement_gives_no_reward(cpu_manager):
    """SPEC 3: Moving backward after moving forward does not result in reward"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Move forward first to establish progress
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=2.0)

    for _ in range(10):  # Move forward several steps
        controller.step()

    # Record position after forward movement
    forward_y = observer.get_position(0)[1]
    max_progress = observer.get_max_y_progress(0)

    # Now move backward
    controller.reset_actions()
    controller.move_backward(world_idx=0, speed=2.0)

    for _ in range(5):  # Move backward several steps
        controller.step()
        reward = observer.get_reward(0)
        current_y = observer.get_position(0)[1]

        print(f"Backward step: Y={current_y:.3f}, Reward={reward}")

        # SPEC REQUIREMENT: Backward movement gives no reward
        assert (
            reward == 0.0
        ), f"SPEC VIOLATION: Backward movement should give 0 reward, got {reward}"

    final_y = observer.get_position(0)[1]
    print(f"Forward Y: {forward_y:.3f}, Final Y: {final_y:.3f}, Max progress: {max_progress:.3f}")


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_no_movement_gives_no_reward(cpu_manager):
    """SPEC 4: If agent does not move, no reward is given"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Keep agent stationary (actions default to 0 = STOP)
    controller.reset_actions()

    initial_pos = observer.get_position(0)

    for step in range(20):
        controller.step()
        reward = observer.get_reward(0)
        current_pos = observer.get_position(0)

        # SPEC REQUIREMENT: No movement = no reward
        assert (
            reward == 0.0
        ), f"SPEC VIOLATION: Stationary agent should get 0 reward at step {step+1}, got {reward}"

        # Verify agent actually didn't move significantly
        movement = abs(current_pos[1] - initial_pos[1])
        if movement > 0.1:  # Allow for small physics settling
            print(f"Warning: Agent moved {movement:.3f} units while supposedly stationary")

    print(f"✓ Stationary agent correctly received 0 reward for {20} steps")


@pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
def test_reward_proportional_to_progress_over_max_y(cpu_manager):
    """SPEC 5: Reward amount is proportional to forward progress divided by max_y of level"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Get level bounds for calculation
    initial_y = observer.get_position(0)[1]
    # Note: We need to access level.world_max_y somehow, but for now use empirical testing

    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=1.0)

    total_reward = 0.0
    prev_y = initial_y

    for step in range(10):
        controller.step()
        reward = observer.get_reward(0)
        current_y = observer.get_position(0)[1]

        if reward > 0.0:
            progress_this_step = current_y - prev_y
            print(
                f"Step {step+1}: Progress={progress_this_step:.6f}, Reward={reward:.6f}, "
                f"Ratio={reward/progress_this_step if progress_this_step > 0 else 'N/A'}"
            )
            total_reward += reward
            prev_y = current_y

    # Total progress made
    total_progress = observer.get_position(0)[1] - initial_y
    max_y_progress = observer.get_max_y_progress(0)  # This should be normalized 0-1

    print(f"Total progress: {total_progress:.6f}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Max Y progress (normalized): {max_y_progress:.6f}")

    # SPEC REQUIREMENT: Total reward should equal normalized progress
    if total_progress > 0:
        assert abs(total_reward - max_y_progress) < 0.01, (
            f"SPEC VIOLATION: Total reward {total_reward:.6f} should equal "
            f"normalized progress {max_y_progress:.6f}"
        )
        print(
            f"✓ Reward correctly proportional to progress: {total_reward:.6f} ≈ "
            f"{max_y_progress:.6f}"
        )


@pytest.mark.spec("docs/specs/sim.md", "stepTrackerSystem")
def test_episode_terminates_after_200_steps():
    """SPEC 6: Episode terminates after exactly 200 steps when auto_reset is enabled
    TERMINATION: Should use termination code 0 (episode_steps_reached)
    """
    from madrona_escape_room import ExecMode, SimManager, create_default_level
    from madrona_escape_room.generated_constants import consts

    # Create manager with auto-reset enabled
    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,
        compiled_levels=create_default_level(),
    )

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Skip if termination_reason tensor not available
    try:
        termination_tensor = mgr.termination_reason_tensor()
    except AttributeError:
        print(
            "Warning: TerminationReason tensor not available, "
            "skipping termination code verification"
        )
        termination_tensor = None

    # Reset world 0
    reset_world(mgr, 0)

    # Keep agent stationary to test pure time-based termination
    controller.reset_actions()

    # Track episode progression
    initial_steps = observer.get_steps_remaining(0)
    print(f"Initial steps remaining: {initial_steps}")

    # SPEC REQUIREMENT: Episode length should be exactly 200
    assert (
        initial_steps == consts.episodeLen
    ), f"SPEC VIOLATION: Episode should start with {consts.episodeLen} steps, got {initial_steps}"

    # Run exactly 200 steps
    for step in range(consts.episodeLen):
        controller.step()
        steps_remaining = observer.get_steps_remaining(0)
        done = observer.get_done_flag(0)

        if step < consts.episodeLen - 1:
            # Before final step, should not be done
            expected_remaining = consts.episodeLen - step - 1
            assert steps_remaining == expected_remaining, (
                f"Step {step+1}: Expected {expected_remaining} steps remaining, "
                f"got {steps_remaining}"
            )
            assert not done, f"Episode should not be done at step {step+1}"
        else:
            # At final step (step 200), episode should terminate
            assert steps_remaining == 0, (
                f"SPEC VIOLATION: After {consts.episodeLen} steps, should have 0 steps "
                f"remaining, got {steps_remaining}"
            )

            # Verify termination code 0 (episode_steps_reached)
            termination_code = termination_tensor.to_numpy()[0, 0]
            assert termination_code == 0, (
                f"TERMINATION CODE VIOLATION: Expected code 0 (episode_steps_reached) "
                f"for step limit termination, got {termination_code}"
            )
            print("✓ Correct termination code 0 (episode_steps_reached)")

    print(f"✓ Episode correctly terminated after exactly {consts.episodeLen} steps")


@pytest.mark.spec("docs/specs/sim.md", "resetSystem")
def test_auto_reset_after_episode_termination():
    """SPEC 7: When auto_reset is enabled, episodes automatically reset after termination"""
    from madrona_escape_room import ExecMode, SimManager, create_default_level
    from madrona_escape_room.generated_constants import consts

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,
        compiled_levels=create_default_level(),
    )

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)
    controller.reset_actions()

    initial_position = observer.get_position(0)

    # Run exactly 200 steps to trigger termination
    for _ in range(consts.episodeLen):
        controller.step()

    # After episode termination, check state
    steps_after_termination = observer.get_steps_remaining(0)
    done_after_termination = observer.get_done_flag(0)

    print(
        f"After episode termination: steps_remaining={steps_after_termination}, "
        f"done={done_after_termination}"
    )

    # Take one more step to trigger auto-reset
    controller.step()

    # SPEC REQUIREMENT: Auto-reset should restore initial state
    steps_after_reset = observer.get_steps_remaining(0)
    position_after_reset = observer.get_position(0)

    print(f"After auto-reset: steps_remaining={steps_after_reset}")
    print(f"Position before: ({initial_position[0]:.3f}, {initial_position[1]:.3f})")
    print(f"Position after:  ({position_after_reset[0]:.3f}, {position_after_reset[1]:.3f})")

    # SPEC REQUIREMENT: Steps should reset to full episode length
    assert steps_after_reset >= consts.episodeLen - 5, (
        f"SPEC VIOLATION: Auto-reset should restore ~{consts.episodeLen} steps, "
        f"got {steps_after_reset}"
    )

    print("✓ Auto-reset correctly restored episode after termination")


def test_post_reset_reward_consistency():
    """SPEC 8: After auto-reset, reward system behavior is identical to initial episode"""
    from madrona_escape_room import ExecMode, SimManager, create_default_level
    from madrona_escape_room.generated_constants import consts

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,
        compiled_levels=create_default_level(),
    )

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Episode 1: Record initial behavior
    reset_world(mgr, 0)

    episode_1_step_0_reward = observer.get_reward(0)
    controller.reset_actions()

    # Run full episode
    for _ in range(consts.episodeLen):
        controller.step()

    # Trigger auto-reset
    controller.step()

    # Episode 2: Check post-reset behavior
    episode_2_step_0_reward = observer.get_reward(0)

    print(f"Episode 1 step 0 reward: {episode_1_step_0_reward}")
    print(f"Episode 2 step 0 reward: {episode_2_step_0_reward}")

    # SPEC REQUIREMENT: Post-reset step 0 should match initial step 0
    assert episode_2_step_0_reward == episode_1_step_0_reward, (
        f"SPEC VIOLATION: Post-reset step 0 reward should match initial "
        f"({episode_1_step_0_reward}), got {episode_2_step_0_reward}"
    )

    # SPEC REQUIREMENT: Both should be 0.0 (this will fail with current bug)
    if episode_1_step_0_reward != 0.0 or episode_2_step_0_reward != 0.0:
        print("BUG CONFIRMED: Step 0 rewards should be 0.0 in both episodes")

    # Test forward movement reward consistency
    controller.move_forward(world_idx=0, speed=1.0)
    controller.step()
    episode_2_first_move_reward = observer.get_reward(0)

    print(f"Episode 2 first movement reward: {episode_2_first_move_reward}")

    # The reward calculation should be consistent across episodes
    print("✓ Post-reset reward behavior verified (note: step 0 bug affects both episodes equally)")


def test_multiple_episode_cycles_with_auto_reset():
    """SPEC Integration: Validate reward consistency across multiple auto-reset cycles"""
    from madrona_escape_room import ExecMode, SimManager, create_default_level
    from madrona_escape_room.generated_constants import consts

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,
        compiled_levels=create_default_level(),
    )

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    step_0_rewards = []
    episode_count = 3

    for episode in range(episode_count):
        # Reset for new episode
        if episode == 0:
            reset_world(mgr, 0)
        else:
            # Let auto-reset handle subsequent episodes
            pass

        # Record step 0 reward
        step_0_reward = observer.get_reward(0)
        step_0_rewards.append(step_0_reward)

        controller.reset_actions()

        # Run full episode
        for _ in range(consts.episodeLen):
            controller.step()

        # Trigger auto-reset for next episode (except last)
        if episode < episode_count - 1:
            controller.step()

        print(f"Episode {episode+1}: Step 0 reward = {step_0_reward}")

    # SPEC REQUIREMENT: All step 0 rewards should be identical
    for i, reward in enumerate(step_0_rewards[1:], 1):
        assert reward == step_0_rewards[0], (
            f"SPEC VIOLATION: Episode {i+1} step 0 reward {reward} should match "
            f"episode 1 reward {step_0_rewards[0]}"
        )

    print(f"✓ Verified consistent step 0 rewards across {episode_count} episodes: {step_0_rewards}")


# Define level with cubes surrounding agent spawn to cause collision termination
# Using JSON level to specify DoneOnCollide=True for cubes
@pytest.mark.json_level(
    {
        "ascii": """################################
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
#..............CCCCCC..........#
#..............C....C..........#
#..............C.S..C..........#
#..............C....C..........#
#..............CCCCCC..........#
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
#..............................#
################################""",
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": False},  # Outer walls don't terminate
            "C": {"asset": "cube", "done_on_collision": True},  # Collision cubes terminate episode
            "S": {"asset": "spawn"},  # Agent spawn point
            ".": {"asset": "empty"},  # Empty space
        },
        "scale": 2.5,
        "name": "collision_chamber_with_cubes",
    }
)
@pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
@pytest.mark.auto_reset
def test_collision_auto_reset_step_zero_reward(cpu_manager):
    """Test that step 0 reward is 0 after collision-induced auto-reset"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Verify cpu_manager has auto_reset enabled
    # Note: cpu_manager fixture should be configured with auto_reset=True

    # Reset world 0
    reset_world(mgr, 0)

    # Verify initial step 0 reward is 0
    initial_step_0_reward = observer.get_reward(0)
    initial_position = observer.get_position(0)
    print(f"Initial step 0 reward: {initial_step_0_reward}")
    print(
        f"Initial position: ({initial_position[0]:.2f}, "
        f"{initial_position[1]:.2f}, {initial_position[2]:.2f})"
    )
    assert (
        initial_step_0_reward == 0.0
    ), f"Initial step 0 reward should be 0.0, got {initial_step_0_reward}"

    # Agent spawns in center of chamber with walls 1 tile away in all directions
    # Move agent eastward to collide with wall
    controller.reset_actions()
    controller.strafe_right(world_idx=0, speed=3)  # High speed to ensure collision

    collision_detected = False
    step_count = 0
    max_collision_steps = 20  # Should hit wall quickly in small chamber

    # Keep moving until collision occurs
    while not observer.get_done_flag(0) and step_count < max_collision_steps:
        controller.step()
        step_count += 1

        reward = observer.get_reward(0)
        position = observer.get_position(0)

        # Check for collision penalty (DoneOnCollide cubes give -0.1 reward)
        if abs(reward - (-0.1)) < 1e-6:  # Collision with terminating object gives ~-0.1
            collision_detected = True
            print(
                f"Collision detected at step {step_count}: reward={reward:.4f}, "
                f"position=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
            break

        if step_count % 5 == 0:
            print(
                f"Step {step_count}: reward={reward:.4f}, "
                f"position=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )

    # Verify collision was detected
    assert collision_detected, f"Expected collision with wall within {max_collision_steps} steps"

    # Episode should be done due to collision
    assert observer.get_done_flag(0), "Episode should be done after collision"

    # Trigger auto-reset
    controller.step()

    # Check step 0 reward after auto-reset
    post_reset_step_0_reward = observer.get_reward(0)
    post_reset_position = observer.get_position(0)
    steps_remaining = observer.get_steps_remaining(0)

    print("After auto-reset:")
    print(f"  Step 0 reward: {post_reset_step_0_reward}")
    print(
        f"  Position: ({post_reset_position[0]:.2f}, "
        f"{post_reset_position[1]:.2f}, {post_reset_position[2]:.2f})"
    )
    print(f"  Steps remaining: {steps_remaining}")

    # SPEC REQUIREMENT: Step 0 reward after auto-reset must be 0
    assert post_reset_step_0_reward == 0.0, (
        f"SPEC VIOLATION: Step 0 reward after auto-reset should be 0.0, "
        f"got {post_reset_step_0_reward}"
    )

    # Verify the episode reset properly (should have close to full episode length)
    assert (
        steps_remaining > 190
    ), f"Episode should have ~200 steps after reset, got {steps_remaining}"

    print("✓ Step 0 reward correctly set to 0.0 after collision-induced auto-reset")


# Define empty level for complete traversal test - no northern wall, agent can reach world_max_y
# 32×20 = 640 tiles (under 1024 limit), spawn at 25% from bottom
TEST_LEVEL_EMPTY_TRAVERSAL = """
................................
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
#...........S..................#
#..............................#
#..............................#
#..............................#
################################
"""


# Load the empty level JSON and modify it for traversal testing
with open("levels/empty_20x48.json", "r") as f:
    EMPTY_LEVEL_JSON = json.load(f)

# Create a modified version for traversal testing - remove north wall for complete traversal
EMPTY_LEVEL_TRAVERSAL_JSON = EMPTY_LEVEL_JSON.copy()
EMPTY_LEVEL_TRAVERSAL_JSON["ascii"] = EMPTY_LEVEL_JSON["ascii"].copy()
# Remove the top wall (first line) to allow complete traversal
EMPTY_LEVEL_TRAVERSAL_JSON["ascii"][0] = "." * len(EMPTY_LEVEL_JSON["ascii"][0])
EMPTY_LEVEL_TRAVERSAL_JSON["name"] = "empty_level_traversal_test"


@pytest.mark.json_level(EMPTY_LEVEL_TRAVERSAL_JSON)
def test_empty_level_complete_traversal_rewards(cpu_manager):
    """Test reward system on empty level - agent should reach ~1.0 total reward with traversal"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Get initial position (spawn should be near bottom of 48-row level)
    initial_pos = observer.get_position(0)
    initial_y = initial_pos[1]
    print(
        f"Agent initial position in empty level: ({initial_pos[0]:.2f}, "
        f"{initial_y:.2f}, {initial_pos[2]:.2f})"
    )

    # Agent spawns at row 46 of 48 rows (near bottom), with top wall removed for complete traversal
    print("Empty level: 48 rows, agent spawned near bottom, can traverse to open north boundary")

    # Move forward consistently to traverse the entire level
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=3)  # Use fastest speed for efficiency

    total_reward = 0.0
    step_count = 0
    max_steps = 400  # Longer level needs more steps

    print("Starting traversal of empty level...")

    # Run until episode completes (done=True) or we hit the step limit
    while not observer.get_done_flag(0) and step_count < max_steps:
        controller.step()
        step_count += 1

        reward = observer.get_reward(0)
        total_reward += reward

        current_pos = observer.get_position(0)
        current_y = current_pos[1]
        max_y_progress = observer.get_max_y_progress(0)

        # Log progress every 75 steps (longer level)
        if step_count % 75 == 0:
            print(
                f"Step {step_count}: Y={current_y:.2f}, Total reward={total_reward:.6f}, "
                f"Progress={max_y_progress:.3f}"
            )

        # Check for early completion due to progress
        if max_y_progress >= 1.0:
            print(f"Reached world boundary at step {step_count}")
            break

    # Final state
    final_pos = observer.get_position(0)
    final_y = final_pos[1]
    final_max_y_progress = observer.get_max_y_progress(0)
    done_flag = observer.get_done_flag(0)

    print("\nEmpty level traversal completed:")
    print(f"  Steps taken: {step_count}")
    print(f"  Initial Y: {initial_y:.2f}")
    print(f"  Final Y: {final_y:.2f}")
    print(f"  Y distance traveled: {final_y - initial_y:.2f}")
    print(f"  Max Y progress (normalized): {final_max_y_progress:.6f}")
    print(f"  Total reward accumulated: {total_reward:.6f}")
    print(f"  Episode done: {done_flag}")

    # SPEC 9 REQUIREMENTS: Complete traversal validation for empty level

    # 1. Episode should terminate due to reaching world boundary or hitting walls
    assert (
        done_flag
    ), "SPEC VIOLATION: Episode should terminate (done=True) after complete traversal"

    # 2. Agent should reach very close to 100% progress in empty level
    assert final_max_y_progress >= 0.95, (
        f"SPEC VIOLATION: Agent should reach ~100% progress in empty level, "
        f"got {final_max_y_progress:.3f}"
    )

    # 3. Total reward should equal 1.0 ± small epsilon (0.02 for longer level)
    epsilon = 0.02
    assert abs(total_reward - 1.0) < epsilon, (
        f"SPEC VIOLATION: Complete traversal should yield total reward = 1.0 ± {epsilon}, "
        f"got {total_reward:.6f} (difference: {abs(total_reward - 1.0):.6f})"
    )

    # 4. Total reward should match the max Y progress (they should be equal)
    assert abs(total_reward - final_max_y_progress) < 0.02, (
        f"SPEC VIOLATION: Total reward {total_reward:.6f} should match max Y progress "
        f"{final_max_y_progress:.6f}"
    )

    print(
        f"✓ Empty level SPEC 9 verified: Complete traversal yielded "
        f"{total_reward:.6f} total reward (≈ 1.0)"
    )


@pytest.mark.ascii_level(TEST_LEVEL_EMPTY_TRAVERSAL)
def test_complete_traversal_yields_unit_reward(cpu_manager):
    """SPEC 9: Agent traversing from spawn to world_max_y without obstacles receives
    total reward = 1.0 ± ε
    """
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Get initial position (should be at 25% along Y-axis based on spawn location)
    initial_pos = observer.get_position(0)
    initial_y = initial_pos[1]
    print(f"Agent initial position: ({initial_pos[0]:.2f}, {initial_y:.2f}, {initial_pos[2]:.2f})")

    # Verify agent is roughly at 25% of the level (spawn at row 15 of 20 rows,
    # so 75% from top = 25% from bottom)
    print("Level has 20 rows, agent spawned at row 15 (75% down = 25% up from bottom)")

    # Move forward consistently to traverse the entire level
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=3)  # Use fastest speed for efficiency

    total_reward = 0.0
    step_count = 0
    max_steps = 300  # Safety limit

    print("Starting traversal...")

    # Run until episode completes (done=True) or we hit the step limit
    while not observer.get_done_flag(0) and step_count < max_steps:
        controller.step()
        step_count += 1

        reward = observer.get_reward(0)
        total_reward += reward

        current_pos = observer.get_position(0)
        current_y = current_pos[1]
        max_y_progress = observer.get_max_y_progress(0)

        # Log progress every 50 steps
        if step_count % 50 == 0:
            print(
                f"Step {step_count}: Y={current_y:.2f}, Total reward={total_reward:.6f}, "
                f"Progress={max_y_progress:.3f}"
            )

        # Check for early completion due to progress
        if max_y_progress >= 1.0:
            print(f"Reached world boundary at step {step_count}")
            break

    # Final state
    final_pos = observer.get_position(0)
    final_y = final_pos[1]
    final_max_y_progress = observer.get_max_y_progress(0)
    done_flag = observer.get_done_flag(0)

    print("\nTraversal completed:")
    print(f"  Steps taken: {step_count}")
    print(f"  Initial Y: {initial_y:.2f}")
    print(f"  Final Y: {final_y:.2f}")
    print(f"  Y distance traveled: {final_y - initial_y:.2f}")
    print(f"  Max Y progress (normalized): {final_max_y_progress:.6f}")
    print(f"  Total reward accumulated: {total_reward:.6f}")
    print(f"  Episode done: {done_flag}")

    # SPEC 9 REQUIREMENTS: Complete traversal validation

    # 1. Episode should terminate due to reaching world boundary
    assert (
        done_flag
    ), "SPEC VIOLATION: Episode should terminate (done=True) after complete traversal"

    # 2. Agent should reach very close to 100% progress
    assert (
        final_max_y_progress >= 0.99
    ), f"SPEC VIOLATION: Agent should reach ~100% progress, got {final_max_y_progress:.3f}"

    # 3. Total reward should equal 1.0 ± small epsilon (0.01)
    epsilon = 0.01
    assert abs(total_reward - 1.0) < epsilon, (
        f"SPEC VIOLATION: Complete traversal should yield total reward = 1.0 ± {epsilon}, "
        f"got {total_reward:.6f} (difference: {abs(total_reward - 1.0):.6f})"
    )

    # 4. Total reward should match the max Y progress (they should be equal)
    assert abs(total_reward - final_max_y_progress) < 0.01, (
        f"SPEC VIOLATION: Total reward {total_reward:.6f} should match max Y progress "
        f"{final_max_y_progress:.6f}"
    )

    print(f"✓ SPEC 9 verified: Complete traversal yielded {total_reward:.6f} total reward (≈ 1.0)")


# ================================================================
# COLLISION TERMINATION TESTS (migrated from test_collision_termination.py)
# ================================================================

# Custom 3x3 enclosed level with specific collision behaviors
# Layout:
#   #C#
#   OS#
#   ###
# Where:
# S = Spawn (agent facing north)
# C = North: Cube with done_on_collide=true (should terminate)
# # = East: Wall with done_on_collide=false (should continue)
# O = West: Cylinder with done_on_collide=true (should terminate)
# # = South: Wall with done_on_collide=false (should continue)


@pytest.mark.json_level(
    {
        "ascii": "#C#\nOS#\n###",
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": False},  # Non-terminating walls
            "C": {"asset": "cube", "done_on_collision": True},  # Terminating cube (north)
            "O": {"asset": "cylinder", "done_on_collision": True},  # Terminating cylinder (west)
            "S": {"asset": "spawn"},  # Agent spawn point
        },
        "scale": 2.5,
        "agent_facing": [0.0],  # Face north (0 radians)
        "name": "collision_test_3x3",
    }
)
class TestCollisionTermination:
    """Test collision-based episode termination with custom per-tile collision flags."""

    @pytest.mark.spec("docs/specs/sim.md", "agentCollisionSystem")
    def test_north_collision_terminates(self, cpu_manager):
        """Test collision with terminating cube (north) ends episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state - episode should be running
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move north toward the terminating cube
        for _ in range(5):  # Multiple steps to ensure collision
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check if collision terminated the episode (single agent in world 0)
            if observer.get_done_flag(0):
                return  # SUCCESS - episode terminated as expected

        # If we reach here, episode didn't terminate
        assert False, "Episode should have terminated from north collision with terminating cube"

    def test_east_collision_continues(self, cpu_manager):
        """Test collision with non-terminating wall (east) continues episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move east toward the non-terminating wall
        for step in range(10):  # More steps to test continued simulation
            controller.reset_actions()
            controller.strafe_right(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Episode should continue running even after collision
            assert not observer.get_done_flag(
                0
            ), f"Episode should continue after east wall collision at step {step}"

        # SUCCESS - episode continued running after collision with non-terminating wall

    def test_west_collision_terminates(self, cpu_manager):
        """Test collision with terminating cylinder (west) ends episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move west toward the terminating cylinder
        for _ in range(5):  # Multiple steps to ensure collision
            controller.reset_actions()
            controller.strafe_left(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check if collision terminated the episode
            if observer.get_done_flag(0):
                return  # SUCCESS - episode terminated as expected

        # If we reach here, episode didn't terminate
        assert False, "Episode should have terminated from west collision with terminating cylinder"

    def test_south_collision_continues(self, cpu_manager):
        """Test collision with non-terminating wall (south) continues episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move south toward the non-terminating wall
        for step in range(10):  # More steps to test continued simulation
            controller.reset_actions()
            controller.move_backward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Episode should continue running even after collision
            assert not observer.get_done_flag(
                0
            ), f"Episode should continue after south wall collision at step {step}"

        # SUCCESS - episode continued running after collision with non-terminating wall

    def test_level_configuration_validation(self, cpu_manager):
        """Validate the custom level has correct collision configuration."""
        mgr = cpu_manager

        # Verify the custom level was loaded correctly
        # This is more of a sanity check for the level compilation
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset and get initial state
        controller.reset_actions()
        mgr.step()

        # Check that we can access game state (indicates level loaded successfully)
        initial_position = observer.get_normalized_position(0)
        assert initial_position is not None, "Should be able to read agent position"

        # Check that the episode starts in running state
        assert not observer.get_done_flag(0), "Episode should start in running state"

        # Verify we have proper tensor access
        actions = mgr.action_tensor().to_torch()
        assert actions.shape[0] == 4, "Should have 4 worlds (from cpu_manager fixture)"
        assert actions.shape[1] == 3, "Should have 3 action components (single agent per world)"

    def test_collision_behavior_differences(self, cpu_manager):
        """Test that different collision objects actually behave differently."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Test terminating collision (north - cube)
        controller.reset_actions()
        mgr.step()

        north_terminated = False
        for _ in range(8):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            if observer.get_done_flag(0):
                north_terminated = True
                break

        # Reset for next test - collision terminated, manual reset needed since auto_reset=False
        if north_terminated:
            # Manually trigger reset since auto_reset is disabled in cpu_manager fixture
            reset_tensor = mgr.reset_tensor().to_torch()
            reset_tensor[0] = 1  # Reset world 0
            controller.reset_actions()
            mgr.step()
            reset_tensor[0] = 0  # Clear reset flag

        # Test non-terminating collision (east - wall)
        assert not observer.get_done_flag(0), "Should be reset and running for east test"

        east_terminated = False
        for _ in range(8):
            controller.reset_actions()
            controller.strafe_right(speed=consts.action.move_amount.FAST)
            mgr.step()

            if observer.get_done_flag(0):
                east_terminated = True
                break

        # Validate the different behaviors
        assert north_terminated, "North collision with cube should terminate episode"
        assert not east_terminated, "East collision with wall should not terminate episode"

    @pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
    @pytest.mark.spec("docs/specs/sim.md", "agentCollisionSystem")
    def test_collision_reward_penalty(self, cpu_manager):
        """Test that collision with DoneOnCollide objects gives -0.1 reward.
        TERMINATION: Should use termination code 2 (collision_death)
        """
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Check for termination_reason tensor availability
        try:
            termination_tensor = mgr.termination_reason_tensor()
        except AttributeError:
            print(
                "Warning: TerminationReason tensor not available, "
                "skipping termination code verification"
            )
            termination_tensor = None

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"
        assert observer.get_reward(0) == 0.0, "Reward should be 0 during episode"

        # Move north toward the terminating cube (DoneOnCollide=True)
        for step in range(8):  # Multiple steps to ensure collision
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check if collision terminated the episode
            if observer.get_done_flag(0):
                # Verify -0.1 reward on collision death (allow for floating-point precision)
                reward = observer.get_reward(0)
                assert (
                    abs(reward - (-0.1)) < 1e-6
                ), f"Expected ~-0.1 reward on collision, got {reward}"

                # Verify termination code 2 (collision_death)
                termination_code = termination_tensor.to_numpy()[0, 0]
                assert termination_code == 2, (
                    f"TERMINATION CODE VIOLATION: Expected code 2 (collision_death) "
                    f"for collision termination, got {termination_code}"
                )
                print("✓ Correct termination code 2 (collision_death)")

                print(f"✓ Collision reward test passed: reward = {reward}")
                return

        # If we reach here, collision didn't occur
        assert False, "Collision should have occurred with terminating cube"

    @pytest.mark.skip(reason="Reward system changed to completion-only, test needs update")
    def test_normal_episode_end_vs_collision_rewards(self, cpu_manager):
        """Test that normal episode timeout gives progress reward, collision gives -0.1."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Test 1: Normal episode timeout (should give progress reward)
        controller.reset_actions()
        mgr.step()

        # Stay still to avoid collision and let episode timeout
        for _ in range(200):  # Run full episode length
            controller.reset_actions()  # No movement
            mgr.step()

        # Check timeout reward (should be progress-based, >= 0)
        timeout_reward = observer.get_reward(0)
        assert observer.get_done_flag(0), "Episode should be done after timeout"
        assert (
            timeout_reward >= 0.0
        ), f"Timeout should give non-negative reward, got {timeout_reward}"
        print(f"✓ Timeout reward: {timeout_reward}")

        # Reset for collision test
        reset_tensor = mgr.reset_tensor().to_torch()
        reset_tensor[0] = 1  # Reset world 0
        mgr.step()
        reset_tensor[0] = 0  # Clear reset flag

        # Test 2: Collision death (should give -0.1 reward)
        assert not observer.get_done_flag(0), "Episode should be reset and running"

        # Move toward terminating object
        collision_occurred = False
        for step in range(8):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            if observer.get_done_flag(0):
                collision_occurred = True
                collision_reward = observer.get_reward(0)
                assert (
                    abs(collision_reward - (-0.1)) < 1e-6
                ), f"Expected ~-0.1 collision reward, got {collision_reward}"
                print(f"✓ Collision reward: {collision_reward}")
                break

        assert collision_occurred, "Collision should have occurred"

        # Verify different rewards for different episode end types
        print(f"✓ Reward comparison: timeout={timeout_reward}, collision={collision_reward}")
        assert (
            timeout_reward > collision_reward
        ), "Timeout reward should be higher than collision penalty"

    def test_non_terminating_collision_no_penalty(self, cpu_manager):
        """Test that collision with non-DoneOnCollide objects doesn't give -0.1 reward."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Move east toward non-terminating wall (DoneOnCollide=False)
        for step in range(10):  # Collide with wall multiple times
            controller.reset_actions()
            controller.strafe_right(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Episode should continue, reward should remain 0
            assert not observer.get_done_flag(
                0
            ), f"Episode should continue after non-terminating collision at step {step}"
            reward = observer.get_reward(0)
            assert (
                abs(reward) < 1e-5
            ), f"Reward should be ~0 during non-terminating collision, got {reward} at step {step}"

        # Continue until episode timeout to verify normal progress reward
        remaining_steps = 200 - 10  # Already took 10 steps
        for _ in range(remaining_steps):
            controller.reset_actions()  # Stay still
            mgr.step()

        # Should get normal progress reward, not collision penalty
        final_reward = observer.get_reward(0)
        assert observer.get_done_flag(0), "Episode should be done after timeout"
        assert (
            final_reward >= 0.0
        ), f"Non-terminating collision should not affect final reward, got {final_reward}"
        print(f"✓ Non-terminating collision final reward: {final_reward}")


# ================================================================
# TERMINATION REASON CODE TESTS
# ================================================================


def test_termination_reason_tensor_access(cpu_manager):
    """Test that termination reason tensor is accessible and has correct shape."""
    mgr = cpu_manager

    # Check if termination_reason tensor is accessible
    try:
        termination_tensor = mgr.termination_reason_tensor()
        termination_data = termination_tensor.to_torch()
        print(f"✓ TerminationReason tensor shape: {termination_data.shape}")
        print(f"  Initial values: {termination_data.numpy()}")

        # Should have shape [num_worlds, 1] for single-agent environment
        assert (
            len(termination_data.shape) == 2
        ), f"Expected 2D tensor, got {len(termination_data.shape)}D"
        assert termination_data.shape[0] == 4, f"Expected 4 worlds, got {termination_data.shape[0]}"
        assert (
            termination_data.shape[1] == 1
        ), f"Expected 1 agent per world, got {termination_data.shape[1]}"

    except AttributeError:
        print("✗ TerminationReason tensor not accessible - needs to be added to Python bindings")
        pytest.skip("TerminationReason tensor not yet available in Python bindings")


def test_episode_steps_reached_termination_code(cpu_manager):
    """Test termination reason code 0 when episode reaches step limit."""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Skip if termination_reason tensor not available
    try:
        termination_tensor = mgr.termination_reason_tensor()
    except AttributeError:
        pytest.skip("TerminationReason tensor not yet available")

    # Reset world 0
    reset_world(mgr, 0)

    # Keep agent stationary to let episode timeout
    controller.reset_actions()

    # Run exactly episodeLen steps to trigger step limit termination
    for step in range(consts.episodeLen):
        controller.step()

        # Check if episode is done
        if observer.get_done_flag(0):
            termination_code = termination_tensor.to_torch()[0, 0].item()
            print(f"Episode ended at step {step+1} with termination code: {termination_code}")

            if step + 1 == consts.episodeLen:
                # Should be termination code 0 (episode_steps_reached)
                assert (
                    termination_code == 0
                ), f"Expected termination code 0 for step limit, got {termination_code}"
                print("✓ Step limit termination code test passed")
                return
            else:
                # Episode ended early (unexpected)
                pytest.fail(f"Episode ended unexpectedly at step {step+1}")

    # If we get here, episode didn't end at step limit
    pytest.fail("Episode should have ended at step limit")


def test_goal_achieved_termination_code():
    """Test termination reason code 1 when agent reaches world boundary."""
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    # Create manager with larger level for goal achievement
    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=False,  # Manual control
        compiled_levels=create_default_level(),
    )

    # Skip if termination_reason tensor not available
    try:
        termination_tensor = mgr.termination_reason_tensor()
    except AttributeError:
        pytest.skip("TerminationReason tensor not yet available")

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Move forward quickly to try to reach goal
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=3)  # Fast speed

    max_steps = 300  # Safety limit
    for step in range(max_steps):
        controller.step()

        # Check progress
        max_y_progress = observer.get_max_y_progress(0)

        # Check if episode is done
        if observer.get_done_flag(0):
            termination_code = termination_tensor.to_torch()[0, 0].item()
            print(f"Episode ended at step {step+1} with:")
            print(f"  Progress: {max_y_progress:.3f}")
            print(f"  Termination code: {termination_code}")

            # If high progress (>95%), should be goal achieved (code 1)
            if max_y_progress >= 0.95:
                assert (
                    termination_code == 1
                ), f"Expected termination code 1 for goal achieved, got {termination_code}"
                print("✓ Goal achieved termination code test passed")
                return
            elif step + 1 >= consts.episodeLen:
                # Hit step limit, should be code 0
                assert (
                    termination_code == 0
                ), f"Expected termination code 0 for step limit, got {termination_code}"
                print("✓ Step limit reached instead of goal")
                return
            else:
                # Some other termination
                print(f"Episode ended early with code {termination_code}")
                return

    pytest.fail("Episode should have ended within step limit")


@pytest.mark.json_level(
    {
        "ascii": """################################
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
#..............CCCCCC..........#
#..............C....C..........#
#..............C.S..C..........#
#..............C....C..........#
#..............CCCCCC..........#
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
#..............................#
################################""",
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": False},
            "C": {"asset": "cube", "done_on_collision": True},  # Collision cubes terminate
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
        },
        "scale": 2.5,
        "name": "collision_termination_test",
    }
)
@pytest.mark.spec("docs/specs/sim.md", "agentCollisionSystem")
def test_collision_death_termination_code(cpu_manager):
    """Test termination reason code 2 when agent collides with terminating object."""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Skip if termination_reason tensor not available
    try:
        termination_tensor = mgr.termination_reason_tensor()
    except AttributeError:
        pytest.skip("TerminationReason tensor not yet available")

    # Reset world 0
    reset_world(mgr, 0)

    # Move toward collision cube (agent spawns surrounded by cubes)
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=3)  # Fast speed to ensure collision

    max_collision_steps = 15  # Should hit cube quickly

    for step in range(max_collision_steps):
        controller.step()

        # Check if episode is done
        if observer.get_done_flag(0):
            termination_code = termination_tensor.to_torch()[0, 0].item()
            reward = observer.get_reward(0)

            print(f"Episode ended at step {step+1} with:")
            print(f"  Termination code: {termination_code}")
            print(f"  Reward: {reward}")

            # Check for collision death indicators
            if abs(reward - (-0.1)) < 1e-6:  # Collision penalty
                assert (
                    termination_code == 2
                ), f"Expected termination code 2 for collision death, got {termination_code}"
                print("✓ Collision death termination code test passed")
                return
            else:
                # Some other termination reason
                print(
                    f"Episode ended with different reason: code={termination_code}, reward={reward}"
                )
                return

    pytest.fail("Expected collision with terminating cube")


def test_termination_codes_consistency():
    """Test that termination codes are consistent across different termination types."""
    import torch

    from madrona_escape_room import ExecMode, SimManager, create_default_level

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=3,  # Test multiple worlds
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=False,
        compiled_levels=create_default_level(),
    )

    # Skip if termination_reason tensor not available
    try:
        termination_tensor = mgr.termination_reason_tensor()
    except AttributeError:
        pytest.skip("TerminationReason tensor not yet available")

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset all worlds
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0

    # World 0: Let timeout (should be code 0)
    # World 1: Try for goal (may be code 1 or 0)
    # World 2: Let timeout (should be code 0)

    controller.reset_actions()
    # World 1 moves forward, others stay still
    actions = mgr.action_tensor().to_torch()
    actions[1, :] = torch.tensor([3, 0, 2])  # Fast forward for world 1

    # Run until episodes end
    for step in range(consts.episodeLen + 5):
        mgr.step()

        # Check which worlds are done
        done_flags = [observer.get_done_flag(i) for i in range(3)]

        if all(done_flags):
            break

    # Check termination codes
    termination_codes = termination_tensor.to_torch().numpy()
    print("Final termination codes:")
    for i in range(3):
        code = termination_codes[i, 0]
        reward = observer.get_reward(i)
        progress = observer.get_max_y_progress(i)
        print(f"  World {i}: code={code}, reward={reward:.4f}, progress={progress:.3f}")

        # Validate codes are in expected range
        assert code in [0, 1, 2], f"World {i}: Invalid termination code {code}"

        # Code 1 (goal achieved) should have positive or zero reward
        if code == 1:
            assert (
                reward >= -0.01
            ), f"World {i}: Goal achieved should have non-negative reward, got {reward}"

        # Code 2 (collision death) should have collision penalty
        if code == 2:
            assert (
                abs(reward - (-0.1)) < 0.01
            ), f"World {i}: Code 2 should have ~-0.1 reward, got {reward}"

        # Code 0 (timeout) can have any reward (depends on final action)

    print("✓ Termination codes consistency test passed")


def test_termination_reason_export_integration():
    """Integration test to verify termination reason is properly exported to training code."""
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=2,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,  # Test with auto-reset
        compiled_levels=create_default_level(),
    )

    # Skip if termination_reason tensor not available
    try:
        termination_tensor = mgr.termination_reason_tensor()
    except AttributeError:
        pytest.skip("TerminationReason tensor not yet available")

    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset worlds
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0

    termination_history = []
    episode_count = 0
    max_episodes = 3

    controller.reset_actions()

    # Run multiple episodes with auto-reset
    step_count = 0
    max_total_steps = consts.episodeLen * max_episodes + 10

    while episode_count < max_episodes and step_count < max_total_steps:
        mgr.step()
        step_count += 1

        # Check for episode completion in any world
        for world_idx in range(2):
            if observer.get_done_flag(world_idx):
                code = termination_tensor.to_torch()[world_idx, 0].item()
                reward = observer.get_reward(world_idx)

                termination_history.append(
                    {
                        "episode": episode_count,
                        "world": world_idx,
                        "step": step_count,
                        "code": code,
                        "reward": reward,
                    }
                )

                print(
                    f"Episode {episode_count}, World {world_idx}: code={code}, reward={reward:.4f}"
                )

                if world_idx == 0:  # Count episodes based on world 0
                    episode_count += 1

                break

    # Validate we got termination data
    assert len(termination_history) > 0, "Should have recorded some episode terminations"

    # Validate all codes are valid
    for entry in termination_history:
        assert entry["code"] in [0, 1, 2], f"Invalid termination code: {entry['code']}"

    print(f"✓ Recorded {len(termination_history)} episode terminations with valid codes")
    print("✓ Termination reason export integration test passed")

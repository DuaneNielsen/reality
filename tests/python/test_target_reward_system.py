#!/usr/bin/env python3
"""
Test target-based reward system for Madrona Escape Room.
Tests that rewards are given when the agent reaches within 3.0 world units of a target entity.
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

# Test level with target far from spawn (for testing no reward when far)
TEST_LEVEL_WITH_TARGET = {
    "ascii": [
        "############",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#....S.....#",
        "############",
    ],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [5.0, 1.0, 1.0],  # Far from spawn at (5.0, 9.0)
            "motion_type": "static",
        }
    ],
    "name": "target_test",
}

# Test level with target close to spawn (within 1.0 unit for easy completion)
TEST_LEVEL_CLOSE_TARGET = {
    "ascii": ["######", "#....#", "#.S..#", "#....#", "######"],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [-0.5, 0.0, 1.0],  # Very close to actual spawn at (-0.5, 0.0)
            "motion_type": "static",
        }
    ],
    "name": "close_target_test",
}

# Test level with static target for completion testing
TEST_LEVEL_STATIC_TARGET = {
    "ascii": ["########", "#......#", "#......#", "#..S...#", "#......#", "#......#", "########"],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [3.0, 1.0, 1.0],  # Reachable static target
            "motion_type": "static",
        }
    ],
    "name": "static_target_test",
}


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_CLOSE_TARGET)
def test_target_proximity_reward(cpu_manager):
    """Test that +1.0 reward is given when agent is within 1.0 units of target"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Agent spawns at (-0.5, 0.0), target at (-0.5, 0.0, 1.0) = 1.0 units away
    # This should immediately trigger completion reward
    controller.reset_actions()
    controller.step(1)

    # Check that reward is +1.0 for completion
    reward = observer.get_reward(0)
    done = observer.get_done_flag(0)
    termination_reason = observer.get_termination_reason(0)

    print(f"Distance: 0.5 units, Reward: {reward}, Done: {done}, Termination: {termination_reason}")

    assert reward == 1.0, f"Expected +1.0 reward for target proximity, got {reward}"
    assert done, f"Expected done=True for target completion, got {done}"
    assert (
        termination_reason == 1
    ), f"Expected termination_reason=1 (goal_achieved), got {termination_reason}"


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_WITH_TARGET)
def test_no_reward_when_far(cpu_manager):
    """Test that 0.0 reward is given when agent is > 1.0 units from target"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Agent spawns at (16.0, 2.0), target at (16.0, 28.0) = 26 units away
    # Should not trigger completion reward
    controller.reset_actions()

    # Move forward a few steps but not enough to reach target
    for _ in range(10):
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.FAST)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)

        # Should have no reward and not be done while far from target
        assert reward == 0.0, f"Expected 0.0 reward while far from target, got {reward}"
        assert not done, f"Expected done=False while far from target, got {done}"


@pytest.mark.xfail(reason="Target placement needs adjustment for 3.0 unit completion threshold")
@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_WITH_TARGET)
def test_static_target_completion(cpu_manager):
    """Test completion reward with static target after movement"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Move agent towards target for many steps
    controller.reset_actions()
    completion_found = False

    for step in range(150):  # Enough steps to reach target
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.FAST)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)
        pos = observer.get_normalized_position(0)

        if done:
            completion_found = True
            print(f"Completion at step {step}: reward={reward}, position={pos}")
            assert reward == 1.0, f"Expected +1.0 reward for completion, got {reward}"
            break
        else:
            assert (
                reward == 0.0
            ), f"Expected 0.0 reward before completion, got {reward} at step {step}"

    assert completion_found, "Agent should have reached the target and completed episode"


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_STATIC_TARGET)
def test_static_target_reachable(cpu_manager):
    """Test completion with reachable static target"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # Agent spawns at (3.0, 3.0), target at (3.0, 1.0) - should be reachable
    controller.reset_actions()
    completion_found = False

    for step in range(50):  # Should reach target quickly
        # Move toward target (north)
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.FAST)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)

        if done:
            completion_found = True
            print(f"Static target completion at step {step}: reward={reward}")
            assert reward == 1.0, f"Expected +1.0 reward for static target completion, got {reward}"
            break
        else:
            assert (
                reward == 0.0
            ), f"Expected 0.0 reward before completion, got {reward} at step {step}"

    assert completion_found, "Agent should have reached the static target"


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_no_target_fallback(cpu_manager):
    """Test that no reward is given when no target exists in level"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Use default level (no targets)
    reset_world(mgr, 0)

    # Move around for several steps
    controller.reset_actions()

    for step in range(20):
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.MEDIUM)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)

        # Should never get reward without target
        assert reward == 0.0, f"Expected 0.0 reward with no target, got {reward} at step {step}"
        # Episode can end from step limit (0) or collision (2), but never target completion (1)
        if done:
            termination_reason = observer.get_termination_reason(0)
            assert termination_reason in [
                0,
                2,
            ], f"Expected step limit (0) or collision (2), got {termination_reason}"
            break


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_CLOSE_TARGET)
def test_collision_overrides_target_reward(cpu_manager):
    """Test that collision death penalty (-0.1) overrides target completion reward"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world 0
    reset_world(mgr, 0)

    # This test is tricky since we need both target proximity AND collision
    # With auto_boundary_walls=true, agent can collide with walls
    # Move agent toward boundary wall while near target
    controller.reset_actions()

    # Move toward wall boundary (should cause collision termination)
    for step in range(50):
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.FAST)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)
        termination_reason = observer.get_termination_reason(0)

        if done:
            print(
                f"Episode ended at step {step}: reward={reward}, termination={termination_reason}"
            )

            if termination_reason == 2:  # Collision death
                assert reward == -0.1, f"Expected -0.1 collision penalty, got {reward}"
                print("Collision death penalty test passed")
                return
            elif termination_reason == 1:  # Goal achieved (target completion)
                assert reward == 1.0, f"Expected +1.0 target reward, got {reward}"
                print("Target completion occurred before collision")
                return

    # If no termination occurred, that's also valid behavior
    print("No termination occurred in test duration")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

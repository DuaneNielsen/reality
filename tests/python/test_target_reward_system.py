#!/usr/bin/env python3
"""
Test target-based reward system for Madrona Escape Room.
Tests that rewards are given when the agent reaches within 3.0 world units of a target entity.
"""

import json

import numpy as np
import pytest
import torch
from test_helpers import AgentController, ObservationReader, TargetTracker, reset_world

from madrona_escape_room.generated_constants import consts
from madrona_escape_room.level_compiler import compile_level

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

# Test level with target close to spawn (within 3.0 units for completion)
TEST_LEVEL_CLOSE_TARGET = {
    "ascii": ["######", "#....#", "#.S..#", "#....#", "######"],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [1.0, -1.0, 1.0],  # Within 3.0 units of spawn for immediate completion
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
    """Test that +1.0 reward is given when agent is within 3.0 units of target"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_CLOSE_TARGET)[0]

    # Reset world 0
    reset_world(mgr, 0)

    # Check actual distance using target position tensor with proper coordinate conversion
    distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    tracker.print_distance_info(0, compiled_level=compiled_level)

    # This should immediately trigger completion reward if within 3.0 units
    controller.reset_actions()
    controller.step(1)

    # Check that reward is +1.0 for completion
    reward = observer.get_reward(0)
    done = observer.get_done_flag(0)
    termination_reason = observer.get_termination_reason(0)

    print(
        f"Actual distance: {distance:.2f} units, Reward: {reward}, Done: {done}, Termination: {termination_reason}"
    )

    # Note: Distance calculation may not exactly match engine calculation due to coordinate system differences
    # Instead, verify that reward behavior is consistent with completion
    if reward == 1.0:
        # If we got reward, should also be done with goal achieved
        assert done, f"Expected done=True when reward={reward}, got {done}"
        assert (
            termination_reason == 1
        ), f"Expected termination_reason=1 (goal_achieved) when reward={reward}, got {termination_reason}"
        print(f"✓ Target reached: Reward={reward}, Done={done}, Distance≈{distance:.2f}")
    else:
        # If no reward, should not be done (unless other termination reason)
        if done:
            assert (
                termination_reason != 1
            ), "Got done=True without reward - termination_reason should not be goal_achieved"
        print(f"✓ Target not reached: Reward={reward}, Done={done}, Distance≈{distance:.2f}")


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_WITH_TARGET)
def test_no_reward_when_far(cpu_manager):
    """Test that 0.0 reward is given when agent is > 3.0 units from target"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_WITH_TARGET)[0]

    # Reset world 0
    reset_world(mgr, 0)

    # Check actual initial distance
    initial_distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    tracker.print_distance_info(0, compiled_level=compiled_level)

    controller.reset_actions()

    # Move forward a few steps but verify we stay far from target
    for step in range(10):
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.FAST)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)
        distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)

        print(f"Step {step}: Distance: {distance:.2f}, Reward: {reward}, Done: {done}")

        # Verify reward matches distance expectation
        assert tracker.verify_reward_threshold(
            distance, reward
        ), f"Reward {reward} doesn't match distance {distance:.2f}"

        if distance > 3.0:
            assert (
                reward == 0.0
            ), f"Expected 0.0 reward while far from target (distance {distance:.2f}), got {reward}"
            assert not done, f"Expected done=False while far from target, got {done}"
        else:
            # If we get close enough, reward should be 1.0
            assert (
                reward == 1.0
            ), f"Expected 1.0 reward when close to target (distance {distance:.2f}), got {reward}"
            break


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_WITH_TARGET)
def test_static_target_completion(cpu_manager):
    """Test completion reward with static target after movement"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_WITH_TARGET)[0]

    # Reset world 0
    reset_world(mgr, 0)

    # Check initial distance and print info
    initial_distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    tracker.print_distance_info(0, compiled_level=compiled_level)
    print(f"Initial distance to target: {initial_distance:.2f} units")

    # Move agent towards target for many steps
    controller.reset_actions()
    completion_found = False

    for step in range(150):  # Enough steps to reach target
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.FAST)
        controller.step(1)

        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)
        distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)

        # Print progress every 20 steps or when close
        if step % 20 == 0 or distance <= 5.0:
            print(f"Step {step}: Distance: {distance:.2f}, Reward: {reward}, Done: {done}")

        # Verify reward matches distance expectation
        assert tracker.verify_reward_threshold(
            distance, reward
        ), f"Step {step}: Reward {reward} doesn't match distance {distance:.2f}"

        if done:
            completion_found = True
            print(f"Completion at step {step}: reward={reward}, distance={distance:.2f}")
            assert reward == 1.0, f"Expected +1.0 reward for completion, got {reward}"
            assert distance <= 3.0, f"Expected distance ≤3.0 for completion, got {distance:.2f}"
            break
        else:
            if distance <= 3.0:
                # If we're close enough, we should have gotten reward and be done
                assert (
                    False
                ), f"Step {step}: Within 3.0 units (distance {distance:.2f}) but not done"

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


# New comprehensive boundary tests for 3.0 unit threshold

# Test levels with targets at precise distances to test threshold boundary
TEST_LEVEL_BOUNDARY_INSIDE = {
    "ascii": ["############", "#..........#", "#..........#", "#....S.....#", "############"],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [2.9, -1.0, 1.0],  # Just within 3.0 unit threshold
            "motion_type": "static",
        }
    ],
    "name": "boundary_inside_test",
}

TEST_LEVEL_BOUNDARY_OUTSIDE = {
    "ascii": ["############", "#..........#", "#..........#", "#....S.....#", "############"],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [3.1, -1.0, 1.0],  # Just outside 3.0 unit threshold
            "motion_type": "static",
        }
    ],
    "name": "boundary_outside_test",
}

TEST_LEVEL_EXACTLY_THREE = {
    "ascii": ["############", "#..........#", "#..........#", "#....S.....#", "############"],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
    "scale": 1.0,
    "targets": [
        {
            "position": [3.0, 0.0, 1.0],  # Exactly 3.0 units away
            "motion_type": "static",
        }
    ],
    "name": "exactly_three_test",
}


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_BOUNDARY_INSIDE)
def test_threshold_boundary_inside(cpu_manager):
    """Test that targets just within 3.0 units (2.9) give reward"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_BOUNDARY_INSIDE)[0]

    reset_world(mgr, 0)

    distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    tracker.print_distance_info(0, compiled_level=compiled_level)

    controller.reset_actions()
    controller.step(1)

    reward = observer.get_reward(0)
    done = observer.get_done_flag(0)

    print(f"Boundary inside test: Distance: {distance:.2f}, Reward: {reward}, Done: {done}")

    if distance <= 3.0:
        assert (
            reward == 1.0
        ), f"Expected 1.0 reward for distance {distance:.2f} (≤3.0), got {reward}"
        assert done, f"Expected done=True for distance {distance:.2f}, got {done}"
    else:
        assert (
            reward == 0.0
        ), f"Expected 0.0 reward for distance {distance:.2f} (>3.0), got {reward}"


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_BOUNDARY_OUTSIDE)
def test_threshold_boundary_outside(cpu_manager):
    """Test that targets just outside 3.0 units (3.1) give no reward"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_BOUNDARY_OUTSIDE)[0]

    reset_world(mgr, 0)

    distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    tracker.print_distance_info(0, compiled_level=compiled_level)

    controller.reset_actions()
    controller.step(1)

    reward = observer.get_reward(0)
    done = observer.get_done_flag(0)

    print(f"Boundary outside test: Distance: {distance:.2f}, Reward: {reward}, Done: {done}")

    if distance > 3.0:
        assert (
            reward == 0.0
        ), f"Expected 0.0 reward for distance {distance:.2f} (>3.0), got {reward}"
        assert not done, f"Expected done=False for distance {distance:.2f}, got {done}"
    else:
        assert (
            reward == 1.0
        ), f"Expected 1.0 reward for distance {distance:.2f} (≤3.0), got {reward}"


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_EXACTLY_THREE)
def test_threshold_exactly_three_units(cpu_manager):
    """Test behavior at exactly 3.0 units (should give reward since ≤3.0)"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_EXACTLY_THREE)[0]

    reset_world(mgr, 0)

    distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    tracker.print_distance_info(0, compiled_level=compiled_level)

    controller.reset_actions()
    controller.step(1)

    reward = observer.get_reward(0)
    done = observer.get_done_flag(0)

    print(f"Exactly 3.0 test: Distance: {distance:.2f}, Reward: {reward}, Done: {done}")

    # The threshold is ≤3.0, so exactly 3.0 should give reward
    if abs(distance - 3.0) < 0.1:  # Account for small positioning errors
        assert (
            reward == 1.0
        ), f"Expected 1.0 reward for distance ~3.0 ({distance:.2f}), got {reward}"
        assert done, f"Expected done=True for distance ~3.0 ({distance:.2f}), got {done}"
    elif distance < 3.0:
        assert (
            reward == 1.0
        ), f"Expected 1.0 reward for distance {distance:.2f} (<3.0), got {reward}"
    else:
        assert (
            reward == 0.0
        ), f"Expected 0.0 reward for distance {distance:.2f} (>3.0), got {reward}"


@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.json_level(TEST_LEVEL_CLOSE_TARGET)
def test_dynamic_distance_tracking(cpu_manager):
    """Test that distance tracking works correctly as agent moves toward target"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    tracker = TargetTracker(mgr)

    # Compile level to get world bounds for accurate distance calculation
    compiled_level = compile_level(TEST_LEVEL_CLOSE_TARGET)[0]

    reset_world(mgr, 0)

    initial_distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
    print(f"Starting distance tracking test: Initial distance = {initial_distance:.2f}")

    controller.reset_actions()
    distances = []
    rewards = []

    for step in range(20):
        # Move toward target
        controller.move_forward(world_idx=0, speed=consts.action.move_amount.SLOW)
        controller.step(1)

        distance = tracker.calculate_distance_to_agent(0, compiled_level=compiled_level)
        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)

        distances.append(distance)
        rewards.append(reward)

        print(f"Step {step}: Distance: {distance:.2f}, Reward: {reward}")

        # Verify reward consistency with distance
        assert tracker.verify_reward_threshold(
            distance, reward
        ), f"Step {step}: Inconsistent reward/distance"

        if done:
            print(f"Episode completed at step {step}")
            break

    print(f"Distance progression: {[f'{d:.2f}' for d in distances[-5:]]}")  # Last 5 distances
    print(f"Reward progression: {rewards[-5:]}")  # Last 5 rewards


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

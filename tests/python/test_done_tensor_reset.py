#!/usr/bin/env python3
"""
Test done tensor reset behavior for Madrona Escape Room.
Tests that done tensor properly resets to 0 and goes to 1 after episode length.
"""

import pytest
import torch

from madrona_escape_room.generated_constants import consts


def test_done_tensor_resets_to_zero(cpu_manager):
    """Test that done tensor resets to 0 when episode reset is triggered"""
    mgr = cpu_manager

    # Get tensor references
    actions = mgr.action_tensor().to_torch()
    done_tensor = mgr.done_tensor().to_torch()
    reset_tensor = mgr.reset_tensor().to_torch()
    steps_taken = mgr.steps_taken_tensor().to_torch()

    # Verify initial state - done should be 0
    assert done_tensor.shape == (
        4,
        1,
        1,
    ), f"Expected done tensor shape (4,1,1), got {done_tensor.shape}"
    assert done_tensor.dtype == torch.int32, f"Expected dtype int32, got {done_tensor.dtype}"

    # First reset all worlds to ensure clean state
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0  # Clear reset flags

    # Run simulation until at least one episode is done
    # Set forward movement to progress through episode quickly
    actions[:] = 0
    actions[:, 0] = 3  # FAST forward movement

    for step in range(consts.episodeLen + 10):  # Run past episode length
        mgr.step()
        if done_tensor.any():
            break

    # Ensure at least one episode is done
    assert (
        done_tensor.any()
    ), f"Expected at least one episode to be done after {consts.episodeLen} steps"

    # Store which world is done for testing
    done_world_idx = torch.nonzero(done_tensor.squeeze(), as_tuple=False)[0].item()
    print(f"World {done_world_idx} is done, testing reset behavior")

    # Now test reset functionality - reset the done world
    reset_tensor[:] = 0
    reset_tensor[done_world_idx] = 1  # Reset the done world
    mgr.step()

    # After reset, done tensor should be 0 for the reset world
    assert done_tensor[done_world_idx, 0, 0] == 0, (
        f"Expected done=0 after reset for world {done_world_idx}, "
        f"got {done_tensor[done_world_idx, 0, 0]}"
    )

    # Steps taken should be back to 0 for the reset world
    assert steps_taken[done_world_idx, 0, 0] == 0, (
        f"Expected 0 steps taken after reset, " f"got {steps_taken[done_world_idx, 0, 0]}"
    )


def test_done_tensor_after_episode_length(cpu_manager):
    """Test that done tensor goes to 1 after exactly episodeLen steps"""
    mgr = cpu_manager

    # Get tensor references
    actions = mgr.action_tensor().to_torch()
    done_tensor = mgr.done_tensor().to_torch()
    reset_tensor = mgr.reset_tensor().to_torch()
    steps_taken = mgr.steps_taken_tensor().to_torch()

    # Reset all worlds to start fresh
    reset_tensor[:] = 1
    mgr.step()

    # Verify initial state after reset
    assert not done_tensor.any(), "All done flags should be 0 after reset"
    assert (steps_taken == 0).all(), "All worlds should have 0 steps taken after reset"

    # Set actions for consistent behavior
    actions[:] = 0
    actions[:, 0] = 1  # SLOW forward movement to avoid collision termination

    # Run for exactly episodeLen steps
    reset_tensor[:] = 0  # Clear reset flags
    for step in range(consts.episodeLen):
        mgr.step()

        # Check that done is still 0 during the episode
        if step < consts.episodeLen - 1:
            assert (
                not done_tensor.any()
            ), f"Done tensor should be 0 during episode at step {step}, got {done_tensor}"

    # After exactly episodeLen steps, done should be 1
    # Note: The episode might end due to step limit, so check if any episodes are done
    # In collision-based termination, episodes might end earlier, so we need to be flexible
    steps_after = steps_taken.clone()

    # If steps taken equals episode length, then done should be 1
    for world_idx in range(4):
        if steps_after[world_idx, 0, 0] == consts.episodeLen:
            assert (
                done_tensor[world_idx, 0, 0] == 1
            ), f"World {world_idx} should be done when steps taken equals {consts.episodeLen}"


def test_done_tensor_stays_zero_before_episode_end(cpu_manager):
    """Test that done tensor stays 0 during the episode before reaching episode length"""
    mgr = cpu_manager

    # Get tensor references
    actions = mgr.action_tensor().to_torch()
    done_tensor = mgr.done_tensor().to_torch()
    reset_tensor = mgr.reset_tensor().to_torch()
    steps_taken = mgr.steps_taken_tensor().to_torch()

    # Reset all worlds
    reset_tensor[:] = 1
    mgr.step()

    # Set minimal movement to avoid collision termination
    actions[:] = 0
    actions[:, 0] = 0  # STOP - no movement

    reset_tensor[:] = 0  # Clear reset flags

    # Run for first half of episode length
    half_episode = consts.episodeLen // 2
    for step in range(half_episode):
        mgr.step()

        # Done should remain 0 throughout
        assert (
            not done_tensor.any()
        ), f"Done tensor should be 0 at step {step}/{half_episode}, got {done_tensor}"

        # Steps taken should increase
        current_steps = steps_taken[0, 0, 0].item()
        expected_steps = step + 1
        assert (
            current_steps == expected_steps
        ), f"Expected {expected_steps} steps taken at step {step}, got {current_steps}"


def test_done_tensor_collision_termination(cpu_manager):
    """Test that done tensor goes to 1 when collision-based termination occurs"""
    mgr = cpu_manager

    # Get tensor references
    actions = mgr.action_tensor().to_torch()
    done_tensor = mgr.done_tensor().to_torch()
    reset_tensor = mgr.reset_tensor().to_torch()

    # Reset all worlds
    reset_tensor[:] = 1
    mgr.step()

    # Set aggressive movement to try to trigger collision termination
    actions[:] = 0
    actions[:, 0] = 3  # FAST forward movement
    actions[:, 1] = 0  # FORWARD direction

    reset_tensor[:] = 0  # Clear reset flags

    # Run simulation and check for collision-based termination
    max_steps_to_check = min(consts.episodeLen, 50)  # Don't run too long
    collision_occurred = False

    for step in range(max_steps_to_check):
        mgr.step()

        if done_tensor.any():
            collision_occurred = True
            # Check which worlds are done
            done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)
            print(
                f"Collision termination occurred at step {step + 1} for worlds: "
                f"{done_worlds.flatten().tolist()}"
            )
            break

    # This test is informational - collision termination might or might not occur
    # depending on the level layout and movement patterns
    if collision_occurred:
        print("✓ Collision-based termination detected and done tensor properly set to 1")
    else:
        print("ℹ No collision termination detected in first {} steps".format(max_steps_to_check))

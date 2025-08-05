#!/usr/bin/env python3
"""
Test reward system for Madrona Escape Room.
Tests that rewards are only given at episode end based on normalized Y progress.
"""

import numpy as np
import pytest
import torch
from test_helpers import AgentController, ObservationReader, reset_world, reset_all_worlds

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def test_forward_movement_reward(test_manager):
    """Test reward for consistent forward movement"""
    mgr = test_manager
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
        assert reward == 0.0, f"Reward should be 0 during episode, got {reward} at step {i+1}"
        
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
    
    print(f"\nFinal state:")
    print(f"  Position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f} (moved Y by {final_pos[1] - initial_y:.2f})")
    print(f"  Max Y progress: {max_y_progress:.3f}")
    print(f"  Final reward: {final_reward:.3f}")
    
    # Verify reward matches progress
    assert final_reward > 0.0, "Should receive positive reward for forward movement"
    assert observer.get_done_flag(0), "Episode should be done"
    assert observer.get_steps_remaining(0) == 0, "Steps should be exhausted"


def test_reward_normalization(test_manager):
    """Test that rewards are properly normalized by world length"""
    mgr = test_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    
    # Reset world 0
    reset_world(mgr, 0)
    
    # Move forward consistently
    controller.reset_actions()
    controller.move_forward(world_idx=0, speed=1.5)
    
    # Run until episode end
    while observer.get_steps_remaining(0) > 0:
        controller.step()
    
    final_reward = observer.get_reward(0)
    max_y = observer.get_max_y_progress(0)
    
    print(f"Normalization test:")
    print(f"  Max Y (normalized): {max_y:.3f}")
    print(f"  Reward: {final_reward:.3f}")
    
    # Since world length is 40 (from consts), reward should be Y/40
    # The normalized observation and reward should be similar
    assert 0.0 < final_reward <= 1.0, "Reward should be normalized between 0 and 1"
    
    # For moderate forward movement, we expect to cover some fraction of the world
    assert 0.05 < final_reward < 0.5, "Reward should reflect partial progress through world"


def test_replay_reward_test_bin(test_manager_from_replay):
    """Test replaying the reward_test.bin recording"""
    import os
    
    # Get path to reward_test.bin
    test_dir = os.path.dirname(os.path.abspath(__file__))
    replay_path = os.path.join(test_dir, "reward_test.bin")
    
    # Create manager from replay
    mgr = test_manager_from_replay(replay_path)
    observer = ObservationReader(mgr)
    
    print(f"\nReplaying reward_test.bin:")
    
    # Track initial state
    initial_pos = observer.get_position(0)
    print(f"Initial position: X={initial_pos[0]:.2f}, Y={initial_pos[1]:.2f}")
    
    # Step through the replay
    current, total = mgr.get_replay_step_count()
    print(f"Replay has {total} steps to process")
    
    step_count = 0
    while True:
        finished = mgr.replay_step()
        mgr.step()  # Execute the simulation step with replayed actions
        step_count += 1
        
        # Print progress periodically
        if step_count % 50 == 0:
            pos = observer.get_position(0)
            max_y = observer.get_max_y_progress(0)
            reward = observer.get_reward(0)
            print(f"Step {step_count}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Max Y={max_y:.3f}, Reward={reward:.3f}")
        
        if finished:
            break
    
    # Check final state
    final_pos = observer.get_position(0)
    final_reward = observer.get_reward(0)
    max_y_progress = observer.get_max_y_progress(0)
    done = observer.get_done_flag(0)
    
    print(f"\nFinal state after {step_count} steps:")
    print(f"  Position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}")
    print(f"  Max Y progress: {max_y_progress:.3f}")
    print(f"  Final reward: {final_reward:.3f}")
    print(f"  Episode done: {done}")
    
    # Verify replay completed successfully
    assert step_count > 0, "Replay should have at least one step"
    
    # If episode ended, verify reward was given
    if done:
        assert final_reward > 0.0, "Should have positive reward if episode completed"
        assert 0.0 < final_reward <= 1.0, "Reward should be normalized"


# NOTE: Removed test_recorded_actions_reward - replaced by comprehensive native recording tests



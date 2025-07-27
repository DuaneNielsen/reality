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


def test_recorded_actions_reward(test_manager):
    """Test that recorded actions from test_reward_system.bin achieve expected reward"""
    import os
    import sys
    import struct
    
    mgr = test_manager
    observer = ObservationReader(mgr)
    
    # Import the load_action_file function from test_replay_actions
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, test_dir)
    from test_replay_actions import load_action_file
    
    # Get the path to the action file
    action_file = os.path.join(test_dir, 'test_reward_system.bin')
    
    if not os.path.exists(action_file):
        pytest.skip(f"Action file not found: {action_file}")
    
    # Load actions from file (single world)
    world_actions = load_action_file(action_file, num_worlds=1)
    actions = world_actions[0]  # Get actions for world 0
    print(f"Loaded {len(actions)} actions from {action_file}")
    
    # Reset world 0
    reset_world(mgr, 0)
    
    # Get initial state
    initial_pos = observer.get_position(0)
    print(f"Initial position: Y={initial_pos[1]:.2f}")
    
    # Get action tensor
    action_tensor = mgr.action_tensor().to_torch()
    
    # Replay actions until episode ends
    max_y = 0.0
    step_count = 0
    
    # Print first 20 actions to understand movement pattern
    print("\nFirst 20 actions:")
    for i in range(min(20, len(actions))):
        move_amount, move_angle, rotate = actions[i]
        print(f"  Step {i}: move_amount={move_amount}, move_angle={move_angle}, rotate={rotate}")
    
    for i, (move_amount, move_angle, rotate) in enumerate(actions):
        # Set the action in the tensor
        action_tensor[0, 0] = move_amount
        action_tensor[0, 1] = move_angle
        action_tensor[0, 2] = rotate
        mgr.step()
        step_count += 1
        
        # Track progress
        pos = observer.get_position(0)
        if pos[1] > max_y:
            max_y = pos[1]
        
        # Print progress at key intervals
        if i in [50, 100, 150]:
            print(f"Step {i}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Max Y={max_y:.2f}, Action: move={move_amount}, angle={move_angle}, rotate={rotate}")
        
        # Check if episode ended
        if observer.get_done_flag(0):
            print(f"\nEpisode ended at step {step_count}")
            break
    
    # Get final reward
    final_reward = observer.get_reward(0)
    max_y_progress = observer.get_max_y_progress(0)
    final_pos = observer.get_position(0)
    
    print(f"\nFinal results:")
    print(f"  Steps taken: {step_count}")
    print(f"  Final position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}")
    print(f"  Maximum Y reached: {max_y:.2f}")
    print(f"  Max Y progress (normalized): {max_y_progress:.3f}")
    print(f"  Final reward: {final_reward:.3f}")
    
    # Verify the reward is reasonable for forward progress
    # The recorded actions achieve about 0.569 reward (normalized max Y progress)
    assert 0.5 < final_reward < 0.7, f"Expected reward between 0.5-0.7, got {final_reward:.3f}"
    
    # Verify reward equals normalized max Y progress (as per game design)
    assert abs(final_reward - max_y_progress) < 0.001, \
        f"Reward ({final_reward:.3f}) should equal max Y progress ({max_y_progress:.3f})"
    
    assert observer.get_done_flag(0), "Episode should be complete"



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
            current_y = observer.get_position(0)[1]
            max_y = observer.get_max_y_progress(0)
            print(f"Step {i}: Y={current_y:.2f}, Max Y progress={max_y:.3f}")
    
    # Final steps to trigger reward
    for _ in range(10):
        controller.step()
    
    # Check final state
    final_y = observer.get_position(0)[1]
    final_reward = observer.get_reward(0)
    max_y_progress = observer.get_max_y_progress(0)
    
    print(f"\nFinal state:")
    print(f"  Y position: {initial_y:.2f} -> {final_y:.2f} (moved {final_y - initial_y:.2f})")
    print(f"  Max Y progress: {max_y_progress:.3f}")
    print(f"  Final reward: {final_reward:.3f}")
    
    # Verify reward matches progress
    assert final_reward > 0.0, "Should receive positive reward for forward movement"
    assert observer.get_done_flag(0), "Episode should be done"
    assert observer.get_steps_remaining(0) == 0, "Steps should be exhausted"


def test_navigation_with_obstacles(test_manager):
    """Test navigating around obstacles"""
    mgr = test_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    
    # Reset world 1
    reset_world(mgr, 1)
    
    # Strategy: Move forward, then try to navigate around obstacles
    controller.reset_actions()
    
    # Phase 1: Move forward
    controller.move_forward(world_idx=1, speed=1.0)
    for _ in range(30):
        controller.step()
    
    y_after_forward = observer.get_position(1)[1]
    print(f"After forward movement: Y={y_after_forward:.2f}")
    
    # Phase 2: Turn right and move
    controller.turn_right(world_idx=1, speed=1.0)
    for _ in range(20):
        controller.step()
        
    # Phase 3: Turn left and continue
    controller.turn_left(world_idx=1, speed=1.0)
    for _ in range(30):
        controller.step()
    
    y_after_navigation = observer.get_position(1)[1]
    print(f"After navigation: Y={y_after_navigation:.2f}")
    
    # Phase 4: Straight forward again
    controller.move_forward(world_idx=1, speed=1.0)
    
    # Continue until episode end
    while observer.get_steps_remaining(1) > 0:
        controller.step()
    
    # Check final results
    final_reward = observer.get_reward(1)
    max_y = observer.get_max_y_progress(1)
    
    print(f"\nFinal navigation results:")
    print(f"  Max Y progress: {max_y:.3f}")
    print(f"  Final reward: {final_reward:.3f}")
    
    assert final_reward > 0.0, "Should receive reward for any forward progress"
    assert observer.get_done_flag(1), "Episode should be done"


def test_backward_movement_no_progress(test_manager):
    """Test that moving backward doesn't increase max Y"""
    mgr = test_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    
    # Reset world 2
    reset_world(mgr, 2)
    
    # Move forward first
    controller.reset_actions()
    controller.move_forward(world_idx=2, speed=2.0)
    controller.step(30)
    
    max_y_after_forward = observer.get_max_y_progress(2)
    print(f"Max Y after forward movement: {max_y_after_forward:.3f}")
    
    # Now move backward
    controller.move_backward(world_idx=2, speed=2.0)
    controller.step(20)
    
    max_y_after_backward = observer.get_max_y_progress(2)
    current_y = observer.get_position(2)[1]
    print(f"After backward movement: Current Y={current_y:.2f}, Max Y={max_y_after_backward:.3f}")
    
    # Max Y should not decrease
    assert max_y_after_backward >= max_y_after_forward, "Max Y should never decrease"
    
    # Continue until episode end
    controller.stop(world_idx=2)
    while observer.get_steps_remaining(2) > 0:
        controller.step()
    
    final_reward = observer.get_reward(2)
    print(f"Final reward: {final_reward:.3f}")
    
    # Reward should be based on max Y reached, not current position
    assert final_reward > 0.0, "Should still get reward based on max Y reached"


def test_multi_world_different_strategies(test_manager):
    """Test different movement strategies across worlds"""
    mgr = test_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)
    
    # Reset all worlds
    reset_all_worlds(mgr)
    
    # World 0: Aggressive forward movement
    # World 1: Careful navigation with turns
    # World 2: Stop and go
    # World 3: No movement (control)
    
    for step in range(195):
        controller.reset_actions()
        
        # World 0: Full speed ahead
        controller.move_forward(world_idx=0, speed=2.0)
        
        # World 1: Alternate between forward and turning
        if step % 20 < 10:
            controller.move_forward(world_idx=1, speed=1.0)
        elif step % 20 < 15:
            controller.turn_right(world_idx=1, speed=0.5)
        else:
            controller.turn_left(world_idx=1, speed=0.5)
            
        # World 2: Stop and go pattern
        if step % 10 < 5:
            controller.move_forward(world_idx=2, speed=1.5)
        else:
            controller.stop(world_idx=2)
            
        # World 3: No movement
        controller.stop(world_idx=3)
        
        controller.step()
        
        # Verify no rewards during episode
        for w in range(4):
            assert observer.get_reward(w) == 0.0, f"No reward during episode for world {w}"
    
    # Final steps to trigger rewards
    controller.step(5)
    
    # Compare final rewards
    print("\nFinal results across worlds:")
    for w in range(4):
        reward = observer.get_reward(w)
        max_y = observer.get_max_y_progress(w)
        final_y = observer.get_position(w)[1]
        print(f"World {w}: Final Y={final_y:.2f}, Max Y={max_y:.3f}, Reward={reward:.3f}")
    
    # Verify reward ordering matches progress
    rewards = [observer.get_reward(w) for w in range(4)]
    
    # World 0 (aggressive) should have highest reward
    assert rewards[0] > rewards[3], "Moving forward should give more reward than staying still"
    # World 3 (no movement) should have lowest reward  
    assert rewards[3] > 0.0, "Even stationary agent gets small reward from spawn position"


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



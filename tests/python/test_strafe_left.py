#!/usr/bin/env python3
"""
Test agent strafe left movement.
"""

import numpy as np
import pytest
import torch
import math


# Action enums for clarity
class MoveAmount:
    STOP = 0
    SLOW = 1
    MEDIUM = 2
    FAST = 3


class MoveAngle:
    FORWARD = 0      # 0 degrees
    FORWARD_RIGHT = 1  # 45 degrees
    RIGHT = 2        # 90 degrees
    BACK_RIGHT = 3   # 135 degrees
    BACKWARD = 4     # 180 degrees
    BACK_LEFT = 5    # 225 degrees
    LEFT = 6         # 270 degrees
    FORWARD_LEFT = 7 # 315 degrees


class Rotate:
    FAST_LEFT = 0
    SLOW_LEFT = 1
    NONE = 2        # Center bucket - no rotation (viewer default)
    SLOW_RIGHT = 3
    FAST_RIGHT = 4


# Self observation indices
class ObsIndex:
    GLOBAL_X = 0
    GLOBAL_Y = 1
    GLOBAL_Z = 2
    MAX_Y = 3
    THETA = 4


def test_strafe_left(test_manager):
    """Test agent strafing left (moving left while maintaining forward orientation)"""
    mgr = test_manager
    
    # Reset to start fresh episode
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1  # Reset all worlds
    mgr.step()
    reset_tensor[:] = 0
    
    # Get initial position
    initial_obs = mgr.self_observation_tensor().to_torch()[0, 0]
    initial_x = initial_obs[ObsIndex.GLOBAL_X].item()
    initial_y = initial_obs[ObsIndex.GLOBAL_Y].item()
    initial_theta = initial_obs[ObsIndex.THETA].item()
    
    print(f"Initial position: x={initial_x:.3f}, y={initial_y:.3f}, theta={initial_theta:.3f}")
    
    # Set strafe left movement for all steps
    actions = mgr.action_tensor().to_torch()
    actions[:] = 0  # Clear all actions first
    
    # Set strafe left for all worlds
    num_worlds = actions.shape[0]
    for world in range(num_worlds):
        actions[world, 0, 0] = MoveAmount.MEDIUM  # Medium speed
        actions[world, 0, 1] = MoveAngle.LEFT     # Left direction (270 degrees)
        actions[world, 0, 2] = Rotate.NONE        # No rotation - maintain orientation
    
    # Run for 100 steps
    for step in range(100):
        mgr.step()
    
    # Check final state
    final_obs = mgr.self_observation_tensor().to_torch()[0, 0]
    final_x = final_obs[ObsIndex.GLOBAL_X].item()
    final_y = final_obs[ObsIndex.GLOBAL_Y].item()
    final_theta = final_obs[ObsIndex.THETA].item()
    
    print(f"Final position: x={final_x:.3f}, y={final_y:.3f}, theta={final_theta:.3f}")
    print(f"Delta: dx={final_x - initial_x:.3f}, dy={final_y - initial_y:.3f}")
    
    # Calculate expected movement direction based on agent's orientation
    # Strafe left means moving 90 degrees to the left of facing direction
    strafe_angle = initial_theta - math.pi/2  # 90 degrees left
    expected_dx = math.sin(strafe_angle)
    expected_dy = math.cos(strafe_angle)
    
    actual_dx = final_x - initial_x
    actual_dy = final_y - initial_y
    
    # Normalize vectors for comparison
    actual_length = math.sqrt(actual_dx**2 + actual_dy**2)
    if actual_length > 0.01:  # Avoid division by zero
        actual_dx /= actual_length
        actual_dy /= actual_length
        
        # Check movement direction (should be roughly 90 degrees left of facing)
        dot_product = actual_dx * expected_dx + actual_dy * expected_dy
        print(f"Movement direction alignment: {dot_product:.3f} (1.0 = perfect)")
        assert dot_product > 0.9, f"Agent should strafe left, but moved in wrong direction"
    
    # Orientation should remain roughly the same (no rotation)
    theta_change = abs(final_theta - initial_theta)
    assert theta_change < 0.1, f"Agent should maintain orientation while strafing, but theta changed by {theta_change:.3f}"
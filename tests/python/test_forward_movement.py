#!/usr/bin/env python3
"""
Test agent forward movement.
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


def test_forward_only(test_manager):
    """Test agent moving forward for entire episode"""
    mgr = test_manager
    
    # Reset to start fresh episode
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1  # Reset all worlds
    mgr.step()
    reset_tensor[:] = 0
    
    # Set forward movement for all steps
    actions = mgr.action_tensor().to_torch()
    actions[:] = 0  # Clear all actions first
    
    # Set same action for ALL worlds (in case viewer shows a different one)
    num_worlds = actions.shape[0]
    for world in range(num_worlds):
        actions[world, 0, 0] = MoveAmount.MEDIUM  # Medium speed
        actions[world, 0, 1] = MoveAngle.FORWARD  # Forward direction
        actions[world, 0, 2] = Rotate.NONE        # No rotation
    
    # Debug: print action values and tensor info
    print(f"Action tensor shape: {actions.shape}")
    print(f"Action values set: moveAmount={actions[0, 0, 0]}, moveAngle={actions[0, 0, 1]}, rotate={actions[0, 0, 2]}")
    print(f"Full action for world 0: {actions[0, 0].tolist()}")
    
    # Run for entire episode (200 steps)
    for step in range(200):
        mgr.step()
        if step < 5:  # Check first few steps
            print(f"Step {step}: actions = {actions[0, 0].tolist()}")
    
    # Check final state
    final_obs = mgr.self_observation_tensor().to_torch()[0, 0]
    final_y = final_obs[ObsIndex.GLOBAL_Y].item()
    max_y = final_obs[ObsIndex.MAX_Y].item()
    
    # Agent should have moved forward significantly
    print(f"Final Y position: {final_y}")
    print(f"Max Y reached: {max_y}")
    
    # Basic sanity check - agent should have moved forward
    assert max_y > 0.1, f"Agent should have moved forward, but max_y = {max_y}"
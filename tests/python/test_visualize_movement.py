#!/usr/bin/env python3
"""
Test specifically designed to demonstrate movement patterns for visualization.
"""

import numpy as np
import pytest
import torch


def test_movement_demo(test_manager):
    """Demonstrate different movement patterns for visualization"""
    mgr = test_manager
    
    # Reset all worlds
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0
    
    actions = mgr.action_tensor().to_torch()
    
    # Different movement pattern for each world:
    # World 0: Move forward only
    # World 1: Rotate in place
    # World 2: Move in circle (forward + rotation)
    # World 3: Zigzag (alternating angles)
    
    for step in range(100):
        # World 0: Forward movement
        actions[0, 0, 0] = 2  # Medium speed forward
        actions[0, 0, 1] = 0  # Straight ahead
        actions[0, 0, 2] = 2  # No rotation (center bucket)
        
        # World 1: Rotation only
        actions[1, 0, 0] = 0  # No movement
        actions[1, 0, 1] = 0  # Doesn't matter
        actions[1, 0, 2] = 3  # Rotate right
        
        # World 2: Circle (move + rotate)
        actions[2, 0, 0] = 1  # Slow forward
        actions[2, 0, 1] = 0  # Forward
        actions[2, 0, 2] = 3  # Rotate right
        
        # World 3: Zigzag
        if step % 20 < 10:
            actions[3, 0, 0] = 2  # Forward
            actions[3, 0, 1] = 1  # Slight right
            actions[3, 0, 2] = 2  # No rotation
        else:
            actions[3, 0, 0] = 2  # Forward
            actions[3, 0, 1] = 7  # Slight left
            actions[3, 0, 2] = 2  # No rotation
        
        mgr.step()
        
        # Print positions every 25 steps
        if step % 25 == 0:
            obs = mgr.self_observation_tensor().to_torch()
            print(f"\nStep {step}:")
            for i in range(4):
                x, y, z = obs[i, 0, 0:3]
                print(f"  World {i}: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    
    print("\nMovement patterns demonstrated:")
    print("  World 0: Forward only (should move along Y axis)")
    print("  World 1: Rotation only (should stay in place)")
    print("  World 2: Circle motion (forward + rotation)")
    print("  World 3: Zigzag pattern")
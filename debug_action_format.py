#!/usr/bin/env python3
"""Debug script to check action data format"""

import numpy as np
from pathlib import Path

# Load the action data
action_file = Path("test_recordings/test_reward_system.py_actions.bin")
if not action_file.exists():
    print(f"Action file not found: {action_file}")
    exit(1)

# Read the binary data
data = np.fromfile(str(action_file), dtype=np.int32)

print(f"Total int32 elements: {len(data)}")
print(f"First 30 values: {data[:30]}")

# Check if data looks reasonable (action values should be in expected ranges)
print("\nChecking value ranges:")
print(f"Min value: {data.min()}")
print(f"Max value: {data.max()}")
print(f"Unique values: {np.unique(data)}")

# Try to reshape based on expected format
# Viewer expects: [steps][worlds][agents][3 components]
num_worlds = 4
num_agents = 2
num_components = 3

elements_per_step = num_worlds * num_agents * num_components
if len(data) % elements_per_step == 0:
    num_steps = len(data) // elements_per_step
    print(f"\nData can be reshaped to {num_steps} steps")
    
    # Reshape and look at first step
    reshaped = data.reshape(num_steps, num_worlds, num_agents, num_components)
    print(f"\nFirst step data shape: {reshaped[0].shape}")
    print("First step values:")
    for w in range(num_worlds):
        for a in range(num_agents):
            print(f"  World {w}, Agent {a}: moveAmount={reshaped[0,w,a,0]}, moveAngle={reshaped[0,w,a,1]}, rotate={reshaped[0,w,a,2]}")
else:
    print(f"\nWarning: Data length {len(data)} is not divisible by {elements_per_step}")
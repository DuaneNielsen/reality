#!/usr/bin/env python3
"""
Test replaying recorded actions from a binary file.
This test loads actions from test_reward_system_actions.bin and replays them.
"""

import numpy as np
import pytest
import torch
import struct
import os
from test_helpers import ObservationReader, reset_world

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_action_file(filepath, num_worlds=1):
    """Load actions from binary file in viewer recording format.
    
    The viewer recording format stores actions as sequential int32_t values:
    - Each timestep contains 3 values per world: [move_amount, move_angle, rotate]
    - For multiple worlds, actions are interleaved: world0_actions, world1_actions, ...
    - Total values per timestep = num_worlds * 3
    
    File structure (multi-world example with 2 worlds):
        [timestep 0] world0_move, world0_angle, world0_rotate, world1_move, world1_angle, world1_rotate
        [timestep 1] world0_move, world0_angle, world0_rotate, world1_move, world1_angle, world1_rotate
        ...
    
    Action value ranges:
        - move_amount: 0-3 (0=stop, 1=slow, 2=medium, 3=fast)
        - move_angle: 0-7 (0=forward, 2=right, 4=backward, 6=left, odd=diagonals)
        - rotate: 0-4 (0=fast left, 1=slow left, 2=none, 3=slow right, 4=fast right)
    
    Args:
        filepath: Path to the binary action file
        num_worlds: Number of worlds in the recording (default: 1)
        
    Returns:
        List of world action sequences, where:
        - Outer list has length num_worlds
        - Each inner list contains action tuples (move_amount, move_angle, rotate) for that world
        - Example: For 2 worlds, returns [[world0_actions], [world1_actions]]
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Each action value is a 4-byte signed integer (int32_t)
    num_values = len(data) // 4
    values = struct.unpack(f'{num_values}i', data)
    
    # Verify file contains complete frames
    values_per_frame = num_worlds * 3
    if len(values) % values_per_frame != 0:
        raise ValueError(f"File contains {len(values)} values, not divisible by {values_per_frame} "
                        f"(num_worlds={num_worlds} * 3 values per action)")
    
    num_timesteps = len(values) // values_per_frame
    
    # Initialize action lists for each world
    world_actions = [[] for _ in range(num_worlds)]
    
    # Parse actions frame by frame
    for t in range(num_timesteps):
        frame_start = t * values_per_frame
        
        # Extract actions for each world in this timestep
        for w in range(num_worlds):
            world_offset = frame_start + (w * 3)
            move_amount = values[world_offset]
            move_angle = values[world_offset + 1]
            rotate = values[world_offset + 2]
            
            world_actions[w].append((move_amount, move_angle, rotate))
    
    return world_actions


def test_replay_recorded_actions(test_manager):
    """Test replaying actions from test_replay_actions.bin"""
    mgr = test_manager
    observer = ObservationReader(mgr)
    
    # Get the path to the action file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    action_file = os.path.join(test_dir, 'test_replay_actions.bin')
    
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
    print(f"Initial position: X={initial_pos[0]:.2f}, Y={initial_pos[1]:.2f}, Z={initial_pos[2]:.2f}")
    
    # Get action tensor
    action_tensor = mgr.action_tensor().to_torch()
    
    # Replay all actions
    max_y = 0.0
    for i, (move_amount, move_angle, rotate) in enumerate(actions):
        # Set the action in the tensor (world 0, agent 0)
        action_tensor[0, 0] = move_amount
        action_tensor[0, 1] = move_angle
        action_tensor[0, 2] = rotate
        mgr.step()
        
        # Track progress
        pos = observer.get_position(0)
        max_y_progress = observer.get_max_y_progress(0)
        reward = observer.get_reward(0)
        done = observer.get_done_flag(0)
        
        if pos[1] > max_y:
            max_y = pos[1]
        
        # Print progress every 50 steps
        if i % 50 == 0 or done:
            print(f"Step {i}: Y={pos[1]:.2f}, Max Y={max_y:.2f}, "
                  f"Max Y Progress={max_y_progress:.3f}, Reward={reward:.3f}, Done={done}")
        
        # Check if episode ended
        if done:
            print(f"Episode ended at step {i} with reward {reward:.3f}")
            break
    
    # Verify we processed all actions or episode ended properly
    final_pos = observer.get_position(0)
    final_max_y = observer.get_max_y_progress(0)
    
    print(f"\nFinal results:")
    print(f"  Actions processed: {i+1}/{len(actions)}")
    print(f"  Final position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}, Z={final_pos[2]:.2f}")
    print(f"  Maximum Y reached: {max_y:.2f}")
    print(f"  Final max Y progress: {final_max_y:.3f}")
    
    # Basic sanity checks
    assert i > 0, "Should have processed at least one action"
    assert final_max_y > 0, "Should have made some forward progress"


def test_action_file_format():
    """Test that we can properly parse the action file format"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    action_file = os.path.join(test_dir, 'test_replay_actions.bin')
    
    if not os.path.exists(action_file):
        pytest.skip(f"Action file not found: {action_file}")
    
    world_actions = load_action_file(action_file, num_worlds=1)
    actions = world_actions[0]  # Get actions for world 0
    
    # Verify actions are valid
    for i, (move_amount, move_angle, rotate) in enumerate(actions[:10]):  # Check first 10
        # Check value ranges based on the game's action space
        assert 0 <= move_amount <= 3, f"Invalid move_amount {move_amount} at action {i}"
        assert 0 <= move_angle <= 7, f"Invalid move_angle {move_angle} at action {i}"
        assert 0 <= rotate <= 4, f"Invalid rotate {rotate} at action {i}"
        
        print(f"Action {i}: move_amount={move_amount}, move_angle={move_angle}, rotate={rotate}")


def test_viewer_compatibility():
    """Test that the action file is compatible with viewer replay format"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    action_file = os.path.join(test_dir, 'test_replay_actions.bin')
    
    if not os.path.exists(action_file):
        pytest.skip(f"Action file not found: {action_file}")
    
    # Check file size
    file_size = os.path.getsize(action_file)
    print(f"File size: {file_size} bytes")
    
    # Should be divisible by 12 (3 int32_t per action = 12 bytes)
    assert file_size % 12 == 0, f"File size {file_size} not divisible by 12 (3 int32_t per action)"
    
    num_actions = file_size // 12
    print(f"Number of actions: {num_actions}")
    
    # Load and verify structure
    world_actions = load_action_file(action_file, num_worlds=1)
    actions = world_actions[0]  # Get actions for world 0
    assert len(actions) == num_actions, f"Expected {num_actions} actions, got {len(actions)}"
    
    # The viewer expects actions for all worlds in each frame
    # For single world recording, each frame should have exactly 3 values
    print(f"Action file contains {len(actions)} frames for a single world")
    print("File is compatible with viewer replay format")


def test_multi_world_loading():
    """Test loading actions for multiple worlds"""
    # Create a test file with 2 worlds, 3 timesteps
    test_data = [
        # Timestep 0
        1, 0, 2,  # World 0: move slow forward, no rotation
        2, 4, 2,  # World 1: move medium backward, no rotation
        # Timestep 1
        3, 2, 3,  # World 0: move fast right, slow right rotation
        0, 0, 2,  # World 1: stop, no rotation
        # Timestep 2
        1, 6, 1,  # World 0: move slow left, slow left rotation
        1, 0, 4,  # World 1: move slow forward, fast right rotation
    ]
    
    # Write test file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        test_file = f.name
        data = struct.pack(f'{len(test_data)}i', *test_data)
        f.write(data)
    
    try:
        # Load with correct number of worlds
        world_actions = load_action_file(test_file, num_worlds=2)
        
        assert len(world_actions) == 2, "Should have 2 world action lists"
        assert len(world_actions[0]) == 3, "World 0 should have 3 timesteps"
        assert len(world_actions[1]) == 3, "World 1 should have 3 timesteps"
        
        # Verify world 0 actions
        assert world_actions[0][0] == (1, 0, 2), "World 0, timestep 0"
        assert world_actions[0][1] == (3, 2, 3), "World 0, timestep 1"
        assert world_actions[0][2] == (1, 6, 1), "World 0, timestep 2"
        
        # Verify world 1 actions
        assert world_actions[1][0] == (2, 4, 2), "World 1, timestep 0"
        assert world_actions[1][1] == (0, 0, 2), "World 1, timestep 1"
        assert world_actions[1][2] == (1, 0, 4), "World 1, timestep 2"
        
        print("Multi-world loading test passed!")
            
    finally:
        # Clean up
        os.unlink(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
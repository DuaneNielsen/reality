#!/usr/bin/env python3
"""
Test native C++ recording/replay functionality on GPU.
"""

import pytest
import tempfile
import os
import torch


def test_gpu_recording_basic(gpu_manager):
    """Test basic recording functionality on GPU"""
    mgr = gpu_manager
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        recording_path = f.name
    
    try:
        # Test recording lifecycle
        assert not mgr.is_recording()
        
        mgr.start_recording(recording_path, seed=999)
        assert mgr.is_recording()
        
        # Run a few steps with actions
        action_tensor = mgr.action_tensor().to_torch()
        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 1  # SLOW movement
            action_tensor[:, 1] = 0  # FORWARD
            action_tensor[:, 2] = 2  # No rotation
            mgr.step()
        
        mgr.stop_recording()
        assert not mgr.is_recording()
        
        # Verify file was created
        assert os.path.exists(recording_path)
        assert os.path.getsize(recording_path) > 0
        
        print(f"✓ GPU recording successful: {os.path.getsize(recording_path)} bytes")
        
    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_gpu_replay_basic(gpu_manager):
    """Test basic replay functionality on GPU"""
    mgr = gpu_manager
    
    # First create a recording
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        recording_path = f.name
    
    try:
        # Record some actions
        mgr.start_recording(recording_path, seed=456)
        action_tensor = mgr.action_tensor().to_torch()
        
        for step in range(2):
            action_tensor.fill_(0)
            action_tensor[:, 0] = 2  # MEDIUM movement
            action_tensor[:, 1] = step % 8  # Different directions
            action_tensor[:, 2] = 2  # No rotation
            mgr.step()
        
        mgr.stop_recording()
        
        # Now test replay
        assert not mgr.has_replay()
        
        mgr.load_replay(recording_path)
        assert mgr.has_replay()
        
        current, total = mgr.get_replay_step_count()
        assert current == 0
        assert total == 2
        
        # Step through replay
        finished1 = mgr.replay_step()
        assert not finished1
        mgr.step()
        
        finished2 = mgr.replay_step()
        assert finished2  # Should finish after 2 steps
        
        print(f"✓ GPU replay successful: {total} steps replayed")
        
    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_gpu_recording_replay_roundtrip(gpu_manager):
    """Test recording and replay round-trip on GPU"""
    mgr = gpu_manager
    
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        recording_path = f.name
    
    try:
        # Record actions and capture action values
        mgr.start_recording(recording_path, seed=123)
        action_tensor = mgr.action_tensor().to_torch()
        
        recorded_actions = []
        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = (step % 3) + 1  # Varying move amounts
            action_tensor[:, 1] = step  # Varying move angles
            action_tensor[:, 2] = 2     # No rotation
            
            recorded_actions.append(action_tensor.clone())
            mgr.step()
        
        mgr.stop_recording()
        
        # Load replay and verify actions match
        mgr.load_replay(recording_path)
        
        for step in range(3):
            finished = mgr.replay_step()
            
            # Get current action tensor after replay step
            current_actions = mgr.action_tensor().to_torch()
            
            # Should match what we recorded
            assert torch.equal(recorded_actions[step], current_actions), f"Action mismatch at step {step}"
            
            if step < 2:
                assert not finished
            else:
                assert finished
            
            mgr.step()
        
        print(f"✓ GPU round-trip test successful: actions match exactly")
        
    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)
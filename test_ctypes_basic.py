#!/usr/bin/env python3
"""
Basic test for ctypes bindings - CPU and GPU manager creation
"""

import sys
import os

# Add the package to Python path
sys.path.insert(0, '/home/duane/madrona/madrona_escape_room')

try:
    import madrona_escape_room as mer
    print("✓ Successfully imported madrona_escape_room")
except ImportError as e:
    print(f"✗ Failed to import madrona_escape_room: {e}")
    sys.exit(1)

def test_cpu_manager():
    """Test CPU manager creation and basic operations"""
    print("\n=== Testing CPU Manager ===")
    
    try:
        # Create CPU manager
        print("Creating CPU manager...")
        mgr = mer.SimManager(
            exec_mode=mer.madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=4,
            rand_seed=42,
            auto_reset=True
        )
        print("✓ CPU manager created successfully")
        
        # Get tensors
        print("Getting tensors...")
        action_tensor = mgr.action_tensor()
        print(f"✓ Action tensor: shape={action_tensor.shape}, dtype={action_tensor.dtype}")
        
        obs_tensor = mgr.self_observation_tensor()
        print(f"✓ Observation tensor: shape={obs_tensor.shape}, dtype={obs_tensor.dtype}")
        
        reward_tensor = mgr.reward_tensor()
        print(f"✓ Reward tensor: shape={reward_tensor.shape}, dtype={reward_tensor.dtype}")
        
        # Run one step
        print("Running simulation step...")
        mgr.step()
        print("✓ Simulation step completed")
        
        print("✓ CPU manager test passed!")
        return True
        
    except Exception as e:
        print(f"✗ CPU manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_manager():
    """Test GPU manager creation and basic operations"""
    print("\n=== Testing GPU Manager ===")
    
    try:
        # Create GPU manager
        print("Creating GPU manager...")
        mgr = mer.SimManager(
            exec_mode=mer.madrona.ExecMode.CUDA,
            gpu_id=0,
            num_worlds=4,
            rand_seed=42,
            auto_reset=True
        )
        print("✓ GPU manager created successfully")
        
        # Get tensors
        print("Getting tensors...")
        action_tensor = mgr.action_tensor()
        print(f"✓ Action tensor: shape={action_tensor.shape}, dtype={action_tensor.dtype}, GPU={action_tensor.isOnGPU()}")
        
        obs_tensor = mgr.self_observation_tensor()
        print(f"✓ Observation tensor: shape={obs_tensor.shape}, dtype={obs_tensor.dtype}, GPU={obs_tensor.isOnGPU()}")
        
        reward_tensor = mgr.reward_tensor()
        print(f"✓ Reward tensor: shape={reward_tensor.shape}, dtype={reward_tensor.dtype}, GPU={reward_tensor.isOnGPU()}")
        
        # Run one step
        print("Running simulation step...")
        mgr.step()
        print("✓ Simulation step completed")
        
        print("✓ GPU manager test passed!")
        return True
        
    except Exception as e:
        print(f"✗ GPU manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing ctypes bindings for Madrona Escape Room")
    print("=" * 50)
    
    # Test CPU first
    cpu_success = test_cpu_manager()
    
    # Test GPU (the main target)
    gpu_success = test_gpu_manager()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"CPU Manager: {'✓ PASS' if cpu_success else '✗ FAIL'}")
    print(f"GPU Manager: {'✓ PASS' if gpu_success else '✗ FAIL'}")
    
    if cpu_success and gpu_success:
        print("\n🎉 All tests passed! ctypes bindings working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
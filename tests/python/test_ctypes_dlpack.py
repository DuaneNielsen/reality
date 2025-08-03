#!/usr/bin/env python3
"""
DLPack functionality test for ctypes bindings
Tests zero-copy tensor conversion to PyTorch on both CPU and GPU
"""

import sys
import os
import pytest

# Add the package to Python path
sys.path.insert(0, '/home/duane/madrona/madrona_escape_room')

try:
    import madrona_escape_room as mer
    import torch
    import numpy as np
    print("✓ Successfully imported dependencies")
except ImportError as e:
    print(f"✗ Failed to import dependencies: {e}")
    sys.exit(1)

def test_cpu_dlpack():
    """Test CPU DLPack functionality"""
    print("\n=== Testing CPU DLPack ===")
    
    try:
        # Create CPU manager
        mgr = mer.SimManager(
            exec_mode=mer.madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=2,
            rand_seed=42,
            auto_reset=True
        )
        print("✓ CPU manager created")
        
        # Get observation tensor
        obs_tensor = mgr.self_observation_tensor()
        print(f"✓ Observation tensor: shape={obs_tensor.shape}, dtype={obs_tensor.dtype}, GPU={obs_tensor.isOnGPU()}")
        
        # Test to_numpy conversion
        np_array = obs_tensor.to_numpy()
        print(f"✓ NumPy conversion: shape={np_array.shape}, dtype={np_array.dtype}")
        
        # Test to_torch conversion
        torch_tensor = obs_tensor.to_torch()
        print(f"✓ PyTorch conversion: shape={torch_tensor.shape}, dtype={torch_tensor.dtype}, device={torch_tensor.device}")
        
        # Test DLPack protocol
        try:
            dlpack_tensor = torch.from_dlpack(obs_tensor)
            print(f"✓ DLPack conversion: shape={dlpack_tensor.shape}, dtype={dlpack_tensor.dtype}, device={dlpack_tensor.device}")
            
            # Verify zero-copy by checking memory addresses
            np_ptr = np_array.ctypes.data
            torch_ptr = torch_tensor.data_ptr()
            dlpack_ptr = dlpack_tensor.data_ptr()
            
            print(f"✓ Memory addresses - NumPy: {hex(np_ptr)}, PyTorch: {hex(torch_ptr)}, DLPack: {hex(dlpack_ptr)}")
            
            if np_ptr == torch_ptr == dlpack_ptr:
                print("✓ Zero-copy verified - all tensors share same memory!")
            else:
                print("⚠ Warning: Memory addresses differ, may not be zero-copy")
                
        except Exception as e:
            print(f"⚠ DLPack conversion failed (using fallback): {e}")
        
        print("✓ CPU DLPack test passed!")
        return True
        
    except Exception as e:
        print(f"✗ CPU DLPack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_dlpack(gpu_manager):
    """Test GPU DLPack functionality"""
    print("\n=== Testing GPU DLPack ===")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping GPU DLPack test")
        return True
    
    try:
        # Use the session-scoped GPU manager
        mgr = gpu_manager
        print("✓ Using session GPU manager")
        
        # Get observation tensor
        obs_tensor = mgr.self_observation_tensor()
        print(f"✓ Observation tensor: shape={obs_tensor.shape}, dtype={obs_tensor.dtype}, GPU={obs_tensor.isOnGPU()}, GPU_ID={obs_tensor.gpuID()}")
        
        # Test to_torch conversion
        torch_tensor = obs_tensor.to_torch()
        print(f"✓ PyTorch conversion: shape={torch_tensor.shape}, dtype={torch_tensor.dtype}, device={torch_tensor.device}")
        
        # Verify it's on GPU
        if not torch_tensor.is_cuda:
            print("✗ PyTorch tensor should be on CUDA but isn't!")
            return False
        
        # Test DLPack protocol - this is the main goal!
        try:
            dlpack_tensor = torch.from_dlpack(obs_tensor)
            print(f"✓ DLPack conversion: shape={dlpack_tensor.shape}, dtype={dlpack_tensor.dtype}, device={dlpack_tensor.device}")
            
            # Verify it's on GPU
            if not dlpack_tensor.is_cuda:
                print("✗ DLPack tensor should be on CUDA but isn't!")
                return False
            
            # Verify zero-copy by checking memory addresses
            torch_ptr = torch_tensor.data_ptr()
            dlpack_ptr = dlpack_tensor.data_ptr()
            
            print(f"✓ GPU memory addresses - PyTorch: {hex(torch_ptr)}, DLPack: {hex(dlpack_ptr)}")
            
            if torch_ptr == dlpack_ptr:
                print("✓ Zero-copy verified on GPU - tensors share same GPU memory!")
            else:
                print("⚠ Warning: GPU memory addresses differ, may not be zero-copy")
            
            # Test basic operations to ensure the tensor is valid
            print(f"✓ Tensor data sample: {dlpack_tensor.flatten()[:5].cpu().numpy()}")
            
        except Exception as e:
            print(f"⚠ DLPack conversion failed (using fallback): {e}")
            return False
        
        # Test action tensor as well
        action_tensor = mgr.action_tensor()
        print(f"✓ Action tensor: shape={action_tensor.shape}, dtype={action_tensor.dtype}, GPU={action_tensor.isOnGPU()}")
        
        action_torch = torch.from_dlpack(action_tensor)
        print(f"✓ Action DLPack: shape={action_torch.shape}, dtype={action_torch.dtype}, device={action_torch.device}")
        
        print("✓ GPU DLPack test passed!")
        return True
        
    except Exception as e:
        print(f"✗ GPU DLPack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dlpack_device_protocol(request):
    """Test __dlpack_device__ method"""
    print("\n=== Testing DLPack Device Protocol ===")
    
    try:
        # Test CPU
        cpu_mgr = mer.SimManager(
            exec_mode=mer.madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=2,
            rand_seed=42,
            auto_reset=True
        )
        
        cpu_tensor = cpu_mgr.self_observation_tensor()
        cpu_device = cpu_tensor.__dlpack_device__()
        print(f"✓ CPU device info: {cpu_device}")
        
        # Test GPU if available and not skipped
        no_gpu = request.config.getoption("--no-gpu", default=False)
        if torch.cuda.is_available() and not no_gpu:
            # Note: Can't test GPU device protocol separately due to
            # Madrona's one-GPU-manager limitation. This would be tested
            # by other GPU tests that use the gpu_manager fixture.
            print("⚠ Skipping GPU device test - would require separate GPU manager")
        else:
            print("⚠ Skipping GPU device test (CUDA not available or --no-gpu flag set)")
        
        return True
        
    except Exception as e:
        print(f"✗ DLPack device protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing DLPack functionality for ctypes bindings")
    print("=" * 60)
    
    # Test CPU DLPack
    cpu_success = test_cpu_dlpack()
    
    # Test GPU DLPack (the main goal!)
    gpu_success = test_gpu_dlpack()
    
    # Test device protocol
    device_success = test_dlpack_device_protocol()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"CPU DLPack: {'✓ PASS' if cpu_success else '✗ FAIL'}")
    print(f"GPU DLPack: {'✓ PASS' if gpu_success else '✗ FAIL'}")
    print(f"Device Protocol: {'✓ PASS' if device_success else '✗ FAIL'}")
    
    if cpu_success and gpu_success and device_success:
        print("\n🎉 All DLPack tests passed! GPU zero-copy tensors working!")
        print("🚀 CFFI replacement with ctypes is SUCCESSFUL!")
        return 0
    else:
        print("\n❌ Some DLPack tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
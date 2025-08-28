#!/usr/bin/env python3
"""
Simple DLPack test - using cpu_manager fixture
"""

import pytest

try:
    import torch

    import madrona_escape_room as mer

    print("✓ Successfully imported dependencies")
except ImportError as e:
    print(f"✗ Failed to import dependencies: {e}")
    pytest.skip(f"Failed to import dependencies: {e}")


def test_cpu_simple(cpu_manager):
    """Simple CPU test"""
    print("\n=== Testing CPU DLPack (Simple) ===")

    try:
        print("✓ Using CPU manager fixture")

        # Get tensor
        obs_tensor = cpu_manager.self_observation_tensor()
        print(f"✓ Tensor: shape={obs_tensor.shape}, GPU={obs_tensor.isOnGPU()}")

        # Test DLPack
        try:
            dlpack_tensor = torch.from_dlpack(obs_tensor)
            print(f"✓ DLPack: shape={dlpack_tensor.shape}, device={dlpack_tensor.device}")
        except Exception as e:
            print(f"⚠ DLPack failed: {e}")
            # Try regular conversion
            torch_tensor = obs_tensor.to_torch()
            print(f"✓ to_torch: shape={torch_tensor.shape}, device={torch_tensor.device}")

    except Exception as e:
        print(f"✗ CPU test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_simple(gpu_manager):
    """Simple GPU test"""
    print("\n=== Testing GPU DLPack (Simple) ===")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available")
        pytest.skip("CUDA not available")

    try:
        # Use the session-scoped GPU manager
        mgr = gpu_manager
        print("✓ Using session GPU manager")

        # Get tensor
        obs_tensor = mgr.self_observation_tensor()
        print(f"✓ Tensor: shape={obs_tensor.shape}, GPU={obs_tensor.isOnGPU()}")

        # Test DLPack - this is the main goal!
        try:
            print("Testing DLPack conversion...")
            dlpack_tensor = torch.from_dlpack(obs_tensor)
            print(f"✓ DLPack SUCCESS: shape={dlpack_tensor.shape}, device={dlpack_tensor.device}")

            if dlpack_tensor.is_cuda:
                print("🎉 GPU DLPack working! Zero-copy GPU tensors achieved!")
            else:
                print("✗ DLPack tensor should be on CUDA")
                assert False, "DLPack tensor should be on CUDA"

        except Exception as e:
            print(f"✗ DLPack conversion failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test the new create_sim_manager factory function
"""

import pytest

from madrona_escape_room import ExecMode, SensorConfig, create_sim_manager


class TestFactoryFunction:
    """Test the create_sim_manager factory function"""

    def test_basic_cpu_creation(self):
        """Test basic CPU manager creation with defaults"""
        mgr = create_sim_manager(ExecMode.CPU)
        assert mgr is not None

        # Test that it works
        mgr.step()
        obs = mgr.self_observation_tensor().to_numpy()
        assert obs.shape == (4, 1, 5)  # 4 worlds, 1 agent, 5 obs components

    def test_sensor_config_integration(self):
        """Test creation with sensor configuration"""
        sensor = SensorConfig.depth_default()
        mgr = create_sim_manager(ExecMode.CPU, sensor_config=sensor)

        # Should have renderer enabled
        mgr.step()
        depth_tensor = mgr.depth_tensor()

        # Check if tensor is on GPU (might happen due to CUDA renderer)
        if depth_tensor.isOnGPU():
            depth = depth_tensor.to_torch().cpu().numpy()
        else:
            depth = depth_tensor.to_numpy()

        assert depth.shape == (4, 1, 64, 64, 1)  # 4 worlds, 1 agent, 64x64x1 depth

    def test_ascii_level_integration(self):
        """Test creation with ASCII level data"""
        level = """#####
#...#
#.S.#
#...#
#####"""
        mgr = create_sim_manager(ExecMode.CPU, level_data=level)

        # Test that manager works with custom level
        mgr.step()
        obs = mgr.self_observation_tensor().to_numpy()
        assert obs.shape == (4, 1, 5)

    def test_json_level_integration(self):
        """Test creation with JSON level data"""
        json_level = """{
            "ascii": "###\\n#S#\\n###",
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "scale": 1.0,
            "name": "test_json_level"
        }"""
        mgr = create_sim_manager(ExecMode.CPU, level_data=json_level)

        # Test that manager works
        mgr.step()
        obs = mgr.self_observation_tensor().to_numpy()
        assert obs.shape == (4, 1, 5)

    def test_combined_sensor_and_level(self):
        """Test creation with both sensor config and custom level"""
        sensor = SensorConfig.lidar_horizontal_128()
        level = "#####\n#.S.#\n#####"

        mgr = create_sim_manager(
            ExecMode.CPU,
            sensor_config=sensor,
            level_data=level,
            num_worlds=2,  # Custom world count
            rand_seed=123,  # Custom seed
        )

        # Test functionality
        mgr.step()
        obs = mgr.self_observation_tensor().to_numpy()
        assert obs.shape == (2, 1, 5)  # 2 worlds as specified

        # Should have lidar depth data
        depth_tensor = mgr.depth_tensor()
        if depth_tensor.isOnGPU():
            depth = depth_tensor.to_torch().cpu().numpy()
        else:
            depth = depth_tensor.to_numpy()
        assert depth.shape == (
            2,
            1,
            1,
            128,
            1,
        )  # 2 worlds, 1 agent, 1x128x1 lidar (sensor config width)

    def test_parameter_forwarding(self):
        """Test that parameters are correctly forwarded"""
        mgr = create_sim_manager(ExecMode.CPU, num_worlds=8, rand_seed=999, auto_reset=True)

        # Check that parameters were applied
        assert mgr._c_config.num_worlds == 8
        assert mgr._c_config.rand_seed == 999
        assert mgr._c_config.auto_reset

    @pytest.mark.slow
    def test_gpu_creation(self):
        """Test GPU manager creation with CUDA - run in subprocess to avoid conflicts"""
        import subprocess
        import sys

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Run the actual test in a subprocess to avoid SimManager conflicts
        test_code = """
import sys
sys.path.insert(0, "/home/duane/madrona_escape_room")

print("Starting subprocess test", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        print("CUDA not available, exiting", flush=True)
        sys.exit(1)

    print("Importing madrona_escape_room...", flush=True)
    from madrona_escape_room import ExecMode, create_sim_manager
    print("Imports successful", flush=True)

    print("Creating CUDA manager...", flush=True)
    mgr = create_sim_manager(ExecMode.CUDA, gpu_id=0)
    print("Manager created successfully", flush=True)
    assert mgr is not None

    print("Stepping simulation...", flush=True)
    mgr.step()
    print("Step completed", flush=True)

    obs_tensor = mgr.self_observation_tensor()
    print("Got observation tensor", flush=True)

    # GPU tensors need to be converted via PyTorch
    if obs_tensor.isOnGPU():
        obs = obs_tensor.to_torch().cpu().numpy()
    else:
        obs = obs_tensor.to_numpy()

    assert obs.shape == (4, 1, 5)
    print(f"GPU manager created successfully, observation shape: {obs.shape}")

    # Cleanup
    del mgr
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Cleanup completed", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"Test failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

        # Copy environment without forcing CUDA libraries - let PyTorch use its own consistent set
        import os

        env = os.environ.copy()

        # Add PyTorch's CUDA library paths so cusparse can find matching nvJitLink
        pytorch_cuda_base = (
            "/home/duane/madrona_escape_room/.venv/lib/python3.12/site-packages/nvidia"
        )
        pytorch_cuda_paths = (
            f"{pytorch_cuda_base}/nvjitlink/lib:"
            f"{pytorch_cuda_base}/cusparse/lib:"
            f"{pytorch_cuda_base}/cuda_runtime/lib"
        )
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{pytorch_cuda_paths}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = pytorch_cuda_paths

        result = subprocess.run(
            [sys.executable, "-c", test_code], capture_output=True, text=True, timeout=60, env=env
        )

        if result.returncode != 0:
            print(f"Subprocess stdout: {result.stdout}")
            print(f"Subprocess stderr: {result.stderr}")
            pytest.fail(f"GPU factory test failed in subprocess: {result.stderr}")

        print(f"✓ Subprocess test passed: {result.stdout.strip()}")

    @pytest.mark.slow
    def test_gpu_with_sensor_config(self):
        """Test GPU manager with sensor configuration - run in subprocess to avoid conflicts"""
        import subprocess
        import sys

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Run the actual test in a subprocess to avoid SimManager conflicts
        test_code = """
import sys
sys.path.insert(0, "/home/duane/madrona_escape_room")

print("Starting sensor config subprocess test", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        print("CUDA not available, exiting", flush=True)
        sys.exit(1)

    print("Importing madrona_escape_room...", flush=True)
    from madrona_escape_room import ExecMode, SensorConfig, create_sim_manager
    print("Imports successful", flush=True)

    print("Creating depth sensor config...", flush=True)
    sensor = SensorConfig.depth_default()
    print("Sensor config created", flush=True)

    print("Creating CUDA manager with sensor...", flush=True)
    mgr = create_sim_manager(ExecMode.CUDA, sensor_config=sensor, gpu_id=0)
    print("Manager created successfully", flush=True)
    assert mgr is not None

    print("Stepping simulation...", flush=True)
    mgr.step()
    print("Step completed", flush=True)

    print("Getting observation tensor...", flush=True)
    obs_tensor = mgr.self_observation_tensor()
    print("Getting depth tensor...", flush=True)
    depth_tensor = mgr.depth_tensor()
    print("Got both tensors", flush=True)

    # Convert GPU tensors
    if obs_tensor.isOnGPU():
        obs = obs_tensor.to_torch().cpu().numpy()
    else:
        obs = obs_tensor.to_numpy()

    if depth_tensor.isOnGPU():
        depth = depth_tensor.to_torch().cpu().numpy()
    else:
        depth = depth_tensor.to_numpy()

    assert obs.shape == (4, 1, 5)
    assert depth.shape == (4, 1, 64, 64, 1)  # 4 worlds, 1 agent, 64x64x1 depth
    print(f"GPU manager with depth sensor: obs {obs.shape}, depth {depth.shape}")

    # Cleanup
    del mgr
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Cleanup completed", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"Test failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

        # Copy environment without forcing CUDA libraries - let PyTorch use its own consistent set
        import os

        env = os.environ.copy()
        # Remove CUDA library forcing to avoid mixing 12.5 system libs with PyTorch's 12.8 libs

        # Add PyTorch's CUDA library paths so cusparse can find matching nvJitLink
        pytorch_cuda_base = (
            "/home/duane/madrona_escape_room/.venv/lib/python3.12/site-packages/nvidia"
        )
        pytorch_cuda_paths = (
            f"{pytorch_cuda_base}/nvjitlink/lib:"
            f"{pytorch_cuda_base}/cusparse/lib:"
            f"{pytorch_cuda_base}/cuda_runtime/lib"
        )
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{pytorch_cuda_paths}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = pytorch_cuda_paths

        result = subprocess.run(
            [sys.executable, "-c", test_code], capture_output=True, text=True, timeout=60, env=env
        )

        if result.returncode != 0:
            print(f"Subprocess stdout: {result.stdout}")
            print(f"Subprocess stderr: {result.stderr}")
            pytest.fail(f"GPU factory sensor test failed in subprocess: {result.stderr}")

        print(f"✓ Subprocess sensor test passed: {result.stdout.strip()}")

    def test_backward_compatibility_with_conftest(self, cpu_manager):
        """Test that old conftest fixtures still work"""
        # This test uses the traditional cpu_manager fixture
        # to ensure backward compatibility
        assert cpu_manager is not None
        cpu_manager.step()
        obs = cpu_manager.self_observation_tensor().to_numpy()
        assert obs.shape == (4, 1, 5)

"""
Test the new create_sim_manager factory function
"""
import pytest
from madrona_escape_room import create_sim_manager, ExecMode, SensorConfig


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
            rand_seed=123  # Custom seed
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
        assert depth.shape == (2, 1, 1, 128, 1)  # 2 worlds, 1 agent, 1x128x1 lidar

    def test_parameter_forwarding(self):
        """Test that parameters are correctly forwarded"""
        mgr = create_sim_manager(
            ExecMode.CPU,
            num_worlds=8,
            rand_seed=999,
            auto_reset=True
        )
        
        # Check that parameters were applied
        assert mgr._c_config.num_worlds == 8
        assert mgr._c_config.rand_seed == 999
        assert mgr._c_config.auto_reset == True

    @pytest.mark.slow
    def test_gpu_creation(self):
        """Test GPU manager creation with CUDA"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        mgr = create_sim_manager(ExecMode.CUDA, gpu_id=0)
        assert mgr is not None
        
        # Test that it works
        mgr.step()
        obs_tensor = mgr.self_observation_tensor()
        
        # GPU tensors need to be converted via PyTorch
        if obs_tensor.isOnGPU():
            obs = obs_tensor.to_torch().cpu().numpy()
        else:
            obs = obs_tensor.to_numpy()
            
        assert obs.shape == (4, 1, 5)
        print(f"GPU manager created successfully, observation shape: {obs.shape}")

    @pytest.mark.slow  
    def test_gpu_with_sensor_config(self):
        """Test GPU manager with sensor configuration"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        sensor = SensorConfig.depth_default()
        mgr = create_sim_manager(ExecMode.CUDA, sensor_config=sensor, gpu_id=0)
        assert mgr is not None
        
        # Test functionality
        mgr.step()
        obs_tensor = mgr.self_observation_tensor()
        depth_tensor = mgr.depth_tensor()
        
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

    def test_backward_compatibility_with_conftest(self, cpu_manager):
        """Test that old conftest fixtures still work"""
        # This test uses the traditional cpu_manager fixture
        # to ensure backward compatibility
        assert cpu_manager is not None
        cpu_manager.step()
        obs = cpu_manager.self_observation_tensor().to_numpy()
        assert obs.shape == (4, 1, 5)
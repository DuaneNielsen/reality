"""Unit tests for the TorchRL environment wrapper"""

import pytest
import torch
import numpy as np
from tensordict import TensorDict

# Import from madrona_escape_room_learn package
from madrona_escape_room_learn import MadronaEscapeRoomEnv


class TestMadronaEscapeRoomEnv:
    """Test suite for the Madrona Escape Room TorchRL wrapper"""
    
    def test_env_creation_cpu(self):
        """Test environment creation on CPU"""
        env = MadronaEscapeRoomEnv(
            num_worlds=4,
            gpu_id=-1,
            rand_seed=42,
            auto_reset=True
        )
        
        assert env.num_worlds == 4
        assert env.num_agents == 1
        assert env.batch_size == torch.Size([4])
        assert env.device == torch.device("cpu")
        
        env.close()
    
    def test_env_creation_cuda(self):
        """Test environment creation on CUDA (if available)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        env = MadronaEscapeRoomEnv(
            num_worlds=8,
            gpu_id=0,
            rand_seed=42,
            auto_reset=True
        )
        
        assert env.num_worlds == 8
        assert env.device == torch.device("cuda:0")
        
        env.close()
    
    def test_action_spec(self):
        """Test action specification"""
        env = MadronaEscapeRoomEnv(num_worlds=2)
        
        # Check action spec structure
        assert "move_amount" in env.action_spec
        assert "move_angle" in env.action_spec
        assert "rotate" in env.action_spec
        
        # Check action dimensions
        assert env.action_spec["move_amount"].n == 4
        assert env.action_spec["move_angle"].n == 8
        assert env.action_spec["rotate"].n == 5
        
        # Check that we can sample valid actions
        action = env.action_spec.rand()
        assert action["move_amount"].shape == torch.Size([2])
        assert action["move_angle"].shape == torch.Size([2])
        assert action["rotate"].shape == torch.Size([2])
        
        env.close()
    
    def test_observation_spec(self):
        """Test observation specification"""
        env = MadronaEscapeRoomEnv(num_worlds=3)
        
        # Check observation spec
        assert "observation" in env.observation_spec
        obs_spec = env.observation_spec["observation"]
        
        # Should have 7 features: 5 self obs + 1 steps + 1 agent_id
        assert obs_spec.shape == torch.Size([3, 7])
        assert obs_spec.dtype == torch.float32
        
        env.close()
    
    def test_reset(self):
        """Test environment reset"""
        env = MadronaEscapeRoomEnv(num_worlds=4)
        
        # Reset environment
        td = env.reset()
        
        # Check output structure
        assert isinstance(td, TensorDict)
        assert "observation" in td
        
        # Check observation shape
        obs = td["observation"]
        assert obs.shape == torch.Size([4, 7])  # 5 self obs + 1 steps + 1 agent_id
        assert obs.dtype == torch.float32
        
        # Check that observations are not NaN or Inf
        assert not torch.isnan(obs).any()
        assert not torch.isinf(obs).any()
        
        env.close()
    
    def test_step(self):
        """Test environment step"""
        env = MadronaEscapeRoomEnv(num_worlds=2)
        
        # Reset first
        td = env.reset()
        
        # Create action
        action = TensorDict({
            "move_amount": torch.tensor([1, 2]),
            "move_angle": torch.tensor([0, 4]),
            "rotate": torch.tensor([2, 2]),
        }, batch_size=[2])
        
        td["action"] = action
        
        # Step environment
        td_out = env.step(td)
        
        # Check output structure - TorchRL puts step results in "next"
        assert "next" in td_out
        assert "observation" in td_out["next"]
        assert "reward" in td_out["next"]
        assert "done" in td_out["next"]
        assert "terminated" in td_out["next"]
        assert "truncated" in td_out["next"]
        
        # Check shapes
        assert td_out["next"]["observation"].shape == torch.Size([2, 7])
        assert td_out["next"]["reward"].shape == torch.Size([2, 1])
        assert td_out["next"]["done"].shape == torch.Size([2, 1])
        
        # Check types
        assert td_out["next"]["reward"].dtype == torch.float32
        assert td_out["next"]["done"].dtype == torch.bool
        
        env.close()
    
    def test_batch_consistency(self):
        """Test that batched environments maintain consistency"""
        num_worlds = 8
        env = MadronaEscapeRoomEnv(num_worlds=num_worlds)
        
        # Reset
        td = env.reset()
        
        # Run several steps with different actions per world
        for step in range(10):
            # Each world gets different actions
            action = TensorDict({
                "move_amount": torch.randint(0, 4, (num_worlds,)),
                "move_angle": torch.randint(0, 8, (num_worlds,)),
                "rotate": torch.randint(0, 5, (num_worlds,)),
            }, batch_size=[num_worlds])
            
            td["action"] = action
            td = env.step(td)
            
            # Verify batch dimensions are preserved
            assert td["next"]["observation"].shape[0] == num_worlds
            assert td["next"]["reward"].shape[0] == num_worlds
            assert td["next"]["done"].shape[0] == num_worlds
        
        env.close()
    
    def test_auto_reset(self):
        """Test automatic reset functionality"""
        env = MadronaEscapeRoomEnv(num_worlds=2, auto_reset=True)
        
        td = env.reset()
        
        # Run many steps to ensure some episodes complete
        episode_ended = False
        for _ in range(300):  # More than episode length
            # Always move forward fast
            action = TensorDict({
                "move_amount": torch.tensor([3, 3]),
                "move_angle": torch.tensor([0, 0]),
                "rotate": torch.tensor([2, 2]),
            }, batch_size=[2])
            
            td["action"] = action
            td = env.step(td)
            
            if td["next"]["done"].any():
                episode_ended = True
                # Next step should have reset the done environments
                # This is handled automatically by Madrona with auto_reset=True
        
        assert episode_ended, "No episode ended in 300 steps"
        env.close()
    
    def test_stateful_behavior(self):
        """Test that the environment maintains state correctly"""
        env = MadronaEscapeRoomEnv(num_worlds=1)
        
        # Reset and get initial observation
        td1 = env.reset()
        obs1 = td1["observation"].clone()
        
        # Take a specific action
        action = TensorDict({
            "move_amount": torch.tensor([3]),  # Fast forward
            "move_angle": torch.tensor([0]),   # Forward
            "rotate": torch.tensor([2]),       # No rotation
        }, batch_size=[1])
        
        td1["action"] = action
        td2 = env.step(td1)
        obs2 = td2["next"]["observation"]
        
        # Position should have changed (Y coordinate increases when moving forward)
        # Observation format: [globalX, globalY, globalZ, maxY, theta, steps_remaining, agent_id]
        assert not torch.allclose(obs1, obs2), "Observation didn't change after movement"
        
        # Specifically, globalY and maxY should increase
        assert obs2[0, 1] > obs1[0, 1], "Global Y position didn't increase"
        assert obs2[0, 3] >= obs1[0, 3], "Max Y didn't update correctly"
        
        env.close()
    
    def test_device_placement(self):
        """Test that tensors are on the correct device"""
        for device_str in ["cpu"]:  # Add "cuda:0" if CUDA available
            if device_str.startswith("cuda") and not torch.cuda.is_available():
                continue
                
            gpu_id = 0 if device_str.startswith("cuda") else -1
            env = MadronaEscapeRoomEnv(
                num_worlds=2,
                gpu_id=gpu_id,
                device=device_str
            )
            
            td = env.reset()
            
            # Check all tensors are on correct device
            assert td["observation"].device.type == device_str.split(":")[0]
            
            # Step and check again
            action = env.action_spec.rand()
            td["action"] = action
            td_out = env.step(td)
            
            assert td_out["next"]["observation"].device.type == device_str.split(":")[0]
            assert td_out["next"]["reward"].device.type == device_str.split(":")[0]
            assert td_out["next"]["done"].device.type == device_str.split(":")[0]
            
            env.close()
    
    def test_seed_determinism(self):
        """Test that setting seed produces deterministic results"""
        # Create two environments with same seed
        env1 = MadronaEscapeRoomEnv(num_worlds=2, rand_seed=123)
        env2 = MadronaEscapeRoomEnv(num_worlds=2, rand_seed=123)
        
        # Reset both
        td1 = env1.reset()
        td2 = env2.reset()
        
        # Initial observations should be identical
        assert torch.allclose(td1["observation"], td2["observation"])
        
        # Take same actions
        action = TensorDict({
            "move_amount": torch.tensor([2, 1]),
            "move_angle": torch.tensor([3, 5]),
            "rotate": torch.tensor([1, 3]),
        }, batch_size=[2])
        
        td1["action"] = action.clone()
        td2["action"] = action.clone()
        
        td1_out = env1.step(td1)
        td2_out = env2.step(td2)
        
        # Results should be identical
        assert torch.allclose(td1_out["next"]["observation"], td2_out["next"]["observation"])
        assert torch.allclose(td1_out["next"]["reward"], td2_out["next"]["reward"])
        assert torch.equal(td1_out["next"]["done"], td2_out["next"]["done"])
        
        env1.close()
        env2.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
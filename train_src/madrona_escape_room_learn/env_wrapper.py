"""PyTorchRL Environment Wrapper for Madrona Escape Room

This module provides a TorchRL-compatible environment wrapper for the Madrona
escape room simulator, enabling seamless integration with TorchRL's RL algorithms.
"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any
import time
import sys
import os

from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, Composite, Categorical, Bounded
from tensordict import TensorDict, TensorDictBase

import madrona_escape_room

# # Add scripts directory to path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
# from step_timer import StepTimer


class MadronaEscapeRoomEnv(EnvBase):
    """TorchRL Environment wrapper for Madrona Escape Room simulator.
    
    This environment wraps the high-performance Madrona escape room simulator,
    providing a stateful, batched environment compatible with TorchRL.
    
    Args:
        num_worlds: Number of parallel worlds to simulate (batch size)
        gpu_id: GPU device ID for CUDA execution (-1 for CPU)
        rand_seed: Random seed for world generation
        auto_reset: Whether to automatically reset completed episodes
        device: PyTorch device for tensors (if None, inferred from gpu_id)
        **kwargs: Additional arguments passed to EnvBase
    """
    
    def __init__(
        self,
        num_worlds: int = 1,
        gpu_id: int = -1,
        rand_seed: int = 0,
        auto_reset: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        # Determine execution mode and device
        if gpu_id >= 0:
            exec_mode = madrona_escape_room.madrona.ExecMode.CUDA
            if device is None:
                device = torch.device(f"cuda:{gpu_id}")
        else:
            exec_mode = madrona_escape_room.madrona.ExecMode.CPU
            if device is None:
                device = torch.device("cpu")
        
        # Ensure device is a torch.device object
        if isinstance(device, str):
            device = torch.device(device)
            
        # Store configuration
        self.num_worlds = num_worlds
        
        # # Initialize timers
        # self._init_timer = StepTimer()
        # self._runtime_timer = StepTimer()
        
        # Initialize the Madrona simulator
        # self._init_timer.start("sim_manager_creation")
        self.sim = madrona_escape_room.SimManager(
            exec_mode=exec_mode,
            gpu_id=gpu_id,
            num_worlds=num_worlds,
            rand_seed=rand_seed,
            auto_reset=auto_reset,
            enable_batch_renderer=False
        )
        if gpu_id >= 0:
            torch.cuda.synchronize()
        # self._init_timer.stop("sim_manager_creation")
        
        # Get tensor references from simulator
        self._action_tensor = self.sim.action_tensor().to_torch()
        self._reward_tensor = self.sim.reward_tensor().to_torch()
        self._done_tensor = self.sim.done_tensor().to_torch()
        self._self_obs_tensor = self.sim.self_observation_tensor().to_torch()
        self._steps_remaining_tensor = self.sim.steps_remaining_tensor().to_torch()
        self._reset_tensor = self.sim.reset_tensor().to_torch()
        
        # Initialize parent class with batch size
        super().__init__(
            device=device,
            batch_size=torch.Size([num_worlds]),
            **kwargs
        )
        
        # Set up action and observation specs
        self._make_specs()
        
    def _make_specs(self) -> None:
        """Define the action and observation specifications."""
        
        # Action spec - composite of three discrete actions
        self.action_spec = Composite(
            move_amount=Categorical(
                n=4,  # 0=stop, 1=slow, 2=medium, 3=fast
                shape=self.batch_size,
                device=self.device
            ),
            move_angle=Categorical(
                n=8,  # 8 directions (0=forward, 2=right, 4=back, 6=left, diagonals)
                shape=self.batch_size,
                device=self.device
            ),
            rotate=Categorical(
                n=5,  # 0=fast left, 1=slow left, 2=none, 3=slow right, 4=fast right
                shape=self.batch_size,
                device=self.device
            ),
            shape=self.batch_size,
            device=self.device
        )
        
        # Observation spec with separate components
        self.observation_spec = Composite(
            observation=Composite(
                self_obs=Bounded(
                    low=-np.inf,
                    high=np.inf,
                    shape=(*self.batch_size, madrona_escape_room.NUM_AGENTS, madrona_escape_room.SELF_OBSERVATION_SIZE),
                    dtype=torch.float32,
                    device=self.device
                ),
                shape=self.batch_size,
                device=self.device
            ),
            shape=self.batch_size,
            device=self.device
        )
        
        # Reward spec
        self.reward_spec = Bounded(
            low=-np.inf,
            high=np.inf,
            shape=(*self.batch_size, madrona_escape_room.NUM_AGENTS, 1),
            dtype=torch.float32,
            device=self.device
        )
        
        # Done spec
        self.done_spec = Categorical(
            n=2,
            shape=(*self.batch_size, madrona_escape_room.NUM_AGENTS, 1),
            dtype=torch.bool,
            device=self.device
        )
        
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment.

        Args:
            tensordict: Input tensordict (may contain reset masks)
            **kwargs: Additional reset arguments
            
        Returns:
            TensorDict containing initial observations
        """
        # self._runtime_timer.start("reset_total")
        
        # Check if we need to do a partial reset (some environments)
        # todo this is lazy, fix it so it correctly resets if a mask is provided, if non mask, then
        # self._runtime_timer.start("reset_setup")
        if tensordict is not None and "_reset" in tensordict:
            reset_mask = tensordict["_reset"]
            # Convert boolean mask to world indices
            if reset_mask.any():
                # For now, we reset all worlds if any need resetting
                # todo reset this properly
                self._reset_tensor.fill_(1)
        else:
            # Full reset - all worlds
            self._reset_tensor.fill_(1)

        self.sim.step()
        
        # Clear the reset tensor
        self._reset_tensor.fill_(0)
        
        # Collect initial observations
        result = self._get_observations().select("observation")
        
        return result
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step in the environment.
        
        Args:
            tensordict: Must contain "action" with move_amount, move_angle, rotate
            
        Returns:
            TensorDict containing observations, rewards, done flags
        """

        action = tensordict["action"]
        
        # Actions are already in the correct shape for single agent per world
        # Just ensure correct dtype
        self._action_tensor[:, 0] = action["move_amount"]
        self._action_tensor[:, 1] = action["move_angle"]
        self._action_tensor[:, 2] = action["rotate"]

        # Step the simulator
        self.sim.step()

        result = self._get_observations()
        
        return result
    
    def _get_observations(self) -> TensorDictBase:
        """Collect observations from the simulator.
        
        Returns:
            TensorDict containing observations, rewards, done flags
        """
        
        batch_shape = self.batch_size
        
        # Use natural tensor shapes from simulator - no views needed!
        self_obs = self._self_obs_tensor  # Shape: [batch, agents_per_world, obs_dim]
        
        # Get steps remaining as integer
        steps_remaining = self._steps_remaining_tensor  # Shape: [batch, agents_per_world, 1]
        
        # Get rewards and done flags
        rewards = self._reward_tensor  # Shape: [batch, agents_per_world, 1]
        dones = self._done_tensor.bool()  # Shape: [batch, agents_per_world, 1]
        
        # Create output tensordict bypassing validation where possible
        # Construct nested structure manually to minimize overhead
        obs_td = TensorDict({}, batch_size=batch_shape, device=self.device)
        obs_td._set_str("self_obs", self_obs, validated=False, inplace=False)
        
        info_td = TensorDict({}, batch_size=batch_shape, device=self.device)
        info_td._set_str("steps_remaining", steps_remaining, validated=False, inplace=False)
        
        result = TensorDict({}, batch_size=batch_shape, device=self.device)
        result._set_str("observation", obs_td, validated=False, inplace=False)
        result._set_str("reward", rewards, validated=False, inplace=False)
        result._set_str("done", dones, validated=False, inplace=False)
        result._set_str("terminated", dones, validated=False, inplace=False)
        result._set_str("truncated", torch.zeros_like(dones), validated=False, inplace=False)
        result._set_str("info", info_td, validated=False, inplace=False)
        return result
    
    def _set_seed(self, seed: Optional[int]) -> Optional[int]:
        """Set the random seed.
        
        Note: The Madrona simulator seed is set at construction time.
        This method is provided for compatibility but doesn't affect
        the already-initialized simulator.
        
        Args:
            seed: Random seed value
            
        Returns:
            The seed value
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return seed
    
    def close(self, *, raise_if_closed: bool = True) -> None:
        """Clean up the environment."""
        # The simulator cleanup is handled by its destructor
        pass
    
    @property
    def num_agents(self) -> int:
        """Number of agents per world."""
        return madrona_escape_room.NUM_AGENTS
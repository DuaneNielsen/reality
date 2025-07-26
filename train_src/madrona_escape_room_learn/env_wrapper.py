"""PyTorchRL Environment Wrapper for Madrona Escape Room

This module provides a TorchRL-compatible environment wrapper for the Madrona
escape room simulator, enabling seamless integration with TorchRL's RL algorithms.
"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any

from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, Composite, Categorical, Bounded
from tensordict import TensorDict, TensorDictBase

import madrona_escape_room


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
        self._num_agents = 1  # Single agent per world in current version
        
        # Initialize the Madrona simulator
        self.sim = madrona_escape_room.SimManager(
            exec_mode=exec_mode,
            gpu_id=gpu_id,
            num_worlds=num_worlds,
            rand_seed=rand_seed,
            auto_reset=auto_reset,
            enable_batch_renderer=False
        )
        
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
            batch_size=torch.Size([num_worlds * self._num_agents]),
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
                    shape=(*self.batch_size, madrona_escape_room.SELF_OBSERVATION_SIZE),
                    dtype=torch.float32,
                    device=self.device
                ),
                steps_remaining=Bounded(
                    low=0.0,
                    high=1.0,
                    shape=(*self.batch_size, 1),
                    dtype=torch.float32,
                    device=self.device
                ),
                agent_id=Bounded(
                    low=0,
                    high=self._num_agents - 1,
                    shape=(*self.batch_size, 1),
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
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        
        # Done spec
        self.done_spec = Categorical(
            n=2,
            shape=(*self.batch_size, 1),
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
        # Check if we need to do a partial reset (some environments)
        if tensordict is not None and "_reset" in tensordict:
            reset_mask = tensordict["_reset"]
            # Convert boolean mask to world indices
            if reset_mask.any():
                # For now, we reset all worlds if any need resetting
                # Future: implement partial reset by world
                self._reset_tensor.fill_(1)
        else:
            # Full reset - all worlds
            self._reset_tensor.fill_(1)
        
        # Step the simulator to process resets
        self.sim.step()
        
        # Clear the reset tensor
        self._reset_tensor.fill_(0)
        
        # Collect initial observations
        return self._get_observations().select("observation")
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step in the environment.
        
        Args:
            tensordict: Must contain "action" with move_amount, move_angle, rotate
            
        Returns:
            TensorDict containing observations, rewards, done flags
        """
        # Extract actions from tensordict
        action = tensordict["action"]
        
        # Actions are already in the correct shape for single agent per world
        # Just ensure correct dtype
        move_amount = action["move_amount"]
        move_angle = action["move_angle"]
        rotate = action["rotate"]
        
        # Write actions to simulator tensor
        # Action tensor is (num_worlds, 3) for single agent case
        self._action_tensor[:, 0] = move_amount.to(self._action_tensor.dtype)
        self._action_tensor[:, 1] = move_angle.to(self._action_tensor.dtype)
        self._action_tensor[:, 2] = rotate.to(self._action_tensor.dtype)
        
        # Step the simulator
        self.sim.step()
        
        # Collect results
        return self._get_observations()
    
    def _get_observations(self) -> TensorDictBase:
        """Collect observations from the simulator.
        
        Returns:
            TensorDict containing observations, rewards, done flags
        """
        # Flatten world and agent dimensions
        batch_shape = self.batch_size
        
        # Get self observations (already normalized in simulator)
        self_obs = self._self_obs_tensor.view(*batch_shape, -1)
        
        # Get steps remaining and normalize
        steps_remaining = self._steps_remaining_tensor.view(*batch_shape, 1).float() / 200.0
        
        # Create agent IDs (useful for multi-agent, but single agent gets 0)
        agent_ids = torch.zeros(*batch_shape, 1, device=self.device)
        
        # Get rewards and done flags
        rewards = self._reward_tensor.view(*batch_shape, 1)
        dones = self._done_tensor.view(*batch_shape, 1).bool()
        
        # Create output tensordict with separate observation components
        return TensorDict(
            {
                "observation": TensorDict(
                    {
                        "self_obs": self_obs,
                        "steps_remaining": steps_remaining,
                        "agent_id": agent_ids,
                    },
                    batch_size=batch_shape,
                    device=self.device,
                ),
                "reward": rewards,
                "done": dones,
                "terminated": dones,  # No truncation in this env
                "truncated": torch.zeros_like(dones),
            },
            batch_size=batch_shape,
            device=self.device,
        )
    
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
        return self._num_agents
    
    @num_agents.setter 
    def num_agents(self, value: int) -> None:
        """Set number of agents (currently fixed at 1)."""
        if value != 1:
            raise ValueError("Current version only supports 1 agent per world")
        self._num_agents = value
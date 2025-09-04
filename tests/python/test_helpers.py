"""
Helper functions for controlling agents in tests.
Provides a cleaner interface for agent movement and actions.
"""

from typing import Optional

import numpy as np

import madrona_escape_room


class AgentController:
    """Simple controller for agent movement in tests"""

    def __init__(self, manager):
        """Initialize controller with a SimManager instance"""
        self.mgr = manager
        self.actions = manager.action_tensor().to_torch()
        self.num_worlds = self.actions.shape[0]
        self.num_agents = self.actions.shape[1]

    def reset_actions(self):
        """Clear all actions to default values"""
        self.actions[:] = 0
        # Set rotation to center position (no rotation)
        self.actions[:, 2] = madrona_escape_room.action.rotate.NONE

    def move_forward(
        self,
        world_idx: Optional[int] = None,
        speed: int = madrona_escape_room.action.move_amount.MEDIUM,
    ):
        """Move agent(s) forward"""
        if world_idx is None:
            # Apply to all worlds
            self.actions[:, 0] = speed
            self.actions[:, 1] = madrona_escape_room.action.move_angle.FORWARD
        else:
            self.actions[world_idx, 0] = speed
            self.actions[world_idx, 1] = madrona_escape_room.action.move_angle.FORWARD

    def move_backward(
        self,
        world_idx: Optional[int] = None,
        speed: int = madrona_escape_room.action.move_amount.MEDIUM,
    ):
        """Move agent(s) backward"""
        if world_idx is None:
            self.actions[:, 0] = speed
            self.actions[:, 1] = madrona_escape_room.action.move_angle.BACKWARD
        else:
            self.actions[world_idx, 0] = speed
            self.actions[world_idx, 1] = madrona_escape_room.action.move_angle.BACKWARD

    def strafe_left(
        self,
        world_idx: Optional[int] = None,
        speed: int = madrona_escape_room.action.move_amount.MEDIUM,
    ):
        """Move agent(s) left (strafe) while maintaining orientation"""
        if world_idx is None:
            self.actions[:, 0] = speed
            self.actions[:, 1] = madrona_escape_room.action.move_angle.LEFT
        else:
            self.actions[world_idx, 0] = speed
            self.actions[world_idx, 1] = madrona_escape_room.action.move_angle.LEFT

    def strafe_right(
        self,
        world_idx: Optional[int] = None,
        speed: int = madrona_escape_room.action.move_amount.MEDIUM,
    ):
        """Move agent(s) right (strafe) while maintaining orientation"""
        if world_idx is None:
            self.actions[:, 0] = speed
            self.actions[:, 1] = madrona_escape_room.action.move_angle.RIGHT
        else:
            self.actions[world_idx, 0] = speed
            self.actions[world_idx, 1] = madrona_escape_room.action.move_angle.RIGHT

    def stop(self, world_idx: Optional[int] = None):
        """Stop agent(s) movement"""
        if world_idx is None:
            self.actions[:, 0] = madrona_escape_room.action.move_amount.STOP
        else:
            self.actions[world_idx, 0] = madrona_escape_room.action.move_amount.STOP

    def rotate_only(
        self,
        world_idx: Optional[int] = None,
        rotation: int = madrona_escape_room.action.rotate.NONE,
    ):
        """Rotate agent(s) in place (no movement)"""
        if world_idx is None:
            self.actions[:, 0] = madrona_escape_room.action.move_amount.STOP
            self.actions[:, 2] = rotation
        else:
            self.actions[world_idx, 0] = madrona_escape_room.action.move_amount.STOP
            self.actions[world_idx, 2] = rotation

    def set_custom_action(
        self,
        world_idx: int,
        move_amount: float,
        move_angle: int,
        rotate: int = madrona_escape_room.action.rotate.NONE,
    ):
        """Set custom action values for specific world"""
        self.actions[world_idx, 0] = move_amount
        self.actions[world_idx, 1] = move_angle
        self.actions[world_idx, 2] = rotate

    def step(self, num_steps: int = 1):
        """Execute simulation steps"""
        for _ in range(num_steps):
            self.mgr.step()


class ObservationReader:
    """Helper for reading and interpreting observations"""

    def __init__(self, manager):
        """Initialize reader with SimManager instance"""
        self.mgr = manager

    def get_position(self, world_idx: int, agent_idx: int = 0) -> np.ndarray:
        """Get agent's current position (x, y, z) - denormalized to world coordinates

        NOTE: This denormalization is incorrect for custom levels with different world sizes.
        The C++ code normalizes using (pos - world_min) / world_length where world_length
        is the Y-axis range. This method incorrectly assumes a fixed world size of 40.0.

        For custom levels, use get_normalized_position() and denormalize manually using
        the actual world boundaries from your compiled level.
        """
        obs = self.mgr.self_observation_tensor().to_torch()
        # WARNING: This assumes default world size - incorrect for custom levels!
        normalized_pos = obs[world_idx, agent_idx, :3].cpu().numpy()
        return normalized_pos * 40.0  # worldLength = 40.0

    def get_normalized_position(self, world_idx: int, agent_idx: int = 0) -> tuple:
        """Get agent's normalized position (x_norm, y_norm, z_norm)"""
        obs = self.mgr.self_observation_tensor().to_torch()
        # Positions are already normalized in observations
        x_norm = obs[world_idx, agent_idx, 0].item()
        y_norm = obs[world_idx, agent_idx, 1].item()
        z_norm = obs[world_idx, agent_idx, 2].item()
        return (x_norm, y_norm, z_norm)

    def get_max_y_progress(self, world_idx: int, agent_idx: int = 0) -> float:
        """Get agent's maximum Y progress (normalized)"""
        obs = self.mgr.self_observation_tensor().to_torch()
        # Index 3 is max Y reached
        return obs[world_idx, agent_idx, 3].item()

    def get_rotation(self, world_idx: int, agent_idx: int = 0) -> float:
        """Get agent's rotation (theta) - normalized to [-1, 1]"""
        obs = self.mgr.self_observation_tensor().to_torch()
        # Index 4 is theta
        return obs[world_idx, agent_idx, 4].item()

    def get_reward(self, world_idx: int, agent_idx: int = 0) -> float:
        """Get agent's current reward"""
        rewards = self.mgr.reward_tensor().to_torch()
        return rewards[world_idx, 0].item()

    def get_done_flag(self, world_idx: int, agent_idx: int = 0) -> bool:
        """Check if episode is done"""
        dones = self.mgr.done_tensor().to_torch()
        return bool(dones[world_idx, 0].item())

    def get_steps_remaining(self, world_idx: int, agent_idx: int = 0) -> int:
        """Get steps remaining in episode"""
        steps_taken = self.mgr.steps_taken_tensor().to_torch()
        episode_length = 200  # consts::episodeLen from consts.hpp
        return max(0, episode_length - int(steps_taken[world_idx, agent_idx, 0].item()))

    def print_agent_state(self, world_idx: int, agent_idx: int = 0):
        """Print current agent state for debugging"""
        pos = self.get_position(world_idx, agent_idx)
        norm_pos = self.get_normalized_position(world_idx, agent_idx)
        max_y = self.get_max_y_progress(world_idx, agent_idx)
        reward = self.get_reward(world_idx, agent_idx)
        done = self.get_done_flag(world_idx, agent_idx)
        steps = self.get_steps_remaining(world_idx, agent_idx)

        print(f"Agent {agent_idx} in World {world_idx}:")
        print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        print(f"  Normalized: ({norm_pos[0]:.3f}, {norm_pos[1]:.3f}, {norm_pos[2]:.3f})")
        print(f"  Max Y Progress: {max_y:.3f}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}, Steps Remaining: {steps}")


def reset_world(manager, world_idx: int):
    """Reset a specific world"""
    reset_tensor = manager.reset_tensor().to_torch()
    reset_tensor[:] = 0
    reset_tensor[world_idx] = 1
    manager.step()
    reset_tensor[:] = 0  # Clear reset flag


def reset_all_worlds(manager):
    """Reset all worlds"""
    reset_tensor = manager.reset_tensor().to_torch()
    reset_tensor[:] = 1
    manager.step()
    reset_tensor[:] = 0  # Clear reset flags

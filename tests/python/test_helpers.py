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

    def get_termination_reason(self, world_idx: int, agent_idx: int = 0) -> int:
        """Get termination reason code

        Returns:
            -1: Not terminated (episode still running)
             0: Episode steps reached (hit 200-step limit)
             1: Goal achieved (reached target/world_max_y)
             2: Collision death (hit DoneOnCollide=true entity)
        """
        termination_tensor = self.mgr.termination_reason_tensor().to_torch()
        return int(termination_tensor[world_idx, agent_idx].item())

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


class TargetTracker:
    """Helper for tracking target positions and calculating distances to agents"""

    def __init__(self, manager):
        """Initialize tracker with SimManager instance"""
        self.mgr = manager

    def get_target_position(self, world_idx: int, target_idx: int = 0) -> np.ndarray:
        """Get current target position in world coordinates"""
        target_tensor = self.mgr.target_position_tensor()
        target_positions = target_tensor.to_numpy()
        return target_positions[world_idx, target_idx, :]  # Shape: [3] (x, y, z)

    def calculate_distance_to_agent(
        self, world_idx: int, agent_idx: int = 0, target_idx: int = 0, compiled_level=None
    ) -> float:
        """Calculate 3D distance between agent and target using proper coordinate conversion

        Args:
            compiled_level: Optional CompiledLevel to get exact world bounds.
                           If None, uses default world bounds (world_length = 40.0)
        """
        # Get agent position from observation tensor (normalized)
        obs = self.mgr.self_observation_tensor().to_torch()
        agent_pos_norm = obs[world_idx, agent_idx, :3].cpu().numpy()  # Normalized position

        # Get target position (already in world coordinates)
        target_pos = self.get_target_position(world_idx, target_idx)

        # Convert agent position to world coordinates using proper denormalization
        if compiled_level is not None:
            # Use exact world bounds from compiled level
            world_width = compiled_level.world_max_x - compiled_level.world_min_x
            world_length = compiled_level.world_max_y - compiled_level.world_min_y
            world_height = compiled_level.world_max_z - compiled_level.world_min_z

            # Apply exact denormalization formula from src/sim.cpp:325-327
            agent_pos_world = np.array(
                [
                    agent_pos_norm[0] * world_width + compiled_level.world_min_x,
                    agent_pos_norm[1] * world_length + compiled_level.world_min_y,
                    agent_pos_norm[2] * world_height + compiled_level.world_min_z,
                ]
            )
        else:
            # Fallback to default world bounds (assumes square world with world_length = 40.0)
            # This works for default levels but not custom test levels
            agent_pos_world = agent_pos_norm * 40.0

        # Calculate 3D distance
        distance = np.linalg.norm(agent_pos_world - target_pos)
        return distance

    def verify_reward_threshold(
        self, distance: float, expected_reward: float, tolerance: float = 0.01
    ) -> bool:
        """Verify that reward matches expected value based on 3.0 unit threshold"""
        if distance <= 3.0:
            return abs(expected_reward - 1.0) < tolerance  # Should get +1.0 reward
        else:
            return abs(expected_reward - 0.0) < tolerance  # Should get 0.0 reward

    def print_distance_info(
        self, world_idx: int, agent_idx: int = 0, target_idx: int = 0, compiled_level=None
    ):
        """Print debugging information about agent-target distance"""
        target_pos = self.get_target_position(world_idx, target_idx)
        distance = self.calculate_distance_to_agent(
            world_idx, agent_idx, target_idx, compiled_level
        )

        # Get properly denormalized agent position
        obs = self.mgr.self_observation_tensor().to_torch()
        agent_pos_norm = obs[world_idx, agent_idx, :3].cpu().numpy()

        if compiled_level is not None:
            world_width = compiled_level.world_max_x - compiled_level.world_min_x
            world_length = compiled_level.world_max_y - compiled_level.world_min_y
            world_height = compiled_level.world_max_z - compiled_level.world_min_z
            agent_pos_world = np.array(
                [
                    agent_pos_norm[0] * world_width + compiled_level.world_min_x,
                    agent_pos_norm[1] * world_length + compiled_level.world_min_y,
                    agent_pos_norm[2] * world_height + compiled_level.world_min_z,
                ]
            )
        else:
            agent_pos_world = agent_pos_norm * 40.0

        print(f"Target {target_idx} in World {world_idx}:")
        print(f"  Target Position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
        print(
            f"  Agent Position: ({agent_pos_world[0]:.2f}, {agent_pos_world[1]:.2f}, {agent_pos_world[2]:.2f})"
        )
        print(f"  Distance: {distance:.2f} units")
        print(f"  Within 3.0 threshold: {distance <= 3.0}")

    def get_all_target_distances(
        self, world_idx: int, agent_idx: int = 0, compiled_level=None
    ) -> list:
        """Get distances to all targets in a world"""
        target_tensor = self.mgr.target_position_tensor()
        target_positions = target_tensor.to_numpy()

        # Get number of targets for this world
        world_targets = target_positions[world_idx]  # Shape: [max_targets, 3]

        distances = []
        for target_idx in range(len(world_targets)):
            # Check if target exists (non-zero position indicates active target)
            target_pos = world_targets[target_idx]
            if np.any(target_pos != 0):  # Active target
                distance = self.calculate_distance_to_agent(
                    world_idx, agent_idx, target_idx, compiled_level
                )
                distances.append((target_idx, distance))

        return distances

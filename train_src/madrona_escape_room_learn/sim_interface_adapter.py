"""
Simple SimInterface adapter for Madrona escape room training.
"""

import madrona_escape_room

from .cfg import SimInterface


def create_sim_interface(manager: madrona_escape_room.SimManager) -> SimInterface:
    """Create SimInterface from SimManager with live tensor references."""

    return SimInterface(
        step=manager.step,
        obs=[
            manager.self_observation_tensor().to_torch(),  # [worlds, agents, 5]
            manager.compass_tensor().to_torch(),  # [worlds, agents, 128]
            manager.depth_tensor().to_torch(),  # [worlds, agents, 128, 1]
        ],
        actions=manager.action_tensor().to_torch(),  # [worlds, 3]
        dones=manager.done_tensor().to_torch(),  # [worlds, agents, 1]
        rewards=manager.reward_tensor().to_torch(),  # [worlds, agents, 1]
    )


def create_training_sim(num_worlds: int, exec_mode, gpu_id: int = -1) -> SimInterface:
    """Create simulator for training with 128x1 horizontal lidar depth."""
    # Use the 128x1 horizontal lidar depth sensor
    depth_config = madrona_escape_room.sensor_config.SensorConfig.lidar_horizontal_128()

    manager = madrona_escape_room.SimManager(
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=42,
        auto_reset=True,
        **depth_config.to_manager_kwargs(),
    )

    return create_sim_interface(manager)

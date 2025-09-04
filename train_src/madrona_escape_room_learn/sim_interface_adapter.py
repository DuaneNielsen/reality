"""
Simple SimInterface adapter for Madrona escape room training.
"""

import madrona_escape_room

from .cfg import SimInterface


class DepthIndex:
    """Depth tensor indices for horizontal lidar"""

    # For tensor shape [worlds, agents, height, width, channels]
    HEIGHT_DIM = 2  # Height dimension index
    WIDTH_DIM = 3  # Width dimension index (beam index)
    CHANNEL_DIM = 4  # Channel dimension index

    # For horizontal lidar [worlds, agents, 1, 128, 1]
    LEFTMOST = 0  # First lidar beam (leftmost)
    CENTER = 64  # Middle lidar beam (center)
    RIGHTMOST = 127  # Last lidar beam (rightmost)

    @staticmethod
    def beam_count():
        """Total number of lidar beams for horizontal lidar"""
        return 128


class CompassIndex:
    """Compass tensor bucket indices (128-bucket one-hot encoding)"""

    FIRST = 0  # First compass bucket (0/128)
    CENTER = 64  # Middle compass bucket (64/128)
    LAST = 127  # Last compass bucket (127/128)

    @staticmethod
    def bucket_count():
        """Total number of compass buckets"""
        return 128


class SelfObsIndex:
    """Indices for self observation tensor components"""

    X = 0  # Position X coordinate
    Y = 1  # Position Y coordinate
    Z = 2  # Position Z coordinate
    PROGRESS = 3  # Progress (maxY reached)
    ROTATION = 4  # Agent rotation


class ObsIndex:
    """Observation tensor indices for SimInterface.obs list"""

    SELF_OBS = 0  # Self observation tensor [worlds, agents, 5]
    COMPASS = 1  # Compass tensor [worlds, agents, 128]
    DEPTH = 2  # Depth tensor [worlds, agents, height, width, channels]


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

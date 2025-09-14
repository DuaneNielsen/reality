"""
Simple SimInterface adapter for Madrona escape room training.
"""

import madrona_escape_room

from .cfg import SimInterface


class LidarIndex:
    """Lidar tensor indices for 128-beam forward arc"""

    # For tensor shape [worlds, agents, samples] - single distance value per beam
    SAMPLES_DIM = 2  # Samples dimension index (128 beams)

    # Beam indices for 128-beam lidar in 360-degree circle
    LEFTMOST = 0  # First lidar beam
    CENTER = 64  # Middle lidar beam
    RIGHTMOST = 127  # Last lidar beam

    @staticmethod
    def beam_count():
        """Total number of lidar beams"""
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
    LIDAR = 2  # Lidar tensor [worlds, agents, 128] - distance values only


def create_sim_interface(manager: madrona_escape_room.SimManager) -> SimInterface:
    """Create SimInterface from SimManager with live tensor references."""

    # Get the full self observation tensor
    self_obs_full = manager.self_observation_tensor().to_torch()  # [worlds, agents, 5]

    # Extract just the progress indicator (maxY reached)
    progress_tensor = self_obs_full[
        :, :, SelfObsIndex.PROGRESS : SelfObsIndex.PROGRESS + 1
    ]  # [worlds, agents, 1]

    return SimInterface(
        step=manager.step,
        obs=[
            progress_tensor,  # [worlds, agents, 1] - just progress
            manager.compass_tensor().to_torch(),  # [worlds, agents, 128]
            manager.lidar_tensor().to_torch(),  # [worlds, agents, 128] - distance only
        ],
        actions=manager.action_tensor().to_torch(),  # [worlds, 3]
        dones=manager.done_tensor().to_torch(),  # [worlds, agents, 1]
        rewards=manager.reward_tensor().to_torch(),  # [worlds, agents, 1]
        manager=manager,
    )


def create_training_sim(
    num_worlds: int, exec_mode, gpu_id: int = -1, rand_seed: int = 42
) -> SimInterface:
    """Create simulator for training with native 128-beam lidar (distance values only)."""
    # Don't use any visual sensor config - just use native lidar system
    manager = madrona_escape_room.SimManager(
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=rand_seed,
        auto_reset=True,
        enable_batch_renderer=False,  # No visual rendering needed
    )

    return create_sim_interface(manager)


def create_training_sim_with_config(
    num_worlds: int, exec_mode, sensor_config, gpu_id: int = -1, rand_seed: int = 42
) -> SimInterface:
    """Create simulator for training with explicit sensor configuration."""
    manager = madrona_escape_room.create_sim_manager(
        exec_mode=exec_mode,
        sensor_config=sensor_config,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=rand_seed,
        auto_reset=True,
    )

    return create_sim_interface(manager)


def setup_training_environment(
    num_worlds: int, exec_mode, gpu_id: int = -1, rand_seed: int = 42
) -> SimInterface:
    """
    Setup training environment with native 128-beam lidar system.

    Args:
        num_worlds: Number of parallel simulation worlds
        exec_mode: madrona.ExecMode.CPU or madrona.ExecMode.CUDA
        gpu_id: GPU device ID (ignored for CPU mode)
        rand_seed: Random seed for reproducible training

    Returns:
        SimInterface configured for training with 128-beam lidar (distance values only)

    Example:
        import madrona
        from madrona_escape_room_learn.sim_interface_adapter import setup_training_environment

        sim = setup_training_environment(
            num_worlds=4096,
            exec_mode=madrona.ExecMode.CUDA,
            gpu_id=0
        )
    """
    return create_training_sim(
        num_worlds=num_worlds, exec_mode=exec_mode, gpu_id=gpu_id, rand_seed=rand_seed
    )


def create_minimal_sim_interface(manager: madrona_escape_room.SimManager) -> SimInterface:
    """Create SimInterface with only compass and progress - no depth sensors."""

    # Get the full self observation tensor
    self_obs_full = manager.self_observation_tensor().to_torch()  # [worlds, agents, 5]

    # Extract just the progress indicator (maxY reached)
    progress_tensor = self_obs_full[
        :, :, SelfObsIndex.PROGRESS : SelfObsIndex.PROGRESS + 1
    ]  # [worlds, agents, 1]

    return SimInterface(
        step=manager.step,
        obs=[
            progress_tensor,  # [worlds, agents, 1] - just progress
            manager.compass_tensor().to_torch(),  # [worlds, agents, 128]
            # No depth tensor - removed for minimal setup
        ],
        actions=manager.action_tensor().to_torch(),  # [worlds, 3]
        dones=manager.done_tensor().to_torch(),  # [worlds, agents, 1]
        rewards=manager.reward_tensor().to_torch(),  # [worlds, agents, 1]
        manager=manager,
    )


def create_lidar_sim_interface(manager: madrona_escape_room.SimManager) -> SimInterface:
    """Create SimInterface with lidar, compass and progress sensors."""

    # Get the full self observation tensor
    self_obs_full = manager.self_observation_tensor().to_torch()  # [worlds, agents, 5]

    # Extract just the progress indicator (maxY reached)
    progress_tensor = self_obs_full[
        :, :, SelfObsIndex.PROGRESS : SelfObsIndex.PROGRESS + 1
    ]  # [worlds, agents, 1]

    # Get lidar tensor - now single-channel distance values only
    lidar_tensor = manager.lidar_tensor().to_torch()  # [worlds, agents, 128] - distance only

    return SimInterface(
        step=manager.step,
        obs=[
            progress_tensor,  # [worlds, agents, 1] - just progress
            manager.compass_tensor().to_torch(),  # [worlds, agents, 128]
            lidar_tensor,  # [worlds, agents, 128] - lidar distance only
        ],
        actions=manager.action_tensor().to_torch(),  # [worlds, 3]
        dones=manager.done_tensor().to_torch(),  # [worlds, agents, 1]
        rewards=manager.reward_tensor().to_torch(),  # [worlds, agents, 1]
        manager=manager,
    )


def setup_lidar_training_environment(
    num_worlds: int,
    exec_mode,
    gpu_id: int = -1,
    rand_seed: int = 42,
    compiled_level=None,
    compiled_levels=None,
) -> SimInterface:
    """
    Setup training environment with native lidar sensor system.

    Args:
        num_worlds: Number of parallel simulation worlds
        exec_mode: madrona.ExecMode.CPU or madrona.ExecMode.CUDA
        gpu_id: GPU device ID (ignored for CPU mode)
        rand_seed: Random seed for reproducible training
        compiled_level: Optional single CompiledLevel to use (deprecated, use compiled_levels)
        compiled_levels: Optional list of CompiledLevel to use for multi-level/curriculum training

    Returns:
        SimInterface with lidar, compass, and progress observations

    Example:
        import madrona
        from madrona_escape_room_learn.sim_interface_adapter import (
            setup_lidar_training_environment,
        )

        sim = setup_lidar_training_environment(
            num_worlds=4096,
            exec_mode=madrona.ExecMode.CUDA,
            gpu_id=0
        )
    """
    # Handle both old and new parameter styles for backward compatibility
    if compiled_levels is not None:
        levels_to_use = compiled_levels
    elif compiled_level is not None:
        levels_to_use = [compiled_level]
    else:
        levels_to_use = None

    manager = madrona_escape_room.SimManager(
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=rand_seed,
        auto_reset=True,
        enable_batch_renderer=False,  # No visual rendering
        compiled_levels=levels_to_use,
    )

    return create_lidar_sim_interface(manager)


def setup_minimal_training_environment(
    num_worlds: int, exec_mode, gpu_id: int = -1, rand_seed: int = 42
) -> SimInterface:
    """
    Setup minimal training environment with no sensors - only compass and progress.

    Args:
        num_worlds: Number of parallel simulation worlds
        exec_mode: madrona.ExecMode.CPU or madrona.ExecMode.CUDA
        gpu_id: GPU device ID (ignored for CPU mode)
        rand_seed: Random seed for reproducible training

    Returns:
        SimInterface with only compass and progress observations

    Example:
        import madrona
        from madrona_escape_room_learn.sim_interface_adapter import (
            setup_minimal_training_environment,
        )

        sim = setup_minimal_training_environment(
            num_worlds=4096,
            exec_mode=madrona.ExecMode.CUDA,
            gpu_id=0
        )
    """
    # Create manager without any sensor config (no visual sensors)
    manager = madrona_escape_room.create_sim_manager(
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=rand_seed,
        auto_reset=True,
        # sensor_config=None,  # Default - no visual sensors
        # enable_batch_renderer=None,  # Default - disabled
    )

    return create_minimal_sim_interface(manager)

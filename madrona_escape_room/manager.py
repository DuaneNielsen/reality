"""
SimManager class for Madrona Escape Room
Handles simulation lifecycle, tensor access, recording and replay
"""

from ctypes import POINTER, byref, c_bool, c_uint32

import numpy as np

# Import from our modules
from .ctypes_bindings import (
    ManagerConfig,
    MER_ManagerHandle,
    MER_Tensor,
    lib,
    validate_compiled_level,
)
from .generated_dataclasses import ReplayMetadata


def _check_result(result):
    """Check C API result and raise exception if error"""
    if result != 0:  # Success = 0
        error_msg = lib.mer_result_to_string(result)
        if error_msg:
            error_str = error_msg.decode("utf-8")
        else:
            error_str = f"Unknown error code: {result}"
        raise RuntimeError(f"Madrona Escape Room error: {error_str}")


class SimManager:
    """Main simulation manager class"""

    def __init__(
        self,
        exec_mode,
        gpu_id,
        num_worlds,
        rand_seed,
        auto_reset,
        enable_batch_renderer=False,
        compiled_levels=None,  # Pass CompiledLevel objects directly
        batch_render_view_width=64,  # Custom render view width
        batch_render_view_height=64,  # Custom render view height
        custom_vertical_fov=0.0,  # Custom vertical FOV in degrees (0 = use default)
        render_mode=None,  # Render mode: RenderMode.RGBD (default) or RenderMode.Depth
    ):
        # Import ExecMode and RenderMode for type checking
        from .generated_constants import ExecMode, RenderMode

        # Create config
        config = ManagerConfig()
        config.exec_mode = exec_mode.value if isinstance(exec_mode, ExecMode) else exec_mode
        config.gpu_id = gpu_id
        config.num_worlds = num_worlds
        config.rand_seed = rand_seed
        config.auto_reset = auto_reset
        config.enable_batch_renderer = enable_batch_renderer
        config.batch_render_view_width = batch_render_view_width
        config.batch_render_view_height = batch_render_view_height
        config.custom_vertical_fov = custom_vertical_fov

        # Set render mode (default to RGBD for backward compatibility)
        if render_mode is None:
            config.render_mode = RenderMode.RGBD
        else:
            config.render_mode = render_mode.value if hasattr(render_mode, "value") else render_mode

        # If no level provided, use default level
        if compiled_levels is None:
            from .default_level import create_default_level

            compiled_levels = create_default_level()

        # Validate compiled levels
        levels_to_validate = (
            compiled_levels if isinstance(compiled_levels, list) else [compiled_levels]
        )
        for level in levels_to_validate:
            if level is not None:
                validate_compiled_level(level)

        # Create handle
        self._handle = MER_ManagerHandle()

        # Use the wrapper function to handle level array properly
        from .ctypes_bindings import create_manager_with_levels

        result, self._c_config, self._levels_array = create_manager_with_levels(
            byref(self._handle),
            config,
            compiled_levels,  # Pass config directly, not byref
        )
        # Store the c_config and levels_array to keep them alive for the lifetime of the manager
        # This prevents segfaults from the C code accessing freed memory
        _check_result(result)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            lib.mer_destroy_manager(self._handle)

    def step(self):
        """Run one simulation step"""
        result = lib.mer_step(self._handle)
        _check_result(result)

    def _get_tensor(self, getter_func):
        """Helper to get tensor from C API"""
        from ctypes import byref

        from .tensor import Tensor

        c_tensor = MER_Tensor()
        result = getter_func(self._handle, byref(c_tensor))
        _check_result(result)
        return Tensor(c_tensor)

    def reset_tensor(self):
        return self._get_tensor(lib.mer_get_reset_tensor)

    def action_tensor(self):
        return self._get_tensor(lib.mer_get_action_tensor)

    def reward_tensor(self):
        return self._get_tensor(lib.mer_get_reward_tensor)

    def done_tensor(self):
        return self._get_tensor(lib.mer_get_done_tensor)

    def termination_reason_tensor(self):
        return self._get_tensor(lib.mer_get_termination_reason_tensor)

    def self_observation_tensor(self):
        return self._get_tensor(lib.mer_get_self_observation_tensor)

    def compass_tensor(self):
        return self._get_tensor(lib.mer_get_compass_tensor)

    def lidar_tensor(self):
        return self._get_tensor(lib.mer_get_lidar_tensor)

    def steps_taken_tensor(self):
        return self._get_tensor(lib.mer_get_steps_taken_tensor)

    def progress_tensor(self):
        return self._get_tensor(lib.mer_get_progress_tensor)

    def target_position_tensor(self):
        return self._get_tensor(lib.mer_get_target_position_tensor)

    def rgb_tensor(self):
        return self._get_tensor(lib.mer_get_rgb_tensor)

    def depth_tensor(self):
        return self._get_tensor(lib.mer_get_depth_tensor)

    def enable_trajectory_logging(self, world_idx, agent_idx, filename=None):
        """Enable trajectory logging for a specific agent"""
        if filename is not None:
            filename_bytes = filename.encode("utf-8")
            result = lib.mer_enable_trajectory_logging(
                self._handle, world_idx, agent_idx, filename_bytes
            )
        else:
            result = lib.mer_enable_trajectory_logging(self._handle, world_idx, agent_idx, None)
        _check_result(result)

    def disable_trajectory_logging(self):
        """Disable trajectory logging"""
        result = lib.mer_disable_trajectory_logging(self._handle)
        _check_result(result)

    # Recording functionality
    def start_recording(self, filepath):
        """Start recording actions to a binary file from the beginning of the simulation.

        Recording can only be started from a fresh simulation (before any steps are taken).
        The recording will use the Manager's original random seed.

        Args:
            filepath: Path where to save the recording

        Raises:
            RuntimeError: If recording cannot be started (already recording or
                simulation has started)
        """
        filepath_bytes = filepath.encode("utf-8")
        result = lib.mer_start_recording(self._handle, filepath_bytes)
        _check_result(result)

    def stop_recording(self):
        """Stop recording actions"""
        result = lib.mer_stop_recording(self._handle)
        _check_result(result)

    def is_recording(self):
        """Check if currently recording

        Returns:
            bool: True if recording is active
        """
        from ctypes import byref

        is_recording = c_bool()
        result = lib.mer_is_recording(self._handle, byref(is_recording))
        _check_result(result)
        return is_recording.value

    # Replay functionality

    def has_replay(self):
        """Check if a replay is currently loaded

        Returns:
            bool: True if replay is loaded
        """
        from ctypes import byref

        has_replay = c_bool()
        result = lib.mer_has_replay(self._handle, byref(has_replay))
        _check_result(result)
        return has_replay.value

    def replay_step(self):
        """Execute one step of replay

        Returns:
            bool: True if replay finished (no more steps), False if more steps remain
        """
        from ctypes import byref

        finished = c_bool()
        result = lib.mer_replay_step(self._handle, byref(finished))
        _check_result(result)
        return finished.value

    def get_replay_step_count(self):
        """Get current and total step counts for loaded replay

        Returns:
            tuple: (current_step, total_steps)
        """
        from ctypes import byref

        current_step = c_uint32()
        total_steps = c_uint32()
        result = lib.mer_get_replay_step_count(
            self._handle, byref(current_step), byref(total_steps)
        )
        _check_result(result)
        return (current_step.value, total_steps.value)

    def has_checksum_failed(self):
        """Check if checksum verification has failed during replay

        Returns:
            bool: True if checksum verification failed, False otherwise
        """
        from ctypes import byref

        has_failed = c_bool()
        result = lib.mer_has_checksum_failed(self._handle, byref(has_failed))
        _check_result(result)
        return has_failed.value

    # Context manager convenience methods
    def debug_session(self, base_path, enable_recording=True, enable_tracing=True, seed=42):
        """Create a debug session context manager that handles both recording and tracing"""
        return DebugSession(self, base_path, enable_recording, enable_tracing, seed)

    def recording(self, filename, seed=42):
        """Create a recording context manager"""
        return Recording(self, filename, seed)

    def trajectory_logging(self, world_idx, agent_idx, filename=None):
        """Create a trajectory logging context manager"""
        return TrajectoryLogging(self, world_idx, agent_idx, filename)

    @staticmethod
    def read_replay_metadata(filepath):
        """Read replay metadata without creating a manager

        Args:
            filepath: Path to replay file

        Returns:
            ReplayMetadata: Replay metadata dataclass with all fields from the binary file

        Example:
            metadata = SimManager.read_replay_metadata("demo.bin")
            print(f"Replay has {metadata.num_worlds} worlds, version {metadata.version}")
        """
        from ctypes import byref

        metadata = ReplayMetadata()
        filepath_bytes = filepath.encode("utf-8")
        # Convert dataclass to ctypes for C API
        c_metadata = metadata.to_ctype()
        result = lib.mer_read_replay_metadata(filepath_bytes, byref(c_metadata))
        _check_result(result)

        # Convert back to dataclass to get updated values
        metadata = ReplayMetadata.from_buffer(bytearray(c_metadata))

        # Return the dataclass directly instead of converting to dict
        return metadata

    @classmethod
    def from_replay(cls, replay_filepath, exec_mode, gpu_id=0, enable_batch_renderer=False):
        """Create SimManager configured for replay from file

        All configuration (num_worlds, seed, auto_reset) comes from the replay file.
        You only specify execution preferences.

        Args:
            replay_filepath: Path to replay file
            exec_mode: CPU or CUDA execution (madrona.ExecMode.CPU or .CUDA)
            gpu_id: GPU device ID (ignored for CPU mode)
            enable_batch_renderer: Enable rendering (CPU mode only)

        Returns:
            SimManager ready to replay from step 0

        Example:
            # Create manager configured exactly like the replay
            sim = SimManager.from_replay("demo.bin", madrona.ExecMode.CUDA, gpu_id=0)

            # Manager is ready to replay - step through it
            while True:
                finished = sim.replay_step()
                if finished:
                    break
                obs = sim.self_observation_tensor().to_torch()
                # ... process replay data
        """
        # Single C call that handles everything
        from .generated_constants import ExecMode

        # Convert exec_mode to integer value
        exec_mode_value = exec_mode.value if isinstance(exec_mode, ExecMode) else exec_mode

        handle = MER_ManagerHandle()
        result = lib.mer_create_manager_from_replay(
            byref(handle),
            replay_filepath.encode("utf-8"),
            exec_mode_value,
            gpu_id,
            enable_batch_renderer,
        )

        if result != 0:  # Success = 0
            error_msg = lib.mer_result_to_string(result)
            if error_msg:
                error_str = error_msg.decode("utf-8")
            else:
                error_str = f"Unknown error code: {result}"
            raise RuntimeError(
                f"Failed to create manager from replay: {replay_filepath}. Error: {error_str}"
            )

        # Create Python wrapper
        manager = cls.__new__(cls)
        manager._handle = handle

        # Read metadata to set Python attributes
        metadata = cls.read_replay_metadata(replay_filepath)
        manager.num_worlds = metadata.num_worlds
        manager.seed = metadata.seed

        print(
            f"Loaded replay with {metadata.num_worlds} worlds, "
            f"{metadata.num_steps} steps, seed {metadata.seed}"
        )

        return manager

    def _load_replay_internal(self, filepath):
        """Internal replay loading without reconfiguration"""
        filepath_bytes = filepath.encode("utf-8")
        result = lib.mer_load_replay(self._handle, filepath_bytes)
        _check_result(result)


# Context Managers for Recording and Tracing
class Recording:
    """Context manager for recording simulation steps to a file"""

    def __init__(self, manager, filename, seed=42):
        self.manager = manager
        self.filename = filename
        self.seed = seed

    def __enter__(self):
        self.manager.start_recording(self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.stop_recording()


class TrajectoryLogging:
    """Context manager for tracking agent trajectories"""

    def __init__(self, manager, world_idx, agent_idx, filename=None):
        self.manager = manager
        self.world_idx = world_idx
        self.agent_idx = agent_idx
        self.filename = filename

    def __enter__(self):
        self.manager.enable_trajectory_logging(self.world_idx, self.agent_idx, self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.disable_trajectory_logging()


def create_sim_manager(
    exec_mode,
    sensor_config=None,
    level_data=None,
    gpu_id=0,
    num_worlds=4,
    rand_seed=42,
    auto_reset=False,
    batch_render_width=64,
    batch_render_height=64,
    custom_vertical_fov=0.0,
    render_mode=None,
    enable_batch_renderer=None,
):
    """Factory function to create SimManager with flexible configuration

    Args:
        exec_mode: ExecMode.CPU or ExecMode.CUDA
        sensor_config: Optional SensorConfig object for sensor settings
        level_data: Optional level data (ASCII string, JSON string, or CompiledLevel object)
        gpu_id: GPU device ID (ignored for CPU mode)
        num_worlds: Number of simulation worlds
        rand_seed: Random seed for simulation
        auto_reset: Whether to enable automatic episode resets
        batch_render_width: Custom render view width (default 64)
        batch_render_height: Custom render view height (default 64)
        custom_vertical_fov: Custom vertical FOV in degrees (0 = use default)
        render_mode: RenderMode enum value (None = use default RGBD)
        enable_batch_renderer: Force enable/disable renderer (None = auto-detect from sensor_config)

    Returns:
        SimManager: Configured simulation manager

    Examples:
        # Simple CPU manager with defaults
        mgr = create_sim_manager(ExecMode.CPU)

        # With custom sensor configuration
        sensor = SensorConfig.lidar_horizontal_128()
        mgr = create_sim_manager(ExecMode.CPU, sensor_config=sensor)

        # With custom ASCII level
        level = "###\\n#@#\\n###"
        mgr = create_sim_manager(ExecMode.CPU, level_data=level)
    """
    from .generated_constants import ExecMode, RenderMode

    # Handle level data
    compiled_level = None
    if level_data is None:
        # Use default level
        from .default_level import create_default_level

        compiled_level = create_default_level()
    elif isinstance(level_data, str):
        # Check if it's JSON or ASCII
        level_str = level_data.strip()
        if level_str.startswith("{") or level_str.startswith("["):
            # JSON level string
            from .level_compiler import compile_level

            compiled_levels = compile_level(level_str)
            compiled_level = compiled_levels[
                0
            ]  # Extract single level (JSON must be single level here)
        else:
            # ASCII level
            from .level_compiler import compile_ascii_level

            compiled_level = compile_ascii_level(level_str, level_name="custom_level")
    else:
        # Assume it's already a CompiledLevel object
        compiled_level = level_data

    # Handle sensor configuration
    if sensor_config is not None:
        # Apply sensor config settings
        enable_renderer = True
        batch_render_width = sensor_config.width
        batch_render_height = sensor_config.height
        custom_vertical_fov = sensor_config.vertical_fov
        render_mode = sensor_config.render_mode
    else:
        # Use provided parameters or defaults
        if enable_batch_renderer is None:
            enable_renderer = False  # Default to no renderer if no sensor config provided
        else:
            enable_renderer = enable_batch_renderer

        # Use default RGBD render mode if not specified
        if render_mode is None:
            render_mode = RenderMode.RGBD

    return SimManager(
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=rand_seed,
        enable_batch_renderer=enable_renderer,
        auto_reset=auto_reset,
        compiled_levels=compiled_level,
        batch_render_view_width=batch_render_width,
        batch_render_view_height=batch_render_height,
        custom_vertical_fov=custom_vertical_fov,
        render_mode=render_mode,
    )


class DebugSession:
    """Master context manager that handles both recording and trajectory logging"""

    def __init__(self, manager, base_path, enable_recording=True, enable_tracing=True, seed=42):
        from pathlib import Path

        self.manager = manager
        self.base_path = Path(base_path)
        self.enable_recording = enable_recording
        self.enable_tracing = enable_tracing
        self.seed = seed

        # Generate file paths - use .name to avoid issues with dots in filename
        base_name = self.base_path.name
        self.recording_path = (
            self.base_path.with_name(f"{base_name}.rec") if enable_recording else None
        )
        self.trajectory_path = (
            self.base_path.with_name(f"{base_name}_trajectory.txt") if enable_tracing else None
        )

    def __enter__(self):
        if self.enable_recording and self.recording_path:
            self.manager.start_recording(str(self.recording_path))

        if self.enable_tracing and self.trajectory_path:
            self.manager.enable_trajectory_logging(0, 0, str(self.trajectory_path))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable_recording:
            self.manager.stop_recording()

        if self.enable_tracing:
            self.manager.disable_trajectory_logging()

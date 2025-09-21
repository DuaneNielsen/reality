"""
Core inference functionality for Madrona Escape Room.

Provides shared API for inference scripts and training evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from madrona_escape_room_learn.moving_avg import EpisodicEMATrackerWithHistogram

import madrona_escape_room


class InferenceConfig:
    """Configuration for inference runs"""

    def __init__(
        self,
        ckpt_path: str,
        compiled_levels: List,
        model_kwargs: Optional[Dict[str, Any]] = None,
        # Optional simulation settings
        num_worlds: int = 4,
        num_steps: int = 1000,
        exec_mode: madrona_escape_room.ExecMode = madrona_escape_room.ExecMode.CPU,
        gpu_id: int = 0,
        sim_seed: int = 0,
        fp16: bool = False,
        # Optional level settings
        level_file: Optional[str] = None,
        # Optional output settings
        recording_path: Optional[str] = None,
        action_dump_path: Optional[str] = None,
        # Optional tracking settings
        track_episodes: bool = True,
        ema_alpha: float = 0.01,
        reward_bins: Optional[List[float]] = None,
        length_bins: Optional[List[int]] = None,
        # Optional display settings
        verbose: bool = True,
        print_interval: int = 100,
        print_probs: bool = False,
    ):
        # Required settings
        self.ckpt_path = ckpt_path
        self.compiled_levels = compiled_levels

        # Model settings - store as dict to pass through
        self.model_kwargs = model_kwargs or {}

        # Simulation settings
        self.num_worlds = num_worlds
        self.num_steps = num_steps
        self.exec_mode = exec_mode
        self.gpu_id = gpu_id
        self.sim_seed = sim_seed
        self.fp16 = fp16

        # Level settings
        self.level_file = level_file

        # Output settings
        self.recording_path = recording_path
        self.action_dump_path = action_dump_path

        # Tracking settings
        self.track_episodes = track_episodes
        self.ema_alpha = ema_alpha
        self.reward_bins = (
            reward_bins if reward_bins is not None else [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
        )
        self.length_bins = length_bins if length_bins is not None else [1, 25, 50, 100, 150, 200]

        # Display settings
        self.verbose = verbose
        self.print_interval = print_interval
        self.print_probs = print_probs


class InferenceRunner:
    """Main inference runner with reusable components"""

    def __init__(self, config: InferenceConfig):
        """Initialize runner with configuration"""
        self.config = config
        self.sim_interface = None
        self.policy = None
        self.episode_tracker: Optional[EpisodicEMATrackerWithHistogram] = None
        self.rnn_states = None
        self._obs = None
        self._action_log = None

    def setup_simulation(self):
        """Set up simulation environment"""
        from madrona_escape_room_learn.sim_interface_adapter import setup_lidar_training_environment

        self.sim_interface = setup_lidar_training_environment(
            num_worlds=self.config.num_worlds,
            exec_mode=self.config.exec_mode,
            gpu_id=self.config.gpu_id,
            rand_seed=self.config.sim_seed,
            compiled_levels=self.config.compiled_levels,
        )
        return self.sim_interface

    def load_policy(self):
        """Load policy from checkpoint"""
        from madrona_escape_room_learn import LearningState
        from policy import make_policy, setup_obs

        # Setup observations
        self._obs, num_obs_features = setup_obs(self.sim_interface.obs)

        # Create and load policy
        self.policy = make_policy(num_obs_features, **self.config.model_kwargs)

        weights = LearningState.load_policy_weights(self.config.ckpt_path)
        self.policy.load_state_dict(weights)

        # Initialize RNN states
        self._init_rnn_states()

        return self.policy

    def _init_rnn_states(self):
        """Initialize RNN states for the policy"""
        self.rnn_states = []
        for shape in self.policy.recurrent_cfg.shapes:
            self.rnn_states.append(
                torch.zeros(
                    *shape[0:2],
                    self.config.num_worlds,
                    shape[2],
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
            )

    def setup_tracking(self):
        """Initialize episode tracking"""
        device = torch.device(
            "cuda" if self.config.exec_mode == madrona_escape_room.ExecMode.CUDA else "cpu"
        )

        self.episode_tracker = EpisodicEMATrackerWithHistogram(
            num_envs=self.config.num_worlds,
            alpha=self.config.ema_alpha,
            device=device,
            reward_bins=self.config.reward_bins,
            length_bins=self.config.length_bins,
            disable=not self.config.track_episodes,
        )
        return self.episode_tracker

    def start_recording(self) -> bool:
        """Start recording if configured"""
        if self.config.recording_path:
            try:
                self.sim_interface.manager.start_recording(self.config.recording_path)
                if self.config.verbose:
                    print(f"Recording started: {self.config.recording_path}")
                return True
            except Exception as e:
                print(f"Failed to start recording: {e}")
                return False
        return False

    def stop_recording(self) -> bool:
        """Stop recording if active"""
        if self.config.recording_path:
            try:
                self.sim_interface.manager.stop_recording()
                if self.config.verbose:
                    print(f"Recording completed: {self.config.recording_path}")
                return True
            except Exception as e:
                print(f"Failed to stop recording: {e}")
                return False
        return False

    def _start_action_logging(self):
        """Start action logging if configured"""
        if self.config.action_dump_path:
            self._action_log = open(self.config.action_dump_path, "wb")

    def _stop_action_logging(self):
        """Stop action logging"""
        if self._action_log:
            self._action_log.close()
            self._action_log = None

    def run_steps(
        self, callback: Optional[Callable] = None, return_trajectory: bool = False
    ) -> Tuple[EpisodicEMATrackerWithHistogram, Optional[Dict]]:
        """
        Run inference for configured number of steps

        Args:
            callback: Optional callback function called each step with:
                     (step_num, obs, actions, rewards, dones, values, action_probs)
            return_trajectory: If True, return full trajectory data

        Returns:
            Tuple of (episode_tracker, trajectory_data)
            - episode_tracker contains all episode statistics
            - trajectory_data is None unless return_trajectory=True
        """
        # Ensure everything is set up
        if not self.sim_interface:
            self.setup_simulation()
        if not self.policy:
            self.load_policy()
        if not self.episode_tracker:
            self.setup_tracking()

        # Start recording and action logging
        recording_started = self.start_recording()
        self._start_action_logging()

        # Trajectory storage
        trajectory = (
            {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "values": [],
                "action_probs": [],
            }
            if return_trajectory
            else None
        )

        # Get references for efficiency
        obs = self._obs
        actions = self.sim_interface.actions
        rewards = self.sim_interface.rewards
        dones = self.sim_interface.dones

        # Main inference loop
        for step in range(self.config.num_steps):
            with torch.no_grad():
                action_dists, values, self.rnn_states = self.policy(self.rnn_states, *obs)
                action_dists.best(actions)
                probs = action_dists.probs()

            # Log actions if configured
            if self._action_log:
                actions.numpy().tofile(self._action_log)

            # Store trajectory if requested
            if return_trajectory:
                trajectory["observations"].append([o.clone() for o in obs])
                trajectory["actions"].append(actions.clone())
                trajectory["values"].append(values.clone())
                trajectory["action_probs"].append([p.clone() for p in probs])

            # Step simulation
            self.sim_interface.step()

            # Process rewards and dones (handle both [worlds, 1] and [worlds] shapes)
            step_rewards = rewards[:, 0] if rewards.dim() == 2 else rewards
            step_dones = dones[:, 0] if dones.dim() == 2 else dones

            # Process termination reasons (similar to rollouts.py)
            step_termination_reasons = (
                self.sim_interface.termination_reasons[:, 0]
                if self.sim_interface.termination_reasons.dim() == 2
                else self.sim_interface.termination_reasons[:, 0, 0]
            )

            # Update tracker with termination reasons
            self.episode_tracker.step_update(
                step_rewards, step_dones.bool(), step_termination_reasons
            )

            # Store post-step data
            if return_trajectory:
                trajectory["rewards"].append(step_rewards.clone())
                trajectory["dones"].append(step_dones.clone())

            # Callback
            if callback:
                callback(step, obs, actions, step_rewards, step_dones, values, probs)

            # Verbose printing
            if self.config.verbose and step % self.config.print_interval == 0:
                self._print_step_info(step, obs, actions, probs, step_rewards)

        # Cleanup
        self._stop_action_logging()
        if recording_started:
            self.stop_recording()

        return self.episode_tracker, trajectory

    def _print_step_info(self, step, obs, actions, probs, rewards):
        """Print step information during verbose mode"""
        stats = self.episode_tracker.get_statistics()
        print(
            f"\nStep {step+1}/{self.config.num_steps} - "
            f"Episodes completed: {stats['episodes/completed']}"
        )

        if self.config.print_probs:
            print("Progress:", obs[0])
            print("Compass:", obs[1])
            print("Actions:", actions.cpu().numpy())
            print(f"Step Rewards: {rewards.cpu().numpy()}")

        reward_ema = stats["episodes/reward_ema"]
        length_ema = stats["episodes/length_ema"]
        print(f"Episode EMA - Reward: {reward_ema:.3f}, Length: {length_ema:.1f}")
        print()

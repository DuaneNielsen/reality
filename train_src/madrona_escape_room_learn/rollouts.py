from dataclasses import dataclass
from typing import List

import torch

from .actor_critic import ActorCritic, RecurrentStateConfig
from .amp import AMPState
from .cfg import SimInterface
from .moving_avg import EMANormalizer, EMATracker
from .profile import profile


@dataclass(frozen=True)
class Rollouts:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]


class RolloutManager:
    def __init__(
        self,
        dev: torch.device,
        sim: SimInterface,
        steps_per_update: int,
        num_bptt_chunks: int,
        amp: AMPState,
        recurrent_cfg: RecurrentStateConfig,
        episode_length_ema_decay: float = 0.95,
        episode_reward_ema_decay: float = 0.95,
    ):
        self.dev = dev
        self.steps_per_update = steps_per_update
        self.num_bptt_chunks = num_bptt_chunks
        assert steps_per_update % num_bptt_chunks == 0
        num_bptt_steps = steps_per_update // num_bptt_chunks
        self.num_bptt_steps = num_bptt_steps

        self.need_obs_copy = sim.obs[0].device != dev

        if dev.type == "cuda":
            float_storage_type = torch.float16
        else:
            float_storage_type = torch.bfloat16

        self.actions = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.actions.shape),
            dtype=sim.actions.dtype,
            device=dev,
        )

        self.log_probs = torch.zeros(self.actions.shape, dtype=float_storage_type, device=dev)

        self.dones = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.dones.shape),
            dtype=torch.bool,
            device=dev,
        )

        self.rewards = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type,
            device=dev,
        )

        self.values = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type,
            device=dev,
        )

        self.bootstrap_values = torch.zeros(sim.rewards.shape, dtype=amp.compute_dtype, device=dev)

        self.obs = []

        for obs_tensor in sim.obs:
            self.obs.append(
                torch.zeros(
                    (num_bptt_chunks, num_bptt_steps, *obs_tensor.shape),
                    dtype=obs_tensor.dtype,
                    device=dev,
                )
            )

        if self.need_obs_copy:
            self.final_obs = []

            for obs_tensor in sim.obs:
                self.final_obs.append(
                    torch.zeros(obs_tensor.shape, dtype=obs_tensor.dtype, device=dev)
                )

        self.rnn_end_states = []
        self.rnn_alt_states = []
        self.rnn_start_states = []
        for rnn_state_shape in recurrent_cfg.shapes:
            # expand shape to batch size
            batched_state_shape = (
                *rnn_state_shape[0:2],
                sim.actions.shape[0],
                rnn_state_shape[2],
            )

            rnn_end_state = torch.zeros(batched_state_shape, dtype=amp.compute_dtype, device=dev)
            rnn_alt_state = torch.zeros_like(rnn_end_state)

            self.rnn_end_states.append(rnn_end_state)
            self.rnn_alt_states.append(rnn_alt_state)

            bptt_starts_shape = (num_bptt_chunks, *batched_state_shape)

            rnn_start_state = torch.zeros(bptt_starts_shape, dtype=amp.compute_dtype, device=dev)

            self.rnn_start_states.append(rnn_start_state)

        self.rnn_end_states = tuple(self.rnn_end_states)
        self.rnn_alt_states = tuple(self.rnn_alt_states)
        self.rnn_start_states = tuple(self.rnn_start_states)

        # Episode length and reward EMA tracking
        self.episode_length_ema = EMATracker(episode_length_ema_decay).to(dev)
        self.episode_reward_ema = EMATracker(episode_reward_ema_decay).to(dev)

        # Track cumulative episode rewards and episode reward max/min from most recent update
        self.episode_reward_accum = torch.zeros(sim.rewards.shape, dtype=torch.float32, device=dev)
        self.episode_reward_max = 0.0
        self.episode_reward_min = 0.0

    def collect(
        self,
        amp: AMPState,
        sim: SimInterface,
        actor_critic: ActorCritic,
        value_normalizer: EMANormalizer,
    ):
        rnn_states_cur_in = self.rnn_end_states
        rnn_states_cur_out = self.rnn_alt_states

        for bptt_chunk in range(0, self.num_bptt_chunks):
            with profile("Cache RNN state"):
                # Cache starting RNN state for this chunk
                for start_state, end_state in zip(self.rnn_start_states, rnn_states_cur_in):
                    start_state[bptt_chunk].copy_(end_state)

            for slot in range(0, self.num_bptt_steps):
                cur_obs_buffers = [obs[bptt_chunk, slot] for obs in self.obs]

                with profile("Policy Infer", gpu=True):
                    for obs_idx, step_obs in enumerate(sim.obs):
                        cur_obs_buffers[obs_idx].copy_(step_obs, non_blocking=True)

                    cur_actions_store = self.actions[bptt_chunk, slot]

                    with amp.enable():
                        actor_critic.fwd_rollout(
                            cur_actions_store,
                            self.log_probs[bptt_chunk, slot],
                            self.values[bptt_chunk, slot],
                            rnn_states_cur_out,
                            rnn_states_cur_in,
                            *cur_obs_buffers,
                        )

                    # Invert normalized values
                    self.values[bptt_chunk, slot] = value_normalizer.invert(
                        amp, self.values[bptt_chunk, slot]
                    )

                    rnn_states_cur_in, rnn_states_cur_out = (
                        rnn_states_cur_out,
                        rnn_states_cur_in,
                    )

                    # This isn't non-blocking because if the sim is running in
                    # CPU mode, the copy needs to be finished before sim.step()
                    # FIXME: proper pytorch <-> madrona cuda stream integration

                    # For now, the Policy Infer profile block ends here to get
                    # a CPU synchronization
                    sim.actions.copy_(cur_actions_store)

                with profile("Simulator Step"):
                    sim.step()

                with profile("Post Step Copy"):
                    cur_rewards_store = self.rewards[bptt_chunk, slot]
                    cur_rewards_store.copy_(sim.rewards, non_blocking=True)

                    # Accumulate rewards for episode totals
                    self.episode_reward_accum += sim.rewards.to(self.episode_reward_accum.dtype)

                    cur_dones_store = self.dones[bptt_chunk, slot]
                    cur_dones_store.copy_(sim.dones, non_blocking=True)

                    # Track episode lengths and rewards when episodes complete
                    done_mask = cur_dones_store.bool()
                    if done_mask.any():
                        # Get current steps taken for worlds that just completed
                        steps_taken = (
                            sim.manager.steps_taken_tensor()
                            .to_torch()
                            .to(self.dev, non_blocking=True)
                        )

                        # Get episode lengths and rewards for all completed episodes
                        completed_episodes = done_mask[:, 0]  # Shape: [num_worlds]
                        if completed_episodes.any():
                            episode_lengths = steps_taken[
                                completed_episodes, 0, 0
                            ]  # Get lengths for completed episodes

                            # Get total accumulated episode rewards for completed episodes
                            episode_rewards = self.episode_reward_accum[completed_episodes, 0]

                            # Update EMAs with mean of completed episodes (vectorized, no CPU sync)
                            if len(episode_lengths) > 0:
                                mean_episode_length = episode_lengths.float().mean()
                                mean_episode_reward = episode_rewards.float().mean()
                                self.episode_length_ema.update(mean_episode_length)
                                self.episode_reward_ema.update(mean_episode_reward)

                                # Track max/min of episode rewards from this update
                                self.episode_reward_max = episode_rewards.float().max().item()
                                self.episode_reward_min = episode_rewards.float().min().item()

                            # Reset accumulator for completed episodes
                            self.episode_reward_accum[completed_episodes] = 0.0

                    for rnn_states in rnn_states_cur_in:
                        rnn_states.masked_fill_(cur_dones_store, 0)

                profile.gpu_measure(sync=True)

        if self.need_obs_copy:
            final_obs = self.final_obs
            for obs_idx, step_obs in enumerate(sim.obs):
                final_obs[obs_idx].copy_(step_obs, non_blocking=True)
        else:
            final_obs = sim.obs

        # rnn_hidden_cur_in and rnn_hidden_cur_out are flipped after each
        # iter so rnn_hidden_cur_in is the final output
        self.rnn_end_states = rnn_states_cur_in
        self.rnn_alt_states = rnn_states_cur_out

        with amp.enable(), profile("Bootstrap Values"):
            actor_critic.fwd_critic(self.bootstrap_values, None, self.rnn_end_states, *final_obs)
            self.bootstrap_values = value_normalizer.invert(amp, self.bootstrap_values)

        # Right now this just returns the rollout manager's pointers,
        # but in the future could return only one set of buffers from a
        # double buffered store, etc

        return Rollouts(
            obs=self.obs,
            actions=self.actions,
            log_probs=self.log_probs,
            dones=self.dones,
            rewards=self.rewards,
            values=self.values,
            bootstrap_values=self.bootstrap_values,
            rnn_start_states=self.rnn_start_states,
        )

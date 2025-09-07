#!/usr/bin/env python3
"""
Script to run inference from a wandb run hash.
Finds the latest checkpoint and creates a recording with the same name but .rec extension.
"""

import argparse
import sys
from pathlib import Path

# Import inference modules
import numpy as np
import torch
from madrona_escape_room_learn import LearningState
from madrona_escape_room_learn.sim_interface_adapter import setup_lidar_training_environment
from policy import make_policy, setup_obs

import madrona_escape_room


def find_latest_checkpoint(wandb_run_path):
    """Find the latest checkpoint in the wandb run directory."""
    checkpoints_dir = Path(wandb_run_path) / "files" / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # Find all .pth files and sort by numerical value
    checkpoint_files = list(checkpoints_dir.glob("*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")

    # Sort by numerical value (extract number from filename)
    checkpoint_files.sort(key=lambda x: int(x.stem))
    latest_checkpoint = checkpoint_files[-1]

    return latest_checkpoint


def run_inference(
    ckpt_path,
    recording_path,
    num_worlds=4,
    num_steps=1000,
    gpu_sim=False,
    fp16=False,
    seed=0,
    gpu_id=0,
    num_channels=256,
    separate_value=False,
    recording_seed=5,
):
    """Run inference with the given parameters."""

    torch.manual_seed(seed)

    exec_mode = madrona_escape_room.ExecMode.CUDA if gpu_sim else madrona_escape_room.ExecMode.CPU

    sim_interface = setup_lidar_training_environment(
        num_worlds=num_worlds, exec_mode=exec_mode, gpu_id=gpu_id, rand_seed=seed
    )

    obs, num_obs_features = setup_obs(sim_interface.obs)
    policy = make_policy(num_obs_features, num_channels, separate_value)

    weights = LearningState.load_policy_weights(ckpt_path)
    policy.load_state_dict(weights)

    actions = sim_interface.actions
    dones = sim_interface.dones
    rewards = sim_interface.rewards

    # Flatten N, A, ... tensors to N * A, ... for rewards and dones (still per-agent)
    # Actions are now per-world, so no flattening needed
    dones = dones.view(-1, *dones.shape[2:])
    rewards = rewards.view(-1, *rewards.shape[2:])

    cur_rnn_states = []

    for shape in policy.recurrent_cfg.shapes:
        cur_rnn_states.append(
            torch.zeros(
                *shape[0:2],
                actions.shape[0],
                shape[2],
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        )

    # Episode tracking by world
    episode_rewards_by_world = [[] for _ in range(num_worlds)]
    episode_lengths_by_world = [[] for _ in range(num_worlds)]
    all_episode_rewards = []
    all_episode_lengths = []

    # Start recording
    if recording_path:
        try:
            # Use the same seed for recording as for inference
            sim_interface.manager.start_recording(str(recording_path), seed)
            print(f"Recording started: {recording_path} with seed {seed}")
        except Exception as e:
            print(f"Failed to start recording: {e}")
            recording_path = None  # Disable recording

    for i in range(num_steps):
        with torch.no_grad():
            action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
            action_dists.best(actions)

        # Check for done episodes
        done_mask = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()

        if done_mask.any():
            # Store completed episode data - reward at done step is the final episode reward
            step_rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            # Get steps taken from the manager's tensor
            steps_taken_tensor = sim_interface.manager.steps_taken_tensor().to_torch()
            steps_taken = (
                steps_taken_tensor.squeeze(-1)
                if steps_taken_tensor.dim() > 1
                else steps_taken_tensor
            )

            # Store by world
            for world_idx in range(num_worlds):
                if done_mask[world_idx]:
                    reward = step_rewards[world_idx].item()
                    length = (
                        steps_taken[world_idx].flatten()[0].item()
                    )  # Get scalar from flattened tensor

                    episode_rewards_by_world[world_idx].append(reward)
                    episode_lengths_by_world[world_idx].append(length)
                    all_episode_rewards.append(reward)
                    all_episode_lengths.append(length)

        # Print details every 100 steps
        if i % 100 == 0:
            print(f"Step {i}/{num_steps} - Episodes completed: {len(all_episode_rewards)}")
            print("Progress:", obs[0])
            print("Compass:", obs[1])
            print("Actions:", actions.cpu().numpy())
            print("Rewards:", rewards)
            print()

        sim_interface.step()

    # Print episode statistics
    print("\n" + "=" * 60)
    print("EPISODE STATISTICS")
    print("=" * 60)

    if all_episode_rewards:
        all_episode_rewards = np.array(all_episode_rewards)
        all_episode_lengths = np.array(all_episode_lengths)

        print(f"Total episodes completed: {len(all_episode_rewards)}")
        print()

        # Overall statistics
        print("OVERALL STATISTICS:")
        reward_mean = np.mean(all_episode_rewards)
        reward_std = np.std(all_episode_rewards)
        length_mean = np.mean(all_episode_lengths)
        length_std = np.std(all_episode_lengths)
        print(f"Reward - Mean: {reward_mean:.3f}, Std: {reward_std:.3f}")
        print(f"Length - Mean: {length_mean:.3f}, Std: {length_std:.3f}")
        print()

        # Per-world statistics
        print("PER-WORLD STATISTICS:")
        for world_idx in range(num_worlds):
            if episode_rewards_by_world[world_idx]:
                world_rewards = np.array(episode_rewards_by_world[world_idx])
                world_lengths = np.array(episode_lengths_by_world[world_idx])
                print(f"World {world_idx}: {len(world_rewards)} episodes")
                print(f"  Rewards: {world_rewards}")
                print(f"  Lengths: {world_lengths}")
                w_reward_mean = np.mean(world_rewards)
                w_reward_std = np.std(world_rewards)
                w_length_mean = np.mean(world_lengths)
                w_length_std = np.std(world_lengths)
                print(f"  Reward - Mean: {w_reward_mean:.3f}, Std: {w_reward_std:.3f}")
                print(f"  Length - Mean: {w_length_mean:.3f}, Std: {w_length_std:.3f}")
                print()
            else:
                print(f"World {world_idx}: 0 episodes")
    else:
        print("No episodes completed during inference")

    # Stop recording if it was started
    if recording_path:
        try:
            sim_interface.manager.stop_recording()
            print(f"Recording completed: {recording_path}")
        except Exception as e:
            print(f"Failed to stop recording: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference from wandb run hash")
    parser.add_argument("wandb_run_hash", help="Wandb run hash (e.g., iecno14m)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--gpu-sim", action="store_true", help="Use GPU simulation")
    parser.add_argument("--num-worlds", type=int, default=4, help="Number of worlds")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--num-channels", type=int, default=256, help="Number of channels")
    parser.add_argument("--separate-value", action="store_true", help="Use separate value network")
    parser.add_argument("--recording-seed", type=int, default=5, help="Seed for recording")

    args = parser.parse_args()

    # Find the wandb run directory
    wandb_base = Path("wandb")
    wandb_run_dirs = list(wandb_base.glob(f"*-{args.wandb_run_hash}"))

    if not wandb_run_dirs:
        print(f"Error: No wandb run found with hash {args.wandb_run_hash}")
        sys.exit(1)

    if len(wandb_run_dirs) > 1:
        print(f"Warning: Multiple runs found with hash {args.wandb_run_hash}, using first one:")
        for run_dir in wandb_run_dirs:
            print(f"  {run_dir}")

    wandb_run_path = wandb_run_dirs[0]
    print(f"Using wandb run: {wandb_run_path}")

    try:
        latest_checkpoint = find_latest_checkpoint(wandb_run_path)
        print(f"Latest checkpoint: {latest_checkpoint}")

        # Create recording path in the same directory as checkpoint
        recording_path = latest_checkpoint.with_suffix(".rec")
        print(f"Recording will be saved to: {recording_path}")

        # Run inference directly
        run_inference(
            ckpt_path=str(latest_checkpoint),
            recording_path=str(recording_path),
            num_worlds=args.num_worlds,
            num_steps=args.num_steps,
            gpu_sim=args.gpu_sim,
            fp16=args.fp16,
            seed=args.seed,
            gpu_id=args.gpu_id,
            num_channels=args.num_channels,
            separate_value=args.separate_value,
            recording_seed=args.recording_seed,
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

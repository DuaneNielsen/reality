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
from madrona_escape_room_learn.moving_avg import EpisodicEMATracker
from madrona_escape_room_learn.sim_interface_adapter import setup_lidar_training_environment
from policy import make_policy, setup_obs
from wandb_utils import find_checkpoint as find_latest_checkpoint

# Import wandb utilities
from wandb_utils import find_wandb_run_dir, get_run_object, lookup_run

import madrona_escape_room
import wandb
from madrona_escape_room.level_io import load_compiled_levels


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
    sim_seed=0,
    compiled_levels=None,
):
    """Run inference with the given parameters."""

    torch.manual_seed(seed)

    exec_mode = madrona_escape_room.ExecMode.CUDA if gpu_sim else madrona_escape_room.ExecMode.CPU

    sim_interface = setup_lidar_training_environment(
        num_worlds=num_worlds,
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        rand_seed=sim_seed,
        compiled_levels=compiled_levels,
    )

    obs, num_obs_features = setup_obs(sim_interface.obs)
    policy = make_policy(num_obs_features, num_channels, separate_value)

    weights = LearningState.load_policy_weights(ckpt_path)
    policy.load_state_dict(weights)

    actions = sim_interface.actions
    dones = sim_interface.dones
    rewards = sim_interface.rewards

    # Keep original tensor shapes:
    # - actions: [worlds, 3]
    # - dones: [worlds, 1] (1 agent per world)
    # - rewards: [worlds, 1] (1 agent per world)

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

    # Initialize episode tracker
    device = torch.device("cuda" if gpu_sim else "cpu")
    episode_tracker = EpisodicEMATracker(num_envs=num_worlds, alpha=0.01, device=device)

    # Keep detailed tracking for per-world analysis
    episode_rewards_by_world = [[] for _ in range(num_worlds)]
    episode_lengths_by_world = [[] for _ in range(num_worlds)]
    all_episode_rewards = []
    all_episode_lengths = []

    # Start recording
    if recording_path:
        try:
            # Start recording (no seed parameter needed - uses simulation's existing seed)
            sim_interface.manager.start_recording(str(recording_path))
            print(f"Recording started: {recording_path}")
        except Exception as e:
            print(f"Failed to start recording: {e}")
            recording_path = None  # Disable recording

    for i in range(num_steps):
        with torch.no_grad():
            action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
            action_dists.best(actions)

        # Step the simulation
        sim_interface.step()

        # Get step rewards and dones (handle both [worlds, 1] and [worlds] shapes)
        step_rewards = rewards[:, 0] if rewards.dim() == 2 else rewards
        step_dones = dones[:, 0] if dones.dim() == 2 else dones

        # Update episode tracker
        completed_episodes = episode_tracker.step_update(step_rewards, step_dones.bool())

        # Process completed episodes for detailed per-world tracking
        if "completed_episodes" in completed_episodes:
            completed_data = completed_episodes["completed_episodes"]
            completed_rewards = completed_data["rewards"]
            completed_lengths = completed_data["lengths"]
            completed_env_indices = completed_data["env_indices"]

            for i, env_idx in enumerate(completed_env_indices):
                episode_return = completed_rewards[i].item()
                episode_length = completed_lengths[i].item()

                episode_rewards_by_world[env_idx].append(episode_return)
                episode_lengths_by_world[env_idx].append(episode_length)
                all_episode_rewards.append(episode_return)
                all_episode_lengths.append(episode_length)

        # Print details every 100 steps
        if i % 100 == 0:
            episode_stats = episode_tracker.get_statistics()
            print(
                f"Step {i+1}/{num_steps} - Episodes completed: {episode_stats['episodes/completed']}"
            )
            print("Progress:", obs[0])
            print("Compass:", obs[1])
            print("Actions:", actions.cpu().numpy())
            print(f"Step Rewards: {step_rewards.cpu().numpy()}")
            print(
                f"Episode EMA - Reward: {episode_stats['episodes/reward_ema']:.3f}, Length: {episode_stats['episodes/length_ema']:.1f}"
            )
            print()

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
        print(f"Episode Return - Mean: {reward_mean:.3f}, Std: {reward_std:.3f}")
        print(f"Episode Length - Mean: {length_mean:.3f}, Std: {length_std:.3f}")
        print()

        # Per-world statistics
        print("PER-WORLD STATISTICS:")
        for world_idx in range(num_worlds):
            if episode_rewards_by_world[world_idx]:
                world_rewards = np.array(episode_rewards_by_world[world_idx])
                world_lengths = np.array(episode_lengths_by_world[world_idx])
                print(f"World {world_idx}: {len(world_rewards)} episodes")
                print(f"  Episode Returns: {world_rewards}")
                print(f"  Episode Lengths: {world_lengths}")
                w_reward_mean = np.mean(world_rewards)
                w_reward_std = np.std(world_rewards)
                w_length_mean = np.mean(world_lengths)
                w_length_std = np.std(world_lengths)
                print(f"  Episode Return - Mean: {w_reward_mean:.3f}, Std: {w_reward_std:.3f}")
                print(f"  Episode Length - Mean: {w_length_mean:.3f}, Std: {w_length_std:.3f}")
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
    parser = argparse.ArgumentParser(description="Run inference from wandb run name or hash")
    parser.add_argument(
        "wandb_run_identifier",
        nargs="?",
        help="Wandb run name or hash (e.g., 'my-training-run' or 'iecno14m')",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const="",
        help="List available wandb runs (optionally filter with regex pattern)",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--gpu-sim", action="store_true", help="Use GPU simulation")
    parser.add_argument("--num-worlds", type=int, default=4, help="Number of worlds")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Number of channels (auto-detected from wandb config)",
    )
    parser.add_argument("--separate-value", action="store_true", help="Use separate value network")
    parser.add_argument("--sim-seed", type=int, default=0, help="Seed for simulation")
    parser.add_argument("--level-file", type=str, help="Path to compiled .lvl level file")
    parser.add_argument(
        "--project", type=str, default="madrona-escape-room", help="Wandb project name"
    )
    parser.add_argument(
        "--lookup", type=str, help="Look up run hash for given human readable name and exit"
    )

    args = parser.parse_args()

    # Handle lookup option
    if args.lookup:
        result = lookup_run(args.lookup, args.project)
        print(result)
        return

    # Handle list option
    if args.list is not None:
        import re

        api = wandb.Api()
        runs = api.runs(args.project)

        # Filter runs if regex pattern provided
        if args.list:  # Non-empty string means regex pattern provided
            pattern = re.compile(args.list, re.IGNORECASE)
            filtered_runs = [run for run in runs if pattern.search(run.name)]
            print(f"Wandb runs matching '{args.list}':")
        else:
            filtered_runs = runs
            print("All wandb runs:")

        for i, run in enumerate(filtered_runs):
            print(f"{i+1:3d}. {run.name} ({run.id}) - {run.state}")

        print(f"Total: {len(filtered_runs)} runs")
        if args.list and len(filtered_runs) != len(runs):
            print(f"(Filtered from {len(runs)} total runs)")
        return

    # Require run identifier if not listing
    if not args.wandb_run_identifier:
        parser.error("wandb_run_identifier is required unless --list is used")

    # Get the wandb run object using the utility function
    identifier = args.wandb_run_identifier
    selected_run = get_run_object(identifier, args.project)

    if isinstance(selected_run, str) and selected_run.startswith("ERROR"):
        print(selected_run)
        sys.exit(1)

    print(f"Using wandb run: {selected_run.name} ({selected_run.id})")

    # Extract model config from wandb config
    if hasattr(selected_run, "config"):
        config = selected_run.config

        # Get num_channels from config (required)
        if args.num_channels is None:
            if "num_channels" in config:
                args.num_channels = config["num_channels"]
                print(f"Using num_channels from wandb config: {args.num_channels}")
            else:
                print(
                    "Error: num_channels not found in wandb config and "
                    "not provided via --num-channels"
                )
                sys.exit(1)

        # Get separate_value from config if not explicitly set
        if not args.separate_value and "separate_value" in config and config["separate_value"]:
            args.separate_value = config["separate_value"]
            print(f"Using separate_value from wandb config: {args.separate_value}")
    else:
        if args.num_channels is None:
            print("Error: No wandb config found and --num-channels not provided")
            sys.exit(1)

    # Convert to local wandb directory path using utility function
    wandb_run_path_result = find_wandb_run_dir(selected_run.id)

    if wandb_run_path_result == "NOT_FOUND":
        print(f"Error: Local wandb directory not found for run {selected_run.id}")
        print("Make sure you have synced this run locally")
        sys.exit(1)
    elif wandb_run_path_result.startswith("ERROR:"):
        print(wandb_run_path_result)
        sys.exit(1)

    wandb_run_path = Path(wandb_run_path_result)

    # Load custom level if provided via --level-file, or check checkpoint directory for .lvl file
    compiled_levels = None
    level_name = "default_16x16_room"
    level_file_to_use = args.level_file

    # If no level file specified, check checkpoint directory for .lvl files
    if not level_file_to_use:
        checkpoints_dir = wandb_run_path / "files" / "checkpoints"
        if checkpoints_dir.exists():
            lvl_files = list(checkpoints_dir.glob("*.lvl"))
            if lvl_files:
                level_file_to_use = str(lvl_files[0])  # Use the first .lvl file found
                print(f"Found level file in checkpoint directory: {level_file_to_use}")

    if level_file_to_use:
        compiled_levels = load_compiled_levels(level_file_to_use)
        num_sublevels = len(compiled_levels)

        if num_sublevels == 1:
            # Single level file - extract level name from the level
            level_name = (
                compiled_levels[0].level_name.decode("utf-8", errors="ignore").strip("\x00")
            )
            if not level_name:
                level_name = Path(level_file_to_use).stem
            print(f"Loaded custom level: {level_name} from {level_file_to_use}")
            print(f"Number of sublevels: {num_sublevels}")
        else:
            # Multi-level file - use filename for display
            level_name = Path(level_file_to_use).stem
            print(
                f"Loaded multi-level file with {num_sublevels} levels from " f"{level_file_to_use}"
            )
            print(f"Using curriculum name: {level_name}")
            print(f"Number of sublevels: {num_sublevels}")

        # Override num_worlds to match number of sublevels
        if args.num_worlds != num_sublevels:
            print(
                f"Overriding --num-worlds from {args.num_worlds} to {num_sublevels} "
                f"to match sublevels"
            )
            args.num_worlds = num_sublevels

    # Find latest checkpoint using utility function
    latest_checkpoint_result = find_latest_checkpoint(str(wandb_run_path))

    if latest_checkpoint_result.startswith("ERROR:"):
        print(f"Error: {latest_checkpoint_result}")
        sys.exit(1)

    latest_checkpoint = Path(latest_checkpoint_result)
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
        sim_seed=args.sim_seed,
        compiled_levels=compiled_levels,
    )


if __name__ == "__main__":
    main()

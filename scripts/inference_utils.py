"""
Utility functions for inference scripts.

Provides configuration conversion and metrics display functions.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from inference_core import InferenceConfig
from madrona_escape_room_learn.moving_avg import EpisodicEMATrackerWithHistogram

import madrona_escape_room


def create_inference_config_from_args(args: argparse.Namespace) -> InferenceConfig:
    """Convert CLI arguments to InferenceConfig"""
    exec_mode = (
        madrona_escape_room.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.ExecMode.CPU
    )

    # Prepare model kwargs for policy creation
    model_kwargs = {
        "num_channels": args.num_channels,
        "separate_value": args.separate_value,
    }

    return InferenceConfig(
        # Required
        ckpt_path=args.ckpt_path,
        compiled_levels=getattr(args, "compiled_levels", []),
        # Model configuration
        model_kwargs=model_kwargs,
        # Simulation
        num_worlds=args.num_worlds,
        num_steps=args.num_steps,
        exec_mode=exec_mode,
        gpu_id=args.gpu_id,
        sim_seed=getattr(args, "sim_seed", getattr(args, "seed", 0)),
        fp16=args.fp16,
        # Level
        level_file=getattr(args, "level_file", None),
        # Output
        recording_path=getattr(args, "recording_path", None),
        action_dump_path=getattr(args, "action_dump_path", None),
        # Display
        verbose=True,  # infer.py is always verbose
        print_interval=1,  # infer.py prints every step
        print_probs=True,  # infer.py shows detailed output
    )


def create_inference_config_from_wandb(
    run_id: str, project: str = "madrona-escape-room", overrides: Optional[Dict] = None
) -> InferenceConfig:
    """Create config from wandb run with optional overrides"""
    from wandb_utils import find_checkpoint, find_wandb_run_dir, get_run_object

    from madrona_escape_room.level_io import load_compiled_levels

    # Get run object
    run = get_run_object(run_id, project)
    if isinstance(run, str) and run.startswith("ERROR"):
        raise ValueError(run)

    # Extract configuration from overrides
    model_kwargs = overrides.get("model_kwargs", {}) if overrides else {}
    compiled_levels = overrides.get("compiled_levels", []) if overrides else []
    level_file = overrides.get("level_file") if overrides else None
    num_worlds = overrides.get("num_worlds", 4) if overrides else 4
    num_steps = overrides.get("num_steps", 1000) if overrides else 1000
    exec_mode_str = overrides.get("exec_mode", "CPU") if overrides else "CPU"
    gpu_id = overrides.get("gpu_id", 0) if overrides else 0
    sim_seed = overrides.get("sim_seed", 0) if overrides else 0
    fp16 = overrides.get("fp16", False) if overrides else False

    # Convert exec_mode string to enum
    import madrona_escape_room

    exec_mode = (
        madrona_escape_room.ExecMode.CUDA
        if exec_mode_str == "CUDA"
        else madrona_escape_room.ExecMode.CPU
    )

    # Find wandb run directory
    run_dir_result = find_wandb_run_dir(run.id)
    if run_dir_result == "NOT_FOUND" or run_dir_result.startswith("ERROR:"):
        raise ValueError(f"Local wandb directory not found for run {run.id}")

    run_dir = Path(run_dir_result)

    # Find checkpoint
    ckpt_result = find_checkpoint(str(run_dir))
    if ckpt_result.startswith("ERROR:"):
        raise ValueError(ckpt_result)

    # Create config with proper constructor
    config = InferenceConfig(
        ckpt_path=ckpt_result,
        compiled_levels=compiled_levels,
        model_kwargs=model_kwargs,
        num_worlds=num_worlds,
        num_steps=num_steps,
        exec_mode=exec_mode,
        gpu_id=gpu_id,
        sim_seed=sim_seed,
        fp16=fp16,
        level_file=level_file,
    )

    return config


def print_episode_statistics(
    tracker: EpisodicEMATrackerWithHistogram, detailed: bool = True, show_histograms: bool = False
) -> None:
    """Pretty print episode statistics from tracker"""

    stats = tracker.get_statistics()

    print("\n" + "=" * 60)
    print("EPISODE STATISTICS")
    print("=" * 60)

    total = stats.get("episodes/completed", 0)
    print(f"Total episodes completed: {total}")

    if total > 0:
        print("\nEMA Statistics:")
        reward_ema = stats["episodes/reward_ema"]
        length_ema = stats["episodes/length_ema"]
        print(f"  Episode Return - Mean: {reward_ema:.3f}")
        print(f"  Episode Length - Mean: {length_ema:.1f}")

        # Termination reasons if available
        if "episodes/termination_time_limit_prob" in stats:
            print("\nTermination Reasons:")
            time_limit = stats["episodes/termination_time_limit_prob"]
            progress_complete = stats["episodes/termination_progress_complete_prob"]
            collision_death = stats["episodes/termination_collision_death_prob"]
            print(f"  Time Limit: {time_limit:.1%}")
            print(f"  Progress Complete: {progress_complete:.1%}")
            print(f"  Collision Death: {collision_death:.1%}")

        if detailed and show_histograms:
            # Get histogram data
            hist_data = tracker.get_histograms()
            if hist_data and len(hist_data) > 0:
                print("\nReward Distribution:")
                reward_probs = hist_data["histograms/reward_distribution"]
                reward_edges = hist_data["histograms/reward_bin_edges"]
                for i, prob in enumerate(reward_probs):
                    if prob > 0.01:  # Only show bins with >1% probability
                        left_edge = reward_edges[i] if i < len(reward_edges) else reward_edges[-1]
                        right_edge = (
                            reward_edges[i + 1] if i + 1 < len(reward_edges) else float("inf")
                        )
                        print(f"  [{left_edge:.1f}, {right_edge:.1f}): {prob:.1%}")

                print("\nLength Distribution:")
                length_probs = hist_data["histograms/length_distribution"]
                length_edges = hist_data["histograms/length_bin_edges"]
                for i, prob in enumerate(length_probs):
                    if prob > 0.01:  # Only show bins with >1% probability
                        left_edge = (
                            int(length_edges[i]) if i < len(length_edges) else int(length_edges[-1])
                        )
                        right_edge = (
                            int(length_edges[i + 1]) if i + 1 < len(length_edges) else float("inf")
                        )
                        print(f"  [{left_edge}, {right_edge}): {prob:.1%}")
    else:
        print("No episodes completed")


def save_metrics_to_file(
    tracker: EpisodicEMATrackerWithHistogram, filepath: str, format: str = "json"
) -> None:
    """Save metrics from tracker to file for later analysis"""

    stats = tracker.get_statistics()
    hist_data = tracker.get_histograms()

    if format == "json":
        # Convert numpy arrays to lists for JSON serialization
        all_data = {**stats}
        if hist_data:
            for key, value in hist_data.items():
                if isinstance(value, np.ndarray):
                    all_data[key] = value.tolist()
                else:
                    all_data[key] = value

        with open(filepath, "w") as f:
            json.dump(all_data, f, indent=2)

    elif format == "npz":
        combined_data = {**stats}
        if hist_data:
            combined_data.update(hist_data)
        np.savez(filepath, **combined_data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def print_per_world_statistics(
    episode_rewards_by_world: list, episode_lengths_by_world: list, num_worlds: int
) -> None:
    """Print per-world statistics (for detailed analysis)"""
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
